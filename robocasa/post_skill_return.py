"""Deterministic task-conditioned arm return after a verified atomic skill."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robocasa.atomic_task_schemas import AtomicTaskCall
from robocasa.held_object_guard import build_held_object_guard
from robocasa.io_utils import write_json
from robocasa.post_grasp_transport import (
    _holding,
    _object_snapshot,
    _raw_task_env,
    resolve_object_alias,
)
from robocasa.work_pose_dataset import (
    current_eef_pose,
    move_eef_to_pose_smooth as move_eef_to_pose,
)


def _finite_vector(value: Any, *, size: int, name: str) -> list[float]:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return vector.tolist()


class PostSkillReturnController:
    """Move each verified skill to a configured base-frame EEF handoff pose.

    Positions are fixed per preset. Orientation may either be fixed in YAML or
    preserved from the end of the skill; preserving it is the conservative
    default for carried bowls and other orientation-sensitive objects.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        log_dir: str | Path | None = None,
    ) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.failure_policy = str(cfg.get("failure_policy", "block"))
        if self.failure_policy not in {"warn", "block"}:
            raise ValueError("post_skill_return.failure_policy must be warn or block")
        self.motion_stability = dict(
            cfg.get("motion_stability", {})
        )
        self.task_presets = {
            str(key): str(value)
            for key, value in dict(cfg.get("task_presets", {})).items()
        }
        self.default_preset = str(cfg.get("default_preset", "neutral"))
        self.presets = {
            str(key): dict(value)
            for key, value in dict(cfg.get("presets", {})).items()
            if isinstance(value, Mapping)
        }
        if self.default_preset not in self.presets:
            raise ValueError(
                "post_skill_return.default_preset must name a configured preset"
            )
        unknown = sorted(set(self.task_presets.values()).difference(self.presets))
        if unknown:
            raise ValueError(
                f"post_skill_return.task_presets reference unknown presets: {unknown}"
            )
        for name, preset in self.presets.items():
            _finite_vector(
                preset.get("target_position"),
                size=3,
                name=f"post_skill_return.presets.{name}.target_position",
            )
            orientation_mode = str(
                preset.get("orientation_mode", "preserve_skill_end")
            )
            if orientation_mode not in {"preserve_skill_end", "fixed"}:
                raise ValueError(
                    f"post_skill_return preset {name!r} has invalid orientation_mode"
                )
            if orientation_mode == "fixed":
                quaternion = _finite_vector(
                    preset.get("target_quaternion_xyzw"),
                    size=4,
                    name=(
                        "post_skill_return.presets."
                        f"{name}.target_quaternion_xyzw"
                    ),
                )
                if float(np.linalg.norm(quaternion)) <= 1e-12:
                    raise ValueError(
                        f"post_skill_return preset {name!r} quaternion is zero"
                    )
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._log_index = 0

    def _save(self, result: Mapping[str, Any]) -> str | None:
        if self.log_dir is None:
            return None
        self._log_index += 1
        name = str(result.get("subgoal_id") or "skill").replace("/", "_")
        path = self.log_dir / "post_skill_return" / f"{self._log_index:06d}_{name}.json"
        write_json(path, result)
        return str(path)

    @staticmethod
    def _target_pose(
        skill_end_pose: Mapping[str, Any], cfg: Mapping[str, Any]
    ) -> dict[str, Any]:
        orientation_mode = str(cfg.get("orientation_mode", "preserve_skill_end"))
        quaternion: Sequence[float]
        if orientation_mode == "fixed":
            quaternion = cfg["target_quaternion_xyzw"]
        else:
            quaternion = skill_end_pose["quaternion_xyzw"]
        return {
            "position": _finite_vector(
                cfg["target_position"], size=3, name="target_position"
            ),
            "quaternion_xyzw": _finite_vector(
                quaternion, size=4, name="target_quaternion_xyzw"
            ),
            "frame": "robot_base",
        }

    def _motion_kwargs(
        self, cfg: Mapping[str, Any],
        *,
        max_steps: int,
        trajectory_steps: int,
    ) -> dict[str, Any]:
        return {
            "max_steps": max_steps,
            "trajectory_steps": min(trajectory_steps, max_steps),
            "translation_tolerance_m": float(
                cfg.get("translation_tolerance_m", 0.03)
            ),
            "orientation_tolerance_deg": float(
                cfg.get("orientation_tolerance_deg", 15.0)
            ),
            "max_translation_command": float(
                cfg.get("max_translation_command", 0.10)
            ),
            "max_rotation_command": float(cfg.get("max_rotation_command", 0.08)),
            "max_translation_command_delta": float(
                cfg.get("max_translation_command_delta", 0.006)
            ),
            "max_rotation_command_delta": float(
                cfg.get("max_rotation_command_delta", 0.004)
            ),
            "translation_gain": float(cfg.get("translation_gain", 3.0)),
            "rotation_gain": float(cfg.get("rotation_gain", 1.5)),
            "stable_steps": int(cfg.get("stable_steps", 5)),
            "linear_velocity_threshold_m_s": float(
                self.motion_stability.get("linear_velocity_threshold_m_s", 0.05)
            ),
            "angular_velocity_threshold_deg_s": float(
                self.motion_stability.get("angular_velocity_threshold_deg_s", 20.0)
            ),
            "near_target_slowdown_radius_m": float(
                self.motion_stability.get("near_target_slowdown_radius_m", 0.08)
            ),
            "near_target_slowdown_radius_deg": float(
                self.motion_stability.get("near_target_slowdown_radius_deg", 30.0)
            ),
            "handoff_max_steps": int(self.motion_stability.get("handoff_max_steps", 30)),
            "handoff_stable_steps": int(self.motion_stability.get("handoff_stable_steps", 2)),
            "handoff_action_tolerance": float(
                self.motion_stability.get("handoff_action_tolerance", 0.006)
            ),
            "simulation_timestep_s": self.motion_stability.get("simulation_timestep_s"),
        }

    def _fixed_return(
        self,
        *,
        env: Any,
        call: AtomicTaskCall,
        preset_name: str,
        cfg: Mapping[str, Any],
        initial_applied_action: Sequence[float] | None,
        simulation_step_offset: int,
    ) -> dict[str, Any]:
        is_carry = preset_name == "carry" and call.atomic_task == "PickObject"
        gripper_action = 1.0 if is_carry else float(cfg.get("gripper_action", -1.0))
        before = current_eef_pose(env)
        target = self._target_pose(before, cfg)

        object_alias = resolve_object_alias(env, call) if is_carry else None
        raw_env = None
        held_before = None
        object_before = None
        guard = None
        if is_carry and object_alias is not None:
            raw_env = _raw_task_env(env)
            held_before = _holding(raw_env, object_alias)
            object_before = _object_snapshot(raw_env, object_alias)
            guard = build_held_object_guard(
                env=env,
                atomic_task_call=call,
                enabled=True,
                hold_confirmation_steps=int(cfg.get("hold_confirmation_steps", 2)),
                drop_confirmation_steps=int(cfg.get("drop_confirmation_steps", 30)),
            )

        # Clear the work surface vertically before moving to the exact fixed
        # pose, so a low object is not swept through a sink or counter edge.
        pre_lift_height = float(cfg.get("pre_lift_height", 0.10))
        pre_lift_minimum_z = float(cfg.get("pre_lift_minimum_z", 0.22))
        pre_lift_target = {
            "position": list(before["position"]),
            "quaternion_xyzw": list(before["quaternion_xyzw"]),
            "frame": "robot_base",
        }
        pre_lift_target["position"][2] = max(
            float(pre_lift_target["position"][2]) + pre_lift_height,
            pre_lift_minimum_z,
        )
        pre_lift = move_eef_to_pose(
            env,
            pre_lift_target,
            guard=guard,
            initial_applied_action=initial_applied_action,
            simulation_step_offset=simulation_step_offset,
            settle_steps=0,
            gripper_action=gripper_action,
            **self._motion_kwargs(
                cfg,
                max_steps=int(cfg.get("pre_lift_max_steps", 160)),
                trajectory_steps=int(cfg.get("pre_lift_trajectory_steps", 60)),
            ),
        )

        dropped_during_lift = bool(
            pre_lift.get("guard_failure") is not None
            or (
                is_carry
                and raw_env is not None
                and object_alias is not None
                and _holding(raw_env, object_alias) is False
            )
        )
        return_translation = None
        return_orientation = None
        handoff = None
        if pre_lift.get("success", False) and not dropped_during_lift:
            lifted_pose = current_eef_pose(env)
            orientation_mode = str(
                cfg.get("orientation_mode", "preserve_skill_end")
            )
            if orientation_mode == "preserve_skill_end":
                target["quaternion_xyzw"] = list(
                    lifted_pose["quaternion_xyzw"]
                )
            translation_target = {
                "position": list(target["position"]),
                "quaternion_xyzw": list(lifted_pose["quaternion_xyzw"]),
                "frame": "robot_base",
            }
            return_budget = int(cfg.get("return_max_steps", 240))
            split_orientation = orientation_mode == "fixed"
            translation_budget = (
                int(
                    cfg.get(
                        "return_translation_max_steps",
                        max(1, round(return_budget * 2.0 / 3.0)),
                    )
                )
                if split_orientation
                else return_budget
            )
            return_translation = move_eef_to_pose(
                env,
                translation_target,
                start_guard=False,
                initial_applied_action=pre_lift.get("last_applied_action"),
                simulation_step_offset=(
                    simulation_step_offset + int(pre_lift.get("steps", 0))
                ),

                guard=guard,
                settle_steps=(
                    0 if split_orientation else int(cfg.get("hold_steps", 10))
                ),
                settle_mode=str(cfg.get("hold_mode", "zero_delta")),
                gripper_action=gripper_action,
                **self._motion_kwargs(
                    cfg,
                    max_steps=translation_budget,
                    trajectory_steps=int(
                        cfg.get("return_translation_trajectory_steps", 120)
                    ),
                ),
            )
            handoff = return_translation
            if return_translation.get("success", False) and split_orientation:
                orientation_budget = int(
                    cfg.get(
                        "return_orientation_max_steps",
                        max(1, return_budget - translation_budget),
                    )
                )
                return_orientation = move_eef_to_pose(
                    env,
                    target,
                    start_guard=False,
                    initial_applied_action=return_translation.get(
                        "last_applied_action"
                    ),
                    simulation_step_offset=(
                        simulation_step_offset
                        + int(pre_lift.get("steps", 0))
                        + int(return_translation.get("steps", 0))
                    ),

                    guard=guard,
                    settle_steps=int(cfg.get("hold_steps", 10)),
                    settle_mode=str(cfg.get("hold_mode", "zero_delta")),
                    gripper_action=gripper_action,
                    **self._motion_kwargs(
                        cfg,
                        max_steps=orientation_budget,
                        trajectory_steps=int(
                            cfg.get("return_orientation_trajectory_steps", 60)
                        ),
                    ),
                )
                handoff = dict(return_orientation)
                handoff["success"] = bool(
                    return_translation.get("success", False)
                    and return_orientation.get("success", False)
                )
                handoff["guard_failure"] = (
                    return_translation.get("guard_failure")
                    or return_orientation.get("guard_failure")
                )
                handoff["steps"] = int(return_translation.get("steps", 0)) + int(
                    return_orientation.get("steps", 0)
                )
                handoff["requested_max_steps"] = (
                    translation_budget + orientation_budget
                )
                handoff["trajectory_type"] = "staged_translation_then_orientation"

        after = current_eef_pose(env)
        held_after = (
            _holding(raw_env, object_alias)
            if is_carry and raw_env is not None and object_alias is not None
            else None
        )
        object_after = (
            _object_snapshot(raw_env, object_alias)
            if raw_env is not None and object_alias is not None
            else None
        )
        dropped = bool(
            dropped_during_lift
            or (handoff is not None and handoff.get("guard_failure") is not None)
            or (is_carry and object_alias is not None and held_after is False)
        )
        if is_carry:
            ready = bool(
                object_alias is not None
                and pre_lift.get("success", False)
                and handoff is not None
                and handoff.get("success", False)
                and held_after is True
            )
        else:
            ready = bool(
                pre_lift.get("success", False)
                and handoff is not None
                and handoff.get("success", False)
            )

        if dropped:
            failure_code = "OBJECT_DROPPED_DURING_PREP"
            reason = failure_code
        elif not pre_lift.get("success", False):
            failure_code = (
                "POST_GRASP_LIFT_FAILED" if is_carry else "POST_SKILL_PRE_LIFT_FAILED"
            )
            reason = failure_code
        elif handoff is None or not handoff.get("success", False):
            failure_code = (
                "POST_GRASP_RETRACT_FAILED"
                if is_carry
                else "POST_SKILL_RETURN_FAILED"
            )
            reason = failure_code
        elif not ready:
            failure_code = (
                "TRANSPORT_READY_FAILED" if is_carry else "POST_SKILL_RETURN_FAILED"
            )
            reason = failure_code
        else:
            failure_code = None
            reason = "TRANSPORT_READY" if is_carry else "POST_SKILL_RETURN_READY"

        return {
            "module": "POST_SKILL_RETURN",
            "subgoal_id": call.subgoal_id,
            "atomic_task": call.atomic_task,
            "preset": preset_name,
            "pose_strategy": "task_conditioned_smooth_staged_eef_trajectory",
            "smooth_trajectory": True,
            "fixed_return_pose": True,
            "orientation_mode": str(
                cfg.get("orientation_mode", "preserve_skill_end")
            ),
            "object_alias": object_alias,
            "before": {"eef_pose": before, "object_pose": object_before},
            "pre_lift": pre_lift,
            "handoff_target_pose": target,
            "return_translation": return_translation,
            "return_orientation": return_orientation,
            "handoff": handoff,
            "after": {"eef_pose": after, "object_pose": object_after},
            "initial_applied_action": (
                list(initial_applied_action)
                if initial_applied_action is not None
                else None
            ),
            "simulation_step_offset": int(simulation_step_offset),
            "total_motion_steps": (
                int(pre_lift.get("steps", 0))
                + int((return_translation or {}).get("steps", 0))
                + int((return_orientation or {}).get("steps", 0))
            ),
            "gripper_action": gripper_action,
            "held_before": held_before,
            "held_after": held_after,
            "transport_ready": ready if is_carry else None,
            "retreat_ready": ready if not is_carry else None,
            "return_ready": ready,
            "navigation_safe": ready,
            "success": ready,
            "reason": reason,
            "failure_code": failure_code,
        }

    def prepare(
        self,
        *,
        env: Any,
        atomic_task_call: AtomicTaskCall | Mapping[str, Any],
        episode_id: int | str | None = None,
        skill_verified: bool = True,
        initial_applied_action: Sequence[float] | None = None,
        simulation_step_offset: int = 0,
    ) -> dict[str, Any]:
        call = (
            atomic_task_call
            if isinstance(atomic_task_call, AtomicTaskCall)
            else AtomicTaskCall.from_mapping(atomic_task_call)
        )
        preset_name = self.task_presets.get(call.atomic_task, self.default_preset)
        cfg = self.presets.get(preset_name)
        if not self.enabled or not skill_verified or cfg is None:
            result: dict[str, Any] = {
                "module": "POST_SKILL_RETURN",
                "episode_id": episode_id,
                "subgoal_id": call.subgoal_id,
                "atomic_task": call.atomic_task,
                "preset": preset_name,
                "success": False,
                "return_ready": False,
                "navigation_safe": False,
                "reason": "POST_SKILL_RETURN_DISABLED_OR_UNVERIFIED",
                "failure_code": "POST_SKILL_RETURN_FAILED",
            }
        else:
            result = self._fixed_return(
                env=env,
                call=call,
                preset_name=preset_name,
                cfg=cfg,
                initial_applied_action=initial_applied_action,
                simulation_step_offset=int(simulation_step_offset),
            )
            result["episode_id"] = episode_id
        result["artifact_path"] = self._save(result)
        print(
            "[POST-SKILL] {} task={} preset={} fixed_pose={} ready={} reason={}".format(
                call.subgoal_id,
                call.atomic_task,
                preset_name,
                result.get("fixed_return_pose", False),
                result.get("return_ready", False),
                result.get("reason"),
            )
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "failure_policy": self.failure_policy,
            "default_preset": self.default_preset,
            "task_presets": dict(self.task_presets),
            "presets": {key: dict(value) for key, value in self.presets.items()},
            "motion_stability": dict(self.motion_stability),
        }
