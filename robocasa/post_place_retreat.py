"""Deterministic arm retreat after a verified object placement."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robocasa.atomic_task_schemas import AtomicTaskCall
from robocasa.io_utils import write_json
from robocasa.post_grasp_transport import (
    _object_snapshot,
    _raw_task_env,
    resolve_object_alias,
)
from robocasa.work_pose_dataset import (
    current_base_pose,
    current_eef_pose,
    move_eef_to_pose,
)


FAILURE_CODES = frozenset(
    {
        "POST_PLACE_LIFT_FAILED",
        "POST_PLACE_RETRACT_FAILED",
        "POST_PLACE_RETREAT_FAILED",
    }
)


class PostPlaceRetreatController:
    """Open the gripper, lift clear, and compact the arm after PlaceObject."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        log_dir: str | Path | None = None,
    ) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.lift_height = float(cfg.get("lift_height", 0.10))
        self.retract_distance = float(cfg.get("retract_distance", 0.12))
        self.minimum_planar_radius = float(
            cfg.get("minimum_planar_radius", 0.35)
        )
        self.minimum_eef_lift_m = float(
            cfg.get("minimum_eef_lift_m", 0.03)
        )
        self.maximum_retreat_planar_radius = float(
            cfg.get("maximum_retreat_planar_radius", 0.72)
        )
        self.lift_max_steps = int(cfg.get("lift_max_steps", 160))
        self.retract_max_steps = int(cfg.get("retract_max_steps", 180))
        self.translation_tolerance_m = float(
            cfg.get("translation_tolerance_m", 0.025)
        )
        self.orientation_tolerance_deg = float(
            cfg.get("orientation_tolerance_deg", 12.0)
        )
        self.max_translation_command = float(
            cfg.get("max_translation_command", 0.12)
        )
        self.max_rotation_command = float(
            cfg.get("max_rotation_command", 0.10)
        )
        self.stable_steps = int(cfg.get("stable_steps", 5))
        self.hold_steps = int(cfg.get("hold_steps", 10))
        self.hold_mode = str(cfg.get("hold_mode", "zero_delta"))
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._log_index = 0

        positive = {
            "lift_height": self.lift_height,
            "retract_distance": self.retract_distance,
            "minimum_planar_radius": self.minimum_planar_radius,
            "minimum_eef_lift_m": self.minimum_eef_lift_m,
            "maximum_retreat_planar_radius": self.maximum_retreat_planar_radius,
            "lift_max_steps": self.lift_max_steps,
            "retract_max_steps": self.retract_max_steps,
            "translation_tolerance_m": self.translation_tolerance_m,
            "orientation_tolerance_deg": self.orientation_tolerance_deg,
            "max_translation_command": self.max_translation_command,
            "max_rotation_command": self.max_rotation_command,
            "stable_steps": self.stable_steps,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid or self.hold_steps < 0:
            raise ValueError(
                "post-place retreat values must be positive "
                f"(hold_steps may be zero); invalid={invalid}"
            )
        if self.hold_mode not in {"zero_delta", "closed_loop"}:
            raise ValueError(
                "post-place hold_mode must be 'zero_delta' or 'closed_loop'"
            )

    def _motion_kwargs(self, *, max_steps: int) -> dict[str, Any]:
        return {
            "max_steps": max_steps,
            "translation_tolerance_m": self.translation_tolerance_m,
            "orientation_tolerance_deg": self.orientation_tolerance_deg,
            "max_translation_command": self.max_translation_command,
            "max_rotation_command": self.max_rotation_command,
            "stable_steps": self.stable_steps,
            # RoboCasa's Panda gripper uses -1 for open and +1 for close.
            "gripper_action": -1.0,
        }

    @staticmethod
    def _optional_object_snapshot(
        raw_env: Any | None, object_alias: str | None
    ) -> dict[str, Any]:
        if raw_env is None or object_alias is None:
            return {
                "alias": object_alias,
                "available": False,
                "reason": "OBJECT_ALIAS_UNRESOLVED",
            }
        snapshot = _object_snapshot(raw_env, object_alias)
        snapshot["available"] = "position" in snapshot
        return snapshot

    def _retreat_ready_check(
        self,
        *,
        before_eef: Mapping[str, Any],
        current_eef: Mapping[str, Any],
    ) -> dict[str, Any]:
        before_position = np.asarray(before_eef["position"], dtype=float)
        current_position = np.asarray(current_eef["position"], dtype=float)
        measured_lift = float(current_position[2] - before_position[2])
        planar_radius = float(np.linalg.norm(current_position[:2]))
        lift_ok = measured_lift >= self.minimum_eef_lift_m
        compactness_ok = (
            planar_radius <= self.maximum_retreat_planar_radius
        )
        return {
            "measured_eef_lift_m": measured_lift,
            "minimum_eef_lift_m": self.minimum_eef_lift_m,
            "current_planar_radius_m": planar_radius,
            "maximum_retreat_planar_radius_m": (
                self.maximum_retreat_planar_radius
            ),
            "lift_ok": lift_ok,
            "compactness_ok": compactness_ok,
            "safe": bool(lift_ok and compactness_ok),
        }

    def _save(self, result: Mapping[str, Any]) -> str | None:
        if self.log_dir is None:
            return None
        self._log_index += 1
        alias = str(result.get("object_alias") or "object").replace("/", "_")
        path = (
            self.log_dir
            / "post_place_retreat"
            / f"{self._log_index:06d}_{alias}.json"
        )
        write_json(path, result)
        return str(path)

    @staticmethod
    def _print_summary(result: Mapping[str, Any]) -> None:
        before = result.get("before", {})
        after = result.get("after", {})
        lift = result.get("lift", {})
        retract = result.get("retract", {})
        print("[POST-PLACE]")
        print(f"  object: {result.get('object_alias')}")
        print(f"  place_verified: {result.get('place_verified')}")
        print(
            "  before: eef_pose={} object_pose={} base_pose={}".format(
                before.get("eef_pose"),
                before.get("object_pose"),
                before.get("base_pose"),
            )
        )
        print(
            "  lift: requested_delta_z={} success={} failure_code={}".format(
                lift.get("requested_delta_z"),
                lift.get("success"),
                lift.get("failure_code"),
            )
        )
        print(
            "  retract: requested_distance={} effective_distance={} "
            "success={} failure_code={}".format(
                retract.get("requested_distance"),
                retract.get("effective_distance"),
                retract.get("success"),
                retract.get("failure_code"),
            )
        )
        print(
            "  after: eef_pose={} object_pose={}".format(
                after.get("eef_pose"), after.get("object_pose")
            )
        )
        print(
            "  retreat_ready: {} reason: {}".format(
                result.get("retreat_ready"), result.get("reason")
            )
        )

    def prepare(
        self,
        *,
        env: Any,
        atomic_task_call: AtomicTaskCall | Mapping[str, Any],
        episode_id: int | str | None = None,
        place_verified: bool = True,
    ) -> dict[str, Any]:
        """Lift and retract after a verified placement with an open gripper."""

        call = (
            atomic_task_call
            if isinstance(atomic_task_call, AtomicTaskCall)
            else AtomicTaskCall.from_mapping(atomic_task_call)
        )
        object_alias = resolve_object_alias(env, call)
        result: dict[str, Any] = {
            "module": "POST_PLACE_RETREAT",
            "episode_id": episode_id,
            "subgoal_id": call.subgoal_id,
            "object_alias": object_alias,
            "enabled": self.enabled,
            "place_verified": bool(place_verified),
            "retreat_ready": False,
            "navigation_safe": False,
            "success": False,
            "reason": None,
            "failure_code": None,
            "fixed_safe_pose": False,
            "safe_pose_strategy": "base_relative_lift_then_planar_radial_retract",
            "gripper_action": -1.0,
        }
        if not self.enabled:
            result.update(
                {
                    "reason": "POST_PLACE_RETREAT_DISABLED",
                    "failure_code": "POST_PLACE_RETREAT_FAILED",
                }
            )
            result["artifact_path"] = self._save(result)
            self._print_summary(result)
            return result
        if call.atomic_task != "PlaceObject" or not place_verified:
            result.update(
                {
                    "reason": "PLACE_NOT_VERIFIED",
                    "failure_code": "POST_PLACE_RETREAT_FAILED",
                }
            )
            result["artifact_path"] = self._save(result)
            self._print_summary(result)
            return result

        # Arm retreat only needs the end-effector pose. Object pose is useful
        # oracle diagnostics, but an unresolved alias must not block a verified
        # placement from moving the gripper to a safer pose.
        raw_env = _raw_task_env(env) if object_alias is not None else None
        before = {
            "eef_pose": current_eef_pose(env),
            "object_pose": self._optional_object_snapshot(
                raw_env, object_alias
            ),
            "base_pose": current_base_pose(env),
        }
        result["before"] = before
        lift_target = {
            "position": list(before["eef_pose"]["position"]),
            "quaternion_xyzw": list(
                before["eef_pose"]["quaternion_xyzw"]
            ),
            "frame": "robot_base",
        }
        lift_target["position"][2] += self.lift_height
        lift_motion = move_eef_to_pose(
            env,
            lift_target,
            settle_steps=0,
            **self._motion_kwargs(max_steps=self.lift_max_steps),
        )
        result["lift"] = {
            "requested_delta_z": self.lift_height,
            **dict(lift_motion),
        }
        after_lift = {
            "eef_pose": current_eef_pose(env),
            "object_pose": self._optional_object_snapshot(
                raw_env, object_alias
            ),
            "base_pose": current_base_pose(env),
        }
        lift_check = self._retreat_ready_check(
            before_eef=before["eef_pose"],
            current_eef=after_lift["eef_pose"],
        )
        result["after_lift"] = after_lift
        result["lift_safety_check"] = lift_check
        if not lift_check["lift_ok"]:
            result.update(
                {
                    "reason": "INSUFFICIENT_EEF_CLEARANCE_AFTER_PLACE",
                    "failure_code": "POST_PLACE_LIFT_FAILED",
                    "after": after_lift,
                }
            )
            result["artifact_path"] = self._save(result)
            self._print_summary(result)
            return result
        if not lift_motion.get("success", False):
            result.setdefault("motion_warnings", []).append(
                "LIFT_TARGET_NOT_CONVERGED_BUT_CLEARANCE_SAFE"
            )

        retract_start = current_eef_pose(env)
        retract_target = {
            "position": list(retract_start["position"]),
            "quaternion_xyzw": list(
                retract_start["quaternion_xyzw"]
            ),
            "frame": "robot_base",
        }
        planar = np.asarray(retract_start["position"][:2], dtype=float)
        radius = float(np.linalg.norm(planar))
        effective_retract = min(
            self.retract_distance,
            max(radius - self.minimum_planar_radius, 0.0),
        )
        if radius > 1e-9 and effective_retract > 0:
            retract_target["position"][:2] = (
                planar * ((radius - effective_retract) / radius)
            ).tolist()
        retract_motion = move_eef_to_pose(
            env,
            retract_target,
            settle_steps=self.hold_steps,
            settle_mode=self.hold_mode,
            **self._motion_kwargs(max_steps=self.retract_max_steps),
        )
        result["retract"] = {
            "requested_distance": self.retract_distance,
            "effective_distance": effective_retract,
            **dict(retract_motion),
        }
        after = {
            "eef_pose": current_eef_pose(env),
            "object_pose": self._optional_object_snapshot(
                raw_env, object_alias
            ),
            "base_pose": current_base_pose(env),
        }
        result["after"] = after
        safety = self._retreat_ready_check(
            before_eef=before["eef_pose"],
            current_eef=after["eef_pose"],
        )
        result["retreat_ready_check"] = {
            **safety,
            "lift_motion_converged": bool(lift_motion.get("success", False)),
            "retract_motion_converged": bool(
                retract_motion.get("success", False)
            ),
        }
        if safety["safe"]:
            if not retract_motion.get("success", False):
                result.setdefault("motion_warnings", []).append(
                    "RETRACT_TARGET_NOT_CONVERGED_BUT_NAVIGATION_SAFE"
                )
            result.update(
                {
                    "success": True,
                    "retreat_ready": True,
                    "navigation_safe": True,
                    "reason": (
                        "POST_PLACE_RETREAT_READY_WITH_MOTION_WARNING"
                        if result.get("motion_warnings")
                        else "POST_PLACE_RETREAT_READY"
                    ),
                    "failure_code": None,
                }
            )
        else:
            lift_failed = not safety["lift_ok"]
            result.update(
                {
                    "reason": (
                        "FINAL_EEF_CLEARANCE_UNSAFE"
                        if lift_failed
                        else "FINAL_ARM_NOT_COMPACT"
                    ),
                    "failure_code": (
                        "POST_PLACE_LIFT_FAILED"
                        if lift_failed
                        else "POST_PLACE_RETRACT_FAILED"
                    ),
                }
            )
        result["artifact_path"] = self._save(result)
        self._print_summary(result)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "lift_height": self.lift_height,
            "retract_distance": self.retract_distance,
            "minimum_planar_radius": self.minimum_planar_radius,
            "minimum_eef_lift_m": self.minimum_eef_lift_m,
            "maximum_retreat_planar_radius": self.maximum_retreat_planar_radius,
            "lift_max_steps": self.lift_max_steps,
            "retract_max_steps": self.retract_max_steps,
            "translation_tolerance_m": self.translation_tolerance_m,
            "orientation_tolerance_deg": self.orientation_tolerance_deg,
            "max_translation_command": self.max_translation_command,
            "max_rotation_command": self.max_rotation_command,
            "stable_steps": self.stable_steps,
            "hold_steps": self.hold_steps,
            "hold_mode": self.hold_mode,
            "fixed_safe_pose": False,
            "gripper_action": -1.0,
        }
