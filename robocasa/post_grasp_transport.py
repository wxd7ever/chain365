"""Deterministic post-grasp handoff from manipulation to base navigation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robocasa.atomic_task_schemas import AtomicTaskCall
from robocasa.held_object_guard import build_held_object_guard
from robocasa.io_utils import write_json
from robocasa.utils import object_utils
from robocasa.work_pose_dataset import (
    current_base_pose,
    current_eef_pose,
    move_eef_to_pose,
)


FAILURE_CODES = frozenset(
    {
        "POST_GRASP_LIFT_FAILED",
        "POST_GRASP_RETRACT_FAILED",
        "TRANSPORT_READY_FAILED",
        "OBJECT_DROPPED_DURING_PREP",
    }
)


def _raw_task_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if all(
            hasattr(current, name)
            for name in ("fixtures", "objects", "sim")
        ):
            return current
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is not None and unwrapped is not current:
            current = unwrapped
        else:
            current = getattr(current, "env", None)
    raise ValueError("could not find the underlying RoboCasa task environment")


def _condition_subjects(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, Mapping)
    ):
        return [
            subject
            for item in value
            for subject in _condition_subjects(item)
        ]
    if not isinstance(value, Mapping):
        return []
    nested: list[str] = []
    for key in ("all_of", "any_of", "not"):
        if key in value:
            nested.extend(_condition_subjects(value[key]))
    subject = value.get("subject")
    if isinstance(subject, str) and subject.strip():
        nested.append(subject.strip())
    return nested


def _object_identifiers(call: AtomicTaskCall) -> list[str]:
    """Return every plausible object identifier without discarding aliases."""

    candidates: list[str] = []
    # Explicit alias-like fields are strongest. A held_object_id is common in
    # decomposed Place calls and reference_object_id is emitted by work-pose
    # planning metadata.
    for key in ("object_id", "source_object_id", "held_object_id"):
        value = call.arguments.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    work_pose = call.metadata.get("work_pose")
    if isinstance(work_pose, Mapping):
        value = work_pose.get("reference_object_id")
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    # Planner termination conditions often retain the stable simulator alias
    # even when arguments contain a natural-language instance name.
    candidates.extend(_condition_subjects(call.termination_condition))

    for key in ("object_name", "source_object_name"):
        value = call.arguments.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    return list(dict.fromkeys(candidates))


def _object_identifier(call: AtomicTaskCall) -> str | None:
    """Backward-compatible first candidate for external debug callers."""

    candidates = _object_identifiers(call)
    return candidates[0] if candidates else None


def _normalise_identifier(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def resolve_object_alias(env: Any, call: AtomicTaskCall | Mapping[str, Any]) -> str | None:
    """Resolve the object produced by a PickObject call without changing state."""

    parsed = (
        call
        if isinstance(call, AtomicTaskCall)
        else AtomicTaskCall.from_mapping(call)
    )
    identifiers = _object_identifiers(parsed)
    if not identifiers:
        return None
    raw_env = _raw_task_env(env)

    # Prefer an exact alias from any candidate over a weaker natural-language
    # match. This resolves e.g. arguments.object_name="bell pepper" together
    # with termination_condition.subject="vegetable" deterministically.
    for identifier in identifiers:
        if identifier in raw_env.objects:
            return str(identifier)

    get_obj_lang = getattr(raw_env, "get_obj_lang", None)
    for identifier in identifiers:
        wanted = _normalise_identifier(identifier)
        matches = []
        for alias, entity in raw_env.objects.items():
            names = {
                _normalise_identifier(alias),
                _normalise_identifier(getattr(entity, "name", "")),
                _normalise_identifier(getattr(entity, "nat_lang", "")),
                _normalise_identifier(getattr(entity, "natural_name", "")),
            }
            if callable(get_obj_lang):
                try:
                    names.add(_normalise_identifier(get_obj_lang(str(alias))))
                except (
                    AssertionError,
                    AttributeError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    pass
            if wanted in names:
                matches.append(str(alias))
        matches = list(dict.fromkeys(matches))
        if len(matches) == 1:
            return matches[0]
    return None


def _holding(raw_env: Any, object_alias: str) -> bool | None:
    try:
        return bool(
            object_utils.check_obj_grasped(
                raw_env,
                object_alias,
                threshold=object_utils.GRASP_CLOSED_THRESHOLD,
            )
        )
    except (
        AssertionError,
        AttributeError,
        IndexError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None


def _object_snapshot(raw_env: Any, object_alias: str) -> dict[str, Any]:
    """Return oracle-only pose and approximate support clearance diagnostics."""

    result: dict[str, Any] = {"alias": object_alias}
    try:
        body_id = raw_env.obj_body_id[object_alias]
        position = np.asarray(
            raw_env.sim.data.body_xpos[body_id], dtype=float
        )
        quaternion_wxyz = np.asarray(
            raw_env.sim.data.body_xquat[body_id], dtype=float
        )
        quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
        result.update(
            {
                "position": position.tolist(),
                "quaternion_xyzw": quaternion_xyzw.tolist(),
            }
        )
        entity = raw_env.objects[object_alias]
        points = np.asarray(
            entity.get_bbox_points(trans=position, rot=quaternion_xyzw),
            dtype=float,
        )
        if points.ndim == 2 and points.shape[1] == 3:
            result["lowest_z"] = float(np.min(points[:, 2]))
            result["highest_z"] = float(np.max(points[:, 2]))

            support_candidates: list[tuple[float, str]] = []
            for alias, fixture in raw_env.fixtures.items():
                try:
                    fixture_points = np.asarray(
                        fixture.get_bbox_points(), dtype=float
                    )
                except (
                    AssertionError,
                    AttributeError,
                    IndexError,
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    continue
                if (
                    fixture_points.ndim != 2
                    or fixture_points.shape[1] != 3
                    or not np.isfinite(fixture_points).all()
                ):
                    continue
                xy_min = fixture_points[:, :2].min(axis=0) - 0.02
                xy_max = fixture_points[:, :2].max(axis=0) + 0.02
                if np.all(position[:2] >= xy_min) and np.all(
                    position[:2] <= xy_max
                ):
                    top_z = float(np.max(fixture_points[:, 2]))
                    if top_z <= result["lowest_z"] + 0.05:
                        support_candidates.append((top_z, str(alias)))
            if support_candidates:
                support_z, support_alias = max(support_candidates)
                result["support_fixture"] = support_alias
                result["support_surface_z"] = support_z
                result["support_clearance_m"] = float(
                    result["lowest_z"] - support_z
                )
    except (
        AssertionError,
        AttributeError,
        IndexError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        result["diagnostic_error"] = str(exc)
    return result


class PostGraspTransportController:
    """Lift and radially retract a held object before global base motion."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        log_dir: str | Path | None = None,
    ) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.lift_height = float(cfg.get("lift_height", 0.10))
        self.retract_distance = float(cfg.get("retract_distance", 0.08))
        self.minimum_planar_radius = float(
            cfg.get("minimum_planar_radius", 0.35)
        )
        self.maximum_transport_planar_radius = float(
            cfg.get("maximum_transport_planar_radius", 0.70)
        )
        self.minimum_object_lift_m = float(
            cfg.get("minimum_object_lift_m", 0.03)
        )
        self.minimum_object_clearance_m = float(
            cfg.get("minimum_object_clearance_m", 0.03)
        )
        self.lift_max_steps = int(cfg.get("lift_max_steps", 160))
        self.retract_max_steps = int(cfg.get("retract_max_steps", 160))
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
        self.hold_confirmation_steps = int(
            cfg.get("hold_confirmation_steps", 2)
        )
        self.drop_confirmation_steps = int(
            cfg.get("drop_confirmation_steps", 2)
        )
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._log_index = 0
        positive = {
            "lift_height": self.lift_height,
            "retract_distance": self.retract_distance,
            "minimum_planar_radius": self.minimum_planar_radius,
            "maximum_transport_planar_radius": self.maximum_transport_planar_radius,
            "minimum_object_lift_m": self.minimum_object_lift_m,
            "minimum_object_clearance_m": self.minimum_object_clearance_m,
            "lift_max_steps": self.lift_max_steps,
            "retract_max_steps": self.retract_max_steps,
            "translation_tolerance_m": self.translation_tolerance_m,
            "orientation_tolerance_deg": self.orientation_tolerance_deg,
            "max_translation_command": self.max_translation_command,
            "max_rotation_command": self.max_rotation_command,
            "stable_steps": self.stable_steps,
            "hold_confirmation_steps": self.hold_confirmation_steps,
            "drop_confirmation_steps": self.drop_confirmation_steps,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid or self.hold_steps < 0:
            raise ValueError(
                "post-grasp transport values must be positive "
                f"(hold_steps may be zero); invalid={invalid}"
            )
        if self.hold_mode not in {"zero_delta", "closed_loop"}:
            raise ValueError(
                "post-grasp hold_mode must be 'zero_delta' or 'closed_loop'"
            )

    def _motion_kwargs(self, *, max_steps: int) -> dict[str, Any]:
        return {
            "max_steps": max_steps,
            "translation_tolerance_m": self.translation_tolerance_m,
            "orientation_tolerance_deg": self.orientation_tolerance_deg,
            "max_translation_command": self.max_translation_command,
            "max_rotation_command": self.max_rotation_command,
            "stable_steps": self.stable_steps,
            "gripper_action": 1.0,
        }

    @staticmethod
    def _motion_dropped(result: Mapping[str, Any]) -> bool:
        return bool(
            result.get("guard_failure")
            or str(result.get("failure_code", "")).startswith(
                "OBJECT_DROPPED"
            )
        )

    def _clearance_check(
        self,
        *,
        before_object: Mapping[str, Any],
        current_object: Mapping[str, Any],
    ) -> dict[str, Any]:
        before_position = before_object.get("position")
        current_position = current_object.get("position")
        measured_lift = None
        if (
            isinstance(before_position, (list, tuple))
            and isinstance(current_position, (list, tuple))
            and len(before_position) >= 3
            and len(current_position) >= 3
        ):
            measured_lift = float(current_position[2]) - float(before_position[2])
        support_clearance = current_object.get("support_clearance_m")
        lift_ok = bool(
            isinstance(measured_lift, (int, float))
            and measured_lift >= self.minimum_object_lift_m
        )
        clearance_ok = bool(
            isinstance(support_clearance, (int, float))
            and support_clearance >= self.minimum_object_clearance_m
        )
        return {
            "measured_object_lift_m": measured_lift,
            "minimum_object_lift_m": self.minimum_object_lift_m,
            "support_clearance_m": support_clearance,
            "minimum_object_clearance_m": self.minimum_object_clearance_m,
            "lift_ok": lift_ok,
            "clearance_ok": clearance_ok,
            # Either signal is sufficient: support-surface inference is not
            # always available for movable receptacles, while object body pose is.
            "safe": bool(lift_ok or clearance_ok),
        }

    def _compactness_check(
        self,
        *,
        before_eef: Mapping[str, Any],
        current_eef: Mapping[str, Any],
    ) -> dict[str, Any]:
        before_radius = float(
            np.linalg.norm(np.asarray(before_eef["position"][:2], dtype=float))
        )
        current_radius = float(
            np.linalg.norm(np.asarray(current_eef["position"][:2], dtype=float))
        )
        return {
            "before_planar_radius_m": before_radius,
            "current_planar_radius_m": current_radius,
            "achieved_retract_m": before_radius - current_radius,
            "maximum_transport_planar_radius_m": self.maximum_transport_planar_radius,
            "safe": current_radius <= self.maximum_transport_planar_radius,
        }

    def _save(self, result: Mapping[str, Any]) -> str | None:
        if self.log_dir is None:
            return None
        self._log_index += 1
        alias = str(result.get("object_alias") or "object").replace("/", "_")
        path = (
            self.log_dir
            / "post_grasp_transport"
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
        print("[POST-GRASP]")
        print(f"  object: {result.get('object_alias')}")
        print(f"  pick_verified: {result.get('pick_verified')}")
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
            "  held_after_lift: {} held_after_retract: {}".format(
                result.get("held_after_lift"),
                result.get("held_after_retract"),
            )
        )
        print(
            "  transport_ready: {} reason: {}".format(
                result.get("transport_ready"), result.get("reason")
            )
        )

    def prepare(
        self,
        *,
        env: Any,
        atomic_task_call: AtomicTaskCall | Mapping[str, Any],
        episode_id: int | str | None = None,
        pick_verified: bool | None = None,
    ) -> dict[str, Any]:
        """Execute lift, retract, and hold while continuously guarding grasp."""

        call = (
            atomic_task_call
            if isinstance(atomic_task_call, AtomicTaskCall)
            else AtomicTaskCall.from_mapping(atomic_task_call)
        )
        object_alias = resolve_object_alias(env, call)
        result: dict[str, Any] = {
            "module": "POST_GRASP_TRANSPORT",
            "episode_id": episode_id,
            "subgoal_id": call.subgoal_id,
            "object_alias": object_alias,
            "enabled": self.enabled,
            "pick_verified": bool(pick_verified),
            "transport_ready": False,
            "navigation_safe": False,
            "success": False,
            "reason": None,
            "failure_code": None,
            "fixed_carry_pose": False,
            "carry_pose_strategy": "base_relative_lift_then_planar_radial_retract",
        }
        if not self.enabled:
            result.update(
                {
                    "reason": "POST_GRASP_TRANSPORT_DISABLED",
                    "failure_code": "TRANSPORT_READY_FAILED",
                }
            )
            result["artifact_path"] = self._save(result)
            self._print_summary(result)
            return result
        if call.atomic_task != "PickObject" or object_alias is None:
            result.update(
                {
                    "reason": "PICK_OBJECT_UNRESOLVED",
                    "failure_code": "TRANSPORT_READY_FAILED",
                }
            )
            result["artifact_path"] = self._save(result)
            self._print_summary(result)
            return result

        raw_env = _raw_task_env(env)
        before = {
            "eef_pose": current_eef_pose(env),
            "object_pose": _object_snapshot(raw_env, object_alias),
            "base_pose": current_base_pose(env),
        }
        result["before"] = before
        held_before = _holding(raw_env, object_alias)
        result["held_before"] = held_before
        if pick_verified is None:
            # Backward-compatible direct use: callers predating the explicit
            # verifier handoff only invoke this primitive after a successful pick.
            result["pick_verified"] = held_before is True
        if held_before is not True:
            result.update(
                {
                    # This controller is only scheduled after the Pick verifier
                    # accepted the grasp.  Losing that verified grasp before the
                    # first lift command is therefore a transport-handoff drop,
                    # not an ordinary convergence failure.
                    "reason": "OBJECT_DROPPED_BEFORE_LIFT",
                    "failure_code": "OBJECT_DROPPED_DURING_PREP",
                    "held_after_lift": held_before,
                    "held_after_retract": held_before,
                    "after": before,
                }
            )
            result["artifact_path"] = self._save(result)
            self._print_summary(result)
            return result

        guard = build_held_object_guard(
            env=env,
            atomic_task_call=call,
            enabled=True,
            hold_confirmation_steps=self.hold_confirmation_steps,
            drop_confirmation_steps=self.drop_confirmation_steps,
        )
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
            guard=guard,
            settle_steps=0,
            **self._motion_kwargs(max_steps=self.lift_max_steps),
        )
        held_after_lift = _holding(raw_env, object_alias)
        after_lift = {
            "eef_pose": current_eef_pose(env),
            "object_pose": _object_snapshot(raw_env, object_alias),
            "base_pose": current_base_pose(env),
        }
        lift_clearance = self._clearance_check(
            before_object=before["object_pose"],
            current_object=after_lift["object_pose"],
        )
        result["after_lift"] = after_lift
        result["lift_safety_check"] = lift_clearance
        result["held_after_lift"] = held_after_lift
        result["lift"] = {
            "requested_delta_z": self.lift_height,
            **dict(lift_motion),
        }
        if self._motion_dropped(lift_motion) or held_after_lift is False:
            result.update(
                {
                    "reason": "OBJECT_DROPPED_DURING_LIFT",
                    "failure_code": "OBJECT_DROPPED_DURING_PREP",
                }
            )
        elif not lift_clearance["safe"]:
            result.update(
                {
                    "reason": "INSUFFICIENT_OBJECT_CLEARANCE_AFTER_LIFT",
                    "failure_code": "POST_GRASP_LIFT_FAILED",
                }
            )
        elif not lift_motion.get("success", False):
            result.setdefault("motion_warnings", []).append(
                "LIFT_TARGET_NOT_CONVERGED_BUT_CLEARANCE_SAFE"
            )
        if result["failure_code"] is not None:
            result["after"] = {
                "eef_pose": current_eef_pose(env),
                "object_pose": _object_snapshot(raw_env, object_alias),
                "base_pose": current_base_pose(env),
            }
            result["held_after_retract"] = held_after_lift
            result["artifact_path"] = self._save(result)
            self._print_summary(result)
            return result

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
            guard=guard,
            settle_steps=self.hold_steps,
            settle_mode=self.hold_mode,
            **self._motion_kwargs(max_steps=self.retract_max_steps),
        )
        held_after_retract = _holding(raw_env, object_alias)
        result["held_after_retract"] = held_after_retract
        result["retract"] = {
            "requested_distance": self.retract_distance,
            "effective_distance": effective_retract,
            **dict(retract_motion),
        }
        if self._motion_dropped(retract_motion) or held_after_retract is False:
            result.update(
                {
                    "reason": "OBJECT_DROPPED_DURING_RETRACT",
                    "failure_code": "OBJECT_DROPPED_DURING_PREP",
                }
            )

        after = {
            "eef_pose": current_eef_pose(env),
            "object_pose": _object_snapshot(raw_env, object_alias),
            "base_pose": current_base_pose(env),
        }
        result["after"] = after
        final_clearance = self._clearance_check(
            before_object=before["object_pose"],
            current_object=after["object_pose"],
        )
        compactness = self._compactness_check(
            before_eef=retract_start,
            current_eef=after["eef_pose"],
        )
        result["measured_object_lift_m"] = final_clearance[
            "measured_object_lift_m"
        ]
        result["transport_ready_check"] = {
            "holding": held_after_retract,
            "clearance": final_clearance,
            "compactness": compactness,
            "lift_motion_converged": bool(lift_motion.get("success", False)),
            "retract_motion_converged": bool(
                retract_motion.get("success", False)
            ),
        }
        if (
            result["failure_code"] is None
            and held_after_retract is True
            and final_clearance["safe"]
            and compactness["safe"]
        ):
            if not retract_motion.get("success", False):
                result.setdefault("motion_warnings", []).append(
                    "RETRACT_TARGET_NOT_CONVERGED_BUT_TRANSPORT_SAFE"
                )
            result.update(
                {
                    "success": True,
                    "transport_ready": True,
                    "navigation_safe": True,
                    "reason": (
                        "TRANSPORT_READY_WITH_MOTION_WARNING"
                        if result.get("motion_warnings")
                        else "TRANSPORT_READY"
                    ),
                    "failure_code": None,
                }
            )
        elif result["failure_code"] is None and held_after_retract is not True:
            result.update(
                {
                    "reason": "FINAL_HOLDING_UNVERIFIABLE",
                    "failure_code": "TRANSPORT_READY_FAILED",
                }
            )
        elif result["failure_code"] is None and not final_clearance["safe"]:
            result.update(
                {
                    "reason": "FINAL_OBJECT_CLEARANCE_UNSAFE",
                    "failure_code": "TRANSPORT_READY_FAILED",
                }
            )
        elif result["failure_code"] is None and not compactness["safe"]:
            result.update(
                {
                    "reason": "FINAL_ARM_NOT_COMPACT",
                    "failure_code": "TRANSPORT_READY_FAILED",
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
            "maximum_transport_planar_radius": self.maximum_transport_planar_radius,
            "minimum_object_lift_m": self.minimum_object_lift_m,
            "minimum_object_clearance_m": self.minimum_object_clearance_m,
            "lift_max_steps": self.lift_max_steps,
            "retract_max_steps": self.retract_max_steps,
            "translation_tolerance_m": self.translation_tolerance_m,
            "orientation_tolerance_deg": self.orientation_tolerance_deg,
            "max_translation_command": self.max_translation_command,
            "max_rotation_command": self.max_rotation_command,
            "stable_steps": self.stable_steps,
            "hold_steps": self.hold_steps,
            "hold_mode": self.hold_mode,
            "hold_confirmation_steps": self.hold_confirmation_steps,
            "drop_confirmation_steps": self.drop_confirmation_steps,
            "fixed_carry_pose": False,
        }
