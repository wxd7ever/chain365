"""Runtime grasp monitor with optional, explicitly enabled action guarding."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from .pick_motion_verification import (
        capture_pick_motion_baseline,
        check_two_finger_grasp,
        evaluate_pick_motion,
    )
except ImportError:
    from pick_motion_verification import (
        capture_pick_motion_baseline,
        check_two_finger_grasp,
        evaluate_pick_motion,
    )


_RELATION_PREDICATES = {
    "dishwasher_rack_contact",
    "inside",
    "inserted",
    "on",
    "object_fixture_relation",
}
_GRIPPER_ACTION_INDEX = 6
_CLOSE_GRIPPER_ACTION = 1.0


def _env_chain(env: Any) -> list[Any]:
    chain: list[Any] = []
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        if hasattr(current, "unwrapped") and current.unwrapped is not current:
            current = current.unwrapped
        elif hasattr(current, "env"):
            current = current.env
        else:
            break
    return chain


def _task_env(env: Any) -> Any:
    for owner in reversed(_env_chain(env)):
        if hasattr(owner, "fixtures") and hasattr(owner, "objects") and hasattr(owner, "sim"):
            return owner
    raise ValueError("could not find a RoboCasa task environment for held-object guarding")


def _normalise_identifier(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _resolve_alias(
    raw_env: Any, identifier: Any, *, expected_kind: str | None = None
) -> tuple[str | None, str | None]:
    text = str(identifier or "").strip()
    if not text:
        return None, None
    collections = (
        (expected_kind + "s",)
        if expected_kind in {"object", "fixture"}
        else ("objects", "fixtures")
    )
    for collection_name in collections:
        if text in getattr(raw_env, collection_name, {}):
            return text, collection_name[:-1]

    wanted = _normalise_identifier(text)
    matches: list[tuple[str, str]] = []
    for collection_name in collections:
        for alias, entity in getattr(raw_env, collection_name, {}).items():
            names = {
                _normalise_identifier(alias),
                _normalise_identifier(getattr(entity, "name", "")),
                _normalise_identifier(getattr(entity, "nat_lang", "")),
            }
            if wanted in names:
                matches.append((str(alias), collection_name[:-1]))
    matches = list(dict.fromkeys(matches))
    return matches[0] if len(matches) == 1 else (None, None)


def _flatten_conditions(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping)):
        result: list[dict[str, Any]] = []
        for item in value:
            result.extend(_flatten_conditions(item))
        return result
    if not isinstance(value, Mapping):
        return []
    for operator in ("all_of", "any_of"):
        if operator in value:
            return _flatten_conditions(value[operator])
    if "not" in value:
        return _flatten_conditions(value["not"])
    return [dict(value)]


def _goal_conditions(call: Any) -> list[dict[str, Any]]:
    metadata = getattr(call, "metadata", {})
    contract = metadata.get("skill_contract", {}) if isinstance(metadata, Mapping) else {}
    goal = contract.get("goal", {}) if isinstance(contract, Mapping) else {}
    conditions = goal.get("conditions") if isinstance(goal, Mapping) else None
    if not conditions:
        conditions = getattr(call, "termination_condition", None)
    return _flatten_conditions(conditions)


def _guard_requested(call: Any, conditions: Sequence[Mapping[str, Any]]) -> bool:
    arguments = getattr(call, "arguments", {})
    metadata = getattr(call, "metadata", {})
    contract = metadata.get("skill_contract", {}) if isinstance(metadata, Mapping) else {}
    family = contract.get("family") if isinstance(contract, Mapping) else None
    predicates = {str(item.get("predicate", "")).lower() for item in conditions}
    return bool(
        family == "object_transfer"
        or arguments.get("held_object_id")
        or str(getattr(call, "atomic_task", "")).startswith("PickPlace")
        or "holding" in predicates
    )


def _guarded_object_alias(
    raw_env: Any, call: Any, conditions: Sequence[Mapping[str, Any]]
) -> str | None:
    arguments = getattr(call, "arguments", {})
    identifiers = [
        arguments.get(key)
        for key in (
            "held_object_id",
            "object_id",
            "source_object_id",
            "object_name",
        )
        if arguments.get(key)
    ]
    identifiers.extend(
        item.get("subject")
        for item in conditions
        if str(item.get("predicate", "")).lower()
        in (_RELATION_PREDICATES | {"holding"})
    )
    for identifier in identifiers:
        alias, kind = _resolve_alias(raw_env, identifier, expected_kind="object")
        if alias and kind == "object":
            return alias
    return None


def _interchangeable_object_candidates(
    raw_env: Any, call: Any, canonical: str
) -> list[str]:
    metadata = getattr(call, "metadata", {})
    raw_candidates = (
        metadata.get("interchangeable_object_candidates", ())
        if isinstance(metadata, Mapping)
        else ()
    )
    if isinstance(raw_candidates, (str, bytes)) or not isinstance(
        raw_candidates, Sequence
    ):
        return [canonical]
    objects = getattr(raw_env, "objects", {})
    candidates = list(
        dict.fromkeys(
            str(alias).strip()
            for alias in raw_candidates
            if str(alias).strip() in objects
        )
    )
    return candidates if canonical in candidates else [canonical]


def _selected_interchangeable_alias(
    result: Mapping[str, Any], candidates: Sequence[str]
) -> str | None:
    evidence = result.get("state_evidence", ())
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        return None
    valid = set(candidates)
    for item in reversed(evidence):
        if not isinstance(item, Mapping):
            continue
        if item.get("source") != "interchangeable_object_pick":
            continue
        selected = str(item.get("selected_object_alias") or "").strip()
        if selected in valid:
            return selected
    return None


def _destination_condition(
    raw_env: Any,
    object_alias: str,
    conditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    for item in conditions:
        predicate = str(item.get("predicate", "")).lower()
        if predicate not in _RELATION_PREDICATES or not bool(
            item.get("desired_value", True)
        ):
            continue
        subject, subject_kind = _resolve_alias(
            raw_env, item.get("subject"), expected_kind="object"
        )
        if subject != object_alias or subject_kind != "object":
            continue
        destination_id = item.get("object", item.get("destination"))
        destination, destination_kind = _resolve_alias(raw_env, destination_id)
        if destination is None or destination_kind is None:
            continue
        condition = dict(item)
        condition["predicate"] = predicate
        condition["subject"] = object_alias
        condition["object"] = destination
        condition["object_kind"] = destination_kind
        condition.pop("destination", None)
        return condition
    return None


def _check_holding(
    raw_env: Any,
    object_alias: str,
    grasp_threshold: float | None = None,
) -> bool | None:
    try:
        from robocasa.utils import object_utils

        return bool(
            object_utils.check_obj_grasped(
                raw_env,
                object_alias,
                threshold=(
                    object_utils.GRASP_CLOSED_THRESHOLD
                    if grasp_threshold is None
                    else float(grasp_threshold)
                ),
            )
        )
    except (
        AssertionError,
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return None


def _object_center_z(raw_env: Any, object_alias: str) -> float | None:
    """Return an object center height for Pick lift gating."""

    try:
        body_id = raw_env.obj_body_id[object_alias]
        position = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None
    if position.shape != (3,) or not np.isfinite(position).all():
        return None
    return float(position[2])


def _check_destination(raw_env: Any, condition: Mapping[str, Any] | None) -> bool | None:
    if condition is None:
        return None
    try:
        from robocasa.utils import object_utils

        predicate = str(condition["predicate"])
        subject = str(condition["subject"])
        destination = str(condition["object"])
        destination_kind = str(condition["object_kind"])
        if predicate == "dishwasher_rack_contact":
            fixture = raw_env.fixtures[destination]
            checker = getattr(fixture, "check_rack_contact", None)
            if not callable(checker):
                raise AttributeError(
                    f"fixture {destination!r} has no callable check_rack_contact"
                )
            return bool(checker(raw_env, subject))
        if destination_kind == "object":
            kwargs = {}
            if condition.get("threshold") is not None:
                kwargs["th"] = float(condition["threshold"])
            return bool(
                object_utils.check_obj_in_receptacle(
                    raw_env, subject, destination, **kwargs
                )
            )
        if predicate in {"inside", "inserted"}:
            return bool(
                object_utils.obj_inside_of(
                    raw_env, subject, destination, partial_check=False
                )
            )
        return bool(
            object_utils.check_obj_fixture_contact(raw_env, subject, destination)
        )
    except (
        AssertionError,
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return None


class HeldObjectGuard:
    """Monitor a confirmed grasp and optionally override gripper actions."""

    def __init__(
        self,
        *,
        enabled: bool,
        object_alias: str | None = None,
        destination_condition: Mapping[str, Any] | None = None,
        holding_checker: Callable[[], bool | None] | None = None,
        destination_checker: Callable[[], bool | None] | None = None,
        lift_delta_checker: Callable[[], float | None] | None = None,
        minimum_lift_delta_m: float | None = None,
        pick_readiness_checker: Callable[[], bool | None] | None = None,
        pick_readiness_mode: str = "vertical_lift",
        activation_requires_verifier_success: bool = False,
        verifier_object_selector: Callable[[Mapping[str, Any]], str | None] | None = None,
        action_override_enabled: bool = False,
        hold_confirmation_steps: int = 2,
        drop_confirmation_steps: int = 30,
        reason: str = "",
    ):
        if hold_confirmation_steps <= 0 or drop_confirmation_steps <= 0:
            raise ValueError("held-object confirmation steps must be positive")
        if minimum_lift_delta_m is not None:
            minimum_lift_delta_m = float(minimum_lift_delta_m)
            if not np.isfinite(minimum_lift_delta_m) or minimum_lift_delta_m <= 0:
                raise ValueError("minimum_lift_delta_m must be positive when provided")
        self.enabled = bool(enabled)
        self.object_alias = object_alias
        self.destination_condition = (
            dict(destination_condition) if destination_condition is not None else None
        )
        self.holding_checker = holding_checker
        self.destination_checker = destination_checker
        self.lift_delta_checker = lift_delta_checker
        self.minimum_lift_delta_m = minimum_lift_delta_m
        self.pick_readiness_checker = pick_readiness_checker
        self.pick_readiness_mode = str(pick_readiness_mode)
        self.activation_requires_verifier_success = bool(
            activation_requires_verifier_success
        )
        self.verifier_object_selector = verifier_object_selector
        self.action_override_enabled = bool(action_override_enabled)
        self.hold_confirmation_steps = int(hold_confirmation_steps)
        self.drop_confirmation_steps = int(drop_confirmation_steps)
        self.reason = reason
        self.latched = False
        self.release_allowed = False
        self.holding_consecutive = 0
        self.lost_consecutive = 0
        self.latched_step: int | None = None
        self.release_allowed_step: int | None = None
        self.dropped_step: int | None = None
        self.drop_active = False
        self.drop_event_count = 0
        self.gripper_override_count = 0
        self.verifier_confirmed = False
        self.verifier_confirmed_step: int | None = None
        self.events: list[dict[str, Any]] = []
        self.trace: list[dict[str, Any]] = []
        self._pending_action: dict[str, Any] | None = None

    def _holding(self) -> bool | None:
        if not self.enabled or self.holding_checker is None:
            return None
        return self.holding_checker()

    def _destination(self) -> bool | None:
        if not self.enabled or self.destination_checker is None:
            return None
        return self.destination_checker()

    def _lift_delta(self) -> float | None:
        if self.minimum_lift_delta_m is None:
            return None
        if not self.enabled or self.lift_delta_checker is None:
            return None
        value = self.lift_delta_checker()
        if value is None:
            return None
        value = float(value)
        return value if np.isfinite(value) else None

    def _lift_ready(self, lift_delta: float | None) -> bool:
        if self.pick_readiness_checker is not None:
            try:
                readiness = self.pick_readiness_checker()
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                return False
            return readiness is True
        if self.minimum_lift_delta_m is None:
            return True
        return bool(
            lift_delta is not None
            and lift_delta >= self.minimum_lift_delta_m
        )

    def confirm_verifier_result(
        self, result: Mapping[str, Any], *, step_index: int
    ) -> bool:
        """Activate a Pick guard after the verifier confirms Pick success.

        Ordinary support-surface Picks inherit the verifier decision and do not
        reintroduce a finger-contact gate. Cabinet and drawer Picks retain the
        existing two-finger-contact plus extraction requirement.
        """

        if not self.enabled or not self.activation_requires_verifier_success:
            return self.latched
        if self.latched:
            return True
        if (
            not isinstance(result, Mapping)
            or result.get("status") != "success"
            or result.get("goal_satisfied") is False
        ):
            return False
        if self.verifier_object_selector is not None:
            selected_alias = self.verifier_object_selector(result)
            if selected_alias:
                previous_alias = self.object_alias
                self.object_alias = selected_alias
                if selected_alias != previous_alias:
                    self.events.append(
                        {
                            "step_index": int(step_index),
                            "event": "guard_bound_to_verified_object",
                            "from_object": previous_alias,
                            "object": selected_alias,
                        }
                    )
        holding = self._holding()
        lift_delta = self._lift_delta()
        lift_ready = self._lift_ready(lift_delta)
        self.verifier_confirmed = True
        self.verifier_confirmed_step = int(step_index)
        holding_required = (
            self.pick_readiness_mode == "cabinet_or_drawer_extraction"
        )
        if (holding_required and holding is not True) or not lift_ready:
            self.events.append(
                {
                    "step_index": int(step_index),
                    "event": "verifier_success_guard_activation_rejected",
                    "object": self.object_alias,
                    "two_finger_contact": holding,
                    "pick_motion_ready": lift_ready,
                    "lift_delta_m": lift_delta,
                }
            )
            return False
        self.latched = True
        self.holding_consecutive = self.hold_confirmation_steps
        self.latched_step = int(step_index)
        self.events.append(
            {
                "step_index": int(step_index),
                "event": "verifier_confirmed_grasp_latched",
                "object": self.object_alias,
                "two_finger_contact": holding,
                "two_finger_contact_required": holding_required,
                "pick_motion_ready": True,
                "lift_delta_m": lift_delta,
                "minimum_lift_delta_m": self.minimum_lift_delta_m,
            }
        )
        return True

    def _allow_release(self, step_index: int) -> None:
        if self.release_allowed:
            return
        self.release_allowed = True
        self.release_allowed_step = step_index
        self.events.append(
            {
                "step_index": step_index,
                "event": "destination_satisfied_release_allowed",
                "object": self.object_alias,
            }
        )

    def start(self) -> None:
        if not self.enabled:
            return
        destination = self._destination()
        holding = self._holding()
        lift_delta = self._lift_delta()
        lift_ready = self._lift_ready(lift_delta)
        if destination is True:
            self._allow_release(0)
        elif self.activation_requires_verifier_success:
            return
        elif holding is True and lift_ready:
            self.latched = True
            self.holding_consecutive = self.hold_confirmation_steps
            self.latched_step = 0
            self.events.append(
                {
                    "step_index": 0,
                    "event": "initial_grasp_latched",
                    "object": self.object_alias,
                    "lift_delta_m": lift_delta,
                    "minimum_lift_delta_m": self.minimum_lift_delta_m,
                }
            )
        elif holding is True and not lift_ready:
            self.events.append(
                {
                    "step_index": 0,
                    "event": "initial_grasp_waiting_for_lift",
                    "object": self.object_alias,
                    "lift_delta_m": lift_delta,
                    "minimum_lift_delta_m": self.minimum_lift_delta_m,
                }
            )

    def apply_action(self, action: np.ndarray, *, step_index: int) -> np.ndarray:
        array = np.asarray(action)
        if array.shape != (12,):
            raise ValueError(f"held-object guard expected a 12-D action, got {array.shape}")
        guarded = array.copy()
        destination = self._destination()
        if destination is True:
            self._allow_release(step_index)
        override = bool(
            self.action_override_enabled
            and self.enabled
            and self.latched
            and not self.release_allowed
        )
        if override:
            guarded[_GRIPPER_ACTION_INDEX] = _CLOSE_GRIPPER_ACTION
            if float(array[_GRIPPER_ACTION_INDEX]) != _CLOSE_GRIPPER_ACTION:
                self.gripper_override_count += 1
        self._pending_action = {
            "step_index": step_index,
            "raw_gripper_action": float(array[_GRIPPER_ACTION_INDEX]),
            "applied_gripper_action": float(guarded[_GRIPPER_ACTION_INDEX]),
            "gripper_overridden": override,
        }
        return guarded

    def observe(self, *, step_index: int) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        holding = self._holding()
        destination = self._destination()
        lift_delta = self._lift_delta()
        lift_ready = self._lift_ready(lift_delta)
        if destination is True:
            self._allow_release(step_index)

        if (
            not self.latched
            and not self.release_allowed
            and not self.activation_requires_verifier_success
        ):
            self.holding_consecutive = (
                self.holding_consecutive + 1
                if holding is True and lift_ready
                else 0
            )
            if self.holding_consecutive >= self.hold_confirmation_steps:
                self.latched = True
                self.latched_step = step_index
                self.events.append(
                    {
                        "step_index": step_index,
                        "event": "grasp_latched",
                        "object": self.object_alias,
                        "holding_consecutive": self.holding_consecutive,
                        "lift_delta_m": lift_delta,
                        "minimum_lift_delta_m": self.minimum_lift_delta_m,
                    }
                )

        if self.latched and not self.release_allowed:
            if holding is False:
                self.lost_consecutive += 1
            elif holding is True:
                self.lost_consecutive = 0
                if self.drop_active:
                    self.drop_active = False
                    self.events.append(
                        {
                            "step_index": step_index,
                            "event": "object_reacquired_after_drop",
                            "object": self.object_alias,
                        }
                    )

        trace = dict(self._pending_action or {"step_index": step_index})
        trace.update(
            {
                "holding": holding,
                "lift_delta_m": lift_delta,
                "pick_readiness_mode": self.pick_readiness_mode,
                "minimum_lift_delta_m": self.minimum_lift_delta_m,
                "lift_ready": lift_ready,
                "activation_requires_verifier_success": (
                    self.activation_requires_verifier_success
                ),
                "verifier_confirmed": self.verifier_confirmed,
                "destination_satisfied": destination,
                "latched": self.latched,
                "release_allowed": self.release_allowed,
                "lost_consecutive": self.lost_consecutive,
            }
        )
        self.trace.append(trace)
        self._pending_action = None

        if (
            self.latched
            and not self.release_allowed
            and self.lost_consecutive >= self.drop_confirmation_steps
            and not self.drop_active
        ):
            self.drop_active = True
            self.drop_event_count += 1
            if self.dropped_step is None:
                self.dropped_step = step_index
            evidence = {
                "source": "held_object_guard",
                "object": self.object_alias,
                "latched_step": self.latched_step,
                "drop_step": step_index,
                "lost_consecutive": self.lost_consecutive,
                "drop_confirmation_steps": self.drop_confirmation_steps,
                "destination_condition": self.destination_condition,
                "destination_satisfied": destination,
                "gripper_override_count": self.gripper_override_count,
            }
            self.events.append(
                {
                    "step_index": step_index,
                    "event": "object_dropped",
                    "object": self.object_alias,
                }
            )
            return {
                "status": "failed",
                "goal_satisfied": False,
                "failure_code": "OBJECT_DROPPED",
                "retryable": True,
                "state_evidence": [{"step_index": step_index}, evidence],
            }
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "reason": self.reason,
            "object_alias": self.object_alias,
            "destination_condition": self.destination_condition,
            "minimum_lift_delta_m": self.minimum_lift_delta_m,
            "pick_readiness_mode": self.pick_readiness_mode,
            "lift_gate_enabled": self.minimum_lift_delta_m is not None,
            "activation_requires_verifier_success": (
                self.activation_requires_verifier_success
            ),
            "verifier_confirmed": self.verifier_confirmed,
            "verifier_confirmed_step": self.verifier_confirmed_step,
            "hold_confirmation_steps": self.hold_confirmation_steps,
            "drop_confirmation_steps": self.drop_confirmation_steps,
            "latched": self.latched,
            "latched_step": self.latched_step,
            "release_allowed": self.release_allowed,
            "release_allowed_step": self.release_allowed_step,
            "dropped_step": self.dropped_step,
            "drop_active": self.drop_active,
            "drop_event_count": self.drop_event_count,
            "action_override_enabled": self.action_override_enabled,
            "gripper_override_count": self.gripper_override_count,
            "events": self.events,
            "gripper_action_trace": self.trace,
        }


def build_held_object_guard(
    *,
    env: Any,
    atomic_task_call: Any,
    enabled: bool = True,
    hold_confirmation_steps: int = 2,
    drop_confirmation_steps: int = 30,
    action_override_enabled: bool = False,
    grasp_threshold: float | None = None,
    pick_min_lift_delta_m: float | None = None,
) -> HeldObjectGuard:
    """Build a guard only for object-transfer or explicit held-object calls."""

    if not enabled:
        return HeldObjectGuard(
            enabled=False,
            hold_confirmation_steps=hold_confirmation_steps,
            drop_confirmation_steps=drop_confirmation_steps,
            reason="disabled_by_configuration",
        )
    conditions = _goal_conditions(atomic_task_call)
    if not _guard_requested(atomic_task_call, conditions):
        return HeldObjectGuard(
            enabled=False,
            hold_confirmation_steps=hold_confirmation_steps,
            drop_confirmation_steps=drop_confirmation_steps,
            reason="atomic_task_does_not_transport_or_hold_an_object",
        )
    try:
        raw_env = _task_env(env)
    except ValueError as exc:
        return HeldObjectGuard(
            enabled=False,
            hold_confirmation_steps=hold_confirmation_steps,
            drop_confirmation_steps=drop_confirmation_steps,
            reason=str(exc),
        )
    object_alias = _guarded_object_alias(raw_env, atomic_task_call, conditions)
    if object_alias is None:
        return HeldObjectGuard(
            enabled=False,
            hold_confirmation_steps=hold_confirmation_steps,
            drop_confirmation_steps=drop_confirmation_steps,
            reason="could_not_resolve_guarded_object",
        )
    destination = _destination_condition(raw_env, object_alias, conditions)
    is_pick = str(getattr(atomic_task_call, "atomic_task", "")) == "PickObject"
    minimum_lift_delta_m = (
        pick_min_lift_delta_m
        if is_pick
        else None
    )
    candidate_aliases = _interchangeable_object_candidates(
        raw_env, atomic_task_call, object_alias
    )
    active = {"alias": object_alias}
    pick_motion_baselines = {
        alias: capture_pick_motion_baseline(
            raw_env=raw_env,
            object_alias=alias,
            atomic_task_call=atomic_task_call,
        )
        for alias in candidate_aliases
    } if minimum_lift_delta_m is not None else {}

    def active_baseline() -> Mapping[str, Any] | None:
        return pick_motion_baselines.get(active["alias"])

    def lift_delta_checker() -> float | None:
        baseline = active_baseline()
        if baseline is None:
            return None
        current_z = _object_center_z(raw_env, active["alias"])
        if current_z is None:
            return None
        return float(current_z - float(baseline["initial_center_z_m"]))

    def pick_readiness_checker() -> bool | None:
        baseline = active_baseline()
        if baseline is None or minimum_lift_delta_m is None:
            return None
        ready, _ = evaluate_pick_motion(
            raw_env=raw_env,
            object_alias=active["alias"],
            baseline=baseline,
            minimum_delta_m=float(minimum_lift_delta_m),
        )
        return ready

    def holding_checker() -> bool | None:
        if is_pick:
            value, _ = check_two_finger_grasp(raw_env, active["alias"])
            return value
        return _check_holding(raw_env, active["alias"], grasp_threshold)

    def verifier_object_selector(result: Mapping[str, Any]) -> str | None:
        selected = _selected_interchangeable_alias(result, candidate_aliases)
        if selected is not None:
            active["alias"] = selected
        return selected

    pick_motion_baseline = active_baseline()
    pick_readiness_mode = (
        "cabinet_or_drawer_extraction"
        if pick_motion_baseline is not None
        and pick_motion_baseline.get("source_is_cabinet_or_drawer")
        else "support_surface_departure"
    )

    return HeldObjectGuard(
        enabled=True,
        object_alias=object_alias,
        destination_condition=destination,
        holding_checker=holding_checker,
        destination_checker=lambda: _check_destination(raw_env, destination),
        lift_delta_checker=(
            lift_delta_checker if minimum_lift_delta_m is not None else None
        ),
        minimum_lift_delta_m=minimum_lift_delta_m,
        pick_readiness_checker=(
            pick_readiness_checker if pick_motion_baseline is not None else None
        ),
        pick_readiness_mode=pick_readiness_mode,
        activation_requires_verifier_success=is_pick,
        verifier_object_selector=(
            verifier_object_selector if len(candidate_aliases) > 1 else None
        ),
        action_override_enabled=action_override_enabled,
        hold_confirmation_steps=hold_confirmation_steps,
        drop_confirmation_steps=drop_confirmation_steps,
        reason="guarding_resolved_object",
    )
