"""Runtime guard that keeps a confirmed grasp closed until placement."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np


_RELATION_PREDICATES = {"inside", "inserted", "on", "object_fixture_relation"}
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


def _check_holding(raw_env: Any, object_alias: str) -> bool | None:
    try:
        from robocasa.utils import object_utils

        return bool(
            object_utils.check_obj_grasped(raw_env, object_alias, threshold=0.05)
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


def _check_destination(raw_env: Any, condition: Mapping[str, Any] | None) -> bool | None:
    if condition is None:
        return None
    try:
        from robocasa.utils import object_utils

        predicate = str(condition["predicate"])
        subject = str(condition["subject"])
        destination = str(condition["object"])
        destination_kind = str(condition["object_kind"])
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
    """Latch a confirmed Panda grasp until the destination relation is true."""

    def __init__(
        self,
        *,
        enabled: bool,
        object_alias: str | None = None,
        destination_condition: Mapping[str, Any] | None = None,
        holding_checker: Callable[[], bool | None] | None = None,
        destination_checker: Callable[[], bool | None] | None = None,
        hold_confirmation_steps: int = 2,
        drop_confirmation_steps: int = 2,
        reason: str = "",
    ):
        if hold_confirmation_steps <= 0 or drop_confirmation_steps <= 0:
            raise ValueError("held-object confirmation steps must be positive")
        self.enabled = bool(enabled)
        self.object_alias = object_alias
        self.destination_condition = (
            dict(destination_condition) if destination_condition is not None else None
        )
        self.holding_checker = holding_checker
        self.destination_checker = destination_checker
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
        self.gripper_override_count = 0
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
        if destination is True:
            self._allow_release(0)
        elif holding is True:
            self.latched = True
            self.holding_consecutive = self.hold_confirmation_steps
            self.latched_step = 0
            self.events.append(
                {
                    "step_index": 0,
                    "event": "initial_grasp_latched",
                    "object": self.object_alias,
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
        override = bool(self.enabled and self.latched and not self.release_allowed)
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
        if destination is True:
            self._allow_release(step_index)

        if not self.latched and not self.release_allowed:
            self.holding_consecutive = (
                self.holding_consecutive + 1 if holding is True else 0
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
                    }
                )

        if self.latched and not self.release_allowed:
            if holding is False:
                self.lost_consecutive += 1
            elif holding is True:
                self.lost_consecutive = 0

        trace = dict(self._pending_action or {"step_index": step_index})
        trace.update(
            {
                "holding": holding,
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
        ):
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
            "hold_confirmation_steps": self.hold_confirmation_steps,
            "drop_confirmation_steps": self.drop_confirmation_steps,
            "latched": self.latched,
            "latched_step": self.latched_step,
            "release_allowed": self.release_allowed,
            "release_allowed_step": self.release_allowed_step,
            "dropped_step": self.dropped_step,
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
    drop_confirmation_steps: int = 2,
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
    return HeldObjectGuard(
        enabled=True,
        object_alias=object_alias,
        destination_condition=destination,
        holding_checker=lambda: _check_holding(raw_env, object_alias),
        destination_checker=lambda: _check_destination(raw_env, destination),
        hold_confirmation_steps=hold_confirmation_steps,
        drop_confirmation_steps=drop_confirmation_steps,
        reason="guarding_resolved_object",
    )
