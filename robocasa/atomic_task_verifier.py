"""Runtime verification of structured atomic-task termination conditions."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Mapping

import numpy as np

try:
    from .atomic_task_schemas import AtomicTaskCall
except ImportError:  # Direct execution from the robocasa script directory.
    from atomic_task_schemas import AtomicTaskCall


_VALID_PREDICATES = {
    "closed",
    "eef_outside_fixture",
    "fixture_state",
    "gripper_far",
    "holding",
    "inserted",
    "inside",
    "navigation_pose",
    "object_fixture_relation",
    "on",
    "open",
    "powered",
    "receptacle_count",
    "released",
    "pressed",
    "turned",
}


def _env_chain(env: Any) -> list[Any]:
    """Collect wrappers once so predicate branches never hard-code env.env.env."""

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


def _task_env(chain: list[Any]) -> Any:
    """Select the RoboCasa task object that owns fixtures, objects, and sim."""

    for owner in reversed(chain):
        if hasattr(owner, "fixtures") and hasattr(owner, "objects") and hasattr(owner, "sim"):
            return owner
    return chain[-1]


def _call_state_method(method: Callable[..., Any], raw_env: Any) -> Any:
    parameters = [
        parameter
        for parameter in inspect.signature(method).parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is parameter.empty
    ]
    if not parameters:
        return method()
    if parameters[0].name == "sim":
        return method(raw_env.sim)
    return method(raw_env)


def _normalise_identifier(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _candidate_names(key: str, value: Any) -> set[str]:
    names = {_normalise_identifier(key), _normalise_identifier(value)}
    if value is not None:
        for attr in ("name", "nat_lang"):
            names.add(_normalise_identifier(getattr(value, attr, "")))
    expanded = set(names)
    for name in names:
        if name:
            expanded.add(name.rsplit("_", 1)[0] if name.rsplit("_", 1)[-1].isdigit() else name)
    return {name for name in expanded if name}


def _resolve_entity(chain: list[Any], identifier: str) -> tuple[Any | None, str | None]:
    wanted = _normalise_identifier(identifier)
    wanted_base = wanted.rsplit("_", 1)[0] if wanted.rsplit("_", 1)[-1].isdigit() else wanted
    for owner in chain:
        if hasattr(owner, identifier):
            return getattr(owner, identifier), identifier
        for collection_name in ("fixtures", "objects"):
            collection = getattr(owner, collection_name, None)
            if not isinstance(collection, Mapping):
                continue
            for key, value in collection.items():
                names = _candidate_names(str(key), value)
                if wanted in names or wanted_base in names:
                    return value, str(key)
    return None, None


def _mapping_state(
    chain: list[Any], info: Mapping[str, Any], predicate: str, subject: str, object_id: str | None
) -> tuple[bool | None, list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    keys = [
        (predicate, subject, object_id),
        f"{predicate}:{subject}:{object_id}" if object_id else f"{predicate}:{subject}",
        predicate,
    ]
    sources: list[tuple[str, Any]] = [("info.atomic_state", info.get("atomic_state"))]
    for index, owner in enumerate(chain):
        sources.extend(
            (
                (f"env[{index}].atomic_state", getattr(owner, "atomic_state", None)),
                (f"env[{index}].world_state", getattr(owner, "world_state", None)),
            )
        )
    for source_name, source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            if key in source and isinstance(source[key], (bool, int, float)):
                value = bool(source[key])
                evidence.append({"source": source_name, "key": str(key), "value": value})
                return value, evidence
        subject_state = source.get(subject)
        if isinstance(subject_state, Mapping) and predicate in subject_state:
            value = subject_state[predicate]
            if isinstance(value, (bool, int, float)):
                evidence.append(
                    {"source": source_name, "key": f"{subject}.{predicate}", "value": bool(value)}
                )
                return bool(value), evidence
    return None, evidence


def _custom_state(
    chain: list[Any], predicate: str, subject: str, object_id: str | None
) -> tuple[bool | None, list[dict[str, Any]]]:
    for index, owner in enumerate(chain):
        method = getattr(owner, "get_atomic_task_state", None)
        if not callable(method):
            continue
        value = method(predicate=predicate, subject=subject, object=object_id)
        if isinstance(value, Mapping):
            state = value.get("value")
            evidence = dict(value)
        else:
            state = value
            evidence = {"value": value}
        if isinstance(state, (bool, int, float)):
            evidence.update({"source": f"env[{index}].get_atomic_task_state"})
            return bool(state), [evidence]
    return None, []


def _fixture_state(
    chain: list[Any], predicate: str, subject: str, desired_state: Any
) -> tuple[bool | None, list[dict[str, Any]]]:
    entity, alias = _resolve_entity(chain, subject)
    if entity is None:
        return None, [{"source": "fixture_lookup", "subject": subject, "resolved": False}]
    raw_env = _task_env(chain)
    if predicate == "fixture_state" and isinstance(desired_state, str):
        state_name = desired_state.strip().lower()
        if state_name in {"open", "closed"}:
            method = getattr(entity, f"is_{state_name}", None)
            if callable(method):
                value = bool(_call_state_method(method, raw_env))
                return value, [
                    {
                        "source": "fixture_method",
                        "entity": alias,
                        "desired_state": state_name,
                        "value": value,
                    }
                ]
    if predicate in {"open", "closed"}:
        method = getattr(entity, f"is_{predicate}", None)
        if callable(method):
            value = bool(_call_state_method(method, raw_env))
            return value, [{"source": "fixture_method", "entity": alias, "value": value}]
    get_state = getattr(entity, "get_state", None)
    if not callable(get_state):
        return None, [{"source": "fixture_lookup", "entity": alias, "state_api": False}]
    state = _call_state_method(get_state, raw_env)
    if not isinstance(state, Mapping):
        return None, [{"source": "fixture_state", "entity": alias, "mapping": False}]
    key_candidates = {
        "powered": ("turned_on", "powered", "power", "on"),
        "pressed": ("pressed", "button_pressed"),
        "turned": ("turned", "angle", "position"),
        "inserted": ("inserted",),
        "fixture_state": (str(desired_state),),
    }.get(predicate, (predicate,))
    for key in key_candidates:
        if key in state:
            raw_value = state[key]
            if isinstance(raw_value, (bool, int, float)):
                value = bool(raw_value)
                return value, [
                    {"source": "fixture_state", "entity": alias, "key": key, "value": raw_value}
                ]
            if predicate == "fixture_state" and isinstance(raw_value, str):
                value = raw_value.strip().lower() == str(desired_state).strip().lower()
                return value, [
                    {"source": "fixture_state", "entity": alias, "key": key, "value": raw_value}
                ]
    return None, [
        {"source": "fixture_state", "entity": alias, "available_keys": sorted(map(str, state))}
    ]


def _object_relation(
    chain: list[Any], predicate: str, subject: str, object_id: str | None
) -> tuple[bool | None, list[dict[str, Any]]]:
    if not object_id and predicate != "holding":
        return None, [{"source": "relation", "error": "missing object/destination"}]
    subject_entity, subject_alias = _resolve_entity(chain, subject)
    object_entity, object_alias = _resolve_entity(chain, object_id or "")
    raw_env = _task_env(chain)
    try:
        from robocasa.utils import object_utils as object_utils
    except (ImportError, ModuleNotFoundError, AssertionError):
        object_utils = None
    if object_utils is None:
        return None, [{"source": "object_utils", "available": False}]
    try:
        if predicate == "holding" and subject_alias:
            value = bool(object_utils.check_obj_grasped(raw_env, subject_alias))
        elif predicate in {"inside", "inserted"} and subject_alias and object_entity is not None:
            value = bool(
                object_utils.obj_inside_of(
                    raw_env, subject_alias, object_entity, partial_check=True
                )
            )
        elif predicate in {"on", "object_fixture_relation"} and subject_alias and object_entity is not None:
            value = bool(
                object_utils.check_obj_fixture_contact(raw_env, subject_alias, object_entity)
            )
        else:
            return None, [
                {
                    "source": "relation_lookup",
                    "subject": subject,
                    "object": object_id,
                    "subject_resolved": subject_entity is not None,
                    "object_resolved": object_entity is not None,
                }
            ]
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return None, [{"source": "object_utils", "error": str(exc)}]
    return value, [
        {
            "source": "simulator_relation",
            "predicate": predicate,
            "subject": subject_alias,
            "object": object_alias,
            "value": value,
        }
    ]


def _receptacle_count(
    chain: list[Any],
    receptacle_id: str,
    object_prefix: str,
    desired_count: int,
) -> tuple[bool | None, list[dict[str, Any]]]:
    _, receptacle_alias = _resolve_entity(chain, receptacle_id)
    if receptacle_alias is None:
        return None, [
            {
                "source": "receptacle_count",
                "receptacle": receptacle_id,
                "resolved": False,
            }
        ]
    raw_env = _task_env(chain)
    normalized_prefix = _normalise_identifier(object_prefix)
    candidates = sorted(
        str(alias)
        for alias in getattr(raw_env, "objects", {})
        if _normalise_identifier(alias).startswith(normalized_prefix)
    )
    if not candidates:
        return None, [
            {
                "source": "receptacle_count",
                "receptacle": receptacle_alias,
                "object_prefix": object_prefix,
                "candidates": [],
            }
        ]
    try:
        from robocasa.utils import object_utils as object_utils
        inside = [
            alias
            for alias in candidates
            if object_utils.check_obj_in_receptacle(
                raw_env, alias, receptacle_alias, th=0.5
            )
        ]
    except (AttributeError, ImportError, KeyError, TypeError, ValueError) as exc:
        return None, [{"source": "receptacle_count", "error": str(exc)}]
    count = len(inside)
    value = count == desired_count
    return value, [
        {
            "source": "receptacle_count",
            "receptacle": receptacle_alias,
            "object_prefix": object_prefix,
            "candidates": candidates,
            "inside": inside,
            "count": count,
            "desired_count": desired_count,
            "value": value,
        }
    ]


def _matching_object_aliases(chain: list[Any], object_prefix: str) -> list[str]:
    raw_env = _task_env(chain)
    normalized_prefix = _normalise_identifier(object_prefix)
    return sorted(
        str(alias)
        for alias in getattr(raw_env, "objects", {})
        if _normalise_identifier(alias).startswith(normalized_prefix)
    )


def _released(
    chain: list[Any], subject: str, object_prefix: str | None
) -> tuple[bool | None, list[dict[str, Any]]]:
    """Check that the named object, or every object in a group, is ungrasped."""

    raw_env = _task_env(chain)
    if object_prefix:
        aliases = _matching_object_aliases(chain, object_prefix)
    else:
        _, alias = _resolve_entity(chain, subject)
        aliases = (
            [alias]
            if alias is not None and alias in getattr(raw_env, "objects", {})
            else []
        )
    if not aliases:
        return None, [
            {
                "source": "released",
                "subject": subject,
                "object_prefix": object_prefix,
                "resolved": False,
            }
        ]
    try:
        from robocasa.utils import object_utils as object_utils

        held = [
            alias
            for alias in aliases
            if object_utils.check_obj_grasped(raw_env, alias)
        ]
    except (AttributeError, ImportError, KeyError, TypeError, ValueError) as exc:
        return None, [{"source": "released", "error": str(exc)}]
    value = not held
    return value, [
        {
            "source": "released",
            "subject": subject,
            "object_prefix": object_prefix,
            "candidates": aliases,
            "held": held,
            "value": value,
        }
    ]


def _gripper_far(
    chain: list[Any], subject: str, threshold: float
) -> tuple[bool | None, list[dict[str, Any]]]:
    entity, alias = _resolve_entity(chain, subject)
    if entity is None or alias is None:
        return None, [
            {"source": "gripper_far", "subject": subject, "resolved": False}
        ]
    raw_env = _task_env(chain)
    try:
        from robocasa.utils import object_utils as object_utils

        if alias in getattr(raw_env, "objects", {}):
            value = bool(object_utils.gripper_obj_far(raw_env, alias, th=threshold))
            entity_kind = "object"
        elif alias in getattr(raw_env, "fixtures", {}):
            fixture_name = str(getattr(entity, "name", alias))
            value = bool(
                object_utils.gripper_fxtr_far(raw_env, fixture_name, th=threshold)
            )
            entity_kind = "fixture"
        else:
            return None, [
                {
                    "source": "gripper_far",
                    "subject": subject,
                    "entity": alias,
                    "supported": False,
                }
            ]
    except (AttributeError, ImportError, KeyError, TypeError, ValueError) as exc:
        return None, [
            {"source": "gripper_far", "entity": alias, "error": str(exc)}
        ]
    return value, [
        {
            "source": "gripper_far",
            "entity": alias,
            "entity_kind": entity_kind,
            "threshold": threshold,
            "value": value,
        }
    ]


def _eef_site_position(raw_env: Any) -> np.ndarray:
    eef_site_id = raw_env.robots[0].eef_site_id
    if isinstance(eef_site_id, Mapping):
        site_id = eef_site_id.get("right")
        if site_id is None and len(eef_site_id) == 1:
            site_id = next(iter(eef_site_id.values()))
    else:
        site_id = eef_site_id
    if site_id is None:
        raise KeyError("robot does not expose a right end-effector site")
    return np.asarray(raw_env.sim.data.site_xpos[site_id], dtype=float)


def _point_inside_fixture_with_margin(
    point: np.ndarray, fixture: Any, margin: float, only_2d: bool
) -> bool:
    """Equivalent to object_utils.point_in_fixture with a metric safety margin."""

    p0, px, py, pz = (
        np.asarray(value, dtype=float)
        for value in fixture.get_ext_sites(relative=False)
    )
    axes = (px - p0, py - p0, pz - p0)
    endpoints = (px, py, pz)
    checks = []
    for axis, endpoint in zip(axes, endpoints):
        axis_length = float(np.linalg.norm(axis))
        if axis_length <= 0:
            raise ValueError("fixture exterior bounding box has a zero-length axis")
        projection = float(np.dot(axis, point))
        checks.append(
            float(np.dot(axis, p0)) - margin * axis_length
            <= projection
            <= float(np.dot(axis, endpoint)) + margin * axis_length
        )
    return bool(all(checks[:2] if only_2d else checks))


def _eef_outside_fixture(
    chain: list[Any], subject: str, margin: float, only_2d: bool
) -> tuple[bool | None, list[dict[str, Any]]]:
    fixture, alias = _resolve_entity(chain, subject)
    raw_env = _task_env(chain)
    if fixture is None or alias not in getattr(raw_env, "fixtures", {}):
        return None, [
            {
                "source": "eef_outside_fixture",
                "subject": subject,
                "resolved_fixture": False,
            }
        ]
    try:
        point = _eef_site_position(raw_env)
        inside = _point_inside_fixture_with_margin(point, fixture, margin, only_2d)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return None, [
            {"source": "eef_outside_fixture", "entity": alias, "error": str(exc)}
        ]
    value = not inside
    return value, [
        {
            "source": "eef_outside_fixture",
            "entity": alias,
            "eef_position": point.tolist(),
            "margin": margin,
            "only_2d": only_2d,
            "inside_expanded_fixture": inside,
            "value": value,
        }
    ]

def _navigation_pose(
    chain: list[Any], subject: str
) -> tuple[bool | None, list[dict[str, Any]]]:
    target_fixture, target_alias = _resolve_entity(chain, subject)
    if target_fixture is None:
        return None, [{"source": "navigation_target", "subject": subject, "resolved": False}]
    raw_env = _task_env(chain)
    try:
        if (
            getattr(raw_env, "target_fixture", None) is target_fixture
            and hasattr(raw_env, "target_pos")
            and hasattr(raw_env, "target_ori")
        ):
            target_pos = np.asarray(raw_env.target_pos, dtype=float)
            target_ori = np.asarray(raw_env.target_ori, dtype=float)
        else:
            from robocasa.utils import env_utils as env_utils
            target_pos, target_ori = env_utils.compute_robot_base_placement_pose(
                raw_env, target_fixture
            )
        robot_id = raw_env.sim.model.body_name2id("mobilebase0_base")
        base_pos = np.asarray(raw_env.sim.data.body_xpos[robot_id], dtype=float)
        from robosuite.utils import transform_utils as transform_utils
        base_ori = transform_utils.mat2euler(
            np.asarray(raw_env.sim.data.body_xmat[robot_id]).reshape((3, 3))
        )
        position_distance = float(np.linalg.norm(target_pos[:2] - base_pos[:2]))
        orientation_cosine = float(np.cos(target_ori[2] - base_ori[2]))
        position_ok = position_distance <= 0.20
        orientation_ok = orientation_cosine >= 0.98
    except (AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        return None, [{"source": "navigation_pose", "entity": target_alias, "error": str(exc)}]
    value = bool(position_ok and orientation_ok)
    return value, [
        {
            "source": "navigation_pose",
            "entity": target_alias,
            "target_position": np.asarray(target_pos).tolist(),
            "base_position": base_pos.tolist(),
            "position_distance": position_distance,
            "position_threshold": 0.20,
            "orientation_cosine": orientation_cosine,
            "orientation_cosine_threshold": 0.98,
            "position_ok": bool(position_ok),
            "orientation_ok": bool(orientation_ok),
            "value": value,
        }
    ]


def _result(
    status: str,
    *,
    goal_satisfied: bool,
    evidence: list[dict[str, Any]],
    failure_code: str | None = None,
    retryable: bool = True,
) -> dict[str, Any]:
    return {
        "status": status,
        "goal_satisfied": goal_satisfied,
        "failure_code": failure_code,
        "retryable": retryable,
        "state_evidence": evidence,
    }


class RuntimeAtomicTaskVerifier:
    """Evaluate generic predicates using current privileged simulator state."""

    def __init__(self, *, default_required_consecutive_successes: int = 1):
        if (
            isinstance(default_required_consecutive_successes, bool)
            or not isinstance(default_required_consecutive_successes, int)
            or default_required_consecutive_successes <= 0
        ):
            raise ValueError(
                "default_required_consecutive_successes must be a positive integer"
            )
        self.default_required_consecutive_successes = (
            default_required_consecutive_successes
        )
        self._consecutive_state: dict[tuple[int, str], dict[str, int]] = {}

    def _evaluate_condition(
        self,
        *,
        env: Any,
        condition: Mapping[str, Any],
        info: Mapping[str, Any],
    ) -> tuple[bool | None, list[dict[str, Any]]]:
        for operator in ("all_of", "any_of"):
            if operator in condition:
                children = condition[operator]
                if not isinstance(children, list) or not all(
                    isinstance(item, Mapping) for item in children
                ):
                    raise ValueError(f"termination_condition.{operator} must be a list of mappings")
                evaluated = [
                    self._evaluate_condition(env=env, condition=item, info=info)
                    for item in children
                ]
                values = [value for value, _ in evaluated]
                evidence = [item for _, items in evaluated for item in items]
                if operator == "all_of":
                    value = False if False in values else (True if all(v is True for v in values) else None)
                else:
                    value = True if True in values else (False if all(v is False for v in values) else None)
                return value, evidence

        predicate = str(condition.get("predicate", "")).strip().lower().replace(" ", "_")
        if predicate not in _VALID_PREDICATES:
            raise ValueError(f"Unsupported atomic termination predicate {predicate!r}")
        subject = str(condition.get("subject", "")).strip()
        if not subject:
            raise ValueError("termination condition requires a non-empty subject")
        object_id = condition.get("object", condition.get("destination"))
        object_id = str(object_id).strip() if object_id is not None else None
        desired = condition.get("desired_value", True)
        if not isinstance(desired, (bool, int, float, str)):
            raise TypeError("termination condition desired_value must be scalar")
        chain = _env_chain(env)
        if predicate == "receptacle_count":
            if isinstance(desired, bool) or not isinstance(desired, int) or desired < 0:
                raise ValueError("receptacle_count desired_value must be a non-negative integer")
            object_prefix = str(condition.get("object_prefix", "")).strip()
            if not object_prefix:
                raise ValueError("receptacle_count requires a non-empty object_prefix")
            return _receptacle_count(
                chain,
                subject,
                object_prefix,
                int(desired),
            )

        threshold = 0.25
        margin = 0.0
        only_2d = False
        object_prefix = None
        if predicate == "released":
            raw_prefix = str(condition.get("object_prefix", "")).strip()
            object_prefix = raw_prefix or None
        elif predicate == "gripper_far":
            raw_threshold = condition.get("threshold", 0.25)
            if isinstance(raw_threshold, bool):
                raise ValueError("gripper_far threshold must be a positive number")
            try:
                threshold = float(raw_threshold)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "gripper_far threshold must be a positive number"
                ) from exc
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError("gripper_far threshold must be a positive number")
        elif predicate == "eef_outside_fixture":
            raw_margin = condition.get("margin", 0.0)
            if isinstance(raw_margin, bool):
                raise ValueError(
                    "eef_outside_fixture margin must be a non-negative number"
                )
            try:
                margin = float(raw_margin)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "eef_outside_fixture margin must be a non-negative number"
                ) from exc
            if not np.isfinite(margin) or margin < 0:
                raise ValueError(
                    "eef_outside_fixture margin must be a non-negative number"
                )
            only_2d = condition.get("only_2d", False)
            if not isinstance(only_2d, bool):
                raise ValueError("eef_outside_fixture only_2d must be boolean")

        value, evidence = _custom_state(chain, predicate, subject, object_id)
        if value is None:
            value, mapping_evidence = _mapping_state(chain, info, predicate, subject, object_id)
            evidence.extend(mapping_evidence)
        if value is None and predicate == "released":
            value, predicate_evidence = _released(chain, subject, object_prefix)
            evidence.extend(predicate_evidence)
        if value is None and predicate == "gripper_far":
            value, predicate_evidence = _gripper_far(chain, subject, threshold)
            evidence.extend(predicate_evidence)
        if value is None and predicate == "eef_outside_fixture":
            value, predicate_evidence = _eef_outside_fixture(
                chain, subject, margin, only_2d
            )
            evidence.extend(predicate_evidence)

        if value is None and predicate in {"inside", "inserted", "on", "holding", "object_fixture_relation"}:
            value, relation_evidence = _object_relation(chain, predicate, subject, object_id)
            evidence.extend(relation_evidence)
        if value is None and predicate == "navigation_pose":
            value, navigation_evidence = _navigation_pose(chain, subject)
            evidence.extend(navigation_evidence)
        if value is None:
            value, fixture_evidence = _fixture_state(chain, predicate, subject, desired)
            evidence.extend(fixture_evidence)
        if value is None:
            return None, evidence
        expected = bool(desired) if not isinstance(desired, str) else True
        return value == expected, evidence

    def _required_consecutive_successes(self, call: AtomicTaskCall) -> int:
        contract = call.metadata.get("skill_contract", {})
        verification = (
            contract.get("verification", {})
            if isinstance(contract, Mapping)
            else {}
        )
        required = (
            verification.get(
                "required_consecutive_successes",
                self.default_required_consecutive_successes,
            )
            if isinstance(verification, Mapping)
            else self.default_required_consecutive_successes
        )
        if isinstance(required, bool) or not isinstance(required, int) or required <= 0:
            raise ValueError(
                "skill contract required_consecutive_successes must be a positive integer"
            )
        return required

    def _record_consecutive_result(
        self,
        *,
        env: Any,
        subgoal_id: str,
        step_index: int,
        satisfied: bool,
    ) -> int:
        key = (id(env), subgoal_id)
        previous = self._consecutive_state.get(
            key, {"last_step": -1, "count": 0}
        )
        last_step = previous["last_step"]
        count = previous["count"]
        if step_index < last_step:
            count = 0
            last_step = -1
        if step_index > last_step:
            count = count + 1 if satisfied else 0
        elif not satisfied:
            count = 0
        self._consecutive_state[key] = {
            "last_step": int(step_index),
            "count": int(count),
        }
        return count

    def __call__(
        self,
        *,
        env: Any,
        atomic_task_call: AtomicTaskCall | Mapping[str, Any],
        observation: Mapping[str, Any],
        step_index: int,
        info: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return success only when the structured predicate has evidence."""

        del observation
        call = (
            atomic_task_call
            if isinstance(atomic_task_call, AtomicTaskCall)
            else AtomicTaskCall.from_mapping(atomic_task_call)
        )
        conditions = call.termination_condition
        if isinstance(conditions, list):
            condition: Mapping[str, Any] = {"all_of": conditions}
        else:
            condition = conditions
        try:
            required = self._required_consecutive_successes(call)
            satisfied, evidence = self._evaluate_condition(
                env=env, condition=condition, info=info
            )
        except (TypeError, ValueError) as exc:
            return _result(
                "failed",
                goal_satisfied=False,
                evidence=[{"step_index": step_index, "error": str(exc)}],
                failure_code="INVALID_TERMINATION_CONDITION",
                retryable=False,
            )
        evidence.insert(0, {"step_index": int(step_index)})
        consecutive_count = self._record_consecutive_result(
            env=env,
            subgoal_id=call.subgoal_id,
            step_index=int(step_index),
            satisfied=satisfied is True,
        )
        if required > 1:
            evidence.append(
                {
                    "source": "consecutive_stability",
                    "instantaneous_satisfied": satisfied is True,
                    "consecutive_successes": consecutive_count,
                    "required_consecutive_successes": required,
                    "stable": consecutive_count >= required,
                }
            )
        if satisfied is True and consecutive_count >= required:
            return _result(
                "success", goal_satisfied=True, evidence=evidence, retryable=False
            )
        return _result(
            "uncertain", goal_satisfied=False, evidence=evidence, retryable=True
        )
