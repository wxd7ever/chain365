"""Runtime verification of structured atomic-task termination conditions."""

from __future__ import annotations

import inspect
import json
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

NAVIGATION_POSITION_THRESHOLD_M = 0.62
NAVIGATION_ORIENTATION_COSINE_THRESHOLD = 0.90


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
    raise ValueError("could not find a RoboCasa task environment for verification")


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


def _coerce_boolean_state(value: Any) -> bool | None:
    """Accept actual booleans and explicit numeric 0/1, never NaN or angles."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if np.isfinite(numeric) and numeric in (0.0, 1.0):
            return bool(numeric)
    return None


def _require_boolean_state(value: Any, source: str) -> bool:
    state = _coerce_boolean_state(value)
    if state is None:
        raise TypeError(f"{source} did not return a boolean state")
    return state


def _candidate_names(key: str, value: Any) -> set[str]:
    names = {_normalise_identifier(key), _normalise_identifier(value)}
    if value is not None:
        for attr in ("name", "nat_lang"):
            names.add(_normalise_identifier(getattr(value, attr, "")))
    return {name for name in names if name}


def _resolve_exact_alias(
    chain: list[Any], identifier: str
) -> tuple[Any | None, str | None]:
    """Resolve only a complete collection key or an explicit environment role."""

    for owner in chain:
        for collection_name in ("fixtures", "objects"):
            collection = getattr(owner, collection_name, None)
            if isinstance(collection, Mapping) and identifier in collection:
                return collection[identifier], identifier
    for owner in chain:
        if not hasattr(owner, identifier):
            continue
        entity = getattr(owner, identifier)
        for collection_name in ("fixtures", "objects"):
            collection = getattr(owner, collection_name, None)
            if not isinstance(collection, Mapping):
                continue
            for key, value in collection.items():
                if value is entity:
                    return value, str(key)
        return entity, identifier
    return None, None


def _resolve_entity(chain: list[Any], identifier: str) -> tuple[Any | None, str | None]:
    """Prefer exact aliases; accept a natural-name fallback only when unique."""

    exact_entity, exact_alias = _resolve_exact_alias(chain, identifier)
    if exact_entity is not None:
        return exact_entity, exact_alias
    wanted = _normalise_identifier(identifier)
    matches: dict[tuple[int, str], tuple[Any, str]] = {}
    for owner in chain:
        for collection_name in ("fixtures", "objects"):
            collection = getattr(owner, collection_name, None)
            if not isinstance(collection, Mapping):
                continue
            for key, value in collection.items():
                if wanted in _candidate_names(str(key), value):
                    alias = str(key)
                    matches[(id(value), alias)] = (value, alias)
    if len(matches) == 1:
        return next(iter(matches.values()))
    return None, None


def _entity_alias_candidates(chain: list[Any]) -> list[dict[str, str]]:
    """Return exact aliases and metadata for an optional fallback resolver."""

    candidates: dict[tuple[str, str], dict[str, str]] = {}
    for owner in chain:
        for kind in ("fixtures", "objects"):
            collection = getattr(owner, kind, None)
            if not isinstance(collection, Mapping):
                continue
            for key, value in collection.items():
                alias = str(key)
                candidates.setdefault(
                    (kind, alias),
                    {
                        "alias": alias,
                        "kind": kind[:-1],
                        "name": str(getattr(value, "name", alias)),
                        "natural_name": str(getattr(value, "nat_lang", "")),
                    },
                )
    return list(candidates.values())


def _entity_kind(raw_env: Any, alias: str | None) -> str | None:
    if alias is None:
        return None
    if alias in getattr(raw_env, "objects", {}):
        return "object"
    if alias in getattr(raw_env, "fixtures", {}):
        return "fixture"
    return None


def _expected_entity_kinds(
    predicate: str, field: str, condition: Mapping[str, Any]
) -> frozenset[str]:
    """Return the entity kinds accepted by a predicate field."""

    relation_predicates = {"inside", "inserted", "on", "object_fixture_relation"}
    fixture_predicates = {
        "closed",
        "eef_outside_fixture",
        "fixture_state",
        "navigation_pose",
        "open",
        "powered",
        "pressed",
        "turned",
    }
    if field == "reference_object":
        return frozenset({"object"})
    if field in {"object", "destination"}:
        if predicate == "object_fixture_relation":
            return frozenset({"fixture"})
        if predicate in relation_predicates:
            return frozenset({"object", "fixture"})
        return frozenset({"object", "fixture"})
    if field != "subject":
        return frozenset({"object", "fixture"})
    if predicate in fixture_predicates:
        return frozenset({"fixture"})
    if predicate in relation_predicates or predicate in {
        "holding",
        "receptacle_count",
        "released",
    }:
        if predicate == "released" and condition.get("object_prefix"):
            return frozenset()
        return frozenset({"object"})
    if predicate == "gripper_far":
        return frozenset({"object", "fixture"})
    return frozenset({"object", "fixture"})


def _mapping_state(
    chain: list[Any], info: Mapping[str, Any], predicate: str, subject: str, object_id: str | None
) -> tuple[bool | None, list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    # Never consume a bare predicate key such as {"inside": True}: it is not
    # tied to the requested entities and can falsely satisfy an unrelated goal.
    keys = [
        (predicate, subject, object_id),
        f"{predicate}:{subject}:{object_id}" if object_id else f"{predicate}:{subject}",
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
            if key in source:
                value = _coerce_boolean_state(source[key])
                if value is not None:
                    evidence.append(
                        {"source": source_name, "key": str(key), "value": value}
                    )
                    return value, evidence
        subject_state = source.get(subject)
        if isinstance(subject_state, Mapping) and predicate in subject_state:
            value = _coerce_boolean_state(subject_state[predicate])
            if value is not None:
                evidence.append(
                    {
                        "source": source_name,
                        "key": f"{subject}.{predicate}",
                        "value": value,
                    }
                )
                return value, evidence
    return None, evidence


def _custom_state(
    chain: list[Any], predicate: str, subject: str, object_id: str | None
) -> tuple[bool | None, list[dict[str, Any]]]:
    for index, owner in enumerate(chain):
        method = getattr(owner, "get_atomic_task_state", None)
        if not callable(method):
            continue
        try:
            value = method(predicate=predicate, subject=subject, object=object_id)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            return None, [
                {
                    "source": f"env[{index}].get_atomic_task_state",
                    "error": str(exc),
                }
            ]
        if isinstance(value, Mapping):
            state = value.get("value")
            evidence = dict(value)
        else:
            state = value
            evidence = {"value": value}
        state = _coerce_boolean_state(state)
        if state is not None:
            evidence.update({"source": f"env[{index}].get_atomic_task_state"})
            return state, [evidence]
    return None, []


def _fixture_state(
    chain: list[Any], predicate: str, subject: str, desired_state: Any
) -> tuple[bool | None, list[dict[str, Any]]]:
    entity, alias = _resolve_entity(chain, subject)
    if entity is None:
        return None, [{"source": "fixture_lookup", "subject": subject, "resolved": False}]
    raw_env = _task_env(chain)
    if alias not in getattr(raw_env, "fixtures", {}):
        return None, [
            {
                "source": "fixture_lookup",
                "subject": subject,
                "entity": alias,
                "expected_kind": "fixture",
                "actual_kind": _entity_kind(raw_env, alias),
            }
        ]
    if predicate == "fixture_state" and isinstance(desired_state, str):
        state_name = desired_state.strip().lower()
        if state_name in {"open", "closed"}:
            method = getattr(entity, f"is_{state_name}", None)
            if callable(method):
                value = _coerce_boolean_state(
                    _call_state_method(method, raw_env)
                )
                if value is None:
                    return None, [
                        {
                            "source": "fixture_method",
                            "entity": alias,
                            "desired_state": state_name,
                            "error": "fixture method did not return a boolean state",
                        }
                    ]
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
            value = _coerce_boolean_state(_call_state_method(method, raw_env))
            if value is None:
                return None, [
                    {
                        "source": "fixture_method",
                        "entity": alias,
                        "error": "fixture method did not return a boolean state",
                    }
                ]
            return value, [{"source": "fixture_method", "entity": alias, "value": value}]
    state_sources: list[tuple[str, Mapping[str, Any]]] = []
    # Sink intentionally exposes faucet, spout, and temperature state through
    # get_handle_state(env), while its inherited get_state() has no mapping.
    # Treat this as a first-class fixture API instead of silently declaring an
    # already-manipulated faucet unverifiable.
    get_handle_state = getattr(entity, "get_handle_state", None)
    if callable(get_handle_state):
        handle_state = _call_state_method(get_handle_state, raw_env)
        if isinstance(handle_state, Mapping):
            state_sources.append(("fixture_handle_state", handle_state))

    get_state = getattr(entity, "get_state", None)
    if callable(get_state):
        state = _call_state_method(get_state, raw_env)
        if isinstance(state, Mapping):
            state_sources.append(("fixture_state", state))
    if not state_sources:
        return None, [{"source": "fixture_lookup", "entity": alias, "state_api": False}]
    key_candidates = {
        "powered": ("turned_on", "water_on", "powered", "power", "on"),
        "pressed": ("pressed", "button_pressed"),
        "turned": ("turned", "angle", "position"),
        "inserted": ("inserted",),
        "fixture_state": (str(desired_state),),
    }.get(predicate, (predicate,))
    for source_name, state in state_sources:
        for key in key_candidates:
            if key in state:
                raw_value = state[key]
                value = _coerce_boolean_state(raw_value)
                if value is not None:
                    return value, [
                        {
                            "source": source_name,
                            "entity": alias,
                            "key": key,
                            "value": raw_value,
                        }
                    ]
                if predicate == "fixture_state" and isinstance(raw_value, str):
                    value = raw_value.strip().lower() == str(desired_state).strip().lower()
                    return value, [
                        {
                            "source": source_name,
                            "entity": alias,
                            "key": key,
                            "value": raw_value,
                        }
                    ]
    return None, [
        {
            "source": "fixture_state",
            "entity": alias,
            "available_keys": {
                source_name: sorted(map(str, state))
                for source_name, state in state_sources
            },
        }
    ]


def _object_relation(
    chain: list[Any],
    predicate: str,
    subject: str,
    object_id: str | None,
    receptacle_threshold: float | None = None,
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
    subject_kind = _entity_kind(raw_env, subject_alias)
    object_kind = _entity_kind(raw_env, object_alias)
    if subject_kind != "object":
        return None, [
            {
                "source": "relation_lookup",
                "predicate": predicate,
                "subject": subject_alias,
                "expected_subject_kind": "object",
                "actual_subject_kind": subject_kind,
            }
        ]
    try:
        relation_api = ""
        if predicate == "holding":
            relation_api = "check_obj_grasped"
            value = _require_boolean_state(
                object_utils.check_obj_grasped(raw_env, subject_alias), relation_api
            )
        elif predicate in {"inside", "inserted", "on"} and object_kind == "object":
            kwargs = (
                {}
                if receptacle_threshold is None
                else {"th": receptacle_threshold}
            )
            relation_api = "check_obj_in_receptacle"
            value = _require_boolean_state(
                object_utils.check_obj_in_receptacle(
                    raw_env, subject_alias, object_alias, **kwargs
                ),
                relation_api,
            )
        elif predicate in {"inside", "inserted"} and object_kind == "fixture":
            relation_api = "obj_inside_of"
            value = _require_boolean_state(
                object_utils.obj_inside_of(
                    raw_env, subject_alias, object_alias, partial_check=False
                ),
                relation_api,
            )
        elif predicate in {"on", "object_fixture_relation"} and object_kind == "fixture":
            relation_api = "check_obj_fixture_contact"
            value = _require_boolean_state(
                object_utils.check_obj_fixture_contact(
                    raw_env, subject_alias, object_alias
                ),
                relation_api,
            )
        else:
            return None, [
                {
                    "source": "relation_lookup",
                    "subject": subject,
                    "object": object_id,
                    "subject_resolved": subject_entity is not None,
                    "object_resolved": object_entity is not None,
                    "subject_kind": subject_kind,
                    "object_kind": object_kind,
                }
            ]
    except (AssertionError, AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        return None, [{"source": "object_utils", "error": str(exc)}]
    return value, [
        {
            "source": "simulator_relation",
            "predicate": predicate,
            "subject": subject_alias,
            "object": object_alias,
            "subject_kind": subject_kind,
            "object_kind": object_kind,
            "relation_api": relation_api,
            "receptacle_threshold": receptacle_threshold,
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
    if receptacle_alias not in getattr(raw_env, "objects", {}):
        return None, [
            {
                "source": "receptacle_count",
                "receptacle": receptacle_alias,
                "expected_kind": "object",
                "actual_kind": _entity_kind(raw_env, receptacle_alias),
            }
        ]
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
        directly_inside = [
            alias
            for alias in candidates
            if _require_boolean_state(
                object_utils.check_obj_in_receptacle(
                    raw_env, alias, receptacle_alias, th=0.5
                ),
                "check_obj_in_receptacle",
            )
        ]
        # PlaceEqualIceCubes counts a cube stacked on a cube that is directly
        # in the cup. Mirror that official one-hop contact rule here.
        check_contact = getattr(raw_env, "check_contact", None)
        touching_inside = (
            [
                alias
                for alias in candidates
                if alias not in directly_inside
                and any(
                    _require_boolean_state(
                        check_contact(
                            raw_env.objects[alias], raw_env.objects[inside_alias]
                        ),
                        "check_contact",
                    )
                    for inside_alias in directly_inside
                )
            ]
            if callable(check_contact)
            else []
        )
        inside = sorted(set(directly_inside + touching_inside))
    except (
        AssertionError,
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        return None, [{"source": "receptacle_count", "error": str(exc)}]
    count = len(inside)
    value = count == desired_count
    return value, [
        {
            "source": "receptacle_count",
            "receptacle": receptacle_alias,
            "object_prefix": object_prefix,
            "candidates": candidates,
            "directly_inside": directly_inside,
            "touching_inside": touching_inside,
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
            if _require_boolean_state(
                object_utils.check_obj_grasped(raw_env, alias),
                "check_obj_grasped",
            )
        ]
    except (
        AssertionError,
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
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
            value = _require_boolean_state(
                object_utils.gripper_obj_far(raw_env, alias, th=threshold),
                "gripper_obj_far",
            )
            entity_kind = "object"
        elif alias in getattr(raw_env, "fixtures", {}):
            fixture_name = str(getattr(entity, "name", alias))
            value = _require_boolean_state(
                object_utils.gripper_fxtr_far(
                    raw_env, fixture_name, th=threshold
                ),
                "gripper_fxtr_far",
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
    except (
        AssertionError,
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
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
    if not all(
        value.shape == (3,) and np.isfinite(value).all()
        for value in (point, p0, px, py, pz)
    ):
        raise ValueError("EEF and fixture bounding-box coordinates must be finite 3D points")
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
    except (AssertionError, AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
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
    chain: list[Any], subject: str, reference_object: str | None = None
) -> tuple[bool | None, list[dict[str, Any]]]:
    target_fixture, target_alias = _resolve_entity(chain, subject)
    if target_fixture is None:
        return None, [{"source": "navigation_target", "subject": subject, "resolved": False}]
    raw_env = _task_env(chain)
    if target_alias not in getattr(raw_env, "fixtures", {}):
        return None, [
            {
                "source": "navigation_target",
                "subject": subject,
                "entity": target_alias,
                "expected_kind": "fixture",
                "actual_kind": _entity_kind(raw_env, target_alias),
            }
        ]
    try:
        applied_reference = None
        if (
            getattr(raw_env, "target_fixture", None) is target_fixture
            and reference_object is None
            and hasattr(raw_env, "target_pos")
            and hasattr(raw_env, "target_ori")
        ):
            target_pos = np.asarray(raw_env.target_pos, dtype=float)
            target_ori = np.asarray(raw_env.target_ori, dtype=float)
        else:
            from robocasa.utils import env_utils as env_utils
            if (
                reference_object is not None
                and reference_object
                not in getattr(raw_env, "object_placements", {})
            ):
                raise KeyError(
                    f"reference object {reference_object!r} has no live placement"
                )
            target_pos, target_ori = env_utils.compute_robot_base_placement_pose(
                raw_env, target_fixture, ref_object=reference_object
            )
            applied_reference = reference_object
        target_pos = np.asarray(target_pos, dtype=float)
        target_ori = np.asarray(target_ori, dtype=float)
        robot_id = raw_env.sim.model.body_name2id("mobilebase0_base")
        base_pos = np.asarray(raw_env.sim.data.body_xpos[robot_id], dtype=float)
        from robosuite.utils import transform_utils as transform_utils
        base_ori = transform_utils.mat2euler(
            np.asarray(raw_env.sim.data.body_xmat[robot_id]).reshape((3, 3))
        )
        if (
            target_pos.size < 2
            or target_ori.size < 3
            or base_pos.size < 2
            or base_ori.size < 3
            or not np.isfinite(target_pos).all()
            or not np.isfinite(target_ori).all()
            or not np.isfinite(base_pos).all()
            or not np.isfinite(base_ori).all()
        ):
            raise ValueError("navigation poses must contain finite position and orientation values")
        position_distance = float(np.linalg.norm(target_pos[:2] - base_pos[:2]))
        orientation_cosine = float(np.cos(target_ori[2] - base_ori[2]))
        position_ok = position_distance <= NAVIGATION_POSITION_THRESHOLD_M
        orientation_ok = (
            orientation_cosine >= NAVIGATION_ORIENTATION_COSINE_THRESHOLD
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
    ) as exc:
        return None, [{"source": "navigation_pose", "entity": target_alias, "error": str(exc)}]
    value = bool(position_ok and orientation_ok)
    return value, [
        {
            "source": "navigation_pose",
            "entity": target_alias,
            "requested_reference_object": reference_object,
            "reference_object": applied_reference,
            "target_position": np.asarray(target_pos).tolist(),
            "base_position": base_pos.tolist(),
            "position_distance": position_distance,
            "position_threshold": NAVIGATION_POSITION_THRESHOLD_M,
            "orientation_cosine": orientation_cosine,
            "orientation_cosine_threshold": NAVIGATION_ORIENTATION_COSINE_THRESHOLD,
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

    def __init__(
        self,
        *,
        default_required_consecutive_successes: int = 1,
        entity_alias_resolver: Callable[..., str | None] | None = None,
    ):
        if (
            isinstance(default_required_consecutive_successes, bool)
            or not isinstance(default_required_consecutive_successes, int)
            or default_required_consecutive_successes <= 0
        ):
            raise ValueError(
                "default_required_consecutive_successes must be a positive integer"
            )
        if entity_alias_resolver is not None and not callable(entity_alias_resolver):
            raise TypeError("entity_alias_resolver must be callable when provided")
        self.default_required_consecutive_successes = (
            default_required_consecutive_successes
        )
        self.entity_alias_resolver = entity_alias_resolver
        self._alias_resolution_cache: dict[tuple[Any, ...], str] = {}
        self._consecutive_state: dict[tuple[int, str, str], dict[str, int]] = {}

    def _ground_condition_aliases(
        self,
        *,
        env: Any,
        call: AtomicTaskCall,
        condition: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Ground every condition identifier before simulator-state evaluation."""

        chain = _env_chain(env)
        raw_env = _task_env(chain)
        candidates = _entity_alias_candidates(chain)
        candidate_signature = tuple(
            sorted(
                (
                    candidate["kind"],
                    candidate["alias"],
                    candidate["name"],
                    candidate["natural_name"],
                )
                for candidate in candidates
            )
        )
        evidence: list[dict[str, Any]] = []

        def ground(item: Mapping[str, Any]) -> dict[str, Any]:
            value = dict(item)
            operators = [
                operator for operator in ("all_of", "any_of") if operator in value
            ]
            if operators:
                if len(operators) != 1 or set(value) != {operators[0]}:
                    raise ValueError(
                        "termination condition must contain exactly one boolean "
                        "operator and no other fields"
                    )
                operator = operators[0]
                children = value[operator]
                if not isinstance(children, list) or not children:
                    raise ValueError(
                        f"termination_condition.{operator} must be a non-empty list"
                    )
                if not all(isinstance(child, Mapping) for child in children):
                    raise ValueError(
                        f"termination_condition.{operator} must contain mappings"
                    )
                value[operator] = [ground(child) for child in children]
                return value

            predicate = (
                str(value.get("predicate", ""))
                .strip()
                .lower()
                .replace(" ", "_")
            )
            if (
                value.get("object") is not None
                and value.get("destination") is not None
                and str(value["object"]).strip() != str(value["destination"]).strip()
            ):
                raise ValueError(
                    "termination condition object and destination must match when "
                    "both are provided"
                )
            for field in ("subject", "object", "destination", "reference_object"):
                identifier = value.get(field)
                if identifier is None:
                    continue
                identifier = str(identifier).strip()
                if not identifier:
                    continue
                if (
                    field == "subject"
                    and predicate == "released"
                    and value.get("object_prefix")
                ):
                    continue

                expected_kinds = _expected_entity_kinds(predicate, field, value)

                _, alias = _resolve_exact_alias(chain, identifier)
                if alias is not None:
                    actual_kind = _entity_kind(raw_env, alias)
                    if expected_kinds and actual_kind not in expected_kinds:
                        raise ValueError(
                            f"{field}={identifier!r} resolves to {actual_kind}, but "
                            f"predicate {predicate!r} requires {sorted(expected_kinds)}"
                        )
                    value[field] = alias
                    continue

                _, alias = _resolve_entity(chain, identifier)
                if alias is not None:
                    actual_kind = _entity_kind(raw_env, alias)
                    if expected_kinds and actual_kind not in expected_kinds:
                        raise ValueError(
                            f"{field}={identifier!r} resolves to {actual_kind}, but "
                            f"predicate {predicate!r} requires {sorted(expected_kinds)}"
                        )
                    value[field] = alias
                    evidence.append(
                        {
                            "source": "unique_entity_alias_fallback",
                            "field": field,
                            "identifier": identifier,
                            "resolved_alias": alias,
                        }
                    )
                    continue

                if self.entity_alias_resolver is None:
                    raise ValueError(
                        f"{field}={identifier!r} is not an exact scene alias and "
                        "has no unique deterministic match"
                    )
                expected_key = tuple(sorted(expected_kinds))
                cache_key = (
                    id(raw_env),
                    field,
                    identifier,
                    expected_key,
                    candidate_signature,
                )
                resolved_alias = self._alias_resolution_cache.get(cache_key)
                if resolved_alias is None:
                    resolver_candidates = [
                        candidate
                        for candidate in candidates
                        if not expected_kinds
                        or candidate.get("kind") in expected_kinds
                    ]
                    try:
                        resolved_alias = self.entity_alias_resolver(
                            identifier=identifier,
                            field=field,
                            candidates=resolver_candidates,
                            condition=dict(value),
                            atomic_task_call=call.to_dict(),
                        )
                    except (OSError, RuntimeError, TimeoutError) as exc:
                        raise ValueError(
                            f"VLM entity alias resolver failed: {exc}"
                        ) from exc
                    if not isinstance(resolved_alias, str) or not resolved_alias.strip():
                        raise ValueError(
                            f"entity alias resolver did not resolve {field}={identifier!r}"
                        )
                    resolved_alias = resolved_alias.strip()

                _, exact_alias = _resolve_exact_alias(chain, resolved_alias)
                if exact_alias is None:
                    raise ValueError(
                        "entity alias resolver returned non-existent alias "
                        f"{resolved_alias!r} for {field}={identifier!r}"
                    )
                actual_kind = _entity_kind(raw_env, exact_alias)
                if expected_kinds and actual_kind not in expected_kinds:
                    raise ValueError(
                        "entity alias resolver returned wrong-kind alias "
                        f"{resolved_alias!r} ({actual_kind}) for predicate "
                        f"{predicate!r}; expected {sorted(expected_kinds)}"
                    )
                # Cache only a validated, exact, correctly typed scene alias.
                # A transient resolver hallucination must not poison later calls.
                self._alias_resolution_cache[cache_key] = exact_alias
                value[field] = exact_alias
                evidence.append(
                    {
                        "source": "vlm_entity_alias_resolver",
                        "field": field,
                        "identifier": identifier,
                        "resolved_alias": exact_alias,
                    }
                )
            return value

        return ground(condition), evidence


    def _evaluate_condition(
        self,
        *,
        env: Any,
        condition: Mapping[str, Any],
        info: Mapping[str, Any],
    ) -> tuple[bool | None, list[dict[str, Any]]]:
        operators = [
            operator for operator in ("all_of", "any_of") if operator in condition
        ]
        if operators:
            if len(operators) != 1 or set(condition) != {operators[0]}:
                raise ValueError(
                    "termination condition must contain exactly one boolean "
                    "operator and no other fields"
                )
            operator = operators[0]
            children = condition[operator]
            if not isinstance(children, list) or not children or not all(
                isinstance(item, Mapping) for item in children
            ):
                raise ValueError(
                    f"termination_condition.{operator} must be a non-empty list "
                    "of mappings"
                )
            evaluated = [
                self._evaluate_condition(env=env, condition=item, info=info)
                for item in children
            ]
            values = [value for value, _ in evaluated]
            evidence = [item for _, items in evaluated for item in items]
            # A malformed or unavailable branch invalidates the complete
            # contract.  Do not let all_of(False, None) become a retryable
            # policy failure, or any_of(True, None) become a false success.
            if None in values:
                return None, evidence
            if operator == "all_of":
                value = all(v is True for v in values)
            else:
                value = any(v is True for v in values)
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
        if isinstance(desired, float) and not np.isfinite(desired):
            raise ValueError("termination condition desired_value must be finite")
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
        if predicate == "fixture_state":
            if not isinstance(desired, (bool, str)):
                raise ValueError(
                    "fixture_state desired_value must be boolean or a state name"
                )
        elif not isinstance(desired, bool):
            raise ValueError(
                f"{predicate} desired_value must be boolean"
            )

        threshold = 0.25
        receptacle_threshold = None
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
        elif predicate in {"inside", "inserted", "on"} and "threshold" in condition:
            raw_threshold = condition["threshold"]
            if isinstance(raw_threshold, bool):
                raise ValueError("relation threshold must be a positive number")
            try:
                receptacle_threshold = float(raw_threshold)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "relation threshold must be a positive number"
                ) from exc
            if not np.isfinite(receptacle_threshold) or receptacle_threshold <= 0:
                raise ValueError("relation threshold must be a positive number")

        # Prefer current privileged simulator state. Custom hooks and exact
        # entity-scoped atomic_state entries are compatibility fallbacks only;
        # they must never override a native false result with stale data.
        value: bool | None = None
        evidence: list[dict[str, Any]] = []
        if predicate == "released":
            value, predicate_evidence = _released(chain, subject, object_prefix)
            evidence.extend(predicate_evidence)
        elif predicate == "gripper_far":
            value, predicate_evidence = _gripper_far(chain, subject, threshold)
            evidence.extend(predicate_evidence)
        elif predicate == "eef_outside_fixture":
            value, predicate_evidence = _eef_outside_fixture(
                chain, subject, margin, only_2d
            )
            evidence.extend(predicate_evidence)
        elif predicate in {"inside", "inserted", "on", "holding", "object_fixture_relation"}:
            value, relation_evidence = _object_relation(
                chain,
                predicate,
                subject,
                object_id,
                receptacle_threshold,
            )
            evidence.extend(relation_evidence)
        elif predicate == "navigation_pose":
            reference_object = str(condition.get("reference_object", "")).strip() or None
            value, navigation_evidence = _navigation_pose(
                chain, subject, reference_object
            )
            evidence.extend(navigation_evidence)
        else:
            value, fixture_evidence = _fixture_state(chain, predicate, subject, desired)
            evidence.extend(fixture_evidence)

        native_error = any("error" in item for item in evidence)
        if value is None and not native_error:
            value, custom_evidence = _custom_state(
                chain, predicate, subject, object_id
            )
            evidence.extend(custom_evidence)
        custom_error = any("error" in item for item in evidence)
        if value is None and not custom_error:
            value, mapping_evidence = _mapping_state(
                chain, info, predicate, subject, object_id
            )
            evidence.extend(mapping_evidence)
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
        condition_key: str,
        step_index: int,
        satisfied: bool,
    ) -> int:
        key = (id(env), subgoal_id, condition_key)
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
        try:
            if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
                raise ValueError("step_index must be a non-negative integer")
            if not isinstance(info, Mapping):
                raise TypeError("info must be a mapping")
            call = (
                atomic_task_call
                if isinstance(atomic_task_call, AtomicTaskCall)
                else AtomicTaskCall.from_mapping(atomic_task_call)
            )
            conditions = call.termination_condition
            if isinstance(conditions, list):
                if not conditions:
                    raise ValueError("termination_condition must not be empty")
                condition: Mapping[str, Any] = {"all_of": conditions}
            else:
                if not conditions:
                    raise ValueError("termination_condition must not be empty")
                condition = conditions
            grounded_condition, grounding_evidence = self._ground_condition_aliases(
                env=env,
                call=call,
                condition=condition,
            )
            required = self._required_consecutive_successes(call)
            satisfied, evidence = self._evaluate_condition(
                env=env, condition=grounded_condition, info=info
            )
            evidence = grounding_evidence + evidence
        except (TypeError, ValueError) as exc:
            return _result(
                "failed",
                goal_satisfied=False,
                evidence=[{"step_index": step_index, "error": str(exc)}],
                failure_code="INVALID_TERMINATION_CONDITION",
                retryable=False,
            )
        except (
            AssertionError,
            AttributeError,
            ImportError,
            IndexError,
            KeyError,
            OSError,
            OverflowError,
            RuntimeError,
        ) as exc:
            return _result(
                "failed",
                goal_satisfied=False,
                evidence=[{"step_index": step_index, "error": str(exc)}],
                failure_code="VERIFIER_EVALUATION_ERROR",
                retryable=False,
            )
        evidence.insert(0, {"step_index": int(step_index)})
        if satisfied is None:
            evaluation_error = any(item.get("error") for item in evidence)
            evidence.append(
                {
                    "source": "verifier",
                    "error": "termination condition could not be evaluated",
                }
            )
            failure_code = (
                "VERIFIER_EVALUATION_ERROR"
                if evaluation_error
                else "INVALID_TERMINATION_CONDITION"
            )
            return _result(
                "failed",
                goal_satisfied=False,
                evidence=evidence,
                failure_code=failure_code,
                retryable=False,
            )
        condition_key = json.dumps(
            grounded_condition,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        consecutive_count = self._record_consecutive_result(
            env=env,
            subgoal_id=call.subgoal_id,
            condition_key=condition_key,
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
