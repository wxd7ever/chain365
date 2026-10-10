"""Source-aware motion checks used after a successful Pick grasp."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _normalise_identifier(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def object_center_position(raw_env: Any, object_alias: str) -> np.ndarray | None:
    """Return one object's world-frame center without changing simulator state."""

    try:
        body_id = raw_env.obj_body_id[object_alias]
        position = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None
    if position.shape != (3,) or not np.isfinite(position).all():
        return None
    return position.copy()


def check_two_finger_grasp(
    raw_env: Any, object_alias: str
) -> tuple[bool | None, dict[str, Any]]:
    """Check a grasp using robosuite's left-and-right finger-pad contacts.

    Unlike the legacy ``check_obj_grasped`` helper, this check does not impose
    a fixed finger-joint aperture. Large objects can therefore be accepted as
    long as both finger pads actually contact the requested object.
    """

    evidence: dict[str, Any] = {
        "source": "two_finger_grasp",
        "subject": object_alias,
        "relation_api": "_check_grasp",
        "required_contact_groups": ["left_fingerpad", "right_fingerpad"],
    }
    try:
        obj = raw_env.objects[object_alias]
        robot = raw_env.robots[0]
        gripper = robot.gripper["right"]
        checker = getattr(raw_env, "_check_grasp")
        if not callable(checker):
            raise AttributeError("RoboCasa environment has no callable _check_grasp")
        value = bool(checker(gripper, obj))
    except (
        AssertionError,
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        evidence.update({"value": None, "error": str(exc)})
        return None, evidence
    evidence["value"] = value
    return value, evidence


def resolve_pick_source_fixture(
    raw_env: Any, atomic_task_call: Any
) -> tuple[str | None, Any | None]:
    """Resolve an explicitly declared Pick source to a unique fixture."""

    arguments = getattr(atomic_task_call, "arguments", {})
    if not isinstance(arguments, Mapping):
        return None, None
    fixtures = getattr(raw_env, "fixtures", {})
    if not isinstance(fixtures, Mapping):
        return None, None
    for key in (
        "source_id",
        "source_name",
        "source_fixture_id",
        "source_fixture_name",
    ):
        identifier = str(arguments.get(key) or "").strip()
        if not identifier:
            continue
        if identifier in fixtures:
            return identifier, fixtures[identifier]
        wanted = _normalise_identifier(identifier)
        matches = [
            (str(alias), fixture)
            for alias, fixture in fixtures.items()
            if wanted
            in {
                _normalise_identifier(alias),
                _normalise_identifier(getattr(fixture, "name", "")),
                _normalise_identifier(getattr(fixture, "nat_lang", "")),
            }
        ]
        if len(matches) == 1:
            return matches[0]
    return None, None


def resolve_pick_source_object(
    raw_env: Any, atomic_task_call: Any
) -> tuple[str | None, Any | None]:
    """Resolve an explicitly declared Pick source to a movable scene object."""

    arguments = getattr(atomic_task_call, "arguments", {})
    if not isinstance(arguments, Mapping):
        return None, None
    objects = getattr(raw_env, "objects", {})
    if not isinstance(objects, Mapping):
        return None, None
    for key in (
        "source_object_id",
        "source_object_name",
        "source_receptacle_id",
        "source_receptacle_name",
        "source_container_id",
        "source_container_name",
        "source_id",
        "source_name",
    ):
        identifier = str(arguments.get(key) or "").strip()
        if not identifier:
            continue
        if identifier in objects:
            return identifier, objects[identifier]
        wanted = _normalise_identifier(identifier)
        matches = [
            (str(alias), obj)
            for alias, obj in objects.items()
            if wanted
            in {
                _normalise_identifier(alias),
                _normalise_identifier(getattr(obj, "name", "")),
                _normalise_identifier(getattr(obj, "nat_lang", "")),
            }
        ]
        if len(matches) == 1:
            return matches[0]
    return None, None


def is_cabinet_or_drawer_fixture(alias: str | None, fixture: Any | None) -> bool:
    """Recognize cabinet and drawer sources without depending on one layout class."""

    if fixture is None:
        return False
    labels = " ".join(
        str(value or "").lower()
        for value in (
            alias,
            type(fixture).__name__,
            getattr(fixture, "name", ""),
            getattr(fixture, "nat_lang", ""),
        )
    )
    return "cabinet" in labels or "drawer" in labels


def _find_enclosing_cabinet_or_drawer(
    raw_env: Any, object_alias: str
) -> tuple[str | None, Any | None]:
    """Recover an enclosure source when a natural-name source is ambiguous."""

    try:
        from robocasa.utils import object_utils
    except (ImportError, ModuleNotFoundError):
        return None, None
    for alias, fixture in getattr(raw_env, "fixtures", {}).items():
        alias = str(alias)
        if not is_cabinet_or_drawer_fixture(alias, fixture):
            continue
        try:
            inside = object_utils.obj_inside_of(
                raw_env, object_alias, fixture, partial_check=True
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
        if bool(inside):
            return alias, fixture
    return None, None


def _fixture_contact(raw_env: Any, object_alias: str, fixture: Any) -> bool | None:
    try:
        from robocasa.utils import object_utils

        return bool(
            object_utils.check_obj_fixture_contact(raw_env, object_alias, fixture)
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


def _object_contact(raw_env: Any, object_alias: str, other_object: Any) -> bool | None:
    try:
        obj = raw_env.objects[object_alias]
        return bool(raw_env.check_contact(obj, other_object))
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


def _inside_source_object(
    raw_env: Any, object_alias: str, source_alias: str | None
) -> bool | None:
    if not source_alias or source_alias == object_alias:
        return None
    try:
        from robocasa.utils import object_utils

        return bool(
            object_utils.check_obj_in_receptacle(
                raw_env,
                object_alias,
                source_alias,
            )
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


def _object_support_relation(
    raw_env: Any,
    object_alias: str,
    candidate_alias: str,
) -> str | None:
    """Return why one contacting object directly supports the Pick target.

    Objects contained by the Pick target are deliberately rejected first. This
    makes food inside a bowl irrelevant when the bowl itself is being picked.
    Conversely, a bowl containing the Pick target is retained as its direct
    movable support.
    """

    if candidate_alias == object_alias:
        return None
    contact = _object_contact(
        raw_env,
        object_alias,
        getattr(raw_env, "objects", {}).get(candidate_alias),
    )
    if contact is not True:
        return None

    target_position = object_center_position(raw_env, object_alias)
    candidate_position = object_center_position(raw_env, candidate_alias)
    if target_position is None or candidate_position is None:
        return None

    candidate_inside_target = _inside_source_object(
        raw_env, candidate_alias, object_alias
    )
    if (
        candidate_inside_target is True
        and float(candidate_position[2]) >= float(target_position[2])
    ):
        return None
    target_inside_candidate = _inside_source_object(
        raw_env, object_alias, candidate_alias
    )
    if (
        target_inside_candidate is True
        and float(target_position[2]) >= float(candidate_position[2])
    ):
        return "target_in_receptacle"
    # Covers plates, trays, and object stacks that are not classified as
    # receptacles. The tolerance handles nearly level body origins.
    if float(candidate_position[2]) <= float(target_position[2]) + 0.01:
        return "contact_below_target"
    return None


def _capture_support_contacts(
    raw_env: Any, object_alias: str
) -> tuple[list[dict[str, str]], tuple[tuple[str, str, Any], ...]]:
    """Capture fixtures and movable objects directly supporting a Pick target.

    A receptacle containing the target is a direct support. Objects contained by
    the target are not: when picking a loaded bowl, its contents must not become
    supports that the bowl is required to leave.
    """

    descriptions: list[dict[str, str]] = []
    entities: list[tuple[str, str, Any]] = []
    for alias, fixture in getattr(raw_env, "fixtures", {}).items():
        if _fixture_contact(raw_env, object_alias, fixture) is True:
            alias = str(alias)
            descriptions.append(
                {"kind": "fixture", "alias": alias, "relation": "contact"}
            )
            entities.append(("fixture", alias, fixture))
    for alias, obj in getattr(raw_env, "objects", {}).items():
        alias = str(alias)
        relation = _object_support_relation(raw_env, object_alias, alias)
        if relation is None:
            continue
        descriptions.append(
            {"kind": "object", "alias": alias, "relation": relation}
        )
        entities.append(("object", alias, obj))
    return descriptions, tuple(entities)


def _current_support_contacts(
    raw_env: Any, object_alias: str, support_entities: Any
) -> tuple[list[dict[str, Any]], bool, bool]:
    states: list[dict[str, Any]] = []
    items = support_entities if isinstance(support_entities, (list, tuple)) else ()
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        kind, alias, entity = item
        if kind == "fixture":
            contact = _fixture_contact(raw_env, object_alias, entity)
        elif kind == "object":
            contact = _object_contact(raw_env, object_alias, entity)
        else:
            contact = None
        states.append({"kind": str(kind), "alias": str(alias), "contact": contact})
    support_available = bool(states) and all(
        item["contact"] is not None for item in states
    )
    left_all_supports = bool(
        support_available and not any(item["contact"] is True for item in states)
    )
    return states, left_all_supports, support_available


def _inside_source_fixture(
    raw_env: Any, object_alias: str, source_fixture: Any | None
) -> bool | None:
    if source_fixture is None:
        return None
    try:
        from robocasa.utils import object_utils

        return bool(
            object_utils.obj_inside_of(
                raw_env,
                object_alias,
                source_fixture,
                partial_check=True,
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


def _is_holding(raw_env: Any, object_alias: str) -> bool | None:
    value, _ = check_two_finger_grasp(raw_env, object_alias)
    return value


def capture_pick_motion_baseline(
    *, raw_env: Any, object_alias: str, atomic_task_call: Any
) -> dict[str, Any] | None:
    """Capture the pose and source relation at the start of one Pick attempt."""

    position = object_center_position(raw_env, object_alias)
    if position is None:
        return None
    source_alias, source_fixture = resolve_pick_source_fixture(
        raw_env, atomic_task_call
    )
    source_object_alias, source_object = resolve_pick_source_object(
        raw_env, atomic_task_call
    )
    if source_object_alias == object_alias:
        source_object_alias, source_object = None, None
    source_is_enclosed = is_cabinet_or_drawer_fixture(
        source_alias, source_fixture
    )
    if not source_is_enclosed:
        enclosed_alias, enclosed_fixture = _find_enclosing_cabinet_or_drawer(
            raw_env, object_alias
        )
        if enclosed_fixture is not None:
            source_alias, source_fixture = enclosed_alias, enclosed_fixture
            source_is_enclosed = True
    support_contacts: list[dict[str, str]] = []
    support_entities: tuple[tuple[str, str, Any], ...] = ()
    if not source_is_enclosed:
        support_contacts, support_entities = _capture_support_contacts(
            raw_env, object_alias
        )
        if source_object is None:
            object_supports = [
                item for item in support_entities if item[0] == "object"
            ]
            if len(object_supports) == 1:
                _, source_object_alias, source_object = object_supports[0]
    return {
        "subject": object_alias,
        "initial_center_position_m": position.tolist(),
        # Retain the old field so existing diagnostics remain easy to compare.
        "initial_center_z_m": float(position[2]),
        "source_fixture_alias": source_alias,
        "source_fixture_type": (
            type(source_fixture).__name__ if source_fixture is not None else None
        ),
        "source_is_cabinet_or_drawer": source_is_enclosed,
        "initially_inside_source_region": (
            _inside_source_fixture(raw_env, object_alias, source_fixture)
            if source_is_enclosed
            else None
        ),
        "holding_at_attempt_start": _is_holding(raw_env, object_alias),
        "initial_support_contacts": support_contacts,
        "source_object_alias": source_object_alias,
        "initially_inside_source_object": _inside_source_object(
            raw_env, object_alias, source_object_alias
        ),
        "_source_fixture": source_fixture,
        "_source_object": source_object,
        "_support_entities": support_entities,
    }


def evaluate_pick_motion(
    *,
    raw_env: Any,
    object_alias: str,
    baseline: Mapping[str, Any],
    minimum_delta_m: float,
) -> tuple[bool | None, dict[str, Any]]:
    """Accept support departure normally and extraction from enclosures."""

    current = object_center_position(raw_env, object_alias)
    try:
        initial = np.asarray(baseline["initial_center_position_m"], dtype=float)
    except (KeyError, TypeError, ValueError):
        initial = np.asarray([], dtype=float)
    if current is None or initial.shape != (3,) or not np.isfinite(initial).all():
        return None, {
            "source": "pick_motion_verification",
            "subject": object_alias,
            "value": None,
            "error": "Pick object pose or attempt baseline is unavailable",
        }

    delta = current - initial
    vertical_delta = float(delta[2])
    horizontal_delta = float(np.linalg.norm(delta[:2]))
    displacement = float(np.linalg.norm(delta))
    threshold = float(minimum_delta_m)
    source_is_enclosed = bool(baseline.get("source_is_cabinet_or_drawer"))
    source_fixture = baseline.get("_source_fixture")
    currently_inside = (
        _inside_source_fixture(raw_env, object_alias, source_fixture)
        if source_is_enclosed
        else None
    )
    initially_inside = baseline.get("initially_inside_source_region")
    left_source_region = bool(
        source_is_enclosed
        and initially_inside is True
        and currently_inside is False
    )
    already_outside_source = bool(
        source_is_enclosed
        and baseline.get("holding_at_attempt_start") is True
        and initially_inside is False
        and currently_inside is False
    )
    vertical_lift_satisfied = vertical_delta >= threshold
    horizontal_extraction_satisfied = (
        source_is_enclosed and horizontal_delta >= threshold
    )
    source_object_alias = baseline.get("source_object_alias")
    initially_inside_source_object = baseline.get(
        "initially_inside_source_object"
    )
    currently_inside_source_object = _inside_source_object(
        raw_env, object_alias, source_object_alias
    )
    left_source_object = bool(
        initially_inside_source_object is True
        and currently_inside_source_object is False
    )
    support_states: list[dict[str, Any]] = []
    left_support_surface: bool | None = None
    support_contacts_available: bool | None = None
    fallback_used = False
    readiness_basis: str
    if source_is_enclosed:
        ready = bool(
            vertical_lift_satisfied
            or horizontal_extraction_satisfied
            or left_source_region
            or already_outside_source
        )
        mode = "cabinet_or_drawer_extraction"
        readiness_basis = "cabinet_or_drawer_extraction"
    else:
        (
            support_states,
            left_support_surface,
            support_contacts_available,
        ) = _current_support_contacts(
            raw_env, object_alias, baseline.get("_support_entities")
        )
        initial_support_resolved = bool(support_states)
        if initial_support_resolved and support_contacts_available:
            ready = bool(left_support_surface)
            readiness_basis = "direct_support_departure"
        elif initial_support_resolved:
            ready = False
            readiness_basis = "direct_support_contact_unavailable"
        else:
            fallback_used = True
            ready = bool(vertical_lift_satisfied or left_source_object)
            if left_source_object:
                readiness_basis = "source_object_departure_fallback"
            elif vertical_lift_satisfied:
                readiness_basis = "vertical_lift_fallback"
            else:
                readiness_basis = "unresolved_support_fallback_pending"
        mode = "support_surface_departure"

    return ready, {
        # Keep the historical source name for log consumers, while making the
        # actual source-aware mode explicit.
        "source": "pick_lift_verification",
        "verification_mode": mode,
        "subject": object_alias,
        "initial_center_position_m": initial.tolist(),
        "current_center_position_m": current.tolist(),
        "initial_center_z_m": float(initial[2]),
        "current_center_z_m": float(current[2]),
        "lift_delta_m": vertical_delta,
        "horizontal_extraction_delta_m": horizontal_delta,
        "displacement_m": displacement,
        "minimum_lift_delta_m": threshold,
        "minimum_extraction_delta_m": threshold if source_is_enclosed else None,
        "source_fixture_alias": baseline.get("source_fixture_alias"),
        "source_fixture_type": baseline.get("source_fixture_type"),
        "initially_inside_source_region": initially_inside,
        "currently_inside_source_region": currently_inside,
        "object_left_source_region": left_source_region,
        "holding_at_attempt_start": baseline.get("holding_at_attempt_start"),
        "already_outside_source_at_attempt_start": already_outside_source,
        "vertical_lift_satisfied": bool(vertical_lift_satisfied),
        "horizontal_extraction_satisfied": bool(
            horizontal_extraction_satisfied
        ),
        "initial_support_contacts": list(
            baseline.get("initial_support_contacts") or []
        ),
        "current_support_contacts": (
            support_states if not source_is_enclosed else []
        ),
        "left_support_surface": (
            bool(left_support_surface) if not source_is_enclosed else None
        ),
        "support_contacts_available": (
            bool(support_contacts_available) if not source_is_enclosed else None
        ),
        "initial_support_resolved": (
            bool(support_states) if not source_is_enclosed else None
        ),
        "source_object_alias": source_object_alias,
        "initially_inside_source_object": initially_inside_source_object,
        "currently_inside_source_object": currently_inside_source_object,
        "object_left_source_object": left_source_object,
        "fallback_used": fallback_used,
        "readiness_basis": readiness_basis,
        "value": ready,
    }
