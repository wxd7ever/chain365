"""Resolve atomic-skill arguments to live object or fixture navigation targets."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .navigation_types import (
    NavigationTarget,
    TargetType,
    coerce_atomic_call,
    raw_task_env,
)


_RELATIONS = {"inside", "inserted", "on", "object_fixture_relation"}


def _normalise(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _names(alias: str, entity: Any) -> set[str]:
    return {
        item
        for item in (
            _normalise(alias),
            _normalise(getattr(entity, "name", "")),
            _normalise(getattr(entity, "nat_lang", "")),
        )
        if item
    }


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
    contract = call.metadata.get("skill_contract", {})
    goal = contract.get("goal", {}) if isinstance(contract, Mapping) else {}
    conditions = goal.get("conditions") if isinstance(goal, Mapping) else None
    return _flatten_conditions(conditions or call.termination_condition)


class TargetResolver:
    """Use privileged runtime state without caching movable-object positions."""

    def _resolve_alias(
        self, raw_env: Any, identifier: Any, expected_kind: str | None = None
    ) -> tuple[str | None, str | None]:
        text = str(identifier or "").strip()
        if not text:
            return None, None
        collections = (
            (expected_kind + "s",)
            if expected_kind in {"object", "fixture"}
            else ("objects", "fixtures")
        )
        for collection in collections:
            if text in getattr(raw_env, collection, {}):
                return text, collection[:-1]
        wanted = _normalise(text)
        matches: list[tuple[str, str]] = []
        for collection in collections:
            for alias, entity in getattr(raw_env, collection, {}).items():
                if wanted in _names(str(alias), entity):
                    matches.append((str(alias), collection[:-1]))
        matches = list(dict.fromkeys(matches))
        return matches[0] if len(matches) == 1 else (None, None)

    @staticmethod
    def _object_pose(raw_env: Any, alias: str) -> tuple[np.ndarray, np.ndarray]:
        obj = raw_env.objects[alias]
        body_ids = getattr(raw_env, "obj_body_id", {})
        body_id = body_ids.get(alias, body_ids.get(getattr(obj, "name", alias)))
        if body_id is None:
            raise ValueError(f"object {alias!r} has no MuJoCo body id")
        position = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float)
        quat_wxyz = np.asarray(raw_env.sim.data.body_xquat[body_id], dtype=float)
        if position.shape != (3,) or quat_wxyz.shape != (4,):
            raise ValueError(f"object {alias!r} has invalid simulator pose")
        quaternion_xyzw = quat_wxyz[[1, 2, 3, 0]]
        return position, quaternion_xyzw

    @staticmethod
    def _fixture_pose(fixture: Any) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(getattr(fixture, "pos", None), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("fixture position is unavailable")
        yaw = float(getattr(fixture, "rot", 0.0))
        quaternion = np.asarray(
            [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)], dtype=float
        )
        return position, quaternion

    @staticmethod
    def _is_enclosed_fixture(fixture: Any) -> bool:
        try:
            from robocasa.models.fixtures import FixtureType, fixture_is_type

            types = (
                FixtureType.MICROWAVE,
                FixtureType.OVEN,
                FixtureType.FRIDGE,
                FixtureType.DISHWASHER,
                FixtureType.TOASTER_OVEN,
                FixtureType.CABINET,
                FixtureType.CABINET_WITH_DOOR,
                FixtureType.CABINET_SINGLE_DOOR,
                FixtureType.CABINET_DOUBLE_DOOR,
                FixtureType.DRAWER,
                FixtureType.TOP_DRAWER,
            )
            return any(fixture_is_type(fixture, value) for value in types)
        except (AttributeError, ImportError, TypeError, ValueError):
            name = type(fixture).__name__.lower()
            return any(
                token in name
                for token in (
                    "cabinet",
                    "drawer",
                    "microwave",
                    "oven",
                    "fridge",
                    "dishwasher",
                )
            )

    def locate_support_fixture(
        self, raw_env: Any, object_alias: str
    ) -> tuple[str | None, tuple[dict[str, Any], ...]]:
        """Select the strongest live containment/contact relation."""

        try:
            from robocasa.utils import object_utils

            position, _ = self._object_pose(raw_env, object_alias)
            matches: list[tuple[int, float, str, str]] = []
            for alias, fixture in raw_env.fixtures.items():
                relation = ""
                score = 0
                try:
                    if object_utils.obj_inside_of(
                        raw_env, object_alias, fixture, partial_check=True
                    ):
                        score, relation = 3, "inside"
                except (AssertionError, AttributeError, KeyError, TypeError, ValueError):
                    pass
                if score < 2:
                    try:
                        if object_utils.check_obj_fixture_contact(
                            raw_env, object_alias, fixture
                        ):
                            score, relation = 2, "contact"
                    except (AttributeError, KeyError, TypeError, ValueError):
                        pass
                if score < 1:
                    try:
                        if object_utils.point_in_fixture(
                            position, fixture, only_2d=True
                        ):
                            score, relation = 1, "xy_bounds"
                    except (AttributeError, TypeError, ValueError):
                        pass
                if score:
                    fixture_pos = np.asarray(getattr(fixture, "pos", position), dtype=float)
                    distance = float(np.linalg.norm(position[:2] - fixture_pos[:2]))
                    matches.append((score, -distance, str(alias), relation))
            if matches:
                score, neg_distance, alias, relation = max(matches)
                return alias, (
                    {
                        "source": "live_object_fixture_relation",
                        "fixture": alias,
                        "relation": relation,
                        "score": score,
                        "xy_distance": -neg_distance,
                    },
                )
        except (AttributeError, ImportError, KeyError, TypeError, ValueError) as exc:
            return None, ({"source": "support_fixture_lookup", "error": str(exc)},)
        return None, ()

    def _destination_identifier(self, call: Any) -> Any:
        arguments = call.arguments
        for key in (
            "destination_id",
            "destination_object_id",
            "receptacle_id",
            "container_id",
            "target_fixture_id",
        ):
            if arguments.get(key):
                return arguments[key]
        for condition in _goal_conditions(call):
            if str(condition.get("predicate", "")).lower() in _RELATIONS:
                return condition.get("object", condition.get("destination"))
        return None

    def _object_identifier(self, call: Any) -> Any:
        for key in ("object_id", "source_object_id", "source_id", "held_object_id"):
            if call.arguments.get(key):
                return call.arguments[key]
        for condition in _goal_conditions(call):
            predicate = str(condition.get("predicate", "")).lower()
            if predicate in (_RELATIONS | {"holding", "released"}):
                return condition.get("subject")
        return None

    def _fixture_identifier(self, call: Any) -> Any:
        for key in ("fixture_id", "target_fixture_id", "appliance_id"):
            if call.arguments.get(key):
                return call.arguments[key]
        for condition in _goal_conditions(call):
            predicate = str(condition.get("predicate", "")).lower()
            if predicate in {
                "open",
                "closed",
                "powered",
                "pressed",
                "turned",
                "fixture_state",
            }:
                return condition.get("subject")
        return None

    def resolve(self, env: Any, skill: Any, target: Any | None = None) -> NavigationTarget:
        call = coerce_atomic_call(skill)
        raw_env = raw_task_env(env)
        source = ""
        if isinstance(target, NavigationTarget):
            target = target.target_alias
            source = "explicit_navigation_target"
        elif isinstance(target, Mapping):
            target = target.get("target_alias", target.get("target_name"))
            source = "explicit_navigation_target"
        identifier = target
        expected_kind: str | None = None

        if identifier is None and call.atomic_task == "PlaceObject":
            identifier = self._destination_identifier(call)
            source = "skill_destination"
        if identifier is None:
            fixture_identifier = self._fixture_identifier(call)
            if fixture_identifier is not None:
                identifier = fixture_identifier
                expected_kind = "fixture"
                source = "controlled_fixture"
        if identifier is None:
            identifier = self._object_identifier(call)
            expected_kind = "object"
            source = "operation_object"
        if identifier is None:
            raise LookupError("TARGET_NOT_FOUND: skill does not identify a target")

        alias, kind = self._resolve_alias(raw_env, identifier, expected_kind)
        if alias is None or kind is None:
            raise LookupError(f"TARGET_NOT_FOUND: could not resolve {identifier!r}")
        entity = getattr(raw_env, kind + "s")[alias]
        target_name = str(
            getattr(entity, "nat_lang", None) or getattr(entity, "name", alias)
        )
        if kind == "fixture":
            position, quaternion = self._fixture_pose(entity)
            return NavigationTarget(
                target_type=TargetType.FIXTURE,
                target_name=target_name,
                target_alias=alias,
                world_position=tuple(float(v) for v in position),
                world_orientation_xyzw=tuple(float(v) for v in quaternion),
                support_fixture=alias,
                fixture_instance=entity,
                source=source,
            )

        position, quaternion = self._object_pose(raw_env, alias)
        support_alias, diagnostics = self.locate_support_fixture(raw_env, alias)
        support = raw_env.fixtures.get(support_alias) if support_alias else None
        target_type = (
            TargetType.OBJECT_IN_FIXTURE
            if support is not None and self._is_enclosed_fixture(support)
            else TargetType.EXPOSED_OBJECT
        )
        return NavigationTarget(
            target_type=target_type,
            target_name=target_name,
            target_alias=alias,
            world_position=tuple(float(v) for v in position),
            world_orientation_xyzw=tuple(float(v) for v in quaternion),
            support_fixture=support_alias,
            fixture_instance=support,
            source=source,
            diagnostics=diagnostics,
        )
