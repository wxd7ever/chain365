"""Runtime entity grounding and on-demand navigation for RoboCasa skills.

The planner deliberately does not own navigation.  Before each manipulation skill,
the scheduler asks :class:`RoboCasaGrounder` where that skill must be executed and
whether the mobile base is already at a valid work pose.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from .atomic_task_prompt_builder import clean_entity_name
    from .atomic_task_schemas import AtomicTaskCall
    from .atomic_task_verifier import (
        NAVIGATION_ORIENTATION_COSINE_THRESHOLD,
        NAVIGATION_POSITION_THRESHOLD_M,
    )
    from .skill_contract_registry import apply_skill_contract
except ImportError:  # Direct execution from the robocasa script directory.
    from atomic_task_prompt_builder import clean_entity_name
    from atomic_task_schemas import AtomicTaskCall
    from atomic_task_verifier import (
        NAVIGATION_ORIENTATION_COSINE_THRESHOLD,
        NAVIGATION_POSITION_THRESHOLD_M,
    )
    from skill_contract_registry import apply_skill_contract


_RELATION_PREDICATES = {"inside", "inserted", "on", "object_fixture_relation"}


@dataclass(frozen=True)
class GroundingResult:
    """Serializable decision made immediately before an operation skill."""

    grounded: bool
    status: str
    target_entity_alias: str | None = None
    target_entity_kind: str | None = None
    target_fixture_alias: str | None = None
    reference_object_alias: str | None = None
    held_object_alias: str | None = None
    target_mode: str | None = None
    reason: str = ""
    evidence: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    chain = _env_chain(env)
    for owner in reversed(chain):
        if hasattr(owner, "fixtures") and hasattr(owner, "objects") and hasattr(owner, "sim"):
            return owner
    raise ValueError("could not find a RoboCasa task environment for grounding")


def _normalise_identifier(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _candidate_names(alias: str, entity: Any) -> set[str]:
    return {
        value
        for value in (
            _normalise_identifier(alias),
            _normalise_identifier(getattr(entity, "name", "")),
            _normalise_identifier(getattr(entity, "nat_lang", "")),
        )
        if value
    }


def _goal_conditions(call: AtomicTaskCall) -> list[dict[str, Any]]:
    contract = call.metadata.get("skill_contract", {})
    goal = contract.get("goal", {}) if isinstance(contract, Mapping) else {}
    conditions = goal.get("conditions") if isinstance(goal, Mapping) else None
    if not isinstance(conditions, (Mapping, list)):
        conditions = call.termination_condition
    if isinstance(conditions, Mapping):
        return [dict(conditions)]
    return [dict(item) for item in conditions]


class RoboCasaGrounder:
    """Ground operation targets against the live privileged simulator state.

    The interface is intentionally pluggable: a visual/VLM grounder can later
    implement the same ``ground`` and ``build_navigation_call`` methods without
    changing the orchestrator.
    """

    def __init__(
        self,
        *,
        scene_context: Mapping[str, Any] | None = None,
        entity_alias_resolver: Callable[..., str | None] | None = None,
        position_threshold_m: float = NAVIGATION_POSITION_THRESHOLD_M,
        orientation_cosine_threshold: float = NAVIGATION_ORIENTATION_COSINE_THRESHOLD,
    ):
        if position_threshold_m <= 0:
            raise ValueError("position_threshold_m must be positive")
        if not -1.0 <= orientation_cosine_threshold <= 1.0:
            raise ValueError("orientation_cosine_threshold must be in [-1, 1]")
        self.scene_context = dict(scene_context or {})
        self.entity_alias_resolver = entity_alias_resolver
        self.position_threshold_m = float(position_threshold_m)
        self.orientation_cosine_threshold = float(orientation_cosine_threshold)

    def _entity_candidates(self, raw_env: Any) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        for kind in ("fixtures", "objects"):
            for alias, entity in getattr(raw_env, kind, {}).items():
                result.append(
                    {
                        "alias": str(alias),
                        "kind": kind[:-1],
                        "name": str(getattr(entity, "name", alias)),
                        "natural_name": str(getattr(entity, "nat_lang", "")),
                    }
                )
        return result

    def _resolve_entity(
        self,
        raw_env: Any,
        identifier: Any,
        *,
        expected_kind: str | None = None,
        call: AtomicTaskCall | None = None,
    ) -> tuple[str | None, str | None, list[dict[str, Any]]]:
        text = str(identifier or "").strip()
        evidence: list[dict[str, Any]] = []
        if not text:
            return None, None, evidence
        collections = (
            (expected_kind + "s",)
            if expected_kind in {"fixture", "object"}
            else ("fixtures", "objects")
        )
        for kind in collections:
            collection = getattr(raw_env, kind, {})
            if text in collection:
                return text, kind[:-1], evidence

        wanted = _normalise_identifier(text)
        matches: list[tuple[str, str]] = []
        for kind in collections:
            for alias, entity in getattr(raw_env, kind, {}).items():
                if wanted in _candidate_names(str(alias), entity):
                    matches.append((str(alias), kind[:-1]))
        matches = list(dict.fromkeys(matches))
        if len(matches) == 1:
            alias, kind = matches[0]
            evidence.append(
                {
                    "source": "unique_entity_alias_fallback",
                    "identifier": text,
                    "resolved_alias": alias,
                    "kind": kind,
                }
            )
            return alias, kind, evidence

        if self.entity_alias_resolver is None or call is None:
            return None, None, evidence
        resolved = self.entity_alias_resolver(
            identifier=text,
            field="grounding_target",
            candidates=self._entity_candidates(raw_env),
            condition={"expected_kind": expected_kind},
            atomic_task_call=call.to_dict(),
        )
        if not isinstance(resolved, str) or not resolved.strip():
            return None, None, evidence
        resolved = resolved.strip()
        for kind in collections:
            if resolved in getattr(raw_env, kind, {}):
                evidence.append(
                    {
                        "source": "vlm_entity_alias_resolver",
                        "identifier": text,
                        "resolved_alias": resolved,
                        "kind": kind[:-1],
                    }
                )
                return resolved, kind[:-1], evidence
        return None, None, evidence

    @staticmethod
    def _is_held(raw_env: Any, object_alias: str | None) -> bool:
        if not object_alias or object_alias not in getattr(raw_env, "objects", {}):
            return False
        try:
            from robocasa.utils import object_utils

            return bool(object_utils.check_obj_grasped(raw_env, object_alias))
        except (AttributeError, KeyError, TypeError, ValueError):
            return False

    def _configured_fixture_for_object(
        self, raw_env: Any, object_alias: str
    ) -> str | None:
        for cfg in getattr(raw_env, "object_cfgs", []) or []:
            if not isinstance(cfg, Mapping) or str(cfg.get("name", "")) != object_alias:
                continue
            placement = cfg.get("placement")
            if not isinstance(placement, Mapping):
                continue
            fixture_ref = placement.get("fixture")
            for alias, fixture in getattr(raw_env, "fixtures", {}).items():
                if fixture_ref is fixture or str(fixture_ref) == str(alias):
                    return str(alias)
        return None

    def _locate_object_fixture(
        self,
        raw_env: Any,
        object_alias: str,
        *,
        _visited: set[str] | None = None,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        evidence: list[dict[str, Any]] = []
        visited = set() if _visited is None else _visited
        if object_alias in visited:
            return None, [
                {
                    "source": "object_receptacle_relation",
                    "object": object_alias,
                    "cycle_detected": True,
                }
            ]
        visited.add(object_alias)
        if object_alias not in getattr(raw_env, "objects", {}):
            return None, evidence
        try:
            from robocasa.utils import object_utils

            body_id = raw_env.obj_body_id[object_alias]
            object_pos = np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float)
            matches: list[tuple[int, float, str, str]] = []
            for alias, fixture in raw_env.fixtures.items():
                alias = str(alias)
                score = 0
                relation = ""
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
                            object_pos, fixture, only_2d=True
                        ):
                            score, relation = 1, "xy_bounds"
                    except (AttributeError, TypeError, ValueError):
                        pass
                if score:
                    fixture_pos = np.asarray(getattr(fixture, "pos", object_pos), dtype=float)
                    distance = float(np.linalg.norm(object_pos[:2] - fixture_pos[:2]))
                    matches.append((score, -distance, alias, relation))
            if matches:
                score, neg_distance, alias, relation = max(matches)
                evidence.append(
                    {
                        "source": "live_object_fixture_relation",
                        "object": object_alias,
                        "fixture": alias,
                        "relation": relation,
                        "score": score,
                        "xy_distance": -neg_distance,
                    }
                )
                return alias, evidence

            # If the target is inside a movable receptacle, ground the receptacle's
            # fixture recursively (for example an ice cube in an ice bowl).
            for receptacle_alias in raw_env.objects:
                receptacle_alias = str(receptacle_alias)
                if receptacle_alias == object_alias:
                    continue
                try:
                    nested = object_utils.check_obj_in_receptacle(
                        raw_env, object_alias, receptacle_alias
                    )
                except (AttributeError, KeyError, TypeError, ValueError):
                    nested = False
                if nested:
                    fixture_alias, nested_evidence = self._locate_object_fixture(
                        raw_env, receptacle_alias, _visited=visited
                    )
                    if fixture_alias:
                        evidence.append(
                            {
                                "source": "object_receptacle_relation",
                                "object": object_alias,
                                "receptacle": receptacle_alias,
                            }
                        )
                        evidence.extend(nested_evidence)
                        return fixture_alias, evidence
        except (AttributeError, ImportError, KeyError, TypeError, ValueError) as exc:
            evidence.append({"source": "live_object_fixture_relation", "error": str(exc)})

        configured = self._configured_fixture_for_object(raw_env, object_alias)
        if configured:
            evidence.append(
                {
                    "source": "object_placement_config_fallback",
                    "object": object_alias,
                    "fixture": configured,
                }
            )
            return configured, evidence
        return None, evidence

    def _navigation_pose(
        self,
        raw_env: Any,
        fixture_alias: str,
        reference_object_alias: str | None,
        position_threshold_m: float,
        orientation_cosine_threshold: float,
    ) -> tuple[bool | None, list[dict[str, Any]]]:
        try:
            from robocasa.utils import env_utils
            from robosuite.utils import transform_utils

            fixture = raw_env.fixtures[fixture_alias]
            reference = (
                reference_object_alias
                if reference_object_alias in getattr(raw_env, "object_placements", {})
                else None
            )
            try:
                target_pos, target_ori = env_utils.compute_robot_base_placement_pose(
                    raw_env, fixture, ref_object=reference
                )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                reference = None
                target_pos, target_ori = env_utils.compute_robot_base_placement_pose(
                    raw_env, fixture
                )
            robot_id = raw_env.sim.model.body_name2id("mobilebase0_base")
            base_pos = np.asarray(raw_env.sim.data.body_xpos[robot_id], dtype=float)
            base_ori = transform_utils.mat2euler(
                np.asarray(raw_env.sim.data.body_xmat[robot_id]).reshape((3, 3))
            )
            position_distance = float(
                np.linalg.norm(np.asarray(target_pos)[:2] - base_pos[:2])
            )
            orientation_cosine = float(np.cos(target_ori[2] - base_ori[2]))
            position_ok = position_distance <= position_threshold_m
            orientation_ok = orientation_cosine >= orientation_cosine_threshold
            grounded = bool(position_ok and orientation_ok)
            return grounded, [
                {
                    "source": "operation_navigation_pose",
                    "fixture": fixture_alias,
                    "reference_object": reference,
                    "target_position": np.asarray(target_pos).tolist(),
                    "base_position": base_pos.tolist(),
                    "position_distance": position_distance,
                    "position_threshold": position_threshold_m,
                    "orientation_cosine": orientation_cosine,
                    "orientation_cosine_threshold": orientation_cosine_threshold,
                    "position_ok": bool(position_ok),
                    "orientation_ok": bool(orientation_ok),
                    "value": grounded,
                }
            ]
        except (AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            return None, [
                {
                    "source": "operation_navigation_pose",
                    "fixture": fixture_alias,
                    "error": str(exc),
                }
            ]

    def ground(
        self, *, env: Any, atomic_task_call: AtomicTaskCall | Mapping[str, Any]
    ) -> GroundingResult:
        call = (
            atomic_task_call
            if isinstance(atomic_task_call, AtomicTaskCall)
            else AtomicTaskCall.from_mapping(atomic_task_call)
        )
        if call.atomic_task == "NavigateKitchen":
            return GroundingResult(
                grounded=True,
                status="not_required",
                reason="navigation calls are already scheduler actions",
            )
        raw_env = _task_env(env)
        evidence: list[dict[str, Any]] = []
        conditions = _goal_conditions(call)

        relation = next(
            (
                item
                for item in conditions
                if str(item.get("predicate", "")).lower() in _RELATION_PREDICATES
            ),
            None,
        )
        primary = conditions[0] if conditions else {}

        object_identifiers: list[Any] = []
        for key in ("object_id", "source_id", "source_object_id"):
            if call.arguments.get(key):
                object_identifiers.append(call.arguments[key])
        if relation is not None:
            object_identifiers.append(relation.get("subject"))
        elif str(primary.get("predicate", "")).lower() not in {
            "receptacle_count",
            "open",
            "closed",
            "powered",
            "pressed",
            "turned",
            "fixture_state",
        }:
            object_identifiers.append(primary.get("subject"))

        operation_object: str | None = None
        for identifier in object_identifiers:
            alias, kind, resolution_evidence = self._resolve_entity(
                raw_env, identifier, expected_kind="object", call=call
            )
            evidence.extend(resolution_evidence)
            if alias and kind == "object":
                operation_object = alias
                break

        held_object = operation_object if self._is_held(raw_env, operation_object) else None
        destination_identifier = None
        if relation is not None:
            destination_identifier = relation.get("object", relation.get("destination"))
        for key in (
            "destination_id",
            "receptacle_id",
            "container_id",
            "target_fixture_id",
        ):
            if destination_identifier is None and call.arguments.get(key):
                destination_identifier = call.arguments[key]

        target_alias: str | None = None
        target_kind: str | None = None
        target_fixture: str | None = None
        reference_object: str | None = None
        target_mode: str | None = None

        if held_object and destination_identifier:
            target_alias, target_kind, resolved = self._resolve_entity(
                raw_env, destination_identifier, call=call
            )
            evidence.extend(resolved)
            target_mode = "held_object_destination"
            if target_kind == "fixture":
                target_fixture = target_alias
            elif target_kind == "object" and target_alias:
                reference_object = target_alias
                target_fixture, located = self._locate_object_fixture(raw_env, target_alias)
                evidence.extend(located)

        if target_fixture is None and operation_object and not held_object:
            target_alias = operation_object
            target_kind = "object"
            reference_object = operation_object
            target_mode = "operation_object_current_location"
            target_fixture, located = self._locate_object_fixture(raw_env, operation_object)
            evidence.extend(located)

        # Receptacle-count policies specify their pickup area through source_id.
        if target_fixture is None and call.arguments.get("source_id"):
            source_alias, source_kind, resolved = self._resolve_entity(
                raw_env, call.arguments["source_id"], call=call
            )
            evidence.extend(resolved)
            if source_kind == "fixture":
                target_alias, target_kind = source_alias, source_kind
                target_fixture = source_alias
            elif source_kind == "object" and source_alias:
                target_alias, target_kind = source_alias, source_kind
                reference_object = source_alias
                target_fixture, located = self._locate_object_fixture(raw_env, source_alias)
                evidence.extend(located)
            target_mode = "declared_source"

        if target_fixture is None:
            fixture_identifiers = [primary.get("subject")]
            fixture_identifiers.extend(
                call.arguments.get(key)
                for key in ("fixture_id", "target_fixture_id")
                if call.arguments.get(key)
            )
            for identifier in fixture_identifiers:
                alias, kind, resolved = self._resolve_entity(
                    raw_env, identifier, expected_kind="fixture", call=call
                )
                evidence.extend(resolved)
                if alias and kind == "fixture":
                    target_alias, target_kind = alias, kind
                    target_fixture = alias
                    target_mode = "controlled_fixture"
                    break

        # Some operation policies control a movable component (for example a
        # blender lid) rather than a fixture. Ground that component's workstation.
        if target_fixture is None and primary.get("subject"):
            alias, kind, resolved = self._resolve_entity(
                raw_env, primary["subject"], call=call
            )
            evidence.extend(resolved)
            if alias and kind == "object":
                target_alias, target_kind = alias, kind
                reference_object = alias
                target_fixture, located = self._locate_object_fixture(raw_env, alias)
                evidence.extend(located)
                target_mode = "controlled_object_current_location"

        if target_fixture is None:
            return GroundingResult(
                grounded=False,
                status="unresolved",
                target_entity_alias=target_alias,
                target_entity_kind=target_kind,
                reference_object_alias=reference_object,
                held_object_alias=held_object,
                target_mode=target_mode,
                reason="could not resolve the operation target to a fixture",
                evidence=tuple(evidence),
            )

        work_pose = call.metadata.get("work_pose", {})
        if not isinstance(work_pose, Mapping):
            work_pose = {}
        try:
            position_threshold = float(
                work_pose.get("position_threshold_m", self.position_threshold_m)
            )
            orientation_threshold = float(
                work_pose.get("orientation_cosine_threshold", self.orientation_cosine_threshold)
            )
            if position_threshold <= 0 or not -1.0 <= orientation_threshold <= 1.0:
                raise ValueError("invalid work-pose thresholds")
        except (TypeError, ValueError) as exc:
            evidence.append({"source": "work_pose", "error": str(exc)})
            position_threshold = self.position_threshold_m
            orientation_threshold = self.orientation_cosine_threshold
        pose_value, pose_evidence = self._navigation_pose(
            raw_env, target_fixture, reference_object,
            position_threshold, orientation_threshold,
        )
        evidence.extend(pose_evidence)
        if pose_value is None:
            return GroundingResult(
                grounded=False,
                status="unresolved",
                target_entity_alias=target_alias,
                target_entity_kind=target_kind,
                target_fixture_alias=target_fixture,
                reference_object_alias=reference_object,
                held_object_alias=held_object,
                target_mode=target_mode,
                reason="target was grounded but its operation pose could not be evaluated",
                evidence=tuple(evidence),
            )
        return GroundingResult(
            grounded=pose_value,
            status="grounded" if pose_value else "navigation_required",
            target_entity_alias=target_alias,
            target_entity_kind=target_kind,
            target_fixture_alias=target_fixture,
            reference_object_alias=reference_object,
            held_object_alias=held_object,
            target_mode=target_mode,
            reason=(
                "robot is at the grounded operation pose"
                if pose_value
                else "target is grounded but robot is outside its operation pose"
            ),
            evidence=tuple(evidence),
        )

    def build_navigation_call(
        self,
        *,
        operation_call: AtomicTaskCall | Mapping[str, Any],
        grounding_result: GroundingResult | Mapping[str, Any],
    ) -> AtomicTaskCall:
        operation = (
            operation_call
            if isinstance(operation_call, AtomicTaskCall)
            else AtomicTaskCall.from_mapping(operation_call)
        )
        result = (
            grounding_result.to_dict()
            if isinstance(grounding_result, GroundingResult)
            else dict(grounding_result)
        )
        fixture_alias = str(result.get("target_fixture_alias") or "").strip()
        if not fixture_alias:
            raise ValueError("grounding result has no target_fixture_alias")
        fixture_name = clean_entity_name(fixture_alias)
        for entity in self.scene_context.get("fixtures", []):
            if isinstance(entity, Mapping) and str(entity.get("alias")) == fixture_alias:
                fixture_name = str(entity.get("natural_name") or fixture_name)
                break
        arguments: dict[str, Any] = {
            "fixture_id": fixture_alias,
            "fixture_name": fixture_name,
        }
        work_pose = operation.metadata.get("work_pose", {})
        if not isinstance(work_pose, Mapping):
            work_pose = {}
        work_pose_mode = str(work_pose.get("mode", ""))
        position_threshold = work_pose.get("position_threshold_m")
        orientation_threshold = work_pose.get("orientation_cosine_threshold")
        reference_object = result.get("reference_object_alias")
        if reference_object:
            arguments["reference_object_id"] = str(reference_object)
        held_object = result.get("held_object_alias")
        conditions: list[dict[str, Any]] = [
            {
                "predicate": "navigation_pose",
                "subject": fixture_alias,
                "desired_value": True,
            }
        ]
        if position_threshold is not None:
            conditions[0]["position_threshold_m"] = float(position_threshold)
        if orientation_threshold is not None:
            conditions[0]["orientation_cosine_threshold"] = float(
                orientation_threshold
            )
        if reference_object:
            conditions[0]["reference_object"] = str(reference_object)
        if held_object:
            arguments["held_object_id"] = str(held_object)
            conditions.append(
                {
                    "predicate": "holding",
                    "subject": str(held_object),
                    "desired_value": True,
                }
            )
        prompt = f"Navigate to the {fixture_name}."
        if work_pose_mode == "standard_pick" and reference_object:
            prompt = (
                f"Navigate to the {fixture_name} and align the mobile base at the "
                f"standard picking pose for the "
                f"{clean_entity_name(str(reference_object))}. Face the object and stop "
                "at the requested work pose."
            )
        if held_object:
            prompt = (
                f"While keeping hold of {clean_entity_name(str(held_object))}, "
                f"navigate to the {fixture_name}."
            )
        navigation_metadata = {
            "inserted_by": "scheduler_grounding",
            "for_subgoal_id": operation.subgoal_id,
            "grounding_target_mode": result.get("target_mode"),
            "work_pose_mode": work_pose_mode or None,
        }
        if position_threshold is not None:
            navigation_metadata["position_threshold_m"] = float(position_threshold)
        if orientation_threshold is not None:
            navigation_metadata["orientation_cosine_threshold"] = float(
                orientation_threshold
            )
        call = AtomicTaskCall.from_mapping(
            {
                "subgoal_id": f"navigate_before_{operation.subgoal_id}",
                "atomic_task": "NavigateKitchen",
                "policy_prompt": prompt,
                "arguments": arguments,
                "termination_condition": conditions[0] if len(conditions) == 1 else conditions,
                "metadata": navigation_metadata,
            }
        )
        contracted, _ = apply_skill_contract(call, scene_context=self.scene_context)
        return contracted


def grounding_result_to_dict(value: GroundingResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, GroundingResult):
        return value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("grounder.ground(...) must return GroundingResult or a mapping")
    return dict(value)
