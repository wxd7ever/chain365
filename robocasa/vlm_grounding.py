"""Image-only multi-view grounding for scheduler atomic-task calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .atomic_task_schemas import AtomicTaskCall
    from .grounding import GroundingResult, RoboCasaGrounder
    from .multi_view_perception import MultiViewObservationProvider
    from .view_scheduler import FixedViewScheduler
    from .vlm_artifacts import save_vlm_artifacts
except ImportError:
    from atomic_task_schemas import AtomicTaskCall
    from grounding import GroundingResult, RoboCasaGrounder
    from multi_view_perception import MultiViewObservationProvider
    from view_scheduler import FixedViewScheduler
    from vlm_artifacts import save_vlm_artifacts


GROUNDING_SYSTEM_PROMPT = """You are a conservative multi-view RoboCasa grounding module.
Use only the labelled camera images to locate the requested operation target and decide
whether the mobile base is coarsely close enough to start the operation or local pose
refinement. For a placement, judge only the destination receptacle (for example, the
bowl), its supporting workstation, and whether navigation is required. Do not judge
whether the source object is held, released, grasped, or otherwise satisfies a task
precondition; those decisions belong to the skill critic and scheduler handoff.
Scene context contains alias candidates only; it is not simulator state. Select aliases
exactly from the supplied candidate sets. Never invent an alias. If images are occluded,
ambiguous, or conflicting, return null/false and lower confidence. Return exactly one JSON
object with keys target_visible, target_entity_alias, target_entity_kind, entity_role,
target_fixture_alias, operation_pose_ready, navigation_required, confidence, reason,
supporting_views.
target_entity_kind is an ontology enum, not the object's natural category: it MUST be
exactly "object" for an alias from candidate_objects, exactly "fixture" for an alias
from candidate_fixtures, or null when target_entity_alias is null. Never put values such
as broccoli, bowl, microwave, appliance, food, or receptacle in target_entity_kind.
entity_role MUST be one of object, fixture, object_receptacle, surface_receptacle,
fixture_destination, or null. A placement destination may be a movable object receptacle
such as a bowl; do not assume every destination is a fixture. target_fixture_alias is the
supporting workstation fixture, while target_entity_alias is the actual operation target.
operation_pose_ready and navigation_required must be logical opposites. Do not claim
metric precision and do not infer positions or relations from privileged state."""


def _entities(scene_context: Mapping[str, Any], key: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in scene_context.get(key, []) or []:
        if not isinstance(item, Mapping):
            continue
        alias = str(item.get("alias", "")).strip()
        if not alias:
            continue
        result.append(
            {
                "alias": alias,
                "name": str(item.get("name", alias)),
                "natural_name": str(item.get("natural_name", item.get("name", alias))),
            }
        )
    return result


def _conditions(call: AtomicTaskCall) -> Any:
    return call.termination_condition


def _operation_semantics(
    call: AtomicTaskCall,
    *,
    object_aliases: set[str],
    fixture_aliases: set[str],
) -> dict[str, Any]:
    raw_conditions = call.termination_condition
    conditions = raw_conditions if isinstance(raw_conditions, list) else [raw_conditions]
    relation = next(
        (
            item
            for item in conditions
            if str(item.get("predicate", "")).lower()
            in {"inside", "inserted", "on", "object_fixture_relation"}
        ),
        None,
    )
    source = None
    destination = None
    predicate = None
    if relation is not None:
        source = str(relation.get("subject") or "").strip() or None
        destination = str(
            relation.get("object", relation.get("destination")) or ""
        ).strip() or None
        predicate = str(relation.get("predicate", "")).lower()
    if source is None:
        for key in ("held_object_id", "object_id", "source_object_id"):
            candidate = str(call.arguments.get(key) or "").strip()
            if candidate in object_aliases:
                source = candidate
                break
    if destination is None:
        for key in ("destination_id", "receptacle_id", "container_id"):
            candidate = str(call.arguments.get(key) or "").strip()
            if candidate in object_aliases or candidate in fixture_aliases:
                destination = candidate
                break
    if destination in object_aliases:
        destination_kind = "object"
        destination_role = "object_receptacle"
    elif destination in fixture_aliases:
        destination_kind = "fixture"
        destination_role = (
            "surface_receptacle" if predicate == "on" else "fixture_destination"
        )
    else:
        destination_kind = None
        destination_role = None
    return {
        "source_object_alias": source if source in object_aliases else None,
        "destination_entity_alias": destination,
        "destination_entity_kind": destination_kind,
        "destination_role": destination_role,
        "relation_predicate": predicate,
    }


def _confidence(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def _bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


class VLMRoboCasaGrounder:
    """Ground operation targets exclusively from fresh multi-view images."""

    def __init__(
        self,
        *,
        client: Any,
        observation_provider: MultiViewObservationProvider,
        view_scheduler: FixedViewScheduler | None = None,
        scene_context: Mapping[str, Any] | None = None,
        grounding_confidence_threshold: float = 0.70,
        max_validation_retries: int = 1,
        log_dir: str | Path | None = None,
    ):
        if not callable(getattr(client, "query_json", None)):
            raise TypeError("client must provide query_json(...)")
        if not callable(getattr(observation_provider, "observe", None)):
            raise TypeError("observation_provider must provide observe(...)")
        if view_scheduler is not None and not callable(
            getattr(view_scheduler, "select_views", None)
        ):
            raise TypeError("view_scheduler must provide select_views(...)")
        self.client = client
        self.observation_provider = observation_provider
        self.view_scheduler = view_scheduler
        self.scene_context = dict(scene_context or {})
        self.grounding_confidence_threshold = _confidence(
            grounding_confidence_threshold, field="grounding_confidence_threshold"
        )
        if (
            isinstance(max_validation_retries, bool)
            or not isinstance(max_validation_retries, int)
            or max_validation_retries < 0
        ):
            raise ValueError("max_validation_retries must be a non-negative integer")
        self.max_validation_retries = max_validation_retries
        self.log_dir = Path(log_dir) if log_dir is not None else None
        # Reuse only the schema/prompt construction path. Its privileged ground()
        # method is never called by this visual implementation.
        self._navigation_builder = RoboCasaGrounder(scene_context=self.scene_context)
        self._request_count = 0

    def build_navigation_call(self, **kwargs: Any) -> AtomicTaskCall:
        return self._navigation_builder.build_navigation_call(**kwargs)

    def _validate(
        self,
        value: Mapping[str, Any],
        *,
        object_aliases: set[str],
        fixture_aliases: set[str],
        camera_names: Sequence[str],
        operation_semantics: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("grounding response must be a JSON object")
        # Discard legacy state/evidence fields such as
        # holding_reference_object. Grounding owns only target localisation and
        # navigation readiness; source-object state is inherited by the scheduler.
        result = {
            key: value.get(key)
            for key in (
                "target_visible",
                "target_entity_alias",
                "target_entity_kind",
                "entity_role",
                "target_fixture_alias",
                "operation_pose_ready",
                "navigation_required",
                "confidence",
                "reason",
                "supporting_views",
            )
        }
        result["target_visible"] = _bool(
            result.get("target_visible"), field="target_visible"
        )
        result["operation_pose_ready"] = _bool(
            result.get("operation_pose_ready"), field="operation_pose_ready"
        )
        result["navigation_required"] = _bool(
            result.get("navigation_required"), field="navigation_required"
        )
        if result["navigation_required"] == result["operation_pose_ready"]:
            raise ValueError(
                "navigation_required must equal not operation_pose_ready"
            )
        result["confidence"] = _confidence(
            result.get("confidence"), field="confidence"
        )
        reason = result.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        result["reason"] = reason.strip()

        alias = result.get("target_entity_alias")
        kind = result.get("target_entity_kind")
        normalizations: list[dict[str, Any]] = []
        expected_destination = operation_semantics.get("destination_entity_alias")
        expected_destination_kind = operation_semantics.get("destination_entity_kind")
        expected_destination_role = operation_semantics.get("destination_role")
        if expected_destination is not None and alias != expected_destination:
            normalizations.append(
                {
                    "field": "target_entity_alias",
                    "from": alias,
                    "value": expected_destination,
                    "reason": "derived_from_place_relation_destination",
                }
            )
            alias = expected_destination
            kind = expected_destination_kind
            result["target_entity_alias"] = alias
            result["target_entity_kind"] = kind
        if alias is None:
            if kind is not None:
                raise ValueError("target_entity_kind must be null when alias is null")
        else:
            if not isinstance(alias, str):
                raise ValueError("target_entity_alias must be a string or null")
            is_object = alias in object_aliases
            is_fixture = alias in fixture_aliases
            if not is_object and not is_fixture:
                raise ValueError(f"target_entity_alias {alias!r} is not a candidate")
            if is_object and is_fixture:
                if kind not in {"object", "fixture"}:
                    raise ValueError(
                        "target_entity_kind is required for an ambiguous candidate alias"
                    )
                expected_kind = kind
            else:
                expected_kind = "object" if is_object else "fixture"
            if kind != expected_kind:
                normalizations.append(
                    {
                        "field": "target_entity_kind",
                        "from": kind,
                        "value": expected_kind,
                        "reason": "derived_from_exact_candidate_alias_membership",
                    }
                )
            result["target_entity_kind"] = expected_kind
        allowed_roles = {
            "object",
            "fixture",
            "object_receptacle",
            "surface_receptacle",
            "fixture_destination",
            None,
        }
        role = result.get("entity_role")
        if role not in allowed_roles:
            raise ValueError("entity_role is not a supported ontology role")
        expected_role = (
            expected_destination_role
            if expected_destination is not None
            else "object"
            if result.get("target_entity_kind") == "object"
            else "fixture"
            if result.get("target_entity_kind") == "fixture"
            else None
        )
        if role != expected_role:
            normalizations.append(
                {
                    "field": "entity_role",
                    "from": role,
                    "value": expected_role,
                    "reason": "derived_from_operation_semantics",
                }
            )
        result["entity_role"] = expected_role

        fixture = result.get("target_fixture_alias")
        if fixture is not None and (
            not isinstance(fixture, str) or fixture not in fixture_aliases
        ):
            raise ValueError(f"target_fixture_alias {fixture!r} is not a candidate")
        views = result.get("supporting_views")
        if not isinstance(views, list) or not all(isinstance(item, str) for item in views):
            raise ValueError("supporting_views must be a list of camera names")
        unknown_views = set(views) - set(camera_names)
        if unknown_views:
            raise ValueError(f"supporting_views contains unknown cameras: {unknown_views}")
        result["supporting_views"] = list(dict.fromkeys(views))
        if normalizations:
            result["validation_normalizations"] = normalizations
        return result

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
        self._request_count += 1
        selected_views = None
        if self.view_scheduler is not None:
            selected_views = self.view_scheduler.select_views(
                atomic_task_call=call,
                purpose="grounding",
                available_views=self.observation_provider.camera_names,
            )
        bundle = self.observation_provider.observe(
            env=env,
            camera_names=selected_views,
            purpose=f"grounding:{call.subgoal_id}",
        )
        objects = _entities(self.scene_context, "objects")
        fixtures = _entities(self.scene_context, "fixtures")
        object_aliases = {item["alias"] for item in objects}
        fixture_aliases = {item["alias"] for item in fixtures}
        operation_semantics = _operation_semantics(
            call,
            object_aliases=object_aliases,
            fixture_aliases=fixture_aliases,
        )
        base_user_prompt = (
            "Decide the visual grounding for this scheduler operation. The alias "
            "candidates are labels, not state evidence. For a pick, locate the requested "
            "object and its visible workstation. For a placement, ignore the source "
            "object and gripper state completely: locate only the destination receptacle "
            "as target_entity_alias and its supporting workstation as "
            "target_fixture_alias. operation_pose_ready only means the mobile base is "
            "coarsely near enough to operate on that target; otherwise set "
            "navigation_required=true. Do not report holding or task-success fields.\n"
            + json.dumps(
                {
                    "atomic_task": call.atomic_task,
                    "policy_prompt": call.policy_prompt,
                    "arguments": call.arguments,
                    "termination_conditions": _conditions(call),
                    "operation_semantics": operation_semantics,
                    "candidate_objects": objects,
                    "candidate_fixtures": fixtures,
                },
                ensure_ascii=False,
            )
        )
        user_prompt = base_user_prompt
        parsed: dict[str, Any] | None = None
        validation_error: str | None = None
        validation_attempts: list[dict[str, Any]] = []
        for attempt_index in range(self.max_validation_retries + 1):
            try:
                raw = self.client.query_json(
                    system_prompt=GROUNDING_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    images=bundle.images,
                )
                parsed = self._validate(
                    raw,
                    object_aliases=object_aliases,
                    fixture_aliases=fixture_aliases,
                    camera_names=bundle.camera_names,
                    operation_semantics=operation_semantics,
                )
                break
            except (
                OSError,
                RuntimeError,
                TimeoutError,
                TypeError,
                ValueError,
            ) as exc:
                validation_error = str(exc)
                validation_attempts.append(
                    {
                        "attempt_index": attempt_index,
                        "error": validation_error,
                        "raw_response": getattr(
                            self.client, "last_response_text", None
                        ),
                        "raw_http_response": getattr(
                            self.client, "last_raw_http_response", None
                        ),
                        "parsed_response": getattr(
                            self.client, "last_parsed_response", None
                        ),
                    }
                )
                if attempt_index >= self.max_validation_retries:
                    break
                user_prompt = (
                    base_user_prompt
                    + "\nCORRECTION REQUIRED: The previous response failed strict "
                    "validation. Return the entire corrected JSON object. Copy aliases "
                    "exactly from the candidate lists and use target_entity_kind only "
                    "as object, fixture, or null. Validation error: "
                    + validation_error
                    + "\nPrevious response: "
                    + json.dumps(
                        getattr(self.client, "last_parsed_response", None),
                        ensure_ascii=False,
                    )
                )

        if parsed is None:
            runtime_decision = {
                "grounded": False,
                "status": "unresolved",
                "reason": f"visual grounding failed: {validation_error}",
                "validation_attempts": validation_attempts,
            }
            artifact_dir, image_paths = save_vlm_artifacts(
                root=self.log_dir,
                category="vlm_grounding",
                request_id=f"{self._request_count:06d}_{call.subgoal_id}",
                bundle=bundle,
                system_prompt=GROUNDING_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                client=self.client,
                parsed_json=getattr(self.client, "last_parsed_response", None),
                runtime_decision=runtime_decision,
            )
            return GroundingResult(
                grounded=False,
                status="unresolved",
                reason=runtime_decision["reason"],
                evidence=(
                    {
                        "source": "vlm_grounding_error",
                        "camera_names": list(bundle.camera_names),
                        "image_paths": image_paths,
                        "artifact_dir": artifact_dir,
                        "raw_response": getattr(self.client, "last_response_text", None),
                        "raw_http_response": getattr(
                            self.client, "last_raw_http_response", None
                        ),
                        "error": validation_error,
                        "validation_attempts": validation_attempts,
                    },
                ),
            )

        target_alias = parsed.get("target_entity_alias")
        target_kind = parsed.get("target_entity_kind")
        entity_role = parsed.get("entity_role")
        fixture_alias = parsed.get("target_fixture_alias")
        source_alias = operation_semantics.get("source_object_alias")
        destination_alias = operation_semantics.get("destination_entity_alias")
        destination_role = operation_semantics.get("destination_role")
        reference_alias = (
            source_alias
            if destination_alias is not None
            else target_alias
            if target_kind == "object"
            else None
        )
        confident = parsed["confidence"] >= self.grounding_confidence_threshold
        grounded = bool(
            parsed["target_visible"]
            and parsed["operation_pose_ready"]
            and confident
        )
        if grounded:
            status = "grounded"
        elif (
            parsed["target_visible"]
            and fixture_alias is not None
            and parsed["navigation_required"]
            and confident
        ):
            status = "navigation_required"
        else:
            status = "unresolved"
        # The orchestrator interprets a non-null fixture as permission to insert
        # navigation. Do not let a low-confidence/unresolved visual guess move the
        # robot; the untrusted selection remains available in parsed_json evidence.
        runtime_fixture_alias = fixture_alias if status != "unresolved" else None
        if destination_alias is not None:
            target_mode = "operation_destination"
        elif target_kind == "object":
            target_mode = "operation_object_current_location"
        elif target_kind == "fixture":
            target_mode = "controlled_fixture"
        else:
            target_mode = None
        runtime_decision = {
            "grounded": grounded,
            "status": status,
            "confidence_threshold": self.grounding_confidence_threshold,
            "decision_scope": "target_and_navigation_only",
            "parsed_json": parsed,
            "validation_attempts": validation_attempts,
        }
        artifact_dir, image_paths = save_vlm_artifacts(
            root=self.log_dir,
            category="vlm_grounding",
            request_id=f"{self._request_count:06d}_{call.subgoal_id}",
            bundle=bundle,
            system_prompt=GROUNDING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            client=self.client,
            parsed_json=parsed,
            runtime_decision=runtime_decision,
        )
        evidence = {
            "source": "multi_view_vlm_grounding",
            "camera_names": list(bundle.camera_names),
            "image_paths": image_paths,
            "purpose": bundle.purpose,
            "timestamp": bundle.timestamp,
            "observation_step": bundle.step_index,
            "system_prompt": GROUNDING_SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "raw_response": getattr(self.client, "last_response_text", None),
            "raw_http_response": getattr(self.client, "last_raw_http_response", None),
            "parsed_json": parsed,
            "validation_attempts": validation_attempts,
            "confidence": parsed["confidence"],
            "supporting_views": parsed["supporting_views"],
            "final_runtime_decision": runtime_decision,
            "artifact_dir": artifact_dir,
        }
        return GroundingResult(
            grounded=grounded,
            status=status,
            target_entity_alias=target_alias,
            target_entity_kind=target_kind,
            entity_role=entity_role,
            target_fixture_alias=runtime_fixture_alias,
            reference_object_alias=reference_alias,
            held_object_alias=None,
            holding_source_object=None,
            source_object_alias=source_alias,
            destination_entity_alias=destination_alias,
            destination_role=destination_role,
            target_mode=target_mode,
            reason=parsed["reason"],
            evidence=(evidence,),
        )
