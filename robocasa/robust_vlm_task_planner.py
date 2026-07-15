"""Validation-retrying VLM planner tuned for small OpenAI-compatible VLMs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .atomic_task_schemas import AtomicTaskCall
    from .skill_contract_registry import apply_skill_contract
    from .vlm_task_planner import OpenAICompatibleVLMPlanner
except ImportError:
    from atomic_task_schemas import AtomicTaskCall
    from skill_contract_registry import apply_skill_contract
    from vlm_task_planner import OpenAICompatibleVLMPlanner


def _controlled_decomposition(
    task: str, scene_context: Mapping[str, Any]
) -> tuple[list[AtomicTaskCall], dict[str, Any]] | None:
    if str(scene_context.get("env_name", "")) != "PlaceEqualIceCubes":
        return None
    object_aliases = {
        str(entity.get("alias"))
        for entity in scene_context.get("objects", [])
        if isinstance(entity, Mapping) and entity.get("alias")
    }
    required = {
        "ice_bowl",
        "ice_cube1",
        "ice_cube2",
        "ice_cube3",
        "ice_cube4",
        "glass_cup1",
        "glass_cup2",
    }
    missing = sorted(required.difference(object_aliases))
    if missing:
        raise ValueError(
            f"PlaceEqualIceCubes controlled decomposition is missing aliases: {missing}"
        )
    calls: list[AtomicTaskCall] = []
    for cup_index in (1, 2):
        cup_alias = f"glass_cup{cup_index}"
        cup_name = f"glass cup {cup_index}"
        for target_count in (1, 2):
            calls.append(
                AtomicTaskCall.from_mapping(
                    {
                        "subgoal_id": f"cup{cup_index}_ice{target_count}",
                        "atomic_task": "MakeIcedCoffee",
                        "policy_prompt": (
                            "Pick up one ice cube from the ice bowl and place it "
                            f"in {cup_name}."
                        ),
                        "arguments": {
                            "source_id": "ice_bowl",
                            "receptacle_id": cup_alias,
                            "object_group": "ice_cube",
                            "target_count": target_count,
                        },
                        "termination_condition": {
                            "predicate": "receptacle_count",
                            "subject": cup_alias,
                            "object_prefix": "ice_cube",
                            "desired_value": target_count,
                        },
                        "metadata": {
                            "controlled_decomposition": "PlaceEqualIceCubes",
                            "cup_index": cup_index,
                            "target_count": target_count,
                        },
                    }
                )
            )
    provenance = {
        "source": "controlled_decomposition",
        "rule": "PlaceEqualIceCubes_to_4x_MakeIcedCoffee",
        "long_horizon_task": task,
        "num_atomic_task_calls": len(calls),
    }
    return calls, provenance


def _validate_execution_plan(
    calls: Sequence[AtomicTaskCall],
    scene_context: Mapping[str, Any],
) -> None:
    """Reject structurally valid plans that cannot be verified or safely ordered."""

    aliases = {
        str(entity.get("alias"))
        for group in ("fixtures", "objects")
        for entity in scene_context.get(group, [])
        if isinstance(entity, Mapping) and entity.get("alias")
    }
    fixture_aliases = {
        str(entity.get("alias"))
        for entity in scene_context.get("fixtures", [])
        if isinstance(entity, Mapping) and entity.get("alias")
    }
    relation_predicates = {"inside", "on", "inserted", "object_fixture_relation"}
    task_text = str(scene_context.get("long_horizon_task", "")).lower()
    fixture_terms = {
        "Microwave": {"microwave"},
        "Stove": {"stove", "stovetop", "burner"},
        "Sink": {"sink", "faucet"},
    }
    explicit_fixtures = {
        family
        for family, terms in fixture_terms.items()
        if any(term in task_text for term in terms)
    }
    for call in calls:
        for family in fixture_terms:
            if (
                family in call.atomic_task
                and explicit_fixtures
                and family not in explicit_fixtures
            ):
                raise ValueError(
                    f"{call.subgoal_id} uses {call.atomic_task}, but the requested fixtures "
                    f"are {sorted(explicit_fixtures)}"
                )
        conditions = (
            call.termination_condition
            if isinstance(call.termination_condition, list)
            else [call.termination_condition]
        )
        for condition in conditions:
            missing = [
                key
                for key in ("predicate", "subject", "desired_value")
                if key not in condition
            ]
            if missing:
                raise ValueError(
                    f"{call.subgoal_id} termination_condition is missing {missing}"
                )
            predicate = str(condition["predicate"])
            if call.atomic_task == "NavigateKitchen":
                target_alias = str(condition["subject"])
                if predicate != "navigation_pose":
                    raise ValueError(
                        f"{call.subgoal_id} NavigateKitchen requires navigation_pose"
                    )
                if fixture_aliases and target_alias not in fixture_aliases:
                    raise ValueError(
                        f"{call.subgoal_id} navigation target must be a fixture alias"
                    )
                if call.arguments.get("fixture_id") != target_alias:
                    raise ValueError(
                        f"{call.subgoal_id} fixture_id must match navigation subject"
                    )
            if predicate in relation_predicates and not condition.get("object"):
                raise ValueError(
                    f"{call.subgoal_id} relation predicate {predicate!r} requires object"
                )
            if aliases:
                for key in ("subject", "object"):
                    identifier = condition.get(key)
                    if (
                        key == "subject"
                        and predicate == "released"
                        and identifier == condition.get("object_prefix")
                    ):
                        continue
                    if identifier and str(identifier) not in aliases:
                        raise ValueError(
                            f"{call.subgoal_id} condition {key}={identifier!r} is not an "
                            f"exact scene alias; available aliases are {sorted(aliases)}"
                        )

    names = [call.atomic_task for call in calls]
    for turn_index, name in enumerate(names):
        if name != "TurnOnMicrowave":
            continue
        last_open_index = max(
            (
                index
                for index, candidate in enumerate(names[:turn_index])
                if candidate in {"OpenMicrowave", "PickPlaceCounterToMicrowave"}
            ),
            default=-1,
        )
        if (
            last_open_index >= 0
            and "CloseMicrowave" not in names[last_open_index + 1 : turn_index]
        ):
            raise ValueError(
                "TurnOnMicrowave requires CloseMicrowave after the preceding microwave "
                "open/pick-place step"
            )


def _normalize_execution_plan(
    calls: Sequence[AtomicTaskCall],
    scene_context: Mapping[str, Any],
) -> tuple[list[AtomicTaskCall], list[dict[str, Any]]]:
    """Apply explicit scheduler safety/default rules and report every change."""

    normalized: list[AtomicTaskCall] = []
    changes: list[dict[str, Any]] = []
    scene_aliases = {
        str(entity.get("alias"))
        for group in ("fixtures", "objects")
        for entity in scene_context.get(group, [])
        if isinstance(entity, Mapping) and entity.get("alias")
    }
    fixture_aliases = {
        str(entity.get("alias"))
        for entity in scene_context.get("fixtures", [])
        if isinstance(entity, Mapping) and entity.get("alias")
    }
    movable_aliases = {
        str(entity.get("alias"))
        for entity in scene_context.get("objects", [])
        if isinstance(entity, Mapping) and entity.get("alias")
    }
    alias_lookup: dict[str, str] = {}
    for group in ("fixtures", "objects"):
        for entity in scene_context.get(group, []):
            if not isinstance(entity, Mapping) or not entity.get("alias"):
                continue
            alias = str(entity["alias"])
            for field in ("alias", "name", "natural_name", "type"):
                candidate = str(entity.get(field, "")).strip().lower().replace(" ", "_")
                if candidate:
                    alias_lookup.setdefault(candidate, alias)

    def resolve_scene_identifier(identifier: Any) -> str | None:
        if identifier is None:
            return None
        text = str(identifier).strip()
        if not text:
            return None
        if text in scene_aliases:
            return text
        return alias_lookup.get(text.lower().replace(" ", "_"))

    def infer_prompt_destination(prompt: Any, subject: Any) -> str | None:
        if not isinstance(prompt, str) or not prompt.strip():
            return None
        normalized_prompt = "_" + "_".join(
            part
            for part in "".join(
                char.lower() if char.isalnum() else " " for char in prompt
            ).split()
            if part
        ) + "_"
        candidates = {
            alias
            for lookup_key, alias in alias_lookup.items()
            if len(lookup_key) >= 3
            and f"_{lookup_key.strip('_')}_" in normalized_prompt
            and alias != subject
        }
        movable_candidates = candidates.intersection(movable_aliases)
        if len(movable_candidates) == 1:
            return next(iter(movable_candidates))
        if len(candidates) == 1:
            return next(iter(candidates))
        return None

    fixture_natural_names = {
        str(entity["alias"]): str(entity.get("natural_name") or entity["alias"])
        for entity in scene_context.get("fixtures", [])
        if isinstance(entity, Mapping) and entity.get("alias")
    }
    microwave_alias = alias_lookup.get("microwave", "microwave")
    microwave_opened = False
    used_ids = {call.subgoal_id for call in calls}
    for call in calls:
        value = call.to_dict()
        conditions = value["termination_condition"]
        condition_items = conditions if isinstance(conditions, list) else [conditions]
        for condition in condition_items:
            if "desired_value" not in condition:
                condition["desired_value"] = True
                changes.append(
                    {
                        "type": "default_desired_value",
                        "subgoal_id": call.subgoal_id,
                        "value": True,
                    }
                )
            for field in ("subject", "object"):
                identifier = condition.get(field)
                if (
                    field == "subject"
                    and condition.get("predicate") == "released"
                    and condition.get("object_prefix")
                ):
                    continue
                if not identifier or str(identifier) in scene_aliases:
                    continue
                lookup_key = str(identifier).strip().lower().replace(" ", "_")
                resolved = alias_lookup.get(lookup_key)
                if (
                    resolved is None
                    and field == "subject"
                    and lookup_key in {"object", "target_object"}
                    and "obj" in scene_aliases
                ):
                    resolved = "obj"
                if resolved is not None:
                    condition[field] = resolved
                    changes.append(
                        {
                            "type": "resolve_scene_alias",
                            "subgoal_id": call.subgoal_id,
                            "field": field,
                            "from": str(identifier),
                            "value": resolved,
                        }
                    )
            predicate = str(condition.get("predicate", "")).strip().lower()
            relation_predicates = {
                "inside",
                "inserted",
                "on",
                "object_fixture_relation",
            }
            if predicate in relation_predicates:
                subject = condition.get("subject")
                destination = condition.get("object")
                if not destination:
                    source = None
                    arguments = value["arguments"]
                    for key in (
                        "destination_id",
                        "destination",
                        "destination_name",
                        "receptacle_id",
                        "receptacle",
                        "receptacle_name",
                        "container_id",
                        "container_name",
                        "fixture_id",
                        "fixture_name",
                        "target_id",
                        "target_name",
                    ):
                        resolved = resolve_scene_identifier(arguments.get(key))
                        if resolved is not None and resolved != subject:
                            destination = resolved
                            source = f"arguments.{key}"
                            break
                    if not destination:
                        destination = infer_prompt_destination(
                            value.get("policy_prompt"), subject
                        )
                        if destination is not None:
                            source = "policy_prompt"
                    if destination is not None:
                        condition["object"] = destination
                        changes.append(
                            {
                                "type": "infer_relation_destination",
                                "subgoal_id": call.subgoal_id,
                                "predicate": predicate,
                                "subject": subject,
                                "object": destination,
                                "source": source,
                            }
                        )

                if (
                    subject in fixture_aliases
                    and destination in movable_aliases
                ):
                    condition["subject"], condition["object"] = destination, subject
                    changes.append(
                        {
                            "type": "swap_relation_direction",
                            "subgoal_id": call.subgoal_id,
                            "predicate": predicate,
                            "subject": destination,
                            "object": subject,
                        }
                    )
            elif "object" in condition:
                removed_object = condition.pop("object")
                changes.append(
                    {
                        "type": "remove_extraneous_relation_object",
                        "subgoal_id": call.subgoal_id,
                        "predicate": predicate,
                        "value": removed_object,
                    }
                )
        if value["atomic_task"] == "NavigateKitchen":
            arguments = value["arguments"]
            target_identifier = (
                arguments.get("fixture_id")
                or arguments.get("fixture_name")
                or condition_items[0].get("subject")
            )
            target_key = (
                str(target_identifier or "")
                .strip()
                .lower()
                .replace(" ", "_")
            )
            target_alias = (
                str(target_identifier)
                if str(target_identifier) in fixture_aliases
                else alias_lookup.get(target_key)
            )
            if target_alias in fixture_aliases:
                arguments["fixture_id"] = target_alias
                arguments["fixture_name"] = fixture_natural_names.get(
                    target_alias, target_key.replace("_", " ")
                )
                for condition in condition_items:
                    predicate = str(condition.get("predicate", "")).lower()
                    if predicate in {"at", "near", "navigation", "navigation_pose"}:
                        condition["predicate"] = "navigation_pose"
                    condition["subject"] = target_alias
                changes.append(
                    {
                        "type": "resolve_navigation_target",
                        "subgoal_id": call.subgoal_id,
                        "subject": target_alias,
                    }
                )
        call = AtomicTaskCall.from_mapping(value)

        if call.atomic_task in {"OpenMicrowave", "PickPlaceCounterToMicrowave"}:
            microwave_opened = True
        elif call.atomic_task == "CloseMicrowave":
            microwave_opened = False
        elif call.atomic_task == "TurnOnMicrowave" and microwave_opened:
            subgoal_id = f"close_microwave_before_{call.subgoal_id}"
            suffix = 2
            while subgoal_id in used_ids:
                subgoal_id = f"close_microwave_before_{call.subgoal_id}_{suffix}"
                suffix += 1
            used_ids.add(subgoal_id)
            close_call = AtomicTaskCall.from_mapping(
                {
                    "subgoal_id": subgoal_id,
                    "atomic_task": "CloseMicrowave",
                    "policy_prompt": "Close the microwave door.",
                    "arguments": {
                        "fixture_id": microwave_alias,
                        "fixture_name": "microwave",
                    },
                    "termination_condition": {
                        "predicate": "closed",
                        "subject": microwave_alias,
                        "desired_value": True,
                    },
                    "metadata": {
                        "inserted_by": "planner_safety_normalizer",
                        "reason": "microwave_must_be_closed_before_turn_on",
                    },
                }
            )
            normalized.append(close_call)
            microwave_opened = False
            changes.append(
                {
                    "type": "insert_prerequisite",
                    "before_subgoal_id": call.subgoal_id,
                    "inserted_subgoal_id": subgoal_id,
                    "atomic_task": "CloseMicrowave",
                }
            )
        normalized.append(call)
    contracted: list[AtomicTaskCall] = []
    for call in normalized:
        enriched_call, contract_changes = apply_skill_contract(
            call, scene_context=scene_context
        )
        contracted.append(enriched_call)
        changes.extend(contract_changes)
    return contracted, changes


def prepare_execution_plan(
    calls: Sequence[AtomicTaskCall],
    scene_context: Mapping[str, Any] | None = None,
) -> tuple[list[AtomicTaskCall], list[dict[str, Any]]]:
    """Normalize, contract-enrich, and validate planner or scheduler calls."""

    context = dict(scene_context or {})
    prepared, changes = _normalize_execution_plan(calls, context)
    _validate_execution_plan(prepared, context)
    return prepared, changes


class RobustOpenAICompatibleVLMPlanner(OpenAICompatibleVLMPlanner):
    """Retry invalid plans with concrete parser feedback and the prior response."""

    def __init__(self, *, max_validation_retries: int = 2, **kwargs: Any):
        if max_validation_retries < 0:
            raise ValueError("max_validation_retries must be non-negative")
        super().__init__(**kwargs)
        self.max_validation_retries = int(max_validation_retries)

    def _system_prompt(self) -> str:
        base = super()._system_prompt()
        return (
            base
            + "\nSTRICT OUTPUT RULES:\n"
            "- termination_condition MUST be a JSON object or a list of JSON objects, "
            "never a sentence or string.\n"
            "- Each condition object MUST contain predicate, subject, and desired_value. "
            "Use object for a destination relation such as inside/on.\n"
            "- arguments MUST be a JSON object even when empty.\n"
            "- NavigateKitchen MUST set arguments.fixture_name to the target fixture and "
            "use {predicate:navigation_pose, subject:<target fixture alias>, "
            "desired_value:true} as its termination condition.\n"
            "- Inspect fixture state in Scene context. Do not open an already-open fixture "
            "or close an already-closed fixture.\n"
            "- A microwave must be closed before TurnOnMicrowave. When an object has just "
            "been placed into an open microwave, schedule CloseMicrowave before "
            "TurnOnMicrowave.\n"
            "Example condition objects: "
            '{"predicate":"inside","subject":"obj","object":"microwave",'
            '"desired_value":true}; '
            '{"predicate":"closed","subject":"microwave","desired_value":true}; '
            '{"predicate":"powered","subject":"microwave","desired_value":true}; '
            '{"predicate":"navigation_pose","subject":"sink",'
            '"desired_value":true}.\n'
        )

    def plan(
        self,
        *,
        task: str,
        images: Sequence[np.ndarray] | None = None,
        scene_context: Mapping[str, Any] | None = None,
    ) -> tuple[list[AtomicTaskCall], dict[str, Any]]:
        attempts: list[dict[str, str]] = []
        context = dict(scene_context or {})
        controlled = _controlled_decomposition(task, context)
        if controlled is not None:
            calls, provenance = controlled
            calls, normalizations = prepare_execution_plan(calls, context)
            if normalizations:
                provenance["plan_normalizations"] = normalizations
            return calls, provenance
        for attempt_index in range(self.max_validation_retries + 1):
            try:
                calls, provenance = super().plan(
                    task=task,
                    images=images,
                    scene_context=context,
                )
                calls, normalizations = prepare_execution_plan(calls, context)
                if normalizations:
                    provenance["plan_normalizations"] = normalizations
                provenance["validation_attempts"] = attempt_index + 1
                if attempts:
                    provenance["invalid_attempts"] = attempts
                return calls, provenance
            except (TypeError, ValueError) as exc:
                raw_response = self.last_response_text or ""
                attempts.append(
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_response": raw_response,
                    }
                )
                if attempt_index >= self.max_validation_retries:
                    raise ValueError(
                        f"VLM plan remained invalid after {attempt_index + 1} attempt(s): {exc}"
                    ) from exc
                context["planner_correction"] = {
                    "instruction": (
                        "The previous response below failed strict validation. Return the "
                        "entire corrected plan as JSON. Do not repeat the same schema error."
                    ),
                    "validation_error": str(exc),
                    "previous_invalid_response": raw_response,
                }
        raise AssertionError("unreachable VLM validation loop")
