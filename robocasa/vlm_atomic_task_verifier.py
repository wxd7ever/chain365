"""Visual task critic with a legacy evidence-verifier baseline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .atomic_task_schemas import AtomicTaskCall
    from .evidence_requirements import (
        HOLDING_SIDE_VIEWS,
        missing_required_views,
    )
    from .multi_view_perception import MultiViewObservationProvider
    from .view_scheduler import FixedViewScheduler
    from .vlm_artifacts import save_vlm_artifacts
except ImportError:
    from atomic_task_schemas import AtomicTaskCall
    from evidence_requirements import (
        HOLDING_SIDE_VIEWS,
        missing_required_views,
    )
    from multi_view_perception import MultiViewObservationProvider
    from view_scheduler import FixedViewScheduler
    from vlm_artifacts import save_vlm_artifacts


CRITIC_SYSTEM_PROMPT = """You are a robot task critic.
Your job is to determine whether the robot successfully completed the atomic task.

Judge only from the observation images captured at the current verification checkpoint.
Do not assume success because the robot attempted the action.
Return success only when there is clear visual evidence.
Judge the task as a whole. Do not predict hidden world state, expose a predicate-by-
predicate analysis, build an evidence graph, or require temporal evidence.

Return exactly one JSON object:
{
  "success": true or false,
  "confidence": a number from 0.0 to 1.0,
  "reason": "brief explanation",
  "failure_reason": "brief reason for failure, or an empty string on success"
}"""


VERIFIER_SYSTEM_PROMPT = """You are a conservative multi-view RoboCasa atomic-task verifier.
Judge every supplied required condition separately using only the fresh labelled images.
The candidate aliases are vocabulary constraints, not simulator-state evidence. In each
condition, value means whether the complete required condition (including desired_value)
is visually satisfied. Use null when occlusion, image quality, or conflicting views make
the condition visually indeterminate. For non-holding predicates, directly judge the
requested visible end condition. One clear supporting view is sufficient; do not invent
extra geometric subconditions. Never guess true merely to finish a task.
Return exactly one JSON object with conditions, overall_status, confidence, reason. Each condition must
contain predicate, subject, object, value, confidence, supporting_views, reason and must
appear in the same order as the required conditions. overall_status is SUCCESS only when
every required all_of branch is confidently satisfied, NOT_YET when visual evidence
clearly shows a required goal is not satisfied, and UNKNOWN otherwise."""


HOLDING_EVIDENCE_SYSTEM_PROMPT = """You extract role-separated visual evidence for a
RoboCasa holding predicate. Do not decide task success. Use each camera only for its
assigned role:
- robot0_eye_in_hand judges object contact/attachment to the end effector and whether the
  object appears fixed relative to the gripper. It must not judge finger-side geometry or
  whether the object left its source region.
- robot0_agentview_left and robot0_agentview_right judge side geometry: whether the object
  is between the gripper fingers and whether the gripper appears closed around it.
- external views, especially robot0_frontview, judge whether the object left the named
  source region. Never infer source-region membership from robot0_eye_in_hand.
Return exactly one JSON object with object_visible, object_location,
object_between_gripper_fingers, gripper_closed, object_contact_with_gripper,
object_left_source_region, source_region, holding_evidence, contradicting_evidence,
contradicting_views, supporting_views, view_evidence, confidence, reason. The global
object_contact_with_gripper summarizes eye-in-hand attachment/contact evidence; the global
object_between_gripper_fingers and gripper_closed summarize side-view evidence; the global
object_left_source_region summarizes external-view evidence. These evidence fields may be
null when their assigned views are genuinely occluded. view_evidence is a list of
per-camera objects with camera_name, object_visible, object_location,
object_between_gripper_fingers, object_left_source_region, confidence, reason. For
robot0_eye_in_hand, both per-view object_between_gripper_fingers and
object_left_source_region MUST be null. holding_evidence MUST be one JSON boolean or null,
never an array or object; put per-camera records only in view_evidence.
contradicting_evidence MUST be a JSON array of strings; use an empty array when there is no
contradictory evidence. supporting_views and contradicting_views MUST be JSON arrays of
camera-name strings. Report contradictory evidence only from a camera assigned to judge
that fact."""


NAVIGATION_SYSTEM_PROMPT = """You are a conservative multi-view RoboCasa navigation verifier.
Use only fresh labelled images. Do not estimate metric distance or orientation and do not
use simulator navigation poses. Success means the robot is visibly in the target fixture's
vicinity and is suitable for handing off to local work-pose refinement, not that it reached
a final metric manipulation pose. If the navigation call requires holding an object, verify
that visually; use null if the grasp is occluded or ambiguous.
Return one JSON object with target_visible, target_reached,
suitable_for_local_refinement, holding_required_object,
confidence, supporting_views, reason."""


def _confidence(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def _required_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _nullable_bool(value: Any, *, field: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean or null")
    return value


def _normalise_evidence_bool(
    value: Any,
    *,
    field: str,
    normalizations: list[dict[str, Any]],
) -> bool | None:
    """Coerce common VLM boolean variants without turning uncertainty into success."""

    if isinstance(value, bool) or value is None:
        return value
    normalized: bool | None
    if isinstance(value, int) and value in (0, 1):
        normalized = bool(value)
    elif isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "yes", "1"}:
            normalized = True
        elif token in {"false", "no", "0"}:
            normalized = False
        elif token in {"null", "none", "unknown", "uncertain", "occluded"}:
            normalized = None
        else:
            raise ValueError(f"{field} must be boolean or null")
    else:
        raise ValueError(f"{field} must be boolean or null")
    normalizations.append(
        {
            "field": field,
            "from_type": type(value).__name__,
            "value": normalized,
            "reason": "coerced common VLM boolean representation",
        }
    )
    return normalized


def _normalise_evidence_confidence(
    value: Any,
    *,
    field: str,
    normalizations: list[dict[str, Any]],
) -> float:
    if isinstance(value, str):
        try:
            normalized = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"{field} must be numeric") from exc
        normalizations.append(
            {
                "field": field,
                "from_type": "str",
                "value": normalized,
                "reason": "parsed numeric VLM confidence string",
            }
        )
        value = normalized
    return _confidence(value, field=field)


def _normalise_evidence_text_list(
    value: Any,
    *,
    field: str,
    normalizations: list[dict[str, Any]],
) -> list[str]:
    """Normalize optional explanatory evidence while retaining its diagnostics."""

    raw_value = value
    if value is None or value is False:
        items: list[Any] = []
    elif value is True:
        items = ["unspecified contradictory evidence"]
    elif isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    elif isinstance(value, Mapping):
        items = [value]
    else:
        raise ValueError(f"{field} must contain textual evidence")

    result: list[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, Mapping):
            text = next(
                (
                    candidate.strip()
                    for key in ("reason", "text", "description", "evidence")
                    if isinstance((candidate := item.get(key)), str)
                    and candidate.strip()
                ),
                "",
            )
            if not text:
                text = json.dumps(dict(item), ensure_ascii=False, sort_keys=True)
        else:
            text = str(item).strip()
        if text:
            result.append(text)
    if not isinstance(raw_value, list) or result != raw_value:
        normalizations.append(
            {
                "field": field,
                "from_type": type(raw_value).__name__,
                "value": result,
                "reason": "normalized optional textual evidence to list[str]",
            }
        )
    return result


def _normalise_region(value: Any) -> str:
    return "_".join(str(value or "").strip().lower().replace("_", " ").split())


def _required_conditions(call: AtomicTaskCall) -> Mapping[str, Any]:
    conditions = call.termination_condition
    if isinstance(conditions, list):
        if not conditions:
            raise ValueError("termination_condition must not be empty")
        return {"all_of": conditions}
    if not isinstance(conditions, Mapping) or not conditions:
        raise ValueError("termination_condition must be a non-empty mapping or list")
    return conditions


def _condition_tree(
    condition: Mapping[str, Any], leaves: list[dict[str, Any]]
) -> dict[str, Any]:
    operators = [name for name in ("all_of", "any_of") if name in condition]
    if operators:
        if len(operators) != 1 or set(condition) != {operators[0]}:
            raise ValueError("boolean termination condition has invalid fields")
        operator = operators[0]
        children = condition[operator]
        if not isinstance(children, list) or not children or not all(
            isinstance(item, Mapping) for item in children
        ):
            raise ValueError(f"{operator} must contain non-empty condition mappings")
        return {operator: [_condition_tree(item, leaves) for item in children]}
    predicate = str(condition.get("predicate", "")).strip()
    subject = str(condition.get("subject", "")).strip()
    if not predicate or not subject:
        raise ValueError("each visual condition requires predicate and subject")
    object_id = condition.get("object", condition.get("destination"))
    leaf = {
        "predicate": predicate,
        "subject": subject,
        "object": None if object_id is None else str(object_id).strip(),
        "desired_value": condition.get("desired_value", True),
    }
    # Retain non-privileged semantic parameters for the visual prompt.
    for key in ("object_prefix", "threshold", "count"):
        if key in condition:
            leaf[key] = condition[key]
    index = len(leaves)
    leaves.append(leaf)
    return {"leaf": index}


def _evaluate_tree(tree: Mapping[str, Any], values: Sequence[bool | None]) -> bool | None:
    if "leaf" in tree:
        return values[int(tree["leaf"])]
    operator = "all_of" if "all_of" in tree else "any_of"
    children = [_evaluate_tree(item, values) for item in tree[operator]]
    if operator == "all_of":
        if any(value is False for value in children):
            return False
        return True if all(value is True for value in children) else None
    if any(value is True for value in children):
        return True
    return False if all(value is False for value in children) else None



def _entities(scene_context: Mapping[str, Any], key: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in scene_context.get(key, []) or []:
        if not isinstance(item, Mapping):
            continue
        alias = str(item.get("alias", "")).strip()
        if alias:
            result.append(
                {
                    "alias": alias,
                    "name": str(item.get("name", alias)),
                    "natural_name": str(
                        item.get("natural_name", item.get("name", alias))
                    ),
                }
            )
    return result


class VLMAtomicTaskVerifier:
    """Drop-in callable verifier that never reads privileged simulator state."""

    def __init__(
        self,
        *,
        client: Any,
        observation_provider: MultiViewObservationProvider,
        view_scheduler: FixedViewScheduler | None = None,
        scene_context: Mapping[str, Any] | None = None,
        success_threshold: float = 0.70,
        default_required_consecutive_successes: int = 1,
        critic_only: bool = False,
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
        if (
            isinstance(default_required_consecutive_successes, bool)
            or not isinstance(default_required_consecutive_successes, int)
            or default_required_consecutive_successes <= 0
        ):
            raise ValueError(
                "default_required_consecutive_successes must be a positive integer"
            )
        self.client = client
        self.observation_provider = observation_provider
        self.view_scheduler = view_scheduler
        self.scene_context = dict(scene_context or {})
        self.success_threshold = _confidence(
            success_threshold, field="success_threshold"
        )
        self.default_required_consecutive_successes = (
            default_required_consecutive_successes
        )
        self.critic_only = bool(critic_only)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._consecutive_state: dict[tuple[int, str, str], dict[str, int]] = {}
        self._request_count = 0

    def _required_consecutive_successes(self, call: AtomicTaskCall) -> int:
        contract = call.metadata.get("skill_contract", {})
        verification = (
            contract.get("verification", {}) if isinstance(contract, Mapping) else {}
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
            raise ValueError("required_consecutive_successes must be positive")
        return required

    def _record_consecutive(
        self,
        *,
        env: Any,
        call: AtomicTaskCall,
        condition_key: str,
        step_index: int,
        satisfied: bool,
    ) -> int:
        key = (id(env), call.subgoal_id, condition_key)
        previous = self._consecutive_state.get(key, {"last_step": -1, "count": 0})
        last_step, count = previous["last_step"], previous["count"]
        if step_index < last_step:
            last_step, count = -1, 0
        if step_index > last_step:
            count = count + 1 if satisfied else 0
        elif not satisfied:
            count = 0
        self._consecutive_state[key] = {"last_step": step_index, "count": count}
        return count

    @staticmethod
    def _validate_views(value: Any, camera_names: Sequence[str]) -> list[str]:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("supporting_views must be a list of camera names")
        if set(value) - set(camera_names):
            raise ValueError("supporting_views contains an unknown camera")
        return list(dict.fromkeys(value))

    def _validate_generic(
        self,
        value: Mapping[str, Any],
        *,
        leaves: Sequence[Mapping[str, Any]],
        camera_names: Sequence[str],
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("verifier response must be a JSON object")
        conditions = value.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != len(leaves):
            raise ValueError("VLM must return exactly one result per required condition")
        validated: list[dict[str, Any]] = []
        for expected, item in zip(leaves, conditions):
            if not isinstance(item, Mapping):
                raise ValueError("each VLM condition result must be an object")
            result = dict(item)
            for field in ("predicate", "subject"):
                if str(result.get(field, "")).strip() != expected[field]:
                    raise ValueError(f"VLM condition {field} does not match request")
            actual_object = result.get("object")
            actual_object = None if actual_object is None else str(actual_object).strip()
            if actual_object != expected.get("object"):
                raise ValueError("VLM condition object does not match request")
            condition_value = result.get("value")
            if condition_value is not None and not isinstance(condition_value, bool):
                raise ValueError("condition value must be true, false, or null")
            result["value"] = condition_value
            result["object"] = actual_object
            result["confidence"] = _confidence(
                result.get("confidence"), field="condition confidence"
            )
            result["supporting_views"] = self._validate_views(
                result.get("supporting_views"), camera_names
            )
            if not isinstance(result.get("reason"), str):
                raise ValueError("condition reason must be a string")
            validated.append(result)
        overall_status = str(value.get("overall_status", "")).strip().upper()
        if overall_status not in {"SUCCESS", "NOT_YET", "UNKNOWN"}:
            raise ValueError("overall_status must be SUCCESS, NOT_YET, or UNKNOWN")
        reason = value.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("verifier reason must be a non-empty string")
        return {
            "conditions": validated,
            "overall_status": overall_status,
            "confidence": _confidence(value.get("confidence"), field="confidence"),
            "reason": reason.strip(),
        }

    def _validate_holding_evidence(
        self,
        value: Mapping[str, Any],
        *,
        camera_names: Sequence[str],
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("holding evidence response must be a JSON object")
        result = dict(value)
        schema_normalizations: list[dict[str, Any]] = []
        for field in (
            "object_visible",
            "object_between_gripper_fingers",
            "gripper_closed",
            "object_contact_with_gripper",
            "object_left_source_region",
        ):
            result[field] = _normalise_evidence_bool(
                result.get(field),
                field=field,
                normalizations=schema_normalizations,
            )
        raw_holding_evidence = result.get("holding_evidence")
        if isinstance(raw_holding_evidence, bool):
            result["holding_evidence"] = raw_holding_evidence
        elif isinstance(raw_holding_evidence, list):
            # Some VLMs confuse the scalar summary with the similarly named
            # per-camera evidence list. The program never trusts this claim for
            # success, so retain it diagnostically and use view_evidence below.
            result["holding_evidence_raw"] = raw_holding_evidence
            result["holding_evidence"] = None
            schema_normalizations.append(
                {
                    "field": "holding_evidence",
                    "from_type": "list",
                    "value": None,
                    "reason": "per-camera records belong in view_evidence",
                }
            )
            if not result.get("view_evidence"):
                result["view_evidence"] = raw_holding_evidence
        elif raw_holding_evidence is None:
            result["holding_evidence"] = None
        elif isinstance(raw_holding_evidence, (str, int)):
            result["holding_evidence"] = _normalise_evidence_bool(
                raw_holding_evidence,
                field="holding_evidence",
                normalizations=schema_normalizations,
            )
        elif (
            isinstance(raw_holding_evidence, Mapping)
            and "value" in raw_holding_evidence
        ):
            result["holding_evidence_raw"] = dict(raw_holding_evidence)
            result["holding_evidence"] = _normalise_evidence_bool(
                raw_holding_evidence.get("value"),
                field="holding_evidence",
                normalizations=schema_normalizations,
            )
        else:
            raise ValueError(
                "holding_evidence must be boolean, list, object with value, or null"
            )
        for field in ("object_location", "source_region"):
            field_value = result.get(field)
            if field_value is not None and not isinstance(field_value, str):
                raise ValueError(f"{field} must be a string or null")
            result[field] = field_value.strip() if isinstance(field_value, str) else None
        result["contradicting_evidence"] = _normalise_evidence_text_list(
            result.get("contradicting_evidence"),
            field="contradicting_evidence",
            normalizations=schema_normalizations,
        )
        result["supporting_views"] = self._validate_views(
            result.get("supporting_views"), camera_names
        )
        result["contradicting_views"] = self._validate_views(
            result.get("contradicting_views", []), camera_names
        )
        result["confidence"] = _normalise_evidence_confidence(
            result.get("confidence"),
            field="holding evidence confidence",
            normalizations=schema_normalizations,
        )
        if not isinstance(result.get("reason"), str) or not result["reason"].strip():
            raise ValueError("holding evidence reason must be a non-empty string")
        result["reason"] = result["reason"].strip()

        raw_view_evidence = result.get("view_evidence", [])
        if not isinstance(raw_view_evidence, list):
            raise ValueError("view_evidence must be a list")
        validated_views: list[dict[str, Any]] = []
        for item in raw_view_evidence:
            if not isinstance(item, Mapping):
                raise ValueError("each view_evidence item must be an object")
            view = dict(item)
            camera_name = str(view.get("camera_name", "")).strip()
            if camera_name not in camera_names:
                raise ValueError("view_evidence contains an unknown camera")
            view["camera_name"] = camera_name
            view["object_visible"] = _normalise_evidence_bool(
                view.get("object_visible"),
                field=f"view_evidence[{camera_name}].object_visible",
                normalizations=schema_normalizations,
            )
            view["object_between_gripper_fingers"] = _normalise_evidence_bool(
                view.get("object_between_gripper_fingers"),
                field=f"view_evidence[{camera_name}].object_between_gripper_fingers",
                normalizations=schema_normalizations,
            )
            view["object_left_source_region"] = _normalise_evidence_bool(
                view.get("object_left_source_region"),
                field=f"view_evidence[{camera_name}].object_left_source_region",
                normalizations=schema_normalizations,
            )
            location = view.get("object_location")
            if camera_name == "robot0_eye_in_hand":
                for field in (
                    "object_between_gripper_fingers",
                    "object_left_source_region",
                ):
                    if view[field] is not None:
                        schema_normalizations.append(
                            {
                                "field": f"view_evidence[{camera_name}].{field}",
                                "value": None,
                                "reason": "outside the eye-in-hand evidence role",
                            }
                        )
                    view[field] = None
            if location is not None and not isinstance(location, str):
                raise ValueError("view object_location must be a string or null")
            view["object_location"] = (
                location.strip() if isinstance(location, str) else None
            )
            view["confidence"] = _normalise_evidence_confidence(
                view.get("confidence"),
                field=f"view_evidence[{camera_name}].confidence",
                normalizations=schema_normalizations,
            )
            if not isinstance(view.get("reason"), str):
                raise ValueError("view evidence reason must be a string")
            validated_views.append(view)
        result["view_evidence"] = validated_views
        if schema_normalizations:
            result["schema_normalizations"] = schema_normalizations
        return result

    def verify_holding(
        self,
        *,
        call: AtomicTaskCall,
        object_alias: str,
        source_region: str | None,
        images: Mapping[str, Any],
        camera_names: Sequence[str],
    ) -> tuple[dict[str, Any], str]:
        object_name = object_alias
        for entity in _entities(self.scene_context, "objects"):
            if entity["alias"] == object_alias:
                object_name = entity["natural_name"]
                break
        user_prompt = "Analyze holding evidence; do not decide success.\n" + json.dumps(
            {
                "atomic_task": call.atomic_task,
                "policy_prompt": call.policy_prompt,
                "object_alias": object_alias,
                "object_name": object_name,
                "source_region": source_region,
                "required_positive_logic": {
                    "all_of": [
                        "eye_in_hand: object_contact_with_gripper",
                        "one_side_view: object_between_gripper_fingers",
                        "side_view_summary: gripper_closed",
                        "external_view: object_left_source_region",
                    ]
                },
                "negative_override": (
                    "a confident external view showing the object still in source"
                ),
            },
            ensure_ascii=False,
        )
        raw = self.client.query_json(
            system_prompt=HOLDING_EVIDENCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            images=images,
        )
        return (
            self._validate_holding_evidence(raw, camera_names=camera_names),
            user_prompt,
        )

    def _resolve_holding_evidence(
        self,
        parsed: Mapping[str, Any],
        *,
        expected: Mapping[str, Any],
        source_region: str | None,
    ) -> tuple[dict[str, Any], str | None, list[dict[str, Any]]]:
        rejected: list[dict[str, Any]] = []
        negative: list[dict[str, Any]] = []
        view_evidence = list(parsed.get("view_evidence", []))
        side_views = [
            item
            for item in view_evidence
            if item["camera_name"] in HOLDING_SIDE_VIEWS
            and item["confidence"] >= self.success_threshold
        ]
        external_views = [
            item
            for item in view_evidence
            if item["camera_name"] != "robot0_eye_in_hand"
            and item["confidence"] >= self.success_threshold
        ]

        source = _normalise_region(source_region or parsed.get("source_region"))
        for item in external_views:
            at_source = bool(
                source
                and item["object_visible"]
                and _normalise_region(item.get("object_location")) == source
            )
            if item["object_left_source_region"] is False or at_source:
                negative.append(
                    {
                        "type": "external_source_region_negative",
                        "camera_name": item["camera_name"],
                        "object_location": item.get("object_location"),
                        "object_left_source_region": item[
                            "object_left_source_region"
                        ],
                        "confidence": item["confidence"],
                        "reason": item.get("reason", ""),
                    }
                )

        side_between_values = [
            item["object_between_gripper_fingers"]
            for item in side_views
            if item["object_between_gripper_fingers"] is not None
        ]
        side_between = any(value is True for value in side_between_values)
        side_between_false = bool(side_between_values) and all(
            value is False for value in side_between_values
        )
        external_left_source = any(
            item["object_left_source_region"] is True for item in external_views
        )
        eye_contact = parsed["object_contact_with_gripper"]
        gripper_closed = parsed["gripper_closed"]
        positive_holding = bool(
            eye_contact is True
            and side_between
            and gripper_closed is True
            and external_left_source
        )

        missing = missing_required_views("holding", parsed["supporting_views"])
        failure_code: str | None = None
        if negative:
            value: bool | None = False
            rejected.extend(negative)
        elif positive_holding:
            if missing:
                value = None
                failure_code = "MISSING_REQUIRED_EVIDENCE"
            else:
                value = True
        elif (
            side_between_false
            or eye_contact is False
            or gripper_closed is False
        ):
            value = False
        elif missing:
            value = None
            failure_code = "MISSING_REQUIRED_EVIDENCE"
        else:
            value = None

        if failure_code:
            rejected.append(
                {
                    "type": "missing_required_views",
                    "predicate": "holding",
                    "missing_views": missing,
                    "supporting_views": parsed["supporting_views"],
                }
            )

        if parsed["holding_evidence"] is True and value is not True:
            rejected.append(
                {
                    "type": "rejected_model_holding_claim",
                    "model_claim": True,
                    "program_decision": value,
                    "reason": "role-separated holding evidence requirements not met",
                }
            )
        reason = (
            "An external view shows the object remains at its source."
            if negative
            else "Required holding evidence views are missing."
            if failure_code
            else "Role-separated holding evidence was evaluated."
        )
        condition = {
            "predicate": expected["predicate"],
            "subject": expected["subject"],
            "object": expected.get("object"),
            "desired_value": expected.get("desired_value", True),
            "value": value,
            "confidence": parsed["confidence"],
            "supporting_views": parsed["supporting_views"],
            "reason": reason,
            "role_evidence": {
                "eye_in_hand_contact": eye_contact,
                "side_view_between_fingers": side_between,
                "side_view_gripper_closed": gripper_closed,
                "external_view_left_source": external_left_source,
                "side_views_used": [
                    item["camera_name"] for item in side_views
                ],
                "external_views_used": [
                    item["camera_name"] for item in external_views
                ],
            },
            "extracted_evidence": dict(parsed),
        }
        return condition, failure_code, rejected

    def _validate_navigation(
        self, value: Mapping[str, Any], camera_names: Sequence[str]
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("navigation verifier response must be an object")
        result = dict(value)
        for field in (
            "target_visible",
            "target_reached",
            "suitable_for_local_refinement",
        ):
            if not isinstance(result.get(field), bool):
                raise ValueError(f"{field} must be boolean")
        holding = result.get("holding_required_object")
        if holding is not None and not isinstance(holding, bool):
            raise ValueError("holding_required_object must be boolean or null")
        result["confidence"] = _confidence(result.get("confidence"), field="confidence")
        result["supporting_views"] = self._validate_views(
            result.get("supporting_views", []), camera_names
        )
        reason = result.get("reason", "")
        if not isinstance(reason, str):
            raise ValueError("navigation reason must be a string")
        result["reason"] = reason
        return result

    @staticmethod
    def _validate_critic_response(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("VLM critic response must be a JSON object")
        success = value.get("success")
        if not isinstance(success, bool):
            raise ValueError("VLM critic success must be boolean")
        confidence = _confidence(value.get("confidence"), field="confidence")
        reason = value.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("VLM critic reason must be a non-empty string")
        failure_reason = value.get("failure_reason", "")
        if failure_reason is None:
            failure_reason = ""
        if not isinstance(failure_reason, str):
            raise ValueError("VLM critic failure_reason must be a string")
        if not success and not failure_reason.strip():
            failure_reason = reason
        return {
            "success": success,
            "confidence": confidence,
            "reason": reason.strip(),
            "failure_reason": failure_reason.strip(),
        }

    @staticmethod
    def _critic_request_id(
        request_count: int,
        atomic_task: str,
        *,
        verification_phase: str,
        step_index: int,
    ) -> str:
        task_name = re.sub(r"(?<!^)(?=[A-Z])", "_", atomic_task).lower()
        task_name = re.sub(r"[^a-z0-9_]+", "_", task_name).strip("_")
        suffix = (
            "final"
            if verification_phase == "skill_end"
            else f"step_{step_index}"
        )
        return f"{request_count:06d}_{task_name or 'atomic_task'}_{suffix}"

    def _run_critic(
        self,
        *,
        env: Any,
        atomic_task_call: AtomicTaskCall | Mapping[str, Any],
        step_index: int,
        verification_phase: str,
    ) -> dict[str, Any]:
        self._request_count += 1
        bundle = None
        parsed: dict[str, Any] | None = None
        user_prompt = ""
        call = (
            atomic_task_call
            if isinstance(atomic_task_call, AtomicTaskCall)
            else AtomicTaskCall.from_mapping(atomic_task_call)
        )
        request_id = self._critic_request_id(
            self._request_count,
            call.atomic_task,
            verification_phase=verification_phase,
            step_index=step_index,
        )
        final_checkpoint = verification_phase in {"skill_end", "task_end"}
        try:
            selected_views = None
            if self.view_scheduler is not None:
                selected_views = self.view_scheduler.select_views(
                    atomic_task_call=call,
                    purpose="critic",
                    available_views=self.observation_provider.camera_names,
                )
            bundle = self.observation_provider.observe(
                env=env,
                camera_names=selected_views,
                purpose=(
                    f"critic:{call.subgoal_id}:{verification_phase}:"
                    f"step_{step_index}"
                ),
            )
            user_prompt = (
                "Atomic task:\n"
                f"{call.policy_prompt}\n\n"
                "Required goal:\n"
                f"{json.dumps(call.termination_condition, ensure_ascii=False)}\n\n"
                "The required goal is context for the task-level decision; do not "
                "score its predicates separately.\n\n"
                f"Observation views at step {step_index}: "
                f"{list(bundle.camera_names)}\n\n"
                "Question:\n"
                "Did the robot successfully complete this atomic task?"
            )
            raw = self.client.query_json(
                system_prompt=CRITIC_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                images=bundle.images,
            )
            parsed = self._validate_critic_response(raw)
            success = parsed["success"]
            status = (
                "success"
                if success
                else "failed"
                if final_checkpoint
                else "uncertain"
            )
            runtime_decision = {
                "status": status,
                "goal_satisfied": success,
                "failure_code": (
                    None
                    if success
                    else "VLM_CRITIC_FAILURE"
                    if final_checkpoint
                    else "VLM_CRITIC_NOT_COMPLETE"
                ),
                "retryable": not success,
                "confidence": parsed["confidence"],
                "reason": parsed["reason"],
                "failure_reason": parsed["failure_reason"],
                "critic_result": parsed,
                "verification_phase": verification_phase,
                "verification_step": step_index,
                "state_evidence": [
                    {
                        "source": "vlm_critic",
                        "step_index": step_index,
                        "camera_names": list(bundle.camera_names),
                        "confidence": parsed["confidence"],
                        "reason": parsed["reason"],
                        "failure_reason": parsed["failure_reason"],
                    }
                ],
            }
            artifact_dir, image_paths = save_vlm_artifacts(
                root=self.log_dir,
                category="vlm_verification",
                request_id=request_id,
                bundle=bundle,
                system_prompt=CRITIC_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                client=self.client,
                parsed_json=parsed,
                runtime_decision=runtime_decision,
            )
            runtime_decision["artifact_dir"] = artifact_dir
            runtime_decision["image_paths"] = image_paths
            return runtime_decision
        except (
            KeyError,
            OSError,
            RuntimeError,
            TimeoutError,
            TypeError,
            ValueError,
        ) as exc:
            status = "failed" if final_checkpoint else "uncertain"
            runtime_decision = {
                "status": status,
                "goal_satisfied": False,
                "failure_code": "VLM_CRITIC_ERROR",
                "retryable": True,
                "confidence": 0.0,
                "reason": "The task critic could not produce a valid decision.",
                "failure_reason": str(exc),
                "critic_result": parsed,
                "verification_phase": verification_phase,
                "verification_step": step_index,
                "state_evidence": [],
            }
            if bundle is not None:
                artifact_dir, image_paths = save_vlm_artifacts(
                    root=self.log_dir,
                    category="vlm_verification",
                    request_id=request_id,
                    bundle=bundle,
                    system_prompt=CRITIC_SYSTEM_PROMPT,
                    user_prompt=user_prompt or "critic request construction failed",
                    client=self.client,
                    parsed_json=(
                        parsed
                        or getattr(self.client, "last_parsed_response", None)
                    ),
                    runtime_decision=runtime_decision,
                )
                runtime_decision["artifact_dir"] = artifact_dir
                runtime_decision["image_paths"] = image_paths
            return runtime_decision

    def __call__(
        self,
        *,
        env: Any,
        atomic_task_call: AtomicTaskCall | Mapping[str, Any],
        observation: Mapping[str, Any],
        step_index: int,
        info: Mapping[str, Any],
    ) -> dict[str, Any]:
        del observation
        verification_mode = str(info.get("verification_mode", "periodic"))
        if verification_mode not in {
            "periodic",
            "periodic_early_stop",
            "skill_end",
        }:
            verification_mode = "periodic"
        default_phase = (
            "skill_end" if verification_mode == "skill_end" else "periodic"
        )
        verification_phase = str(
            info.get("verification_phase", default_phase)
        )
        if verification_mode == "skill_end" or self.critic_only:
            return self._run_critic(
                env=env,
                atomic_task_call=atomic_task_call,
                step_index=step_index,
                verification_phase=verification_phase,
            )
        self._request_count += 1
        bundle = None
        system_prompt = VERIFIER_SYSTEM_PROMPT
        user_prompt = ""
        parsed: dict[str, Any] | None = None
        decision_failure_code: str | None = None
        rejected_evidence: list[dict[str, Any]] = []
        evidence_payload: dict[str, Any] = {}
        try:
            if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
                raise ValueError("step_index must be a non-negative integer")
            call = (
                atomic_task_call
                if isinstance(atomic_task_call, AtomicTaskCall)
                else AtomicTaskCall.from_mapping(atomic_task_call)
            )
            condition = _required_conditions(call)
            leaves: list[dict[str, Any]] = []
            tree = _condition_tree(condition, leaves)
            configured_required = self._required_consecutive_successes(call)
            required = configured_required
            if call.atomic_task == "NavigateKitchen":
                view_selection_purpose = "navigation_verification"
            elif call.atomic_task == "PickObject" or any(
                item["predicate"].strip().lower() == "holding"
                and item.get("desired_value") is True
                for item in leaves
            ):
                view_selection_purpose = "holding_verification"
            elif call.atomic_task == "PlaceObject":
                view_selection_purpose = "placement_verification"
            else:
                # Task-specific generic `verification` entries in the YAML are
                # used as the fallback for non-holding operation predicates.
                view_selection_purpose = "placement_verification"
            selected_views = None
            if self.view_scheduler is not None:
                selected_views = self.view_scheduler.select_views(
                    atomic_task_call=call,
                    purpose=view_selection_purpose,
                    available_views=self.observation_provider.camera_names,
                )
            bundle = self.observation_provider.observe(
                env=env,
                camera_names=selected_views,
                purpose=(
                    f"verification:{call.subgoal_id}:"
                    f"{verification_phase}:step_{step_index}"
                ),
            )
            query_bundle = bundle
            objects = _entities(self.scene_context, "objects")
            fixtures = _entities(self.scene_context, "fixtures")

            if call.atomic_task == "NavigateKitchen":
                system_prompt = NAVIGATION_SYSTEM_PROMPT
                holding_alias = next(
                    (
                        item["subject"]
                        for item in leaves
                        if item["predicate"].strip().lower() == "holding"
                    ),
                    call.arguments.get("held_object_id"),
                )
                user_prompt = "Judge coarse navigation completion.\n" + json.dumps(
                    {
                        "atomic_task": call.atomic_task,
                        "policy_prompt": call.policy_prompt,
                        "target_fixture_alias": call.arguments.get("fixture_id"),
                        "target_fixture_name": call.arguments.get("fixture_name"),
                        "holding_required_object_alias": holding_alias,
                        "candidate_objects": objects,
                        "candidate_fixtures": fixtures,
                    },
                    ensure_ascii=False,
                )
                raw = self.client.query_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    images=query_bundle.images,
                )
                parsed = self._validate_navigation(raw, bundle.camera_names)
                nav_value = bool(
                    parsed["target_visible"]
                    and parsed["target_reached"]
                    and parsed["suitable_for_local_refinement"]
                )
                holding_value = parsed["holding_required_object"]
                if holding_alias is None:
                    holding_value = True
                elif holding_value is True:
                    missing = missing_required_views(
                        "holding", parsed["supporting_views"]
                    )
                    if missing:
                        holding_value = None
                        decision_failure_code = "MISSING_REQUIRED_EVIDENCE"
                        rejected_evidence.append(
                            {
                                "type": "missing_required_views",
                                "predicate": "holding",
                                "missing_views": missing,
                                "supporting_views": parsed["supporting_views"],
                            }
                        )
                condition_results = [
                    {
                        "predicate": "navigation_coarse_visual_completion",
                        "subject": call.arguments.get("fixture_id"),
                        "object": None,
                        "value": nav_value,
                        "confidence": parsed["confidence"],
                        "supporting_views": parsed["supporting_views"],
                        "reason": parsed["reason"],
                    }
                ]
                if holding_alias is not None:
                    condition_results.append(
                        {
                            "predicate": "holding",
                            "subject": holding_alias,
                            "object": None,
                            "value": holding_value,
                            "confidence": parsed["confidence"],
                            "supporting_views": parsed["supporting_views"],
                            "reason": parsed["reason"],
                        }
                    )
                values = [item["value"] for item in condition_results]
                instantaneous = all(value is True for value in values)
                unknown = any(value is None for value in values)
                model_success = instantaneous
                overall_confidence = parsed["confidence"]
                evidence_payload = {
                    "type": "navigation_visual_evidence",
                    "extracted": parsed,
                }
            else:
                holding_leaves = [
                    item
                    for item in leaves
                    if item["predicate"].strip().lower() == "holding"
                ]
                if holding_leaves:
                    if len(leaves) != 1:
                        raise ValueError(
                            "holding predicates must be verified as a standalone "
                            "evidence condition"
                        )
                    system_prompt = HOLDING_EVIDENCE_SYSTEM_PROMPT
                    source_region = (
                        call.arguments.get("source_id")
                        or call.arguments.get("source_name")
                    )
                    parsed, user_prompt = self.verify_holding(
                        call=call,
                        object_alias=holding_leaves[0]["subject"],
                        source_region=(
                            str(source_region) if source_region is not None else None
                        ),
                        images=query_bundle.images,
                        camera_names=bundle.camera_names,
                    )
                    condition_result, decision_failure_code, rejected = (
                        self._resolve_holding_evidence(
                            parsed,
                            expected=holding_leaves[0],
                            source_region=(
                                str(source_region)
                                if source_region is not None
                                else None
                            ),
                        )
                    )
                    rejected_evidence.extend(rejected)
                    condition_results = [condition_result]
                    values = [condition_result["value"]]
                    tree_value = _evaluate_tree(tree, values)
                    instantaneous = tree_value is True
                    unknown = tree_value is None
                    model_success = tree_value is True
                    overall_confidence = parsed["confidence"]
                    evidence_payload = {
                        "type": "holding_evidence",
                        "extracted": parsed,
                        "program_condition": condition_result,
                    }
                else:
                    user_prompt = (
                        "Judge each required condition in order.\n"
                        + json.dumps(
                            {
                                "atomic_task": call.atomic_task,
                                "policy_prompt": call.policy_prompt,
                                "arguments": call.arguments,
                                "required_conditions": leaves,
                                "candidate_objects": objects,
                                "candidate_fixtures": fixtures,
                            },
                            ensure_ascii=False,
                        )
                    )
                    raw = self.client.query_json(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        images=query_bundle.images,
                    )
                    parsed = self._validate_generic(
                        raw, leaves=leaves, camera_names=bundle.camera_names
                    )
                    condition_results = parsed["conditions"]
                    model_values = [item["value"] for item in condition_results]
                    model_tree_value = _evaluate_tree(tree, model_values)
                    model_status = (
                        "SUCCESS"
                        if model_tree_value is True
                        else "NOT_YET"
                        if model_tree_value is False
                        else "UNKNOWN"
                    )
                    if parsed["overall_status"] != model_status:
                        raise ValueError(
                            "overall_status conflicts with per-condition values: "
                            f"reported={parsed['overall_status']}, "
                            f"derived={model_status}"
                        )
                    for item in condition_results:
                        if item["value"] is not True:
                            continue
                        missing = missing_required_views(
                            item["predicate"], item["supporting_views"]
                        )
                        if not item["supporting_views"]:
                            missing = ["at_least_one_supporting_view"]
                        if missing:
                            item["value"] = None
                            decision_failure_code = "MISSING_REQUIRED_EVIDENCE"
                            rejected_evidence.append(
                                {
                                    "type": "missing_required_views",
                                    "predicate": item["predicate"],
                                    "subject": item["subject"],
                                    "missing_views": missing,
                                    "supporting_views": item["supporting_views"],
                                }
                            )
                    values = [item["value"] for item in condition_results]
                    tree_value = _evaluate_tree(tree, values)
                    instantaneous = tree_value is True
                    unknown = tree_value is None
                    model_success = tree_value is True
                    overall_confidence = parsed["confidence"]
                    evidence_payload = {
                        "type": "generic_predicate_evidence",
                        "extracted": parsed,
                        "program_conditions": condition_results,
                    }

            confident = bool(
                overall_confidence >= self.success_threshold
                and all(
                    item["confidence"] >= self.success_threshold
                    for item in condition_results
                )
            )
            instantaneous_success = bool(instantaneous and model_success and confident)
            condition_key = json.dumps(
                {"call": call.subgoal_id, "conditions": leaves},
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            consecutive = self._record_consecutive(
                env=env,
                call=call,
                condition_key=condition_key,
                step_index=step_index,
                satisfied=instantaneous_success,
            )
            success = instantaneous_success and consecutive >= required
            evidence = [
                {
                    "step_index": step_index,
                    "verification_phase": verification_phase,
                    "verification_mode": verification_mode,
                    "verification_step": step_index,
                },
                {
                    "source": "multi_view_vlm_verifier",
                    "camera_names": list(query_bundle.camera_names),
                    "selected_camera_names": list(bundle.camera_names),
                    "purpose": bundle.purpose,
                    "verification_mode": verification_mode,
                    "verification_step": step_index,
                    "view_selection_purpose": view_selection_purpose,
                    "timestamp": bundle.timestamp,
                    "observation_step": bundle.step_index,
                    "conditions": condition_results,
                    "confidence": overall_confidence,
                    "supporting_views": sorted(
                        {
                            view
                            for item in condition_results
                            for view in item["supporting_views"]
                        }
                    ),
                    "raw_response": getattr(self.client, "last_response_text", None),
                    "raw_http_response": getattr(
                        self.client, "last_raw_http_response", None
                    ),
                    "parsed_json": parsed,
                    "success_threshold": self.success_threshold,
                    "evidence": evidence_payload,
                    "rejected_evidence": rejected_evidence,
                },
                {
                    "source": "consecutive_stability",
                    "instantaneous_satisfied": instantaneous_success,
                    "consecutive_successes": consecutive,
                    "required_consecutive_successes": required,
                    "configured_required_consecutive_successes": configured_required,
                    "stable": success,
                },
            ]
            runtime_decision = {
                "status": "success" if success else "uncertain",
                "goal_satisfied": success,
                "failure_code": None if success else decision_failure_code,
                "retryable": not success,
                "state_evidence": evidence,
            }
            request_id = (
                f"{self._request_count:06d}_{call.subgoal_id}_final_step_{step_index}"
                if verification_mode == "skill_end"
                else f"{self._request_count:06d}_{call.subgoal_id}_step_{step_index}"
            )
            artifact_dir, image_paths = save_vlm_artifacts(
                root=self.log_dir,
                category="vlm_verification",
                request_id=request_id,
                bundle=query_bundle,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                client=self.client,
                parsed_json=parsed,
                runtime_decision=runtime_decision,
                evidence=evidence_payload,
                rejected_evidence=rejected_evidence,
                final_decision=runtime_decision,
            )
            evidence[1]["artifact_dir"] = artifact_dir
            evidence[1]["image_paths"] = image_paths
            if success:
                runtime_decision["retryable"] = False
            elif unknown:
                evidence[1]["visual_outcome"] = "UNKNOWN"
            else:
                evidence[1]["visual_outcome"] = "NOT_YET"
            return runtime_decision
        except (
            KeyError,
            OSError,
            RuntimeError,
            TimeoutError,
            TypeError,
            ValueError,
        ) as exc:
            evidence: list[dict[str, Any]] = [
                {
                    "step_index": step_index,
                    "verification_phase": verification_phase,
                    "verification_mode": verification_mode,
                    "verification_step": step_index,
                },
                {
                    "source": "multi_view_vlm_verifier_error",
                    "error": str(exc),
                    "camera_names": list(bundle.camera_names) if bundle else [],
                    "raw_response": getattr(self.client, "last_response_text", None),
                    "raw_http_response": getattr(
                        self.client, "last_raw_http_response", None
                    ),
                    "parsed_json": parsed,
                },
            ]
            invalid_output = isinstance(exc, (KeyError, TypeError, ValueError))
            runtime_decision = {
                "status": "uncertain" if invalid_output else "failed",
                "goal_satisfied": False,
                "failure_code": (
                    "VLM_VERIFIER_OUTPUT_INVALID"
                    if invalid_output
                    else "VLM_VERIFIER_ERROR"
                ),
                "retryable": invalid_output,
                "state_evidence": evidence,
            }
            if bundle is not None:
                artifact_dir, image_paths = save_vlm_artifacts(
                    root=self.log_dir,
                    category="vlm_verification",
                    request_id=(
                        f"{self._request_count:06d}_error_final_step_{step_index}"
                        if verification_mode == "skill_end"
                        else f"{self._request_count:06d}_error_step_{step_index}"
                    ),
                    bundle=query_bundle,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt or "request construction failed",
                    client=self.client,
                    parsed_json=getattr(self.client, "last_parsed_response", None),
                    runtime_decision=runtime_decision,
                )
                evidence[1]["artifact_dir"] = artifact_dir
                evidence[1]["image_paths"] = image_paths
            return runtime_decision
