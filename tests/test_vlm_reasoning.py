from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from robocasa.atomic_task_schemas import AtomicTaskCall
from robocasa.eval_vlm_pi05 import parse_args
from robocasa.local_work_pose_refiner import OpenAICompatibleLocalPoseVLM
from robocasa.navigation.navigation_pipeline import load_navigation_config
from robocasa.multi_view_perception import (
    DEFAULT_REASONING_CAMERA_NAMES,
    MultiViewObservationProvider,
)
from robocasa.oracle_logging import OracleLoggingVerifier
from robocasa.pi05_env import POLICY_CAMERA_NAMES
from robocasa.view_scheduler import FixedViewScheduler
from robocasa.vlm_atomic_task_verifier import VLMAtomicTaskVerifier
from robocasa.vlm_grounding import VLMRoboCasaGrounder
from robocasa.vlm_client import extract_assistant_response


SCENE_CONTEXT = {
    "objects": [
        {"alias": "vegetable", "natural_name": "potato"},
        {"alias": "bowl", "natural_name": "bowl"},
    ],
    "fixtures": [
        {"alias": "sink_main_group", "natural_name": "sink"},
        {"alias": "microwave_main_group", "natural_name": "microwave"},
    ],
}


class FakeVisionClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.last_response_text = None
        self.last_raw_http_response = None
        self.last_parsed_response = None
        self.last_http_status = 200
        self.last_finish_reason = "stop"
        self.last_usage = {"completion_tokens": 10}

    def query_json(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        self.last_parsed_response = dict(response)
        self.last_response_text = json.dumps(response)
        self.last_raw_http_response = json.dumps(
            {"choices": [{"message": {"content": self.last_response_text}}]}
        )
        return dict(response)


class ImageOnlyEnv:
    render_count = 0

    def render_camera_views(self, camera_names, *, height, width):
        self.render_count += 1
        return {
            name: np.full((height, width, 3), index, dtype=np.uint8)
            for index, name in enumerate(camera_names, start=1)
        }

    @property
    def sim(self):
        raise AssertionError("VLM decision path must not access simulator state")

    @property
    def fixtures(self):
        raise AssertionError("VLM decision path must not access fixture state")

    @property
    def objects(self):
        raise AssertionError("VLM decision path must not access object state")


def provider():
    return MultiViewObservationProvider(image_size=8)


def pick_call(*, consecutive=1):
    return AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "pick_vegetable",
            "atomic_task": "PickObject",
            "policy_prompt": "Pick up the potato and keep holding it.",
            "arguments": {"object_id": "vegetable"},
            "termination_condition": {
                "predicate": "holding",
                "subject": "vegetable",
                "desired_value": True,
            },
            "metadata": {
                "skill_contract": {
                    "verification": {
                        "required_consecutive_successes": consecutive
                    }
                }
            },
        }
    )


def grounding_response(**overrides):
    value = {
        "target_visible": True,
        "target_entity_alias": "vegetable",
        "target_entity_kind": "object",
        "entity_role": "object",
        "target_fixture_alias": "sink_main_group",
        "operation_pose_ready": True,
        "navigation_required": False,
        "confidence": 0.9,
        "reason": "The potato and sink workstation are clearly visible and nearby.",
        "supporting_views": ["robot0_frontview"],
    }
    value.update(overrides)
    return value


def condition_result(value, confidence=0.9, *, predicate="holding", subject="vegetable", object_id=None):
    return {
        "predicate": predicate,
        "subject": subject,
        "object": object_id,
        "value": value,
        "confidence": confidence,
        "supporting_views": ["robot0_eye_in_hand"],
        "reason": "Visible evidence for this condition.",
    }


def verifier_response(value, *, overall=None, confidence=0.9):
    del overall
    positive_geometry = value is not False
    supporting_views = [
        "robot0_eye_in_hand",
        "robot0_agentview_left",
        "robot0_frontview",
    ]
    if value is None:
        supporting_views = ["robot0_eye_in_hand"]

    view_evidence = []
    for camera in supporting_views:
        is_eye_in_hand = camera == "robot0_eye_in_hand"
        is_side_view = camera in {
            "robot0_agentview_left",
            "robot0_agentview_right",
        }
        view_evidence.append(
            {
                "camera_name": camera,
                "object_visible": True,
                "object_location": (
                    "attached to gripper" if positive_geometry else "sink"
                ),
                "object_between_gripper_fingers": (
                    positive_geometry if is_side_view else None
                ),
                "object_left_source_region": (
                    positive_geometry if not is_eye_in_hand else None
                ),
                "confidence": confidence,
                "reason": "Direct role-specific evidence from this camera.",
            }
        )

    return {
        "object_visible": True,
        "object_location": "attached to gripper" if positive_geometry else "sink",
        "object_between_gripper_fingers": positive_geometry,
        "gripper_closed": positive_geometry,
        "object_contact_with_gripper": positive_geometry,
        "object_left_source_region": positive_geometry,
        "source_region": "sink",
        "holding_evidence": positive_geometry,
        "contradicting_evidence": [] if positive_geometry else ["object remains in sink"],
        "contradicting_views": [] if positive_geometry else supporting_views,
        "supporting_views": supporting_views,
        "view_evidence": view_evidence,
        "confidence": confidence,
        "reason": "Fresh views were checked conservatively.",
    }


def place_in_bowl_call():
    return AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "place_vegetable_in_bowl",
            "atomic_task": "PlaceObject",
            "policy_prompt": "Put the potato in the bowl.",
            "arguments": {
                "object_id": "vegetable",
                "destination_id": "bowl",
            },
            "termination_condition": {
                "predicate": "inside",
                "subject": "vegetable",
                "object": "bowl",
                "desired_value": True,
            },
        }
    )


def test_vlm_grounder_rejects_alias_outside_candidate_set():
    invalid = grounding_response(target_entity_alias="hallucinated_potato")
    grounder = VLMRoboCasaGrounder(
        client=FakeVisionClient(invalid, invalid),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    )
    result = grounder.ground(env=ImageOnlyEnv(), atomic_task_call=pick_call())
    assert result.grounded is False
    assert result.status == "unresolved"
    assert "not a candidate" in result.evidence[0]["error"]


def test_vlm_grounder_normalizes_natural_kind_from_exact_alias():
    result = VLMRoboCasaGrounder(
        client=FakeVisionClient(
            grounding_response(target_entity_kind="broccoli")
        ),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    ).ground(env=ImageOnlyEnv(), atomic_task_call=pick_call())

    assert result.grounded is True
    assert result.target_entity_kind == "object"
    parsed = result.evidence[0]["parsed_json"]
    assert parsed["validation_normalizations"] == [
        {
            "field": "target_entity_kind",
            "from": "broccoli",
            "value": "object",
            "reason": "derived_from_exact_candidate_alias_membership",
        }
    ]


def test_vlm_grounder_retries_once_after_schema_validation_error():
    client = FakeVisionClient(
        grounding_response(target_entity_alias="hallucinated_potato"),
        grounding_response(),
    )
    result = VLMRoboCasaGrounder(
        client=client,
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    ).ground(env=ImageOnlyEnv(), atomic_task_call=pick_call())

    assert result.grounded is True
    assert len(client.calls) == 2
    assert "CORRECTION REQUIRED" in client.calls[1]["user_prompt"]
    assert len(result.evidence[0]["validation_attempts"]) == 1


def test_vlm_grounder_low_confidence_is_not_grounded():
    result = VLMRoboCasaGrounder(
        client=FakeVisionClient(grounding_response(confidence=0.2)),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    ).ground(env=ImageOnlyEnv(), atomic_task_call=pick_call())
    assert result.grounded is False
    assert result.status == "unresolved"


def test_vlm_grounder_invisible_target_is_not_grounded():
    result = VLMRoboCasaGrounder(
        client=FakeVisionClient(
            grounding_response(
                target_visible=False,
                operation_pose_ready=False,
                navigation_required=True,
            )
        ),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    ).ground(env=ImageOnlyEnv(), atomic_task_call=pick_call())
    assert result.grounded is False
    assert result.status == "unresolved"


def test_vlm_grounder_supports_object_receptacle_destination():
    response = grounding_response(
        target_entity_alias="bowl",
        target_entity_kind="object",
        entity_role="object_receptacle",
        target_fixture_alias="sink_main_group",
        supporting_views=["robot0_agentview_right", "robot0_frontview"],
    )
    result = VLMRoboCasaGrounder(
        client=FakeVisionClient(response),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    ).ground(env=ImageOnlyEnv(), atomic_task_call=place_in_bowl_call())

    assert result.grounded is True
    assert result.target_entity_alias == "bowl"
    assert result.target_entity_kind == "object"
    assert result.entity_role == "object_receptacle"
    assert result.destination_entity_alias == "bowl"
    assert result.destination_role == "object_receptacle"
    assert result.source_object_alias == "vegetable"
    assert result.held_object_alias is None
    assert result.holding_source_object is None
    assert "holding_reference_object" not in result.evidence[0]["parsed_json"]


def test_vlm_grounder_ignores_legacy_holding_field_for_place_navigation():
    response = grounding_response(
        target_entity_alias="bowl",
        target_entity_kind="object",
        entity_role="object_receptacle",
        target_fixture_alias="sink_main_group",
        operation_pose_ready=False,
        navigation_required=True,
        supporting_views=["robot0_agentview_right", "robot0_frontview"],
        holding_reference_object=False,
    )
    client = FakeVisionClient(response)
    result = VLMRoboCasaGrounder(
        client=client,
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    ).ground(env=ImageOnlyEnv(), atomic_task_call=place_in_bowl_call())

    assert result.grounded is False
    assert result.status == "navigation_required"
    assert result.target_entity_alias == "bowl"
    assert result.target_fixture_alias == "sink_main_group"
    assert result.holding_source_object is None
    assert "holding_reference_object" not in result.evidence[0]["parsed_json"]
    assert "Do not report holding" in client.calls[0]["user_prompt"]


def test_vlm_verifier_false_all_of_predicate_cannot_succeed():
    call = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "place_and_close",
            "atomic_task": "PlaceObject",
            "policy_prompt": "Place the potato and close the microwave.",
            "arguments": {},
            "termination_condition": [
                {"predicate": "inside", "subject": "vegetable", "object": "microwave_main_group", "desired_value": True},
                {"predicate": "closed", "subject": "microwave_main_group", "desired_value": True},
            ],
        }
    )
    response = {
        "conditions": [
            condition_result(True, predicate="inside", object_id="microwave_main_group"),
            condition_result(False, predicate="closed", subject="microwave_main_group"),
        ],
        "overall_status": "NOT_YET",
        "confidence": 0.9,
        "reason": "The door is still open.",
    }
    result = VLMAtomicTaskVerifier(
        client=FakeVisionClient(response),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    )(
        env=ImageOnlyEnv(), atomic_task_call=call, observation={}, step_index=20, info={}
    )
    assert result["status"] == "uncertain"
    assert result["goal_satisfied"] is False


def test_non_holding_predicate_accepts_one_clear_supporting_view():
    response = {
        "conditions": [
            condition_result(
                True,
                predicate="inside",
                object_id="bowl",
            )
        ],
        "overall_status": "SUCCESS",
        "confidence": 0.9,
        "reason": "One clear view shows the potato inside the bowl.",
    }
    result = VLMAtomicTaskVerifier(
        client=FakeVisionClient(response),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    )(
        env=ImageOnlyEnv(),
        atomic_task_call=place_in_bowl_call(),
        observation={},
        step_index=20,
        info={},
    )

    assert result["status"] == "success"
    assert result["goal_satisfied"] is True


def test_vlm_verifier_unknown_is_uncertain():
    result = VLMAtomicTaskVerifier(
        client=FakeVisionClient(verifier_response(None)),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    )(
        env=ImageOnlyEnv(), atomic_task_call=pick_call(), observation={}, step_index=20, info={}
    )
    assert result["status"] == "uncertain"
    assert result["goal_satisfied"] is False
    assert result["state_evidence"][1]["visual_outcome"] == "UNKNOWN"
    assert result["failure_code"] == "MISSING_REQUIRED_EVIDENCE"


def test_holding_model_claim_cannot_bypass_program_evidence_rule():
    response = verifier_response(True)
    response["gripper_closed"] = False
    result = VLMAtomicTaskVerifier(
        client=FakeVisionClient(response),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    )(
        env=ImageOnlyEnv(),
        atomic_task_call=pick_call(),
        observation={},
        step_index=20,
        info={},
    )

    assert result["goal_satisfied"] is False
    rejected = result["state_evidence"][1]["rejected_evidence"]
    assert any(item["type"] == "rejected_model_holding_claim" for item in rejected)


def test_eye_in_hand_source_region_claim_is_ignored_when_external_views_are_positive():
    response = verifier_response(True)
    response["view_evidence"][0].update(
        {
            "object_location": "sink",
            "object_between_gripper_fingers": False,
            "object_left_source_region": False,
            "reason": "Eye-in-hand incorrectly claims the object remains in the sink.",
        }
    )
    response["contradicting_views"] = ["robot0_eye_in_hand"]
    response["contradicting_evidence"] = ["eye-in-hand claims object is in source"]
    result = VLMAtomicTaskVerifier(
        client=FakeVisionClient(response),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    )(
        env=ImageOnlyEnv(),
        atomic_task_call=pick_call(),
        observation={},
        step_index=20,
        info={},
    )

    assert result["goal_satisfied"] is True
    condition = result["state_evidence"][1]["conditions"][0]
    assert condition["value"] is True
    parsed = result["state_evidence"][1]["parsed_json"]
    eye_evidence = next(
        item
        for item in parsed["view_evidence"]
        if item["camera_name"] == "robot0_eye_in_hand"
    )
    assert eye_evidence["object_between_gripper_fingers"] is None
    assert eye_evidence["object_left_source_region"] is None
    normalized_fields = {
        item["field"] for item in parsed["schema_normalizations"]
    }
    assert {
        "view_evidence[robot0_eye_in_hand].object_between_gripper_fingers",
        "view_evidence[robot0_eye_in_hand].object_left_source_region",
    } <= normalized_fields


def test_holding_evidence_list_and_occluded_null_are_normalized():
    response = verifier_response(False)
    per_view_records = [dict(item) for item in response["view_evidence"]]
    per_view_records[0]["object_visible"] = False
    per_view_records[0]["object_location"] = "not_visible"
    per_view_records[0]["object_left_source_region"] = None
    response["holding_evidence"] = per_view_records
    result = VLMAtomicTaskVerifier(
        client=FakeVisionClient(response),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    )(
        env=ImageOnlyEnv(),
        atomic_task_call=pick_call(),
        observation={},
        step_index=20,
        info={"verification_phase": "periodic"},
    )

    assert result["status"] == "uncertain"
    assert result["failure_code"] is None
    assert result["state_evidence"][1]["conditions"][0]["value"] is False
    parsed = result["state_evidence"][1]["parsed_json"]
    assert parsed["holding_evidence"] is None
    assert parsed["schema_normalizations"][0]["field"] == "holding_evidence"


def test_holding_auxiliary_format_variants_do_not_discard_positive_evidence():
    # Mirrors the latest real task-end response: strong required-view positive
    # evidence, one weaker side-view contradiction, and false instead of [].
    response = verifier_response(True)
    response["contradicting_evidence"] = False
    response["confidence"] = "0.95"
    response["view_evidence"][0]["confidence"] = "0.94"
    response["view_evidence"].append(
        {
            "camera_name": "robot0_agentview_right",
            "object_visible": True,
            "object_location": "on counter",
            "object_between_gripper_fingers": False,
            "object_left_source_region": False,
            "confidence": 0.4,
            "reason": "This side view appears contradictory.",
        }
    )
    result = VLMAtomicTaskVerifier(
        client=FakeVisionClient(response),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    )(
        env=ImageOnlyEnv(),
        atomic_task_call=pick_call(),
        observation={},
        step_index=60,
        info={"verification_phase": "periodic"},
    )

    assert result["status"] == "success"
    assert result["goal_satisfied"] is True
    parsed = result["state_evidence"][1]["parsed_json"]
    assert parsed["contradicting_evidence"] == []
    assert parsed["confidence"] == 0.95
    assert {
        item["field"] for item in parsed["schema_normalizations"]
    } >= {
        "contradicting_evidence",
        "holding evidence confidence",
        "view_evidence[robot0_eye_in_hand].confidence",
    }


def test_holding_nullable_geometry_becomes_unknown_not_invalid_or_success():
    response = verifier_response(True)
    response["object_between_gripper_fingers"] = None
    for item in response["view_evidence"]:
        if item["camera_name"] in {
            "robot0_agentview_left",
            "robot0_agentview_right",
        }:
            item["object_between_gripper_fingers"] = None

    response["holding_evidence"] = True
    result = VLMAtomicTaskVerifier(
        client=FakeVisionClient(response),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    )(
        env=ImageOnlyEnv(),
        atomic_task_call=pick_call(),
        observation={},
        step_index=60,
        info={"verification_phase": "periodic"},
    )

    assert result["status"] == "uncertain"
    assert result["goal_satisfied"] is False
    assert result["failure_code"] is None
    rejected = result["state_evidence"][1]["rejected_evidence"]
    assert any(item["type"] == "rejected_model_holding_claim" for item in rejected)


def test_invalid_verifier_output_is_retryable_uncertain():
    invalid = verifier_response(False)
    invalid["confidence"] = "high"
    result = VLMAtomicTaskVerifier(
        client=FakeVisionClient(invalid),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    )(
        env=ImageOnlyEnv(),
        atomic_task_call=pick_call(),
        observation={},
        step_index=20,
        info={},
    )

    assert result["status"] == "uncertain"
    assert result["failure_code"] == "VLM_VERIFIER_OUTPUT_INVALID"
    assert result["retryable"] is True


def test_vlm_verifier_requires_two_fresh_consecutive_successes():
    env = ImageOnlyEnv()
    verifier = VLMAtomicTaskVerifier(
        client=FakeVisionClient(verifier_response(True), verifier_response(True)),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    )
    first = verifier(
        env=env, atomic_task_call=pick_call(consecutive=2), observation={}, step_index=20, info={}
    )
    second = verifier(
        env=env, atomic_task_call=pick_call(consecutive=2), observation={}, step_index=40, info={}
    )
    assert first["goal_satisfied"] is False
    assert second["goal_satisfied"] is True
    assert env.render_count == 2


def test_skill_end_critic_uses_only_final_views_and_saves_simple_artifacts(
    tmp_path: Path,
):
    env = ImageOnlyEnv()
    client = FakeVisionClient(
        {
            "success": True,
            "confidence": 0.91,
            "reason": "The broccoli is clearly secured by the gripper.",
            "failure_reason": "",
        }
    )
    verifier = VLMAtomicTaskVerifier(
        client=client,
        observation_provider=provider(),
        view_scheduler=FixedViewScheduler(),
        scene_context=SCENE_CONTEXT,
        log_dir=tmp_path,
    )

    result = verifier(
        env=env,
        atomic_task_call=pick_call(consecutive=2),
        observation={},
        step_index=600,
        info={
            "verification_mode": "skill_end",
            "verification_phase": "skill_end",
            "verification_step": 600,
        },
    )

    assert result["status"] == "success"
    assert result["goal_satisfied"] is True
    assert result["state_evidence"][0]["source"] == "vlm_critic"
    assert "conditions" not in result["state_evidence"][0]
    assert tuple(client.calls[0]["images"]) == (
        "robot0_eye_in_hand",
        "robot0_agentview_left",
        "robot0_agentview_right",
    )
    assert "required goal is context" in client.calls[0]["user_prompt"].lower()
    assert "holding" in client.calls[0]["user_prompt"]
    artifact_dirs = list((tmp_path / "vlm_verification").iterdir())
    assert len(artifact_dirs) == 1
    assert artifact_dirs[0].name == "000001_pick_object_final"
    assert len(list(artifact_dirs[0].glob("robot0_*.png"))) == 3
    assert (artifact_dirs[0] / "request.json").is_file()
    assert (artifact_dirs[0] / "raw_http_response.json").is_file()
    assert (artifact_dirs[0] / "parsed_response.json").is_file()
    assert not (artifact_dirs[0] / "evidence.json").exists()
    assert not (artifact_dirs[0] / "final_decision.json").exists()


def test_skill_end_critic_failure_is_retryable():
    result = VLMAtomicTaskVerifier(
        client=FakeVisionClient(
            {
                "success": False,
                "confidence": 0.88,
                "reason": "The object is still visibly in the sink.",
                "failure_reason": "The pick was not completed.",
            }
        ),
        observation_provider=provider(),
        view_scheduler=FixedViewScheduler(),
        scene_context=SCENE_CONTEXT,
    )(
        env=ImageOnlyEnv(),
        atomic_task_call=pick_call(consecutive=2),
        observation={},
        step_index=600,
        info={"verification_mode": "skill_end"},
    )

    assert result["status"] == "failed"
    assert result["goal_satisfied"] is False
    assert result["failure_code"] == "VLM_CRITIC_FAILURE"
    assert result["retryable"] is True
    assert result["failure_reason"] == "The pick was not completed."


def test_periodic_critic_uses_simple_completion_decisions_and_step_artifacts(
    tmp_path: Path,
):
    verifier = VLMAtomicTaskVerifier(
        client=FakeVisionClient(
            {
                "success": False,
                "confidence": 0.82,
                "reason": "The object is still in the sink.",
                "failure_reason": "The pick is not complete yet.",
            },
            {
                "success": True,
                "confidence": 0.93,
                "reason": "The object is clearly secured in the gripper.",
                "failure_reason": "",
            },
        ),
        observation_provider=provider(),
        view_scheduler=FixedViewScheduler(),
        scene_context=SCENE_CONTEXT,
        critic_only=True,
        log_dir=tmp_path,
    )

    first = verifier(
        env=ImageOnlyEnv(),
        atomic_task_call=pick_call(),
        observation={},
        step_index=60,
        info={
            "verification_mode": "periodic",
            "verification_phase": "periodic",
        },
    )
    second = verifier(
        env=ImageOnlyEnv(),
        atomic_task_call=pick_call(),
        observation={},
        step_index=120,
        info={
            "verification_mode": "periodic",
            "verification_phase": "periodic",
        },
    )

    assert first["status"] == "uncertain"
    assert first["goal_satisfied"] is False
    assert first["failure_code"] == "VLM_CRITIC_NOT_COMPLETE"
    assert second["status"] == "success"
    assert second["goal_satisfied"] is True
    artifacts = sorted(
        path.name for path in (tmp_path / "vlm_verification").iterdir()
    )
    assert artifacts == [
        "000001_pick_object_step_60",
        "000002_pick_object_step_120",
    ]
    for artifact in artifacts:
        artifact_dir = tmp_path / "vlm_verification" / artifact
        assert (artifact_dir / "request.json").is_file()
        assert (artifact_dir / "parsed_response.json").is_file()
        assert not (artifact_dir / "evidence.json").exists()


def test_navigation_uses_coarse_visual_completion_without_navigation_pose():
    call = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "navigate_microwave",
            "atomic_task": "NavigateKitchen",
            "policy_prompt": "Navigate to the microwave.",
            "arguments": {"fixture_id": "microwave_main_group", "fixture_name": "microwave"},
            "termination_condition": {"predicate": "navigation_pose", "subject": "microwave_main_group", "desired_value": True},
        }
    )
    response = {
        "target_visible": True,
        "target_reached": True,
        "suitable_for_local_refinement": True,
        "holding_required_object": None,
        "confidence": 0.9,
        "supporting_views": ["robot0_frontview"],
        "reason": "The robot is visibly beside the microwave workstation.",
    }
    result = VLMAtomicTaskVerifier(
        client=FakeVisionClient(response),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    )(
        env=ImageOnlyEnv(), atomic_task_call=call, observation={}, step_index=20, info={}
    )
    assert result["status"] == "success"
    assert result["state_evidence"][1]["conditions"][0]["predicate"] == "navigation_coarse_visual_completion"


def test_navigation_requires_visually_confirmed_held_object_when_requested():
    call = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "navigate_microwave_holding_vegetable",
            "atomic_task": "NavigateKitchen",
            "policy_prompt": "Carry the potato to the microwave.",
            "arguments": {
                "fixture_id": "microwave_main_group",
                "fixture_name": "microwave",
                "held_object_id": "vegetable",
            },
            "termination_condition": [
                {
                    "predicate": "navigation_pose",
                    "subject": "microwave_main_group",
                    "desired_value": True,
                },
                {
                    "predicate": "holding",
                    "subject": "vegetable",
                    "desired_value": True,
                },
            ],
        }
    )
    response = {
        "target_visible": True,
        "target_reached": True,
        "suitable_for_local_refinement": True,
        "holding_required_object": False,
        "confidence": 0.9,
        "supporting_views": ["robot0_frontview", "robot0_eye_in_hand"],
        "reason": "The workstation is reached, but the gripper is visibly empty.",
    }
    result = VLMAtomicTaskVerifier(
        client=FakeVisionClient(response),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    )(
        env=ImageOnlyEnv(), atomic_task_call=call, observation={}, step_index=20, info={}
    )
    assert result["status"] == "uncertain"
    assert result["goal_satisfied"] is False


def test_vlm_runtime_path_does_not_read_privileged_env_state():
    env = ImageOnlyEnv()
    grounder_result = VLMRoboCasaGrounder(
        client=FakeVisionClient(grounding_response()),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    ).ground(env=env, atomic_task_call=pick_call())
    verifier_result = VLMAtomicTaskVerifier(
        client=FakeVisionClient(verifier_response(True)),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    )(
        env=env, atomic_task_call=pick_call(), observation={}, step_index=20, info={}
    )
    assert grounder_result.grounded is True
    assert verifier_result["goal_satisfied"] is True


def test_oracle_logging_never_changes_primary_runtime_decision(tmp_path: Path):
    primary_result = {
        "status": "uncertain",
        "goal_satisfied": False,
        "failure_code": None,
        "retryable": True,
        "state_evidence": [],
    }
    wrapper = OracleLoggingVerifier(
        primary=lambda **kwargs: dict(primary_result),
        oracle=lambda **kwargs: {"status": "success", "goal_satisfied": True},
        log_dir=tmp_path,
    )
    result = wrapper(env=object())
    assert result == primary_result
    record = json.loads((tmp_path / "oracle_verification" / "000001.json").read_text())
    assert record["agreement"] is False
    assert record["vlm_prediction"] is False
    assert record["simulator_prediction"] is True
    assert record["false_positive"] is False
    assert record["false_negative"] is True
    assert record["runtime_source"] == "vlm_result"


def test_reasoning_views_do_not_change_pi05_camera_contract():
    assert POLICY_CAMERA_NAMES == (
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    )
    assert DEFAULT_REASONING_CAMERA_NAMES == (
        "robot0_topview",
        "robot0_frontview",
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    )


def test_shared_client_extracts_direct_json_content():
    content = {"overall_status": "UNKNOWN", "conditions": []}
    response = {"choices": [{"message": {"content": content}}]}
    assert json.loads(extract_assistant_response(response)) == content


def test_vlm_verifier_api_error_is_terminal_failed():
    result = VLMAtomicTaskVerifier(
        client=FakeVisionClient(RuntimeError("backend unavailable")),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
    )(
        env=ImageOnlyEnv(),
        atomic_task_call=pick_call(),
        observation={},
        step_index=20,
        info={},
    )
    assert result["status"] == "failed"
    assert result["failure_code"] == "VLM_VERIFIER_ERROR"
    assert result["retryable"] is False


def test_vlm_mode_uses_lower_frequency_default_verification(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_vlm_pi05.py",
            "--long_horizon_task",
            "pick the potato",
            "--reasoning_mode",
            "vlm_critic",
            "--plan_only",
            "--no_env",
        ],
    )
    args = parse_args()
    assert args.reasoning_mode == "vlm_critic"
    assert args.verification_mode == "periodic"
    assert args.pi05_verify_interval == 60
    assert args.pi05_min_steps_before_verify == 60


def test_simulator_mode_keeps_baseline_verification_frequency(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_vlm_pi05.py",
            "--long_horizon_task",
            "pick the potato",
            "--plan_only",
            "--no_env",
        ],
    )
    args = parse_args()
    assert args.reasoning_mode == "simulator"
    assert args.verification_mode == "skill_end"
    assert args.pi05_verify_interval == 5
    assert args.pi05_min_steps_before_verify == 10


def test_steam_in_microwave_4_mapped_navigation_profile_is_isolated(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_vlm_pi05.py",
            "--long_horizon_task",
            "pick and heat the vegetable",
            "--plan_only",
            "--no_env",
            "--system_profile",
            "steam_in_microwave_4_mapped_navigation",
            "--verification_mode",
            "skill_end",
            "--pi05_atomic_task_horizon",
            "100",
        ],
    )

    args = parse_args()
    navigation = load_navigation_config(args.navigation_config)

    assert args.reasoning_mode == "simulator"
    assert args.verification_mode == "periodic"
    assert args.pi05_verify_interval == 5
    assert args.pi05_min_steps_before_verify == 10
    assert args.pi05_atomic_task_horizon == 600
    assert args.pi05_success_handoff_steps == 5
    assert args.pi05_task_retries == 3
    assert args.retry_recovery_mode == "same_skill_no_drop_retry"
    assert args.pi05_base_action_mode == "full"
    assert args.atomic_prompt_style == "legacy_handoff"
    assert args.target_aware_navigation is True
    assert args.enable_local_refiner is True
    assert args.local_pose_enable_thinking is True
    assert args.simulator_grounding_position_threshold_m == 0.6
    assert args.simulator_grounding_orientation_cosine_threshold == 0.5
    assert args.simulator_verifier_holding_threshold == 0.035
    assert args.simulator_verifier_pick_min_lift_m == 0.04
    assert args.pi05_held_object_guard_grasp_threshold == 0.05
    assert args.pi05_held_object_drop_stops_rollout is False
    assert navigation["map"]["resolution"] == 0.025
    assert navigation["robot"]["half_length"] == 0.35
    assert navigation["robot"]["half_width"] == 0.25
    assert navigation["robot"]["radius"] == 0.4301162634
    assert navigation["robot"]["safety_margin"] == 0.01
    assert navigation["navigation"]["enabled"] is True
    assert navigation["navigation"]["align_yaw_to_path"] is True
    assert navigation["navigation"]["segment_heading_tolerance_deg"] == 10.0
    assert navigation["navigation"]["segment_heading_max_steps"] == 240
    assert navigation["navigation"]["enable_local_refiner"] is True
    assert navigation["navigation"]["local_refiner_after_navigation_only"] is True
    assert (
        navigation["navigation"][
            "local_refiner_require_transport_ready_when_holding"
        ]
        is False
    )
    assert navigation["post_grasp_transport"]["enabled"] is True
    assert (
        navigation["post_grasp_transport"]["enforce_before_navigation"]
        is True
    )
    assert navigation["post_place_retreat"]["enabled"] is False
    assert navigation["post_skill_return"]["enabled"] is False
    assert (
        navigation["post_skill_return"]["defer_until_navigation"] is False
    )
    assert navigation["navigation"]["start_collision_recovery"]["enabled"] is True


def test_periodic_verification_mode_remains_available(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_vlm_pi05.py",
            "--long_horizon_task",
            "pick the potato",
            "--plan_only",
            "--no_env",
            "--verification_mode",
            "periodic",
            "--pi05_verify_interval",
            "20",
        ],
    )
    args = parse_args()
    assert args.verification_mode == "periodic"
    assert args.pi05_verify_interval == 20


def test_vlm_module_parameters_are_independently_configurable(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_vlm_pi05.py",
            "--long_horizon_task",
            "pick the potato",
            "--plan_only",
            "--no_env",
            "--vlm_model",
            "planner-model",
            "--vlm_max_tokens",
            "4096",
            "--grounding_vlm_model",
            "grounding-model",
            "--grounding_vlm_max_tokens",
            "1024",
            "--grounding_vlm_validation_retries",
            "3",
            "--verifier_vlm_model",
            "verifier-model",
            "--verifier_vlm_max_tokens",
            "3072",
            "--verifier_vlm_enable_thinking",
        ],
    )
    args = parse_args()
    assert args.vlm_model == "planner-model"
    assert args.vlm_max_tokens == 4096
    assert args.grounding_vlm_model == "grounding-model"
    assert args.grounding_vlm_max_tokens == 1024
    assert args.grounding_vlm_enable_thinking is False
    assert args.grounding_vlm_validation_retries == 3
    assert args.verifier_vlm_model == "verifier-model"
    assert args.verifier_vlm_max_tokens == 3072
    assert args.verifier_vlm_enable_thinking is True


def test_vlm_module_parameters_inherit_planner_endpoint_by_default(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_vlm_pi05.py",
            "--long_horizon_task",
            "pick the potato",
            "--plan_only",
            "--no_env",
            "--vlm_base_url",
            "http://shared-vlm/v1",
            "--vlm_model",
            "shared-model",
        ],
    )
    args = parse_args()
    assert args.grounding_vlm_base_url == "http://shared-vlm/v1"
    assert args.verifier_vlm_base_url == "http://shared-vlm/v1"
    assert args.grounding_vlm_model == "shared-model"
    assert args.verifier_vlm_model == "shared-model"


def test_local_pose_vlm_delegates_to_shared_vision_client():
    response = {
        "distance_alignment": "ALIGNED",
        "lateral_alignment": "ALIGNED",
        "yaw_alignment": "ALIGNED",
        "action": "STOP",
        "confidence": 0.9,
        "target_visible": True,
        "operation_ready": True,
        "reason": "ready",
    }
    local_pose = OpenAICompatibleLocalPoseVLM(
        base_url="http://vlm.invalid/v1",
        model="qwen-vl",
        api_key=None,
    )
    fake = FakeVisionClient(response)
    local_pose._vision_client = fake
    decision = local_pose.decide(
        images={"robot0_topview": np.zeros((8, 8, 3), dtype=np.uint8)},
        context={"atomic_task": "PickObject"},
    )
    assert decision["action"] == "STOP"
    assert decision["operation_ready"] is True
    assert fake.calls[0]["images"].keys() == {"robot0_topview"}
    assert local_pose.last_raw_http_response == fake.last_raw_http_response


def test_grounding_and_verification_save_complete_multiview_artifacts(tmp_path: Path):
    env = ImageOnlyEnv()
    VLMRoboCasaGrounder(
        client=FakeVisionClient(grounding_response()),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
        log_dir=tmp_path,
    ).ground(env=env, atomic_task_call=pick_call())
    VLMAtomicTaskVerifier(
        client=FakeVisionClient(verifier_response(True)),
        observation_provider=provider(),
        scene_context=SCENE_CONTEXT,
        log_dir=tmp_path,
    )(
        env=env,
        atomic_task_call=pick_call(),
        observation={},
        step_index=20,
        info={},
    )

    grounding_dir = next((tmp_path / "vlm_grounding").iterdir())
    verifier_dir = next((tmp_path / "vlm_verification").iterdir())
    for artifact_dir in (grounding_dir, verifier_dir):
        assert len(list(artifact_dir.glob("robot0_*.png"))) == 5
        request = json.loads((artifact_dir / "request.json").read_text())
        assert request["camera_names"] == list(DEFAULT_REASONING_CAMERA_NAMES)
        assert request["system_prompt"]
        assert request["user_prompt"]
        assert (artifact_dir / "raw_http_response.json").is_file()
        assert (artifact_dir / "parsed_response.json").is_file()
        assert (artifact_dir / "runtime_decision.json").is_file()
    assert (verifier_dir / "evidence.json").is_file()
    assert (verifier_dir / "rejected_evidence.json").is_file()
    assert (verifier_dir / "final_decision.json").is_file()


def test_fixed_scheduler_limits_only_vlm_reasoning_views():
    scheduler = FixedViewScheduler()
    grounding_client = FakeVisionClient(grounding_response())
    verification_client = FakeVisionClient(verifier_response(True))
    env = ImageOnlyEnv()

    VLMRoboCasaGrounder(
        client=grounding_client,
        observation_provider=provider(),
        view_scheduler=scheduler,
        scene_context=SCENE_CONTEXT,
    ).ground(env=env, atomic_task_call=pick_call())
    VLMAtomicTaskVerifier(
        client=verification_client,
        observation_provider=provider(),
        view_scheduler=scheduler,
        scene_context=SCENE_CONTEXT,
    )(
        env=env,
        atomic_task_call=pick_call(),
        observation={},
        step_index=60,
        info={},
    )

    assert tuple(grounding_client.calls[0]["images"]) == (
        "robot0_eye_in_hand",
        "robot0_frontview",
    )
    assert tuple(verification_client.calls[0]["images"]) == (
        "robot0_eye_in_hand",
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_frontview",
    )
    assert POLICY_CAMERA_NAMES == (
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    )
