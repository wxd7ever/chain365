from __future__ import annotations

from pathlib import Path

import numpy as np

from robocasa.atomic_task_schemas import AtomicTaskCall
from robocasa.local_pose_policy_adapter import LocalPoseRefiningPolicyAdapter
from robocasa.local_work_pose_refiner import (
    LocalWorkPoseRefiner,
    OpenAICompatibleLocalPoseVLM,
    _extract_response_text,
    _summarize_vlm_usage,
    _vlm_usage_breakdown,
)
from robocasa.pi05_env import POLICY_CAMERA_NAMES, RawRoboCasaPi05Env
from robocasa.utils.camera_utils import get_robot_cam_configs
from robocasa.view_scheduler import FixedViewScheduler


def test_extract_response_text_accepts_openai_and_vllm_variants():
    expected = '{"action":"TURN_RIGHT_SMALL"}'

    def response(message):
        return {"choices": [{"message": message}]}

    assert _extract_response_text(response({"content": expected})) == expected
    assert (
        _extract_response_text(
            response(
                {"content": [
                    "{\"action\":",
                    {"type": "text", "text": "\"TURN_RIGHT_SMALL\""},
                    {"text": {"value": "}"}},
                ]}
            )
        )
        == expected
    )
    assert _extract_response_text(
        response({"content": None, "reasoning_content": expected})
    ) == expected
    assert _extract_response_text(
        response({"content": {"type": "text", "text": expected}})
    ) == expected


def _operation_call(task: str = "OpenMicrowave") -> AtomicTaskCall:
    return AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "operate",
            "atomic_task": task,
            "policy_prompt": "Open the microwave door.",
            "arguments": {"fixture_id": "microwave"},
            "termination_condition": {
                "predicate": "open",
                "subject": "microwave",
                "desired_value": True,
            },
        }
    )


class _DecisionMaker:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.camera_history = []
        self.context_history = []

    def decide(self, *, images, context):
        self.camera_history.append(tuple(images))
        self.context_history.append(dict(context))
        return self.decisions.pop(0)


class _FakeRefinerEnv:
    def __init__(self):
        self.observation = {
            "robot0_base_pos": np.zeros(3, dtype=np.float32),
            "robot0_base_quat": np.array([1, 0, 0, 0], dtype=np.float32),
        }
        self.actions = []

    def get_observation(self):
        return self.observation

    def render_camera_views(self, camera_names, *, height, width):
        return {
            name: np.full((height, width, 3), index * 20, dtype=np.uint8)
            for index, name in enumerate(camera_names, start=1)
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.actions.append(action.copy())
        self.observation = dict(self.observation)
        base_pos = np.asarray(self.observation["robot0_base_pos"]).copy()
        base_pos[:2] += action[7:9] * 0.05
        self.observation["robot0_base_pos"] = base_pos
        quaternion = np.asarray(self.observation["robot0_base_quat"]).copy()
        yaw = 2.0 * np.arctan2(quaternion[2], quaternion[3])
        yaw += float(action[9]) * 0.05
        self.observation["robot0_base_quat"] = np.array(
            [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)],
            dtype=np.float32,
        )
        return self.observation, 0.0, False, {}


def test_held_mode_uses_normal_adaptive_step_profiles(tmp_path: Path):
    refiner = LocalWorkPoseRefiner(
        decision_maker=_DecisionMaker([]),
        log_dir=tmp_path,
        camera_names=("robot0_topview",),
        translation_distance_m=0.30,
        medium_translation_distance_m=0.12,
        fine_translation_distance_m=0.05,
        rotation_angle_deg=25.0,
        medium_rotation_angle_deg=10.0,
        fine_rotation_angle_deg=5.0,
        held_translation_distance_m=0.01,
        held_rotation_angle_deg=1.0,
        held_object_guard=False,
    )

    for axis in ("distance", "lateral", "yaw"):
        for level in range(3):
            assert refiner._step_profile(
                axis, level, held_mode=True
            ) == refiner._step_profile(axis, level, held_mode=False)

    assert refiner._step_profile("distance", 0, True)["translation_m"] == 0.30
    assert refiner._step_profile("distance", 1, True)["translation_m"] == 0.12
    assert refiner._step_profile("distance", 2, True)["translation_m"] == 0.05
    assert refiner._step_profile("yaw", 0, True)["rotation_deg"] == 25.0
    assert refiner._step_profile("yaw", 1, True)["rotation_deg"] == 10.0
    assert refiner._step_profile("yaw", 2, True)["rotation_deg"] == 5.0


def test_top_camera_is_registered_but_not_added_to_pi05_policy_contract():
    config = get_robot_cam_configs("PandaOmron")
    assert config["robot0_topview"]["parent_body"] == "mobilebase0_support"
    assert config["robot0_topview"]["quat"] == [1.0, 0.0, 0.0, 0.0]
    assert "robot0_topview" not in POLICY_CAMERA_NAMES


def test_named_camera_render_is_upright():
    raw_frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    class _Sim:
        def render(self, **kwargs):
            assert kwargs["camera_name"] == "robot0_topview"
            return raw_frame

    class _Raw:
        sim = _Sim()

    env = RawRoboCasaPi05Env(_Raw())
    rendered = env.render_camera("robot0_topview", height=2, width=3)
    np.testing.assert_array_equal(rendered, raw_frame[::-1])


def test_refiner_executes_one_small_base_command_then_stops(tmp_path: Path):
    decision_maker = _DecisionMaker(
        [
            {
                "action": "FORWARD_SMALL",
                "confidence": 0.9,
                "target_visible": True,
                "operation_ready": False,
                "reason": "too far",
            },
            {
                "action": "STOP",
                "confidence": 0.95,
                "target_visible": True,
                "operation_ready": True,
                "reason": "aligned",
            },
        ]
    )
    env = _FakeRefinerEnv()
    refiner = LocalWorkPoseRefiner(
        decision_maker=decision_maker,
        log_dir=tmp_path,
        camera_names=("robot0_topview", "robot0_frontview"),
        image_size=16,
        max_decisions=3,
        action_steps=2,
        settle_steps=1,
        held_object_guard=False,
    )
    result = refiner.refine(
        env=env,
        atomic_task_call=_operation_call(),
        grounding_result={"target_fixture_alias": "microwave"},
        episode_id="episode/1",
    )

    assert result["success"] is True
    assert result["uses_top_camera"] is True
    assert result["num_executed_actions"] == 1
    assert len(env.actions) > 3
    assert result["decisions"][0]["motion"]["converged"] is True
    assert result["decisions"][0]["motion"]["requested_translation_m"] == 0.10
    assert abs(env.observation["robot0_base_pos"][0] - 0.10) <= 0.005
    assert env.actions[0][7] > 0
    assert env.actions[0][8] == 0
    assert env.actions[0][11] == 1
    assert decision_maker.camera_history == [
        ("robot0_topview", "robot0_frontview"),
        ("robot0_topview", "robot0_frontview"),
    ]
    assert decision_maker.context_history[1]["previous_corrections"][0][
        "converged"
    ] is True
    assert Path(result["artifact_dir"], "result.json").is_file()


def test_refiner_does_not_move_on_low_confidence(tmp_path: Path):
    env = _FakeRefinerEnv()
    refiner = LocalWorkPoseRefiner(
        decision_maker=_DecisionMaker(
            [
                {
                    "action": "STRAFE_LEFT_SMALL",
                    "confidence": 0.2,
                    "target_visible": False,
                    "operation_ready": False,
                    "reason": "uncertain",
                }
            ]
        ),
        log_dir=tmp_path,
        camera_names=("robot0_topview",),
        image_size=8,
        held_object_guard=False,
    )
    result = refiner.refine(
        env=env,
        atomic_task_call=_operation_call(),
        grounding_result={},
        episode_id=0,
    )
    assert result["success"] is False
    assert result["failure_code"] == "LOCAL_POSE_LOW_CONFIDENCE"
    assert env.actions == []


def test_structured_alignment_is_mapped_to_closed_loop_strafe(tmp_path: Path):
    env = _FakeRefinerEnv()
    decision_maker = _DecisionMaker(
        [
            {
                "distance_alignment": "ALIGNED",
                "lateral_alignment": "TARGET_LEFT",
                "yaw_alignment": "STRAIGHT",
                "confidence": 0.9,
                "target_visible": True,
                "operation_ready": False,
                "reason": "target is left of the workspace center",
            },
            {
                "distance_alignment": "ALIGNED",
                "lateral_alignment": "ALIGNED",
                "yaw_alignment": "ALIGNED",
                "confidence": 0.95,
                "target_visible": True,
                "operation_ready": True,
                "reason": "aligned",
            },
        ]
    )
    refiner = LocalWorkPoseRefiner(
        decision_maker=decision_maker,
        log_dir=tmp_path,
        camera_names=("robot0_topview",),
        image_size=8,
        held_object_guard=False,
    )
    result = refiner.refine(
        env=env,
        atomic_task_call=_operation_call(),
        grounding_result={},
        episode_id=0,
    )

    assert result["success"] is True
    assert result["decisions"][0]["action"] == "STRAFE_LEFT_SMALL"
    assert abs(env.observation["robot0_base_pos"][1] - 0.10) <= 0.005


def test_invalid_stop_is_reprompted_without_moving(tmp_path: Path):
    env = _FakeRefinerEnv()
    decision_maker = _DecisionMaker(
        [
            {
                "action": "STOP",
                "confidence": 0.9,
                "target_visible": True,
                "operation_ready": False,
                "reason": "visible",
            },
            {
                "action": "FORWARD_SMALL",
                "confidence": 0.9,
                "target_visible": True,
                "operation_ready": False,
                "reason": "too far",
            },
            {
                "action": "STOP",
                "confidence": 0.9,
                "target_visible": True,
                "operation_ready": True,
                "reason": "ready",
            },
        ]
    )
    refiner = LocalWorkPoseRefiner(
        decision_maker=decision_maker,
        log_dir=tmp_path,
        camera_names=("robot0_topview",),
        image_size=8,
        held_object_guard=False,
    )
    result = refiner.refine(
        env=env,
        atomic_task_call=_operation_call(),
        grounding_result={},
        episode_id=0,
    )

    assert result["success"] is True
    assert result["invalid_stop_count"] == 1
    assert result["num_executed_actions"] == 1
    assert decision_maker.context_history[1]["last_validation_error"]


def test_invalid_structured_response_is_reprompted_without_moving(tmp_path: Path):
    env = _FakeRefinerEnv()
    decision_maker = _DecisionMaker(
        [
            {
                "distance_alignment": "ALIGNED",
                "lateral_alignment": "ALIGNED",
                "yaw_alignment": "FACE_DIAGONALLY",
                "confidence": 0.9,
                "target_visible": True,
                "operation_ready": False,
                "reason": "invalid enum",
            },
            {
                "distance_alignment": "ALIGNED",
                "lateral_alignment": "ALIGNED",
                "yaw_alignment": "ALIGNED",
                "confidence": 0.9,
                "target_visible": True,
                "operation_ready": True,
                "reason": "ready",
            },
        ]
    )
    refiner = LocalWorkPoseRefiner(
        decision_maker=decision_maker,
        log_dir=tmp_path,
        camera_names=("robot0_topview",),
        image_size=8,
        held_object_guard=False,
    )
    result = refiner.refine(
        env=env,
        atomic_task_call=_operation_call(),
        grounding_result={},
        episode_id=0,
    )

    assert result["success"] is True
    assert result["invalid_decision_count"] == 1
    assert result["num_executed_actions"] == 0
    assert env.actions == []
    assert "FACE_DIAGONALLY" in result["decisions"][0]["validation_error"]
    assert decision_maker.context_history[1]["last_validation_error"]


def test_converged_axis_repeat_uses_other_vlm_misalignment(tmp_path: Path):
    env = _FakeRefinerEnv()
    decision_maker = _DecisionMaker(
        [
            {
                "distance_alignment": "ALIGNED",
                "lateral_alignment": "TARGET_LEFT",
                "yaw_alignment": "TURN_RIGHT",
                "confidence": 0.9,
                "target_visible": True,
                "operation_ready": False,
                "reason": "turn and move left",
            },
            {
                "distance_alignment": "ALIGNED",
                "lateral_alignment": "TARGET_LEFT",
                "yaw_alignment": "TURN_RIGHT",
                "confidence": 0.9,
                "target_visible": True,
                "operation_ready": False,
                "reason": "still reports both errors",
            },
            {
                "distance_alignment": "ALIGNED",
                "lateral_alignment": "ALIGNED",
                "yaw_alignment": "ALIGNED",
                "confidence": 0.9,
                "target_visible": True,
                "operation_ready": True,
                "reason": "ready",
            },
        ]
    )
    refiner = LocalWorkPoseRefiner(
        decision_maker=decision_maker,
        log_dir=tmp_path,
        camera_names=("robot0_topview",),
        image_size=8,
        motion_max_steps=250,
        held_object_guard=False,
    )
    result = refiner.refine(
        env=env,
        atomic_task_call=_operation_call(),
        grounding_result={},
        episode_id=0,
    )

    assert result["success"] is True
    assert result["decisions"][0]["action"] == "TURN_RIGHT_SMALL"
    assert result["decisions"][1]["proposed_action"] == "TURN_RIGHT_SMALL"
    assert result["decisions"][1]["action"] == "STRAFE_LEFT_SMALL"
    assert (
        result["decisions"][1]["action_override"]
        == "skip_completed_action_repeat"
    )
    assert result["decisions"][1]["motion"]["requested_translation_m"] == 0.10


def test_closed_loop_motion_reports_stall(tmp_path: Path):
    class _StalledEnv(_FakeRefinerEnv):
        def step(self, action):
            self.actions.append(np.asarray(action, dtype=np.float32).copy())
            return self.observation, 0.0, False, {}

    env = _StalledEnv()
    refiner = LocalWorkPoseRefiner(
        decision_maker=_DecisionMaker(
            [
                {
                    "action": "FORWARD_SMALL",
                    "confidence": 0.9,
                    "target_visible": True,
                    "operation_ready": False,
                    "reason": "too far",
                }
            ]
        ),
        log_dir=tmp_path,
        camera_names=("robot0_topview",),
        image_size=8,
        max_decisions=1,
        motion_max_steps=150,
        held_object_guard=False,
    )
    result = refiner.refine(
        env=env,
        atomic_task_call=_operation_call(),
        grounding_result={},
        episode_id=0,
    )

    assert result["success"] is False
    assert result["failure_code"] == "LOCAL_POSE_MOTION_STALLED"
    assert len(env.actions) == 50


def test_translation_deadzone_is_logged_as_partial_convergence(tmp_path: Path):
    class _DeadzoneEnv(_FakeRefinerEnv):
        def step(self, action):
            observation, reward, done, info = super().step(action)
            position = np.asarray(observation["robot0_base_pos"]).copy()
            position[0] = min(float(position[0]), 0.08)
            observation["robot0_base_pos"] = position
            self.observation = observation
            return observation, reward, done, info

    env = _DeadzoneEnv()
    refiner = LocalWorkPoseRefiner(
        decision_maker=_DecisionMaker(
            [
                {
                    "action": "FORWARD_SMALL",
                    "confidence": 0.9,
                    "target_visible": True,
                    "operation_ready": False,
                    "reason": "too far",
                },
                {
                    "action": "STOP",
                    "confidence": 0.9,
                    "target_visible": True,
                    "operation_ready": True,
                    "reason": "ready after useful correction",
                },
            ]
        ),
        log_dir=tmp_path,
        camera_names=("robot0_topview",),
        image_size=8,
        motion_max_steps=250,
        held_object_guard=False,
    )
    result = refiner.refine(
        env=env,
        atomic_task_call=_operation_call(),
        grounding_result={},
        episode_id=0,
    )

    motion = result["decisions"][0]["motion"]
    assert result["success"] is True
    assert motion["converged"] is True
    assert motion["convergence_mode"] == "partial_deadzone"
    assert 0.075 <= motion["actual_translation_m"] <= 0.081
    assert motion["final_translation_error_m"] > 0.005


def test_rotation_closed_loop_compensates_translation_drift(tmp_path: Path):
    class _DriftingRotationEnv(_FakeRefinerEnv):
        def step(self, action):
            observation, reward, done, info = super().step(action)
            action = np.asarray(action, dtype=np.float32)
            base_pos = np.asarray(observation["robot0_base_pos"]).copy()
            base_pos[0] += abs(float(action[9])) * 0.002
            observation["robot0_base_pos"] = base_pos
            self.observation = observation
            return observation, reward, done, info

    env = _DriftingRotationEnv()
    refiner = LocalWorkPoseRefiner(
        decision_maker=_DecisionMaker(
            [
                {
                    "action": "TURN_LEFT_SMALL",
                    "confidence": 0.9,
                    "target_visible": True,
                    "operation_ready": False,
                    "reason": "turn left",
                },
                {
                    "action": "STOP",
                    "confidence": 0.9,
                    "target_visible": True,
                    "operation_ready": True,
                    "reason": "ready",
                },
            ]
        ),
        log_dir=tmp_path,
        camera_names=("robot0_topview",),
        image_size=8,
        motion_max_steps=200,
        held_object_guard=False,
    )
    result = refiner.refine(
        env=env,
        atomic_task_call=_operation_call(),
        grounding_result={},
        episode_id=0,
    )

    motion = result["decisions"][0]["motion"]
    assert result["success"] is True
    assert motion["converged"] is True
    assert motion["actual_translation_m"] <= 0.005
    assert abs(motion["actual_rotation_deg"] - 8.0) <= 0.5
    assert any(abs(float(action[7])) > 0 for action in env.actions[1:])


def test_translation_closed_loop_compensates_yaw_drift(tmp_path: Path):
    class _YawDriftingTranslationEnv(_FakeRefinerEnv):
        def step(self, action):
            action = np.asarray(action, dtype=np.float32).copy()
            observation, reward, done, info = super().step(action)
            if abs(float(action[7])) + abs(float(action[8])) > 0:
                quaternion = np.asarray(observation["robot0_base_quat"]).copy()
                yaw = 2.0 * np.arctan2(quaternion[2], quaternion[3])
                yaw += abs(float(action[7])) * 0.01
                observation["robot0_base_quat"] = np.array(
                    [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)],
                    dtype=np.float32,
                )
                self.observation = observation
            return observation, reward, done, info

    env = _YawDriftingTranslationEnv()
    refiner = LocalWorkPoseRefiner(
        decision_maker=_DecisionMaker(
            [
                {
                    "action": "FORWARD_SMALL",
                    "confidence": 0.9,
                    "target_visible": True,
                    "operation_ready": False,
                    "reason": "too far",
                },
                {
                    "action": "STOP",
                    "confidence": 0.9,
                    "target_visible": True,
                    "operation_ready": True,
                    "reason": "ready",
                },
            ]
        ),
        log_dir=tmp_path,
        camera_names=("robot0_topview",),
        image_size=8,
        motion_max_steps=250,
        held_object_guard=False,
    )
    result = refiner.refine(
        env=env,
        atomic_task_call=_operation_call(),
        grounding_result={},
        episode_id=0,
    )
    motion = result["decisions"][0]["motion"]
    assert result["success"] is True
    assert motion["converged"] is True
    assert abs(motion["actual_translation_m"] - 0.10) <= 0.005
    assert motion["actual_rotation_deg"] <= 0.5
    assert any(abs(float(action[9])) > 0 for action in env.actions[1:])


def test_policy_adapter_skips_inserted_navigation_and_refines_operation():
    class _Policy:
        def __init__(self):
            self.calls = []

        def execute(self, **kwargs):
            call = AtomicTaskCall.from_mapping(kwargs["scheduler_query"]["atomic_task_call"])
            self.calls.append(call.atomic_task)
            return {"status": "success", "success": True, "atomic_task": call.atomic_task}

    class _Grounding:
        def to_dict(self):
            return {"grounded": True, "target_fixture_alias": "microwave"}

    class _Grounder:
        def ground(self, **kwargs):
            return _Grounding()

    class _Refiner:
        camera_names = ("robot0_topview",)

        def __init__(self):
            self.calls = []

        def refine(self, **kwargs):
            self.calls.append(kwargs["atomic_task_call"].atomic_task)
            return {
                "status": "success",
                "success": True,
                "failure_code": None,
                "num_executed_actions": 1,
            }

    policy = _Policy()
    refiner = _Refiner()
    adapter = LocalPoseRefiningPolicyAdapter(
        policy_adapter=policy,
        refiner=refiner,
        grounder=_Grounder(),
    )
    navigation = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "navigate",
            "atomic_task": "NavigateKitchen",
            "arguments": {"fixture_id": "microwave"},
            "termination_condition": {
                "predicate": "navigation_pose",
                "subject": "microwave",
                "desired_value": True,
            },
        }
    )
    adapter.execute(
        env=object(),
        scheduler_query={"atomic_task_call": navigation.to_dict()},
        episode_id=0,
    )
    operation_result = adapter.execute(
        env=object(),
        scheduler_query={"atomic_task_call": _operation_call().to_dict()},
        episode_id=1,
    )

    assert policy.calls == ["NavigateKitchen", "OpenMicrowave"]
    assert refiner.calls == ["OpenMicrowave"]
    assert operation_result["local_pose_refinement_result"]["success"] is True



def test_refiner_exposes_task_conditioned_camera_selection(tmp_path: Path):
    refiner = LocalWorkPoseRefiner(
        decision_maker=_DecisionMaker([]),
        log_dir=tmp_path,
        camera_names=(
            "robot0_topview",
            "robot0_frontview",
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ),
        view_scheduler=FixedViewScheduler(),
        held_object_guard=False,
    )
    call = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "pick",
            "atomic_task": "PickObject",
            "arguments": {"object_id": "vegetable"},
            "termination_condition": {
                "predicate": "holding",
                "subject": "vegetable",
                "desired_value": True,
            },
        }
    )

    assert refiner.select_camera_names(call) == (
        "robot0_topview",
        "robot0_frontview",
        "robot0_eye_in_hand",
    )

def test_vlm_usage_breakdown_separates_server_reasoning_tokens():
    raw = """{"choices":[{"finish_reason":"stop","message":{"content":"{\\"action\\":\\"STOP\\"}","reasoning_content":"brief analysis"}}]}"""
    breakdown = _vlm_usage_breakdown(
        {
            "prompt_tokens": 120,
            "completion_tokens": 80,
            "total_tokens": 200,
            "completion_tokens_details": {"reasoning_tokens": 60},
        },
        raw_http_response=raw,
        finish_reason="stop",
    )

    assert breakdown["input_tokens"] == 120
    assert breakdown["completion_tokens"] == 80
    assert breakdown["reasoning_tokens"] == 60
    assert breakdown["visible_output_tokens"] == 20
    assert breakdown["server_reported_reasoning_tokens"] is True
    assert breakdown["dominant_usage"] == "reasoning"
    summary = _summarize_vlm_usage(
        [{"decision_index": 3, "vlm_usage_breakdown": breakdown}]
    )
    assert summary["finish_reason_counts"] == {"stop": 1}
    assert summary["totals"]["reasoning_tokens"] == 60
    assert summary["rounds"][0]["decision_index"] == 3


def test_refiner_prompt_keeps_only_recent_compact_history(tmp_path: Path):
    env = _FakeRefinerEnv()
    decision_maker = _DecisionMaker(
        [
            {
                "action": "FORWARD_SMALL",
                "confidence": 0.9,
                "target_visible": True,
                "operation_ready": False,
                "reason": "first reason that must not be copied into history",
            },
            {
                "action": "STRAFE_LEFT_SMALL",
                "confidence": 0.9,
                "target_visible": True,
                "operation_ready": False,
                "reason": "second reason that must not be copied into history",
            },
            {
                "action": "TURN_LEFT_SMALL",
                "confidence": 0.9,
                "target_visible": True,
                "operation_ready": False,
                "reason": "third reason that must not be copied into history",
            },
            {
                "action": "STOP",
                "confidence": 0.9,
                "target_visible": True,
                "operation_ready": True,
                "reason": "ready",
            },
        ]
    )
    refiner = LocalWorkPoseRefiner(
        decision_maker=decision_maker,
        log_dir=tmp_path,
        camera_names=("robot0_topview",),
        image_size=8,
        max_decisions=4,
        held_object_guard=False,
    )

    result = refiner.refine(
        env=env,
        atomic_task_call=_operation_call(),
        grounding_result={},
        episode_id=0,
    )

    assert result["success"] is True
    final_context = decision_maker.context_history[-1]
    assert final_context["recent_actions"] == [
        "FORWARD_SMALL",
        "STRAFE_LEFT_SMALL",
        "TURN_LEFT_SMALL",
    ]
    assert len(final_context["previous_corrections"]) == 2
    assert all("reason" not in item for item in final_context["previous_corrections"])
    assert final_context["cumulative_state"]["executed_action_count"] == 3
    assert Path(result["artifact_dir"], "vlm_usage_summary.json").is_file()


def test_primary_error_stall_detects_motion_in_wrong_direction(tmp_path: Path):
    class _WrongDirectionEnv(_FakeRefinerEnv):
        def step(self, action):
            action = np.asarray(action, dtype=np.float32)
            self.actions.append(action.copy())
            self.observation = dict(self.observation)
            position = np.asarray(self.observation["robot0_base_pos"]).copy()
            position[1] += 0.002
            self.observation["robot0_base_pos"] = position
            return self.observation, 0.0, False, {}

    env = _WrongDirectionEnv()
    refiner = LocalWorkPoseRefiner(
        decision_maker=_DecisionMaker(
            [
                {
                    "action": "FORWARD_SMALL",
                    "confidence": 0.9,
                    "target_visible": True,
                    "operation_ready": False,
                    "reason": "too far",
                }
            ]
        ),
        log_dir=tmp_path,
        camera_names=("robot0_topview",),
        image_size=8,
        max_decisions=1,
        motion_max_steps=200,
        stall_window_steps=25,
        held_object_guard=False,
    )

    result = refiner.refine(
        env=env,
        atomic_task_call=_operation_call(),
        grounding_result={},
        episode_id=0,
    )

    motion = result["decisions"][0]["motion"]
    assert result["failure_code"] == "LOCAL_POSE_MOTION_STALLED"
    assert motion["steps"] == 25
    assert motion["stall_trigger"] == "primary_error_not_improving"
    assert motion["last_window_primary_improvement"] < 0.0



def test_local_pose_prompt_is_scoped_to_one_current_visual_decision():
    class _PromptClient:
        endpoint = "http://example.invalid/v1/chat/completions"
        last_response_text = None
        last_raw_http_response = None
        last_http_status = 200
        last_finish_reason = "stop"
        last_usage = {"prompt_tokens": 100, "completion_tokens": 20}

        def query_json(self, **kwargs):
            self.kwargs = kwargs
            return {
                "distance_alignment": "ALIGNED",
                "lateral_alignment": "ALIGNED",
                "yaw_alignment": "ALIGNED",
                "action": "STOP",
                "confidence": 0.9,
                "target_visible": True,
                "operation_ready": True,
                "reason": "Target is visibly reachable.",
            }

    model = OpenAICompatibleLocalPoseVLM(
        base_url="http://example.invalid/v1",
        model="test-model",
        api_key=None,
        max_tokens=12288,
        enable_thinking=True,
    )
    client = _PromptClient()
    model._vision_client = client

    decision = model.decide(
        images={"robot0_topview": np.zeros((8, 8, 3), dtype=np.uint8)},
        context={"atomic_task": "PlaceObject"},
    )

    assert decision["action"] == "STOP"
    assert client.kwargs["max_tokens"] == 12288
    assert "current atomic operation" in client.kwargs["user_prompt"]
    assert "Do not analyze the complete task" in client.kwargs["user_prompt"]
    assert "hypothetical action sequences" in client.kwargs["user_prompt"]
    assert "one-step RoboCasa" in client.kwargs["system_prompt"]
    assert "operation_ready=false" in client.kwargs["system_prompt"]
