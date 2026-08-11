from __future__ import annotations

from pathlib import Path

import numpy as np

from robocasa.atomic_task_schemas import AtomicTaskCall
from robocasa.local_pose_policy_adapter import LocalPoseRefiningPolicyAdapter
from robocasa.local_work_pose_refiner import LocalWorkPoseRefiner
from robocasa.pi05_env import POLICY_CAMERA_NAMES, RawRoboCasaPi05Env
from robocasa.utils.camera_utils import get_robot_cam_configs


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

    def decide(self, *, images, context):
        self.camera_history.append(tuple(images))
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
        return self.observation, 0.0, False, {}


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
    assert len(env.actions) == 3
    assert env.actions[0][7] > 0
    assert env.actions[0][8] == 0
    assert env.actions[0][11] == 1
    assert decision_maker.camera_history == [
        ("robot0_topview", "robot0_frontview"),
        ("robot0_topview", "robot0_frontview"),
    ]
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

