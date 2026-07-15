from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from atomic_task_verifier import RuntimeAtomicTaskVerifier  # noqa: E402


class Fixture:
    name = "sink_main_group"
    nat_lang = "sink"


class Model:
    @staticmethod
    def body_name2id(name):
        assert name == "mobilebase0_base"
        return 0


class Data:
    def __init__(self, position):
        self.body_xpos = np.asarray([position], dtype=float)
        self.body_xmat = np.asarray([np.eye(3).reshape(-1)], dtype=float)


class Sim:
    def __init__(self, position):
        self.model = Model()
        self.data = Data(position)


class NavigationEnv:
    def __init__(self, position):
        fixture = Fixture()
        self.fixtures = {"sink_main_group": fixture}
        self.objects = {}
        self.target_fixture = fixture
        self.target_pos = np.asarray([1.0, 2.0, 0.0])
        self.target_ori = np.asarray([0.0, 0.0, 0.0])
        self.sim = Sim(position)


def navigation_call():
    return {
        "subgoal_id": "navigate_sink",
        "atomic_task": "NavigateKitchen",
        "policy_prompt": "Navigate to the sink.",
        "arguments": {
            "fixture_id": "sink_main_group",
            "fixture_name": "sink",
        },
        "termination_condition": {
            "predicate": "navigation_pose",
            "subject": "sink_main_group",
            "desired_value": True,
        },
    }


def verify(env):
    return RuntimeAtomicTaskVerifier()(
        env=env,
        atomic_task_call=navigation_call(),
        observation={},
        step_index=10,
        info={},
    )


def test_navigation_pose_matches_robocasa_position_and_orientation_thresholds():
    result = verify(NavigationEnv([1.1, 2.1, 0.0]))
    assert result["status"] == "success"
    evidence = result["state_evidence"][-1]
    assert evidence["position_ok"] is True
    assert evidence["orientation_ok"] is True

    result = verify(NavigationEnv([1.21, 2.0, 0.0]))
    assert result["status"] == "uncertain"
    assert result["state_evidence"][-1]["position_ok"] is False


class CountEnv:
    def __init__(self, inside):
        self.fixtures = {}
        self.objects = {
            "ice_cube1": object(),
            "ice_cube2": object(),
            "ice_cube3": object(),
            "ice_cube4": object(),
            "glass_cup1": object(),
        }
        self.inside = set(inside)
        self.sim = object()


def count_call():
    return {
        "subgoal_id": "cup1_ice2",
        "atomic_task": "MakeIcedCoffee",
        "policy_prompt": "Place another ice cube in glass cup 1.",
        "arguments": {},
        "termination_condition": {
            "predicate": "receptacle_count",
            "subject": "glass_cup1",
            "object_prefix": "ice_cube",
            "desired_value": 2,
        },
    }


def test_receptacle_count_accepts_any_matching_ice_cube_ids(monkeypatch):
    from robocasa.utils import object_utils

    monkeypatch.setattr(
        object_utils,
        "check_obj_in_receptacle",
        lambda env, obj_name, receptacle_name, th: (
            receptacle_name == "glass_cup1" and obj_name in env.inside and th == 0.5
        ),
    )
    verifier = RuntimeAtomicTaskVerifier()
    result = verifier(
        env=CountEnv({"ice_cube2", "ice_cube4"}),
        atomic_task_call=count_call(),
        observation={},
        step_index=20,
        info={},
    )
    assert result["status"] == "success"
    evidence = result["state_evidence"][-1]
    assert evidence["count"] == 2
    assert evidence["inside"] == ["ice_cube2", "ice_cube4"]

    result = verifier(
        env=CountEnv({"ice_cube2"}),
        atomic_task_call=count_call(),
        observation={},
        step_index=20,
        info={},
    )
    assert result["status"] == "uncertain"
    assert result["state_evidence"][-1]["count"] == 1


class HandoffFixture:
    name = "microwave_main_group"

    @staticmethod
    def get_ext_sites(relative=False):
        assert relative is False
        return (
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
        )


class HandoffEnv:
    def __init__(self, eef_position=(0.5, 0.5, 0.5)):
        self.fixtures = {"microwave_main_group": HandoffFixture()}
        self.objects = {
            "obj": object(),
            "ice_cube1": object(),
            "ice_cube2": object(),
        }
        self.held = set()
        self.far = False
        self.robots = [
            type("Robot", (), {"eef_site_id": {"right": 0}})()
        ]
        data = type(
            "HandoffData",
            (),
            {"site_xpos": np.asarray([eef_position], dtype=float)},
        )()
        self.sim = type("HandoffSim", (), {"data": data})()


def handoff_call(subgoal_id, condition, *, metadata=None):
    return {
        "subgoal_id": subgoal_id,
        "atomic_task": "MakeIcedCoffee",
        "policy_prompt": "Complete a safe handoff.",
        "arguments": {},
        "termination_condition": condition,
        "metadata": metadata or {},
    }


def test_released_supports_object_groups_and_gripper_far_uses_threshold(monkeypatch):
    from robocasa.utils import object_utils

    monkeypatch.setattr(
        object_utils,
        "check_obj_grasped",
        lambda env, obj_name: obj_name in env.held,
    )
    monkeypatch.setattr(
        object_utils,
        "gripper_obj_far",
        lambda env, obj_name, th: (
            obj_name == "obj" and th == 0.4 and env.far
        ),
    )
    env = HandoffEnv()
    verifier = RuntimeAtomicTaskVerifier()

    env.held = {"ice_cube2"}
    released_call = handoff_call(
        "released_cubes",
        {
            "predicate": "released",
            "subject": "ice_cube",
            "object_prefix": "ice_cube",
            "desired_value": True,
        },
    )
    result = verifier(
        env=env,
        atomic_task_call=released_call,
        observation={},
        step_index=10,
        info={},
    )
    assert result["status"] == "uncertain"
    assert result["state_evidence"][-1]["held"] == ["ice_cube2"]

    env.held.clear()
    result = verifier(
        env=env,
        atomic_task_call=released_call,
        observation={},
        step_index=15,
        info={},
    )
    assert result["status"] == "success"

    far_call = handoff_call(
        "gripper_far",
        {
            "predicate": "gripper_far",
            "subject": "obj",
            "threshold": 0.4,
            "desired_value": True,
        },
    )
    result = verifier(
        env=env,
        atomic_task_call=far_call,
        observation={},
        step_index=10,
        info={},
    )
    assert result["status"] == "uncertain"
    env.far = True
    result = verifier(
        env=env,
        atomic_task_call=far_call,
        observation={},
        step_index=15,
        info={},
    )
    assert result["status"] == "success"
    assert result["state_evidence"][-1]["threshold"] == 0.4


def test_eef_outside_fixture_respects_expanded_safety_margin():
    env = HandoffEnv(eef_position=(1.01, 0.5, 0.5))
    call = handoff_call(
        "eef_clear",
        {
            "predicate": "eef_outside_fixture",
            "subject": "microwave_main_group",
            "margin": 0.02,
            "desired_value": True,
        },
    )
    verifier = RuntimeAtomicTaskVerifier()

    result = verifier(
        env=env,
        atomic_task_call=call,
        observation={},
        step_index=10,
        info={},
    )
    assert result["status"] == "uncertain"
    assert result["state_evidence"][-1]["inside_expanded_fixture"] is True

    env.sim.data.site_xpos[0] = np.asarray([1.03, 0.5, 0.5])
    result = verifier(
        env=env,
        atomic_task_call=call,
        observation={},
        step_index=15,
        info={},
    )
    assert result["status"] == "success"
    assert result["state_evidence"][-1]["inside_expanded_fixture"] is False


def test_contract_requires_consecutive_success_and_resets_after_instability():
    call = navigation_call()
    call["metadata"] = {
        "skill_contract": {
            "verification": {"required_consecutive_successes": 2}
        }
    }
    env = NavigationEnv([1.1, 2.1, 0.0])
    verifier = RuntimeAtomicTaskVerifier()

    first = verifier(
        env=env,
        atomic_task_call=call,
        observation={},
        step_index=10,
        info={},
    )
    assert first["status"] == "uncertain"
    assert first["state_evidence"][-1]["consecutive_successes"] == 1

    second = verifier(
        env=env,
        atomic_task_call=call,
        observation={},
        step_index=15,
        info={},
    )
    assert second["status"] == "success"
    assert second["state_evidence"][-1]["stable"] is True

    env.sim.data.body_xpos[0] = np.asarray([1.3, 2.0, 0.0])
    unstable = verifier(
        env=env,
        atomic_task_call=call,
        observation={},
        step_index=20,
        info={},
    )
    assert unstable["status"] == "uncertain"
    assert unstable["state_evidence"][-1]["consecutive_successes"] == 0

    env.sim.data.body_xpos[0] = np.asarray([1.1, 2.1, 0.0])
    rebuilding = verifier(
        env=env,
        atomic_task_call=call,
        observation={},
        step_index=25,
        info={},
    )
    assert rebuilding["status"] == "uncertain"
    recovered = verifier(
        env=env,
        atomic_task_call=call,
        observation={},
        step_index=30,
        info={},
    )
    assert recovered["status"] == "success"

