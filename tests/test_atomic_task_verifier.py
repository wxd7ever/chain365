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
    def __init__(self, position, yaw=0.0):
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        self.body_xpos = np.asarray([position], dtype=float)
        self.body_xmat = np.asarray([rotation.reshape(-1)], dtype=float)


class Sim:
    def __init__(self, position, yaw=0.0):
        self.model = Model()
        self.data = Data(position, yaw)


class NavigationEnv:
    def __init__(self, position, yaw=0.0):
        fixture = Fixture()
        self.fixtures = {"sink_main_group": fixture}
        self.objects = {}
        self.target_fixture = fixture
        self.target_pos = np.asarray([1.0, 2.0, 0.0])
        self.target_ori = np.asarray([0.0, 0.0, 0.0])
        self.sim = Sim(position, yaw)


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


def test_navigation_pose_uses_relaxed_position_and_orientation_thresholds():
    result = verify(NavigationEnv([1.61, 2.0, 0.0], yaw=np.deg2rad(19.0)))
    assert result["status"] == "success"
    evidence = result["state_evidence"][-1]
    assert evidence["position_threshold"] == 0.62
    assert evidence["orientation_cosine_threshold"] == 0.90
    assert evidence["position_ok"] is True
    assert evidence["orientation_ok"] is True

    result = verify(NavigationEnv([1.63, 2.0, 0.0]))
    assert result["status"] == "uncertain"
    assert result["state_evidence"][-1]["position_ok"] is False

    result = verify(NavigationEnv([1.1, 2.1, 0.0], yaw=np.deg2rad(27.0)))
    assert result["status"] == "uncertain"
    assert result["state_evidence"][-1]["orientation_ok"] is False


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


class ReceptacleObject:
    @staticmethod
    def get_bbox_points(trans=None, rot=None):
        return [
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0]),
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0]),
            np.asarray([1.0, 1.0, 1.0]),
            np.asarray([0.0, 1.0, 1.0]),
            np.asarray([1.0, 0.0, 1.0]),
            np.asarray([1.0, 1.0, 0.0]),
        ]


class ReceptacleEnv:
    def __init__(self, eef_position):
        self.fixtures = {}
        self.objects = {"bowl": ReceptacleObject()}
        self.obj_body_id = {"bowl": 0}
        self.robots = [type("Robot", (), {"eef_site_id": {"right": 0}})()]
        data = type(
            "ReceptacleData",
            (),
            {
                "site_xpos": np.asarray([eef_position], dtype=float),
                "body_xpos": np.asarray([[0.0, 0.0, 0.0]], dtype=float),
                "body_xquat": np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float),
            },
        )()
        self.sim = type("ReceptacleSim", (), {"data": data})()


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


def test_eef_outside_object_receptacle_respects_world_boundary_and_margin():
    env = ReceptacleEnv(eef_position=(1.01, 0.5, 0.5))
    call = handoff_call(
        "eef_clear_bowl",
        {
            "predicate": "eef_outside_receptacle",
            "subject": "bowl",
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
    assert result["state_evidence"][-1]["entity_kind"] == "object"
    assert result["state_evidence"][-1]["inside_expanded_receptacle"] is True

    env.sim.data.site_xpos[0] = np.asarray([1.03, 0.5, 0.5])
    result = verifier(
        env=env,
        atomic_task_call=call,
        observation={},
        step_index=15,
        info={},
    )
    assert result["status"] == "success"
    assert result["state_evidence"][-1]["inside_expanded_receptacle"] is False


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

    env.sim.data.body_xpos[0] = np.asarray([1.7, 2.0, 0.0])
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



class DrawerAliasFixture:
    nat_lang = "drawer"

    def __init__(self, name, opened):
        self.name = name
        self.opened = opened

    def is_open(self, env):
        assert self in env.fixtures.values()
        return self.opened


class DrawerAliasEnv:
    def __init__(self):
        self.fixtures = {
            "stack_4_main_group_1": DrawerAliasFixture(
                "stack_4_main_group_1", False
            ),
            "stack_4_main_group_2": DrawerAliasFixture(
                "stack_4_main_group_2", True
            ),
        }
        self.objects = {}
        self.sim = object()


def drawer_alias_call(subject):
    return {
        "subgoal_id": "open_left_drawer",
        "atomic_task": "OpenDrawer",
        "policy_prompt": "Open the left drawer.",
        "arguments": {"drawer_side": "left"},
        "termination_condition": {
            "predicate": "open",
            "subject": subject,
            "desired_value": True,
        },
    }


def test_exact_alias_does_not_collapse_numeric_fixture_suffix():
    result = RuntimeAtomicTaskVerifier()(
        env=DrawerAliasEnv(),
        atomic_task_call=drawer_alias_call("stack_4_main_group_2"),
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "success"
    fixture_evidence = next(
        item
        for item in result["state_evidence"]
        if item.get("source") == "fixture_method"
    )
    assert fixture_evidence["entity"] == "stack_4_main_group_2"
    assert fixture_evidence["value"] is True


def test_vlm_alias_fallback_is_cached_and_revalidated_exactly():
    calls = []

    def resolver(**request):
        calls.append(request)
        assert {
            item["alias"] for item in request["candidates"]
        } >= {"stack_4_main_group_1", "stack_4_main_group_2"}
        return "stack_4_main_group_2"

    env = DrawerAliasEnv()
    verifier = RuntimeAtomicTaskVerifier(entity_alias_resolver=resolver)
    for step_index in (10, 15):
        result = verifier(
            env=env,
            atomic_task_call=drawer_alias_call("left drawer"),
            observation={},
            step_index=step_index,
            info={},
        )
        assert result["status"] == "success"
        fallback = next(
            item
            for item in result["state_evidence"]
            if item.get("source") == "vlm_entity_alias_resolver"
        )
        assert fallback["resolved_alias"] == "stack_4_main_group_2"

    assert len(calls) == 1


def test_vlm_alias_fallback_rejects_invented_alias():
    verifier = RuntimeAtomicTaskVerifier(
        entity_alias_resolver=lambda **request: "invented_drawer"
    )
    result = verifier(
        env=DrawerAliasEnv(),
        atomic_task_call=drawer_alias_call("left drawer"),
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "INVALID_TERMINATION_CONDITION"
    assert result["retryable"] is False
    assert "non-existent alias" in result["state_evidence"][0]["error"]


class RelationFixture:
    def __init__(self, name):
        self.name = name
        self.nat_lang = name


class RelationEnv:
    def __init__(self):
        self.fixtures = {"microwave": RelationFixture("microwave")}
        self.objects = {"vegetable": object(), "bowl": object()}
        self.sim = object()


def relation_call(predicate, subject, destination, *, threshold=None):
    condition = {
        "predicate": predicate,
        "subject": subject,
        "object": destination,
        "desired_value": True,
    }
    if threshold is not None:
        condition["threshold"] = threshold
    return {
        "subgoal_id": "place_object",
        "atomic_task": "PickPlaceSinkToCounter",
        "policy_prompt": "Place the object.",
        "arguments": {},
        "termination_condition": condition,
    }


def test_inside_dispatches_object_receptacle_and_preserves_threshold(monkeypatch):
    from robocasa.utils import object_utils

    calls = []

    def check_receptacle(env, subject, destination, th=None):
        calls.append((subject, destination, th))
        return True

    monkeypatch.setattr(object_utils, "check_obj_in_receptacle", check_receptacle)
    monkeypatch.setattr(
        object_utils,
        "obj_inside_of",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fixture API must not handle object receptacles")
        ),
    )
    result = RuntimeAtomicTaskVerifier()(
        env=RelationEnv(),
        atomic_task_call=relation_call(
            "inside", "vegetable", "bowl", threshold=0.5
        ),
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "success"
    assert calls == [("vegetable", "bowl", 0.5)]
    relation = next(
        item
        for item in result["state_evidence"]
        if item.get("source") == "simulator_relation"
    )
    assert relation["relation_api"] == "check_obj_in_receptacle"
    assert relation["object_kind"] == "object"


def test_inside_dispatches_full_object_fixture_containment(monkeypatch):
    from robocasa.utils import object_utils

    calls = []

    def inside_fixture(env, subject, destination, partial_check):
        calls.append((subject, destination, partial_check))
        return True

    monkeypatch.setattr(object_utils, "obj_inside_of", inside_fixture)
    monkeypatch.setattr(
        object_utils,
        "check_obj_in_receptacle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("receptacle API must not handle fixtures")
        ),
    )
    result = RuntimeAtomicTaskVerifier()(
        env=RelationEnv(),
        atomic_task_call=relation_call("inside", "vegetable", "microwave"),
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "success"
    assert calls == [("vegetable", "microwave", False)]


def test_relation_rejects_wrong_subject_kind_without_retry():
    result = RuntimeAtomicTaskVerifier()(
        env=RelationEnv(),
        atomic_task_call=relation_call("inside", "microwave", "bowl"),
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "INVALID_TERMINATION_CONDITION"
    assert result["retryable"] is False
    assert "requires ['object']" in result["state_evidence"][0]["error"]


def test_relation_api_error_is_not_retried_as_policy_failure(monkeypatch):
    from robocasa.utils import object_utils

    monkeypatch.setattr(
        object_utils,
        "check_obj_in_receptacle",
        lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("bad API call")),
    )
    result = RuntimeAtomicTaskVerifier()(
        env=RelationEnv(),
        atomic_task_call=relation_call("inside", "vegetable", "bowl"),
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "VERIFIER_EVALUATION_ERROR"
    assert result["retryable"] is False
    assert any(
        item.get("source") == "object_utils" and "bad API call" in item["error"]
        for item in result["state_evidence"]
    )


def test_bare_global_predicate_state_cannot_satisfy_an_unrelated_goal(monkeypatch):
    from robocasa.utils import object_utils

    env = RelationEnv()
    env.atomic_state = {"inside": True}
    monkeypatch.setattr(
        object_utils, "check_obj_in_receptacle", lambda *args, **kwargs: False
    )
    result = RuntimeAtomicTaskVerifier()(
        env=env,
        atomic_task_call=relation_call("inside", "vegetable", "bowl"),
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "uncertain"
    assert result["goal_satisfied"] is False


def test_empty_boolean_condition_is_invalid_and_non_retryable():
    call = navigation_call()
    call["termination_condition"] = []
    result = RuntimeAtomicTaskVerifier()(
        env=NavigationEnv([1.0, 2.0, 0.0]),
        atomic_task_call=call,
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "INVALID_TERMINATION_CONDITION"
    assert result["retryable"] is False


def test_stability_count_does_not_leak_across_changed_conditions():
    env = DrawerAliasEnv()
    for fixture in env.fixtures.values():
        fixture.opened = True
    verifier = RuntimeAtomicTaskVerifier()

    first_call = drawer_alias_call("stack_4_main_group_1")
    second_call = drawer_alias_call("stack_4_main_group_2")
    for call in (first_call, second_call):
        call["metadata"] = {
            "skill_contract": {
                "verification": {"required_consecutive_successes": 2}
            }
        }

    first = verifier(
        env=env,
        atomic_task_call=first_call,
        observation={},
        step_index=10,
        info={},
    )
    changed = verifier(
        env=env,
        atomic_task_call=second_call,
        observation={},
        step_index=15,
        info={},
    )
    stable = verifier(
        env=env,
        atomic_task_call=second_call,
        observation={},
        step_index=20,
        info={},
    )

    assert first["status"] == "uncertain"
    assert changed["status"] == "uncertain"
    assert changed["state_evidence"][-1]["consecutive_successes"] == 1
    assert stable["status"] == "success"


def test_receptacle_count_includes_one_hop_stacked_objects(monkeypatch):
    from robocasa.utils import object_utils

    env = CountEnv({"ice_cube2"})
    env.check_contact = lambda first, second: {
        first,
        second,
    } == {env.objects["ice_cube2"], env.objects["ice_cube4"]}
    monkeypatch.setattr(
        object_utils,
        "check_obj_in_receptacle",
        lambda env, obj_name, receptacle_name, th: (
            obj_name in env.inside and receptacle_name == "glass_cup1" and th == 0.5
        ),
    )
    result = RuntimeAtomicTaskVerifier()(
        env=env,
        atomic_task_call=count_call(),
        observation={},
        step_index=20,
        info={},
    )

    assert result["status"] == "success"
    evidence = result["state_evidence"][-1]
    assert evidence["directly_inside"] == ["ice_cube2"]
    assert evidence["touching_inside"] == ["ice_cube4"]
    assert evidence["count"] == 2


class SinkFixture:
    name = "sink"
    nat_lang = "sink"

    def __init__(self, water_on):
        self.water_on = water_on

    def get_handle_state(self, env):
        assert self is env.fixtures["sink"]
        return {"water_on": self.water_on, "spout_ori": "center"}

    @staticmethod
    def get_state():
        return None


class SinkEnv:
    def __init__(self, water_on):
        self.fixtures = {"sink": SinkFixture(water_on)}
        self.objects = {}
        self.sim = object()


def sink_power_call(desired=True):
    return {
        "subgoal_id": "sink_faucet",
        "atomic_task": "TurnOnSinkFaucet" if desired else "TurnOffSinkFaucet",
        "policy_prompt": "Manipulate the sink faucet.",
        "arguments": {},
        "termination_condition": {
            "predicate": "powered",
            "subject": "sink",
            "desired_value": desired,
        },
    }


def test_sink_powered_uses_handle_state_instead_of_empty_get_state():
    result = RuntimeAtomicTaskVerifier()(
        env=SinkEnv(True),
        atomic_task_call=sink_power_call(True),
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "success"
    evidence = result["state_evidence"][-1]
    assert evidence["source"] == "fixture_handle_state"
    assert evidence["key"] == "water_on"


def test_native_simulator_state_cannot_be_overridden_by_stale_atomic_state(monkeypatch):
    from robocasa.utils import object_utils

    monkeypatch.setattr(
        object_utils, "check_obj_in_receptacle", lambda *args, **kwargs: False
    )
    result = RuntimeAtomicTaskVerifier()(
        env=RelationEnv(),
        atomic_task_call=relation_call("inside", "vegetable", "bowl"),
        observation={},
        step_index=10,
        info={"atomic_state": {"inside:vegetable:bowl": True}},
    )

    assert result["status"] == "uncertain"
    assert result["goal_satisfied"] is False
    assert not any(
        item.get("source") == "info.atomic_state"
        for item in result["state_evidence"]
    )


def test_unverifiable_boolean_branch_cannot_be_masked_by_false_branch():
    env = SinkEnv(False)
    call = sink_power_call(True)
    call["termination_condition"] = {
        "all_of": [
            {
                "predicate": "powered",
                "subject": "sink",
                "desired_value": True,
            },
            {
                "predicate": "pressed",
                "subject": "sink",
                "desired_value": True,
            },
        ]
    }
    result = RuntimeAtomicTaskVerifier()(
        env=env,
        atomic_task_call=call,
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "INVALID_TERMINATION_CONDITION"
    assert result["retryable"] is False


def test_boolean_operator_rejects_unrelated_extra_fields():
    call = navigation_call()
    call["termination_condition"] = {
        "all_of": [call["termination_condition"]],
        "subject": "sink_main_group",
    }
    result = RuntimeAtomicTaskVerifier()(
        env=NavigationEnv([1.0, 2.0, 0.0]),
        atomic_task_call=call,
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "INVALID_TERMINATION_CONDITION"


def test_alias_cache_is_invalidated_when_scene_candidates_change():
    resolved = ["stack_4_main_group_2", "stack_4_main_group_1"]
    requests = []

    def resolver(**request):
        requests.append(request)
        return resolved[len(requests) - 1]

    env = DrawerAliasEnv()
    verifier = RuntimeAtomicTaskVerifier(entity_alias_resolver=resolver)
    first = verifier(
        env=env,
        atomic_task_call=drawer_alias_call("left drawer"),
        observation={},
        step_index=10,
        info={},
    )
    assert first["status"] == "success"

    env.fixtures["stack_4_main_group_2"].nat_lang = "right drawer"
    second = verifier(
        env=env,
        atomic_task_call=drawer_alias_call("left drawer"),
        observation={},
        step_index=15,
        info={},
    )

    assert second["status"] == "uncertain"
    assert len(requests) == 2


def test_conflicting_object_and_destination_are_rejected():
    call = relation_call("inside", "vegetable", "bowl")
    call["termination_condition"]["destination"] = "microwave"
    result = RuntimeAtomicTaskVerifier()(
        env=RelationEnv(),
        atomic_task_call=call,
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "INVALID_TERMINATION_CONDITION"
    assert "object and destination must match" in result["state_evidence"][0]["error"]


def test_invalid_alias_resolution_is_not_cached():
    aliases = iter(("invented_drawer", "stack_4_main_group_2"))
    requests = []

    def resolver(**request):
        requests.append(request)
        return next(aliases)

    env = DrawerAliasEnv()
    verifier = RuntimeAtomicTaskVerifier(entity_alias_resolver=resolver)
    first = verifier(
        env=env,
        atomic_task_call=drawer_alias_call("left drawer"),
        observation={},
        step_index=10,
        info={},
    )
    second = verifier(
        env=env,
        atomic_task_call=drawer_alias_call("left drawer"),
        observation={},
        step_index=15,
        info={},
    )

    assert first["status"] == "failed"
    assert second["status"] == "success"
    assert len(requests) == 2


def test_nan_eef_position_cannot_satisfy_outside_fixture():
    env = HandoffEnv(eef_position=(np.nan, 0.5, 0.5))
    call = handoff_call(
        "eef_clear",
        {
            "predicate": "eef_outside_fixture",
            "subject": "microwave_main_group",
            "desired_value": True,
        },
    )
    result = RuntimeAtomicTaskVerifier()(
        env=env,
        atomic_task_call=call,
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "VERIFIER_EVALUATION_ERROR"
    assert result["retryable"] is False


def test_nan_navigation_pose_is_an_evaluation_error_not_a_retry():
    result = verify(NavigationEnv([np.nan, 2.0, 0.0]))

    assert result["status"] == "failed"
    assert result["failure_code"] == "VERIFIER_EVALUATION_ERROR"
    assert result["retryable"] is False


def test_nan_relation_result_cannot_be_coerced_to_success(monkeypatch):
    from robocasa.utils import object_utils

    monkeypatch.setattr(
        object_utils,
        "check_obj_in_receptacle",
        lambda *args, **kwargs: np.nan,
    )
    result = RuntimeAtomicTaskVerifier()(
        env=RelationEnv(),
        atomic_task_call=relation_call("inside", "vegetable", "bowl"),
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "VERIFIER_EVALUATION_ERROR"
    assert result["retryable"] is False


def test_nan_fixture_state_cannot_be_coerced_to_true():
    result = RuntimeAtomicTaskVerifier()(
        env=SinkEnv(np.nan),
        atomic_task_call=sink_power_call(True),
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "failed"
    assert result["goal_satisfied"] is False


def test_navigation_verifier_recomputes_pose_for_reference_object(monkeypatch):
    from robocasa.utils import env_utils

    env = NavigationEnv([1.0, 2.0, 0.0])
    env.objects["potato"] = object()
    env.object_placements = {"potato": (np.asarray([1.0, 2.0, 0.0]), None)}
    calls = []

    def compute_pose(raw_env, fixture, ref_object=None):
        calls.append((fixture, ref_object))
        return np.asarray([1.0, 2.0, 0.0]), np.asarray([0.0, 0.0, 0.0])

    monkeypatch.setattr(env_utils, "compute_robot_base_placement_pose", compute_pose)
    call = navigation_call()
    call["termination_condition"]["reference_object"] = "potato"
    result = RuntimeAtomicTaskVerifier()(
        env=env,
        atomic_task_call=call,
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "success"
    assert calls == [(env.fixtures["sink_main_group"], "potato")]
    evidence = result["state_evidence"][-1]
    assert evidence["reference_object"] == "potato"


def test_navigation_reference_object_without_live_placement_fails_fast():
    env = NavigationEnv([1.0, 2.0, 0.0])
    env.objects["potato"] = object()
    env.object_placements = {}
    call = navigation_call()
    call["termination_condition"]["reference_object"] = "potato"
    result = RuntimeAtomicTaskVerifier()(
        env=env,
        atomic_task_call=call,
        observation={},
        step_index=10,
        info={},
    )

    assert result["status"] == "failed"
    assert result["failure_code"] == "VERIFIER_EVALUATION_ERROR"
    assert result["retryable"] is False
    assert any(
        "has no live placement" in item.get("error", "")
        for item in result["state_evidence"]
    )
