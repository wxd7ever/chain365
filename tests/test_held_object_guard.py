from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from atomic_task_schemas import AtomicTaskCall  # noqa: E402
from held_object_guard import HeldObjectGuard, build_held_object_guard  # noqa: E402


def test_confirmed_grasp_is_latched_and_drop_fails_early():
    state = {"holding": False, "destination": False}
    guard = HeldObjectGuard(
        enabled=True,
        object_alias="bowl",
        destination_condition={
            "predicate": "inside",
            "subject": "bowl",
            "object": "microwave",
        },
        holding_checker=lambda: state["holding"],
        destination_checker=lambda: state["destination"],
        hold_confirmation_steps=2,
        drop_confirmation_steps=2,
        action_override_enabled=True,
    )
    guard.start()
    opening = np.zeros(12, dtype=np.float32)
    opening[6] = -1.0

    state["holding"] = True
    assert guard.apply_action(opening, step_index=1)[6] == -1.0
    assert guard.observe(step_index=1) is None
    assert guard.apply_action(opening, step_index=2)[6] == -1.0
    assert guard.observe(step_index=2) is None
    assert guard.latched is True

    state["holding"] = False
    assert guard.apply_action(opening, step_index=3)[6] == 1.0
    assert guard.observe(step_index=3) is None
    assert guard.apply_action(opening, step_index=4)[6] == 1.0
    failure = guard.observe(step_index=4)

    assert failure["status"] == "failed"
    assert failure["failure_code"] == "OBJECT_DROPPED"
    assert failure["retryable"] is True
    assert guard.dropped_step == 4
    assert guard.gripper_override_count == 2

    # A continuing rollout receives only one failure for a continuous drop.
    assert guard.observe(step_index=5) is None
    assert guard.drop_event_count == 1

    # Reacquiring the object clears the active drop episode. A later drop can
    # be reported again without changing the first dropped_step diagnostic.
    state["holding"] = True
    assert guard.observe(step_index=6) is None
    assert guard.drop_active is False
    state["holding"] = False
    assert guard.observe(step_index=7) is None
    second_failure = guard.observe(step_index=8)
    assert second_failure["failure_code"] == "OBJECT_DROPPED"
    assert guard.dropped_step == 4
    assert guard.drop_event_count == 2


def test_pick_guard_waits_for_holding_and_lift_before_latching():
    state = {"holding": False, "lift_delta": 0.0}
    guard = HeldObjectGuard(
        enabled=True,
        object_alias="bowl",
        holding_checker=lambda: state["holding"],
        lift_delta_checker=lambda: state["lift_delta"],
        minimum_lift_delta_m=0.04,
        hold_confirmation_steps=2,
        drop_confirmation_steps=2,
        action_override_enabled=True,
    )
    guard.start()
    opening = np.zeros(12, dtype=np.float32)
    opening[6] = -1.0

    state["holding"] = True
    state["lift_delta"] = 0.03
    for step in (1, 2):
        assert guard.apply_action(opening, step_index=step)[6] == -1.0
        assert guard.observe(step_index=step) is None
    assert guard.latched is False
    assert guard.holding_consecutive == 0

    state["lift_delta"] = 0.05
    for step in (3, 4):
        assert guard.apply_action(opening, step_index=step)[6] == -1.0
        assert guard.observe(step_index=step) is None
    assert guard.latched is True
    assert guard.latched_step == 4
    assert guard.apply_action(opening, step_index=5)[6] == 1.0
    assert guard.trace[-1]["lift_ready"] is True
    assert guard.trace[-1]["lift_delta_m"] == 0.05


def test_monitor_only_tracks_destination_without_overriding_policy_action():
    state = {"holding": True, "destination": False}
    guard = HeldObjectGuard(
        enabled=True,
        object_alias="bowl",
        destination_condition={
            "predicate": "inside",
            "subject": "bowl",
            "object": "microwave",
        },
        holding_checker=lambda: state["holding"],
        destination_checker=lambda: state["destination"],
    )
    guard.start()
    opening = np.zeros(12, dtype=np.float32)
    opening[6] = -1.0

    assert guard.apply_action(opening, step_index=1)[6] == -1.0
    assert guard.observe(step_index=1) is None

    state["destination"] = True
    assert guard.apply_action(opening, step_index=2)[6] == -1.0
    state["holding"] = False
    assert guard.observe(step_index=2) is None
    assert guard.release_allowed is True
    assert guard.release_allowed_step == 2
    assert guard.action_override_enabled is False
    assert guard.gripper_override_count == 0
    assert guard.dropped_step is None


class Entity:
    def __init__(self, name):
        self.name = name
        self.nat_lang = name


class GuardEnv:
    def __init__(self):
        self.objects = {"bowl": Entity("bowl")}
        self.fixtures = {"microwave_main_group": Entity("microwave")}
        self.gripper = object()
        self.robots = [
            type("GuardRobot", (), {"gripper": {"right": self.gripper}})()
        ]
        self.grasped = False
        self.sim = object()

    def _check_grasp(self, gripper, obj):
        return bool(
            self.grasped
            and gripper is self.gripper
            and obj is self.objects["bowl"]
        )


def test_factory_enables_lift_gate_only_for_pick_object():
    env = GuardEnv()
    env.obj_body_id = {"bowl": 0}
    data = type(
        "GuardData",
        (),
        {"body_xpos": np.asarray([[0.0, 0.0, 0.80]], dtype=float)},
    )()
    env.sim = type("GuardSim", (), {"data": data})()
    call = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "pick_bowl",
            "atomic_task": "PickObject",
            "policy_prompt": "Pick up the bowl.",
            "arguments": {"object_id": "bowl"},
            "termination_condition": {
                "predicate": "holding",
                "subject": "bowl",
                "desired_value": True,
            },
        }
    )

    guard = build_held_object_guard(
        env=env,
        atomic_task_call=call,
        pick_min_lift_delta_m=0.04,
    )

    assert guard.minimum_lift_delta_m == 0.04
    assert guard.activation_requires_verifier_success is True
    assert guard._lift_delta() == 0.0
    env.sim.data.body_xpos[0, 2] = 0.85
    assert np.isclose(guard._lift_delta(), 0.05)



def test_support_pick_guard_inherits_verifier_success_without_finger_gate(monkeypatch):
    from robocasa.utils import object_utils

    class CounterFixture:
        name = "counter_main_group"
        nat_lang = "counter"

    env = GuardEnv()
    env.support = CounterFixture()
    env.fixtures = {"counter_main_group": env.support}
    env.support_contact = True
    env.obj_body_id = {"bowl": 0}
    data = type(
        "SupportGuardData",
        (),
        {"body_xpos": np.asarray([[0.0, 0.0, 0.80]], dtype=float)},
    )()
    env.sim = type("SupportGuardSim", (), {"data": data})()
    monkeypatch.setattr(
        object_utils,
        "check_obj_fixture_contact",
        lambda raw_env, object_alias, fixture: (
            raw_env.support_contact and fixture is raw_env.support
        ),
    )
    call = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "pick_bowl",
            "atomic_task": "PickObject",
            "policy_prompt": "Pick up the bowl.",
            "arguments": {"object_id": "bowl"},
            "termination_condition": {
                "predicate": "holding",
                "subject": "bowl",
                "desired_value": True,
            },
        }
    )
    guard = build_held_object_guard(
        env=env,
        atomic_task_call=call,
        pick_min_lift_delta_m=0.04,
    )
    guard.start()
    assert guard.pick_readiness_mode == "support_surface_departure"
    assert guard.latched is False

    env.support_contact = False
    verifier_success = {
        "status": "success",
        "goal_satisfied": True,
        "failure_code": None,
        "retryable": False,
        "state_evidence": [],
    }
    assert env.grasped is False
    assert guard.confirm_verifier_result(verifier_success, step_index=5) is True
    assert guard.latched is True
    assert guard.events[-1]["two_finger_contact"] is False
    assert guard.events[-1]["two_finger_contact_required"] is False


def test_pick_guard_latches_after_cabinet_horizontal_extraction(monkeypatch):
    from robocasa.utils import object_utils

    state = {"holding": False, "inside": True}

    class CabinetFixture:
        name = "cabinet_main_group"
        nat_lang = "cabinet"

    env = GuardEnv()
    env.fixtures = {"cabinet_main_group": CabinetFixture()}
    env.obj_body_id = {"bowl": 0}
    data = type(
        "CabinetGuardData",
        (),
        {"body_xpos": np.asarray([[0.0, 0.0, 1.45]], dtype=float)},
    )()
    env.sim = type("CabinetGuardSim", (), {"data": data})()
    monkeypatch.setattr(
        object_utils,
        "check_obj_grasped",
        lambda *args, **kwargs: state["holding"],
    )
    monkeypatch.setattr(
        object_utils,
        "obj_inside_of",
        lambda *args, **kwargs: state["inside"],
    )
    call = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "pick_bowl",
            "atomic_task": "PickObject",
            "policy_prompt": "Pick up the bowl from the cabinet.",
            "arguments": {
                "object_id": "bowl",
                "source_name": "cabinet_main_group",
            },
            "termination_condition": {
                "predicate": "holding",
                "subject": "bowl",
                "desired_value": True,
            },
        }
    )
    guard = build_held_object_guard(
        env=env,
        atomic_task_call=call,
        pick_min_lift_delta_m=0.04,
        hold_confirmation_steps=2,
    )
    guard.start()
    assert guard.pick_readiness_mode == "cabinet_or_drawer_extraction"
    assert guard.latched is False

    state["holding"] = True
    state["inside"] = False
    env.sim.data.body_xpos[0, 0] = 0.05
    assert guard.observe(step_index=1) is None
    assert guard.observe(step_index=2) is None
    assert guard.latched is False
    verifier_success = {
        "status": "success",
        "goal_satisfied": True,
        "failure_code": None,
        "retryable": False,
        "state_evidence": [],
    }
    assert guard.confirm_verifier_result(
        verifier_success, step_index=2
    ) is False
    assert guard.latched is False
    env.grasped = True
    assert guard.confirm_verifier_result(
        verifier_success, step_index=3
    ) is True
    assert guard.latched is True
    assert guard.events[-1]["event"] == "verifier_confirmed_grasp_latched"
    assert guard.trace[-1]["lift_delta_m"] == 0.0
    assert guard.trace[-1]["lift_ready"] is True
    assert (
        guard.to_dict()["pick_readiness_mode"]
        == "cabinet_or_drawer_extraction"
    )


def test_factory_resolves_pickplace_object_and_destination():
    call = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "place_bowl",
            "atomic_task": "PickPlaceCounterToMicrowave",
            "policy_prompt": "Place the bowl in the microwave.",
            "arguments": {"object_name": "bowl"},
            "termination_condition": {
                "predicate": "inside",
                "subject": "bowl",
                "object": "microwave_main_group",
                "desired_value": True,
            },
        }
    )

    guard = build_held_object_guard(env=GuardEnv(), atomic_task_call=call)

    assert guard.enabled is True
    assert guard.action_override_enabled is False
    assert guard.object_alias == "bowl"
    assert guard.destination_condition["object"] == "microwave_main_group"
    assert guard.destination_condition["object_kind"] == "fixture"


def test_pick_guard_binds_to_interchangeable_object_selected_by_verifier(monkeypatch):
    from robocasa.utils import object_utils

    class EquivalentGuardEnv:
        def __init__(self):
            self.fixtures = {}
            self.objects = {"ice_cube1": object(), "ice_cube2": object()}
            self.obj_body_id = {"ice_cube1": 0, "ice_cube2": 1}
            data = type(
                "EquivalentGuardData",
                (),
                {"body_xpos": np.asarray([[0.0, 0.0, 0.8], [0.05, 0.0, 0.8]])},
            )()
            self.sim = type("EquivalentGuardSim", (), {"data": data})()
            self.gripper = object()
            self.robots = [
                type("EquivalentRobot", (), {"gripper": {"right": self.gripper}})()
            ]

        def check_contact(self, first, second):
            return False

        def _check_grasp(self, gripper, obj):
            return gripper is self.gripper and obj is self.objects["ice_cube2"]

    monkeypatch.setattr(
        object_utils, "check_obj_in_receptacle", lambda *args, **kwargs: False
    )
    env = EquivalentGuardEnv()
    call = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "pick_ice_cube",
            "atomic_task": "PickObject",
            "policy_prompt": "Pick up an ice cube.",
            "arguments": {"object_name": "ice_cube1"},
            "termination_condition": {
                "predicate": "holding", "subject": "ice_cube1",
                "desired_value": True,
            },
            "metadata": {
                "interchangeable_object_canonical": "ice_cube1",
                "interchangeable_object_candidates": ["ice_cube1", "ice_cube2"],
            },
        }
    )
    guard = build_held_object_guard(
        env=env, atomic_task_call=call, pick_min_lift_delta_m=0.04
    )
    guard.start()
    env.sim.data.body_xpos[1, 2] += 0.05
    verified = {
        "status": "success",
        "goal_satisfied": True,
        "state_evidence": [
            {
                "source": "interchangeable_object_pick",
                "selected_object_alias": "ice_cube2",
            }
        ],
    }

    assert guard.confirm_verifier_result(verified, step_index=5) is True
    assert guard.object_alias == "ice_cube2"
    assert guard.latched is True
    assert any(
        event.get("event") == "guard_bound_to_verified_object"
        for event in guard.events
    )
