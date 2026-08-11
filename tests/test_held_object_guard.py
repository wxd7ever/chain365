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


def test_destination_relation_unlocks_policy_release():
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

    assert guard.apply_action(opening, step_index=1)[6] == 1.0
    assert guard.observe(step_index=1) is None

    state["destination"] = True
    assert guard.apply_action(opening, step_index=2)[6] == -1.0
    state["holding"] = False
    assert guard.observe(step_index=2) is None
    assert guard.release_allowed is True
    assert guard.release_allowed_step == 2
    assert guard.dropped_step is None


class Entity:
    def __init__(self, name):
        self.name = name
        self.nat_lang = name


class GuardEnv:
    def __init__(self):
        self.objects = {"bowl": Entity("bowl")}
        self.fixtures = {"microwave_main_group": Entity("microwave")}
        self.sim = object()


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
    assert guard.object_alias == "bowl"
    assert guard.destination_condition["object"] == "microwave_main_group"
    assert guard.destination_condition["object_kind"] == "fixture"
