from __future__ import annotations

import sys
from pathlib import Path


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from atomic_task_schemas import AtomicTaskCall, load_available_atomic_tasks  # noqa: E402
from skill_contract_registry import (  # noqa: E402
    apply_skill_contract,
    load_skill_contract_registry,
)


def test_registry_has_complete_contract_for_every_atomic_task():
    registry = load_skill_contract_registry()

    assert set(registry) == load_available_atomic_tasks()
    for atomic_task, contract in registry.items():
        assert contract.atomic_task == atomic_task
        assert contract.family
        assert contract.precondition
        assert contract.goal == {"source": "atomic_task_call.termination_condition"}
        assert isinstance(contract.handoff, tuple)
        assert contract.failure
        assert contract.recovery
        assert contract.required_consecutive_successes == 2


def test_object_transfer_contract_adds_safe_handoff_and_is_idempotent():
    call = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "place_in_microwave",
            "atomic_task": "PickPlaceCounterToMicrowave",
            "policy_prompt": "Place the object in the microwave.",
            "arguments": {},
            "termination_condition": {
                "predicate": "inside",
                "subject": "obj",
                "object": "microwave_main_group",
                "desired_value": True,
            },
        }
    )
    context = {
        "fixtures": [{"alias": "microwave_main_group"}],
        "objects": [{"alias": "obj"}],
    }

    enriched, changes = apply_skill_contract(call, context)

    assert isinstance(enriched.termination_condition, list)
    assert [item["predicate"] for item in enriched.termination_condition] == [
        "inside",
        "released",
        "gripper_far",
        "eef_outside_fixture",
    ]
    contract = enriched.metadata["skill_contract"]
    assert contract["precondition"]
    assert contract["goal"]["conditions"] == [call.termination_condition]
    assert contract["handoff"]["conditions"] == enriched.termination_condition[1:]
    assert contract["failure"]
    assert contract["recovery"]
    assert contract["verification"]["required_consecutive_successes"] == 2
    assert changes[0]["type"] == "apply_skill_contract"

    enriched_again, repeated_changes = apply_skill_contract(enriched, context)
    assert enriched_again.termination_condition == enriched.termination_condition
    assert repeated_changes == []


def test_iced_coffee_contract_releases_any_matching_cube_then_retracts_from_cup():
    call = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "cup1_ice1",
            "atomic_task": "MakeIcedCoffee",
            "policy_prompt": "Put one ice cube in glass cup 1.",
            "arguments": {},
            "termination_condition": {
                "predicate": "receptacle_count",
                "subject": "glass_cup1",
                "object_prefix": "ice_cube",
                "desired_value": 1,
            },
        }
    )

    enriched, _ = apply_skill_contract(
        call,
        {
            "fixtures": [],
            "objects": [
                {"alias": "ice_cube1"},
                {"alias": "ice_cube2"},
                {"alias": "glass_cup1"},
            ],
        },
    )

    assert isinstance(enriched.termination_condition, list)
    assert enriched.termination_condition[1] == {
        "predicate": "released",
        "subject": "ice_cube",
        "object_prefix": "ice_cube",
        "desired_value": True,
    }
    assert enriched.termination_condition[2]["predicate"] == "gripper_far"
    assert enriched.termination_condition[2]["subject"] == "glass_cup1"

