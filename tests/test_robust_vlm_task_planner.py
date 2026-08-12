from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from atomic_task_schemas import AtomicTaskCall  # noqa: E402
from robust_vlm_task_planner import (  # noqa: E402
    _controlled_decomposition,
    _normalize_execution_plan,
    _validate_execution_plan,
    prepare_execution_plan,
)


def test_normalizer_resolves_aliases_defaults_condition_and_inserts_close():
    calls = [
        AtomicTaskCall.from_mapping(
            {
                "subgoal_id": "g1",
                "atomic_task": "PickPlaceCounterToMicrowave",
                "policy_prompt": "Put the food in the microwave.",
                "arguments": {},
                "termination_condition": {
                    "predicate": "inside",
                    "subject": "microwave",
                    "object": "chicken drumstick",
                },
            }
        ),
        AtomicTaskCall.from_mapping(
            {
                "subgoal_id": "g2",
                "atomic_task": "TurnOnMicrowave",
                "policy_prompt": "Press the microwave start button.",
                "arguments": {},
                "termination_condition": {
                    "predicate": "powered",
                    "subject": "microwave",
                },
            }
        ),
    ]
    context = {
        "fixtures": [
            {
                "alias": "microwave_main_group",
                "name": "microwave_main_group",
                "natural_name": "microwave",
                "type": "Microwave",
            }
        ],
        "objects": [
            {"alias": "obj", "name": "obj", "natural_name": "chicken drumstick"}
        ],
    }

    normalized, changes = _normalize_execution_plan(calls, context)

    assert [call.atomic_task for call in normalized] == [
        "PickPlaceCounterToMicrowave",
        "CloseMicrowave",
        "TurnOnMicrowave",
    ]
    first_conditions = normalized[0].termination_condition
    assert isinstance(first_conditions, list)
    assert first_conditions[0]["subject"] == "obj"
    assert first_conditions[0]["object"] == "microwave_main_group"
    assert first_conditions[0]["desired_value"] is True
    assert [item["predicate"] for item in first_conditions] == [
        "inside", "released", "eef_outside_receptacle"
    ]
    assert normalized[0].metadata["skill_contract"]["family"] == "object_transfer"
    assert normalized[1].termination_condition["subject"] == "microwave_main_group"
    assert any(change["type"] == "insert_prerequisite" for change in changes)
    _validate_execution_plan(normalized, context)


@pytest.mark.parametrize(
    ("arguments", "policy_prompt", "expected_source"),
    (
        (
            {"destination_name": "tray"},
            "Place sweet1 inside the receptacle.",
            "arguments.destination_name",
        ),
        (
            {},
            "Place sweet1 inside tray and release it.",
            "policy_prompt",
        ),
    ),
)
def test_normalizer_infers_unambiguous_missing_relation_destination(
    arguments, policy_prompt, expected_source
):
    calls = [
        AtomicTaskCall.from_mapping(
            {
                "subgoal_id": "4",
                "atomic_task": "PackDessert",
                "policy_prompt": policy_prompt,
                "arguments": arguments,
                "termination_condition": {
                    "predicate": "inside",
                    "subject": "sweet1",
                    "desired_value": True,
                },
            }
        )
    ]
    context = {
        "fixtures": [],
        "objects": [
            {"alias": "sweet1", "natural_name": "cake"},
            {"alias": "tray", "natural_name": "tray"},
        ],
    }

    normalized, changes = _normalize_execution_plan(calls, context)

    assert isinstance(normalized[0].termination_condition, list)
    goal = normalized[0].termination_condition[0]
    assert goal["object"] == "tray"
    inference = next(
        change
        for change in changes
        if change["type"] == "infer_relation_destination"
    )
    assert inference["source"] == expected_source
    _validate_execution_plan(normalized, context)



def test_normalizer_resolves_navigation_target_and_predicate():
    calls = [
        AtomicTaskCall.from_mapping(
            {
                "subgoal_id": "nav_sink",
                "atomic_task": "NavigateKitchen",
                "policy_prompt": "Navigate to the sink.",
                "arguments": {"fixture_name": "sink"},
                "termination_condition": {
                    "predicate": "near",
                    "subject": "sink",
                },
            }
        )
    ]
    context = {
        "fixtures": [
            {
                "alias": "sink_main_group",
                "name": "sink_main_group",
                "natural_name": "sink",
                "type": "Sink",
            }
        ],
        "objects": [],
    }

    normalized, changes = _normalize_execution_plan(calls, context)

    call = normalized[0]
    assert call.arguments == {
        "fixture_name": "sink",
        "fixture_id": "sink_main_group",
    }
    assert call.termination_condition == {
        "predicate": "navigation_pose",
        "subject": "sink_main_group",
        "desired_value": True,
    }
    assert any(
        change["type"] == "resolve_navigation_target" for change in changes
    )
    _validate_execution_plan(normalized, context)


def test_prepare_plan_defers_vlm_navigation_to_scheduler():
    calls = [
        AtomicTaskCall.from_mapping(
            {
                "subgoal_id": "nav_sink",
                "atomic_task": "NavigateKitchen",
                "policy_prompt": "Navigate to the sink.",
                "arguments": {"fixture_name": "sink"},
                "termination_condition": {
                    "predicate": "navigation_pose",
                    "subject": "sink_main_group",
                    "desired_value": True,
                },
            }
        ),
        AtomicTaskCall.from_mapping(
            {
                "subgoal_id": "turn_on_sink",
                "atomic_task": "TurnOnSinkFaucet",
                "policy_prompt": "Turn on the sink faucet.",
                "arguments": {"fixture_id": "sink_main_group"},
                "termination_condition": {
                    "predicate": "powered",
                    "subject": "sink_main_group",
                    "desired_value": True,
                },
            }
        ),
    ]
    context = {
        "fixtures": [
            {
                "alias": "sink_main_group",
                "natural_name": "sink",
                "type": "Sink",
            }
        ],
        "objects": [],
    }

    prepared, changes = prepare_execution_plan(calls, context)

    assert [call.atomic_task for call in prepared] == ["TurnOnSinkFaucet"]
    assert any(
        change["type"] == "defer_navigation_to_scheduler"
        and change["skipped_subgoal_id"] == "nav_sink"
        for change in changes
    )


def test_normalizer_overrides_relation_predicate_for_navigation_task():
    """A resolved NavigateKitchen target always has navigation-pose semantics."""

    calls = [
        AtomicTaskCall.from_mapping(
            {
                "subgoal_id": "1",
                "atomic_task": "NavigateKitchen",
                "policy_prompt": "Navigate to the sink.",
                "arguments": {},
                "termination_condition": {
                    "predicate": "inside",
                    "subject": "sink_main_group",
                    "desired_value": True,
                },
            }
        )
    ]
    context = {
        "fixtures": [
            {
                "alias": "sink_main_group",
                "name": "sink_main_group",
                "natural_name": "sink",
                "type": "Sink",
            }
        ],
        "objects": [],
    }

    normalized, changes = _normalize_execution_plan(calls, context)

    call = normalized[0]
    assert call.arguments == {
        "fixture_name": "sink",
        "fixture_id": "sink_main_group",
    }
    assert call.termination_condition == {
        "predicate": "navigation_pose",
        "subject": "sink_main_group",
        "desired_value": True,
    }
    assert any(
        change["type"] == "normalize_navigation_predicate"
        and change["from"] == "inside"
        for change in changes
    )
    _validate_execution_plan(normalized, context)


def test_validation_rejects_unrequested_fixture_family():
    calls = [
        AtomicTaskCall.from_mapping(
            {
                "subgoal_id": "wrong_appliance",
                "atomic_task": "TurnOnStove",
                "policy_prompt": "Turn on the stove.",
                "arguments": {},
                "termination_condition": {
                    "predicate": "powered",
                    "subject": "microwave_main_group",
                    "desired_value": True,
                },
            }
        )
    ]
    context = {
        "long_horizon_task": (
            "Turn on the microwave, navigate to the sink, and turn on the sink faucet."
        ),
        "fixtures": [
            {"alias": "microwave_main_group"},
            {"alias": "sink_main_group"},
        ],
        "objects": [],
    }


    with pytest.raises(ValueError, match="TurnOnStove"):
        _validate_execution_plan(calls, context)


def test_controlled_place_equal_ice_cubes_decomposes_to_four_counted_calls():
    context = {
        "env_name": "PlaceEqualIceCubes",
        "long_horizon_task": (
            "There are four ice cubes in the ice bowl. "
            "Place two ice cubes in each glass of water."
        ),
        "fixtures": [],
        "objects": [
            {"alias": alias}
            for alias in (
                "ice_bowl",
                "ice_cube1",
                "ice_cube2",
                "ice_cube3",
                "ice_cube4",
                "glass_cup1",
                "glass_cup2",
            )
        ],
    }

    controlled = _controlled_decomposition(
        context["long_horizon_task"], context
    )

    assert controlled is not None
    calls, provenance = controlled
    assert [call.atomic_task for call in calls] == ["MakeIcedCoffee"] * 4
    assert [call.subgoal_id for call in calls] == [
        "cup1_ice1",
        "cup1_ice2",
        "cup2_ice1",
        "cup2_ice2",
    ]
    assert [
        (call.termination_condition["subject"], call.termination_condition["desired_value"])
        for call in calls
    ] == [
        ("glass_cup1", 1),
        ("glass_cup1", 2),
        ("glass_cup2", 1),
        ("glass_cup2", 2),
    ]
    assert all(
        call.termination_condition["predicate"] == "receptacle_count"
        for call in calls
    )
    assert provenance["rule"] == "PlaceEqualIceCubes_to_4x_MakeIcedCoffee"
    _validate_execution_plan(calls, context)


    prepared, changes = prepare_execution_plan(calls, context)
    assert all(isinstance(call.termination_condition, list) for call in prepared)
    assert all(
        [item["predicate"] for item in call.termination_condition]
        == ["receptacle_count", "released", "gripper_far"]
        for call in prepared
    )
    assert all(
        call.metadata["skill_contract"]["verification"][
            "required_consecutive_successes"
        ]
        == 2
        for call in prepared
    )
    assert len(
        [change for change in changes if change["type"] == "apply_skill_contract"]
    ) == 4



def test_controlled_retrieve_ice_tray_uses_scene_roles_and_preserves_grasp():
    context = {
        "env_name": "RetrieveIceTray",
        "long_horizon_task": (
            "Open the freezer door, pick up the ice cube tray from the freezer, "
            "and place it on the dining counter."
        ),
        "fixture_refs": {
            "fridge": "fridgesidebyside_right_group_1",
            "dining_counter": "island_island_group_1",
        },
        "fixtures": [
            {"alias": "fridgesidebyside_right_group_1"},
            {"alias": "island_island_group_1"},
        ],
        "objects": [{"alias": "ice_cube_tray"}],
    }

    controlled = _controlled_decomposition(context["long_horizon_task"], context)

    assert controlled is not None
    calls, provenance = controlled
    assert [call.atomic_task for call in calls] == [
        "OpenFridge",
        "PickPlaceFridgeShelfToDrawer",
        "PickPlaceCabinetToCounter",
    ]
    assert calls[1].termination_condition["predicate"] == "holding"
    assert calls[2].termination_condition["object"] == "island_island_group_1"
    assert provenance["rule"] == (
        "RetrieveIceTray_to_open_grasp_place_runtime_navigation"
    )

    prepared, changes = prepare_execution_plan(calls, context)
    assert [call.atomic_task for call in prepared] == [
        "OpenFridge",
        "PickPlaceFridgeShelfToDrawer",
        "PickPlaceCabinetToCounter",
    ]
    placement_conditions = prepared[2].termination_condition
    assert [condition["predicate"] for condition in placement_conditions] == [
        "on",
        "released",
        "eef_outside_receptacle",
    ]
    assert prepared[1].metadata["policy_proxy"] == "fridge_source_grasp"
    assert prepared[2].metadata["policy_proxy"] == "held_object_to_counter"
    assert len(
        [change for change in changes if change["type"] == "apply_skill_contract"]
    ) == 3


def test_normalizer_skips_navigation_to_initialized_object():
    calls = [
        AtomicTaskCall.from_mapping(
            {
                "subgoal_id": "nav_counter",
                "atomic_task": "NavigateKitchen",
                "policy_prompt": (
                    "Navigate to the counter where the blender jug is located."
                ),
                "arguments": {"fixture_name": "counter"},
                "termination_condition": {
                    "predicate": "navigation_pose",
                    "subject": "counter_main_group",
                    "desired_value": True,
                },
            }
        ),
        AtomicTaskCall.from_mapping(
            {
                "subgoal_id": "place_jug",
                "atomic_task": "PickPlaceCounterToSink",
                "policy_prompt": "Pick the blender jug and place it in the sink.",
                "arguments": {"object_name": "blender jug"},
                "termination_condition": {
                    "predicate": "inside",
                    "subject": "blender_jug",
                    "object": "sink_main_group",
                    "desired_value": True,
                },
            }
        ),
    ]
    context = {
        "robot_initial_state": {
            "fixture": "counter_main_group",
            "object": "blender_jug",
        },
        "fixtures": [
            {"alias": "counter_main_group", "natural_name": "counter"},
            {"alias": "sink_main_group", "natural_name": "sink"},
        ],
        "objects": [
            {"alias": "blender_jug", "natural_name": "blender jug"},
        ],
    }

    normalized, changes = prepare_execution_plan(calls, context)

    assert [call.atomic_task for call in normalized] == ["PickPlaceCounterToSink"]
    skipped = next(
        change
        for change in changes
        if change["type"] == "skip_redundant_initial_navigation"
    )
    assert skipped["fixture"] == "counter_main_group"
    assert skipped["object"] == "blender_jug"
