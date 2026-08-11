from __future__ import annotations

import sys
from pathlib import Path


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from atomic_task_schemas import AtomicTaskCall  # noqa: E402
from grounding import GroundingResult, RoboCasaGrounder  # noqa: E402


def test_grounder_uses_relaxed_default_position_threshold():
    assert RoboCasaGrounder().position_threshold_m == 0.62


def test_dynamic_navigation_call_preserves_held_object_and_reference():
    operation = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "place_tray",
            "atomic_task": "PickPlaceCabinetToCounter",
            "policy_prompt": "Place the ice cube tray on the dining counter.",
            "arguments": {"object_id": "ice_cube_tray"},
            "termination_condition": {
                "predicate": "on",
                "subject": "ice_cube_tray",
                "object": "dining_counter",
                "desired_value": True,
            },
        }
    )
    result = GroundingResult(
        grounded=False,
        status="navigation_required",
        target_entity_alias="dining_counter",
        target_entity_kind="fixture",
        target_fixture_alias="dining_counter",
        reference_object_alias="serving_tray",
        held_object_alias="ice_cube_tray",
        target_mode="held_object_destination",
    )
    grounder = RoboCasaGrounder(
        scene_context={
            "fixtures": [
                {
                    "alias": "dining_counter",
                    "natural_name": "dining counter",
                }
            ],
            "objects": [
                {"alias": "ice_cube_tray"},
                {"alias": "serving_tray"},
            ],
        }
    )

    navigation = grounder.build_navigation_call(
        operation_call=operation,
        grounding_result=result,
    )

    assert navigation.atomic_task == "NavigateKitchen"
    assert navigation.arguments["fixture_id"] == "dining_counter"
    assert navigation.arguments["reference_object_id"] == "serving_tray"
    assert navigation.arguments["held_object_id"] == "ice_cube_tray"
    assert [item["predicate"] for item in navigation.termination_condition] == [
        "navigation_pose",
        "holding",
    ]
    assert navigation.termination_condition[0]["reference_object"] == "serving_tray"
    assert navigation.metadata["inserted_by"] == "scheduler_grounding"
