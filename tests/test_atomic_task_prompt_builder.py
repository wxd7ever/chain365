from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from atomic_task_prompt_builder import (  # noqa: E402
    build_atomic_task_prompt,
    clean_entity_name,
)


def call(**overrides):
    value = {
        "subgoal_id": "g1",
        "atomic_task": "OpenMicrowave",
        "arguments": {"fixture_id": "microwave_2", "fixture_name": "microwave"},
        "termination_condition": {"predicate": "open", "subject": "microwave_2"},
    }
    value.update(overrides)
    return value


def test_explicit_policy_prompt_has_priority():
    value = call(policy_prompt="  Open the microwave door now.  ")
    assert build_atomic_task_prompt(value) == "Open the microwave door now."


def test_pick_object_prompt_is_reduced_to_the_action_only():
    value = call(
        atomic_task="PickObject",
        policy_prompt=(
            "Pick up the lemon wedge from the sink, lift it clear, retract to a "
            "stable carrying pose, and keep holding it."
        ),
        arguments={
            "object_id": "vegetable",
            "object_name": "vegetable",
            "source_name": "sink",
        },
        termination_condition={
            "predicate": "holding",
            "subject": "vegetable",
            "desired_value": True,
        },
    )
    assert build_atomic_task_prompt(value) == "Pick up the lemon wedge from the sink."


def test_place_object_prompt_is_reduced_to_the_action_only():
    value = call(
        atomic_task="PlaceObject",
        policy_prompt=(
            "Place the held lemon wedge inside the bowl, release it, and move "
            "the gripper away."
        ),
        arguments={
            "object_id": "vegetable",
            "object_name": "vegetable",
            "destination_id": "bowl",
            "destination_name": "bowl",
            "destination_preposition": "inside",
        },
        termination_condition={
            "predicate": "inside",
            "subject": "vegetable",
            "object": "bowl",
        },
    )
    assert build_atomic_task_prompt(value) == "Place the lemon wedge inside the bowl."


def test_legacy_handoff_prompt_style_restores_august_pick_place_language():
    pick = call(
        atomic_task="PickObject",
        policy_prompt="Pick up the lemon wedge from the sink.",
        arguments={
            "object_id": "vegetable",
            "object_name": "lemon wedge",
            "source_name": "sink",
        },
        termination_condition={
            "predicate": "holding",
            "subject": "vegetable",
        },
    )
    place = call(
        atomic_task="PlaceObject",
        policy_prompt="Place the lemon wedge inside the bowl.",
        arguments={
            "object_id": "vegetable",
            "object_name": "lemon wedge",
            "destination_id": "bowl",
            "destination_name": "bowl",
            "destination_preposition": "inside",
        },
        termination_condition={
            "predicate": "inside",
            "subject": "vegetable",
            "object": "bowl",
        },
    )

    assert build_atomic_task_prompt(pick, prompt_style="legacy_handoff") == (
        "Pick up the lemon wedge from the sink and keep holding it."
    )
    assert build_atomic_task_prompt(place, prompt_style="legacy_handoff") == (
        "Place the held lemon wedge inside the bowl, release it, and move the "
        "gripper away."
    )


def test_registry_template_and_argument_replacement():
    value = call(
        atomic_task="PickPlaceCounterToMicrowave",
        arguments={"object_id": "apple_1"},
        termination_condition={
            "predicate": "inside",
            "subject": "apple_1",
            "object": "microwave_1",
        },
    )
    assert build_atomic_task_prompt(value) == (
        "Pick the apple from the counter and place it in the microwave."
    )


def test_missing_template_argument_is_clear():
    with pytest.raises(ValueError, match="drawer_side"):
        build_atomic_task_prompt(
            call(atomic_task="OpenDrawer"),
            prompt_registry={"OpenDrawer": ["Open the {drawer_side} drawer."]},
        )


def test_unknown_atomic_task_is_rejected_before_prompt_lookup():
    with pytest.raises(ValueError, match="Unknown RoboCasa atomic task"):
        build_atomic_task_prompt(call(atomic_task="PickPlaceObject"))


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError, match="policy_prompt"):
        build_atomic_task_prompt(call(policy_prompt=" "))


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("apple_1", "apple"),
        ("microwave_2", "microwave"),
        ("fridge_drawer_1", "fridge drawer"),
    ],
)
def test_entity_id_cleaning(raw, cleaned):
    assert clean_entity_name(raw) == cleaned
