from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from atomic_task_schemas import (  # noqa: E402
    AtomicTaskCall,
    load_available_atomic_tasks,
    validate_atomic_task_call,
)


def valid_call(**overrides):
    value = {
        "subgoal_id": "g1",
        "atomic_task": "OpenMicrowave",
        "policy_prompt": "Open the microwave door.",
        "arguments": {"fixture_id": "microwave_1"},
        "termination_condition": {
            "predicate": "open",
            "subject": "microwave_1",
            "desired_value": True,
        },
    }
    value.update(overrides)
    return value


def test_allowlist_comes_from_current_dataset_metadata():
    tasks = load_available_atomic_tasks()
    assert "OpenMicrowave" in tasks
    assert "PickPlaceCounterToMicrowave" in tasks
    assert "OpenSingleDoor" not in tasks
    assert "PickPlaceObject" not in tasks


def test_valid_atomic_task_call():
    call = AtomicTaskCall.from_mapping(valid_call())
    validate_atomic_task_call(call)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"subgoal_id": ""}, ValueError),
        ({"atomic_task": ""}, ValueError),
        ({"arguments": []}, TypeError),
        ({"termination_condition": {}}, ValueError),
        ({"atomic_task": "OpenSingleDoor"}, ValueError),
        ({"policy_prompt": "  "}, ValueError),
    ],
)
def test_invalid_atomic_task_call(overrides, error):
    with pytest.raises(error):
        call = AtomicTaskCall.from_mapping(valid_call(**overrides))
        validate_atomic_task_call(call)


def test_missing_termination_condition():
    value = valid_call()
    del value["termination_condition"]
    with pytest.raises(ValueError, match="termination_condition"):
        AtomicTaskCall.from_mapping(value)
