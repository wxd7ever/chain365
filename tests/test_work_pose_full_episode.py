from __future__ import annotations

import pytest

from robocasa.atomic_task_schemas import validate_atomic_task_call
from robocasa.scripts.work_pose.eval_work_pose_full_episode import (
    aggregate,
    fixture_call,
    microwave_alias,
    select_episode_workflows,
    target_degraded_pose,
)


def workflow_data(num_episodes: int = 2):
    records = []
    samples = []
    for episode_index in range(num_episodes):
        for stage_index in (0, 1, 2, 4):
            record_index = len(records)
            records.append(
                {
                    "episode_index": episode_index,
                    "segment": {"subtask_idx": stage_index},
                }
            )
            for rank in range(2):
                samples.append(
                    {
                        "sample_id": (
                            f"ep{episode_index:06d}_s{stage_index:02d}_"
                            f"mild_{rank:02d}"
                        ),
                        "record_index": record_index,
                        "difficulty": "mild",
                        "valid": True,
                    }
                )
    return records, samples


def test_select_episode_workflows_counts_complete_episodes_not_samples():
    records, samples = workflow_data(3)

    workflows = select_episode_workflows(
        samples=samples,
        records=records,
        difficulty="mild",
        sample_rank=1,
        episode_start=1,
        episode_count=2,
    )

    assert [value["episode_index"] for value in workflows] == [1, 2]
    assert all(
        set(value["stages"]) == {0, 1, 2, 4} for value in workflows
    )
    assert workflows[0]["stages"][0]["sample"]["sample_id"].endswith("_01")


def test_select_episode_workflows_rejects_incomplete_manifest():
    records, samples = workflow_data(1)
    samples = [
        sample
        for sample in samples
        if "_s04_" not in sample["sample_id"]
    ]

    with pytest.raises(ValueError, match="complete episodes"):
        select_episode_workflows(
            samples=samples,
            records=records,
            difficulty="mild",
            sample_rank=0,
            episode_start=0,
            episode_count=1,
        )


@pytest.mark.parametrize(
    ("stage_index", "task", "predicate"),
    [
        (5, "CloseMicrowave", "closed"),
        (6, "TurnOnMicrowave", "powered"),
    ],
)
def test_fixture_calls_are_valid(stage_index, task, predicate):
    call = fixture_call(stage_index, "microwave_left_group_1")

    validate_atomic_task_call(call)
    assert call.atomic_task == task
    assert call.termination_condition["predicate"] == predicate
    assert call.termination_condition["subject"] == "microwave_left_group_1"


def test_microwave_alias_prefers_exact_alias():
    context = {
        "fixtures": [
            {
                "alias": "microwave_left_group_1",
                "name": "microwave",
                "natural_name": "microwave",
            },
            {
                "alias": "microwave",
                "name": "microwave",
                "natural_name": "microwave",
            },
        ]
    }

    assert microwave_alias(context) == "microwave"


def test_target_degraded_pose_prefers_physically_reached_pose():
    sample = {
        "target_degraded_base_pose": {"position": [1, 2, 3]},
        "base_movement": {
            "final_pose": {
                "position": [4, 5, 6],
                "quaternion_xyzw": [0, 0, 0, 1],
            }
        },
    }

    assert target_degraded_pose(sample)["position"] == [4, 5, 6]


def test_full_episode_aggregate_reports_stage_and_task_success():
    results = [
        {
            "success": True,
            "operation_results": [
                {"stage_index": 0, "atomic_task": "PickObject", "success": True},
                {"stage_index": 1, "atomic_task": "PlaceObject", "success": True},
            ],
        },
        {
            "success": False,
            "operation_results": [
                {"stage_index": 0, "atomic_task": "PickObject", "success": False}
            ],
        },
    ]

    summary = aggregate(results)

    assert summary["num_episodes"] == 2
    assert summary["num_success"] == 1
    assert summary["full_task_success_rate"] == 0.5
    assert summary["stage_metrics"]["stage_00_PickObject"] == {
        "attempts": 2,
        "successes": 1,
        "success_rate": 0.5,
    }
