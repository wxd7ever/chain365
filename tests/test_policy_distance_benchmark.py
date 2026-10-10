import math

import pytest

from robocasa.eval_policy_distance_benchmark import (
    desired_start_pose,
    pose_on_target_ray,
    summarize_trials,
)


def test_desired_start_pose_has_exact_distance_and_faces_target():
    pose = desired_start_pose([0.0, 0.0], [-3.0, 0.0], 1.8)

    assert pose["x"] == pytest.approx(-1.8)
    assert pose["y"] == pytest.approx(0.0)
    assert pose["yaw"] == pytest.approx(0.0)
    assert math.hypot(pose["x"], pose["y"]) == pytest.approx(1.8)


def test_desired_start_pose_preserves_reference_ray():
    pose = desired_start_pose([1.0, 2.0], [4.0, 6.0], 2.0)

    assert math.hypot(pose["x"] - 1.0, pose["y"] - 2.0) == pytest.approx(2.0)
    assert pose["yaw"] == pytest.approx(
        math.atan2(2.0 - pose["y"], 1.0 - pose["x"])
    )


def test_pose_on_target_ray_varies_only_distance():
    near = pose_on_target_ray([1.0, -2.0], 0.8, 30.0)
    far = pose_on_target_ray([1.0, -2.0], 2.8, 30.0)

    assert math.hypot(near["x"] - 1.0, near["y"] + 2.0) == pytest.approx(0.8)
    assert math.hypot(far["x"] - 1.0, far["y"] + 2.0) == pytest.approx(2.8)
    assert near["yaw"] == pytest.approx(far["yaw"])


def test_desired_start_pose_rejects_degenerate_reference():
    with pytest.raises(ValueError, match="coincides"):
        desired_start_pose([1.0, 2.0], [1.0, 2.0], 1.0)


def test_summary_separates_invalid_starts_from_policy_failures():
    trials = [
        {
            "requested_distance_m": 0.8,
            "eligible": True,
            "executed": True,
            "success": True,
            "entered_operation_region": True,
        },
        {
            "requested_distance_m": 0.8,
            "eligible": True,
            "executed": True,
            "success": False,
            "entered_operation_region": True,
        },
        {
            "requested_distance_m": 1.8,
            "eligible": False,
            "executed": False,
            "success": False,
            "entered_operation_region": False,
        },
    ]

    summary = summarize_trials(trials, [0.8, 1.8])
    near, far = summary["by_distance"]

    assert near["successes"] == 1
    assert near["executed_trials"] == 2
    assert near["success_rate_executed"] == pytest.approx(0.5)
    assert near["operation_region_entry_rate"] == pytest.approx(1.0)
    assert far["executed_trials"] == 0
    assert far["success_rate_executed"] is None
    assert far["invalid_or_invisible"] == 1
