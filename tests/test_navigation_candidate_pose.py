from __future__ import annotations

import math

import numpy as np
import pytest

from robocasa.navigation.candidate_pose_generator import CandidatePoseGenerator
from robocasa.navigation.navigation_types import (
    FixtureGeometry,
    NavigationTarget,
    OccupancyMap,
    TargetType,
)


def generator():
    return CandidatePoseGenerator(
        candidate_config={
            "min_distance": 0.5,
            "preferred_distance": 0.7,
            "max_distance": 0.9,
            "radial_step": 0.2,
            "angular_step_deg": 90,
        },
        fixture_config={
            "min_distance": 0.45,
            "preferred_distance": 0.7,
            "max_distance": 0.95,
            "distance_step": 0.25,
            "lateral_range": 0.2,
            "lateral_step": 0.2,
        },
    )


def occupancy():
    geometry = FixtureGeometry(
        alias="microwave",
        polygon_xy=np.array([[-0.5, 0], [0.5, 0], [0.5, 0.5], [-0.5, 0.5]]),
        center_xy=np.array([0.0, 0.25]),
        yaw=0.0,
        front_center_xy=np.array([0.0, 0.0]),
        front_normal_xy=np.array([0.0, -1.0]),
        lateral_axis_xy=np.array([1.0, 0.0]),
    )
    return OccupancyMap(
        grid=np.zeros((100, 100), dtype=np.uint8),
        raw_grid=np.zeros((100, 100), dtype=np.uint8),
        clearance_m=np.ones((100, 100)),
        resolution=0.05,
        origin_x=-2.5,
        origin_y=-2.5,
        fixture_geometries={"microwave": geometry},
    )


def test_exposed_object_ring_respects_distance_and_faces_target():
    target = NavigationTarget(
        TargetType.EXPOSED_OBJECT, "bowl", "bowl", (0.0, 0.0, 0.8)
    )
    candidates = generator().generate(target, occupancy())
    assert candidates
    for candidate in candidates:
        distance = np.linalg.norm(candidate.xy - target.xy)
        assert 0.5 <= distance <= 0.9 + 1e-9
        heading = np.array([math.cos(candidate.yaw), math.sin(candidate.yaw)])
        assert np.dot(heading, (target.xy - candidate.xy) / distance) == pytest.approx(1.0)


def test_fixture_candidates_are_only_on_front_side_and_face_object():
    target = NavigationTarget(
        TargetType.OBJECT_IN_FIXTURE,
        "vegetable",
        "vegetable",
        (0.1, 0.25, 0.8),
        support_fixture="microwave",
    )
    candidates = generator().generate(target, occupancy())
    assert candidates
    assert all(candidate.y < 0 for candidate in candidates)
    assert all(-0.2 - 1e-9 <= candidate.x <= 0.2 + 1e-9 for candidate in candidates)


def test_occupancy_filter_rejects_inflated_obstacles():
    target = NavigationTarget(
        TargetType.EXPOSED_OBJECT, "bowl", "bowl", (0.0, 0.0, 0.8)
    )
    grid = occupancy()
    candidates = generator().generate(target, grid)
    blocked = grid.world_to_grid(candidates[0].xy)
    grid.grid[blocked] = 1
    valid = generator().filter_occupancy(candidates, grid)
    assert candidates[0] not in valid
    assert candidates[0].rejection_reason == "GOAL_IN_COLLISION"
