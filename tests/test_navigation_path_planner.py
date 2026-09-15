from __future__ import annotations

import numpy as np
import pytest

from robocasa.navigation.navigation_types import OccupancyMap
from robocasa.navigation.path_planner import AStarPathPlanner, PathPlanningError


def occupancy(grid):
    return OccupancyMap(
        grid=np.asarray(grid, dtype=np.uint8),
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
    )


def test_astar_reaches_goal_without_crossing_obstacle_and_smooths():
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[1:9, 5] = 1
    grid[5, 5] = 0
    planner = AStarPathPlanner()
    raw = planner.plan_cells(grid, (2, 2), (7, 7))
    assert raw[0] == (2, 2) and raw[-1] == (7, 7)
    assert all(grid[cell] == 0 for cell in raw)
    smoothed = planner.smooth_cells(grid, raw)
    assert len(smoothed) <= len(raw)
    assert all(
        planner.line_of_sight(grid, first, second)
        for first, second in zip(smoothed, smoothed[1:])
    )


def test_blocked_goal_returns_explicit_failure():
    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[4, 4] = 1
    with pytest.raises(PathPlanningError, match="GOAL_IN_COLLISION") as error:
        AStarPathPlanner().plan_cells(grid, (0, 0), (4, 4))
    assert error.value.code == "GOAL_IN_COLLISION"


def test_unreachable_goal_returns_path_not_found():
    grid = np.zeros((7, 7), dtype=np.uint8)
    grid[3, :] = 1
    with pytest.raises(PathPlanningError, match="PATH_NOT_FOUND") as error:
        AStarPathPlanner().plan_cells(grid, (1, 1), (5, 5))
    assert error.value.code == "PATH_NOT_FOUND"
