from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from robocasa.navigation.debug_artifacts import save_navigation_plan_artifacts


class FakeMap:
    def __init__(self):
        self.static_grid = np.zeros((8, 9), dtype=np.uint8)
        self.articulated_grid = np.zeros((8, 9), dtype=np.uint8)
        self.articulated_grid[3, 4] = 1
        self.raw_grid = np.maximum(self.static_grid, self.articulated_grid)
        self.grid = self.raw_grid.copy()
        self.grid[2:5, 3:6] = 1
        self.world_bounds = (0.0, 0.9, 0.0, 0.8)
        self.resolution = 0.1
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.inflation_radius = 0.25
        self.map_debug = {"static_map_cached": False}
        self.articulated_geometries = []


def test_save_navigation_plan_artifacts_writes_viewable_map_and_grids(tmp_path):
    occupancy_map = FakeMap()
    target = SimpleNamespace(
        target_name="microwave",
        target_alias="microwave_main_group",
        target_type="FIXTURE",
        world_position=np.asarray([0.75, 0.6, 1.2]),
    )
    candidate = SimpleNamespace(
        xy=np.asarray([0.65, 0.55]),
        yaw=0.25,
        operation_distance=0.7,
        clearance_m=0.3,
        path_cost_m=0.8,
        score=1.2,
        valid=True,
        rejection_reason=None,
    )
    pipeline = SimpleNamespace(
        last_debug={
            "map": occupancy_map,
            "target": target,
            "start_xy": np.asarray([0.1, 0.1]),
            "start_collision_xy": np.asarray([0.15, 0.1]),
            "raw_candidates": [candidate],
            "valid_candidates": [candidate],
            "selected": candidate,
            "path_xy": [(0.1, 0.1), (0.65, 0.55)],
            "collision_center_path_xy": [(0.15, 0.1), (0.65, 0.55)],
            "skip_evidence": {"current_free": True},
        },
        _footprint_center=lambda xy, yaw: np.asarray(xy, dtype=float),
    )

    saved = save_navigation_plan_artifacts(
        output_root=tmp_path,
        artifact_index=1,
        subgoal_id="place bowl/in microwave",
        pipeline=pipeline,
        navigation_result={"success": True, "reason": "PLANNED"},
    )

    output_dir = tmp_path / "000001_place_bowl_in_microwave"
    assert saved["directory"] == str(output_dir)
    assert (output_dir / "overview.png").stat().st_size > 0
    for name in (
        "static_occupancy.npy",
        "articulated_occupancy.npy",
        "raw_occupancy.npy",
        "inflated_occupancy.npy",
    ):
        assert np.load(output_dir / name).shape == (8, 9)
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["navigation_result"]["reason"] == "PLANNED"
    assert metadata["map"]["resolution"] == 0.1
    assert metadata["selected_candidate"]["score"] == 1.2


def test_save_navigation_plan_artifacts_records_failure_without_map(tmp_path):
    pipeline = SimpleNamespace(last_debug={})
    saved = save_navigation_plan_artifacts(
        output_root=tmp_path,
        artifact_index=2,
        subgoal_id="missing_target",
        pipeline=pipeline,
        navigation_result={"success": False, "reason": "TARGET_NOT_FOUND"},
    )

    assert saved["map_available"] is False
    assert saved["overview"] is None
    metadata = json.loads(
        (tmp_path / "000002_missing_target" / "metadata.json").read_text()
    )
    assert metadata["navigation_result"]["reason"] == "TARGET_NOT_FOUND"
