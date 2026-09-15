from __future__ import annotations

import numpy as np

from robocasa.navigation.candidate_pose_generator import CandidatePoseGenerator
from robocasa.navigation.map_builder import MapBuilder, MapCache
from robocasa.navigation.navigation_types import CandidatePose
from robocasa.navigation.path_planner import AStarPathPlanner


class DoorFixture:
    name = "microwave"
    rot = 0.0
    _joint_infos = {"microwave_microjoint": {"range": [-np.pi / 2, 0.0]}}

    @property
    def door_joint_names(self):
        return ["microwave_microjoint"]

    def get_ext_sites(self, relative=False):
        assert relative is False
        return np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )


class FakeModel:
    def __init__(self):
        self.ngeom = 4
        self.body_parentid = np.asarray([0, 0, 1])
        self.jnt_bodyid = np.asarray([2])
        self.geom_bodyid = np.asarray([2, 2, 2, 2])
        self.geom_type = np.asarray([6, 6, 6, 6])
        self.geom_size = np.asarray(
            [
                [0.40, 0.02, 0.40],
                [0.40, 0.02, 0.40],
                [0.005, 0.005, 0.02],
                [0.30, 0.02, 0.20],
            ]
        )
        self.geom_contype = np.asarray([1, 0, 1, 1])
        self.geom_conaffinity = np.asarray([1, 0, 1, 1])
        self.geom_rbound = np.asarray([0.6, 0.6, 0.03, 0.4])
        self._joint_names = ["microwave_microjoint"]
        self._body_names = ["world", "microwave", "microwave_door"]
        self._geom_names = [
            "microwave_door_main",
            "microwave_door_visual",
            "microwave_handle",
            "microwave_high_trim",
        ]

    def joint_name2id(self, name):
        return self._joint_names.index(name)

    def body_id2name(self, index):
        return self._body_names[index]

    def geom_id2name(self, index):
        return self._geom_names[index]


class FakeData:
    def __init__(self):
        self.geom_xpos = np.asarray(
            [
                [0.5, 0.0, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.0, 3.0],
            ],
            dtype=float,
        )
        self.geom_xmat = np.tile(np.eye(3).reshape(1, 9), (4, 1))
        self.qpos = 0.0

    def get_joint_qpos(self, name):
        assert name == "microwave_microjoint"
        return self.qpos


class FakeEnv:
    def __init__(self):
        self.fixtures = {"microwave": DoorFixture()}
        self.objects = {}
        self.layout_id = 1
        self.sim = type("Sim", (), {"model": FakeModel(), "data": FakeData()})()

    def set_open(self, opened: bool):
        data = self.sim.data
        if opened:
            data.qpos = -np.pi / 2
            data.geom_xpos[0] = [0.5, -0.4, 0.5]
            data.geom_xpos[1] = [0.5, -0.4, 0.5]
            data.geom_xpos[2] = [0.5, -0.4, 0.5]
            rotation = np.asarray(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            )
            data.geom_xmat[:3] = rotation.reshape(1, 9)
        else:
            data.qpos = 0.0
            data.geom_xpos[:3] = np.asarray(
                [[0.5, 0.0, 0.5], [0.5, 0.0, 0.5], [0.5, 0.0, 0.5]]
            )
            data.geom_xmat[:3] = np.eye(3).reshape(1, 9)


def builder(cache=None):
    return MapBuilder(
        resolution=0.05,
        boundary_padding=1.5,
        robot_radius=0.05,
        robot_half_length=0.05,
        robot_half_width=0.05,
        safety_margin=0.0,
        border_obstacle_cells=0,
        cache=cache,
    )


def maps():
    env = FakeEnv()
    value = builder(MapCache())
    env.set_open(False)
    closed = value.build(env, layout_id=1)
    env.set_open(True)
    opened = value.build(env, layout_id=1)
    return env, value, closed, opened


def test_closed_and_open_articulated_maps_differ():
    _, _, closed, opened = maps()
    assert np.any(closed.articulated_grid != opened.articulated_grid)
    assert np.any(closed.raw_grid != opened.raw_grid)


def test_open_door_polygon_cells_are_occupied():
    _, _, _, opened = maps()
    for point in ((0.5, -0.2), (0.5, -0.5), (0.5, -0.75)):
        assert opened.articulated_grid[opened.world_to_grid(point)] == 1
        assert opened.raw_grid[opened.world_to_grid(point)] == 1


def test_closing_door_frees_previous_swept_region():
    _, _, closed, opened = maps()
    swept = opened.world_to_grid((0.5, -0.5))
    assert opened.raw_grid[swept] == 1
    assert closed.raw_grid[swept] == 0


def test_astar_does_not_cross_open_door_polygon():
    _, _, _, opened = maps()
    planner = AStarPathPlanner()
    start, goal = (-0.4, -0.5), (1.4, -0.5)
    assert not planner.line_of_sight(
        opened.grid, opened.world_to_grid(start), opened.world_to_grid(goal)
    )
    path = planner.plan(opened, start, goal, smooth=True)
    assert all(
        planner.line_of_sight(opened.grid, first, second)
        for first, second in zip(path.cells, path.cells[1:])
    )


def test_smoothing_does_not_shortcut_through_open_door():
    _, _, _, opened = maps()
    planner = AStarPathPlanner()
    cells = [
        opened.world_to_grid(point)
        for point in ((-0.4, -0.5), (-0.4, -1.0), (1.4, -1.0), (1.4, -0.5))
    ]
    smoothed = planner.smooth_cells(opened.grid, cells)
    assert len(smoothed) > 2
    assert all(
        planner.line_of_sight(opened.grid, first, second)
        for first, second in zip(smoothed, smoothed[1:])
    )


def test_candidate_on_open_door_inflation_is_rejected():
    _, _, _, opened = maps()
    generator = CandidatePoseGenerator(
        candidate_config={
            "min_distance": 0.5,
            "preferred_distance": 0.7,
            "max_distance": 0.9,
            "radial_step": 0.1,
            "angular_step_deg": 30,
        },
        fixture_config={
            "min_distance": 0.5,
            "preferred_distance": 0.7,
            "max_distance": 0.9,
            "distance_step": 0.1,
            "lateral_range": 0.2,
            "lateral_step": 0.1,
        },
    )
    candidate = CandidatePose(
        x=0.5,
        y=-0.5,
        yaw=0.0,
        operation_distance=0.5,
        source="test",
    )
    assert generator.filter_occupancy([candidate], opened) == []
    assert candidate.rejection_reason == "GOAL_IN_COLLISION"


def test_static_cache_never_reuses_old_articulated_state():
    env = FakeEnv()
    cache = MapCache()
    value = builder(cache)
    env.set_open(True)
    opened = value.build(env, layout_id=1)
    env.set_open(False)
    closed = value.build(env, layout_id=1)
    assert opened.static_map_cached is False
    assert closed.static_map_cached is True
    assert np.array_equal(opened.static_grid, closed.static_grid)
    assert not np.array_equal(opened.articulated_grid, closed.articulated_grid)


def test_visual_tiny_and_above_robot_geoms_are_filtered():
    env = FakeEnv()
    env.set_open(True)
    current = builder().build(env, use_cache=False)
    names = {item["geom_name"] for item in current.articulated_geometries}
    assert "microwave_door_main" in names
    assert "microwave_door_visual" not in names
    assert "microwave_handle" not in names
    assert "microwave_high_trim" not in names

