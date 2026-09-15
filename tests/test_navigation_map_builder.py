from __future__ import annotations

import numpy as np
import pytest

from robocasa.navigation.map_builder import MapBuilder, fixture_geometry


class BoxFixture:
    def __init__(self, alias="counter", x0=0.0, y0=0.0, width=1.0, depth=1.0):
        self.name = alias
        self.nat_lang = alias
        self.pos = np.array([x0 + width / 2, y0 + depth / 2, 0.5])
        self.rot = 0.0
        self._points = np.array(
            [
                [x0, y0, 0.0],
                [x0 + width, y0, 0.0],
                [x0, y0 + depth, 0.0],
                [x0, y0, 1.0],
            ]
        )

    def get_ext_sites(self, relative=False):
        assert relative is False
        return self._points.copy()


class FakeEnv:
    def __init__(self):
        self.fixtures = {"counter": BoxFixture()}
        self.objects = {}
        self.sim = object()
        self.layout_id = 1


def test_world_grid_conversion_round_trip_is_cell_center_accurate():
    occupancy = MapBuilder(
        resolution=0.05,
        boundary_padding=1.0,
        robot_radius=0.2,
        safety_margin=0.05,
    ).build(FakeEnv(), robot_xy=(-0.8, -0.8), use_cache=False)
    point = (-0.73, -0.61)
    recovered = occupancy.grid_to_world(occupancy.world_to_grid(point))
    assert np.linalg.norm(np.asarray(point) - recovered) <= occupancy.resolution


def test_fixture_rasterization_and_inflation_use_runtime_bounds():
    occupancy = MapBuilder(
        resolution=0.05,
        boundary_padding=1.0,
        robot_radius=0.2,
        safety_margin=0.05,
    ).build(FakeEnv(), robot_xy=(-0.8, -0.8), use_cache=False)
    assert occupancy.raw_grid[occupancy.world_to_grid((0.5, 0.5))] == 1
    assert occupancy.raw_grid[occupancy.world_to_grid((-0.15, 0.5))] == 0
    assert occupancy.grid[occupancy.world_to_grid((-0.15, 0.5))] == 1
    assert occupancy.grid[occupancy.world_to_grid((-0.40, 0.5))] == 0


def test_fixture_front_matches_robocasa_local_negative_y_convention():
    geometry = fixture_geometry("counter", BoxFixture())
    assert geometry.front_center_xy == pytest.approx([0.5, 0.0])
    assert geometry.front_normal_xy == pytest.approx([0.0, -1.0])
    assert geometry.lateral_axis_xy == pytest.approx([1.0, 0.0])


class Floor(BoxFixture):
    pass


def test_floor_defines_bounds_without_becoming_an_obstacle():
    env = FakeEnv()
    env.fixtures["floor"] = Floor("floor", -2.0, -2.0, 4.0, 4.0)
    occupancy = MapBuilder(
        resolution=0.05,
        boundary_padding=0.5,
        robot_radius=0.2,
        safety_margin=0.0,
    ).build(env, use_cache=False)
    assert occupancy.raw_grid[occupancy.world_to_grid((1.5, 1.5))] == 0
    assert "floor" in occupancy.traversable_fixture_aliases


def test_rectangular_inflation_rotates_with_robot_yaw():
    builder = MapBuilder(
        resolution=0.05,
        boundary_padding=1.0,
        robot_radius=0.43,
        robot_half_length=0.35,
        robot_half_width=0.20,
        safety_margin=0.0,
    )
    parallel = builder.build(
        FakeEnv(), robot_xy=(-0.3, 0.5), robot_yaw=0.0, use_cache=False
    )
    perpendicular = builder.build(
        FakeEnv(),
        robot_xy=(-0.3, 0.5),
        robot_yaw=np.pi / 2.0,
        use_cache=False,
    )
    point = (-0.3, 0.5)
    assert parallel.grid[parallel.world_to_grid(point)] == 1
    assert perpendicular.grid[perpendicular.world_to_grid(point)] == 0
