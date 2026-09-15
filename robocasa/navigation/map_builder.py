"""Build static-layout plus live-articulated RoboCasa occupancy maps.

Historically this builder only rasterized ``fixture.get_ext_sites``. Those
layout exterior sites do not follow articulated joint qpos, so a microwave
door opened by task reset was absent from the map while the cached closed-body
footprint remained unchanged. Static fixture footprints are now cached alone;
live collision geoms below door / drawer / rack joints are merged every build.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from .articulated_geometry import (
    ArticulatedCollisionGeometry,
    get_articulated_collision_geometries,
    get_robot_navigation_collision_z_range,
)
from .navigation_types import FixtureGeometry, OccupancyMap, raw_task_env


def _unit(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=float)[:2]
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        raise ValueError("fixture boundary has zero length")
    return value / norm


def fixture_geometry(alias: str, fixture: Any) -> FixtureGeometry:
    """Read a fixture's live world footprint and RoboCasa front direction.

    RoboCasa's placement code treats the local ``-y`` edge (p0 -> px) as the
    default front.  Reusing that convention keeps candidate generation aligned
    with ``compute_robot_base_placement_pose``.
    """

    points = np.asarray(fixture.get_ext_sites(relative=False), dtype=float)
    if points.shape != (4, 3) or not np.isfinite(points).all():
        raise ValueError(f"fixture {alias!r} returned invalid exterior sites")
    p0, px, py, _ = points
    corner = px + py - p0
    polygon = np.asarray([p0[:2], px[:2], corner[:2], py[:2]], dtype=float)
    lateral = _unit(px[:2] - p0[:2])
    front_normal = np.asarray([lateral[1], -lateral[0]], dtype=float)
    front_center = 0.5 * (p0[:2] + px[:2])
    center = polygon.mean(axis=0)
    return FixtureGeometry(
        alias=str(alias),
        polygon_xy=polygon,
        center_xy=center,
        yaw=float(getattr(fixture, "rot", math.atan2(lateral[1], lateral[0]))),
        front_center_xy=front_center,
        front_normal_xy=front_normal,
        lateral_axis_xy=lateral,
    )


def _rasterize_parallelogram(
    grid: np.ndarray,
    geometry: FixtureGeometry,
    *,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> None:
    polygon = geometry.polygon_xy
    x_min, y_min = polygon.min(axis=0)
    x_max, y_max = polygon.max(axis=0)
    col0 = max(0, int(math.floor((x_min - origin_x) / resolution)) - 1)
    col1 = min(
        grid.shape[1] - 1,
        int(math.floor((x_max - origin_x) / resolution)) + 1,
    )
    row0 = max(0, int(math.floor((y_min - origin_y) / resolution)) - 1)
    row1 = min(
        grid.shape[0] - 1,
        int(math.floor((y_max - origin_y) / resolution)) + 1,
    )
    p0, px, _, py = polygon
    basis = np.column_stack((px - p0, py - p0))
    determinant = float(np.linalg.det(basis))
    if abs(determinant) <= 1e-10:
        raise ValueError(f"fixture {geometry.alias!r} has degenerate footprint")
    inverse = np.linalg.inv(basis)
    for row in range(row0, row1 + 1):
        y = origin_y + (row + 0.5) * resolution
        for col in range(col0, col1 + 1):
            x = origin_x + (col + 0.5) * resolution
            uv = inverse @ (np.asarray([x, y]) - p0)
            if np.all(uv >= -1e-9) and np.all(uv <= 1.0 + 1e-9):
                grid[row, col] = 1


def _point_in_polygon(point: Sequence[float], polygon: np.ndarray) -> bool:
    """Return whether a point is inside or on a polygon."""

    x, y = float(point[0]), float(point[1])
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x0, y0 = float(previous[0]), float(previous[1])
        x1, y1 = float(current[0]), float(current[1])
        cross = (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)
        on_edge = (
            abs(cross) <= 1e-10
            and min(x0, x1) - 1e-10 <= x <= max(x0, x1) + 1e-10
            and min(y0, y1) - 1e-10 <= y <= max(y0, y1) + 1e-10
        )
        if on_edge:
            return True
        if (y0 > y) != (y1 > y):
            crossing_x = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x <= crossing_x + 1e-12:
                inside = not inside
        previous = current
    return inside


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    delta_ab = b - a
    delta_ac = c - a
    return float(delta_ab[0] * delta_ac[1] - delta_ab[1] * delta_ac[0])


def _segments_intersect(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> bool:
    tolerance = 1e-12
    if (
        max(float(a[0]), float(b[0])) + tolerance
        < min(float(c[0]), float(d[0]))
        or max(float(c[0]), float(d[0])) + tolerance
        < min(float(a[0]), float(b[0]))
        or max(float(a[1]), float(b[1])) + tolerance
        < min(float(c[1]), float(d[1]))
        or max(float(c[1]), float(d[1])) + tolerance
        < min(float(a[1]), float(b[1]))
    ):
        return False
    values = (
        _orientation(a, b, c),
        _orientation(a, b, d),
        _orientation(c, d, a),
        _orientation(c, d, b),
    )
    return (
        values[0] * values[1] <= tolerance
        and values[2] * values[3] <= tolerance
    )


def _polygon_intersects_cell(
    polygon: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> bool:
    corners = np.asarray(
        ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)),
        dtype=float,
    )
    if any(_point_in_polygon(corner, polygon) for corner in corners):
        return True
    if any(
        x_min - 1e-12 <= point[0] <= x_max + 1e-12
        and y_min - 1e-12 <= point[1] <= y_max + 1e-12
        for point in polygon
    ):
        return True
    polygon_edges = zip(polygon, np.roll(polygon, -1, axis=0))
    cell_edges = list(zip(corners, np.roll(corners, -1, axis=0)))
    return any(
        _segments_intersect(first, second, cell_first, cell_second)
        for first, second in polygon_edges
        for cell_first, cell_second in cell_edges
    )


def rasterize_polygon(
    grid: np.ndarray,
    polygon: Sequence[Sequence[float]],
    *,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> int:
    """Conservatively rasterize an oriented polygon, including thin doors."""

    points = np.asarray(polygon, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError("polygon must be Nx2 with N >= 3")
    lower, upper = points.min(axis=0), points.max(axis=0)
    col0 = max(0, int(math.floor((lower[0] - origin_x) / resolution)))
    col1 = min(
        grid.shape[1] - 1,
        int(math.floor((upper[0] - origin_x) / resolution)),
    )
    row0 = max(0, int(math.floor((lower[1] - origin_y) / resolution)))
    row1 = min(
        grid.shape[0] - 1,
        int(math.floor((upper[1] - origin_y) / resolution)),
    )
    before = int(np.count_nonzero(grid))
    for row in range(row0, row1 + 1):
        y_min = origin_y + row * resolution
        for col in range(col0, col1 + 1):
            x_min = origin_x + col * resolution
            if _polygon_intersects_cell(
                points,
                x_min=x_min,
                x_max=x_min + resolution,
                y_min=y_min,
                y_max=y_min + resolution,
            ):
                grid[row, col] = 1
    return int(np.count_nonzero(grid)) - before


@dataclass
class MapCacheEntry:
    occupancy_map: OccupancyMap
    geometry_signature: str


class MapCache:
    """Cache static layout grids while retaining live dynamic target queries."""

    def __init__(self) -> None:
        self._entries: dict[tuple[int, float, float], MapCacheEntry] = {}

    def get(
        self, layout_id: int, resolution: float, inflation_radius: float, signature: str
    ) -> OccupancyMap | None:
        entry = self._entries.get((int(layout_id), resolution, inflation_radius))
        if entry is None or entry.geometry_signature != signature:
            return None
        return entry.occupancy_map

    def put(
        self,
        layout_id: int,
        occupancy_map: OccupancyMap,
        signature: str,
    ) -> None:
        key = (
            int(layout_id),
            occupancy_map.resolution,
            occupancy_map.inflation_radius,
        )
        self._entries[key] = MapCacheEntry(occupancy_map, signature)

    def clear(self) -> None:
        self._entries.clear()


class MapBuilder:
    """Rasterize live fixture exterior bounds and inflate for the mobile base."""

    def __init__(
        self,
        *,
        resolution: float = 0.05,
        boundary_padding: float = 1.0,
        robot_radius: float = 0.4301162634,
        robot_half_length: float | None = None,
        robot_half_width: float | None = None,
        safety_margin: float = 0.05,
        border_obstacle_cells: int = 1,
        articulated_enabled: bool = True,
        articulated_min_projected_area_m2: float = 0.001,
        articulated_min_projected_span_m: float = 0.015,
        robot_collision_z_fallback: Sequence[float] = (0.0, 1.8),
        debug_articulated_map: bool = False,
        traversable_fixture_types: Sequence[str] = ("Floor",),
        cache: MapCache | None = None,
    ) -> None:
        if resolution <= 0 or boundary_padding <= 0:
            raise ValueError("resolution and boundary_padding must be positive")
        if robot_radius <= 0 or safety_margin < 0:
            raise ValueError("robot radius must be positive and margin non-negative")
        if robot_half_length is not None and robot_half_length <= 0:
            raise ValueError("robot_half_length must be positive")
        if robot_half_width is not None and robot_half_width <= 0:
            raise ValueError("robot_half_width must be positive")
        if border_obstacle_cells < 0:
            raise ValueError("border_obstacle_cells must be non-negative")
        if articulated_min_projected_area_m2 < 0:
            raise ValueError("articulated minimum projected area must be non-negative")
        if articulated_min_projected_span_m < 0:
            raise ValueError("articulated minimum projected span must be non-negative")
        z_fallback = np.asarray(robot_collision_z_fallback, dtype=float).reshape(-1)
        if (
            z_fallback.size != 2
            or not np.isfinite(z_fallback).all()
            or z_fallback[0] >= z_fallback[1]
        ):
            raise ValueError("robot_collision_z_fallback must be [min, max]")
        self.resolution = float(resolution)
        self.boundary_padding = float(boundary_padding)
        self.robot_radius = float(robot_radius)
        # A legacy radius remains accepted for callers that do not provide the
        # Omron footprint dimensions. The production configuration supplies
        # both dimensions so flat counter faces do not inherit corner-radius
        # over-inflation from a circular approximation.
        self.robot_half_length = float(
            robot_radius if robot_half_length is None else robot_half_length
        )
        self.robot_half_width = float(
            robot_radius if robot_half_width is None else robot_half_width
        )
        self.safety_margin = float(safety_margin)
        self.border_obstacle_cells = int(border_obstacle_cells)
        self.articulated_enabled = bool(articulated_enabled)
        self.articulated_min_projected_area_m2 = float(
            articulated_min_projected_area_m2
        )
        self.articulated_min_projected_span_m = float(
            articulated_min_projected_span_m
        )
        self.robot_collision_z_fallback = tuple(float(value) for value in z_fallback)
        self.debug_articulated_map = bool(debug_articulated_map)
        self.traversable_fixture_types = frozenset(
            str(value) for value in traversable_fixture_types
        )
        self.cache = cache

    @property
    def inflation_radius(self) -> float:
        return math.hypot(
            self.robot_half_length + self.safety_margin,
            self.robot_half_width + self.safety_margin,
        )

    def _footprint_structure(self, yaw: float) -> np.ndarray:
        """Return a grid-aligned mask for a rectangle rotated by yaw."""

        half_length = self.robot_half_length + self.safety_margin
        half_width = self.robot_half_width + self.safety_margin
        radius = math.hypot(half_length, half_width)
        cells = int(math.ceil(radius / self.resolution))
        offsets = np.arange(-cells, cells + 1, dtype=float) * self.resolution
        dx, dy = np.meshgrid(offsets, offsets)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        return np.asarray(
            (np.abs(local_x) <= half_length + 1e-9)
            & (np.abs(local_y) <= half_width + 1e-9),
            dtype=bool,
        )

    def footprint_is_free(
        self,
        occupancy_map: OccupancyMap,
        center_xy: Sequence[float],
        yaw: float,
    ) -> bool:
        """Check a candidate oriented footprint against raw obstacles."""

        center = occupancy_map.world_to_grid(center_xy)
        structure = self._footprint_structure(yaw)
        half_row = structure.shape[0] // 2
        half_col = structure.shape[1] // 2
        for row_offset, col_offset in np.argwhere(structure) - np.asarray(
            [half_row, half_col]
        ):
            cell = (center[0] + int(row_offset), center[1] + int(col_offset))
            if not occupancy_map.in_bounds(cell):
                return False
            if occupancy_map.raw_grid[cell] != 0:
                return False
        return True

    def _geometries(self, raw_env: Any) -> tuple[dict[str, FixtureGeometry], list[str]]:
        geometries: dict[str, FixtureGeometry] = {}
        skipped: list[str] = []
        by_identity: dict[int, FixtureGeometry] = {}
        for alias, fixture in getattr(raw_env, "fixtures", {}).items():
            try:
                identity = id(fixture)
                if identity in by_identity:
                    source = by_identity[identity]
                    geometries[str(alias)] = FixtureGeometry(
                        alias=str(alias),
                        polygon_xy=source.polygon_xy,
                        center_xy=source.center_xy,
                        yaw=source.yaw,
                        front_center_xy=source.front_center_xy,
                        front_normal_xy=source.front_normal_xy,
                        lateral_axis_xy=source.lateral_axis_xy,
                    )
                else:
                    value = fixture_geometry(str(alias), fixture)
                    geometries[str(alias)] = value
                    by_identity[identity] = value
            except (AttributeError, TypeError, ValueError):
                skipped.append(str(alias))
        if not geometries:
            raise ValueError("no fixture exterior geometry was available")
        return geometries, skipped

    @staticmethod
    def _signature(geometries: Mapping[str, FixtureGeometry]) -> str:
        payload = [
            (alias, np.round(value.polygon_xy, 5).tolist())
            for alias, value in sorted(geometries.items())
        ]
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _inflate(self, raw_grid: np.ndarray, robot_yaw: float) -> tuple[np.ndarray, np.ndarray]:
        clearance = distance_transform_edt(raw_grid == 0) * self.resolution
        inflated = np.asarray(
            binary_dilation(
                raw_grid != 0,
                structure=self._footprint_structure(float(robot_yaw)),
            ),
            dtype=np.uint8,
        )
        return inflated, clearance

    @staticmethod
    def _articulated_summary(
        geometries: Sequence[ArticulatedCollisionGeometry],
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for geometry in geometries:
            fixture = summary.setdefault(
                geometry.fixture_name,
                {
                    "fixture_type": geometry.fixture_type,
                    "components": set(),
                    "joint_states": {},
                    "num_collision_geoms": 0,
                    "projected_bounds": None,
                },
            )
            fixture["components"].add(geometry.component)
            fixture["joint_states"][geometry.joint_name] = geometry.joint_state
            fixture["num_collision_geoms"] += 1
            bounds = np.asarray(
                [
                    geometry.polygon_xy[:, 0].min(),
                    geometry.polygon_xy[:, 1].min(),
                    geometry.polygon_xy[:, 0].max(),
                    geometry.polygon_xy[:, 1].max(),
                ],
                dtype=float,
            )
            current = fixture["projected_bounds"]
            if current is None:
                fixture["projected_bounds"] = bounds
            else:
                current = np.asarray(current, dtype=float)
                fixture["projected_bounds"] = np.asarray(
                    [
                        min(current[0], bounds[0]),
                        min(current[1], bounds[1]),
                        max(current[2], bounds[2]),
                        max(current[3], bounds[3]),
                    ]
                )
        for fixture in summary.values():
            fixture["components"] = sorted(fixture["components"])
            bounds = fixture["projected_bounds"]
            fixture["projected_bounds"] = (
                None if bounds is None else np.asarray(bounds, dtype=float).tolist()
            )
        return summary

    def _merge_articulated_layer(
        self,
        env: Any,
        static_map: OccupancyMap,
        *,
        robot_yaw: float,
        static_map_cached: bool,
        debug_articulated_map: bool,
    ) -> OccupancyMap:
        static_grid = np.asarray(static_map.raw_grid, dtype=np.uint8).copy()
        articulated_grid = np.zeros_like(static_grid, dtype=np.uint8)
        z_range = get_robot_navigation_collision_z_range(
            env, fallback=self.robot_collision_z_fallback
        )
        articulated = (
            get_articulated_collision_geometries(
                env,
                robot_z_range=z_range,
                minimum_projected_area_m2=self.articulated_min_projected_area_m2,
                minimum_projected_span_m=self.articulated_min_projected_span_m,
            )
            if self.articulated_enabled
            else []
        )
        for geometry in articulated:
            rasterize_polygon(
                articulated_grid,
                geometry.polygon_xy,
                origin_x=static_map.origin_x,
                origin_y=static_map.origin_y,
                resolution=static_map.resolution,
            )
        raw_grid = np.asarray(
            (static_grid != 0) | (articulated_grid != 0), dtype=np.uint8
        )
        inflated, clearance = self._inflate(raw_grid, robot_yaw)
        summary = self._articulated_summary(articulated)
        added = int(np.count_nonzero((articulated_grid != 0) & (static_grid == 0)))
        result = OccupancyMap(
            grid=inflated,
            raw_grid=raw_grid,
            clearance_m=clearance,
            resolution=static_map.resolution,
            origin_x=static_map.origin_x,
            origin_y=static_map.origin_y,
            fixture_geometries=static_map.fixture_geometries,
            inflation_radius=static_map.inflation_radius,
        )
        result.static_grid = static_grid
        result.articulated_grid = articulated_grid
        result.articulated_geometries = [value.to_dict() for value in articulated]
        result.static_map_cached = bool(static_map_cached)
        result.articulated_summary = summary
        result.map_debug = {
            "static_map_cached": bool(static_map_cached),
            "robot_collision_z_range": list(z_range),
            "articulated_fixtures": summary,
            "articulated_cells": int(np.count_nonzero(articulated_grid)),
            "articulated_cells_added": added,
            "static_occupied_cells": int(np.count_nonzero(static_grid)),
            "final_occupied_cells": int(np.count_nonzero(raw_grid)),
            "inflated_occupied_cells": int(np.count_nonzero(inflated)),
        }
        result.skipped_fixture_aliases = list(
            getattr(static_map, "skipped_fixture_aliases", ())
        )
        result.traversable_fixture_aliases = list(
            getattr(static_map, "traversable_fixture_aliases", ())
        )
        print(
            "[NAV-MAP] static_cached={} articulated_fixtures={} "
            "articulated_added={} final={} inflated={} z_range={}".format(
                static_map_cached,
                len(summary),
                added,
                result.map_debug["final_occupied_cells"],
                result.map_debug["inflated_occupied_cells"],
                [round(value, 3) for value in z_range],
            )
        )
        if debug_articulated_map:
            for fixture_name, fixture in summary.items():
                print(
                    "[NAV-MAP] fixture={} type={} components={} joints={} "
                    "collision_geoms={} bounds={}".format(
                        fixture_name,
                        fixture["fixture_type"],
                        fixture["components"],
                        fixture["joint_states"],
                        fixture["num_collision_geoms"],
                        fixture["projected_bounds"],
                    )
                )
        return result

    def build(
        self,
        env: Any,
        *,
        layout_id: int | None = None,
        robot_xy: Sequence[float] | None = None,
        robot_yaw: float = 0.0,
        use_cache: bool = True,
        debug_articulated_map: bool | None = None,
    ) -> OccupancyMap:
        """Return cached static layout merged with current articulated geometry."""

        raw_env = raw_task_env(env)
        geometries, skipped = self._geometries(raw_env)
        traversable_aliases = {
            str(alias)
            for alias, fixture in raw_env.fixtures.items()
            if type(fixture).__name__ in self.traversable_fixture_types
        }
        signature = (
            self._signature(geometries)
            + f":footprint={self.robot_half_length:.5f},"
            f"{self.robot_half_width:.5f},{self.safety_margin:.5f}:"
            f"yaw={float(robot_yaw):.5f}"
        )
        resolved_layout = int(
            layout_id if layout_id is not None else getattr(raw_env, "layout_id", -1)
        )
        static_map = None
        static_map_cached = False
        if use_cache and self.cache is not None:
            static_map = self.cache.get(
                resolved_layout, self.resolution, self.inflation_radius, signature
            )
            static_map_cached = static_map is not None

        if static_map is None:
            unique_polygons: list[np.ndarray] = []
            seen_polygons: set[tuple[float, ...]] = set()
            for value in geometries.values():
                key = tuple(np.round(value.polygon_xy.reshape(-1), 5))
                if key not in seen_polygons:
                    seen_polygons.add(key)
                    unique_polygons.append(value.polygon_xy)
            points = np.concatenate(unique_polygons, axis=0)
            if robot_xy is not None:
                robot_point = np.asarray(robot_xy, dtype=float).reshape(-1)
                if robot_point.size >= 2 and np.isfinite(robot_point[:2]).all():
                    points = np.vstack((points, robot_point[:2]))
            lower = points.min(axis=0) - self.boundary_padding
            upper = points.max(axis=0) + self.boundary_padding
            origin_x = math.floor(lower[0] / self.resolution) * self.resolution
            origin_y = math.floor(lower[1] / self.resolution) * self.resolution
            width = int(math.ceil((upper[0] - origin_x) / self.resolution))
            height = int(math.ceil((upper[1] - origin_y) / self.resolution))
            static_grid = np.zeros((height, width), dtype=np.uint8)

            rasterized: set[tuple[float, ...]] = set()
            for alias, value in geometries.items():
                if alias in traversable_aliases:
                    continue
                key = tuple(np.round(value.polygon_xy.reshape(-1), 5))
                if key in rasterized:
                    continue
                _rasterize_parallelogram(
                    static_grid,
                    value,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    resolution=self.resolution,
                )
                rasterized.add(key)
            border = self.border_obstacle_cells
            if border:
                static_grid[:border, :] = 1
                static_grid[-border:, :] = 1
                static_grid[:, :border] = 1
                static_grid[:, -border:] = 1
            inflated, clearance = self._inflate(static_grid, robot_yaw)
            static_map = OccupancyMap(
                grid=inflated,
                raw_grid=static_grid,
                clearance_m=clearance,
                resolution=self.resolution,
                origin_x=origin_x,
                origin_y=origin_y,
                fixture_geometries=geometries,
                inflation_radius=self.inflation_radius,
            )
            static_map.skipped_fixture_aliases = skipped
            static_map.traversable_fixture_aliases = sorted(traversable_aliases)
            if use_cache and self.cache is not None:
                self.cache.put(resolved_layout, static_map, signature)

        debug = (
            self.debug_articulated_map
            if debug_articulated_map is None
            else bool(debug_articulated_map)
        )
        return self._merge_articulated_layer(
            env,
            static_map,
            robot_yaw=robot_yaw,
            static_map_cached=static_map_cached,
            debug_articulated_map=debug,
        )
