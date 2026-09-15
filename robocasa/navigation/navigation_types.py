"""Serializable types shared by the target-aware navigation package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


class TargetType(str, Enum):
    """Geometric treatment required by a live navigation target."""

    EXPOSED_OBJECT = "EXPOSED_OBJECT"
    OBJECT_IN_FIXTURE = "OBJECT_IN_FIXTURE"
    FIXTURE = "FIXTURE"


@dataclass(frozen=True)
class FixtureGeometry:
    """One runtime fixture footprint and its accessible front side."""

    alias: str
    polygon_xy: np.ndarray
    center_xy: np.ndarray
    yaw: float
    front_center_xy: np.ndarray
    front_normal_xy: np.ndarray
    lateral_axis_xy: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "polygon_xy": self.polygon_xy.tolist(),
            "center_xy": self.center_xy.tolist(),
            "yaw": self.yaw,
            "front_center_xy": self.front_center_xy.tolist(),
            "front_normal_xy": self.front_normal_xy.tolist(),
            "lateral_axis_xy": self.lateral_axis_xy.tolist(),
        }


@dataclass
class OccupancyMap:
    """A world-aligned 2-D occupancy grid (0 free, 1 obstacle)."""

    grid: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    raw_grid: np.ndarray | None = None
    clearance_m: np.ndarray | None = None
    fixture_geometries: dict[str, FixtureGeometry] = field(default_factory=dict)
    inflation_radius: float = 0.0

    def __post_init__(self) -> None:
        self.grid = np.asarray(self.grid, dtype=np.uint8)
        if self.grid.ndim != 2:
            raise ValueError("occupancy grid must be two-dimensional")
        if self.resolution <= 0:
            raise ValueError("occupancy resolution must be positive")
        if self.raw_grid is None:
            self.raw_grid = self.grid.copy()
        else:
            self.raw_grid = np.asarray(self.raw_grid, dtype=np.uint8)
            if self.raw_grid.shape != self.grid.shape:
                raise ValueError("raw and inflated occupancy grids must have equal shape")
        if self.clearance_m is not None:
            self.clearance_m = np.asarray(self.clearance_m, dtype=float)
            if self.clearance_m.shape != self.grid.shape:
                raise ValueError("clearance map must match occupancy grid shape")

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    @property
    def world_bounds(self) -> tuple[float, float, float, float]:
        return (
            self.origin_x,
            self.origin_x + self.width * self.resolution,
            self.origin_y,
            self.origin_y + self.height * self.resolution,
        )

    def world_to_grid(self, xy: Sequence[float]) -> tuple[int, int]:
        point = np.asarray(xy, dtype=float).reshape(-1)
        if point.size < 2 or not np.isfinite(point[:2]).all():
            raise ValueError("world coordinate must contain two finite values")
        col = int(np.floor((point[0] - self.origin_x) / self.resolution))
        row = int(np.floor((point[1] - self.origin_y) / self.resolution))
        return row, col

    def grid_to_world(self, cell: Sequence[int]) -> tuple[float, float]:
        row, col = int(cell[0]), int(cell[1])
        return (
            self.origin_x + (col + 0.5) * self.resolution,
            self.origin_y + (row + 0.5) * self.resolution,
        )

    def in_bounds(self, cell: Sequence[int]) -> bool:
        row, col = int(cell[0]), int(cell[1])
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free_world(self, xy: Sequence[float]) -> bool:
        cell = self.world_to_grid(xy)
        return self.in_bounds(cell) and self.grid[cell] == 0

    def clearance_at_world(self, xy: Sequence[float]) -> float:
        cell = self.world_to_grid(xy)
        if not self.in_bounds(cell):
            return 0.0
        if self.clearance_m is None:
            return float("inf") if self.raw_grid[cell] == 0 else 0.0
        return float(self.clearance_m[cell])

    def to_dict(self, *, include_grids: bool = False) -> dict[str, Any]:
        result = {
            "resolution": self.resolution,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "width": self.width,
            "height": self.height,
            "world_bounds": list(self.world_bounds),
            "inflation_radius": self.inflation_radius,
            "fixtures": {
                alias: geometry.to_dict()
                for alias, geometry in self.fixture_geometries.items()
            },
        }
        map_debug = getattr(self, "map_debug", None)
        if isinstance(map_debug, Mapping):
            result["layers"] = dict(map_debug)
        articulated = getattr(self, "articulated_geometries", None)
        if articulated is not None:
            result["articulated_geometries"] = list(articulated)
        if include_grids:
            result["grid"] = self.grid.tolist()
            result["raw_grid"] = self.raw_grid.tolist()
            static_grid = getattr(self, "static_grid", None)
            articulated_grid = getattr(self, "articulated_grid", None)
            if static_grid is not None:
                result["static_grid"] = np.asarray(static_grid).tolist()
            if articulated_grid is not None:
                result["articulated_grid"] = np.asarray(articulated_grid).tolist()
        return result


@dataclass(frozen=True)
class NavigationTarget:
    target_type: TargetType
    target_name: str
    target_alias: str
    world_position: tuple[float, float, float]
    world_orientation_xyzw: tuple[float, float, float, float] | None = None
    support_fixture: str | None = None
    fixture_instance: Any | None = field(default=None, repr=False, compare=False)
    source: str = ""
    diagnostics: tuple[dict[str, Any], ...] = ()

    @property
    def xy(self) -> np.ndarray:
        return np.asarray(self.world_position[:2], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_type"] = self.target_type.value
        result.pop("fixture_instance", None)
        return result


@dataclass
class CandidatePose:
    x: float
    y: float
    yaw: float
    operation_distance: float
    source: str
    valid: bool = False
    rejection_reason: str | None = None
    clearance_m: float = 0.0
    path_cost_m: float | None = None
    score: float | None = None
    path_xy: list[tuple[float, float]] = field(default_factory=list)
    collision_center_path_xy: list[tuple[float, float]] = field(default_factory=list)

    @property
    def xy(self) -> np.ndarray:
        return np.asarray([self.x, self.y], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NavigationResult:
    success: bool
    reason: str
    target_name: str | None = None
    target_type: str | None = None
    target_position: list[float] | None = None
    selected_goal: dict[str, Any] | None = None
    path_length: float = 0.0
    waypoint_count: int = 0
    skipped: bool = False
    plan_only: bool = False
    raw_candidate_count: int = 0
    valid_candidate_count: int = 0
    execution: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



@dataclass
class NavigationExecutionResult:
    """Outcome of executing a smoothed world-frame navigation path."""

    success: bool
    reason: str
    reached_waypoints: int
    total_waypoints: int
    final_pose: dict[str, Any] | None = None
    position_error: float | None = None
    yaw_error: float | None = None
    steps: int = 0
    waypoint_traces: list[dict[str, Any]] = field(default_factory=list)
    held_object_guard: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Retain field spellings used by the earlier navigation result logs.
        result["failure_code"] = None if self.success else self.reason
        result["final_position_error_m"] = self.position_error
        result["final_yaw_error_deg"] = self.yaw_error
        result["guard"] = self.held_object_guard
        return result


FAILURE_REASONS = frozenset(
    {
        "TARGET_NOT_FOUND",
        "TARGET_POSE_UNAVAILABLE",
        "NO_VALID_WORK_POSE",
        "START_IN_COLLISION",
        "GOAL_IN_COLLISION",
        "PATH_NOT_FOUND",
        "PATH_INVALIDATED_BY_SCENE_STATE",
        "MOTION_STALLED",
        "OBJECT_DROPPED",
        "OBJECT_DROPPED_DURING_PREP",
        "POST_GRASP_LIFT_FAILED",
        "POST_GRASP_RETRACT_FAILED",
        "TRANSPORT_NOT_READY",
        "TRANSPORT_READY_FAILED",
        "POST_PLACE_LIFT_FAILED",
        "POST_PLACE_RETRACT_FAILED",
        "POST_PLACE_RETREAT_FAILED",
        "FINAL_POSE_NOT_REACHED",
    }
)


def raw_task_env(env: Any) -> Any:
    """Return the wrapped RoboCasa owner of fixtures, objects and simulator."""

    current = env
    seen: set[int] = set()
    chain: list[Any] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        if hasattr(current, "unwrapped") and current.unwrapped is not current:
            current = current.unwrapped
        elif hasattr(current, "env"):
            current = current.env
        else:
            break
    for owner in reversed(chain):
        if all(hasattr(owner, name) for name in ("fixtures", "objects", "sim")):
            return owner
    raise ValueError("could not find the underlying RoboCasa task environment")


def coerce_atomic_call(value: Any) -> Any:
    """Accept AtomicTaskCall or a mapping without importing it at module import."""

    if hasattr(value, "atomic_task") and hasattr(value, "arguments"):
        return value
    if isinstance(value, Mapping):
        from robocasa.atomic_task_schemas import AtomicTaskCall

        return AtomicTaskCall.from_mapping(value)
    raise TypeError("skill must be an AtomicTaskCall or mapping")
