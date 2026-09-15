"""Generate task-conditioned coarse manipulation work-pose candidates."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import numpy as np

from .navigation_types import CandidatePose, NavigationTarget, OccupancyMap, TargetType


def _frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("candidate step must be positive")
    count = int(math.floor((stop - start) / step + 1e-9))
    values = [start + index * step for index in range(count + 1)]
    if not values or values[-1] < stop - 1e-8:
        values.append(stop)
    return values


class CandidatePoseGenerator:
    """Generate rings for exposed objects and front strips for fixtures."""

    def __init__(
        self,
        *,
        candidate_config: Mapping[str, Any],
        fixture_config: Mapping[str, Any],
    ) -> None:
        self.object_min = float(candidate_config["min_distance"])
        self.object_preferred = float(candidate_config["preferred_distance"])
        self.object_max = float(candidate_config["max_distance"])
        self.radial_step = float(candidate_config["radial_step"])
        self.angular_step_deg = float(candidate_config["angular_step_deg"])
        self.fixture_min = float(fixture_config["min_distance"])
        self.fixture_preferred = float(fixture_config["preferred_distance"])
        self.fixture_max = float(fixture_config["max_distance"])
        self.fixture_distance_step = float(
            fixture_config.get("distance_step", self.radial_step)
        )
        self.lateral_range = float(fixture_config["lateral_range"])
        self.lateral_step = float(fixture_config["lateral_step"])
        if not (
            0 < self.object_min <= self.object_preferred <= self.object_max
            and 0 < self.fixture_min <= self.fixture_preferred <= self.fixture_max
        ):
            raise ValueError("candidate distance ranges are inconsistent")
        if min(
            self.radial_step,
            self.angular_step_deg,
            self.fixture_distance_step,
            self.lateral_step,
        ) <= 0:
            raise ValueError("candidate sampling steps must be positive")

    def preferred_distance(self, target: NavigationTarget) -> float:
        return (
            self.object_preferred
            if target.target_type == TargetType.EXPOSED_OBJECT
            else self.fixture_preferred
        )

    def distance_range(self, target: NavigationTarget) -> tuple[float, float]:
        if target.target_type == TargetType.EXPOSED_OBJECT:
            return self.object_min, self.object_max
        return self.fixture_min, self.fixture_max

    def generate(
        self, target: NavigationTarget, occupancy_map: OccupancyMap
    ) -> list[CandidatePose]:
        if target.target_type == TargetType.EXPOSED_OBJECT:
            return self._object_ring(target)
        fixture_alias = target.support_fixture or target.target_alias
        geometry = occupancy_map.fixture_geometries.get(fixture_alias)
        if geometry is None:
            raise ValueError(
                f"target fixture {fixture_alias!r} has no occupancy geometry"
            )
        candidates: list[CandidatePose] = []
        for distance in _frange(
            self.fixture_min, self.fixture_max, self.fixture_distance_step
        ):
            for lateral in _frange(
                -self.lateral_range, self.lateral_range, self.lateral_step
            ):
                xy = (
                    geometry.front_center_xy
                    + geometry.front_normal_xy * distance
                    + geometry.lateral_axis_xy * lateral
                )
                yaw = math.atan2(target.xy[1] - xy[1], target.xy[0] - xy[0])
                candidates.append(
                    CandidatePose(
                        x=float(xy[0]),
                        y=float(xy[1]),
                        yaw=yaw,
                        operation_distance=distance,
                        source="fixture_front_strip",
                    )
                )
        return candidates

    def _object_ring(self, target: NavigationTarget) -> list[CandidatePose]:
        candidates: list[CandidatePose] = []
        angle_count = max(1, int(math.ceil(360.0 / self.angular_step_deg)))
        for radius in _frange(self.object_min, self.object_max, self.radial_step):
            for index in range(angle_count):
                angle = math.radians(index * 360.0 / angle_count)
                xy = target.xy + radius * np.asarray(
                    [math.cos(angle), math.sin(angle)], dtype=float
                )
                yaw = math.atan2(target.xy[1] - xy[1], target.xy[0] - xy[0])
                candidates.append(
                    CandidatePose(
                        x=float(xy[0]),
                        y=float(xy[1]),
                        yaw=yaw,
                        operation_distance=radius,
                        source="object_ring",
                    )
                )
        return candidates

    @staticmethod
    def filter_occupancy(
        candidates: list[CandidatePose],
        occupancy_map: OccupancyMap,
        *,
        position_fn: Callable[[CandidatePose], np.ndarray] | None = None,
        footprint_validator: Callable[[CandidatePose, np.ndarray], bool] | None = None,
    ) -> list[CandidatePose]:
        valid: list[CandidatePose] = []
        for candidate in candidates:
            checked_xy = (
                candidate.xy
                if position_fn is None
                else np.asarray(position_fn(candidate), dtype=float)
            )
            cell = occupancy_map.world_to_grid(checked_xy)
            if not occupancy_map.in_bounds(cell):
                candidate.rejection_reason = "OUT_OF_BOUNDS"
                continue
            if footprint_validator is None and occupancy_map.grid[cell] != 0:
                candidate.rejection_reason = "GOAL_IN_COLLISION"
                continue
            if footprint_validator is not None and not footprint_validator(
                candidate, checked_xy
            ):
                candidate.rejection_reason = "GOAL_FOOTPRINT_IN_COLLISION"
                continue
            candidate.clearance_m = occupancy_map.clearance_at_world(checked_xy)
            candidate.valid = True
            valid.append(candidate)
        return valid
