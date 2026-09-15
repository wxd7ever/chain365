"""Eight-connected A* and grid line-of-sight path shortcutting."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .navigation_types import OccupancyMap


@dataclass(frozen=True)
class PlannedPath:
    cells: tuple[tuple[int, int], ...]
    world_xy: tuple[tuple[float, float], ...]
    cost_m: float


class PathPlanningError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class AStarPathPlanner:
    _NEIGHBORS = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    @staticmethod
    def _heuristic(a: Sequence[int], b: Sequence[int]) -> float:
        dx, dy = abs(a[1] - b[1]), abs(a[0] - b[0])
        return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)

    @staticmethod
    def _line_cells(a: Sequence[int], b: Sequence[int]) -> list[tuple[int, int]]:
        """Return a conservative supercover of cells crossed by one segment."""

        y0, x0 = int(a[0]), int(a[1])
        y1, x1 = int(b[0]), int(b[1])
        dx, dy = x1 - x0, y1 - y0
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return [(y0, x0)]
        result: list[tuple[int, int]] = []
        for index in range(steps + 1):
            fraction = index / steps
            cell = (int(round(y0 + fraction * dy)), int(round(x0 + fraction * dx)))
            if not result or result[-1] != cell:
                result.append(cell)
        return result

    @classmethod
    def line_of_sight(
        cls, grid: np.ndarray, a: Sequence[int], b: Sequence[int]
    ) -> bool:
        cells = cls._line_cells(a, b)
        for row, col in cells:
            if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
                return False
            if grid[row, col] != 0:
                return False
        # Forbid squeezing diagonally between two touching obstacles.
        for first, second in zip(cells, cells[1:]):
            dr, dc = second[0] - first[0], second[1] - first[1]
            if abs(dr) == 1 and abs(dc) == 1:
                if grid[first[0] + dr, first[1]] or grid[first[0], first[1] + dc]:
                    return False
        return True

    def plan_cells(
        self,
        grid: np.ndarray,
        start: Sequence[int],
        goal: Sequence[int],
    ) -> list[tuple[int, int]]:
        grid = np.asarray(grid, dtype=np.uint8)
        start_cell = (int(start[0]), int(start[1]))
        goal_cell = (int(goal[0]), int(goal[1]))
        for cell, code in ((start_cell, "START_IN_COLLISION"), (goal_cell, "GOAL_IN_COLLISION")):
            if not (0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]):
                raise PathPlanningError(code, f"cell {cell} is outside the map")
            if grid[cell] != 0:
                raise PathPlanningError(code, f"cell {cell} is occupied")
        if start_cell == goal_cell:
            return [start_cell]

        frontier: list[tuple[float, float, tuple[int, int]]] = []
        heapq.heappush(frontier, (self._heuristic(start_cell, goal_cell), 0.0, start_cell))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        costs = {start_cell: 0.0}
        while frontier:
            _, current_cost, current = heapq.heappop(frontier)
            if current == goal_cell:
                break
            if current_cost > costs.get(current, float("inf")) + 1e-12:
                continue
            for dr, dc, move_cost in self._NEIGHBORS:
                neighbor = (current[0] + dr, current[1] + dc)
                if not (
                    0 <= neighbor[0] < grid.shape[0]
                    and 0 <= neighbor[1] < grid.shape[1]
                ):
                    continue
                if grid[neighbor] != 0:
                    continue
                if dr and dc and (
                    grid[current[0] + dr, current[1]] != 0
                    or grid[current[0], current[1] + dc] != 0
                ):
                    continue
                new_cost = current_cost + move_cost
                if new_cost + 1e-12 >= costs.get(neighbor, float("inf")):
                    continue
                costs[neighbor] = new_cost
                came_from[neighbor] = current
                priority = new_cost + self._heuristic(neighbor, goal_cell)
                heapq.heappush(frontier, (priority, new_cost, neighbor))
        if goal_cell not in came_from:
            raise PathPlanningError("PATH_NOT_FOUND", "A* frontier was exhausted")
        path = [goal_cell]
        while path[-1] != start_cell:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

    def smooth_cells(
        self, grid: np.ndarray, cells: Iterable[Sequence[int]]
    ) -> list[tuple[int, int]]:
        path = [(int(cell[0]), int(cell[1])) for cell in cells]
        if len(path) <= 2:
            return path
        result = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            candidate = len(path) - 1
            while candidate > anchor + 1 and not self.line_of_sight(
                grid, path[anchor], path[candidate]
            ):
                candidate -= 1
            result.append(path[candidate])
            anchor = candidate
        return result

    def plan(
        self,
        occupancy_map: OccupancyMap,
        start_xy: Sequence[float],
        goal_xy: Sequence[float],
        *,
        smooth: bool = True,
    ) -> PlannedPath:
        cells = self.plan_cells(
            occupancy_map.grid,
            occupancy_map.world_to_grid(start_xy),
            occupancy_map.world_to_grid(goal_xy),
        )
        output_cells = self.smooth_cells(occupancy_map.grid, cells) if smooth else cells
        world = [occupancy_map.grid_to_world(cell) for cell in output_cells]
        world[0] = (float(start_xy[0]), float(start_xy[1]))
        world[-1] = (float(goal_xy[0]), float(goal_xy[1]))
        cost = sum(
            math.dist(first, second) for first, second in zip(world, world[1:])
        )
        return PlannedPath(tuple(output_cells), tuple(world), float(cost))
