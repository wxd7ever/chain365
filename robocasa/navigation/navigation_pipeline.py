"""End-to-end target-aware geometric navigation, independent of orchestrator."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from robocasa.held_object_guard import HeldObjectGuard
from robocasa.utils import object_utils
from robocasa.work_pose_dataset import current_base_pose

from .base_navigator import BaseNavigator
from .candidate_pose_generator import CandidatePoseGenerator
from .map_builder import MapBuilder, MapCache
from .navigation_types import (
    CandidatePose,
    NavigationResult,
    NavigationTarget,
    TargetType,
    coerce_atomic_call,
    raw_task_env,
)
from .path_planner import AStarPathPlanner, PathPlanningError
from .target_resolver import TargetResolver


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "navigation.yaml"


def load_navigation_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"navigation config {config_path} must be a mapping")
    required = {
        "map",
        "robot",
        "candidate_generation",
        "fixture_candidate",
        "candidate_scoring",
        "navigation",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"navigation config is missing sections: {missing}")
    return {str(key): dict(section) for key, section in value.items()}


class TargetAwareNavigationPipeline:
    """Resolve target, plan a coarse work pose, and optionally navigate there."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
        map_cache: MapCache | None = None,
        target_resolver: TargetResolver | None = None,
        path_planner: AStarPathPlanner | None = None,
        base_navigator: BaseNavigator | None = None,
    ) -> None:
        self.config = (
            {str(key): dict(value) for key, value in config.items()}
            if config is not None
            else load_navigation_config(config_path)
        )
        map_cfg = self.config["map"]
        robot_cfg = self.config["robot"]
        self.map_cache = map_cache if map_cache is not None else MapCache()
        offset = np.asarray(
            robot_cfg.get("footprint_center_offset_xy", (-0.20, 0.0)), dtype=float
        )
        if offset.shape != (2,) or not np.isfinite(offset).all():
            raise ValueError("robot.footprint_center_offset_xy must have two finite values")
        self.footprint_center_offset_xy = offset
        self.map_builder = MapBuilder(
            resolution=float(map_cfg["resolution"]),
            boundary_padding=float(map_cfg["boundary_padding"]),
            border_obstacle_cells=int(map_cfg.get("border_obstacle_cells", 1)),
            traversable_fixture_types=tuple(
                map_cfg.get("traversable_fixture_types", ("Floor",))
            ),
            robot_radius=float(robot_cfg["radius"]),
            robot_half_length=float(
                robot_cfg.get("half_length", robot_cfg["radius"])
            ),
            robot_half_width=float(
                robot_cfg.get("half_width", robot_cfg["radius"])
            ),
            safety_margin=float(robot_cfg["safety_margin"]),
            articulated_enabled=bool(map_cfg.get("articulated_enabled", True)),
            articulated_min_projected_area_m2=float(
                map_cfg.get("articulated_min_projected_area_m2", 0.001)
            ),
            articulated_min_projected_span_m=float(
                map_cfg.get("articulated_min_projected_span_m", 0.015)
            ),
            robot_collision_z_fallback=tuple(
                map_cfg.get("robot_collision_z_fallback", (0.0, 1.8))
            ),
            debug_articulated_map=bool(
                map_cfg.get("debug_articulated_map", False)
            ),
            cache=self.map_cache,
        )
        self.target_resolver = target_resolver or TargetResolver()
        self.candidate_generator = CandidatePoseGenerator(
            candidate_config=self.config["candidate_generation"],
            fixture_config=self.config["fixture_candidate"],
        )
        self.path_planner = path_planner or AStarPathPlanner()
        self.base_navigator = base_navigator or BaseNavigator(self.config["navigation"])
        self.last_debug: dict[str, Any] = {}

    def _footprint_center(self, base_xy: Any, yaw: float) -> np.ndarray:
        cosine, sine = math.cos(yaw), math.sin(yaw)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=float)
        return np.asarray(base_xy, dtype=float) + rotation @ self.footprint_center_offset_xy

    def _base_waypoints_from_center_path(
        self, center_path: Any, *, current_yaw: float, goal: CandidatePose
    ) -> list[tuple[float, float]]:
        cosine, sine = math.cos(current_yaw), math.sin(current_yaw)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=float)
        world_offset = rotation @ self.footprint_center_offset_xy
        result = [
            tuple((np.asarray(point, dtype=float) - world_offset).tolist())
            for point in center_path
        ]
        result[-1] = (goal.x, goal.y)
        return result

    @staticmethod
    def validate_path_against_current_map(
        occupancy_map: Any,
        collision_center_path_xy: Any,
    ) -> tuple[bool, dict[str, Any]]:
        """Validate every smoothed segment against one current inflated map."""

        points = np.asarray(collision_center_path_xy, dtype=float)
        if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] != 2:
            return False, {"reason": "EMPTY_OR_INVALID_PATH"}
        cells = [occupancy_map.world_to_grid(point) for point in points]
        for index, cell in enumerate(cells):
            if not occupancy_map.in_bounds(cell) or occupancy_map.grid[cell] != 0:
                return False, {
                    "reason": "PATH_POINT_IN_COLLISION",
                    "point_index": index,
                    "point_xy": points[index].tolist(),
                    "cell": list(cell),
                }
        for index, (first, second) in enumerate(zip(cells, cells[1:])):
            if not AStarPathPlanner.line_of_sight(
                occupancy_map.grid, first, second
            ):
                return False, {
                    "reason": "PATH_SEGMENT_IN_COLLISION",
                    "segment_index": index,
                    "start_cell": list(first),
                    "end_cell": list(second),
                }
        return True, {
            "reason": "PATH_VALID",
            "point_count": int(len(points)),
            "segment_count": max(0, int(len(points) - 1)),
        }

    def _replan_selected_goal(
        self,
        *,
        occupancy_map: Any,
        selected: CandidatePose,
        current_pose: Mapping[str, Any],
    ) -> None:
        """Replan once to the selected goal on a freshly rebuilt current map."""

        start_xy = np.asarray(current_pose["position"][:2], dtype=float)
        start_yaw = float(current_pose["yaw_rad"])
        start_collision_xy = self._footprint_center(start_xy, start_yaw)
        goal_collision_xy = self._footprint_center(selected.xy, selected.yaw)
        if not self.map_builder.footprint_is_free(
            occupancy_map, goal_collision_xy, selected.yaw
        ):
            raise PathPlanningError(
                "GOAL_IN_COLLISION",
                "selected candidate footprint is blocked by current scene state",
            )
        path = self.path_planner.plan(
            occupancy_map,
            start_collision_xy,
            goal_collision_xy,
            smooth=True,
        )
        selected.collision_center_path_xy = [
            (float(point[0]), float(point[1])) for point in path.world_xy
        ]
        selected.path_xy = self._base_waypoints_from_center_path(
            path.world_xy, current_yaw=start_yaw, goal=selected
        )
        selected.path_cost_m = path.cost_m

    def should_navigate(
        self,
        current_pose: Mapping[str, Any],
        target: NavigationTarget,
        occupancy_map: Any,
        *,
        grounding_result: Mapping[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        current_xy = np.asarray(current_pose["position"][:2], dtype=float)
        current_yaw = float(current_pose["yaw_rad"])
        footprint_center = self._footprint_center(current_xy, current_yaw)
        free = bool(occupancy_map.is_free_world(footprint_center))
        minimum, maximum = self.candidate_generator.distance_range(target)
        if target.target_type == TargetType.EXPOSED_OBJECT:
            operation_distance = float(np.linalg.norm(target.xy - current_xy))
            accessible_side = True
        else:
            fixture_alias = target.support_fixture or target.target_alias
            geometry = occupancy_map.fixture_geometries.get(fixture_alias)
            if geometry is None:
                return True, {"reason": "fixture_geometry_unavailable"}
            relative = current_xy - geometry.front_center_xy
            operation_distance = float(np.dot(relative, geometry.front_normal_xy))
            half_width = 0.5 * float(
                np.linalg.norm(geometry.polygon_xy[1] - geometry.polygon_xy[0])
            )
            lateral = abs(float(np.dot(relative, geometry.lateral_axis_xy)))
            accessible_side = operation_distance > 0 and lateral <= (
                half_width + self.candidate_generator.lateral_range
            )
        to_target = target.xy - current_xy
        distance_to_target = float(np.linalg.norm(to_target))
        if distance_to_target <= 1e-9:
            facing_cosine = -1.0
        else:
            heading = np.asarray([math.cos(current_yaw), math.sin(current_yaw)])
            facing_cosine = float(np.dot(heading, to_target / distance_to_target))
        threshold = float(self.config["navigation"]["skip_facing_cosine"])
        reasonable_distance = minimum <= operation_distance <= maximum
        # Occupancy is used to plan and validate base motion, but it must not
        # force navigation away from an otherwise valid manipulation pose.
        # Reset and post-skill poses close to fixtures can conservatively land
        # inside the inflated map even though the simulator pose is usable.
        geometric_operation_ready = bool(
            accessible_side
            and reasonable_distance
            and facing_cosine >= threshold
        )
        grounded_override = None
        if isinstance(grounding_result, Mapping) and isinstance(
            grounding_result.get("grounded"), (bool, np.bool_)
        ):
            grounded_override = bool(grounding_result["grounded"])
        # The orchestrator grounding decision is the single source of truth
        # when available. Retain geometry only as disagreement diagnostics.
        operation_ready = (
            geometric_operation_ready
            if grounded_override is None
            else grounded_override
        )
        navigate = not operation_ready
        return navigate, {
            "current_free": free,
            "operation_ready": operation_ready,
            "geometric_operation_ready": geometric_operation_ready,
            "decision_source": (
                "target_aware_geometry"
                if grounded_override is None
                else "grounding_result"
            ),
            "grounding_status": (
                grounding_result.get("status")
                if isinstance(grounding_result, Mapping)
                else None
            ),
            "footprint_center_xy": footprint_center.tolist(),
            "accessible_side": accessible_side,
            "operation_distance_m": operation_distance,
            "required_distance_range_m": [minimum, maximum],
            "facing_cosine": facing_cosine,
            "required_facing_cosine": threshold,
        }

    def _score_candidate(
        self,
        candidate: CandidatePose,
        target: NavigationTarget,
        occupancy_map: Any,
    ) -> float:
        cfg = self.config["candidate_scoring"]
        saturation = float(cfg["clearance_saturation_m"])
        clearance_score = min(candidate.clearance_m / saturation, 1.0)
        direction = target.xy - candidate.xy
        direction_norm = float(np.linalg.norm(direction))
        heading = np.asarray([math.cos(candidate.yaw), math.sin(candidate.yaw)])
        facing_score = (
            float(np.dot(heading, direction / direction_norm))
            if direction_norm > 1e-9
            else 0.0
        )
        preferred = self.candidate_generator.preferred_distance(target)
        minimum, maximum = self.candidate_generator.distance_range(target)
        scale = max(maximum - minimum, preferred, 1e-6)
        distance_penalty = abs(candidate.operation_distance - preferred) / scale
        bounds = occupancy_map.world_bounds
        map_diagonal = math.hypot(bounds[1] - bounds[0], bounds[3] - bounds[2])
        path_penalty = float(candidate.path_cost_m or 0.0) / max(map_diagonal, 1e-6)
        score = (
            float(cfg["w_clearance"]) * clearance_score
            + float(cfg["w_facing"]) * facing_score
            - float(cfg["w_distance"]) * distance_penalty
            - float(cfg["w_path"]) * path_penalty
        )
        candidate.score = score
        return score

    def _held_object_guard(self, env: Any) -> HeldObjectGuard | None:
        cfg = self.config["navigation"]
        if not bool(cfg.get("held_object_guard", True)):
            return None
        raw_env = raw_task_env(env)
        for alias in raw_env.objects:
            try:
                holding = bool(object_utils.check_obj_grasped(raw_env, alias))
            except (AssertionError, AttributeError, KeyError, TypeError, ValueError):
                holding = False
            if not holding:
                continue
            guard = HeldObjectGuard(
                enabled=True,
                object_alias=str(alias),
                holding_checker=lambda alias=str(alias): bool(
                    object_utils.check_obj_grasped(raw_env, alias)
                ),
                hold_confirmation_steps=int(cfg.get("hold_confirmation_steps", 2)),
                drop_confirmation_steps=int(cfg.get("drop_confirmation_steps", 2)),
                action_override_enabled=bool(
                    cfg.get("held_object_guard_action_override", True)
                ),
                reason="target_aware_navigation_currently_held_object",
            )
            guard.start()
            return guard
        return None

    def restore_coarse_pose(
        self,
        env: Any,
        coarse_pose: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        """Return to a recorded coarse pose without resolving or replanning globally."""

        current = current_base_pose(env)
        context = dict(context or {})
        guard = context.get("held_object_guard") or self._held_object_guard(env)
        return self.base_navigator.execute_path(
            env=env,
            path_world=[current["position"][:2], coarse_pose["position"][:2]],
            goal_yaw=float(coarse_pose["yaw_rad"]),
            held_object_context=guard,
        )

    def navigate_to_manipulation_target(
        self,
        env: Any,
        skill: Any,
        target: Any | None = None,
        context: Mapping[str, Any] | None = None,
        *,
        plan_only: bool = False,
        before_navigation: Callable[[], Mapping[str, Any]] | None = None,
    ) -> NavigationResult:
        call = coerce_atomic_call(skill)
        context = dict(context or {})
        # Avoid accidentally saving a prior call's map when target resolution
        # fails before the current occupancy map can be constructed.
        self.last_debug = {}
        try:
            resolved_target = self.target_resolver.resolve(env, call, target)
        except LookupError as exc:
            reason = "TARGET_NOT_FOUND"
            return NavigationResult(
                False, reason, plan_only=plan_only, diagnostics={"error": str(exc)}
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            return NavigationResult(
                False, "TARGET_POSE_UNAVAILABLE", diagnostics={"error": str(exc)}, plan_only=plan_only
            )

        current = current_base_pose(env)
        start_xy = np.asarray(current["position"][:2], dtype=float)
        start_yaw = float(current["yaw_rad"])
        start_collision_xy = self._footprint_center(start_xy, start_yaw)
        raw_env = raw_task_env(env)
        layout_id = int(context.get("layout_id", getattr(raw_env, "layout_id", -1)))
        debug_articulated_map = bool(
            context.get("debug_articulated_map", False)
        )
        occupancy_map = self.map_builder.build(
            env,
            layout_id=layout_id,
            robot_xy=start_xy,
            robot_yaw=start_yaw,
            debug_articulated_map=debug_articulated_map,
        )
        grounding_result = context.get("grounding_result")
        navigate, skip_evidence = self.should_navigate(
            current,
            resolved_target,
            occupancy_map,
            grounding_result=(
                grounding_result
                if isinstance(grounding_result, Mapping)
                else None
            ),
        )
        self.last_debug = {
            "map": occupancy_map,
            "target": resolved_target,
            "start_xy": start_xy,
            "start_collision_xy": start_collision_xy,
            "raw_candidates": [],
            "valid_candidates": [],
            "selected": None,
            "path_xy": [],
            "collision_center_path_xy": [],
            "skip_evidence": skip_evidence,
        }
        if not navigate:
            return NavigationResult(
                True,
                "ALREADY_AT_WORK_REGION",
                target_name=resolved_target.target_name,
                target_type=resolved_target.target_type.value,
                target_position=list(resolved_target.world_position),
                skipped=True,
                plan_only=plan_only,
                diagnostics={"should_navigate": skip_evidence},
            )
        start_cell = occupancy_map.world_to_grid(start_collision_xy)
        if not occupancy_map.in_bounds(start_cell) or occupancy_map.grid[start_cell] != 0:
            return NavigationResult(
                False,
                "START_IN_COLLISION",
                target_name=resolved_target.target_name,
                target_type=resolved_target.target_type.value,
                target_position=list(resolved_target.world_position),
                plan_only=plan_only,
                diagnostics={
                    "start_xy": start_xy.tolist(),
                    "footprint_center_xy": start_collision_xy.tolist(),
                    "start_cell": list(start_cell),
                    "should_navigate": skip_evidence,
                },
            )

        try:
            raw_candidates = self.candidate_generator.generate(
                resolved_target, occupancy_map
            )
        except (KeyError, TypeError, ValueError) as exc:
            return NavigationResult(
                False,
                "NO_VALID_WORK_POSE",
                target_name=resolved_target.target_name,
                target_type=resolved_target.target_type.value,
                target_position=list(resolved_target.world_position),
                plan_only=plan_only,
                diagnostics={"error": str(exc)},
            )
        occupancy_candidates = self.candidate_generator.filter_occupancy(
            raw_candidates,
            occupancy_map,
            position_fn=lambda candidate: self._footprint_center(
                candidate.xy, candidate.yaw
            ),
            footprint_validator=(
                (lambda candidate, center: self.map_builder.footprint_is_free(
                    occupancy_map, center, candidate.yaw
                ))
                if hasattr(self.map_builder, "footprint_is_free")
                else None
            ),
        )
        valid: list[CandidatePose] = []
        for candidate in occupancy_candidates:
            try:
                path = self.path_planner.plan(
                    occupancy_map,
                    start_collision_xy,
                    self._footprint_center(candidate.xy, candidate.yaw),
                    smooth=True,
                )
            except PathPlanningError as exc:
                candidate.valid = False
                candidate.rejection_reason = exc.code
                continue
            candidate.collision_center_path_xy = [
                (float(point[0]), float(point[1])) for point in path.world_xy
            ]
            candidate.path_xy = self._base_waypoints_from_center_path(
                path.world_xy, current_yaw=start_yaw, goal=candidate
            )
            candidate.path_cost_m = path.cost_m
            self._score_candidate(candidate, resolved_target, occupancy_map)
            valid.append(candidate)
        self.last_debug.update(
            {"raw_candidates": raw_candidates, "valid_candidates": valid}
        )
        if not valid:
            failure = (
                "PATH_NOT_FOUND" if occupancy_candidates else "NO_VALID_WORK_POSE"
            )
            return NavigationResult(
                False,
                failure,
                target_name=resolved_target.target_name,
                target_type=resolved_target.target_type.value,
                target_position=list(resolved_target.world_position),
                raw_candidate_count=len(raw_candidates),
                valid_candidate_count=0,
                plan_only=plan_only,
                diagnostics={
                    "occupancy_valid_candidate_count": len(occupancy_candidates),
                    "should_navigate": skip_evidence,
                },
            )
        selected = max(valid, key=lambda value: float(value.score))
        self.last_debug.update(
            {
                "selected": selected,
                "path_xy": selected.path_xy,
                "collision_center_path_xy": selected.collision_center_path_xy,
            }
        )
        result = NavigationResult(
            True,
            "PLANNED" if plan_only else "NAVIGATION_PENDING",
            target_name=resolved_target.target_name,
            target_type=resolved_target.target_type.value,
            target_position=list(resolved_target.world_position),
            selected_goal=selected.to_dict(),
            path_length=float(selected.path_cost_m or 0.0),
            waypoint_count=len(selected.path_xy),
            plan_only=plan_only,
            raw_candidate_count=len(raw_candidates),
            valid_candidate_count=len(valid),
            diagnostics={
                "should_navigate": skip_evidence,
                "map": occupancy_map.to_dict(),
            },
        )
        if plan_only:
            return result

        preparation_result = None
        if before_navigation is not None:
            preparation_result = dict(before_navigation())
            result.diagnostics["navigation_preparation"] = preparation_result
            module = preparation_result.get("module")
            if (
                module == "POST_GRASP_TRANSPORT"
                or "transport_ready" in preparation_result
            ):
                result.diagnostics["post_grasp_transport"] = preparation_result
            elif module == "POST_PLACE_RETREAT":
                result.diagnostics["post_place_retreat"] = preparation_result
            navigation_safe = preparation_result.get("navigation_safe")
            if navigation_safe is None:
                navigation_safe = preparation_result.get(
                    "transport_ready", False
                )
            if not navigation_safe:
                result.success = False
                result.reason = str(
                    preparation_result.get("failure_code")
                    or "PRE_NAVIGATION_PREPARATION_FAILED"
                )
                return result

        guard = context.get("held_object_guard") or self._held_object_guard(env)
        transport_cfg = self.config.get("post_grasp_transport", {})
        enforce_transport_ready = bool(
            transport_cfg.get("enforce_before_navigation", False)
        )
        if (
            enforce_transport_ready
            and guard is not None
            and bool(getattr(guard, "enabled", False))
        ):
            guarded_alias = str(getattr(guard, "object_alias", "") or "")
            ready_aliases = {
                str(value)
                for value in context.get("transport_ready_objects", ())
            }
            prepared_alias = (
                str(preparation_result.get("object_alias", ""))
                if preparation_result is not None
                else ""
            )
            prepared_now = bool(
                preparation_result is not None
                and preparation_result.get("transport_ready", False)
                and prepared_alias == guarded_alias
            )
            if guarded_alias not in ready_aliases and not prepared_now:
                result.success = False
                result.reason = "TRANSPORT_NOT_READY"
                result.diagnostics["transport_ready_gate"] = {
                    "required": True,
                    "held_object": guarded_alias,
                    "ready_objects": sorted(ready_aliases),
                    "prepared_now": prepared_now,
                }
                return result
        # The arm preparation above advances MuJoCo and a fixture could have
        # changed state since planning. Rebuild only the live articulated layer,
        # then validate the exact smoothed collision-center path before moving.
        validation_pose = current_base_pose(env)
        validation_xy = np.asarray(validation_pose["position"][:2], dtype=float)
        validation_yaw = float(validation_pose["yaw_rad"])
        current_map = self.map_builder.build(
            env,
            layout_id=layout_id,
            robot_xy=validation_xy,
            robot_yaw=validation_yaw,
            debug_articulated_map=debug_articulated_map,
        )
        path_valid, path_validation = self.validate_path_against_current_map(
            current_map, selected.collision_center_path_xy
        )
        validation_diagnostics = {
            "initial_validation": path_validation,
            "replan_attempted": False,
            "replan_succeeded": False,
            "current_map": current_map.to_dict(),
        }
        if not path_valid:
            validation_diagnostics["replan_attempted"] = True
            try:
                self._replan_selected_goal(
                    occupancy_map=current_map,
                    selected=selected,
                    current_pose=validation_pose,
                )
            except PathPlanningError as exc:
                result.success = False
                result.reason = "PATH_INVALIDATED_BY_SCENE_STATE"
                validation_diagnostics["replan_error"] = {
                    "code": exc.code,
                    "message": str(exc),
                }
                result.diagnostics["final_path_validation"] = validation_diagnostics
                return result
            replanned_valid, replanned_validation = (
                self.validate_path_against_current_map(
                    current_map, selected.collision_center_path_xy
                )
            )
            validation_diagnostics["replanned_validation"] = replanned_validation
            validation_diagnostics["replan_succeeded"] = bool(replanned_valid)
            if not replanned_valid:
                result.success = False
                result.reason = "PATH_INVALIDATED_BY_SCENE_STATE"
                result.diagnostics["final_path_validation"] = validation_diagnostics
                return result
            result.selected_goal = selected.to_dict()
            result.path_length = float(selected.path_cost_m or 0.0)
            result.waypoint_count = len(selected.path_xy)
        result.diagnostics["final_path_validation"] = validation_diagnostics
        result.diagnostics["map"] = current_map.to_dict()
        self.last_debug.update(
            {
                "map": current_map,
                "start_xy": validation_xy,
                "start_collision_xy": self._footprint_center(
                    validation_xy, validation_yaw
                ),
                "selected": selected,
                "path_xy": selected.path_xy,
                "collision_center_path_xy": selected.collision_center_path_xy,
            }
        )
        execution = self.base_navigator.navigate_path(
            env, selected.path_xy, selected.yaw, guard=guard
        )
        result.execution = execution
        result.success = bool(execution.get("success", False))
        result.reason = (
            "SUCCESS" if result.success else str(execution.get("failure_code"))
        )
        return result


def navigate_to_manipulation_target(
    env: Any,
    skill: Any,
    target: Any | None = None,
    context: Mapping[str, Any] | None = None,
    *,
    plan_only: bool = False,
    before_navigation: Callable[[], Mapping[str, Any]] | None = None,
    config_path: str | Path | None = None,
) -> NavigationResult:
    """Convenience entry point for standalone callers and future orchestrator use."""

    return TargetAwareNavigationPipeline(config_path=config_path).navigate_to_manipulation_target(
        env,
        skill,
        target,
        context,
        plan_only=plan_only,
        before_navigation=before_navigation,
    )
