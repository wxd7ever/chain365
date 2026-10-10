from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robocasa.atomic_task_schemas import AtomicTaskCall
from robocasa.navigation.navigation_pipeline import TargetAwareNavigationPipeline
from robocasa.navigation.navigation_types import NavigationTarget, OccupancyMap, TargetType
from robocasa.navigation.path_planner import PathPlanningError


CONFIG = {
    "map": {"resolution": 0.1, "boundary_padding": 1.0, "border_obstacle_cells": 0},
    "robot": {"radius": 0.1, "safety_margin": 0.0},
    "candidate_generation": {
        "min_distance": 0.5,
        "preferred_distance": 0.7,
        "max_distance": 0.9,
        "radial_step": 0.2,
        "angular_step_deg": 90,
    },
    "fixture_candidate": {
        "min_distance": 0.45,
        "preferred_distance": 0.7,
        "max_distance": 0.95,
        "distance_step": 0.25,
        "lateral_range": 0.2,
        "lateral_step": 0.2,
    },
    "candidate_scoring": {
        "w_clearance": 1.0,
        "w_facing": 0.5,
        "w_distance": 1.0,
        "w_path": 0.75,
        "clearance_saturation_m": 1.0,
    },
    "navigation": {
        "waypoint_tolerance": 0.12,
        "goal_position_tolerance": 0.08,
        "goal_yaw_tolerance_deg": 10,
        "waypoint_max_steps": 10,
        "goal_max_steps": 10,
        "max_translation_command": 0.5,
        "max_rotation_command": 0.5,
        "skip_facing_cosine": 0.5,
        "held_object_guard": False,
    },
}


class FakeEnv:
    def __init__(self):
        self.fixtures = {}
        self.objects = {}
        self.sim = object()
        self.layout_id = 1

    def get_observation(self):
        return {
            "robot0_base_pos": np.array([-2.0, 0.0, 0.7]),
            "robot0_base_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        }


class Resolver:
    def __init__(self, error=None):
        self.error = error

    def resolve(self, env, call, target):
        if self.error:
            raise self.error
        return NavigationTarget(
            TargetType.EXPOSED_OBJECT, "bowl", "bowl", (0.0, 0.0, 0.8)
        )


class Builder:
    def __init__(self, grid=None):
        self.map = OccupancyMap(
            grid=np.zeros((60, 60), dtype=np.uint8) if grid is None else grid,
            raw_grid=np.zeros((60, 60), dtype=np.uint8),
            clearance_m=np.ones((60, 60)),
            resolution=0.1,
            origin_x=-3.0,
            origin_y=-3.0,
        )

    def build(self, *args, **kwargs):
        return self.map


def call():
    return AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "pick",
            "atomic_task": "PickObject",
            "arguments": {"object_id": "bowl"},
            "termination_condition": {
                "predicate": "holding",
                "subject": "bowl",
                "desired_value": True,
            },
        }
    )


def pipeline(resolver=None, builder=None):
    value = TargetAwareNavigationPipeline(config=CONFIG, target_resolver=resolver or Resolver())
    value.map_builder = builder or Builder()
    return value


def test_pipeline_reports_target_not_found():
    result = pipeline(resolver=Resolver(LookupError("missing"))).navigate_to_manipulation_target(
        FakeEnv(), call(), plan_only=True
    )
    assert not result.success
    assert result.reason == "TARGET_NOT_FOUND"


def test_pipeline_reports_no_valid_work_pose():
    grid = np.ones((60, 60), dtype=np.uint8)
    # Current Omron footprint center is 0.20 m behind the base control origin.
    grid[29:32, 6:10] = 0
    builder = Builder(grid)
    builder.map.raw_grid = grid.copy()
    builder.map.clearance_m = np.zeros_like(grid, dtype=float)
    result = pipeline(builder=builder).navigate_to_manipulation_target(
        FakeEnv(), call(), plan_only=True
    )
    assert not result.success
    assert result.reason == "NO_VALID_WORK_POSE"


def test_pipeline_reports_path_not_found(monkeypatch):
    value = pipeline()
    monkeypatch.setattr(
        value.path_planner,
        "plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PathPlanningError("PATH_NOT_FOUND", "blocked")
        ),
    )
    result = value.navigate_to_manipulation_target(FakeEnv(), call(), plan_only=True)
    assert not result.success
    assert result.reason == "PATH_NOT_FOUND"


def test_pipeline_plan_only_selects_reachable_goal():
    result = pipeline().navigate_to_manipulation_target(
        FakeEnv(), call(), plan_only=True
    )
    assert result.success
    assert result.reason == "PLANNED"
    assert result.selected_goal is not None
    assert result.valid_candidate_count > 0
    assert result.waypoint_count >= 2


def test_pipeline_refuses_held_object_without_transport_ready(monkeypatch):
    value = pipeline()
    value.config["post_grasp_transport"] = {
        "enforce_before_navigation": True
    }
    guard = SimpleNamespace(enabled=True, object_alias="bowl")
    monkeypatch.setattr(value, "_held_object_guard", lambda env: guard)
    monkeypatch.setattr(
        value.base_navigator,
        "navigate_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("base motion must not start")
        ),
    )
    result = value.navigate_to_manipulation_target(FakeEnv(), call())
    assert not result.success
    assert result.reason == "TRANSPORT_NOT_READY"
    assert result.execution is None


def test_pipeline_prepares_held_object_before_base_motion(monkeypatch):
    value = pipeline()
    value.config["post_grasp_transport"] = {
        "enforce_before_navigation": True
    }
    guard = SimpleNamespace(enabled=True, object_alias="bowl")
    events = []
    monkeypatch.setattr(value, "_held_object_guard", lambda env: guard)

    def navigate(*args, **kwargs):
        events.append("navigate")
        assert kwargs["guard"] is guard
        return {"success": True, "failure_code": None, "steps": 3}

    monkeypatch.setattr(value.base_navigator, "navigate_path", navigate)

    def prepare():
        events.append("prepare")
        return {
            "success": True,
            "transport_ready": True,
            "object_alias": "bowl",
            "reason": "TRANSPORT_READY",
        }


    result = value.navigate_to_manipulation_target(
        FakeEnv(), call(), before_navigation=prepare
    )
    assert result.success
    assert result.reason == "SUCCESS"
    assert events == ["prepare", "navigate"]
    assert result.diagnostics["post_grasp_transport"]["transport_ready"]


class RefreshingBuilder(Builder):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.blocked = OccupancyMap(
            grid=np.ones_like(self.map.grid),
            raw_grid=np.ones_like(self.map.raw_grid),
            clearance_m=np.zeros_like(self.map.clearance_m),
            resolution=self.map.resolution,
            origin_x=self.map.origin_x,
            origin_y=self.map.origin_y,
        )

    def build(self, *args, **kwargs):
        self.calls += 1
        return self.map if self.calls == 1 else self.blocked

    def footprint_is_free(self, occupancy_map, center, yaw):
        return occupancy_map.is_free_world(center)


def test_execution_revalidates_current_map_and_replans_only_once(monkeypatch):
    refreshed = RefreshingBuilder()
    value = pipeline(builder=refreshed)
    monkeypatch.setattr(
        value.base_navigator,
        "navigate_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalidated path must not execute")
        ),
    )
    result = value.navigate_to_manipulation_target(FakeEnv(), call())
    assert not result.success
    assert result.reason == "PATH_INVALIDATED_BY_SCENE_STATE"
    assert refreshed.calls == 2
    validation = result.diagnostics["final_path_validation"]
    assert validation["replan_attempted"] is True
    assert validation["replan_succeeded"] is False
    assert validation["replan_error"]["code"] == "GOAL_IN_COLLISION"


def test_start_collision_recovery_prefers_shortest_monotonic_escape():
    config = {
        **CONFIG,
        "navigation": {
            **CONFIG["navigation"],
            "start_collision_recovery": {
                "enabled": True,
                "search_step_m": 0.1,
                "execution_step_m": 0.05,
                "max_distance_m": 0.3,
                "direction_order": ["backward", "left", "right", "forward"],
            },
        },
    }
    value = TargetAwareNavigationPipeline(config=config, target_resolver=Resolver())
    occupancy = OccupancyMap(
        grid=np.zeros((40, 40), dtype=np.uint8),
        raw_grid=np.zeros((40, 40), dtype=np.uint8),
        clearance_m=np.ones((40, 40), dtype=float),
        resolution=0.1,
        origin_x=-2.0,
        origin_y=-2.0,
    )
    for obstacle_y in np.arange(-0.2, 0.21, 0.1):
        occupancy.raw_grid[
            occupancy.world_to_grid((-0.1, float(obstacle_y)))
        ] = 1

    recovery = value._plan_start_collision_recovery(
        occupancy, base_xy=(0.0, 0.0), yaw=0.0
    )

    assert recovery is not None
    assert recovery["direction"] == "backward"
    overlaps = [
        recovery["initial_overlap_cells"],
        *[
            sample["overlap_cells"]
            for sample in recovery["search_samples"]
        ],
    ]
    assert overlaps[-1] == 0
    assert overlaps == sorted(overlaps, reverse=True)
    assert recovery["execution_points"][-1][0] < 0.0


def test_start_collision_prepares_carry_pose_before_escape_and_only_once(
    monkeypatch,
):
    config = {
        **CONFIG,
        "navigation": {
            **CONFIG["navigation"],
            "start_collision_recovery": {"enabled": True},
        },
    }
    occupied_builder = Builder()
    start_center = (-2.2, 0.0)
    start_cell = occupied_builder.map.world_to_grid(start_center)
    occupied_builder.map.grid[start_cell] = 1
    occupied_builder.map.raw_grid[start_cell] = 1
    free_map = Builder().map

    class SequenceBuilder:
        def __init__(self):
            self.calls = 0

        def build(self, *args, **kwargs):
            self.calls += 1
            return occupied_builder.map if self.calls <= 2 else free_map

    builder = SequenceBuilder()
    value = TargetAwareNavigationPipeline(config=config, target_resolver=Resolver())
    value.map_builder = builder
    events = []

    def prepare():
        events.append("carry")
        return {
            "module": "POST_SKILL_RETURN",
            "success": True,
            "return_ready": True,
            "object_alias": "bowl",
        }

    def recover(**kwargs):
        events.append("escape")
        assert kwargs["occupancy_map"] is occupied_builder.map
        return (
            True,
            "START_COLLISION_RECOVERED",
            free_map,
            {"position": [-2.0, 0.0, 0.7], "yaw_rad": 0.0},
            {"success": True, "reason": "START_COLLISION_RECOVERED"},
        )

    def navigate(*args, **kwargs):
        events.append("navigate")
        return {"success": True, "failure_code": None, "steps": 2}

    monkeypatch.setattr(value, "_recover_start_collision", recover)
    monkeypatch.setattr(value.base_navigator, "navigate_path", navigate)

    result = value.navigate_to_manipulation_target(
        FakeEnv(), call(), before_navigation=prepare
    )

    assert result.success
    assert result.reason == "SUCCESS"
    assert events == ["carry", "escape", "navigate"]
    assert builder.calls == 3
    assert result.diagnostics["navigation_preparation"]["return_ready"]
    assert result.diagnostics["start_collision_recovery"]["success"]


def test_pipeline_skips_preparation_when_navigation_is_not_required(
    monkeypatch,
):
    value = pipeline()
    monkeypatch.setattr(
        value,
        "should_navigate",
        lambda *args, **kwargs: (False, {"reason": "test"}),
    )
    result = value.navigate_to_manipulation_target(
        FakeEnv(),
        call(),
        before_navigation=lambda: (_ for _ in ()).throw(
            AssertionError("preparation must not run")
        ),
    )
    assert result.success
    assert result.skipped
    assert result.reason == "ALREADY_AT_WORK_REGION"


def test_pipeline_keeps_valid_operation_pose_when_inflated_map_marks_it_occupied(
):
    value = pipeline()
    occupied_map = OccupancyMap(
        grid=np.ones((60, 60), dtype=np.uint8),
        raw_grid=np.ones((60, 60), dtype=np.uint8),
        clearance_m=np.zeros((60, 60), dtype=float),
        resolution=0.1,
        origin_x=-3.0,
        origin_y=-3.0,
    )
    target = NavigationTarget(
        TargetType.EXPOSED_OBJECT, "bowl", "bowl", (0.0, 0.0, 0.8)
    )

    navigate, evidence = value.should_navigate(
        {
            "position": [-0.7, 0.0, 0.7],
            "yaw_rad": 0.0,
        },
        target,
        occupied_map,
    )

    assert navigate is False
    assert evidence["current_free"] is False
    assert evidence["operation_ready"] is True


def test_pipeline_uses_grounding_as_readiness_source():
    value = pipeline()
    target = NavigationTarget(
        TargetType.EXPOSED_OBJECT, "bowl", "bowl", (0.0, 0.0, 0.8)
    )
    current = {"position": [-2.0, 0.0, 0.7], "yaw_rad": 0.0}

    navigate, evidence = value.should_navigate(
        current,
        target,
        value.map_builder.map,
        grounding_result={"grounded": True, "status": "grounded"},
    )

    assert navigate is False
    assert evidence["operation_ready"] is True
    assert evidence["geometric_operation_ready"] is False
    assert evidence["decision_source"] == "grounding_result"


def test_navigation_guard_starts_latched_and_protects_gripper(monkeypatch):
    config = {
        **CONFIG,
        "navigation": {
            **CONFIG["navigation"],
            "held_object_guard": True,
            "held_object_guard_action_override": True,
        },
    }
    env = FakeEnv()
    env.objects = {"bowl": object()}
    monkeypatch.setattr(
        "robocasa.navigation.navigation_pipeline.check_two_finger_grasp",
        lambda raw_env, alias: (alias == "bowl", {"source": "test"}),
    )
    value = TargetAwareNavigationPipeline(config=config)

    guard = value._held_object_guard(env)

    assert guard is not None
    assert guard.enabled is True
    assert guard.latched is True
    assert guard.latched_step == 0
    assert guard.action_override_enabled is True
    opening = np.zeros(12, dtype=np.float32)
    opening[6] = -1.0
    assert guard.apply_action(opening, step_index=1)[6] == 1.0
