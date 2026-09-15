from __future__ import annotations

import math

from robocasa.navigation.base_navigator import BaseNavigator
from robocasa.navigation.navigation_types import (
    NavigationExecutionResult,
    NavigationResult,
)
from robocasa.navigation.skill_adapter import TargetAwarePolicyAdapter
from robocasa.orchestrator import RoboCasaOrchestrator


CONFIG = {
    "waypoint_position_tolerance": 0.12,
    "final_position_tolerance": 0.08,
    "final_yaw_tolerance_deg": 10.0,
    "max_waypoint_retries": 1,
    "waypoint_max_steps": 10,
    "final_position_max_steps": 10,
    "final_yaw_max_steps": 10,
    "max_translation_command": 0.5,
    "max_rotation_command": 0.5,
}


class PoseEnv:
    def __init__(self):
        self.pose = {"position": [0.0, 0.0, 0.7], "yaw_rad": 0.0}


def test_base_navigator_executes_xy_then_final_yaw(monkeypatch):
    env = PoseEnv()
    calls = []

    def current(value):
        return {"position": list(value.pose["position"]), "yaw_rad": value.pose["yaw_rad"]}

    def move(value, target, **kwargs):
        calls.append({"target": target, **kwargs})
        value.pose = {
            "position": list(target["position"]),
            "yaw_rad": float(target["yaw_rad"]),
        }
        return {"success": True, "steps": 3, "guard_failure": None}

    monkeypatch.setattr("robocasa.navigation.base_navigator.current_base_pose", current)
    monkeypatch.setattr("robocasa.navigation.base_navigator.move_base_to_pose", move)
    result = BaseNavigator(CONFIG).execute_path(
        env=env,
        path_world=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        goal_yaw=math.pi / 2,
    )
    assert result.success
    assert result.reached_waypoints == 2
    assert result.total_waypoints == 2
    assert result.position_error == 0.0
    assert result.yaw_error == 0.0
    assert [item["yaw_tolerance_deg"] for item in calls] == [180.0, 180.0, 10.0]
    assert calls[0]["target"]["yaw_rad"] == 0.0
    assert calls[1]["target"]["yaw_rad"] == 0.0
    assert calls[2]["target"]["yaw_rad"] == math.pi / 2


def test_base_navigator_retries_one_stalled_waypoint(monkeypatch):
    env = PoseEnv()
    call_count = 0

    monkeypatch.setattr(
        "robocasa.navigation.base_navigator.current_base_pose",
        lambda value: {"position": list(value.pose["position"]), "yaw_rad": value.pose["yaw_rad"]},
    )

    def move(value, target, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            value.pose = {
                "position": list(target["position"]),
                "yaw_rad": float(target["yaw_rad"]),
            }
        return {"success": call_count > 1, "steps": 2, "guard_failure": None}

    monkeypatch.setattr("robocasa.navigation.base_navigator.move_base_to_pose", move)
    result = BaseNavigator(CONFIG).execute_path(
        env=env, path_world=[(0.0, 0.0), (1.0, 0.0)], goal_yaw=0.0
    )
    assert result.success
    assert call_count == 3
    assert [item["attempt"] for item in result.waypoint_traces[:2]] == [1, 2]


def test_base_navigator_stops_when_post_waypoint_guard_reports_drop(monkeypatch):
    env = PoseEnv()

    monkeypatch.setattr(
        "robocasa.navigation.base_navigator.current_base_pose",
        lambda value: {"position": list(value.pose["position"]), "yaw_rad": value.pose["yaw_rad"]},
    )

    def move(value, target, **kwargs):
        value.pose = {
            "position": list(target["position"]),
            "yaw_rad": float(target["yaw_rad"]),
        }
        return {"success": True, "steps": 2, "guard_failure": None}

    class Guard:
        def observe(self, *, step_index):
            return {"failure_code": "OBJECT_DROPPED", "step_index": step_index}

        def to_dict(self):
            return {"dropped_step": 2}

    monkeypatch.setattr("robocasa.navigation.base_navigator.move_base_to_pose", move)
    result = BaseNavigator(CONFIG).execute_path(
        env=env,
        path_world=[(0.0, 0.0), (1.0, 0.0)],
        goal_yaw=0.0,
        held_object_context=Guard(),
    )
    assert not result.success
    assert result.reason == "OBJECT_DROPPED"
    assert result.reached_waypoints == 0


class FakePolicy:
    def __init__(self):
        self.calls = 0
        self.log_dir = "/tmp"

    def execute(self, **kwargs):
        self.calls += 1
        return {
            "success": True,
            "status": "success",
            "verifier_result": {"goal_satisfied": True, "retryable": False},
        }


class FakePipeline:
    def __init__(self, result):
        self.result = result
        self.restore_calls = 0

    def navigate_to_manipulation_target(self, *args, **kwargs):
        return self.result

    def restore_coarse_pose(self, env, coarse_pose, context=None):
        self.restore_calls += 1
        env.pose = {
            "position": list(coarse_pose["position"]),
            "yaw_rad": float(coarse_pose["yaw_rad"]),
        }
        return NavigationExecutionResult(
            True, "SUCCESS", 1, 1, final_pose=_pose(env),
            position_error=0.0, yaw_error=0.0,
        )


class FakeRefiner:
    def __init__(self, success):
        self.success = success
        self.atomic_task_call = None

    def refine(self, *, env, **kwargs):
        self.atomic_task_call = kwargs["atomic_task_call"]
        env.pose["position"][0] += 0.1
        env.pose["yaw_rad"] += math.radians(5.0)
        return {
            "success": self.success,
            "status": "success" if self.success else "failed",
            "failure_code": None if self.success else "LOCAL_POSE_UNRESOLVED",
        }


def call():
    return {
        "subgoal_id": "pick_bowl",
        "atomic_task": "PickObject",
        "arguments": {"object_id": "bowl"},
        "termination_condition": {
            "predicate": "holding",
            "subject": "bowl",
            "desired_value": True,
        },
        "policy_prompt": "Pick up the bowl and keep holding it.",
    }


def _pose(value):
    return {"position": list(value.pose["position"]), "yaw_rad": value.pose["yaw_rad"]}


def test_adapter_navigation_failure_prevents_policy(monkeypatch):
    monkeypatch.setattr("robocasa.navigation.skill_adapter.current_base_pose", _pose)
    policy = FakePolicy()
    navigation = NavigationResult(False, "PATH_NOT_FOUND")
    adapter = TargetAwarePolicyAdapter(
        policy_adapter=policy,
        navigation_pipeline=FakePipeline(navigation),
        navigation_enabled=True,
        local_refiner_enabled=False,
    )
    result = adapter.execute(
        env=PoseEnv(), scheduler_query={"atomic_task_call": call()}, episode_id=0
    )
    assert not result["success"]
    assert result["stage_status"] == "NAVIGATION_FAILED"
    assert result["failure_code"] == "PATH_NOT_FOUND"
    assert result["target_aware_stage"]["policy_started"] is False
    assert result["target_aware_stage"]["local_refinement_used"] is False
    assert policy.calls == 0


def test_adapter_continues_policy_after_configured_navigation_collision(monkeypatch):
    monkeypatch.setattr("robocasa.navigation.skill_adapter.current_base_pose", _pose)
    policy = FakePolicy()

    class CollisionPipeline(FakePipeline):
        def __init__(self):
            super().__init__(NavigationResult(False, "START_IN_COLLISION"))
            self.config = {
                "navigation": {
                    "continue_policy_after_failure_reasons": [
                        "START_IN_COLLISION"
                    ]
                }
            }
            self.navigation_context = None

        def navigate_to_manipulation_target(self, *args, **kwargs):
            self.navigation_context = kwargs.get("context")
            return self.result

    pipeline = CollisionPipeline()
    adapter = TargetAwarePolicyAdapter(
        policy_adapter=policy,
        navigation_pipeline=pipeline,
        navigation_enabled=True,
        local_refiner_enabled=False,
    )

    result = adapter.execute(
        env=PoseEnv(),
        scheduler_query={
            "atomic_task_call": call(),
            "grounding_result": {
                "grounded": False,
                "status": "navigation_required",
            },
        },
        episode_id=0,
    )

    assert result["success"] is True
    assert result["navigation_result"]["success"] is False
    assert result["target_aware_stage"]["navigation_bypassed"] is True
    assert result["target_aware_stage"]["navigation_warning"] == "START_IN_COLLISION"
    assert result["target_aware_stage"]["policy_started"] is True
    assert pipeline.navigation_context["grounding_result"]["grounded"] is False
    assert policy.calls == 1


def test_adapter_records_refiner_delta_and_runs_existing_policy(monkeypatch):
    monkeypatch.setattr("robocasa.navigation.skill_adapter.current_base_pose", _pose)
    policy = FakePolicy()
    navigation = NavigationResult(
        True,
        "ALREADY_AT_WORK_REGION",
        target_name="bowl",
        target_type="EXPOSED_OBJECT",
        skipped=True,
    )
    adapter = TargetAwarePolicyAdapter(
        policy_adapter=policy,
        navigation_pipeline=FakePipeline(navigation),
        refiner=FakeRefiner(True),
        navigation_enabled=True,
        local_refiner_enabled=True,
    )
    result = adapter.execute(
        env=PoseEnv(), scheduler_query={"atomic_task_call": call()}, episode_id=0
    )
    stage = result["target_aware_stage"]
    assert result["success"]
    assert stage["navigation_required"] is False
    assert stage["local_refinement_used"] is True
    assert isinstance(adapter.refiner.atomic_task_call, dict)
    assert adapter.refiner.atomic_task_call["subgoal_id"] == "pick_bowl"
    assert stage["local_refinement_delta"]["delta_translation_m"] == 0.1
    assert abs(stage["local_refinement_delta"]["delta_yaw_deg"] - 5.0) < 1e-9
    assert stage["policy_started"] is True
    assert stage["verifier_success"] is True
    assert policy.calls == 1


def test_use_coarse_pose_restores_before_policy(monkeypatch):
    monkeypatch.setattr("robocasa.navigation.skill_adapter.current_base_pose", _pose)
    policy = FakePolicy()
    pipeline = FakePipeline(NavigationResult(True, "SUCCESS", skipped=False))
    adapter = TargetAwarePolicyAdapter(
        policy_adapter=policy,
        navigation_pipeline=pipeline,
        refiner=FakeRefiner(False),
        navigation_enabled=True,
        local_refiner_enabled=True,
        local_refiner_failure_policy="use_coarse_pose",
    )
    result = adapter.execute(
        env=PoseEnv(), scheduler_query={"atomic_task_call": call()}, episode_id=0
    )
    stage = result["target_aware_stage"]
    assert result["success"]
    assert pipeline.restore_calls == 1
    assert stage["local_refinement_attempt_delta"]["delta_translation_m"] == 0.1
    assert stage["local_refinement_delta"]["delta_translation_m"] == 0.0
    assert stage["coarse_pose_restore"]["success"] is True
    assert policy.calls == 1


def test_adapter_abort_policy_stops_after_refiner_failure(monkeypatch):
    monkeypatch.setattr("robocasa.navigation.skill_adapter.current_base_pose", _pose)
    policy = FakePolicy()
    navigation = NavigationResult(True, "SUCCESS", skipped=False)
    adapter = TargetAwarePolicyAdapter(
        policy_adapter=policy,
        navigation_pipeline=FakePipeline(navigation),
        refiner=FakeRefiner(False),
        navigation_enabled=True,
        local_refiner_enabled=True,
        local_refiner_failure_policy="abort",
    )
    result = adapter.execute(
        env=PoseEnv(), scheduler_query={"atomic_task_call": call()}, episode_id=0
    )
    assert not result["success"]
    assert result["stage_status"] == "LOCAL_REFINEMENT_FAILED"
    assert policy.calls == 0


def test_orchestrator_defers_ungrounded_target_to_target_aware_adapter(monkeypatch):
    monkeypatch.setattr("robocasa.navigation.skill_adapter.current_base_pose", _pose)

    class Grounder:
        def ground(self, **kwargs):
            return {
                "grounded": False,
                "status": "navigation_required",
                "target_fixture_alias": "counter",
            }

        def build_navigation_call(self, **kwargs):
            raise AssertionError("legacy NavigateKitchen must not be inserted")

    policy = FakePolicy()
    adapter = TargetAwarePolicyAdapter(
        policy_adapter=policy,
        navigation_pipeline=FakePipeline(
            NavigationResult(
                True, "ALREADY_AT_WORK_REGION", target_name="bowl", skipped=True
            )
        ),
        navigation_enabled=True,
        local_refiner_enabled=False,
    )
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter, grounder=Grounder()
    ).run_task_plan(
        env=PoseEnv(), task_plan=[call()], episode_id=0
    )
    assert result["success"]
    assert result["num_inserted_navigation_steps"] == 0
    assert result["num_target_aware_stages"] == 1
    assert policy.calls == 1


def test_navigation_off_keeps_legacy_orchestrator_capability():
    adapter = TargetAwarePolicyAdapter(
        policy_adapter=FakePolicy(),
        navigation_pipeline=FakePipeline(NavigationResult(True, "SUCCESS")),
        navigation_enabled=False,
        local_refiner_enabled=False,
    )
    assert adapter.handles_target_aware_navigation is False


def test_adapter_prepares_verified_pick_before_following_navigation(monkeypatch):
    monkeypatch.setattr(
        "robocasa.navigation.skill_adapter.current_base_pose", _pose
    )
    monkeypatch.setattr(
        "robocasa.navigation.skill_adapter.resolve_object_alias",
        lambda env, call: "bowl",
    )
    events = []

    class Controller:
        def prepare(self, **kwargs):
            assert kwargs["pick_verified"] is True
            events.append("prepare")
            return {
                "success": True,
                "transport_ready": True,
                "object_alias": "bowl",
                "reason": "TRANSPORT_READY",
            }

    class Pipeline:
        def navigate_to_manipulation_target(
            self, env, atomic_call, *, before_navigation=None, **kwargs
        ):
            if atomic_call.atomic_task == "PlaceObject":
                assert before_navigation is not None
                preparation = before_navigation()
                assert preparation["transport_ready"]
                events.append("navigate")
                return NavigationResult(
                    True,
                    "SUCCESS",
                    target_name="microwave",
                    target_type="FIXTURE",
                    skipped=False,
                )
            assert before_navigation is None
            return NavigationResult(
                True,
                "ALREADY_AT_WORK_REGION",
                target_name="bowl",
                target_type="EXPOSED_OBJECT",
                skipped=True,
            )

    place = {
        "subgoal_id": "place_bowl",
        "atomic_task": "PlaceObject",
        "arguments": {
            "held_object_id": "bowl",
            "destination_id": "microwave",
        },
        "termination_condition": {
            "predicate": "inside",
            "subject": "bowl",
            "object": "microwave",
            "desired_value": True,
        },
        "policy_prompt": "Place the bowl in the microwave.",
    }
    policy = FakePolicy()
    adapter = TargetAwarePolicyAdapter(
        policy_adapter=policy,
        navigation_pipeline=Pipeline(),
        post_grasp_transport_controller=Controller(),
        navigation_enabled=True,
        local_refiner_enabled=False,
    )
    pick_result = adapter.execute(
        env=PoseEnv(),
        scheduler_query={"atomic_task_call": call()},
        episode_id="pick",
    )
    assert pick_result["success"]
    assert (
        pick_result["target_aware_stage"][
            "post_grasp_transport_pending"
        ]
        is True
    )
    place_result = adapter.execute(
        env=PoseEnv(),
        scheduler_query={"atomic_task_call": place},
        episode_id="place",
    )
    assert place_result["success"]
    assert events == ["prepare", "navigate"]
    stage = place_result["target_aware_stage"]
    assert stage["post_grasp_transport_required"] is True
    assert stage["transport_ready"] is True


def test_target_aware_adapter_retracts_after_verified_place(monkeypatch):
    monkeypatch.setattr(
        "robocasa.navigation.skill_adapter.current_base_pose", _pose
    )
    events = []

    class Policy(FakePolicy):
        def execute(self, **kwargs):
            events.append("place_policy")
            return super().execute(**kwargs)

    class Pipeline:
        def navigate_to_manipulation_target(self, *args, **kwargs):
            return NavigationResult(
                True,
                "ALREADY_AT_WORK_REGION",
                target_name="bowl",
                target_type="EXPOSED_OBJECT",
                skipped=True,
            )

    class Retreat:
        def prepare(self, **kwargs):
            assert kwargs["place_verified"] is True
            events.append("post_place_retreat")
            return {
                "success": True,
                "retreat_ready": True,
                "object_alias": "vegetable",
                "reason": "POST_PLACE_RETREAT_READY",
            }

    place = {
        "subgoal_id": "place_vegetable",
        "atomic_task": "PlaceObject",
        "arguments": {
            "object_id": "vegetable",
            "destination_id": "bowl",
        },
        "termination_condition": {
            "predicate": "inside",
            "subject": "vegetable",
            "object": "bowl",
            "desired_value": True,
        },
        "policy_prompt": "Place the vegetable in the bowl.",
    }
    adapter = TargetAwarePolicyAdapter(
        policy_adapter=Policy(),
        navigation_pipeline=Pipeline(),
        post_place_retreat_controller=Retreat(),
        navigation_enabled=True,
        local_refiner_enabled=False,
    )
    result = adapter.execute(
        env=PoseEnv(),
        scheduler_query={"atomic_task_call": place},
        episode_id="place",
    )
    assert result["success"]
    assert events == ["place_policy", "post_place_retreat"]
    stage = result["target_aware_stage"]
    assert stage["post_place_retreat_required"] is True
    assert stage["post_place_retreat_success"] is True
    assert stage["retreat_ready"] is True


def test_unified_post_skill_return_supersedes_legacy_place_retreat(monkeypatch):
    monkeypatch.setattr(
        "robocasa.navigation.skill_adapter.current_base_pose", _pose
    )
    events = []

    class Pipeline:
        config = {}

        def navigate_to_manipulation_target(self, *args, **kwargs):
            return NavigationResult(True, "ALREADY_AT_WORK_REGION", skipped=True)

    class ReturnController:
        def prepare(self, **kwargs):
            events.append("post_skill_return")
            return {
                "success": True,
                "return_ready": True,
                "navigation_safe": True,
                "preset": "released",
            }

    class LegacyRetreat:
        def prepare(self, **kwargs):
            events.append("legacy_retreat")
            return {"success": True, "retreat_ready": True}

    place = {
        "subgoal_id": "place",
        "atomic_task": "PlaceObject",
        "arguments": {"object_id": "vegetable", "destination_id": "bowl"},
        "termination_condition": {
            "predicate": "inside",
            "subject": "vegetable",
            "object": "bowl",
        },
        "policy_prompt": "Place the vegetable in the bowl.",
    }
    adapter = TargetAwarePolicyAdapter(
        policy_adapter=FakePolicy(),
        navigation_pipeline=Pipeline(),
        post_place_retreat_controller=LegacyRetreat(),
        post_skill_return_controller=ReturnController(),
        navigation_enabled=True,
        local_refiner_enabled=False,
    )
    result = adapter.execute(
        env=PoseEnv(),
        scheduler_query={"atomic_task_call": place},
        episode_id=0,
    )

    assert result["success"] is True
    assert events == ["post_skill_return"]
    assert result["target_aware_stage"]["post_skill_return_success"] is True



def test_failed_post_skill_return_is_diagnostic_when_not_enforced(monkeypatch):
    monkeypatch.setattr(
        "robocasa.navigation.skill_adapter.current_base_pose", _pose
    )

    class Pipeline:
        config = {
            "post_skill_return": {
                "enabled": True,
                "enforce_result": False,
                "failure_policy": "block",
            }
        }

        def navigate_to_manipulation_target(self, *args, **kwargs):
            return NavigationResult(True, "ALREADY_AT_WORK_REGION", skipped=True)

    class FailedReturnController:
        def prepare(self, **kwargs):
            return {
                "success": False,
                "return_ready": False,
                "failure_code": "POST_SKILL_RETURN_FAILED",
            }

    place = {
        "subgoal_id": "place",
        "atomic_task": "PlaceObject",
        "arguments": {"object_id": "vegetable", "destination_id": "bowl"},
        "termination_condition": {
            "predicate": "inside",
            "subject": "vegetable",
            "object": "bowl",
        },
        "policy_prompt": "Place the vegetable in the bowl.",
    }
    adapter = TargetAwarePolicyAdapter(
        policy_adapter=FakePolicy(),
        navigation_pipeline=Pipeline(),
        post_skill_return_controller=FailedReturnController(),
        navigation_enabled=True,
        local_refiner_enabled=False,
    )

    result = adapter.execute(
        env=PoseEnv(),
        scheduler_query={"atomic_task_call": place},
        episode_id=0,
    )

    stage = result["target_aware_stage"]
    assert result["success"] is True
    assert stage["post_skill_return_enforce_result"] is False
    assert stage["post_skill_return_required"] is True
    assert stage["post_skill_return_success"] is False
    assert stage["post_skill_return_warning"] == "POST_SKILL_RETURN_FAILED"


def test_failed_post_place_retreat_preserves_place_and_retries_before_navigation(
    monkeypatch,
):
    monkeypatch.setattr(
        "robocasa.navigation.skill_adapter.current_base_pose", _pose
    )
    events = []

    class Pipeline:
        def navigate_to_manipulation_target(
            self, env, atomic_call, *, before_navigation=None, **kwargs
        ):
            if atomic_call.atomic_task == "PlaceObject":
                assert before_navigation is None
                return NavigationResult(
                    True,
                    "ALREADY_AT_WORK_REGION",
                    target_name="bowl",
                    skipped=True,
                )
            assert before_navigation is not None
            preparation = before_navigation()
            events.append("pre_navigation_retreat")
            assert preparation["navigation_safe"]
            return NavigationResult(
                True,
                "SUCCESS",
                target_name="bowl",
                target_type="EXPOSED_OBJECT",
                skipped=False,
            )

    class Retreat:
        def __init__(self):
            self.calls = 0

        def prepare(self, **kwargs):
            self.calls += 1
            ready = self.calls == 2
            events.append(
                "retreat_ready" if ready else "retreat_warning"
            )
            return {
                "module": "POST_PLACE_RETREAT",
                "success": ready,
                "retreat_ready": ready,
                "navigation_safe": ready,
                "object_alias": "vegetable",
                "reason": (
                    "POST_PLACE_RETREAT_READY"
                    if ready
                    else "FINAL_ARM_NOT_COMPACT"
                ),
                "failure_code": (
                    None if ready else "POST_PLACE_RETRACT_FAILED"
                ),
            }

    place = {
        "subgoal_id": "place_vegetable",
        "atomic_task": "PlaceObject",
        "arguments": {
            "object_id": "vegetable",
            "destination_id": "bowl",
        },
        "termination_condition": {
            "predicate": "inside",
            "subject": "vegetable",
            "object": "bowl",
            "desired_value": True,
        },
        "policy_prompt": "Place the vegetable in the bowl.",
    }
    retreat = Retreat()
    policy = FakePolicy()
    adapter = TargetAwarePolicyAdapter(
        policy_adapter=policy,
        navigation_pipeline=Pipeline(),
        post_place_retreat_controller=retreat,
        navigation_enabled=True,
        local_refiner_enabled=False,
    )

    place_result = adapter.execute(
        env=PoseEnv(),
        scheduler_query={"atomic_task_call": place},
        episode_id="place",
    )
    assert place_result["success"]
    place_stage = place_result["target_aware_stage"]
    assert place_stage["post_place_retreat_warning"] == (
        "POST_PLACE_RETRACT_FAILED"
    )
    assert adapter._pending_post_place_retreat is not None

    next_result = adapter.execute(
        env=PoseEnv(),
        scheduler_query={"atomic_task_call": call()},
        episode_id="next_pick",
    )
    assert next_result["success"]
    next_stage = next_result["target_aware_stage"]
    assert next_stage["pre_navigation_post_place_retreat_required"]
    assert next_stage["pre_navigation_post_place_retreat_success"]
    assert adapter._pending_post_place_retreat is None
    assert retreat.calls == 2
    assert events == [
        "retreat_warning",
        "retreat_ready",
        "pre_navigation_retreat",
    ]
