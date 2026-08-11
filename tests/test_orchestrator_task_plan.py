from __future__ import annotations

import sys
from pathlib import Path


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from orchestrator import RoboCasaOrchestrator  # noqa: E402


def call(subgoal_id, task):
    return {
        "subgoal_id": subgoal_id,
        "atomic_task": task,
        "policy_prompt": f"Do {task}.",
        "arguments": {},
        "termination_condition": {"predicate": "open", "subject": "microwave"},
    }


class Adapter:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.env_ids = []
        self.atomic_tasks = []

    def execute(self, *, env, scheduler_query, episode_id):
        self.env_ids.append(id(env))
        success = next(self.statuses)
        atomic_call = scheduler_query["atomic_task_call"]
        self.atomic_tasks.append(atomic_call["atomic_task"])
        return {
            "subgoal_id": atomic_call["subgoal_id"],
            "status": "success" if success else "uncertain",
            "success": success,
            "verifier_result": {"retryable": not success},
        }


def test_plan_reuses_env_and_executes_in_order():
    adapter = Adapter([True, True])
    orchestrator = RoboCasaOrchestrator(atomic_task_policy_adapter=adapter)
    env = object()
    result = orchestrator.run_task_plan(
        env=env,
        task_plan=[call("g1", "OpenMicrowave"), call("g2", "TurnOnMicrowave")],
        episode_id=0,
    )
    assert result["success"] is True
    assert result["num_executed_steps"] == 2
    assert adapter.env_ids == [id(env), id(env)]


def test_plan_stops_after_uncertain_step():
    adapter = Adapter([False, True])
    orchestrator = RoboCasaOrchestrator(atomic_task_policy_adapter=adapter)
    result = orchestrator.run_task_plan(
        env=object(),
        task_plan=[call("g1", "OpenMicrowave"), call("g2", "TurnOnMicrowave")],
        episode_id=0,
    )
    assert result["success"] is False
    assert result["num_executed_steps"] == 1


def test_plan_retries_retryable_step_without_resetting_env():
    adapter = Adapter([False, True, True])
    orchestrator = RoboCasaOrchestrator(atomic_task_policy_adapter=adapter)
    env = object()
    result = orchestrator.run_task_plan(
        env=env,
        task_plan=[call("g1", "CloseMicrowave"), call("g2", "TurnOnMicrowave")],
        episode_id=0,
        max_task_retries=1,
    )
    assert result["success"] is True
    assert result["num_executed_steps"] == 2
    assert result["num_atomic_attempts"] == 3
    assert result["step_results"][0]["num_attempts"] == 2
    assert len(result["step_results"][0]["attempt_results"]) == 2
    assert adapter.env_ids == [id(env), id(env), id(env)]


def test_plan_does_not_retry_non_retryable_failure():
    class NonRetryableAdapter(Adapter):
        def execute(self, **kwargs):
            result = super().execute(**kwargs)
            result["verifier_result"]["retryable"] = False
            return result

    adapter = NonRetryableAdapter([False, True])
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter
    ).run_task_plan(
        env=object(),
        task_plan=[call("g1", "CloseMicrowave"), call("g2", "TurnOnMicrowave")],
        episode_id=0,
        max_task_retries=1,
    )
    assert result["num_executed_steps"] == 1
    assert result["num_atomic_attempts"] == 1


class Grounder:
    def __init__(self, results):
        self.results = iter(results)
        self.num_ground_calls = 0

    def ground(self, *, env, atomic_task_call):
        self.num_ground_calls += 1
        return next(self.results)

    def build_navigation_call(self, *, operation_call, grounding_result):
        return {
            "subgoal_id": f"navigate_before_{operation_call.subgoal_id}",
            "atomic_task": "NavigateKitchen",
            "policy_prompt": "Navigate to the sink.",
            "arguments": {
                "fixture_id": grounding_result["target_fixture_alias"],
                "fixture_name": "sink",
            },
            "termination_condition": {
                "predicate": "navigation_pose",
                "subject": grounding_result["target_fixture_alias"],
                "desired_value": True,
            },
        }


def grounding(grounded, *, fixture="sink_main_group", status=None):
    return {
        "grounded": grounded,
        "status": status or ("grounded" if grounded else "navigation_required"),
        "target_fixture_alias": fixture,
        "target_entity_alias": "blender_jug",
        "target_entity_kind": "object",
        "target_mode": "operation_object_current_location",
        "evidence": [],
    }


def test_grounded_operation_does_not_insert_navigation():
    adapter = Adapter([True])
    grounder = Grounder([grounding(True)])
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter,
        grounder=grounder,
    ).run_task_plan(
        env=object(),
        task_plan=[call("g1", "PickPlaceCounterToSink")],
        episode_id=0,
    )

    assert result["success"] is True
    assert result["num_inserted_navigation_steps"] == 0
    assert adapter.atomic_tasks == ["PickPlaceCounterToSink"]
    assert grounder.num_ground_calls == 1


def test_scheduler_inserts_navigation_then_regrounds_before_operation():
    adapter = Adapter([True, True])
    grounder = Grounder([grounding(False), grounding(True)])
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter,
        grounder=grounder,
    ).run_task_plan(
        env=object(),
        task_plan=[call("g1", "PickPlaceCounterToSink")],
        episode_id=0,
    )

    assert result["success"] is True
    assert result["num_inserted_navigation_steps"] == 1
    assert result["num_atomic_attempts"] == 2
    assert adapter.atomic_tasks == ["NavigateKitchen", "PickPlaceCounterToSink"]
    assert [event["phase"] for event in result["grounding_events"]] == [
        "before_operation",
        "after_inserted_navigation",
    ]


def test_operation_is_blocked_when_regrounding_after_navigation_fails():
    adapter = Adapter([True])
    grounder = Grounder([grounding(False), grounding(False)])
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter,
        grounder=grounder,
    ).run_task_plan(
        env=object(),
        task_plan=[call("g1", "PickPlaceCounterToSink")],
        episode_id=0,
    )

    assert result["success"] is False
    assert result["status"] == "failed"
    assert result["step_results"][0]["failure_code"] == (
        "GROUNDING_FAILED_AFTER_NAVIGATION"
    )
    assert adapter.atomic_tasks == ["NavigateKitchen"]
