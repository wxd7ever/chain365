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

    def execute(self, *, env, scheduler_query, episode_id):
        self.env_ids.append(id(env))
        success = next(self.statuses)
        atomic_call = scheduler_query["atomic_task_call"]
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
