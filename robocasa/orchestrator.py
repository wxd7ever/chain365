"""Minimal planner/scheduler-facing RoboCasa orchestration interface."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

try:
    from .atomic_task_schemas import AtomicTaskCall
except ImportError:
    from atomic_task_schemas import AtomicTaskCall


class RoboCasaOrchestrator:
    """Dispatch scheduler atomic-task queries without owning environment state."""

    def __init__(self, *, atomic_task_policy_adapter: Any):
        if not hasattr(atomic_task_policy_adapter, "execute"):
            raise TypeError("atomic_task_policy_adapter must provide execute(...) ")
        self.atomic_task_policy_adapter = atomic_task_policy_adapter

    def run_atomic_task_call(
        self,
        *,
        env: Any,
        scheduler_query: Mapping[str, Any],
        episode_id: int | str,
    ) -> dict[str, Any]:
        """Execute one scheduler-selected task in the unchanged environment."""

        return self.atomic_task_policy_adapter.execute(
            env=env,
            scheduler_query=scheduler_query,
            episode_id=episode_id,
        )

    def run_task_plan(
        self,
        *,
        env: Any,
        task_plan: Sequence[AtomicTaskCall | Mapping[str, Any]],
        episode_id: int | str,
        stop_on_unsuccessful: bool = True,
        max_task_retries: int = 0,
    ) -> dict[str, Any]:
        """Run ordered atomic calls in the same environment and policy session.

        The orchestrator does not reset or replace ``env``. A failed or uncertain
        step stops the plan by default so later actions do not run on an unknown state.
        """

        if not isinstance(max_task_retries, int) or max_task_retries < 0:
            raise ValueError("max_task_retries must be a non-negative integer")
        if not isinstance(task_plan, Sequence) or isinstance(task_plan, (str, bytes)):
            raise TypeError("task_plan must be a sequence of AtomicTaskCall mappings")
        if not task_plan:
            raise ValueError("task_plan must not be empty")
        step_results: list[dict[str, Any]] = []
        num_atomic_attempts = 0
        for index, item in enumerate(task_plan):
            call = item if isinstance(item, AtomicTaskCall) else AtomicTaskCall.from_mapping(item)
            attempt_results: list[dict[str, Any]] = []
            for attempt_index in range(max_task_retries + 1):
                attempt_episode_id: int | str = episode_id
                if attempt_index:
                    attempt_episode_id = (
                        f"{episode_id}_step{index + 1}_retry{attempt_index}"
                    )
                result = self.run_atomic_task_call(
                    env=env,
                    scheduler_query={"atomic_task_call": call.to_dict()},
                    episode_id=attempt_episode_id,
                )
                result["attempt_index"] = attempt_index
                attempt_results.append(result)
                num_atomic_attempts += 1
                verifier_result = result.get("verifier_result")
                retryable = bool(
                    isinstance(verifier_result, Mapping)
                    and verifier_result.get("retryable", False)
                )
                if result.get("success", False) or not retryable:
                    break
            result = dict(attempt_results[-1])
            result["num_attempts"] = len(attempt_results)
            if len(attempt_results) > 1:
                result["attempt_results"] = attempt_results
            result["plan_step_index"] = index
            step_results.append(result)
            if stop_on_unsuccessful and not result.get("success", False):
                break

        all_completed = len(step_results) == len(task_plan)
        success = all_completed and all(
            bool(result.get("success", False)) for result in step_results
        )
        if success:
            status = "success"
        elif step_results and step_results[-1].get("status") == "failed":
            status = "failed"
        else:
            status = "uncertain"
        return {
            "module": "LONG_HORIZON_TASK",
            "status": status,
            "success": success,
            "num_planned_steps": len(task_plan),
            "num_executed_steps": len(step_results),
            "num_atomic_attempts": num_atomic_attempts,
            "completed_all_steps": all_completed,
            "step_results": step_results,
        }
