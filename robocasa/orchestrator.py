"""Minimal planner/scheduler-facing RoboCasa orchestration interface."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

try:
    from .atomic_task_schemas import AtomicTaskCall
    from .grounding import grounding_result_to_dict
except ImportError:
    from atomic_task_schemas import AtomicTaskCall
    from grounding import grounding_result_to_dict


class RoboCasaOrchestrator:
    """Dispatch scheduler atomic-task queries without owning environment state."""

    def __init__(self, *, atomic_task_policy_adapter: Any, grounder: Any | None = None):
        if not hasattr(atomic_task_policy_adapter, "execute"):
            raise TypeError("atomic_task_policy_adapter must provide execute(...) ")
        if grounder is not None and (
            not hasattr(grounder, "ground")
            or not hasattr(grounder, "build_navigation_call")
        ):
            raise TypeError(
                "grounder must provide ground(...) and build_navigation_call(...)"
            )
        self.atomic_task_policy_adapter = atomic_task_policy_adapter
        self.grounder = grounder

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
        """Run operation calls, grounding each target and navigating only on demand.

        The orchestrator never resets or replaces env. After inserted navigation it
        grounds the operation again before dispatching the manipulation policy.
        """

        if not isinstance(max_task_retries, int) or max_task_retries < 0:
            raise ValueError("max_task_retries must be a non-negative integer")
        if not isinstance(task_plan, Sequence) or isinstance(task_plan, (str, bytes)):
            raise TypeError("task_plan must be a sequence of AtomicTaskCall mappings")
        if not task_plan:
            raise ValueError("task_plan must not be empty")
        step_results: list[dict[str, Any]] = []
        grounding_events: list[dict[str, Any]] = []
        inserted_navigation_results: list[dict[str, Any]] = []
        num_atomic_attempts = 0

        def execute_with_retries(
            call: AtomicTaskCall,
            *,
            base_episode_id: int | str,
            retry_label: str,
        ) -> tuple[dict[str, Any], int]:
            attempt_results: list[dict[str, Any]] = []
            for attempt_index in range(max_task_retries + 1):
                attempt_episode_id: int | str = base_episode_id
                if attempt_index:
                    attempt_episode_id = (
                        f"{base_episode_id}_{retry_label}_retry{attempt_index}"
                    )
                attempt = self.run_atomic_task_call(
                    env=env,
                    scheduler_query={"atomic_task_call": call.to_dict()},
                    episode_id=attempt_episode_id,
                )
                attempt["attempt_index"] = attempt_index
                attempt_results.append(attempt)
                verifier_result = attempt.get("verifier_result")
                retryable = bool(
                    isinstance(verifier_result, Mapping)
                    and verifier_result.get("retryable", False)
                )
                if attempt.get("success", False) or not retryable:
                    break
            result = dict(attempt_results[-1])
            result["num_attempts"] = len(attempt_results)
            if len(attempt_results) > 1:
                result["attempt_results"] = attempt_results
            return result, len(attempt_results)

        def grounding_failure(
            call: AtomicTaskCall,
            *,
            code: str,
            grounding: Mapping[str, Any],
            navigation_result: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            result: dict[str, Any] = {
                "subgoal_id": call.subgoal_id,
                "atomic_task": call.atomic_task,
                "status": "failed",
                "success": False,
                "failure_code": code,
                "grounding_result": dict(grounding),
                "verifier_result": {"retryable": False, "failure_code": code},
                "num_attempts": 0,
            }
            if navigation_result is not None:
                result["inserted_navigation_result"] = dict(navigation_result)
            return result

        for index, item in enumerate(task_plan):
            call = item if isinstance(item, AtomicTaskCall) else AtomicTaskCall.from_mapping(item)
            navigation_result: dict[str, Any] | None = None
            grounding_after_navigation: dict[str, Any] | None = None

            if self.grounder is not None and call.atomic_task != "NavigateKitchen":
                grounding_before = grounding_result_to_dict(
                    self.grounder.ground(env=env, atomic_task_call=call)
                )
                grounding_before.update(
                    {
                        "phase": "before_operation",
                        "plan_step_index": index,
                        "subgoal_id": call.subgoal_id,
                    }
                )
                grounding_events.append(grounding_before)
                print(
                    f"[Grounding] {call.subgoal_id}: "
                    f"status={grounding_before.get('status')} "
                    f"fixture={grounding_before.get('target_fixture_alias')}"
                )
                if not grounding_before.get("grounded", False):
                    target_fixture = grounding_before.get("target_fixture_alias")
                    if not target_fixture:
                        result = grounding_failure(
                            call,
                            code="GROUNDING_TARGET_UNRESOLVED",
                            grounding=grounding_before,
                        )
                        result["plan_step_index"] = index
                        step_results.append(result)
                        if stop_on_unsuccessful:
                            break
                        continue

                    navigation_call = self.grounder.build_navigation_call(
                        operation_call=call,
                        grounding_result=grounding_before,
                    )
                    if not isinstance(navigation_call, AtomicTaskCall):
                        navigation_call = AtomicTaskCall.from_mapping(navigation_call)
                    print(
                        f"[Scheduler] Insert {navigation_call.atomic_task} before "
                        f"{call.atomic_task}: {target_fixture}"
                    )
                    navigation_result, attempts = execute_with_retries(
                        navigation_call,
                        base_episode_id=f"{episode_id}_step{index + 1}_navigate",
                        retry_label="navigation",
                    )
                    num_atomic_attempts += attempts
                    navigation_result["inserted_before_plan_step_index"] = index
                    navigation_result["inserted_for_subgoal_id"] = call.subgoal_id
                    inserted_navigation_results.append(navigation_result)
                    if not navigation_result.get("success", False):
                        result = grounding_failure(
                            call,
                            code="DYNAMIC_NAVIGATION_FAILED",
                            grounding=grounding_before,
                            navigation_result=navigation_result,
                        )
                        result["plan_step_index"] = index
                        step_results.append(result)
                        if stop_on_unsuccessful:
                            break
                        continue

                    grounding_after_navigation = grounding_result_to_dict(
                        self.grounder.ground(env=env, atomic_task_call=call)
                    )
                    grounding_after_navigation.update(
                        {
                            "phase": "after_inserted_navigation",
                            "plan_step_index": index,
                            "subgoal_id": call.subgoal_id,
                        }
                    )
                    grounding_events.append(grounding_after_navigation)
                    print(
                        f"[Grounding] {call.subgoal_id} after navigation: "
                        f"status={grounding_after_navigation.get('status')}"
                    )
                    if not grounding_after_navigation.get("grounded", False):
                        result = grounding_failure(
                            call,
                            code="GROUNDING_FAILED_AFTER_NAVIGATION",
                            grounding=grounding_after_navigation,
                            navigation_result=navigation_result,
                        )
                        result["grounding_before_navigation"] = grounding_before
                        result["plan_step_index"] = index
                        step_results.append(result)
                        if stop_on_unsuccessful:
                            break
                        continue
            else:
                grounding_before = None

            result, attempts = execute_with_retries(
                call,
                base_episode_id=f"{episode_id}_step{index + 1}",
                retry_label="operation",
            )
            num_atomic_attempts += attempts
            if grounding_before is not None:
                result["grounding_before_operation"] = grounding_before
            if navigation_result is not None:
                result["inserted_navigation_result"] = navigation_result
            if grounding_after_navigation is not None:
                result["grounding_after_navigation"] = grounding_after_navigation
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
            "num_inserted_navigation_steps": len(inserted_navigation_results),
            "completed_all_steps": all_completed,
            "grounding_events": grounding_events,
            "inserted_navigation_results": inserted_navigation_results,
            "step_results": step_results,
        }
