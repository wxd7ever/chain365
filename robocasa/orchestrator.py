"""Minimal planner/scheduler-facing RoboCasa orchestration interface."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

try:
    from .atomic_task_schemas import AtomicTaskCall
    from .grounding import grounding_result_to_dict
except ImportError:
    from atomic_task_schemas import AtomicTaskCall
    from grounding import grounding_result_to_dict


_RELATION_PREDICATES = {"inside", "inserted", "on", "object_fixture_relation"}
_RETRY_RECOVERY_MODES = {
    "same_skill",
    "same_skill_no_drop_retry",
    "dependency_recovery",
}


def _condition_items(call: AtomicTaskCall) -> list[Mapping[str, Any]]:
    value = call.termination_condition
    return value if isinstance(value, list) else [value]


def _required_holding_object(call: AtomicTaskCall) -> str | None:
    if call.atomic_task != "PlaceObject":
        return None
    relation = next(
        (
            item
            for item in _condition_items(call)
            if str(item.get("predicate", "")).lower() in _RELATION_PREDICATES
        ),
        None,
    )
    return str(
        (relation or {}).get("subject")
        or call.arguments.get("held_object_id")
        or call.arguments.get("object_id")
        or ""
    ).strip() or None


def _produced_holding_object(call: AtomicTaskCall) -> str | None:
    holding = next(
        (
            item
            for item in _condition_items(call)
            if str(item.get("predicate", "")).lower() == "holding"
            and item.get("desired_value", True) is True
        ),
        None,
    )
    if holding is None:
        return None
    return str(holding.get("subject") or "").strip() or None


def _result_failure_code(result: Mapping[str, Any]) -> str | None:
    """Return the reported atomic failure code without reading environment state."""

    direct = result.get("failure_code")
    if direct:
        return str(direct)
    verifier_result = result.get("verifier_result")
    if isinstance(verifier_result, Mapping) and verifier_result.get("failure_code"):
        return str(verifier_result["failure_code"])
    rollout_logs = result.get("rollout_logs")
    if isinstance(rollout_logs, Mapping):
        final_verification = rollout_logs.get("Final_Verification")
        if isinstance(final_verification, Mapping) and final_verification.get(
            "failure_code"
        ):
            return str(final_verification["failure_code"])
    return None


def _is_object_drop(result: Mapping[str, Any]) -> bool:
    return _result_failure_code(result) in {
        "OBJECT_DROPPED",
        "OBJECT_DROPPED_DURING_PREP",
    }


def _summarize_target_aware_stages(
    stages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    deltas = []
    for stage in stages:
        delta = stage.get("local_refinement_attempt_delta") or stage.get(
            "local_refinement_delta"
        )
        if isinstance(delta, Mapping):
            deltas.append(delta)
    translations = [
        float(item.get("delta_translation_m", 0.0)) for item in deltas
    ]
    yaw_errors = [abs(float(item.get("delta_yaw_deg", 0.0))) for item in deltas]
    return {
        "num_stages": len(stages),
        "navigation_required": sum(
            bool(stage.get("navigation_required")) for stage in stages
        ),
        "navigation_success": sum(
            stage.get("navigation_success") is True for stage in stages
        ),
        "post_grasp_transport_required": sum(
            bool(stage.get("post_grasp_transport_required"))
            for stage in stages
        ),
        "post_grasp_transport_success": sum(
            stage.get("post_grasp_transport_success") is True
            for stage in stages
        ),
        "post_grasp_transport_failed": sum(
            stage.get("post_grasp_transport_success") is False
            and bool(stage.get("post_grasp_transport_required"))
            for stage in stages
        ),
        "post_place_retreat_required": sum(
            bool(stage.get("post_place_retreat_required"))
            for stage in stages
        ),
        "post_place_retreat_success": sum(
            stage.get("post_place_retreat_success") is True
            for stage in stages
        ),
        "post_place_retreat_failed": sum(
            stage.get("post_place_retreat_success") is False
            and bool(stage.get("post_place_retreat_required"))
            for stage in stages
        ),
        "local_refinement_used": sum(
            bool(stage.get("local_refinement_used")) for stage in stages
        ),
        "local_refinement_success": sum(
            stage.get("local_refinement_success") is True for stage in stages
        ),
        "policy_started": sum(
            bool(stage.get("policy_started")) for stage in stages
        ),
        "policy_success": sum(
            stage.get("policy_success") is True for stage in stages
        ),
        "verifier_success": sum(
            stage.get("verifier_success") is True for stage in stages
        ),
        "mean_local_refinement_translation_m": (
            sum(translations) / len(translations) if translations else None
        ),
        "mean_abs_local_refinement_yaw_deg": (
            sum(yaw_errors) / len(yaw_errors) if yaw_errors else None
        ),
    }


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
        retry_recovery_mode: str = "same_skill",
    ) -> dict[str, Any]:
        """Run operation calls, grounding each target and navigating only on demand.

        The orchestrator never resets or replaces env. After inserted navigation it
        grounds the operation again before dispatching the manipulation policy.
        """

        if not isinstance(max_task_retries, int) or max_task_retries < 0:
            raise ValueError("max_task_retries must be a non-negative integer")
        if retry_recovery_mode not in _RETRY_RECOVERY_MODES:
            raise ValueError(
                "retry_recovery_mode must be one of "
                f"{sorted(_RETRY_RECOVERY_MODES)}, got {retry_recovery_mode!r}"
            )
        if not isinstance(task_plan, Sequence) or isinstance(task_plan, (str, bytes)):
            raise TypeError("task_plan must be a sequence of AtomicTaskCall mappings")
        if not task_plan:
            raise ValueError("task_plan must not be empty")
        calls = [
            item
            if isinstance(item, AtomicTaskCall)
            else AtomicTaskCall.from_mapping(item)
            for item in task_plan
        ]
        step_results: list[dict[str, Any]] = []
        grounding_events: list[dict[str, Any]] = []
        inserted_navigation_results: list[dict[str, Any]] = []
        recovery_events: list[dict[str, Any]] = []
        transition_events: list[dict[str, Any]] = []
        num_atomic_attempts = 0
        num_dependency_recoveries = 0
        # A successful PickObject critic establishes the handoff consumed by the
        # following PlaceObject. Visual grounding locates the destination only and
        # must not independently overturn this verified scheduler transition.
        verified_holding_objects: set[str] = set()

        def attach_verified_handoff(
            grounding: Mapping[str, Any], object_alias: str | None
        ) -> dict[str, Any]:
            result = dict(grounding)
            if object_alias is None or object_alias not in verified_holding_objects:
                return result
            result["held_object_alias"] = object_alias
            result["scheduler_handoff"] = {
                "holding_object_alias": object_alias,
                "source": "prior_successful_atomic_task",
            }
            if result.get("target_fixture_alias"):
                result["target_mode"] = "held_object_destination"
            return result

        def find_holding_producer(
            *, before_index: int, object_alias: str
        ) -> AtomicTaskCall | None:
            for candidate in reversed(calls[:before_index]):
                if _produced_holding_object(candidate) == object_alias:
                    return candidate
            return None

        def retry_previous_holding_skill(
            *,
            current_call: AtomicTaskCall,
            current_index: int,
            object_alias: str,
            recovery_phase: str = "precondition",
            trigger_failure_code: str = "PRECONDITION_NOT_SATISFIED",
            recovery_cycle_index: int = 0,
        ) -> tuple[bool, AtomicTaskCall | None, list[dict[str, Any]]]:
            nonlocal num_atomic_attempts
            rollback = find_holding_producer(
                before_index=current_index, object_alias=object_alias
            )
            attempts: list[dict[str, Any]] = []
            if rollback is None or max_task_retries == 0:
                return False, rollback, attempts
            for recovery_index in range(1, max_task_retries + 1):
                if recovery_phase == "precondition" and recovery_cycle_index == 0:
                    recovery_episode_id = (
                        f"{episode_id}_step{current_index + 1}_"
                        f"rollback_{rollback.subgoal_id}_retry{recovery_index}"
                    )
                else:
                    recovery_episode_id = (
                        f"{episode_id}_step{current_index + 1}_"
                        f"rollback_{rollback.subgoal_id}_{recovery_phase}_"
                        f"cycle{recovery_cycle_index}_retry{recovery_index}"
                    )
                recovery_result = self.run_atomic_task_call(
                    env=env,
                    scheduler_query={"atomic_task_call": rollback.to_dict()},
                    episode_id=recovery_episode_id,
                )
                num_atomic_attempts += 1
                event = {
                    "recovery_type": "RETRY_PREVIOUS_SKILL",
                    "from_task": current_call.atomic_task,
                    "from_subgoal_id": current_call.subgoal_id,
                    "rollback_task": rollback.atomic_task,
                    "rollback_subgoal_id": rollback.subgoal_id,
                    "failed_predicate": f"holding({object_alias})",
                    "trigger_failure_code": trigger_failure_code,
                    "recovery_phase": recovery_phase,
                    "recovery_cycle_index": recovery_cycle_index,
                    "recovery_attempt_index": recovery_index,
                    "result": recovery_result,
                }
                recovery_events.append(event)
                attempts.append(event)
                if recovery_result.get("success", False):
                    return True, rollback, attempts
            return False, rollback, attempts

        def execute_with_retries(
            call: AtomicTaskCall,
            *,
            base_episode_id: int | str,
            retry_label: str,
            holding_dependency: str | None = None,
            scheduler_context: Mapping[str, Any] | None = None,
        ) -> tuple[dict[str, Any], int]:
            attempt_results: list[dict[str, Any]] = []
            for attempt_index in range(max_task_retries + 1):
                attempt_episode_id: int | str = base_episode_id
                if attempt_index:
                    attempt_episode_id = (
                        f"{base_episode_id}_{retry_label}_retry{attempt_index}"
                    )
                scheduler_query: dict[str, Any] = {
                    "atomic_task_call": call.to_dict()
                }
                if scheduler_context is not None:
                    scheduler_query.update(dict(scheduler_context))
                attempt = self.run_atomic_task_call(
                    env=env,
                    scheduler_query=scheduler_query,
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
                if (
                    retry_recovery_mode == "same_skill_no_drop_retry"
                    and holding_dependency is not None
                    and _is_object_drop(attempt)
                ):
                    attempt["retry_suppressed"] = True
                    attempt["retry_suppressed_reason"] = (
                        "OBJECT_DROPPED_WITHOUT_ROLLBACK"
                    )
                    break
                if (
                    retry_recovery_mode == "dependency_recovery"
                    and holding_dependency is not None
                    and _is_object_drop(attempt)
                ):
                    # Re-run the producer PickObject instead of spending the
                    # remaining budget on Navigate/Place from a dropped state.
                    # same_skill deliberately retains the legacy behavior.
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

        def execute_plan_step(
            *,
            index: int,
            call: AtomicTaskCall,
            dependency_recovery_count: int = 0,
            accumulated_recovery_attempts: list[dict[str, Any]] | None = None,
        ) -> dict[str, Any]:
            nonlocal num_atomic_attempts, num_dependency_recoveries
            required_holding = _required_holding_object(call)
            navigation_result: dict[str, Any] | None = None
            grounding_after_navigation: dict[str, Any] | None = None
            recovery_attempts = list(accumulated_recovery_attempts or [])

            def recovery_episode_id(base: str) -> str:
                if dependency_recovery_count == 0:
                    return base
                return f"{base}_recovery{dependency_recovery_count}"

            def with_recovery_attempts(result: dict[str, Any]) -> dict[str, Any]:
                if recovery_attempts:
                    result["recovery_attempts"] = list(recovery_attempts)
                return result

            def recover_dropped_holding(
                *, phase: str, trigger_failure_code: str
            ) -> dict[str, Any] | None:
                nonlocal num_dependency_recoveries
                if (
                    retry_recovery_mode != "dependency_recovery"
                    or required_holding is None
                ):
                    return None
                # Invalidate the scheduler handoff immediately. This uses only the
                # failure reported by the active verifier / guard and never queries
                # a different reasoning backend.
                verified_holding_objects.discard(required_holding)
                transition = {
                    "type": "HOLDING_DEPENDENCY_INVALIDATED",
                    "plan_step_index": index,
                    "subgoal_id": call.subgoal_id,
                    "atomic_task": call.atomic_task,
                    "failed_predicate": f"holding({required_holding})",
                    "trigger_failure_code": trigger_failure_code,
                    "recovery_phase": phase,
                    "recovery_cycle_index": dependency_recovery_count + 1,
                }
                transition_events.append(transition)
                if dependency_recovery_count >= max_task_retries:
                    return {
                        "recovered": False,
                        "reason": "dependency_recovery_budget_exhausted",
                        "transition": transition,
                        "rollback_subgoal_id": None,
                    }
                recovered, rollback_call, attempts = retry_previous_holding_skill(
                    current_call=call,
                    current_index=index,
                    object_alias=required_holding,
                    recovery_phase=phase,
                    trigger_failure_code=trigger_failure_code,
                    recovery_cycle_index=dependency_recovery_count + 1,
                )
                recovery_attempts.extend(attempts)
                if not recovered:
                    return {
                        "recovered": False,
                        "reason": "holding_producer_recovery_failed",
                        "transition": transition,
                        "rollback_subgoal_id": (
                            rollback_call.subgoal_id if rollback_call else None
                        ),
                    }
                verified_holding_objects.add(required_holding)
                num_dependency_recoveries += 1
                return {"recovered": True, "transition": transition}

            if self.grounder is not None and call.atomic_task != "NavigateKitchen":
                grounding_before = grounding_result_to_dict(
                    self.grounder.ground(env=env, atomic_task_call=call)
                )
                grounding_before = attach_verified_handoff(
                    grounding_before, required_holding
                )
                grounding_before.update(
                    {
                        "phase": (
                            "before_operation"
                            if dependency_recovery_count == 0
                            else "after_drop_recovery"
                        ),
                        "plan_step_index": index,
                        "subgoal_id": call.subgoal_id,
                        "dependency_recovery_count": dependency_recovery_count,
                    }
                )
                grounding_events.append(grounding_before)
                print(
                    f"[Grounding] {call.subgoal_id}: "
                    f"status={grounding_before.get('status')} "
                    f"fixture={grounding_before.get('target_fixture_alias')}"
                )
                if (
                    required_holding is not None
                    and required_holding not in verified_holding_objects
                ):
                    transition = {
                        "type": "PRECONDITION_NOT_SATISFIED",
                        "plan_step_index": index,
                        "subgoal_id": call.subgoal_id,
                        "atomic_task": call.atomic_task,
                        "failed_predicate": f"holding({required_holding})",
                    }
                    transition_events.append(transition)
                    recovered = False
                    rollback_call = None
                    precondition_attempts: list[dict[str, Any]] = []
                    if retry_recovery_mode != "same_skill_no_drop_retry":
                        (
                            recovered,
                            rollback_call,
                            precondition_attempts,
                        ) = retry_previous_holding_skill(
                            current_call=call,
                            current_index=index,
                            object_alias=required_holding,
                        )
                    recovery_attempts.extend(precondition_attempts)
                    if recovered:
                        verified_holding_objects.add(required_holding)
                        grounding_after_recovery = grounding_result_to_dict(
                            self.grounder.ground(env=env, atomic_task_call=call)
                        )
                        grounding_after_recovery = attach_verified_handoff(
                            grounding_after_recovery, required_holding
                        )
                        grounding_after_recovery.update(
                            {
                                "phase": "after_previous_skill_recovery",
                                "plan_step_index": index,
                                "subgoal_id": call.subgoal_id,
                            }
                        )
                        grounding_events.append(grounding_after_recovery)
                        grounding_before = grounding_after_recovery
                    if not recovered:
                        result = grounding_failure(
                            call,
                            code="PRECONDITION_NOT_SATISFIED",
                            grounding=grounding_before,
                        )
                        result.update(transition)
                        result["rollback_subgoal_id"] = (
                            rollback_call.subgoal_id if rollback_call else None
                        )
                        result["plan_step_index"] = index
                        return with_recovery_attempts(result)
                if (
                    not grounding_before.get("grounded", False)
                    and not bool(
                        getattr(
                            self.atomic_task_policy_adapter,
                            "handles_target_aware_navigation",
                            False,
                        )
                    )
                ):
                    target_fixture = grounding_before.get("target_fixture_alias")
                    if not target_fixture:
                        result = grounding_failure(
                            call,
                            code="GROUNDING_TARGET_UNRESOLVED",
                            grounding=grounding_before,
                        )
                        result["plan_step_index"] = index
                        return with_recovery_attempts(result)

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
                        base_episode_id=recovery_episode_id(
                            f"{episode_id}_step{index + 1}_navigate"
                        ),
                        retry_label="navigation",
                        holding_dependency=required_holding,
                    )
                    num_atomic_attempts += attempts
                    navigation_result["inserted_before_plan_step_index"] = index
                    navigation_result["inserted_for_subgoal_id"] = call.subgoal_id
                    navigation_result[
                        "dependency_recovery_count"
                    ] = dependency_recovery_count
                    inserted_navigation_results.append(navigation_result)
                    if not navigation_result.get("success", False):
                        recovery = None
                        if _is_object_drop(navigation_result):
                            recovery = recover_dropped_holding(
                                phase="navigation",
                                trigger_failure_code=(
                                    _result_failure_code(navigation_result)
                                    or "OBJECT_DROPPED"
                                ),
                            )
                            if recovery is not None and recovery["recovered"]:
                                return execute_plan_step(
                                    index=index,
                                    call=call,
                                    dependency_recovery_count=(
                                        dependency_recovery_count + 1
                                    ),
                                    accumulated_recovery_attempts=recovery_attempts,
                                )
                        result = grounding_failure(
                            call,
                            code="DYNAMIC_NAVIGATION_FAILED",
                            grounding=grounding_before,
                            navigation_result=navigation_result,
                        )
                        result["plan_step_index"] = index
                        if recovery is not None:
                            result["dependency_recovery"] = recovery
                        return with_recovery_attempts(result)

                    grounding_after_navigation = grounding_result_to_dict(
                        self.grounder.ground(env=env, atomic_task_call=call)
                    )
                    grounding_after_navigation = attach_verified_handoff(
                        grounding_after_navigation, required_holding
                    )
                    grounding_after_navigation.update(
                        {
                            "phase": "after_inserted_navigation",
                            "plan_step_index": index,
                            "subgoal_id": call.subgoal_id,
                            "dependency_recovery_count": dependency_recovery_count,
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
                        return with_recovery_attempts(result)
            else:
                grounding_before = None

            result, attempts = execute_with_retries(
                call,
                base_episode_id=recovery_episode_id(
                    f"{episode_id}_step{index + 1}"
                ),
                retry_label="operation",
                holding_dependency=required_holding,
                scheduler_context=(
                    {"grounding_result": grounding_before}
                    if grounding_before is not None
                    else None
                ),
            )
            num_atomic_attempts += attempts
            if (
                not result.get("success", False)
                and _is_object_drop(result)
            ):
                recovery = recover_dropped_holding(
                    phase="operation",
                    trigger_failure_code=(
                        _result_failure_code(result) or "OBJECT_DROPPED"
                    ),
                )
                if recovery is not None and recovery["recovered"]:
                    return execute_plan_step(
                        index=index,
                        call=call,
                        dependency_recovery_count=dependency_recovery_count + 1,
                        accumulated_recovery_attempts=recovery_attempts,
                    )
                if recovery is not None:
                    result["dependency_recovery"] = recovery
            if grounding_before is not None:
                result["grounding_before_operation"] = grounding_before
            if navigation_result is not None:
                result["inserted_navigation_result"] = navigation_result
            if grounding_after_navigation is not None:
                result["grounding_after_navigation"] = grounding_after_navigation
            result["plan_step_index"] = index
            result["dependency_recovery_count"] = dependency_recovery_count
            if result.get("success", False):
                produced_holding = _produced_holding_object(call)
                if produced_holding is not None:
                    verified_holding_objects.add(produced_holding)
                if required_holding is not None and call.atomic_task == "PlaceObject":
                    verified_holding_objects.discard(required_holding)
            return with_recovery_attempts(result)

        for index, call in enumerate(calls):
            result = execute_plan_step(index=index, call=call)
            step_results.append(result)
            if stop_on_unsuccessful and not result.get("success", False):
                break

        all_completed = len(step_results) == len(calls)
        success = all_completed and all(
            bool(result.get("success", False)) for result in step_results
        )
        if success:
            status = "success"
        elif step_results and step_results[-1].get("status") == "failed":
            status = "failed"
        else:
            status = "uncertain"
        target_aware_stages = list(
            getattr(self.atomic_task_policy_adapter, "stage_results", [])
        )
        target_aware_summary = _summarize_target_aware_stages(
            target_aware_stages
        )
        return {
            "module": "LONG_HORIZON_TASK",
            "status": status,
            "success": success,
            "num_planned_steps": len(calls),
            "num_executed_steps": len(step_results),
            "num_atomic_attempts": num_atomic_attempts,
            "num_inserted_navigation_steps": len(inserted_navigation_results),
            "num_target_aware_stages": len(target_aware_stages),
            "target_aware_stages": target_aware_stages,
            "target_aware_summary": target_aware_summary,
            "retry_recovery_mode": retry_recovery_mode,
            "num_dependency_recoveries": num_dependency_recoveries,
            "completed_all_steps": all_completed,
            "grounding_events": grounding_events,
            "inserted_navigation_results": inserted_navigation_results,
            "transition_events": transition_events,
            "recovery_events": recovery_events,
            "step_results": step_results,
        }
