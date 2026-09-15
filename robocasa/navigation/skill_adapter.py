"""Compose target-aware navigation, local refinement, and an existing policy adapter."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from robocasa.atomic_task_schemas import AtomicTaskCall
from robocasa.navigation.debug_artifacts import save_navigation_plan_artifacts
from robocasa.post_grasp_transport import (
    PostGraspTransportController,
    resolve_object_alias,
)
from robocasa.post_place_retreat import PostPlaceRetreatController
from robocasa.post_skill_return import PostSkillReturnController
from robocasa.work_pose_dataset import current_base_pose


def _yaw_delta_deg(before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
    delta = (float(after["yaw_rad"]) - float(before["yaw_rad"]) + math.pi) % (
        2.0 * math.pi
    ) - math.pi
    return math.degrees(delta)


def _pose_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    dx = float(after["position"][0]) - float(before["position"][0])
    dy = float(after["position"][1]) - float(before["position"][1])
    return {
        "delta_x_m": dx,
        "delta_y_m": dy,
        "delta_translation_m": math.hypot(dx, dy),
        "delta_yaw_deg": _yaw_delta_deg(before, after),
    }


def _navigation_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    execution = result.get("execution")
    execution_summary = None
    if isinstance(execution, Mapping):
        execution_summary = {
            key: execution.get(key)
            for key in (
                "success",
                "reason",
                "failure_code",
                "reached_waypoints",
                "total_waypoints",
                "steps",
                "position_error",
                "yaw_error",
                "final_pose",
            )
        }
    return {
        key: result.get(key)
        for key in (
            "success",
            "reason",
            "target_name",
            "target_type",
            "target_position",
            "selected_goal",
            "path_length",
            "waypoint_count",
            "skipped",
            "raw_candidate_count",
            "valid_candidate_count",
            "artifacts",
        )
    } | {"execution": execution_summary}


def _refinement_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "module",
            "status",
            "success",
            "failure_code",
            "num_decisions",
            "num_executed_actions",
            "total_env_steps",
            "total_translation_m",
            "total_rotation_deg",
            "artifact_dir",
        )
    }


def _verifier_success(result: Mapping[str, Any]) -> bool | None:
    verifier = result.get("verifier_result")
    if not isinstance(verifier, Mapping):
        return None
    if "goal_satisfied" in verifier:
        return bool(verifier["goal_satisfied"])
    if "success" in verifier:
        return bool(verifier["success"])
    return None


class TargetAwarePolicyAdapter:
    """Run optional coarse navigation and local refinement before π0.5."""

    handles_target_aware_navigation = True

    def __init__(
        self,
        *,
        policy_adapter: Any,
        navigation_pipeline: Any,
        refiner: Any | None = None,
        navigation_enabled: bool = True,
        local_refiner_enabled: bool = True,
        local_refiner_failure_policy: str = "use_coarse_pose",
        post_grasp_transport_controller: Any | None = None,
        post_place_retreat_controller: Any | None = None,
        post_skill_return_controller: Any | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(getattr(policy_adapter, "execute", None)):
            raise TypeError("policy_adapter must provide execute(...)")
        if not callable(
            getattr(navigation_pipeline, "navigate_to_manipulation_target", None)
        ):
            raise TypeError("navigation_pipeline must provide navigate_to_manipulation_target(...)")
        if refiner is not None and not callable(getattr(refiner, "refine", None)):
            raise TypeError("refiner must provide refine(...)")
        if local_refiner_enabled and refiner is None:
            raise ValueError("local_refiner_enabled requires a refiner")
        if local_refiner_failure_policy not in {"use_coarse_pose", "abort"}:
            raise ValueError(
                "local_refiner_failure_policy must be use_coarse_pose or abort"
            )
        self.policy_adapter = policy_adapter
        self.navigation_pipeline = navigation_pipeline
        self.refiner = refiner
        self.navigation_enabled = bool(navigation_enabled)
        self.handles_target_aware_navigation = self.navigation_enabled
        self.local_refiner_enabled = bool(local_refiner_enabled)
        self.local_refiner_failure_policy = local_refiner_failure_policy
        self.context = dict(context or {})
        self.stage_results: list[dict[str, Any]] = []
        pipeline_config = getattr(navigation_pipeline, "config", {})
        navigation_config = (
            pipeline_config.get("navigation", {})
            if isinstance(pipeline_config, Mapping)
            else {}
        )
        fail_open_reasons = navigation_config.get(
            "continue_policy_after_failure_reasons", ()
        )
        if isinstance(fail_open_reasons, (str, bytes)) or not isinstance(
            fail_open_reasons, (list, tuple, set, frozenset)
        ):
            raise ValueError(
                "navigation.continue_policy_after_failure_reasons must be a list"
            )
        self.navigation_fail_open_reasons = frozenset(
            str(reason).strip()
            for reason in fail_open_reasons
            if str(reason).strip()
        )
        self.navigation_artifacts_enabled = bool(
            navigation_config.get("save_debug_artifacts", False)
        )
        self.navigation_artifact_subdir = str(
            navigation_config.get(
                "debug_artifact_subdir", "navigation_planning"
            )
        ).strip() or "navigation_planning"
        self._navigation_artifact_index = 0
        transport_config = (
            pipeline_config.get("post_grasp_transport", {})
            if isinstance(pipeline_config, Mapping)
            else {}
        )
        if (
            post_grasp_transport_controller is None
            and transport_config
            and bool(transport_config.get("enabled", True))
        ):
            post_grasp_transport_controller = PostGraspTransportController(
                transport_config,
                log_dir=getattr(policy_adapter, "log_dir", None),
            )
        if (
            post_grasp_transport_controller is not None
            and not callable(
                getattr(post_grasp_transport_controller, "prepare", None)
            )
        ):
            raise TypeError(
                "post_grasp_transport_controller must provide prepare(...)"
            )
        self.post_grasp_transport_controller = post_grasp_transport_controller
        self.post_grasp_transport_enabled = bool(
            post_grasp_transport_controller is not None
        )
        retreat_config = (
            pipeline_config.get("post_place_retreat", {})
            if isinstance(pipeline_config, Mapping)
            else {}
        )
        if (
            post_place_retreat_controller is None
            and retreat_config
            and bool(retreat_config.get("enabled", True))
        ):
            post_place_retreat_controller = PostPlaceRetreatController(
                retreat_config,
                log_dir=getattr(policy_adapter, "log_dir", None),
            )
        if (
            post_place_retreat_controller is not None
            and not callable(
                getattr(post_place_retreat_controller, "prepare", None)
            )
        ):
            raise TypeError(
                "post_place_retreat_controller must provide prepare(...)"
            )
        self.post_place_retreat_controller = post_place_retreat_controller
        self.post_place_retreat_enabled = bool(
            post_place_retreat_controller is not None
        )
        self.post_place_retreat_failure_policy = str(
            retreat_config.get("failure_policy", "warn")
        )
        if self.post_place_retreat_failure_policy not in {"warn", "block"}:
            raise ValueError(
                "post_place_retreat.failure_policy must be warn or block"
            )
        return_config = (
            pipeline_config.get("post_skill_return", {})
            if isinstance(pipeline_config, Mapping)
            else {}
        )
        if (
            post_skill_return_controller is None
            and return_config
            and bool(return_config.get("enabled", True))
        ):
            post_skill_return_controller = PostSkillReturnController(
                return_config,
                log_dir=getattr(policy_adapter, "log_dir", None),
            )
        if post_skill_return_controller is not None and not callable(
            getattr(post_skill_return_controller, "prepare", None)
        ):
            raise TypeError("post_skill_return_controller must provide prepare(...)")
        self.post_skill_return_controller = post_skill_return_controller
        self.post_skill_return_enabled = post_skill_return_controller is not None
        self.post_skill_return_enforce_result = bool(
            return_config.get("enforce_result", True)
        )
        self.post_skill_return_failure_policy = str(
            return_config.get("failure_policy", "warn")
        )
        self._pending_post_grasp: dict[str, Any] | None = None
        self._pending_post_place_retreat: dict[str, Any] | None = None
        self._transport_ready_objects: set[str] = set()

    @property
    def log_dir(self) -> Any:
        return getattr(self.policy_adapter, "log_dir", None)

    def _save_navigation_artifacts(
        self,
        *,
        call: AtomicTaskCall,
        navigation_result: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not self.navigation_artifacts_enabled:
            return None
        log_dir = self.log_dir
        if log_dir is None:
            return {"saved": False, "error": "policy adapter has no log_dir"}
        self._navigation_artifact_index += 1
        try:
            saved = save_navigation_plan_artifacts(
                output_root=Path(log_dir) / self.navigation_artifact_subdir,
                artifact_index=self._navigation_artifact_index,
                subgoal_id=call.subgoal_id,
                pipeline=self.navigation_pipeline,
                navigation_result=navigation_result,
            )
            saved["saved"] = True
            print(
                "[NavigationArtifacts] {} reason={} directory={}".format(
                    call.subgoal_id,
                    navigation_result.get("reason"),
                    saved["directory"],
                )
            )
            return saved
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(
                "[NavigationArtifacts] {} save failed: {}".format(
                    call.subgoal_id, exc
                )
            )
            return {"saved": False, "error": str(exc)}

    @staticmethod
    def _failure_result(
        call: AtomicTaskCall,
        *,
        failure_code: str,
        stage_status: str,
        retryable: bool,
        stage: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "module": "ATOMIC_TASK_POLICY",
            "subgoal_id": call.subgoal_id,
            "atomic_task": call.atomic_task,
            "status": "failed",
            "stage_status": stage_status,
            "success": False,
            "failure_code": failure_code,
            "verifier_result": {
                "status": "failed",
                "goal_satisfied": False,
                "failure_code": failure_code,
                "retryable": retryable,
            },
            "target_aware_stage": dict(stage),
        }

    def execute(
        self,
        *,
        env: Any,
        scheduler_query: Mapping[str, Any],
        episode_id: int | str,
    ) -> dict[str, Any]:
        if not isinstance(scheduler_query, Mapping) or "atomic_task_call" not in scheduler_query:
            raise ValueError("scheduler_query is missing atomic_task_call")
        call = AtomicTaskCall.from_mapping(scheduler_query["atomic_task_call"])
        if call.atomic_task == "NavigateKitchen":
            result = dict(
                self.policy_adapter.execute(
                    env=env,
                    scheduler_query=scheduler_query,
                    episode_id=episode_id,
                )
            )
            verified = bool(
                result.get("success", False)
                and _verifier_success(result) is not False
            )
            if self.post_skill_return_enabled and verified:
                returned = dict(
                    self.post_skill_return_controller.prepare(
                        env=env,
                        atomic_task_call=call,
                        episode_id=episode_id,
                        skill_verified=True,
                    )
                )
                result["post_skill_return_result"] = returned
                result["post_skill_return_enforce_result"] = (
                    self.post_skill_return_enforce_result
                )
                result["post_skill_return_success"] = bool(
                    returned.get("return_ready", False)
                )
                if (
                    self.post_skill_return_enforce_result
                    and not returned.get("return_ready", False)
                ):
                    completed = dict(result)
                    failure_code = str(
                        returned.get("failure_code")
                        or "POST_SKILL_RETURN_FAILED"
                    )
                    result.update(
                        {
                            "status": "failed",
                            "success": False,
                            "failure_code": failure_code,
                            "verifier_result": {
                                "status": "failed",
                                "goal_satisfied": False,
                                "failure_code": failure_code,
                                "retryable": True,
                            },
                            "completed_policy_result": completed,
                        }
                    )
            return result

        pose_before = current_base_pose(env)
        stage: dict[str, Any] = {
            "skill": call.atomic_task,
            "subgoal_id": call.subgoal_id,
            "target": None,
            "target_type": None,
            "current_base_pose": pose_before,
            "navigation_enabled": self.navigation_enabled,
            "navigation_required": False,
            "navigation_success": None,
            "navigation_reason": "DISABLED" if not self.navigation_enabled else None,
            "navigation_bypassed": False,
            "navigation_warning": None,
            "navigation_artifacts": None,
            "post_grasp_transport_enabled": self.post_grasp_transport_enabled,
            "post_grasp_transport_required": False,
            "post_grasp_transport_success": None,
            "transport_ready": None,
            "post_place_retreat_enabled": self.post_place_retreat_enabled,
            "post_place_retreat_required": False,
            "post_place_retreat_success": None,
            "retreat_ready": None,
            "post_place_retreat_warning": None,
            "pre_navigation_post_place_retreat_required": False,
            "pre_navigation_post_place_retreat_success": None,
            "local_refinement_used": False,
            "local_refinement_success": None,
            "local_refinement_delta": None,
            "policy_started": False,
            "policy_success": None,
            "verifier_success": None,
            "post_skill_return_enabled": self.post_skill_return_enabled,
            "post_skill_return_enforce_result": self.post_skill_return_enforce_result,
            "post_skill_return_required": False,
            "post_skill_return_success": None,
            "post_skill_return_warning": None,
        }
        navigation_result = None
        navigation_value = None
        if self.navigation_enabled:
            navigation_context = dict(self.context)
            grounding_result = scheduler_query.get("grounding_result")
            if isinstance(grounding_result, Mapping):
                navigation_context["grounding_result"] = dict(grounding_result)
            navigation_context["transport_ready_objects"] = sorted(
                self._transport_ready_objects
            )
            grasp_preparation_holder: dict[str, Any] = {}
            retreat_preparation_holder: dict[str, Any] = {}

            def prepare_pending_retreat() -> Mapping[str, Any]:
                if (
                    self._pending_post_place_retreat is None
                    or self.post_place_retreat_controller is None
                ):
                    return {
                        "success": False,
                        "retreat_ready": False,
                        "navigation_safe": False,
                        "failure_code": "POST_PLACE_RETREAT_FAILED",
                        "reason": "NO_PENDING_POST_PLACE_RETREAT",
                    }
                pending = self._pending_post_place_retreat
                preparation = dict(
                    self.post_place_retreat_controller.prepare(
                        env=env,
                        atomic_task_call=pending["call"],
                        episode_id=episode_id,
                        place_verified=True,
                    )
                )
                preparation["navigation_safe"] = bool(
                    preparation.get("retreat_ready", False)
                )
                retreat_preparation_holder.update(preparation)
                if preparation["navigation_safe"]:
                    self._pending_post_place_retreat = None
                return preparation

            def prepare_pending_grasp() -> Mapping[str, Any]:
                if (
                    self._pending_post_grasp is None
                    or self.post_grasp_transport_controller is None
                ):
                    return {
                        "success": False,
                        "transport_ready": False,
                        "failure_code": "TRANSPORT_READY_FAILED",
                        "reason": "NO_PENDING_VERIFIED_PICK",
                    }
                pending = self._pending_post_grasp
                preparation = dict(
                    self.post_grasp_transport_controller.prepare(
                        env=env,
                        atomic_task_call=pending["call"],
                        episode_id=episode_id,
                        pick_verified=bool(pending["pick_verified"]),
                    )
                )
                preparation["navigation_safe"] = bool(
                    preparation.get("transport_ready", False)
                )
                grasp_preparation_holder.update(preparation)
                alias = str(preparation.get("object_alias") or "")
                if preparation.get("transport_ready", False) and alias:
                    self._transport_ready_objects.add(alias)
                    self._pending_post_grasp = None
                elif (
                    preparation.get("failure_code")
                    == "OBJECT_DROPPED_DURING_PREP"
                ):
                    self._transport_ready_objects.discard(alias)
                    self._pending_post_grasp = None
                return preparation

            def prepare_for_navigation() -> Mapping[str, Any]:
                if self._pending_post_place_retreat is not None:
                    retreat_preparation = prepare_pending_retreat()
                    if not retreat_preparation.get(
                        "navigation_safe", False
                    ):
                        return retreat_preparation
                if self._pending_post_grasp is not None:
                    return prepare_pending_grasp()
                return {
                    "success": True,
                    "navigation_safe": True,
                    "reason": "PRE_NAVIGATION_SAFETY_READY",
                }

            before_navigation = (
                prepare_for_navigation
                if (
                    (
                        self.post_grasp_transport_enabled
                        and self._pending_post_grasp is not None
                    )
                    or (
                        self.post_place_retreat_enabled
                        and self._pending_post_place_retreat is not None
                    )
                )
                else None
            )
            navigation_result = self.navigation_pipeline.navigate_to_manipulation_target(
                env,
                call,
                context=navigation_context,
                plan_only=False,
                before_navigation=before_navigation,
            )
            navigation_value = navigation_result.to_dict()
            navigation_artifacts = self._save_navigation_artifacts(
                call=call,
                navigation_result=navigation_value,
            )
            if navigation_artifacts is not None:
                navigation_value["artifacts"] = navigation_artifacts
                stage["navigation_artifacts"] = navigation_artifacts
            if grasp_preparation_holder:
                stage["post_grasp_transport_required"] = True
                stage["post_grasp_transport_success"] = bool(
                    grasp_preparation_holder.get("success", False)
                )
                stage["transport_ready"] = bool(
                    grasp_preparation_holder.get("transport_ready", False)
                )
                stage["post_grasp_transport_result"] = dict(
                    grasp_preparation_holder
                )
            if retreat_preparation_holder:
                stage["pre_navigation_post_place_retreat_required"] = True
                stage["pre_navigation_post_place_retreat_success"] = bool(
                    retreat_preparation_holder.get("retreat_ready", False)
                )
                stage["pre_navigation_post_place_retreat_result"] = dict(
                    retreat_preparation_holder
                )
            stage["target"] = navigation_result.target_name
            stage["target_type"] = navigation_result.target_type
            stage["navigation_required"] = not bool(navigation_result.skipped)
            stage["navigation_success"] = bool(navigation_result.success)
            stage["navigation_reason"] = navigation_result.reason
            stage["coarse_goal"] = navigation_result.selected_goal
            stage["navigation_result"] = _navigation_summary(navigation_value)
            if not navigation_result.success:
                reason = str(navigation_result.reason)
                if reason in self.navigation_fail_open_reasons:
                    stage["navigation_bypassed"] = True
                    stage["navigation_warning"] = reason
                    print(
                        "[AtomicSkill] {} navigation=WARNING reason={} "
                        "policy_started=True fail_open=True".format(
                            call.subgoal_id, reason
                        )
                    )
                else:
                    retryable = reason in {
                        "MOTION_STALLED",
                        "OBJECT_DROPPED",
                        "OBJECT_DROPPED_DURING_PREP",
                        "POST_GRASP_LIFT_FAILED",
                        "POST_GRASP_RETRACT_FAILED",
                        "POST_PLACE_LIFT_FAILED",
                        "POST_PLACE_RETRACT_FAILED",
                        "POST_PLACE_RETREAT_FAILED",
                        "TRANSPORT_NOT_READY",
                        "TRANSPORT_READY_FAILED",
                        "FINAL_POSE_NOT_REACHED",
                    }
                    self.stage_results.append(stage)
                    print(
                        "[AtomicSkill] {} navigation=FAILED reason={} "
                        "policy_started=False".format(call.subgoal_id, reason)
                    )
                    failure = self._failure_result(
                        call,
                        failure_code=reason,
                        stage_status=(
                            "PRE_NAVIGATION_PREPARATION_FAILED"
                            if reason
                            in {
                                "OBJECT_DROPPED_DURING_PREP",
                                "POST_GRASP_LIFT_FAILED",
                                "POST_GRASP_RETRACT_FAILED",
                                "TRANSPORT_NOT_READY",
                                "TRANSPORT_READY_FAILED",
                                "POST_PLACE_LIFT_FAILED",
                                "POST_PLACE_RETRACT_FAILED",
                                "POST_PLACE_RETREAT_FAILED",
                            }
                            else "NAVIGATION_FAILED"
                        ),
                        retryable=retryable,
                        stage=stage,
                    )
                    failure["navigation_result"] = navigation_value
                    return failure

        if self.local_refiner_enabled:
            stage["local_refinement_used"] = True
            refiner_before = current_base_pose(env)
            grounding_for_refiner = None
            if navigation_result is not None:
                grounding_for_refiner = {
                    "status": (
                        "grounded"
                        if navigation_result.success
                        else "navigation_failed_continued"
                        if stage["navigation_bypassed"]
                        else "failed"
                    ),
                    "target_name": navigation_result.target_name,
                    "target_type": navigation_result.target_type,
                    "target_position": navigation_result.target_position,
                    "selected_goal": navigation_result.selected_goal,
                }
            refinement = dict(
                self.refiner.refine(
                    env=env,
                    atomic_task_call=call.to_dict(),
                    grounding_result=grounding_for_refiner,
                    episode_id=episode_id,
                )
            )
            refiner_after = current_base_pose(env)
            delta = _pose_delta(refiner_before, refiner_after)
            stage["local_refinement_success"] = bool(refinement.get("success", False))
            stage["local_refinement_delta"] = delta
            stage["local_refiner_result"] = _refinement_summary(refinement)
            refiner_failure = str(refinement.get("failure_code") or "LOCAL_REFINEMENT_FAILED")
            must_abort = (
                refiner_failure == "OBJECT_DROPPED"
                or self.local_refiner_failure_policy == "abort"
            )
            if not refinement.get("success", False) and must_abort:
                self.stage_results.append(stage)
                print(
                    "[AtomicSkill] {} refiner=FAILED reason={} policy_started=False".format(
                        call.subgoal_id, refiner_failure
                    )
                )
                failure = self._failure_result(
                    call,
                    failure_code=(
                        "OBJECT_DROPPED"
                        if refiner_failure == "OBJECT_DROPPED"
                        else "LOCAL_REFINEMENT_FAILED"
                    ),
                    stage_status="LOCAL_REFINEMENT_FAILED",
                    retryable=True,
                    stage=stage,
                )
                failure["navigation_result"] = (
                    navigation_result.to_dict()
                    if navigation_result is not None
                    else None
                )
                failure["local_pose_refinement_result"] = refinement
                return failure
            stage["local_refinement_failure_policy_applied"] = bool(
                not refinement.get("success", False)
            )
            if not refinement.get("success", False) and not must_abort:
                stage["local_refinement_attempt_delta"] = delta
                restore = self.navigation_pipeline.restore_coarse_pose(
                    env, refiner_before, context=self.context
                )
                restore_value = restore.to_dict()
                stage["coarse_pose_restore"] = {
                    key: restore_value.get(key)
                    for key in (
                        "success",
                        "reason",
                        "failure_code",
                        "reached_waypoints",
                        "total_waypoints",
                        "steps",
                        "position_error",
                        "yaw_error",
                        "final_pose",
                    )
                }
                restored_pose = current_base_pose(env)
                stage["local_refinement_delta"] = _pose_delta(
                    refiner_before, restored_pose
                )
                if not restore.success:
                    self.stage_results.append(stage)
                    failure_code = (
                        "OBJECT_DROPPED"
                        if restore.reason == "OBJECT_DROPPED"
                        else "LOCAL_REFINEMENT_FAILED"
                    )
                    failure = self._failure_result(
                        call,
                        failure_code=failure_code,
                        stage_status="LOCAL_REFINEMENT_FAILED",
                        retryable=True,
                        stage=stage,
                    )
                    failure["coarse_pose_restore_result"] = restore_value
                    failure["local_pose_refinement_result"] = refinement
                    return failure
        else:
            stage["local_refinement_success"] = None

        if self._pending_post_place_retreat is not None:
            # A skipped navigation means the next policy will deliberately take
            # control of the arm at the current work region. Do not replay a
            # stale open-gripper retreat after that newer manipulation begins.
            stage["pending_post_place_retreat_superseded_by_policy"] = True
            self._pending_post_place_retreat = None

        stage["policy_started"] = True
        policy_result = dict(
            self.policy_adapter.execute(
                env=env,
                scheduler_query=scheduler_query,
                episode_id=episode_id,
            )
        )
        stage["policy_success"] = bool(policy_result.get("success", False))
        stage["verifier_success"] = _verifier_success(policy_result)
        skill_verified = bool(
            policy_result.get("success", False)
            and stage["verifier_success"] is not False
        )
        post_skill_return = None
        if self.post_skill_return_enabled and skill_verified:
            post_skill_return = dict(
                self.post_skill_return_controller.prepare(
                    env=env,
                    atomic_task_call=call,
                    episode_id=episode_id,
                    skill_verified=True,
                )
            )
            return_ready = bool(post_skill_return.get("return_ready", False))
            stage["post_skill_return_required"] = True
            stage["post_skill_return_success"] = return_ready
            stage["post_skill_return_result"] = post_skill_return
            if not return_ready:
                failure_code = str(
                    post_skill_return.get("failure_code")
                    or "POST_SKILL_RETURN_FAILED"
                )
                stage["post_skill_return_warning"] = failure_code
                dropped = failure_code in {
                    "OBJECT_DROPPED_DURING_PREP",
                    "OBJECT_DROPPED_DURING_POST_SKILL_RETURN",
                }
                # A Pick that cannot reach its carry pose must never fall back
                # to immediate base navigation, even if an external config
                # selects the legacy best-effort warning policy.
                if (
                    self.post_skill_return_enforce_result
                    and (
                        call.atomic_task == "PickObject"
                        or dropped
                        or self.post_skill_return_failure_policy == "block"
                    )
                ):
                    stage["final_base_pose"] = current_base_pose(env)
                    self.stage_results.append(stage)
                    failure = self._failure_result(
                        call,
                        failure_code=failure_code,
                        stage_status="POST_SKILL_RETURN_FAILED",
                        retryable=True,
                        stage=stage,
                    )
                    failure["completed_policy_result"] = policy_result
                    failure["post_skill_return_result"] = post_skill_return
                    return failure
        if call.atomic_task == "PickObject":
            object_alias = (
                resolve_object_alias(env, call)
                if (
                    self.post_grasp_transport_enabled
                    or self.post_skill_return_enabled
                )
                else None
            )
            pick_verified = bool(
                policy_result.get("success", False)
                and stage["verifier_success"] is not False
                and object_alias
            )
            return_ready = bool(
                post_skill_return
                and post_skill_return.get("return_ready", False)
            )
            stage["post_grasp_transport_pending"] = bool(
                pick_verified
                and not return_ready
                and self.post_skill_return_enforce_result
            )
            stage["post_grasp_object_alias"] = object_alias
            if pick_verified and object_alias is not None and return_ready:
                self._transport_ready_objects.add(object_alias)
                self._pending_post_grasp = None
                stage["transport_ready"] = True
                stage["post_grasp_transport_success"] = True
            elif (
                pick_verified
                and object_alias is not None
                and not self.post_skill_return_enabled
            ):
                self._transport_ready_objects.discard(object_alias)
                self._pending_post_grasp = {
                    "call": call,
                    "object_alias": object_alias,
                    "episode_id": episode_id,
                    "pick_verified": True,
                }
            elif (
                self._pending_post_grasp is not None
                and self._pending_post_grasp.get("object_alias")
                == object_alias
            ):
                self._pending_post_grasp = None
        elif call.atomic_task == "PlaceObject" and policy_result.get(
            "success", False
        ):
            place_verified = stage["verifier_success"] is not False
            if (
                self.post_place_retreat_enabled
                and place_verified
                and not self.post_skill_return_enabled
            ):
                retreat = dict(
                    self.post_place_retreat_controller.prepare(
                        env=env,
                        atomic_task_call=call,
                        episode_id=episode_id,
                        place_verified=True,
                    )
                )
                stage["post_place_retreat_required"] = True
                stage["post_place_retreat_success"] = bool(
                    retreat.get("success", False)
                )
                stage["retreat_ready"] = bool(
                    retreat.get("retreat_ready", False)
                )
                stage["post_place_retreat_result"] = retreat
                if not retreat.get("retreat_ready", False):
                    failure_code = str(
                        retreat.get("failure_code")
                        or "POST_PLACE_RETREAT_FAILED"
                    )
                    stage["post_place_retreat_warning"] = failure_code
                    self._pending_post_place_retreat = {
                        "call": call,
                        "episode_id": episode_id,
                    }
                    if self.post_place_retreat_failure_policy == "warn":
                        print(
                            "[AtomicSkill] {} post_place=WARNING reason={} "
                            "place_preserved=True navigation_will_retry=True".format(
                                call.subgoal_id, failure_code
                            )
                        )
                    else:
                        stage["final_base_pose"] = current_base_pose(env)
                        self.stage_results.append(stage)
                        failure = self._failure_result(
                            call,
                            failure_code=failure_code,
                            stage_status="POST_PLACE_RETREAT_FAILED",
                            retryable=False,
                            stage=stage,
                        )
                        failure["completed_place_result"] = policy_result
                        failure["post_place_retreat_result"] = retreat
                        failure["navigation_result"] = navigation_value
                        print(
                            "[AtomicSkill] {} post_place=FAILED reason={} "
                            "next_skill_blocked=True".format(
                                call.subgoal_id, failure_code
                            )
                        )
                        return failure
                else:
                    self._pending_post_place_retreat = None
            for key in ("held_object_id", "object_id", "source_object_id"):
                value = call.arguments.get(key)
                if isinstance(value, str) and value:
                    self._transport_ready_objects.discard(value)
                    if (
                        self._pending_post_grasp is not None
                        and self._pending_post_grasp.get("object_alias")
                        == value
                    ):
                        self._pending_post_grasp = None
        stage["final_base_pose"] = current_base_pose(env)
        policy_result["target_aware_stage"] = stage
        policy_result["navigation_result"] = navigation_value
        if self.local_refiner_enabled:
            policy_result["local_pose_refinement_result"] = refinement
        self.stage_results.append(stage)
        print(
            "[AtomicSkill] {} navigation={} refiner={} policy={} verifier={}".format(
                call.subgoal_id,
                stage["navigation_success"],
                stage["local_refinement_success"],
                stage["policy_success"],
                stage["verifier_success"],
            )
        )
        return policy_result
