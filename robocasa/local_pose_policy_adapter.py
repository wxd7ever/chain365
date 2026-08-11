"""Policy-adapter decorator that runs local base refinement before operations."""

from __future__ import annotations

from typing import Any, Mapping

try:
    from .atomic_task_schemas import AtomicTaskCall
except ImportError:
    from atomic_task_schemas import AtomicTaskCall


class LocalPoseRefiningPolicyAdapter:
    """Keep inserted navigation unchanged and refine immediately before operations."""

    def __init__(self, *, policy_adapter: Any, refiner: Any, grounder: Any):
        if not callable(getattr(policy_adapter, "execute", None)):
            raise TypeError("policy_adapter must provide execute(...)")
        if not callable(getattr(refiner, "refine", None)):
            raise TypeError("refiner must provide refine(...)")
        if not callable(getattr(grounder, "ground", None)):
            raise TypeError("grounder must provide ground(...)")
        self.policy_adapter = policy_adapter
        self.refiner = refiner
        self.grounder = grounder
        self.refinement_results: list[dict[str, Any]] = []

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
        refinement_result: dict[str, Any] | None = None
        if call.atomic_task != "NavigateKitchen":
            grounding = self.grounder.ground(env=env, atomic_task_call=call)
            grounding_value = (
                grounding.to_dict()
                if callable(getattr(grounding, "to_dict", None))
                else dict(grounding)
            )
            print(
                f"[LocalPose] Refine base before {call.atomic_task}: "
                f"cameras={self.refiner.camera_names}"
            )
            refinement_result = dict(
                self.refiner.refine(
                    env=env,
                    atomic_task_call=call,
                    grounding_result=grounding_value,
                    episode_id=episode_id,
                )
            )
            self.refinement_results.append(refinement_result)
            print(
                f"[LocalPose] {call.subgoal_id}: "
                f"status={refinement_result.get('status')} "
                f"actions={refinement_result.get('num_executed_actions')}"
            )
            if refinement_result.get("failure_code") == "OBJECT_DROPPED":
                return {
                    "module": "ATOMIC_TASK_POLICY",
                    "subgoal_id": call.subgoal_id,
                    "atomic_task": call.atomic_task,
                    "status": "failed",
                    "success": False,
                    "verifier_result": {
                        "status": "failed",
                        "goal_satisfied": False,
                        "failure_code": "LOCAL_POSE_OBJECT_DROPPED",
                        "retryable": True,
                        "state_evidence": [refinement_result],
                    },
                    "local_pose_refinement_result": refinement_result,
                }

        result = dict(
            self.policy_adapter.execute(
                env=env,
                scheduler_query=scheduler_query,
                episode_id=episode_id,
            )
        )
        if refinement_result is not None:
            result["local_pose_refinement_result"] = refinement_result
        return result

