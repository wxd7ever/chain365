"""Scheduler-facing adapter for the shared remote pi0.5 policy client."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

try:
    from .atomic_task_horizons import load_atomic_task_horizons
    from .atomic_task_prompt_builder import build_atomic_task_prompt
    from .atomic_task_schemas import AtomicTaskCall, validate_atomic_task_call
except ImportError:  # Direct execution from the robocasa script directory.
    from atomic_task_horizons import load_atomic_task_horizons
    from atomic_task_prompt_builder import build_atomic_task_prompt
    from atomic_task_schemas import AtomicTaskCall, validate_atomic_task_call


class RemoteAtomicTaskPolicyAdapter:
    """Convert a scheduler query into one remote policy rollout."""

    def __init__(
        self,
        *,
        client: Any,
        verifier: Any,
        log_dir: str | Path,
        resize_size: int = 224,
        replan_steps: int = 5,
        atomic_task_horizon: int = 300,
        use_registry_horizons: bool = False,
        verify_interval: int = 5,
        min_steps_before_verify: int = 10,
        base_action_mode: str = "residual",
        base_residual_limit: float = 0.15,
        held_object_guard: bool = True,
        held_object_hold_confirmation_steps: int = 2,
        held_object_drop_confirmation_steps: int = 2,
        success_handoff_steps: int = 30,
        render: bool = True,
        video_skip: int = 2,
    ):
        if not callable(getattr(client, "infer", None)):
            raise TypeError("client must provide infer(observation) -> mapping")
        if not callable(verifier):
            raise TypeError("verifier must be callable")
        if held_object_hold_confirmation_steps <= 0:
            raise ValueError("held_object_hold_confirmation_steps must be positive")
        if held_object_drop_confirmation_steps <= 0:
            raise ValueError("held_object_drop_confirmation_steps must be positive")
        if success_handoff_steps < 0:
            raise ValueError("success_handoff_steps must be non-negative")
        self.client = client
        self.verifier = verifier
        self.log_dir = Path(log_dir)
        self.resize_size = resize_size
        self.replan_steps = replan_steps
        self.atomic_task_horizon = atomic_task_horizon
        self.use_registry_horizons = bool(use_registry_horizons)
        self.verify_interval = verify_interval
        self.min_steps_before_verify = min_steps_before_verify
        self.base_action_mode = base_action_mode
        self.base_residual_limit = base_residual_limit
        self.held_object_guard = bool(held_object_guard)
        self.held_object_hold_confirmation_steps = int(
            held_object_hold_confirmation_steps
        )
        self.held_object_drop_confirmation_steps = int(
            held_object_drop_confirmation_steps
        )
        self.success_handoff_steps = int(success_handoff_steps)
        self.render = render
        self.video_skip = video_skip

    def execute(
        self,
        *,
        env: Any,
        scheduler_query: Mapping[str, Any],
        episode_id: int | str,
    ) -> dict[str, Any]:
        """Execute one validated call while retaining the caller-owned env/client."""

        if not isinstance(scheduler_query, Mapping):
            raise TypeError("scheduler_query must be a mapping")
        if "atomic_task_call" not in scheduler_query:
            raise ValueError("scheduler_query is missing 'atomic_task_call'")
        call = AtomicTaskCall.from_mapping(scheduler_query["atomic_task_call"])
        validate_atomic_task_call(call)
        prompt = build_atomic_task_prompt(call)
        try:
            from .pi05_rollout import execute_pi05_atomic_task_policy
        except ImportError:
            from pi05_rollout import execute_pi05_atomic_task_policy

        configured_horizon = self.atomic_task_horizon
        horizon_source = "fixed"
        if self.use_registry_horizons:
            registry_horizons = load_atomic_task_horizons()
            configured_horizon = registry_horizons.get(
                call.atomic_task, self.atomic_task_horizon
            )
            horizon_source = (
                "robocasa_dataset_registry"
                if call.atomic_task in registry_horizons
                else "fixed_fallback"
            )

        success, rollout_logs = execute_pi05_atomic_task_policy(
            env=env,
            client=self.client,
            atomic_task_call=call,
            verifier=self.verifier,
            log_dir=self.log_dir,
            episode_id=episode_id,
            horizon=configured_horizon,
            replan_steps=self.replan_steps,
            resize_size=self.resize_size,
            verify_interval=self.verify_interval,
            min_steps_before_verify=self.min_steps_before_verify,
            render=self.render,
            video_skip=self.video_skip,
            base_action_mode=self.base_action_mode,
            base_residual_limit=self.base_residual_limit,
            held_object_guard=self.held_object_guard,
            held_object_hold_confirmation_steps=(
                self.held_object_hold_confirmation_steps
            ),
            held_object_drop_confirmation_steps=(
                self.held_object_drop_confirmation_steps
            ),
            success_handoff_steps=self.success_handoff_steps,
        )
        rollout_logs["Configured_Horizon"] = int(configured_horizon)
        rollout_logs["Horizon_Source"] = horizon_source
        verifier_result = rollout_logs.get("Final_Verification")
        if success:
            status = "success"
        elif isinstance(verifier_result, Mapping) and verifier_result.get("status") == "failed":
            status = "failed"
        else:
            status = "uncertain"
        return {
            "module": "ATOMIC_TASK_POLICY",
            "subgoal_id": call.subgoal_id,
            "atomic_task": call.atomic_task,
            "prompt": prompt,
            "configured_horizon": int(configured_horizon),
            "horizon_source": horizon_source,
            "status": status,
            "success": bool(success),
            "rollout_logs": rollout_logs,
            "verifier_result": verifier_result,
        }
