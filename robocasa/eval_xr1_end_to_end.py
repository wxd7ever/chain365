#!/usr/bin/env python3
"""Evaluate XR-1 on one complete RoboCasa task without task decomposition.

This entrypoint is intentionally separate from ``eval_vlm_pi05.py``.  It sends
the complete task instruction to XR-1 exactly once and then executes a single
continuous policy rollout.  It does not instantiate the VLM planner, atomic
orchestrator, navigation pipeline, verifier, held-object guard, retry logic, or
post-skill return controller.  Privileged RoboCasa state is read only once,
after the rollout, to report the official task success predicate.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .eval_vlm_pi05 import (
        _evaluation_only_privileged_oracle,
        _save_initial_views,
        create_pi05_env,
    )
    from .io_utils import make_run_dir, write_json
    from .pi05_rollout import _step_env
    from .xr1_policy_client import (
        DEFAULT_XR1_HOST,
        DEFAULT_XR1_MODEL_PATH,
        DEFAULT_XR1_PORT,
        XR1RoboCasaPolicy,
        check_xr1_connectivity,
    )
except ImportError:  # Direct ``python robocasa/eval_xr1_end_to_end.py`` use.
    from eval_vlm_pi05 import (  # type: ignore
        _evaluation_only_privileged_oracle,
        _save_initial_views,
        create_pi05_env,
    )
    from io_utils import make_run_dir, write_json  # type: ignore
    from pi05_rollout import _step_env  # type: ignore
    from xr1_policy_client import (  # type: ignore
        DEFAULT_XR1_HOST,
        DEFAULT_XR1_MODEL_PATH,
        DEFAULT_XR1_PORT,
        XR1RoboCasaPolicy,
        check_xr1_connectivity,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one direct XR-1 rollout using the complete RoboCasa task "
            "instruction, without planning or atomic-task orchestration."
        )
    )
    parser.add_argument("--env_name", required=True)
    parser.add_argument("--layout_id", type=int, required=True)
    parser.add_argument("--style_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--horizon", type=int, default=3600)
    parser.add_argument("--camera_size", type=int, default=256)
    parser.add_argument("--video_skip", type=int, default=2)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("robocasa/outputs_xr1_end_to_end"),
    )
    parser.add_argument("--xr1_host", default=DEFAULT_XR1_HOST)
    parser.add_argument("--xr1_port", type=int, default=DEFAULT_XR1_PORT)
    parser.add_argument("--xr1_model_path", default=DEFAULT_XR1_MODEL_PATH)
    parser.add_argument("--xr1_connect_timeout_s", type=float, default=15.0)
    parser.add_argument("--xr1_infer_timeout_s", type=float, default=300.0)
    parser.add_argument("--xr1_obs_history", type=int, default=4)
    parser.add_argument("--xr1_obs_interval", type=int, default=2)
    parser.add_argument("--xr1_replan_steps", type=int, default=16)
    parser.add_argument("--xr1_crop_ratio", type=float, default=0.95)
    args = parser.parse_args()
    if args.layout_id < 0 or args.seed < 0:
        parser.error("--layout_id and --seed must be non-negative")
    if args.horizon <= 0 or args.camera_size <= 0 or args.video_skip <= 0:
        parser.error("--horizon, --camera_size, and --video_skip must be positive")
    if not args.instruction.strip():
        parser.error("--instruction must be non-empty")
    return args


def run_direct_rollout(
    *,
    env: Any,
    client: Any,
    instruction: str,
    horizon: int,
) -> dict[str, Any]:
    """Execute one uninterrupted XR-1 rollout and return non-privileged logs."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be non-empty")

    client.set_instruction(instruction.strip())
    rewards: list[float] = []
    action_min = np.inf
    action_max = -np.inf
    completed_steps = 0
    environment_done = False
    final_info: Mapping[str, Any] = {}

    for _ in range(horizon):
        action = np.asarray(client.act(), dtype=np.float32).reshape(-1)
        if action.shape != (12,):
            raise RuntimeError(
                f"XR-1 action must resolve to 12 values, got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise RuntimeError("XR-1 returned an action containing NaN or Inf")
        action_min = min(action_min, float(action.min()))
        action_max = max(action_max, float(action.max()))
        _, reward, environment_done, final_info = _step_env(
            env,
            action,
            convert_robocasa365_action=True,
        )
        rewards.append(float(reward))
        completed_steps += 1
        if environment_done:
            break

    return {
        "completed_steps": int(completed_steps),
        "requested_horizon": int(horizon),
        "environment_done": bool(environment_done),
        "reward_sum": float(sum(rewards)),
        "reward_max": float(max(rewards)) if rewards else None,
        "last_reward": float(rewards[-1]) if rewards else None,
        "action_min": float(action_min) if rewards else None,
        "action_max": float(action_max) if rewards else None,
        "final_info": dict(final_info),
    }


def main() -> int:
    args = parse_args()
    run_dir = make_run_dir(
        args.output_root,
        env_name=f"{args.env_name}_xr1_end_to_end",
    )
    config = vars(args).copy()
    config["output_root"] = str(args.output_root)
    config["execution_mode"] = "direct_xr1_end_to_end"
    config["uses_vlm_planner"] = False
    config["uses_atomic_orchestrator"] = False
    config["uses_target_aware_navigation"] = False
    config["uses_runtime_verifier"] = False
    config["uses_retry"] = False
    config["uses_post_skill_return"] = False
    write_json(run_dir / "config.json", config)

    env = None
    client = None
    video_metadata: dict[str, Any] | None = None
    try:
        print(
            f"[Env] Creating {args.env_name} layout={args.layout_id} "
            f"style={args.style_id} seed={args.seed}"
        )
        env = create_pi05_env(
            env_name=args.env_name,
            layout_id=args.layout_id,
            style_id=args.style_id,
            seed=args.seed,
            camera_size=args.camera_size,
        )
        observation = env.reset()
        _save_initial_views(run_dir, observation)
        if not args.no_video:
            video_metadata = env.start_episode_video_recording(
                run_dir / "episode_full.mp4",
                camera_name="robot0_agentview_left",
                fps=20,
                frame_stride=args.video_skip,
            )

        check_xr1_connectivity(
            args.xr1_host,
            args.xr1_port,
            args.xr1_connect_timeout_s,
        )
        client = XR1RoboCasaPolicy(
            host=args.xr1_host,
            port=args.xr1_port,
            model_path=args.xr1_model_path,
            connect_timeout_s=args.xr1_connect_timeout_s,
            infer_timeout_s=args.xr1_infer_timeout_s,
            obs_history=args.xr1_obs_history,
            obs_interval=args.xr1_obs_interval,
            replan_steps=args.xr1_replan_steps,
            crop_ratio=args.xr1_crop_ratio,
        )
        add_listener = getattr(env, "add_observation_listener", None)
        if callable(add_listener):
            add_listener(client.observe)
            client.environment_observer_attached = True
        client.reset(observation)

        rollout = run_direct_rollout(
            env=env,
            client=client,
            instruction=args.instruction,
            horizon=args.horizon,
        )
        # This is intentionally the first privileged task-state query.  It is
        # evaluation-only and cannot affect any policy action or rollout stop.
        oracle = _evaluation_only_privileged_oracle(env)
        if not args.no_video:
            video_metadata = env.finalize_episode_video_recording()
            write_json(run_dir / "full_episode_video.json", video_metadata)
        success = bool(oracle.get("available") and oracle.get("success"))
        result = {
            "status": "success" if success else "failed",
            "success": success,
            "failure_code": None if success else "END_TO_END_TASK_INCOMPLETE",
            "instruction": args.instruction,
            "rollout": rollout,
            "policy": client.diagnostics(),
            "evaluation_only_privileged_oracle": oracle,
            "full_episode_video": video_metadata,
        }
        write_json(run_dir / "long_horizon_result.json", result)
        print(
            f"[Done] success={success} steps={rollout['completed_steps']} "
            f"output={run_dir}"
        )
        return 0 if success else 2
    except Exception as exc:
        write_json(
            run_dir / "run_error.json",
            {"error": str(exc), "traceback": traceback.format_exc()},
        )
        print(f"[Error] {type(exc).__name__}: {exc}")
        print(f"[Error] Details saved to {run_dir / 'run_error.json'}")
        return 1
    finally:
        if env is not None and not args.no_video:
            try:
                env.finalize_episode_video_recording()
            except Exception:
                pass
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
