#!/usr/bin/env python3
"""Run pi0.5 directly on one complete RoboCasa task instruction.

This entrypoint deliberately bypasses task decomposition, target-aware
navigation, atomic verifiers, held-object guards, retries, and post-skill
controllers. After every environment step, the direct rollout reads RoboCasa's
official sparse task reward, which is exactly the native _check_success result,
and stops immediately on success.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

try:
    from .eval_vlm_pi05 import (
        _evaluation_only_privileged_oracle,
        _save_initial_views,
    )
    from .io_utils import make_run_dir, write_json
    from .openpi_client import OpenPIWebsocketClient
    from .pi05_env import create_pi05_env
    from .pi05_rollout import execute_pi05_manipulation_policy
except ImportError:
    from eval_vlm_pi05 import (
        _evaluation_only_privileged_oracle,
        _save_initial_views,
    )
    from io_utils import make_run_dir, write_json
    from openpi_client import OpenPIWebsocketClient
    from pi05_env import create_pi05_env
    from pi05_rollout import execute_pi05_manipulation_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one direct pi0.5 rollout with the complete RoboCasa task "
            "instruction and stop on the first official environment success."
        )
    )
    parser.add_argument("--env_name", required=True)
    parser.add_argument("--layout_id", type=int, required=True)
    parser.add_argument("--style_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--horizon", type=int, default=3600)
    parser.add_argument("--camera_size", type=int, default=256)
    parser.add_argument("--pi05_host", default="172.16.11.115")
    parser.add_argument("--pi05_port", type=int, default=8000)
    parser.add_argument("--pi05_connect_timeout_s", type=float, default=15.0)
    parser.add_argument("--pi05_infer_timeout_s", type=float, default=300.0)
    parser.add_argument("--pi05_max_retries", type=int, default=1)
    parser.add_argument("--pi05_replan_steps", type=int, default=5)
    parser.add_argument("--pi05_resize_size", type=int, default=224)
    parser.add_argument("--video_skip", type=int, default=2)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("robocasa/outputs_pi05_end_to_end"),
    )
    args = parser.parse_args()
    if args.layout_id <= 0 or args.style_id <= 0 or args.seed < 0:
        parser.error("layout/style must be positive and seed non-negative")
    positive = (
        args.horizon,
        args.camera_size,
        args.pi05_port,
        args.pi05_connect_timeout_s,
        args.pi05_infer_timeout_s,
        args.pi05_replan_steps,
        args.pi05_resize_size,
        args.video_skip,
    )
    if any(value <= 0 for value in positive):
        parser.error(
            "horizon, sizes, port, timeouts, replanning, and video values "
            "must be positive"
        )
    if args.pi05_max_retries < 0:
        parser.error("--pi05_max_retries must be non-negative")
    if not args.instruction.strip():
        parser.error("--instruction must be non-empty")
    return args


def main() -> int:
    args = parse_args()
    run_dir = make_run_dir(
        args.output_root,
        env_name=f"{args.env_name}_pi05_end_to_end",
    )
    config = vars(args).copy()
    config["output_root"] = str(args.output_root)
    config.update(
        {
            "execution_mode": "direct_pi05_end_to_end",
            "terminate_on_official_success": True,
            "success_check_frequency_steps": 1,
            "uses_vlm_planner": False,
            "uses_atomic_orchestrator": False,
            "uses_target_aware_navigation": False,
            "uses_runtime_atomic_verifier": False,
            "uses_held_object_guard": False,
            "uses_retry": False,
            "uses_post_skill_return": False,
        }
    )
    write_json(run_dir / "config.json", config)

    env: Any | None = None
    client: OpenPIWebsocketClient | None = None
    video_started = False
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
            env.start_episode_video_recording(
                run_dir / "episode_full.mp4",
                camera_name="robot0_agentview_left",
                fps=20,
                frame_stride=args.video_skip,
            )
            video_started = True

        client = OpenPIWebsocketClient(
            host=args.pi05_host,
            port=args.pi05_port,
            connect_timeout_s=args.pi05_connect_timeout_s,
            infer_timeout_s=args.pi05_infer_timeout_s,
            max_retries=args.pi05_max_retries,
        )
        write_json(
            run_dir / "pi05_server.json",
            {
                "uri": client.uri,
                "server_metadata": dict(client.server_metadata),
            },
        )
        print(f"[Pi05] Connected to {client.uri}")
        print(f"[Pi05] complete-task instruction: {args.instruction.strip()}")

        success, rollout = execute_pi05_manipulation_policy(
            env=env,
            client=client,
            log_dir=str(run_dir),
            episode_id=0,
            reset_before_rollout=False,
            horizon=args.horizon,
            replan_steps=args.pi05_replan_steps,
            resize_size=args.pi05_resize_size,
            render=False,
            video_skip=args.video_skip,
            terminate_on_success=True,
            prompt_override=args.instruction,
        )

        oracle = _evaluation_only_privileged_oracle(env)
        if video_started:
            video_metadata = env.finalize_episode_video_recording()
            video_started = False
            write_json(run_dir / "full_episode_video.json", video_metadata)

        success_bool = bool(success)
        result = {
            "status": "success" if success_bool else "failed",
            "success": success_bool,
            "failure_code": (
                None if success_bool else "END_TO_END_TASK_INCOMPLETE"
            ),
            "instruction": args.instruction.strip(),
            "rollout": rollout,
            "early_stop": {
                "enabled": True,
                "condition": "native RoboCasa sparse reward == _check_success()",
                "check_interval_steps": 1,
                "triggered": bool(rollout.get("First_Success_Step")),
                "first_success_step": rollout.get("First_Success_Step"),
                "source": rollout.get("Success_Source"),
            },
            "evaluation_only_privileged_oracle": oracle,
            "full_episode_video": video_metadata,
        }
        write_json(run_dir / "long_horizon_result.json", result)
        print(
            f"[Done] success={success_bool} steps={rollout['Horizon']} "
            f"first_success={rollout.get('First_Success_Step')} output={run_dir}"
        )
        return 0 if success_bool else 2
    except Exception as exc:
        write_json(
            run_dir / "run_error.json",
            {"error": str(exc), "traceback": traceback.format_exc()},
        )
        print(f"[Error] {type(exc).__name__}: {exc}")
        print(f"[Error] Details saved to {run_dir / 'run_error.json'}")
        return 1
    finally:
        if env is not None and video_started:
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
