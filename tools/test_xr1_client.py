#!/usr/bin/env python3
"""Real XR-1 RoboCasa365 connection/inference smoke test (no mock actions)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robocasa.pi05_env import create_pi05_env  # noqa: E402
from robocasa.utils.env_utils import convert_action  # noqa: E402
from robocasa.xr1_policy_client import (  # noqa: E402
    DEFAULT_XR1_HOST,
    DEFAULT_XR1_MODEL_PATH,
    DEFAULT_XR1_PORT,
    XR1RoboCasaPolicy,
    check_xr1_connectivity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_XR1_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_XR1_PORT)
    parser.add_argument("--model_path", default=DEFAULT_XR1_MODEL_PATH)
    parser.add_argument("--connect_timeout_s", type=float, default=5.0)
    parser.add_argument("--infer_timeout_s", type=float, default=120.0)
    parser.add_argument("--env_name", default="SteamInMicrowave")
    parser.add_argument("--layout_id", type=int, default=1)
    parser.add_argument("--style_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--camera_size", type=int, default=256)
    parser.add_argument(
        "--instruction", default="Pick up the vegetable from the sink."
    )
    parser.add_argument(
        "--execute_steps",
        type=int,
        default=0,
        help="Optionally execute this many decoded actions after the first query.",
    )
    args = parser.parse_args()
    if args.port <= 0 or args.connect_timeout_s <= 0 or args.infer_timeout_s <= 0:
        parser.error("port and timeouts must be positive")
    if args.execute_steps < 0 or args.execute_steps > 16:
        parser.error("--execute_steps must be between 0 and 16")
    return args


def main() -> int:
    args = parse_args()
    print(f"[XR1 smoke] checking TCP {args.host}:{args.port}")
    check_xr1_connectivity(args.host, args.port, args.connect_timeout_s)
    print("[XR1 smoke] socket reachable")

    env = None
    policy = None
    try:
        env = create_pi05_env(
            env_name=args.env_name,
            layout_id=args.layout_id,
            style_id=args.style_id,
            seed=args.seed,
            camera_size=args.camera_size,
        )
        observation = env.reset()
        policy = XR1RoboCasaPolicy(
            host=args.host,
            port=args.port,
            model_path=args.model_path,
            connect_timeout_s=args.connect_timeout_s,
            infer_timeout_s=args.infer_timeout_s,
        )
        policy.reset(observation)
        env.add_observation_listener(policy.observe)
        policy.set_instruction(args.instruction)
        first_action = policy.act()
        chunk_shape = (
            policy.action_chunk_lengths[-1], first_action.shape[0]
        )
        print(f"action chunk shape: {chunk_shape}")
        print(f"first action shape: {first_action.shape}")
        print(f"first action: {np.array2string(first_action, precision=4)}")

        actions = [first_action]
        for _ in range(1, args.execute_steps):
            actions.append(policy.act())
        for action in actions[: args.execute_steps]:
            observation, _, _, _ = env.step(convert_action(action))
        if args.execute_steps:
            print(f"[XR1 smoke] executed {args.execute_steps} environment step(s)")
        return 0
    finally:
        if policy is not None:
            policy.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
