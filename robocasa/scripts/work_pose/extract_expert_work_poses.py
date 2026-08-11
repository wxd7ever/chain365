#!/usr/bin/env python3
"""Extract official SteamInMicrowave Pick/Place entry snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robocasa.work_pose_dataset import extract_expert_stages, resolve_lerobot_root


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = (
    REPO_ROOT
    / "datasets/v1.0/pretrain/composite/SteamInMicrowave/20250714/lerobot"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "robocasa/outputs_work_pose_benchmark/SteamInMicrowave/expert"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use official human subtask annotations to extract exact pre-Pick "
            "and pre-Place MuJoCo snapshots and expert base poses."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--episode_start", type=int, default=0)
    parser.add_argument(
        "--episode_count",
        type=int,
        default=20,
        help="Number of episodes when --episodes is omitted; use 0 for all.",
    )
    parser.add_argument("--stability_window", type=int, default=10)
    args = parser.parse_args()
    if args.episode_start < 0 or args.episode_count < 0:
        parser.error("episode start/count must be non-negative")
    if args.stability_window <= 0:
        parser.error("--stability_window must be positive")
    return args


def selected_episodes(args: argparse.Namespace, dataset: Path) -> list[int]:
    info = json.loads((dataset / "meta" / "info.json").read_text())
    total = int(info["total_episodes"])
    if args.episodes:
        values = list(dict.fromkeys(args.episodes))
    else:
        stop = total if args.episode_count == 0 else min(
            total, args.episode_start + args.episode_count
        )
        values = list(range(args.episode_start, stop))
    invalid = [value for value in values if value < 0 or value >= total]
    if invalid:
        raise ValueError(f"episode indices outside [0, {total}): {invalid}")
    return values


def main() -> int:
    args = parse_args()
    dataset = resolve_lerobot_root(args.dataset)
    episodes = selected_episodes(args, dataset)
    result = extract_expert_stages(
        dataset=dataset,
        output_dir=args.output,
        episode_indices=episodes,
        stability_window=args.stability_window,
    )
    counts: dict[str, int] = {}
    for record in result["records"]:
        key = f"{record['operation']}_{record['object_id']}"
        counts[key] = counts.get(key, 0) + 1
    print(f"[WorkPose] dataset={dataset}")
    print(f"[WorkPose] episodes={len(episodes)} records={result['num_records']}")
    print(f"[WorkPose] stages={counts}")
    print(f"[WorkPose] index={Path(args.output).resolve() / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
