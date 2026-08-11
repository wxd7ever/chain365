#!/usr/bin/env python3
"""Generate physically reached degraded poses from official expert snapshots."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from robocasa.atomic_task_schemas import AtomicTaskCall
from robocasa.held_object_guard import build_held_object_guard
from robocasa.pi05_env import LOCAL_POSE_CAMERA_NAMES, POLICY_CAMERA_NAMES
from robocasa.work_pose_dataset import (
    apply_local_pose_delta,
    current_base_pose,
    pose_error,
    resolve_lerobot_root,
    set_flattened_state,
    create_dataset_environment,
    generate_pose_perturbations,
    holding_state,
    load_record_state,
    move_base_to_pose,
    raw_task_env,
    resolve_record_asset,
    restore_official_state,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERT_INDEX = (
    REPO_ROOT
    / "robocasa/outputs_work_pose_benchmark/SteamInMicrowave/expert/index.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "robocasa/outputs_work_pose_benchmark/SteamInMicrowave/degraded"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample target-frame pose errors, move the real simulated base from "
            "each official state, validate held-object continuity, and save the "
            "resulting degraded MuJoCo states."
        )
    )
    parser.add_argument("--expert_index", type=Path, default=DEFAULT_EXPERT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--difficulties",
        nargs="+",
        choices=("mild", "moderate", "severe"),
        default=("mild", "moderate", "severe"),
    )
    parser.add_argument("--samples_per_stage", type=int, default=3)
    parser.add_argument(
        "--max_candidate_attempts",
        type=int,
        default=5,
        help=(
            "Maximum random pose candidates tried for each requested physical "
            "sample. Unreachable or dropped-object candidates are rejected."
        ),
    )
    parser.add_argument(
        "--operation", choices=("all", "pick", "place"), default="all"
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--camera_size", type=int, default=256)
    parser.add_argument("--max_move_steps", type=int, default=240)
    parser.add_argument("--translation_tolerance_m", type=float, default=0.025)
    parser.add_argument("--yaw_tolerance_deg", type=float, default=3.0)
    parser.add_argument(
        "--specs_only",
        action="store_true",
        help="Write deterministic pose targets without starting MuJoCo.",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if (
        args.samples_per_stage <= 0
        or args.max_candidate_attempts <= 0
        or args.camera_size <= 0
    ):
        parser.error(
            "samples_per_stage, max_candidate_attempts, and camera_size must be positive"
        )
    if args.max_move_steps <= 0:
        parser.error("--max_move_steps must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def _materialize(
    *,
    env: Any,
    expert_index_path: Path,
    expert_record: dict[str, Any],
    sample: dict[str, Any],
    output: Path,
    camera_names: tuple[str, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    state = load_record_state(expert_index_path, expert_record)
    restore_official_state(
        env,
        ep_meta_path=resolve_record_asset(
            expert_index_path, expert_record, "ep_meta"
        ),
        model_xml_gz_path=resolve_record_asset(
            expert_index_path, expert_record, "model_xml_gz"
        ),
        state=state,
        camera_names=camera_names,
    )
    expected_holding = bool(expert_record["expected_holding_at_entry"])
    holding_before = holding_state(env, expert_record["object_id"])
    resolved_entry_frame = int(expert_record["frame_index"])
    if expected_holding and holding_before is not True:
        dataset = resolve_lerobot_root(expert_record["dataset"])
        episode_states = np.load(
            dataset
            / "extras"
            / f"episode_{int(expert_record['episode_index']):06d}"
            / "states.npz"
        )["states"]
        segment = expert_record["segment"]
        first_held_frame = None
        consecutive = 0
        for frame in range(
            int(segment["start_frame"]),
            int(segment["end_frame"]) + 1,
        ):
            set_flattened_state(env, episode_states[frame])
            if holding_state(env, expert_record["object_id"]) is True:
                consecutive += 1
                if consecutive == 1:
                    first_held_frame = frame
                if consecutive >= 2:
                    resolved_entry_frame = int(first_held_frame)
                    state = episode_states[resolved_entry_frame]
                    set_flattened_state(env, state)
                    holding_before = True
                    break
            else:
                consecutive = 0
                first_held_frame = None
        if holding_before is not True:
            return {
                **sample,
                "valid": False,
                "failure_code": "PLACE_SEGMENT_HAS_NO_STABLE_HOLDING_STATE",
                "holding_before": holding_before,
            }

    resolved_expert_pose = current_base_pose(env)
    delta = sample["perturbation"]
    target_degraded_pose = apply_local_pose_delta(
        resolved_expert_pose,
        forward_m=float(delta["forward_m"]),
        left_m=float(delta["left_m"]),
        yaw_rad=float(delta["yaw_rad"]),
    )
    sample = {
        **sample,
        "resolved_entry_frame": resolved_entry_frame,
        "expert_base_pose": resolved_expert_pose,
        "target_degraded_base_pose": target_degraded_pose,
        "initial_pose_error": pose_error(
            target_degraded_pose, resolved_expert_pose
        ),
    }

    guard = None
    if expected_holding:
        guard = build_held_object_guard(
            env=env,
            atomic_task_call=AtomicTaskCall.from_mapping(
                expert_record["atomic_task_call"]
            ),
            enabled=True,
        )
    movement = move_base_to_pose(
        env,
        target_degraded_pose,
        guard=guard,
        max_steps=args.max_move_steps,
        translation_tolerance_m=args.translation_tolerance_m,
        yaw_tolerance_deg=args.yaw_tolerance_deg,
    )
    holding_after = holding_state(env, expert_record["object_id"])
    valid = bool(
        movement["success"]
        and (not expected_holding or holding_after is True)
    )
    sample_dir = output / "samples" / sample["sample_id"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    degraded_state_path = sample_dir / "state.npz"
    np.savez_compressed(
        degraded_state_path,
        state=np.asarray(raw_task_env(env).sim.get_state().flatten()),
    )
    result = {
        **sample,
        "valid": valid,
        "failure_code": (
            None
            if valid
            else (
                "HELD_OBJECT_LOST"
                if expected_holding and holding_after is not True
                else "BASE_MOVE_NOT_CONVERGED"
            )
        ),
        "holding_before": holding_before,
        "holding_after": holding_after,
        "base_movement": movement,
        "degraded_state": str(degraded_state_path.relative_to(output)),
    }
    write_json(sample_dir / "metadata.json", result)
    return result


def main() -> int:
    args = parse_args()
    expert_index_path = args.expert_index.expanduser().resolve()
    expert_index = json.loads(expert_index_path.read_text())
    records = list(expert_index["records"])
    candidate_multiplier = 1 if args.specs_only else args.max_candidate_attempts
    samples = generate_pose_perturbations(
        records,
        difficulties=args.difficulties,
        samples_per_stage=args.samples_per_stage * candidate_multiplier,
        seed=args.seed,
    )
    output = args.output.expanduser().resolve()
    if args.operation != "all":
        samples = [s for s in samples if s["operation"] == args.operation]
    if args.limit is not None and args.specs_only:
        samples = samples[: args.limit]
    output.mkdir(parents=True, exist_ok=True)
    camera_names = tuple(
        dict.fromkeys(POLICY_CAMERA_NAMES + LOCAL_POSE_CAMERA_NAMES)
    )
    env = None
    results: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_by_group: dict[tuple[int, str], int] = {}
    selected_record_indices = {
        int(sample["record_index"]) for sample in samples
    }
    requested_samples = (
        args.limit
        if args.limit is not None
        else len(selected_record_indices)
        * len(args.difficulties)
        * args.samples_per_stage
    )
    try:
        if not args.specs_only:
            env = create_dataset_environment(
                expert_index["dataset"],
                camera_names=camera_names,
                camera_size=args.camera_size,
            )
        for index, sample in enumerate(samples, start=1):
            group = (int(sample["record_index"]), str(sample["difficulty"]))
            if not args.specs_only:
                if len(results) >= requested_samples:
                    break
                if (
                    args.limit is None
                    and accepted_by_group.get(group, 0)
                    >= args.samples_per_stage
                ):
                    continue
            print(
                f"[WorkPose] {index}/{len(samples)} {sample['sample_id']} "
                f"operation={sample['operation']}"
            )
            if args.specs_only:
                results.append({**sample, "valid": None, "materialized": False})
                continue
            record = records[int(sample["record_index"])]
            try:
                value = _materialize(
                    env=env,
                    expert_index_path=expert_index_path,
                    expert_record=record,
                    sample=sample,
                    output=output,
                    camera_names=camera_names,
                    args=args,
                )
            except Exception as exc:
                value = {
                    **sample,
                    "valid": False,
                    "failure_code": "MATERIALIZATION_ERROR",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            if value.get("valid") is True or args.specs_only:
                results.append(value)
                if value.get("valid") is True:
                    accepted_by_group[group] = accepted_by_group.get(group, 0) + 1
            else:
                rejected.append(value)
    finally:
        if env is not None:
            env.close()

    manifest = {
        "benchmark": "RoboCasaOfficialWorkPosePerturbations",
        "version": 1,
        "expert_index": str(expert_index_path),
        "dataset": expert_index["dataset"],
        "seed": args.seed,
        "difficulties": list(args.difficulties),
        "samples_per_stage": args.samples_per_stage,
        "max_candidate_attempts": args.max_candidate_attempts,
        "materialized": not args.specs_only,
        "requested_samples": requested_samples,
        "num_samples": len(results),
        "num_valid": sum(value.get("valid") is True for value in results),
        "num_rejected": len(rejected),
        "samples": results,
        "rejected": rejected,
    }
    write_json(output / "perturbations.json", manifest)
    print(
        f"[WorkPose] samples={len(results)} valid={manifest['num_valid']} "
        f"manifest={output / 'perturbations.json'}"
    )
    return (
        0
        if args.specs_only or manifest["num_valid"] >= requested_samples
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
