#!/usr/bin/env python3
"""Evaluate VLM work-pose correction on paired official degraded snapshots."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robocasa.atomic_task_policy_adapter import RemoteAtomicTaskPolicyAdapter
from robocasa.atomic_task_schemas import AtomicTaskCall
from robocasa.atomic_task_verifier import RuntimeAtomicTaskVerifier
from robocasa.local_work_pose_refiner import (
    LocalWorkPoseRefiner,
    OpenAICompatibleLocalPoseVLM,
)
from robocasa.openpi_client import OpenPIWebsocketClient
from robocasa.pi05_env import POLICY_CAMERA_NAMES
from robocasa.skill_contract_registry import apply_skill_contract
from robocasa.work_pose_dataset import (
    create_dataset_environment,
    current_base_pose,
    pose_error,
    resolve_record_asset,
    restore_official_state,
    scene_context,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "robocasa/outputs_work_pose_benchmark/SteamInMicrowave/degraded"
    / "perturbations.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "robocasa/outputs_work_pose_benchmark/SteamInMicrowave/evaluation"
)
DEFAULT_REFINER_CAMERAS = (
    "robot0_topview",
    "robot0_frontview",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore the same physically materialized degraded snapshot for a "
            "baseline or multi-view VLM-refined PickObject/PlaceObject rollout."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--condition",
        choices=("baseline", "refiner"),
        required=True,
    )
    parser.add_argument("--operation", choices=("all", "pick", "place"), default="all")
    parser.add_argument(
        "--difficulty",
        choices=("all", "mild", "moderate", "severe"),
        default="all",
    )
    parser.add_argument("--sample_ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--camera_size", type=int, default=256)
    parser.add_argument(
        "--local_pose_cameras",
        nargs="+",
        default=list(DEFAULT_REFINER_CAMERAS),
    )
    parser.add_argument("--local_pose_base_url")
    parser.add_argument("--local_pose_model", default="qwen2.5vl:3b")
    parser.add_argument("--local_pose_api_key")
    parser.add_argument("--local_pose_timeout_s", type=float, default=120.0)
    parser.add_argument("--local_pose_max_decisions", type=int, default=8)
    parser.add_argument("--local_pose_action_steps", type=int, default=5)
    parser.add_argument("--local_pose_settle_steps", type=int, default=2)
    parser.add_argument("--local_pose_translation_command", type=float, default=0.20)
    parser.add_argument("--local_pose_rotation_command", type=float, default=0.25)
    parser.add_argument("--local_pose_min_confidence", type=float, default=0.55)
    parser.add_argument("--pi05_host", default="172.16.36.10")
    parser.add_argument("--pi05_port", type=int, default=8000)
    parser.add_argument("--pi05_api_key")
    parser.add_argument("--pi05_connect_timeout_s", type=float, default=15.0)
    parser.add_argument("--pi05_infer_timeout_s", type=float, default=120.0)
    parser.add_argument("--pi05_max_retries", type=int, default=1)
    parser.add_argument("--pi05_resize_size", type=int, default=224)
    parser.add_argument(
        "--pi05_horizon",
        type=int,
        default=600,
        help="Maximum steps for each PickObject or PlaceObject rollout.",
    )
    parser.add_argument("--pi05_replan_steps", type=int, default=5)
    parser.add_argument("--pi05_verify_interval", type=int, default=5)
    parser.add_argument("--pi05_min_steps_before_verify", type=int, default=5)
    parser.add_argument(
        "--pi05_base_action_mode",
        choices=("frozen", "residual", "full"),
        default="frozen",
        help="Use frozen to isolate the effect of the pre-operation work pose.",
    )
    parser.add_argument("--pi05_base_residual_limit", type=float, default=0.15)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.condition == "refiner" and not args.local_pose_base_url:
        parser.error("--local_pose_base_url is required for refiner condition")
    positive = (
        args.camera_size,
        args.pi05_port,
        args.pi05_resize_size,
        args.pi05_horizon,
        args.pi05_verify_interval,
        args.local_pose_max_decisions,
        args.local_pose_action_steps,
    )
    if any(value <= 0 for value in positive):
        parser.error("camera, policy, and local-pose integer limits must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def select_samples(
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    wanted_ids = set(args.sample_ids or [])
    selected = [
        sample
        for sample in samples
        if sample.get("valid") is True
        and (args.operation == "all" or sample["operation"] == args.operation)
        and (
            args.difficulty == "all"
            or sample["difficulty"] == args.difficulty
        )
        and (not wanted_ids or sample["sample_id"] in wanted_ids)
    ]
    return selected[: args.limit] if args.limit else selected


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [value for value in results if value.get("executed")]
    successes = [value for value in completed if value.get("success")]
    groups: dict[str, dict[str, Any]] = {}
    for result in completed:
        key = f"{result['operation']}:{result['difficulty']}"
        group = groups.setdefault(key, {"attempts": 0, "successes": 0})
        group["attempts"] += 1
        group["successes"] += int(bool(result.get("success")))
    for group in groups.values():
        group["success_rate"] = (
            group["successes"] / group["attempts"] if group["attempts"] else None
        )
    return {
        "num_results": len(results),
        "num_executed": len(completed),
        "num_success": len(successes),
        "success_rate": len(successes) / len(completed) if completed else None,
        "groups": groups,
    }


def _sample_state(manifest_path: Path, sample: Mapping[str, Any]) -> np.ndarray:
    path = manifest_path.parent / str(sample["degraded_state"])
    data = np.load(path)
    key = "state" if "state" in data else "states"
    value = np.asarray(data[key])
    return value[0] if value.ndim == 2 and len(value) == 1 else value


def run_sample(
    *,
    env: Any,
    client: OpenPIWebsocketClient,
    manifest_path: Path,
    expert_index_path: Path,
    expert_record: dict[str, Any],
    sample: dict[str, Any],
    sample_dir: Path,
    camera_names: tuple[str, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    restore_official_state(
        env,
        ep_meta_path=resolve_record_asset(
            expert_index_path, expert_record, "ep_meta"
        ),
        model_xml_gz_path=resolve_record_asset(
            expert_index_path, expert_record, "model_xml_gz"
        ),
        state=_sample_state(manifest_path, sample),
        camera_names=camera_names,
    )
    base_before = current_base_pose(env)
    call, contract_changes = apply_skill_contract(
        AtomicTaskCall.from_mapping(expert_record["atomic_task_call"]),
        scene_context(env),
    )
    write_json(sample_dir / "atomic_task_call.json", call.to_dict())

    refinement_result = None
    if args.condition == "refiner":
        decision_maker = OpenAICompatibleLocalPoseVLM(
            base_url=args.local_pose_base_url,
            model=args.local_pose_model,
            api_key=args.local_pose_api_key,
            timeout_s=args.local_pose_timeout_s,
        )
        refiner = LocalWorkPoseRefiner(
            decision_maker=decision_maker,
            log_dir=sample_dir,
            camera_names=args.local_pose_cameras,
            image_size=args.camera_size,
            max_decisions=args.local_pose_max_decisions,
            action_steps=args.local_pose_action_steps,
            settle_steps=args.local_pose_settle_steps,
            translation_command=args.local_pose_translation_command,
            rotation_command=args.local_pose_rotation_command,
            min_confidence=args.local_pose_min_confidence,
            held_object_guard=True,
        )
        refinement_result = refiner.refine(
            env=env,
            atomic_task_call=call,
            grounding_result={
                "source": "official_dataset",
                "target_id": expert_record["target_id"],
                "expert_base_pose": sample["expert_base_pose"],
            },
            episode_id=sample["sample_id"],
        )
        if refinement_result.get("failure_code") == "OBJECT_DROPPED":
            return {
                **sample,
                "executed": False,
                "success": False,
                "failure_code": "LOCAL_POSE_OBJECT_DROPPED",
                "base_pose_before": base_before,
                "base_pose_after_refinement": current_base_pose(env),
                "refinement": refinement_result,
                "contract_changes": contract_changes,
            }

    base_after_refinement = current_base_pose(env)
    adapter = RemoteAtomicTaskPolicyAdapter(
        client=client,
        verifier=RuntimeAtomicTaskVerifier(),
        log_dir=sample_dir,
        resize_size=args.pi05_resize_size,
        replan_steps=args.pi05_replan_steps,
        atomic_task_horizon=args.pi05_horizon,
        use_registry_horizons=False,
        verify_interval=args.pi05_verify_interval,
        min_steps_before_verify=args.pi05_min_steps_before_verify,
        base_action_mode=args.pi05_base_action_mode,
        base_residual_limit=args.pi05_base_residual_limit,
        held_object_guard=True,
        render=args.video,
    )
    client.reset()
    policy_result = adapter.execute(
        env=env,
        scheduler_query={"atomic_task_call": call.to_dict()},
        episode_id=sample["sample_id"],
    )
    base_after_policy = current_base_pose(env)
    return {
        **sample,
        "executed": True,
        "success": bool(policy_result["success"]),
        "failure_code": (
            policy_result.get("verifier_result", {}) or {}
        ).get("failure_code"),
        "base_pose_before": base_before,
        "base_pose_after_refinement": base_after_refinement,
        "base_pose_after_policy": base_after_policy,
        "pose_error_before": pose_error(
            base_before, sample["expert_base_pose"]
        ),
        "pose_error_after_refinement": pose_error(
            base_after_refinement, sample["expert_base_pose"]
        ),
        "refinement": refinement_result,
        "policy_result": policy_result,
        "contract_changes": contract_changes,
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    expert_index_path = Path(manifest["expert_index"]).expanduser().resolve()
    expert_index = json.loads(expert_index_path.read_text())
    samples = select_samples(list(manifest["samples"]), args)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir = args.output_root.expanduser().resolve() / args.condition / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["manifest"] = str(manifest_path)
    config["output_root"] = str(args.output_root)
    config["local_pose_api_key"] = "***" if args.local_pose_api_key else None
    config["pi05_api_key"] = "***" if args.pi05_api_key else None
    write_json(run_dir / "config.json", config)

    camera_names = tuple(
        dict.fromkeys(POLICY_CAMERA_NAMES + tuple(args.local_pose_cameras))
    )
    env = None
    client = None
    results: list[dict[str, Any]] = []
    try:
        env = create_dataset_environment(
            expert_index["dataset"],
            camera_names=camera_names,
            camera_size=args.camera_size,
        )
        client = OpenPIWebsocketClient(
            host=args.pi05_host,
            port=args.pi05_port,
            api_key=args.pi05_api_key,
            connect_timeout_s=args.pi05_connect_timeout_s,
            infer_timeout_s=args.pi05_infer_timeout_s,
            max_retries=args.pi05_max_retries,
        )
        for index, sample in enumerate(samples, start=1):
            sample_dir = run_dir / "samples" / sample["sample_id"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[WorkPose] {index}/{len(samples)} {sample['sample_id']} "
                f"condition={args.condition}"
            )
            expert_record = expert_index["records"][int(sample["record_index"])]
            try:
                result = run_sample(
                    env=env,
                    client=client,
                    manifest_path=manifest_path,
                    expert_index_path=expert_index_path,
                    expert_record=expert_record,
                    sample=sample,
                    sample_dir=sample_dir,
                    camera_names=camera_names,
                    args=args,
                )
            except Exception as exc:
                result = {
                    **sample,
                    "executed": False,
                    "success": False,
                    "failure_code": "EVALUATION_ERROR",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            write_json(sample_dir / "result.json", result)
            results.append(result)
    finally:
        if client is not None:
            client.close()
        if env is not None:
            env.close()

    summary = {
        "benchmark": "RoboCasaOfficialWorkPoseEvaluation",
        "version": 1,
        "condition": args.condition,
        "manifest": str(manifest_path),
        "camera_names": list(camera_names),
        "base_action_mode": args.pi05_base_action_mode,
        "results": results,
        **aggregate(results),
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"[WorkPose] condition={args.condition} "
        f"success={summary['num_success']}/{summary['num_executed']} "
        f"output={run_dir}"
    )
    return 0 if summary["num_executed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
