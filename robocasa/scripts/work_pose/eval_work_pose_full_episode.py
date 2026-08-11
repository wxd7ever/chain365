#!/usr/bin/env python3
"""Run complete SteamInMicrowave episodes from paired work-pose samples."""

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
from robocasa.held_object_guard import build_held_object_guard
from robocasa.local_work_pose_refiner import (
    LocalWorkPoseRefiner,
    OpenAICompatibleLocalPoseVLM,
)
from robocasa.openpi_client import OpenPIWebsocketClient
from robocasa.pi05_env import POLICY_CAMERA_NAMES
from robocasa.skill_contract_registry import apply_skill_contract
from robocasa.work_pose_dataset import (
    annotation_segments,
    base_pose_from_state,
    create_dataset_environment,
    current_base_pose,
    holding_state,
    load_episode_dataframe,
    load_task_labels,
    move_base_to_pose,
    pose_error,
    raw_task_env,
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
    / "robocasa/outputs_work_pose_benchmark/SteamInMicrowave/full_episode"
)
DEFAULT_REFINER_CAMERAS = (
    "robot0_topview",
    "robot0_frontview",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
REQUIRED_OPERATION_STAGES = (0, 1, 2, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Pick vegetable, Place vegetable, Pick bowl, dataset-guided "
            "navigation, Place bowl, Close microwave, and Turn on microwave "
            "continuously without restoring state between subgoals."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--condition", choices=("baseline", "refiner"), required=True
    )
    parser.add_argument(
        "--difficulty",
        choices=("mild", "moderate", "severe"),
        default="mild",
    )
    parser.add_argument("--episode_start", type=int, default=0)
    parser.add_argument("--episode_count", type=int, default=5)
    parser.add_argument(
        "--sample_rank",
        type=int,
        default=0,
        help="Choose the Nth valid perturbation for every operation stage.",
    )
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
    parser.add_argument("--pi05_horizon", type=int, default=600)
    parser.add_argument("--pi05_replan_steps", type=int, default=5)
    parser.add_argument("--pi05_verify_interval", type=int, default=5)
    parser.add_argument("--pi05_min_steps_before_verify", type=int, default=5)
    parser.add_argument(
        "--pi05_base_action_mode",
        choices=("frozen", "residual", "full"),
        default="frozen",
    )
    parser.add_argument("--pi05_base_residual_limit", type=float, default=0.15)
    parser.add_argument("--max_station_move_steps", type=int, default=360)
    parser.add_argument("--station_translation_tolerance_m", type=float, default=0.03)
    parser.add_argument("--station_yaw_tolerance_deg", type=float, default=4.0)
    parser.add_argument("--held_max_translation_command", type=float, default=0.20)
    parser.add_argument("--held_max_rotation_command", type=float, default=0.15)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.condition == "refiner" and not args.local_pose_base_url:
        parser.error("--local_pose_base_url is required for refiner condition")
    positive = (
        args.episode_count,
        args.camera_size,
        args.pi05_port,
        args.pi05_resize_size,
        args.pi05_horizon,
        args.pi05_replan_steps,
        args.pi05_verify_interval,
        args.local_pose_max_decisions,
        args.local_pose_action_steps,
        args.max_station_move_steps,
        args.held_max_translation_command,
        args.held_max_rotation_command,
    )
    if any(value <= 0 for value in positive):
        parser.error("episode, policy, camera, and movement limits must be positive")
    if args.episode_start < 0 or args.sample_rank < 0:
        parser.error("episode_start and sample_rank must be non-negative")
    return args


def select_episode_workflows(
    *,
    samples: list[dict[str, Any]],
    records: list[dict[str, Any]],
    difficulty: str,
    sample_rank: int,
    episode_start: int,
    episode_count: int,
) -> list[dict[str, Any]]:
    """Select complete episode groups instead of a flat number of samples."""

    grouped: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for sample in samples:
        if sample.get("valid") is not True or sample.get("difficulty") != difficulty:
            continue
        record_index = int(sample["record_index"])
        record = records[record_index]
        episode_index = int(record["episode_index"])
        stage_index = int(record["segment"]["subtask_idx"])
        if stage_index not in REQUIRED_OPERATION_STAGES:
            continue
        grouped.setdefault(episode_index, {}).setdefault(stage_index, []).append(
            {
                "sample": sample,
                "record": record,
                "record_index": record_index,
            }
        )

    complete: list[dict[str, Any]] = []
    for episode_index in sorted(grouped):
        if episode_index < episode_start:
            continue
        stages = grouped[episode_index]
        if any(stage not in stages for stage in REQUIRED_OPERATION_STAGES):
            continue
        chosen: dict[int, dict[str, Any]] = {}
        for stage in REQUIRED_OPERATION_STAGES:
            candidates = sorted(
                stages[stage], key=lambda value: value["sample"]["sample_id"]
            )
            if sample_rank >= len(candidates):
                chosen = {}
                break
            chosen[stage] = candidates[sample_rank]
        if chosen:
            complete.append(
                {"episode_index": episode_index, "stages": chosen}
            )
        if len(complete) >= episode_count:
            break
    if len(complete) < episode_count:
        raise ValueError(
            f"requested {episode_count} complete episodes from {episode_start}, "
            f"but manifest contains only {len(complete)} for difficulty "
            f"{difficulty!r} and sample_rank={sample_rank}"
        )
    return complete


def _sample_state(manifest_path: Path, sample: Mapping[str, Any]) -> np.ndarray:
    path = manifest_path.parent / str(sample["degraded_state"])
    data = np.load(path)
    key = "state" if "state" in data else "states"
    value = np.asarray(data[key])
    return value[0] if value.ndim == 2 and len(value) == 1 else value


def official_stage_poses(
    dataset: str | Path,
    episode_index: int,
    task_labels: Mapping[int, str],
) -> dict[int, dict[str, Any]]:
    dataframe = load_episode_dataframe(dataset, episode_index)
    observations = np.stack(dataframe["observation.state"].to_numpy())
    return {
        segment.subtask_idx: base_pose_from_state(
            observations[segment.start_frame]
        )
        for segment in annotation_segments(dataframe, task_labels)
        if segment.stage not in {"done"}
    }


def target_degraded_pose(sample: Mapping[str, Any]) -> dict[str, Any]:
    movement = sample.get("base_movement")
    if isinstance(movement, Mapping) and isinstance(
        movement.get("final_pose"), Mapping
    ):
        return dict(movement["final_pose"])
    return dict(sample["target_degraded_base_pose"])


def microwave_alias(context: Mapping[str, Any]) -> str:
    fixtures = context.get("fixtures", [])
    candidates: list[tuple[int, str]] = []
    for fixture in fixtures:
        if not isinstance(fixture, Mapping):
            continue
        alias = str(fixture.get("alias", ""))
        searchable = " ".join(
            str(fixture.get(key, ""))
            for key in ("alias", "name", "natural_name")
        ).lower()
        if alias and "microwave" in searchable:
            score = 0 if alias.lower() == "microwave" else 1
            candidates.append((score, alias))
    if not candidates:
        raise ValueError("could not resolve a microwave fixture alias")
    return sorted(candidates)[0][1]


def fixture_call(stage_index: int, fixture_id: str) -> AtomicTaskCall:
    if stage_index == 5:
        return AtomicTaskCall.from_mapping(
            {
                "subgoal_id": "dataset_5_close_microwave",
                "atomic_task": "CloseMicrowave",
                "arguments": {
                    "fixture_id": fixture_id,
                    "fixture_name": "microwave",
                },
                "termination_condition": {
                    "predicate": "closed",
                    "subject": fixture_id,
                    "desired_value": True,
                },
                "policy_prompt": "Close the microwave door.",
                "metadata": {"source": "official_dataset_workflow"},
            }
        )
    if stage_index == 6:
        return AtomicTaskCall.from_mapping(
            {
                "subgoal_id": "dataset_6_turn_on_microwave",
                "atomic_task": "TurnOnMicrowave",
                "arguments": {
                    "fixture_id": fixture_id,
                    "fixture_name": "microwave",
                },
                "termination_condition": {
                    "predicate": "powered",
                    "subject": fixture_id,
                    "desired_value": True,
                },
                "policy_prompt": "Press the start button on the microwave.",
                "metadata": {"source": "official_dataset_workflow"},
            }
        )
    raise ValueError(f"unsupported fixture stage {stage_index}")


def move_to_stage(
    *,
    env: Any,
    target_pose: Mapping[str, Any],
    upcoming_call: AtomicTaskCall | None,
    expected_holding_object: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    holding_before = (
        holding_state(env, expected_holding_object)
        if expected_holding_object
        else None
    )
    if expected_holding_object and holding_before is not True:
        return {
            "success": False,
            "failure_code": "EXPECTED_HELD_OBJECT_MISSING_BEFORE_MOVE",
            "object_id": expected_holding_object,
            "holding_before": holding_before,
        }
    guard = None
    if expected_holding_object and upcoming_call is not None:
        guard = build_held_object_guard(
            env=env,
            atomic_task_call=upcoming_call,
            enabled=True,
        )
    result = move_base_to_pose(
        env,
        target_pose,
        guard=guard,
        max_steps=args.max_station_move_steps,
        translation_tolerance_m=args.station_translation_tolerance_m,
        yaw_tolerance_deg=args.station_yaw_tolerance_deg,
        max_translation_command=(
            args.held_max_translation_command if expected_holding_object else 1.0
        ),
        max_rotation_command=(
            args.held_max_rotation_command if expected_holding_object else 0.50
        ),
    )
    holding_after = (
        holding_state(env, expected_holding_object)
        if expected_holding_object
        else None
    )
    success = bool(
        result["success"]
        and (
            expected_holding_object is None
            or holding_after is True
        )
    )
    return {
        **result,
        "success": success,
        "failure_code": (
            None
            if success
            else (
                "HELD_OBJECT_LOST_DURING_DATASET_NAVIGATION"
                if expected_holding_object and holding_after is not True
                else "DATASET_STATION_MOVE_NOT_CONVERGED"
            )
        ),
        "holding_before": holding_before,
        "holding_after": holding_after,
    }


def execute_current_skill(
    *,
    env: Any,
    client: OpenPIWebsocketClient,
    call: AtomicTaskCall,
    target_id: str,
    expert_pose: Mapping[str, Any],
    step_dir: Path,
    step_id: str,
    decision_maker: OpenAICompatibleLocalPoseVLM | None,
    allow_refinement: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    call, contract_changes = apply_skill_contract(call, scene_context(env))
    write_json(step_dir / "atomic_task_call.json", call.to_dict())
    base_before = current_base_pose(env)
    refinement = None
    if allow_refinement and decision_maker is not None:
        refiner = LocalWorkPoseRefiner(
            decision_maker=decision_maker,
            log_dir=step_dir,
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
        refinement = refiner.refine(
            env=env,
            atomic_task_call=call,
            grounding_result={
                "source": "official_dataset",
                "target_id": target_id,
                "expert_base_pose": dict(expert_pose),
            },
            episode_id=step_id,
        )
        if refinement.get("failure_code") == "OBJECT_DROPPED":
            return {
                "executed": False,
                "success": False,
                "status": "failed",
                "failure_code": "LOCAL_POSE_OBJECT_DROPPED",
                "base_pose_before": base_before,
                "base_pose_after_refinement": current_base_pose(env),
                "refinement": refinement,
                "contract_changes": contract_changes,
            }

    base_after_refinement = current_base_pose(env)
    adapter = RemoteAtomicTaskPolicyAdapter(
        client=client,
        verifier=RuntimeAtomicTaskVerifier(),
        log_dir=step_dir,
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
        episode_id=step_id,
    )
    return {
        "executed": True,
        "success": bool(policy_result["success"]),
        "status": policy_result["status"],
        "failure_code": (
            policy_result.get("verifier_result", {}) or {}
        ).get("failure_code"),
        "base_pose_before": base_before,
        "base_pose_after_refinement": base_after_refinement,
        "base_pose_after_policy": current_base_pose(env),
        "pose_error_before": pose_error(base_before, expert_pose),
        "pose_error_after_refinement": pose_error(
            base_after_refinement, expert_pose
        ),
        "refinement": refinement,
        "policy_result": policy_result,
        "contract_changes": contract_changes,
    }


def run_episode(
    *,
    env: Any,
    client: OpenPIWebsocketClient,
    workflow: Mapping[str, Any],
    manifest_path: Path,
    expert_index_path: Path,
    dataset: str | Path,
    task_labels: Mapping[int, str],
    episode_dir: Path,
    camera_names: tuple[str, ...],
    decision_maker: OpenAICompatibleLocalPoseVLM | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    episode_index = int(workflow["episode_index"])
    stages = workflow["stages"]
    first = stages[0]
    restore_official_state(
        env,
        ep_meta_path=resolve_record_asset(
            expert_index_path, first["record"], "ep_meta"
        ),
        model_xml_gz_path=resolve_record_asset(
            expert_index_path, first["record"], "model_xml_gz"
        ),
        state=_sample_state(manifest_path, first["sample"]),
        camera_names=camera_names,
    )
    poses = official_stage_poses(dataset, episode_index, task_labels)
    operation_results: list[dict[str, Any]] = []
    navigation_results: list[dict[str, Any]] = []
    stopped_at = None

    for stage_index in REQUIRED_OPERATION_STAGES:
        entry = stages[stage_index]
        sample = entry["sample"]
        record = entry["record"]
        call = AtomicTaskCall.from_mapping(record["atomic_task_call"])
        step_dir = episode_dir / f"stage_{stage_index:02d}_{record['operation']}"
        step_dir.mkdir(parents=True, exist_ok=True)
        if stage_index != 0:
            held_object = record["object_id"] if record["operation"] == "place" else None
            navigation = move_to_stage(
                env=env,
                target_pose=target_degraded_pose(sample),
                upcoming_call=call,
                expected_holding_object=held_object,
                args=args,
            )
            navigation["target_stage"] = stage_index
            navigation["sample_id"] = sample["sample_id"]
            write_json(step_dir / "dataset_navigation.json", navigation)
            navigation_results.append(navigation)
            if not navigation["success"]:
                stopped_at = f"navigation_to_stage_{stage_index}"
                break
        result = execute_current_skill(
            env=env,
            client=client,
            call=call,
            target_id=record["target_id"],
            expert_pose=sample["expert_base_pose"],
            step_dir=step_dir,
            step_id=f"ep{episode_index:06d}_s{stage_index:02d}",
            decision_maker=decision_maker,
            allow_refinement=True,
            args=args,
        )
        result.update(
            {
                "stage_index": stage_index,
                "sample_id": sample["sample_id"],
                "atomic_task": call.atomic_task,
                "operation": record["operation"],
            }
        )
        write_json(step_dir / "result.json", result)
        operation_results.append(result)
        if not result["success"]:
            stopped_at = f"operation_stage_{stage_index}"
            break

    if stopped_at is None:
        context = scene_context(env)
        fixture_id = microwave_alias(context)
        for stage_index in (5, 6):
            step_dir = episode_dir / (
                "stage_05_close_microwave"
                if stage_index == 5
                else "stage_06_turn_on_microwave"
            )
            step_dir.mkdir(parents=True, exist_ok=True)
            navigation = move_to_stage(
                env=env,
                target_pose=poses[stage_index],
                upcoming_call=None,
                expected_holding_object=None,
                args=args,
            )
            navigation["target_stage"] = stage_index
            navigation["source"] = "official_execute_stage_pose"
            write_json(step_dir / "dataset_navigation.json", navigation)
            navigation_results.append(navigation)
            if not navigation["success"]:
                stopped_at = f"navigation_to_stage_{stage_index}"
                break
            call = fixture_call(stage_index, fixture_id)
            result = execute_current_skill(
                env=env,
                client=client,
                call=call,
                target_id=fixture_id,
                expert_pose=poses[stage_index],
                step_dir=step_dir,
                step_id=f"ep{episode_index:06d}_s{stage_index:02d}",
                decision_maker=None,
                allow_refinement=False,
                args=args,
            )
            result.update(
                {
                    "stage_index": stage_index,
                    "atomic_task": call.atomic_task,
                    "operation": "execute",
                }
            )
            write_json(step_dir / "result.json", result)
            operation_results.append(result)
            if not result["success"]:
                stopped_at = f"operation_stage_{stage_index}"
                break

    expected_operations = (0, 1, 2, 4, 5, 6)
    completed_stages = tuple(
        int(result["stage_index"]) for result in operation_results
    )
    policy_success = bool(
        completed_stages == expected_operations
        and all(result["success"] for result in operation_results)
        and all(result["success"] for result in navigation_results)
    )
    try:
        environment_success = bool(raw_task_env(env)._check_success())
    except (AttributeError, KeyError, TypeError):
        environment_success = None
    success = bool(
        policy_success
        and environment_success is not False
    )
    return {
        "episode_index": episode_index,
        "condition": args.condition,
        "difficulty": args.difficulty,
        "success": success,
        "policy_sequence_success": policy_success,
        "environment_success": environment_success,
        "stopped_at": stopped_at,
        "completed_stages": list(completed_stages),
        "operation_results": operation_results,
        "navigation_results": navigation_results,
        "final_base_pose": current_base_pose(env),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(result.get("success")) for result in results)
    stage_attempts: dict[str, dict[str, int]] = {}
    for episode in results:
        for result in episode.get("operation_results", []):
            key = f"stage_{int(result['stage_index']):02d}_{result['atomic_task']}"
            value = stage_attempts.setdefault(
                key, {"attempts": 0, "successes": 0}
            )
            value["attempts"] += 1
            value["successes"] += int(bool(result.get("success")))
    for value in stage_attempts.values():
        value["success_rate"] = value["successes"] / value["attempts"]
    return {
        "num_episodes": len(results),
        "num_success": successes,
        "full_task_success_rate": successes / len(results) if results else None,
        "stage_metrics": stage_attempts,
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    expert_index_path = Path(manifest["expert_index"]).expanduser().resolve()
    expert_index = json.loads(expert_index_path.read_text())
    workflows = select_episode_workflows(
        samples=list(manifest["samples"]),
        records=list(expert_index["records"]),
        difficulty=args.difficulty,
        sample_rank=args.sample_rank,
        episode_start=args.episode_start,
        episode_count=args.episode_count,
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_dir = (
        args.output_root.expanduser().resolve()
        / args.condition
        / timestamp
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["manifest"] = str(manifest_path)
    config["output_root"] = str(args.output_root)
    config["selected_episodes"] = [
        workflow["episode_index"] for workflow in workflows
    ]
    config["local_pose_api_key"] = "***" if args.local_pose_api_key else None
    config["pi05_api_key"] = "***" if args.pi05_api_key else None
    write_json(run_dir / "config.json", config)

    camera_names = tuple(
        dict.fromkeys(POLICY_CAMERA_NAMES + tuple(args.local_pose_cameras))
    )
    dataset = expert_index["dataset"]
    task_labels = load_task_labels(dataset)
    decision_maker = (
        OpenAICompatibleLocalPoseVLM(
            base_url=args.local_pose_base_url,
            model=args.local_pose_model,
            api_key=args.local_pose_api_key,
            timeout_s=args.local_pose_timeout_s,
        )
        if args.condition == "refiner"
        else None
    )
    env = None
    client = None
    results: list[dict[str, Any]] = []
    try:
        env = create_dataset_environment(
            dataset,
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
        for index, workflow in enumerate(workflows, start=1):
            episode_index = int(workflow["episode_index"])
            episode_dir = run_dir / f"episode_{episode_index:06d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[FullEpisode] {index}/{len(workflows)} "
                f"episode={episode_index} condition={args.condition}"
            )
            try:
                result = run_episode(
                    env=env,
                    client=client,
                    workflow=workflow,
                    manifest_path=manifest_path,
                    expert_index_path=expert_index_path,
                    dataset=dataset,
                    task_labels=task_labels,
                    episode_dir=episode_dir,
                    camera_names=camera_names,
                    decision_maker=decision_maker,
                    args=args,
                )
            except Exception as exc:
                result = {
                    "episode_index": episode_index,
                    "condition": args.condition,
                    "difficulty": args.difficulty,
                    "success": False,
                    "failure_code": "FULL_EPISODE_ERROR",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            write_json(episode_dir / "episode_result.json", result)
            results.append(result)
            print(
                f"[FullEpisode] episode={episode_index} "
                f"success={result.get('success')} "
                f"stopped_at={result.get('stopped_at')}"
            )
    finally:
        if client is not None:
            client.close()
        if env is not None:
            env.close()

    summary = {
        "benchmark": "RoboCasaOfficialWorkPoseFullEpisode",
        "version": 1,
        "condition": args.condition,
        "difficulty": args.difficulty,
        "manifest": str(manifest_path),
        "results": results,
        **aggregate(results),
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"[FullEpisode] success={summary['num_success']}/"
        f"{summary['num_episodes']} output={run_dir}"
    )
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
