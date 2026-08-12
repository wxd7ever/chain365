#!/usr/bin/env python3
"""Run complete SteamInMicrowave episodes from paired work-pose samples."""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robocasa.atomic_task_policy_adapter import RemoteAtomicTaskPolicyAdapter
from robocasa.atomic_task_prompt_builder import build_atomic_task_prompt
from robocasa.atomic_task_schemas import AtomicTaskCall
from robocasa.atomic_task_verifier import RuntimeAtomicTaskVerifier
from robocasa.held_object_guard import build_held_object_guard
from robocasa.io_utils import save_video
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
    move_eef_to_pose,
    object_eef_diagnostics,
    pose_error,
    raw_task_env,
    resolve_record_asset,
    restore_official_state,
    scene_context,
    terminal_eef_pose,
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
# Held runs advanced 0.24-0.34 mm/step at command 0.20. Budget against
# 0.20 mm/step so arm-pose-dependent slowdown does not exhaust the horizon.
STATION_PROGRESS_PER_UNIT_COMMAND_M = 0.001
STATION_MOVE_SETTLING_MARGIN_STEPS = 120

VLA_COMPLETION_INSTRUCTIONS = {
    "PickObject": (
        "After grasping it securely, lift it completely clear of the source, "
        "retract the arm to a stable carrying pose, keep the gripper closed, "
        "and hold the object steadily."
    ),
    "PlaceObject": (
        "After placing and releasing it, move the gripper completely clear of "
        "the destination and finish with the arm in a stable pose."
    ),
}
DEFAULT_VLA_COMPLETION_INSTRUCTION = (
    "After completing the operation, release contact with the fixture and "
    "retract the end effector to a stable pose without undoing the task."
)


def prepare_vla_completion_call(
    call: AtomicTaskCall,
) -> tuple[AtomicTaskCall, list[dict[str, Any]]]:
    """Ask pi0.5 to finish in a safe semantic pose, not a recorded EEF pose."""

    value = call.to_dict()
    instruction = VLA_COMPLETION_INSTRUCTIONS.get(
        call.atomic_task, DEFAULT_VLA_COMPLETION_INSTRUCTION
    )
    original_prompt = build_atomic_task_prompt(call).rstrip()
    if instruction not in original_prompt:
        original_prompt = original_prompt.rstrip(". ") + ". " + instruction
    value["policy_prompt"] = original_prompt

    added_predicates: list[str] = []
    if call.atomic_task == "PickObject":
        source_id = str(call.arguments.get("source_id", "")).strip()
        conditions = value["termination_condition"]
        if isinstance(conditions, Mapping):
            conditions = [dict(conditions)]
        else:
            conditions = [dict(item) for item in conditions]
        if source_id and not any(
            condition.get("predicate") == "eef_outside_fixture"
            and condition.get("subject") == source_id
            for condition in conditions
        ):
            conditions.append(
                {
                    "predicate": "eef_outside_fixture",
                    "subject": source_id,
                    "margin": 0.02,
                    "desired_value": True,
                }
            )
            added_predicates.append("eef_outside_fixture")
        value["termination_condition"] = conditions

    value["metadata"]["operation_completion"] = {
        "mode": "vla",
        "instruction": instruction,
        "added_predicates": added_predicates,
    }
    changes = [
        {
            "type": "prepare_vla_completion",
            "subgoal_id": call.subgoal_id,
            "atomic_task": call.atomic_task,
            "added_predicates": added_predicates,
        }
    ]
    return AtomicTaskCall.from_mapping(value), changes


def station_move_step_budget(
    *,
    initial_translation_m: float,
    translation_tolerance_m: float,
    max_translation_command: float,
    minimum_steps: int,
    maximum_steps: int,
) -> int:
    """Estimate a safe movement horizon from distance and command limit."""

    if initial_translation_m < 0 or translation_tolerance_m < 0:
        raise ValueError("station translation distances must be non-negative")
    if max_translation_command <= 0:
        raise ValueError("max_translation_command must be positive")
    if minimum_steps <= 0 or maximum_steps < minimum_steps:
        raise ValueError("invalid station movement step bounds")
    remaining_distance = max(
        0.0, initial_translation_m - translation_tolerance_m
    )
    estimated_travel_steps = math.ceil(
        remaining_distance
        / (
            STATION_PROGRESS_PER_UNIT_COMMAND_M
            * max_translation_command
        )
    )
    return min(
        maximum_steps,
        max(
            minimum_steps,
            estimated_travel_steps + STATION_MOVE_SETTLING_MARGIN_STEPS,
        ),
    )


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
    parser.add_argument(
        "--pi05_task_retries",
        type=int,
        default=2,
        help=(
            "Retry verifier-marked retryable atomic operations from the current "
            "environment state."
        ),
    )
    parser.add_argument("--pi05_resize_size", type=int, default=224)
    parser.add_argument("--pi05_horizon", type=int, default=600)
    parser.add_argument("--pi05_replan_steps", type=int, default=5)
    parser.add_argument("--pi05_verify_interval", type=int, default=5)
    parser.add_argument("--pi05_min_steps_before_verify", type=int, default=5)
    parser.add_argument(
        "--pi05_success_handoff_steps",
        type=int,
        default=50,
        help=(
            "Additional pi0.5-controlled steps after the verifier first sees "
            "success, allowing the VLA to retract into a stable pose."
        ),
    )
    parser.add_argument(
        "--operation_completion_mode",
        choices=("vla", "expert"),
        default="vla",
        help=(
            "Use language-conditioned pi0.5 for the post-operation stable pose "
            "(default), or the legacy recorded expert EEF pose controller."
        ),
    )
    parser.add_argument(
        "--pi05_base_action_mode",
        choices=("frozen", "residual", "full"),
        default="frozen",
    )
    parser.add_argument("--pi05_base_residual_limit", type=float, default=0.15)
    parser.add_argument(
        "--min_station_move_steps",
        type=int,
        default=360,
        help="Minimum horizon for every dataset-station movement.",
    )
    parser.add_argument(
        "--max_station_move_steps",
        type=int,
        default=25000,
        help=(
            "Hard cap for the distance-adaptive dataset-station movement "
            "horizon."
        ),
    )
    parser.add_argument("--station_translation_tolerance_m", type=float, default=0.03)
    parser.add_argument("--station_yaw_tolerance_deg", type=float, default=4.0)
    parser.add_argument("--held_max_translation_command", type=float, default=0.20)
    parser.add_argument("--held_max_rotation_command", type=float, default=0.15)
    parser.add_argument(
        "--post_pick_stabilization_steps",
        type=int,
        default=0,
        help="Optional stationary hold verification after VLA/expert completion.",
    )
    parser.add_argument("--expert_handoff_window", type=int, default=10)
    parser.add_argument("--expert_handoff_max_steps", type=int, default=400)
    parser.add_argument(
        "--expert_handoff_translation_tolerance_m", type=float, default=0.05
    )
    parser.add_argument(
        "--expert_handoff_orientation_tolerance_deg", type=float, default=12.0
    )
    parser.add_argument(
        "--expert_handoff_max_translation_command", type=float, default=0.20
    )
    parser.add_argument(
        "--expert_handoff_max_rotation_command", type=float, default=0.15
    )
    parser.add_argument("--expert_handoff_stable_steps", type=int, default=8)
    parser.add_argument("--expert_handoff_settle_steps", type=int, default=0)
    parser.add_argument(
        "--navigation_hold_confirmation_steps",
        type=int,
        default=8,
        help="Consecutive holding checks required immediately before held navigation.",
    )
    parser.add_argument(
        "--held_drop_confirmation_steps",
        type=int,
        default=5,
        help="Consecutive failed holding checks required before declaring a drop.",
    )
    parser.add_argument("--held_slow_start_steps", type=int, default=150)
    parser.add_argument(
        "--held_slow_start_translation_command", type=float, default=0.08
    )
    parser.add_argument("--navigation_video_stride", type=int, default=10)
    parser.add_argument(
        "--navigation_video_camera", default="robot0_agentview_left"
    )
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
        args.min_station_move_steps,
        args.max_station_move_steps,
        args.expert_handoff_window,
        args.expert_handoff_max_steps,
        args.expert_handoff_translation_tolerance_m,
        args.expert_handoff_orientation_tolerance_deg,
        args.expert_handoff_max_translation_command,
        args.expert_handoff_max_rotation_command,
        args.expert_handoff_stable_steps,
        args.held_max_translation_command,
        args.held_max_rotation_command,
        args.navigation_hold_confirmation_steps,
        args.held_drop_confirmation_steps,
        args.navigation_video_stride,
    )
    if any(value <= 0 for value in positive):
        parser.error("episode, policy, camera, and movement limits must be positive")
    if args.min_station_move_steps > args.max_station_move_steps:
        parser.error(
            "--min_station_move_steps cannot exceed --max_station_move_steps"
        )
    if (
        args.post_pick_stabilization_steps < 0
        or args.held_slow_start_steps < 0
        or args.pi05_success_handoff_steps < 0
        or args.expert_handoff_settle_steps < 0
    ):
        parser.error(
            "stabilization, settle, slow-start, and policy handoff steps must "
            "be non-negative"
        )
    if (
        args.post_pick_stabilization_steps
        and args.post_pick_stabilization_steps
        < args.navigation_hold_confirmation_steps
    ):
        parser.error(
            "--post_pick_stabilization_steps must be at least "
            "--navigation_hold_confirmation_steps"
        )
    if not (
        0 < args.held_slow_start_translation_command
        <= args.held_max_translation_command
    ):
        parser.error(
            "--held_slow_start_translation_command must be positive and cannot "
            "exceed --held_max_translation_command"
        )
    if args.held_max_translation_command > 1.0:
        parser.error(
            "--held_max_translation_command cannot exceed controller limit 1.0"
        )
    if (
        args.expert_handoff_max_translation_command > 1.0
        or args.expert_handoff_max_rotation_command > 1.0
    ):
        parser.error(
            "expert handoff commands cannot exceed controller limit 1.0"
        )
    if (
        args.episode_start < 0
        or args.sample_rank < 0
        or args.pi05_task_retries < 0
    ):
        parser.error("episode_start, sample_rank, and retry counts must be non-negative")
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


def official_stage_handoff_poses(
    dataset: str | Path,
    episode_index: int,
    task_labels: Mapping[int, str],
    *,
    window: int = 10,
) -> dict[int, dict[str, Any]]:
    """Return official operation-terminal EEF poses in the robot base frame."""

    dataframe = load_episode_dataframe(dataset, episode_index)
    observations = np.stack(dataframe["observation.state"].to_numpy())
    result: dict[int, dict[str, Any]] = {}
    for segment in annotation_segments(dataframe, task_labels):
        if segment.stage in {"navigate", "done"}:
            continue
        pose = terminal_eef_pose(observations, segment, window=window)
        pose.update(
            {
                "subtask_idx": segment.subtask_idx,
                "subtask": segment.subtask,
                "source_atomic_task": segment.source_atomic_task,
                "stage": segment.stage,
            }
        )
        result[segment.subtask_idx] = pose
    return result


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


def stabilize_held_object(
    *,
    env: Any,
    atomic_task_call: AtomicTaskCall,
    object_id: str,
    total_steps: int,
    required_consecutive: int,
    drop_confirmation_steps: int,
    stop_when_confirmed: bool = False,
) -> dict[str, Any]:
    """Hold the arm/base still and verify a grasp over consecutive sim steps."""

    if total_steps <= 0 or required_consecutive <= 0:
        raise ValueError("held-object stabilization steps must be positive")
    if total_steps < required_consecutive:
        raise ValueError("total stabilization steps cannot be below confirmation steps")
    guard = build_held_object_guard(
        env=env,
        atomic_task_call=atomic_task_call,
        enabled=True,
        hold_confirmation_steps=required_consecutive,
        drop_confirmation_steps=drop_confirmation_steps,
    )
    guard.start()
    trace: list[dict[str, Any]] = []
    consecutive = 0
    guard_failure = None
    environment_done = False
    for step_index in range(1, total_steps + 1):
        action = np.zeros(12, dtype=np.float32)
        action[11] = 1.0
        applied = guard.apply_action(action, step_index=step_index)
        _, _, environment_done, _ = env.step(applied)
        guard_failure = guard.observe(step_index=step_index)
        diagnostics = object_eef_diagnostics(env, object_id)
        holding = diagnostics.get("holding")
        consecutive = consecutive + 1 if holding is True else 0
        trace.append(
            {
                "step": step_index,
                "holding_consecutive": consecutive,
                **diagnostics,
            }
        )
        if guard_failure is not None or environment_done:
            break
        if stop_when_confirmed and consecutive >= required_consecutive:
            break
    holding_after = holding_state(env, object_id)
    success = bool(
        guard_failure is None
        and not environment_done
        and holding_after is True
        and consecutive >= required_consecutive
    )
    return {
        "success": success,
        "failure_code": (
            None
            if success
            else (
                "OBJECT_DROPPED_DURING_STABILIZATION"
                if guard_failure is not None
                else "HELD_OBJECT_NOT_STABLY_CONFIRMED"
            )
        ),
        "object_id": object_id,
        "steps": len(trace),
        "requested_steps": total_steps,
        "required_consecutive": required_consecutive,
        "final_consecutive": consecutive,
        "holding_after": holding_after,
        "guard_failure": guard_failure,
        "guard": guard.to_dict(),
        "trace": trace,
    }


def move_to_stage(
    *,
    env: Any,
    target_pose: Mapping[str, Any],
    upcoming_call: AtomicTaskCall | None,
    expected_holding_object: str | None,
    args: argparse.Namespace,
    video_path: Path | None = None,
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
    pre_navigation_confirmation = None
    if expected_holding_object and upcoming_call is not None:
        pre_navigation_confirmation = stabilize_held_object(
            env=env,
            atomic_task_call=upcoming_call,
            object_id=expected_holding_object,
            total_steps=max(
                args.navigation_hold_confirmation_steps * 2,
                args.navigation_hold_confirmation_steps,
            ),
            required_consecutive=args.navigation_hold_confirmation_steps,
            drop_confirmation_steps=args.held_drop_confirmation_steps,
            stop_when_confirmed=True,
        )
        if not pre_navigation_confirmation["success"]:
            return {
                "success": False,
                "failure_code": "HELD_OBJECT_NOT_STABLE_BEFORE_NAVIGATION",
                "object_id": expected_holding_object,
                "holding_before": holding_before,
                "pre_navigation_hold_confirmation": pre_navigation_confirmation,
            }
        guard = build_held_object_guard(
            env=env,
            atomic_task_call=upcoming_call,
            enabled=True,
            hold_confirmation_steps=args.navigation_hold_confirmation_steps,
            drop_confirmation_steps=args.held_drop_confirmation_steps,
        )
    max_translation_command = (
        args.held_max_translation_command if expected_holding_object else 1.0
    )
    max_rotation_command = (
        args.held_max_rotation_command if expected_holding_object else 0.50
    )
    initial_pose = current_base_pose(env)
    initial_error = pose_error(initial_pose, target_pose)
    step_budget = station_move_step_budget(
        initial_translation_m=initial_error["translation_m"],
        translation_tolerance_m=args.station_translation_tolerance_m,
        max_translation_command=max_translation_command,
        minimum_steps=args.min_station_move_steps,
        maximum_steps=args.max_station_move_steps,
    )
    result = move_base_to_pose(
        env,
        target_pose,
        guard=guard,
        max_steps=step_budget,
        translation_tolerance_m=args.station_translation_tolerance_m,
        yaw_tolerance_deg=args.station_yaw_tolerance_deg,
        max_translation_command=max_translation_command,
        max_rotation_command=max_rotation_command,
        slow_start_steps=(
            args.held_slow_start_steps if expected_holding_object else 0
        ),
        slow_start_translation_command=(
            args.held_slow_start_translation_command
            if expected_holding_object
            else None
        ),
        diagnostic_object_id=expected_holding_object,
        capture_video=bool(args.video and video_path is not None),
        video_stride=args.navigation_video_stride,
        video_camera_name=args.navigation_video_camera,
        video_height=args.camera_size,
        video_width=args.camera_size,
    )
    video_frames = result.pop("video_frames", [])
    if args.video and video_path is not None and video_frames:
        try:
            save_video(video_path, video_frames, fps=20)
            result["video_path"] = str(video_path)
            result["video_frame_count"] = len(video_frames)
            result["video_stride"] = args.navigation_video_stride
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            result["video_save_error"] = str(exc)
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
        "initial_pose": initial_pose,
        "initial_error": initial_error,
        "minimum_step_budget": args.min_station_move_steps,
        "maximum_step_budget": args.max_station_move_steps,
        "effective_step_budget": step_budget,
        "max_translation_command": max_translation_command,
        "max_rotation_command": max_rotation_command,
        "slow_start_steps": (
            args.held_slow_start_steps if expected_holding_object else 0
        ),
        "slow_start_translation_command": (
            args.held_slow_start_translation_command
            if expected_holding_object
            else None
        ),
        "pre_navigation_hold_confirmation": pre_navigation_confirmation,
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
    if args.operation_completion_mode == "vla":
        call, completion_changes = prepare_vla_completion_call(call)
        contract_changes.extend(completion_changes)
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
        success_handoff_steps=(
            args.pi05_success_handoff_steps
            if args.operation_completion_mode == "vla"
            else 0
        ),
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
        "operation_completion_mode": args.operation_completion_mode,
    }



def execute_expert_handoff_with_retries(
    *,
    env: Any,
    call: AtomicTaskCall,
    expert_handoff_pose: Mapping[str, Any],
    step_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Reach the expert terminal EEF pose without repeating the completed skill."""

    handoff_attempts: list[dict[str, Any]] = []
    for attempt_index in range(args.pi05_task_retries + 1):
        guard = None
        if call.atomic_task == "PickObject":
            guard = build_held_object_guard(
                env=env,
                atomic_task_call=call,
                enabled=True,
                hold_confirmation_steps=args.navigation_hold_confirmation_steps,
                drop_confirmation_steps=args.held_drop_confirmation_steps,
            )
        handoff = move_eef_to_pose(
            env,
            expert_handoff_pose,
            guard=guard,
            max_steps=args.expert_handoff_max_steps,
            translation_tolerance_m=(
                args.expert_handoff_translation_tolerance_m
            ),
            orientation_tolerance_deg=(
                args.expert_handoff_orientation_tolerance_deg
            ),
            max_translation_command=(
                args.expert_handoff_max_translation_command
            ),
            max_rotation_command=args.expert_handoff_max_rotation_command,
            stable_steps=args.expert_handoff_stable_steps,
            settle_steps=args.expert_handoff_settle_steps,
            gripper_action=(1.0 if call.atomic_task == "PickObject" else 0.0),
        )
        handoff["attempt_index"] = attempt_index
        handoff_attempts.append(handoff)
        handoff_dir = (
            step_dir
            if attempt_index == 0
            else step_dir / f"retry_{attempt_index}"
        )
        handoff_dir.mkdir(parents=True, exist_ok=True)
        write_json(handoff_dir / "expert_handoff.json", handoff)
        if handoff["success"]:
            break
        if handoff.get("failure_code") == "OBJECT_DROPPED_DURING_EXPERT_HANDOFF":
            break

    result = dict(handoff_attempts[-1])
    result["num_attempts"] = len(handoff_attempts)
    if len(handoff_attempts) > 1:
        result["attempt_results"] = handoff_attempts
    return result

def execute_current_skill_with_retries(
    *,
    env: Any,
    client: OpenPIWebsocketClient,
    call: AtomicTaskCall,
    target_id: str,
    expert_pose: Mapping[str, Any],
    expert_handoff_pose: Mapping[str, Any] | None,
    step_dir: Path,
    step_id: str,
    decision_maker: OpenAICompatibleLocalPoseVLM | None,
    allow_refinement: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Retry one operation in-place using the old orchestrator semantics."""

    attempt_results: list[dict[str, Any]] = []
    for attempt_index in range(args.pi05_task_retries + 1):
        attempt_dir = (
            step_dir
            if attempt_index == 0
            else step_dir / f"retry_{attempt_index}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_step_id = (
            step_id
            if attempt_index == 0
            else f"{step_id}_operation_retry{attempt_index}"
        )
        attempt = execute_current_skill(
            env=env,
            client=client,
            call=call,
            target_id=target_id,
            expert_pose=expert_pose,
            step_dir=attempt_dir,
            step_id=attempt_step_id,
            decision_maker=decision_maker,
            allow_refinement=bool(allow_refinement and attempt_index == 0),
            args=args,
        )
        attempt["attempt_index"] = attempt_index
        if attempt.get("success") and args.operation_completion_mode == "expert":
            if expert_handoff_pose is None:
                raise ValueError("expert completion requires an expert handoff pose")
            handoff = execute_expert_handoff_with_retries(
                env=env,
                call=call,
                expert_handoff_pose=expert_handoff_pose,
                step_dir=attempt_dir,
                args=args,
            )
            attempt["expert_handoff"] = handoff
            if not handoff["success"]:
                attempt["policy_success"] = True
                attempt["success"] = False
                attempt["status"] = "uncertain"
                attempt["failure_code"] = handoff["failure_code"]
        if (
            attempt.get("success")
            and call.atomic_task == "PickObject"
            and args.post_pick_stabilization_steps > 0
        ):
            object_id = str(call.arguments["object_id"])
            stabilization = stabilize_held_object(
                env=env,
                atomic_task_call=call,
                object_id=object_id,
                total_steps=args.post_pick_stabilization_steps,
                required_consecutive=args.navigation_hold_confirmation_steps,
                drop_confirmation_steps=args.held_drop_confirmation_steps,
            )
            attempt["post_pick_stabilization"] = stabilization
            write_json(
                attempt_dir / "post_pick_stabilization.json",
                stabilization,
            )
            if not stabilization["success"]:
                attempt["policy_success"] = True
                attempt["success"] = False
                attempt["status"] = "uncertain"
                attempt["failure_code"] = "POST_PICK_STABILIZATION_FAILED"
        attempt_results.append(attempt)
        if attempt_index:
            write_json(attempt_dir / "result.json", attempt)

        handoff = attempt.get("expert_handoff")
        stabilization = attempt.get("post_pick_stabilization")
        retryable = bool(
            isinstance(handoff, Mapping)
            and handoff.get("failure_code")
            == "OBJECT_DROPPED_DURING_EXPERT_HANDOFF"
        )
        if not retryable:
            retryable = bool(
                isinstance(stabilization, Mapping)
                and not stabilization.get("success", False)
            )
        if not retryable:
            policy_result = attempt.get("policy_result")
            verifier_result = (
                policy_result.get("verifier_result")
                if isinstance(policy_result, Mapping)
                else None
            )
            retryable = bool(
                isinstance(verifier_result, Mapping)
                and verifier_result.get("retryable", False)
            )
        if attempt.get("success", False) or not retryable:
            break

    result = dict(attempt_results[-1])
    result["num_attempts"] = len(attempt_results)
    if len(attempt_results) > 1:
        result["attempt_results"] = attempt_results
    return result


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
    handoff_poses: dict[int, dict[str, Any]] = {}
    if args.operation_completion_mode == "expert":
        handoff_poses = official_stage_handoff_poses(
            dataset,
            episode_index,
            task_labels,
            window=args.expert_handoff_window,
        )
        write_json(episode_dir / "expert_handoff_poses.json", handoff_poses)
    operation_results: list[dict[str, Any]] = []
    navigation_results: list[dict[str, Any]] = []
    stabilization_results: list[dict[str, Any]] = []
    num_atomic_attempts = 0
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
                video_path=step_dir / "dataset_navigation.mp4",
            )
            navigation["target_stage"] = stage_index
            navigation["sample_id"] = sample["sample_id"]
            write_json(step_dir / "dataset_navigation.json", navigation)
            navigation_results.append(navigation)
            if not navigation["success"]:
                stopped_at = f"navigation_to_stage_{stage_index}"
                break
        result = execute_current_skill_with_retries(
            env=env,
            client=client,
            call=call,
            target_id=record["target_id"],
            expert_pose=sample["expert_base_pose"],
            expert_handoff_pose=handoff_poses.get(stage_index),
            step_dir=step_dir,
            step_id=f"ep{episode_index:06d}_s{stage_index:02d}",
            decision_maker=decision_maker,
            allow_refinement=True,
            args=args,
        )
        num_atomic_attempts += int(result["num_attempts"])
        result.update(
            {
                "stage_index": stage_index,
                "sample_id": sample["sample_id"],
                "atomic_task": call.atomic_task,
                "operation": record["operation"],
            }
        )
        attempts = result.get("attempt_results", [result])
        for attempt in attempts:
            stabilization = attempt.get("post_pick_stabilization")
            if isinstance(stabilization, Mapping):
                recorded_stabilization = dict(stabilization)
                recorded_stabilization["stage_index"] = stage_index
                recorded_stabilization["attempt_index"] = attempt.get(
                    "attempt_index", 0
                )
                stabilization_results.append(recorded_stabilization)
        final_stabilization = result.get("post_pick_stabilization")
        if isinstance(final_stabilization, Mapping):
            write_json(
                step_dir / "post_pick_stabilization.json",
                final_stabilization,
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
                video_path=step_dir / "dataset_navigation.mp4",
            )
            navigation["target_stage"] = stage_index
            navigation["source"] = "official_execute_stage_pose"
            write_json(step_dir / "dataset_navigation.json", navigation)
            navigation_results.append(navigation)
            if not navigation["success"]:
                stopped_at = f"navigation_to_stage_{stage_index}"
                break
            call = fixture_call(stage_index, fixture_id)
            result = execute_current_skill_with_retries(
                env=env,
                client=client,
                call=call,
                target_id=fixture_id,
                expert_pose=poses[stage_index],
                expert_handoff_pose=handoff_poses.get(stage_index),
                step_dir=step_dir,
                step_id=f"ep{episode_index:06d}_s{stage_index:02d}",
                decision_maker=None,
                allow_refinement=False,
                args=args,
            )
            num_atomic_attempts += int(result["num_attempts"])
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
        "stabilization_results": stabilization_results,
        "num_atomic_attempts": num_atomic_attempts,
        "max_task_retries": args.pi05_task_retries,
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
        dict.fromkeys(
            POLICY_CAMERA_NAMES
            + tuple(args.local_pose_cameras)
            + (args.navigation_video_camera,)
        )
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
                f"stopped_at={result.get('stopped_at')} "
                f"failure_code={result.get('failure_code')}"
            )
            if result.get("error"):
                print(f"[FullEpisode][Error] {result['error']}")
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
