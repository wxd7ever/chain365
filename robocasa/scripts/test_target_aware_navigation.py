#!/usr/bin/env python3
"""Plan or execute one target-aware navigation probe in SteamInMicrowave."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robocasa.atomic_task_policy_adapter import RemoteAtomicTaskPolicyAdapter
from robocasa.atomic_task_schemas import AtomicTaskCall
from robocasa.atomic_task_verifier import RuntimeAtomicTaskVerifier
from robocasa.local_work_pose_refiner import (
    LocalWorkPoseRefiner,
    OpenAICompatibleLocalPoseVLM,
)
from robocasa.navigation.navigation_pipeline import (
    TargetAwareNavigationPipeline,
    load_navigation_config,
)
from robocasa.navigation.skill_adapter import TargetAwarePolicyAdapter
from robocasa.navigation.navigation_types import TargetType
from robocasa.openpi_client import OpenPIWebsocketClient
from robocasa.pi05_env import create_pi05_env
from robocasa.view_scheduler import FixedViewScheduler
from robocasa.work_pose_dataset import current_base_pose


OPERATIONS = (
    "pick_vegetable",
    "place_vegetable_in_bowl",
    "pick_bowl",
    "place_bowl_in_microwave",
    "close_microwave",
    "turn_on_microwave",
)


def _call(operation: str) -> AtomicTaskCall:
    specs: dict[str, dict[str, Any]] = {
        "pick_vegetable": {
            "atomic_task": "PickObject",
            "arguments": {"object_id": "vegetable", "source_id": "sink"},
            "condition": {
                "predicate": "holding",
                "subject": "vegetable",
                "desired_value": True,
            },
            "prompt": "Pick up the vegetable from the sink and keep holding it.",
        },
        "place_vegetable_in_bowl": {
            "atomic_task": "PlaceObject",
            "arguments": {
                "object_id": "vegetable",
                "held_object_id": "vegetable",
                "destination_id": "bowl",
            },
            "condition": {
                "predicate": "inside",
                "subject": "vegetable",
                "object": "bowl",
                "desired_value": True,
            },
            "prompt": "Place the held vegetable inside the bowl.",
        },
        "pick_bowl": {
            "atomic_task": "PickObject",
            "arguments": {"object_id": "bowl"},
            "condition": {
                "predicate": "holding",
                "subject": "bowl",
                "desired_value": True,
            },
            "prompt": "Pick up the bowl and keep holding it.",
        },
        "place_bowl_in_microwave": {
            "atomic_task": "PlaceObject",
            "arguments": {
                "object_id": "bowl",
                "held_object_id": "bowl",
                "destination_id": "microwave",
            },
            "condition": {
                "predicate": "inside",
                "subject": "bowl",
                "object": "microwave",
                "desired_value": True,
            },
            "prompt": "Place the held bowl inside the microwave.",
        },
        "close_microwave": {
            "atomic_task": "CloseMicrowave",
            "arguments": {"fixture_id": "microwave"},
            "condition": {
                "predicate": "closed",
                "subject": "microwave",
                "desired_value": True,
            },
            "prompt": "Close the microwave door.",
        },
        "turn_on_microwave": {
            "atomic_task": "TurnOnMicrowave",
            "arguments": {"fixture_id": "microwave"},
            "condition": {
                "predicate": "powered",
                "subject": "microwave",
                "desired_value": True,
            },
            "prompt": "Press the microwave start button to turn it on.",
        },
    }
    spec = specs[operation]
    return AtomicTaskCall.from_mapping(
        {
            "subgoal_id": operation,
            "atomic_task": spec["atomic_task"],
            "arguments": spec["arguments"],
            "termination_condition": spec["condition"],
            "policy_prompt": spec["prompt"],
            "metadata": {"navigation_probe": True},
        }
    )


def _plot_debug(path: Path, pipeline: TargetAwareNavigationPipeline, title: str) -> None:
    debug = pipeline.last_debug
    occupancy_map = debug.get("map")
    if occupancy_map is None:
        return
    bounds = occupancy_map.world_bounds
    extent = [bounds[0], bounds[1], bounds[2], bounds[3]]
    static_grid = getattr(occupancy_map, "static_grid", occupancy_map.raw_grid)
    articulated_grid = getattr(
        occupancy_map, "articulated_grid", np.zeros_like(occupancy_map.raw_grid)
    )
    figure, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    for axis, grid, label in (
        (axes[0], static_grid, "Static occupancy (cached)"),
        (axes[1], articulated_grid, "Current articulated occupancy"),
        (axes[2], occupancy_map.grid, "Current inflated occupancy + plan"),
    ):
        axis.imshow(
            grid,
            origin="lower",
            extent=extent,
            cmap="Greys",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        axis.set_aspect("equal")
        axis.set_title(label)
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world y [m]")

    for item in getattr(occupancy_map, "articulated_geometries", ()):
        polygon = np.asarray(item.get("polygon_xy", ()), dtype=float)
        if polygon.ndim != 2 or len(polygon) < 3:
            continue
        closed = np.vstack((polygon, polygon[0]))
        for axis in (axes[1], axes[2]):
            axis.plot(
                closed[:, 0],
                closed[:, 1],
                color="red",
                linewidth=1.2,
                alpha=0.8,
            )

    start = np.asarray(debug["start_xy"], dtype=float)
    collision_start = np.asarray(debug.get("start_collision_xy", start), dtype=float)
    target = debug["target"]
    axes[0].scatter(*start, c="royalblue", s=90, marker="o", label="base origin")
    axes[2].scatter(
        *collision_start,
        c="deepskyblue",
        s=70,
        marker="s",
        label="robot footprint center",
    )
    for axis in axes:
        axis.scatter(*target.xy, c="crimson", s=100, marker="*", label="target")
        fixture_alias = target.support_fixture or (
            target.target_alias if target.target_type == TargetType.FIXTURE else None
        )
        geometry = occupancy_map.fixture_geometries.get(fixture_alias)
        if geometry is not None:
            polygon = np.vstack((geometry.polygon_xy, geometry.polygon_xy[0]))
            axis.plot(polygon[:, 0], polygon[:, 1], color="darkorange", linewidth=2)
            axis.arrow(
                geometry.front_center_xy[0],
                geometry.front_center_xy[1],
                geometry.front_normal_xy[0] * 0.35,
                geometry.front_normal_xy[1] * 0.35,
                color="darkorange",
                width=0.015,
                length_includes_head=True,
            )
    raw_candidates = debug.get("raw_candidates", [])
    valid_candidates = debug.get("valid_candidates", [])
    if raw_candidates:
        raw_centers = [
            pipeline._footprint_center(value.xy, value.yaw)
            for value in raw_candidates
        ]
        axes[2].scatter(
            [value[0] for value in raw_centers],
            [value[1] for value in raw_centers],
            c="silver",
            s=8,
            alpha=0.45,
            label="raw candidates",
        )
    if valid_candidates:
        valid_centers = [
            pipeline._footprint_center(value.xy, value.yaw)
            for value in valid_candidates
        ]
        axes[2].scatter(
            [value[0] for value in valid_centers],
            [value[1] for value in valid_centers],
            c="limegreen",
            s=13,
            alpha=0.75,
            label="valid candidates",
        )
    selected = debug.get("selected")
    if selected is not None:
        selected_center = pipeline._footprint_center(selected.xy, selected.yaw)
        axes[2].scatter(
            selected_center[0],
            selected_center[1],
            c="magenta",
            s=90,
            marker="X",
            label="goal footprint center",
        )
        axes[2].arrow(
            selected_center[0],
            selected_center[1],
            0.30 * np.cos(selected.yaw),
            0.30 * np.sin(selected.yaw),
            color="magenta",
            width=0.015,
            length_includes_head=True,
        )
    path_xy = np.asarray(
        debug.get("collision_center_path_xy", debug.get("path_xy", [])), dtype=float
    )
    if path_xy.ndim == 2 and len(path_xy):
        axes[2].plot(
            path_xy[:, 0],
            path_xy[:, 1],
            "c-o",
            linewidth=2,
            markersize=4,
            label="A* shortcut path",
        )
    axes[2].legend(loc="best", fontsize=8)
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _print_articulated_debug(pipeline: TargetAwareNavigationPipeline) -> None:
    occupancy_map = pipeline.last_debug.get("map")
    if occupancy_map is None:
        return
    print("[ArticulatedMap] " + json.dumps(
        getattr(occupancy_map, "map_debug", {}),
        ensure_ascii=False,
        default=_json_default,
    ))
    for item in getattr(occupancy_map, "articulated_geometries", ()):
        if "microwave" not in str(item.get("fixture_name", "")).lower():
            continue
        print("[ArticulatedGeom] " + json.dumps(
            {
                "fixture_name": item.get("fixture_name"),
                "component": item.get("component"),
                "joint_name": item.get("joint_name"),
                "joint_state": item.get("joint_state"),
                "geom_name": item.get("geom_name"),
                "geom_type": item.get("geom_type"),
                "world_position": item.get("world_position"),
                "world_rotation": item.get("world_rotation"),
                "projected_polygon": item.get("polygon_xy"),
                "z_range": item.get("z_range"),
            },
            ensure_ascii=False,
        ))


def _base_pose_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    dx = float(after["position"][0]) - float(before["position"][0])
    dy = float(after["position"][1]) - float(before["position"][1])
    yaw = (float(after["yaw_rad"]) - float(before["yaw_rad"]) + np.pi) % (
        2.0 * np.pi
    ) - np.pi
    return {
        "delta_x_m": dx,
        "delta_y_m": dy,
        "delta_translation_m": float(np.hypot(dx, dy)),
        "delta_yaw_deg": float(np.degrees(yaw)),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", "--env_name", dest="task", default="SteamInMicrowave")
    parser.add_argument("--layout", "--layout_id", dest="layout", type=int, default=4)
    parser.add_argument("--style", "--style_id", dest="style", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--operation", choices=OPERATIONS, default="pick_bowl")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--run-policy",
        action="store_true",
        help="after navigation/refinement, run the existing π0.5 adapter and verifier",
    )
    parser.add_argument(
        "--run-local-refiner",
        action="store_true",
        help="run the existing VLM local refiner without running π0.5",
    )
    parser.add_argument(
        "--skip-local-refiner",
        action="store_true",
        help="with --run-policy, skip local refinement for ablation",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--view-selection-config", type=Path, default=Path("configs/view_selection.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("navigation_debug"))
    parser.add_argument(
        "--debug-articulated-map",
        action="store_true",
        help="print live joint/geoms and save three-layer occupancy visualization",
    )
    parser.add_argument("--pi05-host", default="172.16.11.115")
    parser.add_argument("--pi05-port", type=int, default=8000)
    parser.add_argument("--pi05-horizon", type=int, default=600)
    parser.add_argument(
        "--pi05-base-action-mode", choices=("full", "residual", "frozen"), default="residual"
    )
    parser.add_argument("--pi05-base-residual-limit", type=float, default=0.15)
    parser.add_argument("--pi05-replan-steps", type=int, default=5)
    parser.add_argument("--pi05-resize-size", type=int, default=224)
    parser.add_argument("--local-pose-base-url", default="http://172.16.11.113:8000/v1")
    parser.add_argument("--local-pose-model", default="/data/vllm/models/Qwen/Qwen-35B-FP8")
    parser.add_argument("--local-pose-api-key")
    parser.add_argument("--local-pose-timeout-s", type=float, default=300.0)
    parser.add_argument("--local-pose-max-tokens", type=int, default=8192)
    parser.add_argument(
        "--local-pose-enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--local-pose-max-decisions", type=int, default=8)
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="record actual execution; ignored in --plan-only mode",
    )
    parser.add_argument("--video-camera", default="robot0_agentview_left")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-skip", type=int, default=1)
    args = parser.parse_args()

    if args.run_policy or args.run_local_refiner:
        args.execute = True
    if not args.execute:
        args.plan_only = True
    if args.skip_local_refiner and args.run_local_refiner:
        parser.error("--skip-local-refiner conflicts with --run-local-refiner")
    if min(
        args.layout,
        args.style,
        args.pi05_port,
        args.pi05_horizon,
        args.pi05_replan_steps,
        args.pi05_resize_size,
        args.local_pose_timeout_s,
        args.local_pose_max_tokens,
        args.local_pose_max_decisions,
        args.video_fps,
        args.video_skip,
    ) <= 0:
        parser.error("layout/style and execution limits must be positive")

    env = None
    client = None
    video_metadata: dict[str, Any] | None = None
    output_stem = f"{args.task}_layout{args.layout:02d}_{args.operation}"
    figure_path = args.output_dir / f"{output_stem}.png"
    result_path = args.output_dir / f"{output_stem}.json"
    video_path = args.output_dir / f"{output_stem}.mp4"
    artifact_dir = args.output_dir / f"{output_stem}_artifacts"
    try:
        print(
            f"[Env] {args.task} layout={args.layout} style={args.style} seed={args.seed}"
        )
        env = create_pi05_env(
            env_name=args.task,
            layout_id=args.layout,
            style_id=args.style,
            seed=args.seed,
        )
        env.reset()
        if args.record_video and args.execute:
            video_metadata = env.start_episode_video_recording(
                video_path,
                camera_name=args.video_camera,
                fps=args.video_fps,
                frame_stride=args.video_skip,
            )
        elif args.record_video:
            print("[Video] --plan-only does not execute steps; recording skipped")

        pipeline = TargetAwareNavigationPipeline(config_path=args.config)
        call = _call(args.operation)
        policy_result = None
        refinement_result = None
        stage_result = None
        navigation_result = None
        local_refinement_delta = None
        local_refinement_fallback = False
        coarse_pose_restore_result = None

        local_refiner_enabled = bool(
            args.run_local_refiner or (args.run_policy and not args.skip_local_refiner)
        )
        local_refiner = None
        if local_refiner_enabled:
            local_refiner = LocalWorkPoseRefiner(
                decision_maker=OpenAICompatibleLocalPoseVLM(
                    base_url=args.local_pose_base_url,
                    model=args.local_pose_model,
                    api_key=args.local_pose_api_key,
                    timeout_s=args.local_pose_timeout_s,
                    max_tokens=args.local_pose_max_tokens,
                    enable_thinking=args.local_pose_enable_thinking,
                ),
                log_dir=artifact_dir,
                camera_names=(
                    "robot0_topview",
                    "robot0_frontview",
                    "robot0_agentview_left",
                    "robot0_agentview_right",
                    "robot0_eye_in_hand",
                ),
                view_scheduler=FixedViewScheduler(args.view_selection_config),
                max_decisions=args.local_pose_max_decisions,
                held_object_guard=True,
            )

        if args.run_policy:
            client = OpenPIWebsocketClient(host=args.pi05_host, port=args.pi05_port)
            verifier = RuntimeAtomicTaskVerifier()
            verifier.verification_mode = "skill_end"
            base_adapter = RemoteAtomicTaskPolicyAdapter(
                client=client,
                verifier=verifier,
                log_dir=artifact_dir,
                resize_size=args.pi05_resize_size,
                replan_steps=args.pi05_replan_steps,
                atomic_task_horizon=args.pi05_horizon,
                verify_interval=60,
                min_steps_before_verify=60,
                base_action_mode=args.pi05_base_action_mode,
                base_residual_limit=args.pi05_base_residual_limit,
                held_object_guard=True,
                render=args.record_video,
                video_skip=args.video_skip,
            )
            composed = TargetAwarePolicyAdapter(
                policy_adapter=base_adapter,
                navigation_pipeline=pipeline,
                refiner=local_refiner,
                navigation_enabled=True,
                local_refiner_enabled=local_refiner_enabled,
                local_refiner_failure_policy=load_navigation_config(args.config)[
                    "navigation"
                ].get("local_refiner_failure_policy", "use_coarse_pose"),
                context={
                    "layout_id": args.layout,
                    "style_id": args.style,
                    "seed": args.seed,
                    "debug_articulated_map": args.debug_articulated_map,
                },
            )
            policy_result = composed.execute(
                env=env,
                scheduler_query={"atomic_task_call": call.to_dict()},
                episode_id=output_stem,
            )
            stage_result = policy_result.get("target_aware_stage")
            if isinstance(stage_result, dict):
                navigation_result = policy_result.get(
                    "navigation_result", stage_result.get("navigation_result")
                )
                refinement_result = policy_result.get(
                    "local_pose_refinement_result",
                    stage_result.get("local_refiner_result"),
                )
            overall_success = bool(policy_result.get("success", False))
        else:
            result = pipeline.navigate_to_manipulation_target(
                env,
                call,
                context={
                    "layout_id": args.layout,
                    "debug_articulated_map": args.debug_articulated_map,
                },
                plan_only=args.plan_only,
            )
            navigation_result = result.to_dict()
            overall_success = bool(result.success)
            if overall_success and local_refiner is not None:
                refinement_pose_before = current_base_pose(env)
                refinement_result = local_refiner.refine(
                    env=env,
                    atomic_task_call=call.to_dict(),
                    grounding_result={
                        "target_name": result.target_name,
                        "target_type": result.target_type,
                        "target_position": result.target_position,
                        "selected_goal": result.selected_goal,
                    },
                    episode_id=output_stem,
                )
                refinement_pose_after = current_base_pose(env)
                local_refinement_delta = _base_pose_delta(
                    refinement_pose_before, refinement_pose_after
                )
                if not refinement_result.get("success", False):
                    local_refinement_fallback = True
                    failure_policy = load_navigation_config(args.config)["navigation"].get(
                        "local_refiner_failure_policy", "use_coarse_pose"
                    )
                    if failure_policy == "abort":
                        overall_success = False
                    else:
                        restore = pipeline.restore_coarse_pose(
                            env, refinement_pose_before,
                            context={"layout_id": args.layout},
                        )
                        coarse_pose_restore_result = restore.to_dict()
                        overall_success = bool(restore.success)

        _plot_debug(figure_path, pipeline, output_stem)
        if args.debug_articulated_map:
            _print_articulated_debug(pipeline)
        if video_metadata is not None:
            video_metadata = env.finalize_episode_video_recording()
        payload = {
            "task": args.task,
            "layout": args.layout,
            "style": args.style,
            "seed": args.seed,
            "operation": args.operation,
            "mode": (
                "navigation_refiner_policy"
                if args.run_policy
                else "navigation_refiner"
                if local_refiner_enabled
                else "execute"
                if args.execute
                else "plan_only"
            ),
            "atomic_task_call": call.to_dict(),
            "success": overall_success,
            "navigation_result": navigation_result,
            "local_refiner_result": refinement_result,
            "local_refinement_delta": (
                stage_result.get("local_refinement_delta")
                if isinstance(stage_result, dict)
                else local_refinement_delta
            ),
            "coarse_pose_restore_result": coarse_pose_restore_result,
            "local_refinement_fallback": (
                stage_result.get("local_refinement_failure_policy_applied", False)
                if isinstance(stage_result, dict)
                else local_refinement_fallback
            ),
            "policy_result": policy_result,
            "target_aware_stage": stage_result,
            "debug_figure": str(figure_path.resolve()),
            "video": video_metadata,
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
            encoding="utf-8",
        )
        reason = (navigation_result or {}).get("reason")
        print(f"[Result] success={overall_success} navigation_reason={reason}")
        if isinstance(navigation_result, dict):
            print(
                "[Plan] candidates={}/{} path={:.2f}m waypoints={}".format(
                    navigation_result.get("valid_candidate_count", 0),
                    navigation_result.get("raw_candidate_count", 0),
                    float(navigation_result.get("path_length", 0.0)),
                    navigation_result.get("waypoint_count", 0),
                )
            )
            execution = navigation_result.get("execution")
            if isinstance(execution, dict):
                print(
                    "[FinalPose] position_error={:.3f}m yaw_error={:.1f}deg".format(
                        float(execution.get("position_error") or 0.0),
                        float(execution.get("yaw_error") or 0.0),
                    )
                )
        print(f"[Output] {result_path.resolve()}")
        if figure_path.exists():
            print(f"[Figure] {figure_path.resolve()}")
        if video_metadata is not None:
            print(
                "[Video] path={} frames={} error={}".format(
                    video_metadata.get("path"),
                    video_metadata.get("frame_count"),
                    video_metadata.get("error"),
                )
            )
        return 0 if overall_success else 2
    except Exception as exc:
        if env is not None and video_metadata is not None:
            video_metadata = env.finalize_episode_video_recording()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        error_path = args.output_dir / f"{output_stem}_error.json"
        error_path.write_text(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "video": video_metadata,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[Error] {type(exc).__name__}: {exc}")
        print(f"[Error] {error_path.resolve()}")
        return 1
    finally:
        if client is not None:
            client.close()
        if env is not None:
            if video_metadata is not None:
                env.finalize_episode_video_recording()
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
