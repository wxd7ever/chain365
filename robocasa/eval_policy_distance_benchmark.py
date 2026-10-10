#!/usr/bin/env python3
"""Measure an operation policy's autonomous base-adjustment range.

Each trial uses the same scene seed and PickObject prompt, places the mobile base
at one exact target distance while facing the target, then runs XR-1 directly.
No global navigator or local pose refiner is instantiated.
"""

from __future__ import annotations

import argparse
import csv
import math
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .atomic_task_policy_adapter import RemoteAtomicTaskPolicyAdapter
    from .atomic_task_schemas import AtomicTaskCall
    from .atomic_task_verifier import RuntimeAtomicTaskVerifier
    from .io_utils import make_run_dir, save_obs_image, write_json
    from .pi05_env import POLICY_CAMERA_NAMES, create_pi05_env
    from .utils.env_utils import detect_robot_collision, set_robot_to_position
    from .work_pose_dataset import current_base_pose
    from .xr1_policy_client import (
        DEFAULT_XR1_HOST,
        DEFAULT_XR1_MODEL_PATH,
        DEFAULT_XR1_PORT,
        XR1RoboCasaPolicy,
        check_xr1_connectivity,
    )
except ImportError:
    from atomic_task_policy_adapter import RemoteAtomicTaskPolicyAdapter
    from atomic_task_schemas import AtomicTaskCall
    from atomic_task_verifier import RuntimeAtomicTaskVerifier
    from io_utils import make_run_dir, save_obs_image, write_json
    from pi05_env import POLICY_CAMERA_NAMES, create_pi05_env
    from utils.env_utils import detect_robot_collision, set_robot_to_position
    from work_pose_dataset import current_base_pose
    from xr1_policy_client import (
        DEFAULT_XR1_HOST,
        DEFAULT_XR1_MODEL_PATH,
        DEFAULT_XR1_PORT,
        XR1RoboCasaPolicy,
        check_xr1_connectivity,
    )


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def desired_start_pose(
    target_xy: Sequence[float],
    reference_xy: Sequence[float],
    distance_m: float,
) -> dict[str, float]:
    """Return a point exactly distance_m from target, facing the target."""

    target = np.asarray(target_xy, dtype=float).reshape(-1)
    reference = np.asarray(reference_xy, dtype=float).reshape(-1)
    if target.size < 2 or reference.size < 2:
        raise ValueError("target_xy and reference_xy must each contain two values")
    if not np.isfinite(target[:2]).all() or not np.isfinite(reference[:2]).all():
        raise ValueError("target_xy and reference_xy must be finite")
    if not np.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("distance_m must be positive")
    ray = reference[:2] - target[:2]
    norm = float(np.linalg.norm(ray))
    if norm < 1e-6:
        raise ValueError("reference base position coincides with the target")
    xy = target[:2] + ray / norm * float(distance_m)
    yaw = math.atan2(float(target[1] - xy[1]), float(target[0] - xy[0]))
    return {"x": float(xy[0]), "y": float(xy[1]), "yaw": _wrap_angle(yaw)}


def pose_on_target_ray(
    target_xy: Sequence[float],
    distance_m: float,
    azimuth_deg: float,
) -> dict[str, float]:
    """Place the base on one target-centred ray and face it toward the target."""

    target = np.asarray(target_xy, dtype=float).reshape(-1)
    if target.size < 2 or not np.isfinite(target[:2]).all():
        raise ValueError("target_xy must contain two finite values")
    if not np.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("distance_m must be positive")
    theta = math.radians(float(azimuth_deg))
    x = float(target[0] + distance_m * math.cos(theta))
    y = float(target[1] + distance_m * math.sin(theta))
    return {"x": x, "y": y, "yaw": _wrap_angle(theta + math.pi)}


def summarize_trials(
    trials: Sequence[Mapping[str, Any]],
    distances: Sequence[float],
) -> dict[str, Any]:
    """Aggregate eligible and executed trial rates separately by distance."""

    buckets: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for trial in trials:
        buckets[float(trial["requested_distance_m"])].append(trial)

    by_distance: list[dict[str, Any]] = []
    for distance in distances:
        rows = buckets.get(float(distance), [])
        eligible = [row for row in rows if bool(row.get("eligible"))]
        executed = [row for row in rows if bool(row.get("executed"))]
        successes = sum(bool(row.get("success")) for row in executed)
        entered = sum(bool(row.get("entered_operation_region")) for row in executed)
        by_distance.append(
            {
                "distance_m": float(distance),
                "total_trials": len(rows),
                "eligible_visible_trials": len(eligible),
                "executed_trials": len(executed),
                "successes": int(successes),
                "success_rate_executed": (
                    float(successes / len(executed)) if executed else None
                ),
                "entered_operation_region": int(entered),
                "operation_region_entry_rate": (
                    float(entered / len(executed)) if executed else None
                ),
                "invalid_or_invisible": len(rows) - len(eligible),
            }
        )
    executed_all = [row for row in trials if bool(row.get("executed"))]
    success_all = sum(bool(row.get("success")) for row in executed_all)
    return {
        "total_trials": len(trials),
        "executed_trials": len(executed_all),
        "successes": int(success_all),
        "success_rate_executed": (
            float(success_all / len(executed_all)) if executed_all else None
        ),
        "by_distance": by_distance,
    }


def _raw_task_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if all(hasattr(current, key) for key in ("sim", "objects", "fixtures")):
            return current
        nested = getattr(current, "env", None)
        current = nested if nested is not current else None
    raise ValueError("could not find the underlying RoboCasa task environment")


def _object_position(raw_env: Any, alias: str) -> np.ndarray:
    if alias not in raw_env.obj_body_id:
        raise KeyError(
            f"object alias {alias!r} is unavailable; "
            f"available={sorted(raw_env.obj_body_id)}"
        )
    body_id = int(raw_env.obj_body_id[alias])
    return np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()


def _object_natural_name(raw_env: Any, alias: str) -> str:
    getter = getattr(raw_env, "get_obj_lang", None)
    if callable(getter):
        try:
            value = str(getter(alias)).strip()
            if value:
                return value
        except (KeyError, TypeError, ValueError):
            pass
    return str(alias).replace("_", " ")


def _set_base_pose(raw_env: Any, *, x: float, y: float, yaw: float) -> None:
    set_robot_to_position(raw_env, np.array([x, y, 0.0], dtype=float))
    anchor = np.asarray(
        getattr(raw_env, "init_robot_base_ori_anchor", [0.0, 0.0, 0.0]),
        dtype=float,
    ).reshape(-1)
    anchor_yaw = float(anchor[2]) if anchor.size >= 3 else 0.0
    yaw_addr = raw_env.sim.model.get_joint_qpos_addr(
        "mobilebase0_joint_mobile_yaw"
    )
    raw_env.sim.data.qpos[yaw_addr] = _wrap_angle(yaw - anchor_yaw)
    raw_env.sim.forward()


def _target_geom_ids(raw_env: Any, alias: str) -> set[int]:
    root_body = int(raw_env.obj_body_id[alias])
    body_parent = np.asarray(raw_env.sim.model.body_parentid, dtype=int)
    ids: set[int] = set()
    for geom_id, body_id_value in enumerate(raw_env.sim.model.geom_bodyid):
        body_id = int(body_id_value)
        cursor = body_id
        while cursor > 0 and cursor != root_body:
            cursor = int(body_parent[cursor])
        if cursor == root_body:
            ids.add(int(geom_id))
    return ids


def _visible_pixels(
    raw_env: Any,
    *,
    alias: str,
    camera_size: int,
) -> dict[str, int]:
    geom_ids = _target_geom_ids(raw_env, alias)
    counts: dict[str, int] = {}
    for camera in POLICY_CAMERA_NAMES:
        seg = np.asarray(
            raw_env.sim.render(
                camera_name=camera,
                height=int(camera_size),
                width=int(camera_size),
                segmentation=True,
            )
        )
        if seg.ndim < 2:
            counts[camera] = 0
            continue
        element_ids = seg[..., -1] if seg.ndim == 3 else seg
        counts[camera] = int(np.isin(element_ids, list(geom_ids)).sum())
    return counts


def _find_common_start_azimuth(
    raw_env: Any,
    *,
    alias: str,
    target_xy: Sequence[float],
    distances_m: Sequence[float],
    camera_size: int,
    search_step_deg: float,
    min_visible_pixels: int,
) -> tuple[float | None, list[dict[str, Any]]]:
    """Find one ray that is collision-free and visible at every distance."""

    candidates: list[dict[str, Any]] = []
    count = max(1, int(math.ceil(360.0 / float(search_step_deg))))
    for index in range(count):
        azimuth = float(index * 360.0 / count)
        per_distance: list[dict[str, Any]] = []
        valid = True
        for distance in distances_m:
            pose = pose_on_target_ray(target_xy, float(distance), azimuth)
            _set_base_pose(raw_env, **pose)
            visibility = _visible_pixels(
                raw_env,
                alias=alias,
                camera_size=camera_size,
            )
            collision = bool(detect_robot_collision(raw_env))
            pixels = max(visibility.values(), default=0)
            accepted = not collision and pixels >= min_visible_pixels
            valid = valid and accepted
            per_distance.append(
                {
                    "distance_m": float(distance),
                    "pose": pose,
                    "visible_pixels": visibility,
                    "visible_pixel_max": int(pixels),
                    "collision": collision,
                    "accepted": accepted,
                }
            )
        candidates.append(
            {
                "azimuth_deg": azimuth,
                "valid_for_all_distances": valid,
                "worst_visible_pixel_count": min(
                    row["visible_pixel_max"] for row in per_distance
                ),
                "per_distance": per_distance,
            }
        )
    valid_candidates = [
        row for row in candidates if row["valid_for_all_distances"]
    ]
    if not valid_candidates:
        return None, candidates
    best = max(valid_candidates, key=lambda row: row["worst_visible_pixel_count"])
    return float(best["azimuth_deg"]), candidates


def _save_views(path: Path, observation: Mapping[str, Any]) -> None:
    for camera in POLICY_CAMERA_NAMES:
        key = f"{camera}_image"
        if key in observation:
            save_obs_image(path / f"{camera}.png", observation[key])


def _make_pick_call(
    *,
    alias: str,
    natural_name: str,
    prompt: str,
    trial_id: str,
) -> AtomicTaskCall:
    return AtomicTaskCall.from_mapping(
        {
            "subgoal_id": trial_id,
            "atomic_task": "PickObject",
            "arguments": {
                "object_id": alias,
                "object_name": natural_name,
            },
            "termination_condition": {
                "predicate": "holding",
                "subject": alias,
                "desired_value": True,
            },
            "policy_prompt": prompt,
            "metadata": {"distance_benchmark": True},
        }
    )


class _PoseTrace:
    def __init__(
        self,
        *,
        env: Any,
        raw_env: Any,
        object_alias: str,
        operation_region_max_m: float,
    ) -> None:
        self.env = env
        self.raw_env = raw_env
        self.object_alias = object_alias
        self.operation_region_max_m = float(operation_region_max_m)
        self.rows: list[dict[str, Any]] = []

    def __call__(self, observation: Mapping[str, Any]) -> None:
        base = np.asarray(observation["robot0_base_pos"], dtype=float).reshape(-1)
        target = _object_position(self.raw_env, self.object_alias)
        distance = float(np.linalg.norm(base[:2] - target[:2]))
        self.rows.append(
            {
                "step": len(self.rows) + 1,
                "base_xy": base[:2].tolist(),
                "target_xy": target[:2].tolist(),
                "target_distance_m": distance,
                "inside_operation_region": distance <= self.operation_region_max_m,
            }
        )

    def metrics(self, initial_distance_m: float) -> dict[str, Any]:
        points = [np.asarray(row["base_xy"], dtype=float) for row in self.rows]
        path_length = float(
            sum(np.linalg.norm(b - a) for a, b in zip(points, points[1:]))
        )
        distances = [float(row["target_distance_m"]) for row in self.rows]
        first_entry = next(
            (
                int(row["step"])
                for row in self.rows
                if bool(row["inside_operation_region"])
            ),
            None,
        )
        return {
            "initial_target_distance_m": float(initial_distance_m),
            "final_target_distance_m": distances[-1] if distances else initial_distance_m,
            "minimum_target_distance_m": min(distances) if distances else initial_distance_m,
            "base_path_length_m": path_length,
            "entered_operation_region": first_entry is not None,
            "operation_region_first_entry_step": first_entry,
            "samples": len(self.rows),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare XR-1 PickObject success at fixed visible, target-facing "
            "initial distances without global navigation or a local refiner."
        )
    )
    parser.add_argument("--env_name", default="SteamInMicrowave")
    parser.add_argument("--layout_id", type=int, default=1)
    parser.add_argument("--style_id", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--distances_m", type=float, nargs="+", default=[0.8, 1.8, 2.8])
    parser.add_argument("--target_object", default="bowl")
    parser.add_argument(
        "--prompt",
        default="Pick up the bowl.",
        help="Exact operation instruction reused for every trial.",
    )
    parser.add_argument("--start_azimuth_deg", type=float)
    parser.add_argument("--azimuth_search_step_deg", type=float, default=15.0)
    parser.add_argument("--horizon", type=int, default=900)
    parser.add_argument("--verify_interval", type=int, default=5)
    parser.add_argument("--min_steps_before_verify", type=int, default=10)
    parser.add_argument("--pick_stable_steps", type=int, default=5)
    parser.add_argument("--pick_min_lift_m", type=float, default=0.04)
    parser.add_argument("--drop_confirmation_steps", type=int, default=30)
    parser.add_argument("--operation_region_max_m", type=float, default=0.95)
    parser.add_argument("--settle_steps", type=int, default=10)
    parser.add_argument("--camera_size", type=int, default=256)
    parser.add_argument("--min_visible_pixels", type=int, default=4)
    parser.add_argument("--allow_invisible", action="store_true")
    parser.add_argument("--allow_initial_collision", action="store_true")
    parser.add_argument("--video_skip", type=int, default=2)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("robocasa/outputs_policy_distance_benchmark"),
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

    if args.layout_id <= 0 or args.style_id <= 0:
        parser.error("--layout_id and --style_id must be positive")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be non-negative")
    if any(distance <= 0 for distance in args.distances_m):
        parser.error("--distances_m must be positive")
    positive = (
        args.horizon,
        args.verify_interval,
        args.min_steps_before_verify,
        args.pick_stable_steps,
        args.drop_confirmation_steps,
        args.operation_region_max_m,
        args.settle_steps,
        args.camera_size,
        args.video_skip,
    )
    if any(value <= 0 for value in positive):
        parser.error("horizon, intervals, thresholds, settle, camera, and video values must be positive")
    if args.min_visible_pixels < 0:
        parser.error("--min_visible_pixels must be non-negative")
    if args.pick_min_lift_m < 0:
        parser.error("--pick_min_lift_m must be non-negative")
    if args.azimuth_search_step_deg <= 0 or args.azimuth_search_step_deg > 360:
        parser.error("--azimuth_search_step_deg must be in (0, 360]")
    if not args.target_object.strip() or not args.prompt.strip():
        parser.error("--target_object and --prompt must be non-empty")
    return args


def _write_csv(path: Path, trials: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "seed",
        "requested_distance_m",
        "actual_initial_distance_m",
        "visible_pixel_max",
        "initial_collision",
        "eligible",
        "executed",
        "success",
        "status",
        "failure_code",
        "completed_steps",
        "entered_operation_region",
        "operation_region_first_entry_step",
        "minimum_target_distance_m",
        "final_target_distance_m",
        "base_path_length_m",
        "trial_dir",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trial in trials:
            writer.writerow({key: trial.get(key) for key in fields})


def main() -> int:
    args = parse_args()
    run_dir = make_run_dir(args.output_root, env_name=f"{args.env_name}_xr1_distance")
    config = vars(args).copy()
    config["output_root"] = str(args.output_root)
    config["distances_m"] = [float(value) for value in args.distances_m]
    config["seeds"] = [int(value) for value in args.seeds]
    config.update(
        {
            "global_navigation": False,
            "local_pose_refiner": False,
            "base_action_mode": "full",
            "same_prompt_all_trials": True,
        }
    )
    write_json(run_dir / "config.json", config)

    check_xr1_connectivity(args.xr1_host, args.xr1_port, args.xr1_connect_timeout_s)
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
    trials: list[dict[str, Any]] = []
    try:
        for seed in args.seeds:
            probe_env = None
            try:
                if args.start_azimuth_deg is not None:
                    chosen_azimuth = float(args.start_azimuth_deg) % 360.0
                    search_record = {
                        "seed": int(seed),
                        "mode": "explicit",
                        "chosen_azimuth_deg": chosen_azimuth,
                        "candidates": [],
                    }
                else:
                    probe_env = create_pi05_env(
                        env_name=args.env_name,
                        layout_id=args.layout_id,
                        style_id=args.style_id,
                        seed=int(seed),
                        camera_size=args.camera_size,
                    )
                    probe_env.reset()
                    probe_raw = _raw_task_env(probe_env)
                    probe_target = _object_position(
                        probe_raw, args.target_object.strip()
                    )
                    chosen_azimuth, candidates = _find_common_start_azimuth(
                        probe_raw,
                        alias=args.target_object.strip(),
                        target_xy=probe_target[:2],
                        distances_m=args.distances_m,
                        camera_size=args.camera_size,
                        search_step_deg=args.azimuth_search_step_deg,
                        min_visible_pixels=args.min_visible_pixels,
                    )
                    common_valid = chosen_azimuth is not None
                    if chosen_azimuth is None:
                        fallback = max(
                            candidates,
                            key=lambda row: row["worst_visible_pixel_count"],
                        )
                        chosen_azimuth = float(fallback["azimuth_deg"])
                    search_record = {
                        "seed": int(seed),
                        "mode": "automatic_common_ray",
                        "common_valid_for_all_distances": common_valid,
                        "chosen_azimuth_deg": chosen_azimuth,
                        "candidates": candidates,
                    }
                write_json(
                    run_dir / f"seed_{seed}_start_search.json", search_record
                )
                print(
                    f"[Start ray] seed={seed} azimuth={chosen_azimuth:.1f} deg "
                    f"mode={search_record['mode']}"
                )
            finally:
                if probe_env is not None:
                    probe_env.close()

            for distance in args.distances_m:
                label = f"seed_{seed}_distance_{distance:.2f}m".replace(".", "p")
                trial_dir = run_dir / "trials" / label
                trial_dir.mkdir(parents=True, exist_ok=False)
                trial: dict[str, Any] = {
                    "seed": int(seed),
                    "requested_distance_m": float(distance),
                    "prompt": args.prompt.strip(),
                    "target_object": args.target_object.strip(),
                    "common_start_azimuth_deg": float(chosen_azimuth),
                    "eligible": False,
                    "executed": False,
                    "success": False,
                    "trial_dir": str(trial_dir),
                }
                env = None
                video_started = False
                try:
                    print(f"[Trial] seed={seed} distance={distance:.2f}m")
                    env = create_pi05_env(
                        env_name=args.env_name,
                        layout_id=args.layout_id,
                        style_id=args.style_id,
                        seed=int(seed),
                        camera_size=args.camera_size,
                    )
                    observation = env.reset()
                    raw_env = _raw_task_env(env)
                    alias = args.target_object.strip()
                    target = _object_position(raw_env, alias)
                    desired = pose_on_target_ray(
                        target[:2],
                        float(distance),
                        chosen_azimuth,
                    )
                    _set_base_pose(raw_env, **desired)
                    zero_action = np.zeros(12, dtype=np.float32)
                    for _ in range(args.settle_steps):
                        observation, _, done, _ = env.step(zero_action)
                        if done:
                            raise RuntimeError("environment ended while settling initial pose")

                    pose = current_base_pose(env)
                    target = _object_position(raw_env, alias)
                    actual_distance = float(
                        np.linalg.norm(
                            np.asarray(pose["position"], dtype=float)[:2] - target[:2]
                        )
                    )
                    visibility = _visible_pixels(
                        raw_env,
                        alias=alias,
                        camera_size=args.camera_size,
                    )
                    collision = bool(detect_robot_collision(raw_env))
                    max_pixels = max(visibility.values(), default=0)
                    visible = max_pixels >= args.min_visible_pixels
                    eligible = (
                        (args.allow_invisible or visible)
                        and (args.allow_initial_collision or not collision)
                    )
                    trial.update(
                        {
                            "desired_start_pose": desired,
                            "actual_start_pose": pose,
                            "target_position": target.tolist(),
                            "actual_initial_distance_m": actual_distance,
                            "distance_error_m": actual_distance - float(distance),
                            "visible_pixels": visibility,
                            "visible_pixel_max": int(max_pixels),
                            "target_visible": bool(visible),
                            "initial_collision": collision,
                            "eligible": bool(eligible),
                        }
                    )
                    _save_views(trial_dir / "initial_views", observation)
                    if not eligible:
                        reasons = []
                        if not visible and not args.allow_invisible:
                            reasons.append("TARGET_NOT_VISIBLE")
                        if collision and not args.allow_initial_collision:
                            reasons.append("INITIAL_COLLISION")
                        trial["status"] = "invalid_start"
                        trial["failure_code"] = "+".join(reasons)
                        print(f"[Skip] {trial['failure_code']}")
                        continue

                    if not args.no_video:
                        env.start_episode_video_recording(
                            trial_dir / "episode.mp4",
                            camera_name="robot0_agentview_left",
                            fps=20,
                            frame_stride=args.video_skip,
                        )
                        video_started = True

                    trace = _PoseTrace(
                        env=env,
                        raw_env=raw_env,
                        object_alias=alias,
                        operation_region_max_m=args.operation_region_max_m,
                    )
                    env.add_observation_listener(trace)
                    env.add_observation_listener(client.observe)
                    client.environment_observer_attached = True
                    client.reset(observation)

                    natural_name = _object_natural_name(raw_env, alias)
                    call = _make_pick_call(
                        alias=alias,
                        natural_name=natural_name,
                        prompt=args.prompt.strip(),
                        trial_id=label,
                    )
                    verifier = RuntimeAtomicTaskVerifier(
                        default_required_consecutive_successes=1,
                        pick_min_lift_delta_m=(
                            args.pick_min_lift_m
                            if args.pick_min_lift_m > 0
                            else None
                        ),
                    )
                    adapter = RemoteAtomicTaskPolicyAdapter(
                        client=client,
                        verifier=verifier,
                        log_dir=trial_dir / "policy_rollout",
                        replan_steps=args.xr1_replan_steps,
                        atomic_task_horizon=args.horizon,
                        verify_interval=args.verify_interval,
                        min_steps_before_verify=args.min_steps_before_verify,
                        base_action_mode="full",
                        held_object_guard=True,
                        held_object_guard_action_override=True,
                        held_object_hold_confirmation_steps=args.pick_stable_steps,
                        held_object_drop_confirmation_steps=args.drop_confirmation_steps,
                        success_handoff_steps=args.pick_stable_steps,
                        verification_mode="periodic_early_stop",
                        held_object_drop_stops_rollout=False,
                        held_object_guard_pick_min_lift_m=(
                            args.pick_min_lift_m
                            if args.pick_min_lift_m > 0
                            else None
                        ),
                        render=True,
                        video_skip=args.video_skip,
                    )
                    trial["executed"] = True
                    result = adapter.execute(
                        env=env,
                        scheduler_query={"atomic_task_call": call.to_dict()},
                        episode_id=label,
                    )
                    metrics = trace.metrics(actual_distance)
                    rollout = result.get("rollout_logs", {})
                    trial.update(metrics)
                    trial.update(
                        {
                            "success": bool(result.get("success")),
                            "status": str(result.get("status", "unknown")),
                            "failure_code": (
                                None
                                if result.get("success")
                                else (
                                    rollout.get("Failure_Reason")
                                    or rollout.get("Termination_Reason")
                                    or "PICK_NOT_VERIFIED"
                                )
                            ),
                            "completed_steps": int(rollout.get("Horizon", len(trace.rows))),
                            "policy_result": result,
                            "xr1_diagnostics": client.diagnostics(),
                        }
                    )
                    write_json(trial_dir / "pose_trace.json", trace.rows)
                except Exception as exc:
                    trial.update(
                        {
                            "status": "error",
                            "failure_code": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(f"[Error] {label}: {type(exc).__name__}: {exc}")
                finally:
                    if env is not None and video_started:
                        try:
                            video = env.finalize_episode_video_recording()
                            trial["video"] = video
                            write_json(trial_dir / "full_episode_video.json", video)
                        except Exception as exc:
                            trial["video_finalize_error"] = str(exc)
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                    write_json(trial_dir / "result.json", trial)
                    trials.append(trial)
                    _write_csv(run_dir / "runs.csv", trials)
                    write_json(
                        run_dir / "summary.json",
                        {
                            "config": config,
                            "summary": summarize_trials(trials, args.distances_m),
                            "trials": trials,
                        },
                    )
    finally:
        client.close()

    summary = summarize_trials(trials, args.distances_m)
    print("[Summary]")
    for row in summary["by_distance"]:
        rate = row["success_rate_executed"]
        shown = "n/a" if rate is None else f"{100.0 * rate:.1f}%"
        print(
            f"  {row['distance_m']:.2f}m: "
            f"{row['successes']}/{row['executed_trials']} success ({shown}), "
            f"eligible={row['eligible_visible_trials']}/{row['total_trials']}"
        )
    print(f"[Output] {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
