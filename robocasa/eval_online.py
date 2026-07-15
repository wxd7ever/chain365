#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from io_utils import make_run_dir, save_obs_image, save_video, write_json
from policy_env import (
    POLICY_CAMERA_NAMES,
    build_env_override,
    create_env,
    get_policy_config,
    layout_style_pairs,
    load_policy,
)


PNIB_CHECKPOINT_ROOT = Path("/data/PNIB/checkpoints")
DEFAULT_PNIB_CKPT = (
    PNIB_CHECKPOINT_ROOT
    / "exp_stage2_online_demo_v3_refine_01_gt_angle_repair_micro"
    / "hybrid_online_best_rft_composite.pt"
)
DEFAULT_QFORMER_CKPT = PNIB_CHECKPOINT_ROOT / "instructblip_vicuna7b_visual_qformer.pt"
DEFAULT_PROCESSOR_DIR = PNIB_CHECKPOINT_ROOT / "instructblip_vicuna7b_processor"
DEFAULT_GLOBAL_NUM_POINTS = 16647
DEFAULT_FLOOR_NUM_POINTS = 10701
PNIB_INSTRUCTION_PREFIX = (
    "Given the image and the instruction, identify the target object and the "
    "affordance region relevant to completing the task. Task instruction: "
)
CLOSE_SINGLE_DOOR_MICROWAVE_LAYOUTS = {1, 4, 9}
CLOSE_SINGLE_DOOR_CABINET_LAYOUTS = {7, 8}
DEFAULT_PI05_HOST = "172.16.36.10"
DEFAULT_PI05_PORT = 8000
DEFAULT_PI05_SERVER_COMMIT = "5a6beda9ff99da30b4e1b59320f6a32971d7c397"
DEFAULT_PI05_POLICY_CONFIG = "pi05_pretrain_human300"
DEFAULT_PI05_CHECKPOINT = (
    "/data/wzh/checkpoints/pi05_pretrain_human300/multitask_learning/75000"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Online MoMa -> base navigation -> manipulation evaluation."
    )
    parser.add_argument("--env_name", type=str, default="CloseSingleDoor")
    parser.add_argument("--layout_id", type=int, default=-1)
    parser.add_argument("--style_id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1, help="Evaluation seed used for initial base pose sampling.")
    parser.add_argument("--num_episodes_per_layout", type=int, default=1)

    parser.add_argument("--ckpt_root_dir", type=str, default="/home/gzz1/mobipi/ckpts")
    parser.add_argument("--data_root_dir", type=str, default="/home/gzz1/mobipi/data")
    parser.add_argument(
        "--policy_name",
        type=str,
        default="bc_xfmr",
        help="Local robomimic checkpoint family used to bootstrap RoboCasa env metadata.",
    )
    parser.add_argument(
        "--policy_seed",
        type=int,
        default=1,
        help="Local robomimic checkpoint seed used to bootstrap RoboCasa env metadata.",
    )
    parser.add_argument("--dataset_name", type=str, default="mg-300")
    parser.add_argument(
        "--manipulation_policy",
        type=str,
        choices=["pi05", "robomimic"],
        default="pi05",
        help="Final manipulation policy. pi05 uses the remote OpenPI service.",
    )
    parser.add_argument("--pi05_host", type=str, default=DEFAULT_PI05_HOST)
    parser.add_argument("--pi05_port", type=int, default=DEFAULT_PI05_PORT)
    parser.add_argument(
        "--pi05_camera_size",
        type=int,
        default=256,
        help="Square render size before OpenPI resize_with_pad(..., 224, 224).",
    )
    parser.add_argument("--pi05_resize_size", type=int, default=224)
    parser.add_argument("--pi05_replan_steps", type=int, default=5)
    parser.add_argument(
        "--atomic_task_call_json",
        type=str,
        default=None,
        help="Optional scheduler query or AtomicTaskCall JSON to execute with the shared pi0.5 policy.",
    )
    parser.add_argument(
        "--long_horizon_task",
        type=str,
        default=None,
        help="Complex task text to decompose with the VLM and execute as atomic tasks.",
    )
    parser.add_argument(
        "--vlm_plan_json",
        type=str,
        default=None,
        help="Replay a saved VLM atomic task plan instead of calling the VLM.",
    )
    parser.add_argument(
        "--vlm_base_url",
        type=str,
        default="http://172.16.11.115:11434/v1",
    )
    parser.add_argument("--vlm_model", type=str, default="qwen2.5vl:3b")
    parser.add_argument("--vlm_api_key", type=str, default="ollama")
    parser.add_argument("--vlm_timeout_s", type=float, default=120.0)
    parser.add_argument(
        "--vlm_text_only",
        action="store_true",
        help="Do not send the three current RoboCasa camera images to the VLM.",
    )
    parser.add_argument("--pi05_atomic_task_horizon", type=int, default=300)
    parser.add_argument("--pi05_verify_interval", type=int, default=5)
    parser.add_argument("--pi05_min_steps_before_verify", type=int, default=10)
    parser.add_argument(
        "--pi05_base_action_mode",
        choices=["full", "residual", "frozen"],
        default="residual",
    )
    parser.add_argument("--pi05_base_residual_limit", type=float, default=0.15)
    parser.add_argument(
        "--skip_navigation",
        action="store_true",
        help="Skip PNIB/MoMa and execute the supplied atomic task from the reset state.",
    )
    parser.add_argument(
        "--pi05_horizon",
        type=int,
        default=None,
        help="Override manipulation horizon; defaults to the local RoboCasa rollout config.",
    )
    parser.add_argument("--pi05_connect_timeout_s", type=float, default=15.0)
    parser.add_argument("--pi05_infer_timeout_s", type=float, default=120.0)
    parser.add_argument("--pi05_max_retries", type=int, default=1)
    parser.add_argument("--pi05_video_skip", type=int, default=2)
    parser.add_argument("--no_pi05_video", action="store_true")
    parser.add_argument(
        "--pi05_server_commit",
        type=str,
        default=DEFAULT_PI05_SERVER_COMMIT,
        help="Provenance only: OpenPI commit expected on the inference server.",
    )
    parser.add_argument(
        "--pi05_policy_config",
        type=str,
        default=DEFAULT_PI05_POLICY_CONFIG,
        help="Provenance only: OpenPI policy config expected on the inference server.",
    )
    parser.add_argument(
        "--pi05_checkpoint",
        type=str,
        default=DEFAULT_PI05_CHECKPOINT,
        help="Provenance only: checkpoint path expected on the inference server.",
    )

    parser.add_argument("--moma_model_name", type=str, default="MOMA", choices=["MOMA", "PNIB"])
    parser.add_argument("--moma_model_path", type=str, default="/home/gzz1/MoMaKitchen/runs/train_Yaw_no_drawer/5.pt")
    parser.add_argument("--moma_config_path", type=str, default="/home/gzz1/MoMaKitchen/conf/config.yaml")
    parser.add_argument("--moma_camera", type=str, default="robot0_agentview_right")
    parser.add_argument("--moma_render_size", type=int, default=640)
    parser.add_argument("--target_class", type=str, default=None)
    parser.add_argument(
        "--disable_close_single_door_layout_targets",
        action="store_true",
        help="Disable CloseSingleDoor layout-aware target override.",
    )
    parser.add_argument(
        "--close_single_door_cabinet_target_classes",
        type=str,
        default="SingleCabinet|HingeCabinet",
        help="Candidate segmentation classes for CloseSingleDoor cabinet layouts 7 and 8.",
    )
    parser.add_argument(
        "--close_single_door_cabinet_instruction_target",
        type=str,
        default="cabinet door",
        help="Target phrase used in PNIB instruction for CloseSingleDoor cabinet layouts 7 and 8.",
    )
    parser.add_argument("--pnib_model_path", type=str, default=str(DEFAULT_PNIB_CKPT))
    parser.add_argument("--pnib_qformer_ckpt", type=str, default=str(DEFAULT_QFORMER_CKPT))
    parser.add_argument("--pnib_processor_dir", type=str, default=str(DEFAULT_PROCESSOR_DIR))
    parser.add_argument(
        "--pnib_instruction",
        type=str,
        default=None,
        help="Language instruction for PNIB. If omitted, it is derived from env_name and target_class.",
    )
    parser.add_argument("--pnib_robot_h", type=float, default=0.85)
    parser.add_argument("--pnib_robot_w", type=float, default=1.03)
    parser.add_argument(
        "--pnib_input_coord",
        type=str,
        default="train_camera",
        choices=["train_camera", "realsense_camera"],
        help="Coordinate convention of the point clouds before PNIB preprocessing.",
    )
    parser.add_argument(
        "--pnib_pc_already_normalized",
        action="store_true",
        help="Set only when PNIB point-cloud xyz is already normalized with global centroid/scale.",
    )
    parser.add_argument("--pnib_max_global_points", type=int, default=DEFAULT_GLOBAL_NUM_POINTS)
    parser.add_argument("--pnib_max_floor_points", type=int, default=DEFAULT_FLOOR_NUM_POINTS)
    parser.add_argument("--pnib_valid_threshold", type=float, default=0.5)
    parser.add_argument("--pnib_amp", action="store_true")
    parser.add_argument(
        "--disable_target_centering",
        action="store_true",
        help="Skip pre-inference base yaw scan that centers the target in the MoMa render.",
    )
    parser.add_argument("--target_center_range", type=float, nargs=2, default=(0.20, 0.80))
    parser.add_argument("--target_center_min_pixels", type=int, default=1)
    parser.add_argument("--target_center_step_degrees", type=float, default=10.0)
    parser.add_argument("--target_center_max_degrees", type=float, default=360.0)
    parser.add_argument(
        "--show_robot_in_moma_render",
        action="store_true",
        help="Do not hide robot geoms during the temporary MoMa RGB/depth/seg render.",
    )

    parser.add_argument("--output_root", type=str, default=str(THIS_DIR / "outputs"))
    parser.add_argument("--no_nav_video", action="store_true")
    parser.add_argument(
        "--clear_frame_stack_before_manip",
        action="store_true",
        help="Fill FrameStack with the current post-navigation observation before manipulation. "
        "Default keeps the original Mobipi behavior; pi05 always consumes only the newest frame.",
    )
    args = parser.parse_args()
    for name in ("pi05_camera_size", "pi05_resize_size", "pi05_replan_steps", "pi05_video_skip"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.pi05_horizon is not None and args.pi05_horizon <= 0:
        parser.error("--pi05_horizon must be positive when provided")
    if args.pi05_atomic_task_horizon <= 0:
        parser.error("--pi05_atomic_task_horizon must be positive")
    if args.pi05_verify_interval <= 0:
        parser.error("--pi05_verify_interval must be positive")
    if args.pi05_min_steps_before_verify < 0:
        parser.error("--pi05_min_steps_before_verify must be non-negative")
    if not math.isfinite(args.pi05_base_residual_limit) or args.pi05_base_residual_limit < 0:
        parser.error("--pi05_base_residual_limit must be finite and non-negative")
    if args.pi05_connect_timeout_s <= 0 or args.pi05_infer_timeout_s <= 0:
        parser.error("--pi05_connect_timeout_s and --pi05_infer_timeout_s must be positive")
    if args.pi05_max_retries < 0:
        parser.error("--pi05_max_retries must be non-negative")
    if args.vlm_timeout_s <= 0:
        parser.error("--vlm_timeout_s must be positive")
    long_horizon_mode = bool(args.long_horizon_task or args.vlm_plan_json)
    if args.atomic_task_call_json is not None and long_horizon_mode:
        parser.error(
            "--atomic_task_call_json cannot be combined with --long_horizon_task or --vlm_plan_json"
        )
    if args.atomic_task_call_json is not None:
        if args.manipulation_policy != "pi05":
            parser.error("--atomic_task_call_json requires --manipulation_policy pi05")
        if not Path(args.atomic_task_call_json).is_file():
            parser.error(f"--atomic_task_call_json does not exist: {args.atomic_task_call_json}")
    if args.vlm_plan_json is not None and not Path(args.vlm_plan_json).is_file():
        parser.error(f"--vlm_plan_json does not exist: {args.vlm_plan_json}")
    if long_horizon_mode and args.manipulation_policy != "pi05":
        parser.error("VLM long-horizon execution requires --manipulation_policy pi05")
    if args.skip_navigation and args.atomic_task_call_json is None and not long_horizon_mode:
        parser.error(
            "--skip_navigation requires --atomic_task_call_json, --long_horizon_task, or --vlm_plan_json"
        )
    return args


def save_prediction(path: Path, prediction) -> None:
    line = (
        f"{prediction.x_world:.6f},{prediction.y_world:.6f},{prediction.yaw_world:.8f},"
        f"{prediction.z_world:.6f},{prediction.score:.8f},{math.degrees(prediction.yaw_world):.4f}\n"
    )
    path.write_text(line, encoding="utf-8")


def layout_dir_name(layout_id: int) -> str:
    return f"layout{layout_id:02d}"


def save_final_policy_views(env, episode_dir: Path) -> dict[str, str]:
    if hasattr(env, "_get_stacked_obs_from_history"):
        obs = env._get_stacked_obs_from_history()
    else:
        obs = env.get_observation()

    view_dir = episode_dir / "final_policy_views"
    saved = {}
    for camera_name in POLICY_CAMERA_NAMES:
        key = f"{camera_name}_image"
        if key not in obs:
            continue
        path = view_dir / f"{camera_name}.png"
        save_obs_image(path, obs[key])
        saved[key] = str(path)
    return saved


def clear_frame_stack_with_current_obs(env) -> None:
    if not hasattr(env, "_get_initial_obs_history") or not hasattr(env, "env"):
        raise RuntimeError("Current env is not a FrameStackWrapper-compatible object")
    current_obs = env.env.get_observation()
    env.obs_history = env._get_initial_obs_history(current_obs)


def execute_atomic_task_from_json(
    *,
    env,
    client,
    args,
    episode_dir: Path,
    episode_id: int,
) -> dict:
    """Load one scheduler call, execute it, and persist atomic-task artifacts."""

    from atomic_task_policy_adapter import RemoteAtomicTaskPolicyAdapter
    from atomic_task_prompt_builder import build_atomic_task_prompt
    from atomic_task_schemas import AtomicTaskCall
    from atomic_task_verifier import RuntimeAtomicTaskVerifier
    from robust_vlm_task_planner import prepare_execution_plan

    source_path = Path(args.atomic_task_call_json)
    loaded = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("atomic task JSON must contain an object")
    scheduler_query = (
        loaded if "atomic_task_call" in loaded else {"atomic_task_call": loaded}
    )
    call = AtomicTaskCall.from_mapping(scheduler_query["atomic_task_call"])
    prepared, _ = prepare_execution_plan(
        [call], _current_scene_context(env, args)
    )
    if len(prepared) != 1:
        raise ValueError("one scheduler call must prepare to exactly one atomic task")
    call = prepared[0]
    scheduler_query = {"atomic_task_call": call.to_dict()}

    prompt = build_atomic_task_prompt(call)
    write_json(episode_dir / "atomic_task_call.json", call.to_dict())
    (episode_dir / "atomic_task_prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    adapter = RemoteAtomicTaskPolicyAdapter(
        client=client,
        verifier=RuntimeAtomicTaskVerifier(),
        log_dir=episode_dir,
        resize_size=args.pi05_resize_size,
        replan_steps=args.pi05_replan_steps,
        atomic_task_horizon=args.pi05_atomic_task_horizon,
        verify_interval=args.pi05_verify_interval,
        min_steps_before_verify=args.pi05_min_steps_before_verify,
        base_action_mode=args.pi05_base_action_mode,
        base_residual_limit=args.pi05_base_residual_limit,
        render=not args.no_pi05_video,
        video_skip=args.pi05_video_skip,
    )
    result = adapter.execute(
        env=env,
        scheduler_query=scheduler_query,
        episode_id=episode_id,
    )
    write_json(episode_dir / "atomic_task_result.json", result)
    return result


def _current_vlm_images(env) -> list[np.ndarray]:
    """Reuse pi0.5's FrameStack-aware camera conversion for VLM planning."""

    from pi05_rollout import _current_observation, _image_to_hwc_uint8

    observation = _current_observation(env)
    keys = (
        "robot0_agentview_left_image",
        "robot0_eye_in_hand_image",
        "robot0_agentview_right_image",
    )
    images = []
    for key in keys:
        if key not in observation:
            raise KeyError(f"Missing required VLM planning camera observation {key!r}")
        images.append(_image_to_hwc_uint8(observation[key], key=key))
    return images


def _current_scene_context(env, args) -> dict:
    """Expose simulator entity aliases so the VLM can emit verifiable predicates."""

    owners = []
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        owners.append(current)
        if hasattr(current, "unwrapped") and current.unwrapped is not current:
            current = current.unwrapped
        elif hasattr(current, "env"):
            current = current.env
        else:
            break
    task_env = next(
        (
            owner
            for owner in reversed(owners)
            if hasattr(owner, "fixtures") and hasattr(owner, "objects")
        ),
        owners[-1],
    )
    fixtures = []
    for alias, fixture in getattr(task_env, "fixtures", {}).items():
        fixtures.append(
            {
                "alias": str(alias),
                "name": str(getattr(fixture, "name", alias)),
                "natural_name": str(getattr(fixture, "nat_lang", alias)),
                "type": type(fixture).__name__,
            }
        )
    objects = []
    for alias, obj in getattr(task_env, "objects", {}).items():
        natural_name = str(getattr(obj, "name", alias))
        get_obj_lang = getattr(task_env, "get_obj_lang", None)
        if callable(get_obj_lang):
            try:
                natural_name = str(get_obj_lang(str(alias)))
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
        objects.append(
            {
                "alias": str(alias),
                "name": str(getattr(obj, "name", alias)),
                "natural_name": natural_name,
                "type": type(obj).__name__,
            }
        )
    return {
        "env_name": args.env_name,
        "layout_id": args.layout_id,
        "style_id": args.style_id,
        "instruction": args.long_horizon_task,
        "fixtures": fixtures,
        "objects": objects,
    }


def execute_long_horizon_task(
    *,
    env,
    client,
    args,
    episode_dir: Path,
    episode_id: int,
) -> dict:
    """Plan once with the VLM, then execute every call in the same environment."""

    from atomic_task_policy_adapter import RemoteAtomicTaskPolicyAdapter
    from atomic_task_verifier import RuntimeAtomicTaskVerifier
    from orchestrator import RoboCasaOrchestrator
    from robust_vlm_task_planner import prepare_execution_plan
    from vlm_task_planner import OpenAICompatibleVLMPlanner, load_vlm_task_plan

    scene_context = _current_scene_context(env, args)
    if args.vlm_plan_json:
        calls = load_vlm_task_plan(args.vlm_plan_json)
        planner_provenance = {
            "source": "saved_plan",
            "path": str(Path(args.vlm_plan_json).resolve()),
        }
    else:
        planner = OpenAICompatibleVLMPlanner(
            base_url=args.vlm_base_url,
            model=args.vlm_model,
            api_key=args.vlm_api_key,
            timeout_s=args.vlm_timeout_s,
            include_images=not args.vlm_text_only,
        )
        images = None if args.vlm_text_only else _current_vlm_images(env)
        try:
            calls, planner_provenance = planner.plan(
                task=args.long_horizon_task,
                images=images,
                scene_context=scene_context,
            )
        except (RuntimeError, TypeError, ValueError):
            if planner.last_response_text:
                (episode_dir / "vlm_raw_response.txt").write_text(
                    planner.last_response_text + "\n", encoding="utf-8"
                )
            raise
        (episode_dir / "vlm_raw_response.txt").write_text(
            (planner.last_response_text or "") + "\n", encoding="utf-8"
        )

    calls, normalizations = prepare_execution_plan(calls, scene_context)
    if normalizations:
        planner_provenance["plan_normalizations"] = normalizations

    task_plan = {
        "long_horizon_task": args.long_horizon_task,
        "planner": planner_provenance,
        "atomic_task_calls": [call.to_dict() for call in calls],
    }
    write_json(episode_dir / "task_plan.json", task_plan)

    adapter = RemoteAtomicTaskPolicyAdapter(
        client=client,
        verifier=RuntimeAtomicTaskVerifier(),
        log_dir=episode_dir,
        resize_size=args.pi05_resize_size,
        replan_steps=args.pi05_replan_steps,
        atomic_task_horizon=args.pi05_atomic_task_horizon,
        verify_interval=args.pi05_verify_interval,
        min_steps_before_verify=args.pi05_min_steps_before_verify,
        base_action_mode=args.pi05_base_action_mode,
        base_residual_limit=args.pi05_base_residual_limit,
        render=not args.no_pi05_video,
        video_skip=args.pi05_video_skip,
    )
    orchestrator = RoboCasaOrchestrator(atomic_task_policy_adapter=adapter)
    result = orchestrator.run_task_plan(
        env=env,
        task_plan=calls,
        episode_id=episode_id,
        stop_on_unsuccessful=True,
    )
    result["long_horizon_task"] = args.long_horizon_task
    result["task_plan_path"] = str(episode_dir / "task_plan.json")
    write_json(episode_dir / "long_horizon_result.json", result)
    return result


def _words_from_camel(name: str) -> list[str]:
    return re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", name or "")


def _task_phrase_from_env(env_name: str, target_name: str | None) -> str:
    words = _words_from_camel(env_name)
    lowered = [w.lower() for w in words]
    target = (target_name or "").replace("_", " ").strip().lower()

    action = None
    rest = lowered
    if lowered[:2] == ["turn", "on"]:
        action = "turn on"
        rest = lowered[2:]
    elif lowered[:2] == ["turn", "off"]:
        action = "turn off"
        rest = lowered[2:]
    elif lowered:
        action = lowered[0]
        rest = lowered[1:]

    if not action:
        action = "complete"
    if not target:
        target = " ".join(rest).strip() or "target object"

    return f"{action} the {target}"


def resolve_episode_target(args, layout_id: int) -> dict[str, str | None]:
    target_class = args.target_class
    instruction_target = args.target_class
    source = "target_class"

    if (
        args.env_name.lower() == "closesingledoor"
        and not args.disable_close_single_door_layout_targets
    ):
        if int(layout_id) in CLOSE_SINGLE_DOOR_MICROWAVE_LAYOUTS:
            target_class = "Microwave"
            instruction_target = "microwave door"
            source = "close_single_door_layout_rule"
        elif int(layout_id) in CLOSE_SINGLE_DOOR_CABINET_LAYOUTS:
            target_class = args.close_single_door_cabinet_target_classes
            instruction_target = args.close_single_door_cabinet_instruction_target
            source = "close_single_door_layout_rule"

    return {
        "target_class": target_class,
        "instruction_target": instruction_target,
        "source": source,
    }


def build_pnib_instruction(
    args,
    moma_obs=None,
    target_class: str | None = None,
    instruction_target: str | None = None,
) -> str:
    if args.pnib_instruction and args.pnib_instruction.strip():
        return args.pnib_instruction.strip()

    target_name = instruction_target or target_class or args.target_class
    if not target_name and moma_obs is not None:
        candidates = getattr(moma_obs, "target_candidates", None) or []
        if candidates:
            target_name = candidates[0]

    return PNIB_INSTRUCTION_PREFIX + _task_phrase_from_env(args.env_name, target_name) + "."


def build_moma_predictor(args):
    if args.moma_model_name == "PNIB":
        from pnib_predictor import PNIBPredictor

        return PNIBPredictor(
            model_path=args.pnib_model_path,
            qformer_ckpt_path=args.pnib_qformer_ckpt,
            processor_dir=args.pnib_processor_dir,
            robot_hw=[args.pnib_robot_h, args.pnib_robot_w],
            input_coord=args.pnib_input_coord,
            pc_already_normalized=args.pnib_pc_already_normalized,
            max_global_points=args.pnib_max_global_points,
            max_floor_points=args.pnib_max_floor_points,
            valid_threshold=args.pnib_valid_threshold,
            amp=args.pnib_amp,
        )
    from moma_predictor import MomaPredictor

    return MomaPredictor(
        model_path=args.moma_model_path,
        config_path=args.moma_config_path,
    )


def close_env(env) -> None:
    try:
        env.env.env.close()
    except Exception:
        try:
            env.close()
        except Exception:
            pass


def run_episode(
    *,
    args,
    config,
    rollout_model,
    pi05_client,
    moma_predictor,
    layout_id: int,
    style_id: int,
    episode_index: int,
    episode_index_in_layout: int,
    run_dir: Path,
) -> dict:
    episode_seed = args.seed * 10000 + episode_index_in_layout
    layout_dir = run_dir / layout_dir_name(layout_id)
    episode_dir = layout_dir / (
        f"ep{episode_index_in_layout:04d}_style{style_id:02d}_seed{episode_seed}"
    )
    episode_dir.mkdir(parents=True, exist_ok=True)

    env = None
    try:
        camera_size = args.pi05_camera_size if args.manipulation_policy == "pi05" else 128
        override = build_env_override(
            layout_id=layout_id,
            style_id=style_id,
            camera_size=camera_size,
        )
        env = create_env(config, override)
        env.unwrapped.env.place_robot_for_nav_rng = np.random.default_rng(episode_seed)

        print(f"[Episode {episode_index}] reset layout={layout_id} style={style_id} rng={episode_seed}")
        env.reset()

        long_horizon_mode = bool(args.long_horizon_task or args.vlm_plan_json)
        if args.skip_navigation or long_horizon_mode:
            final_views = save_final_policy_views(env, episode_dir)
            if args.clear_frame_stack_before_manip:
                clear_frame_stack_with_current_obs(env)
            if pi05_client is None:
                raise RuntimeError("pi0.5 task execution selected but no policy client is available")
            if long_horizon_mode:
                long_horizon_result = execute_long_horizon_task(
                    env=env,
                    client=pi05_client,
                    args=args,
                    episode_dir=episode_dir,
                    episode_id=episode_index_in_layout,
                )
                atomic_result = None
                execution_result = long_horizon_result
            else:
                atomic_result = execute_atomic_task_from_json(
                    env=env,
                    client=pi05_client,
                    args=args,
                    episode_dir=episode_dir,
                    episode_id=episode_index_in_layout,
                )
                long_horizon_result = None
                execution_result = atomic_result
            success = bool(execution_result["success"])
            marker = episode_dir / (
                f"ep{episode_index_in_layout:04d}_{'s' if success else 'f'}.txt"
            )
            marker.touch()
            summary = {
                "episode_index": episode_index,
                "episode_index_in_layout": episode_index_in_layout,
                "layout_id": layout_id,
                "style_id": style_id,
                "episode_seed": episode_seed,
                "success": float(success),
                "success_bool": success,
                "prediction": None,
                "target_alignment": {"skipped": True},
                "navigation": {"skipped": True},
                "manipulation_policy": args.manipulation_policy,
                "atomic_task_result": atomic_result,
                "long_horizon_result": long_horizon_result,
                "rollout_logs": (
                    atomic_result["rollout_logs"] if atomic_result is not None else None
                ),
                "final_policy_views": final_views,
                "frame_stack_cleared_before_manip": bool(
                    args.clear_frame_stack_before_manip
                ),
                "robot_hidden_for_moma_render": None,
            }
            write_json(episode_dir / "episode_summary.json", summary)
            print(
                f"[Episode {episode_index}] task execution ended with "
                f"[{'success' if success else execution_result['status']}]"
            )
            return summary

        effective_target = resolve_episode_target(args, layout_id)
        print(
            f"[Episode {episode_index}] target_class={effective_target['target_class']} "
            f"source={effective_target['source']}"
        )
        from moma_observation import save_observation_artifacts
        from navigation import navigate_to_world_pose
        from target_alignment import capture_aligned_moma_observation
        from visualization import save_prediction_topdown

        moma_obs, target_alignment = capture_aligned_moma_observation(
            env=env,
            env_name=args.env_name,
            camera_name=args.moma_camera,
            width=args.moma_render_size,
            height=args.moma_render_size,
            target_class=effective_target["target_class"],
            seed=episode_seed,
            hide_robot=not args.show_robot_in_moma_render,
            enabled=not args.disable_target_centering,
            center_range=args.target_center_range,
            min_pixels=args.target_center_min_pixels,
            step_degrees=args.target_center_step_degrees,
            max_degrees=args.target_center_max_degrees,
        )
        target_alignment["effective_target"] = effective_target
        write_json(episode_dir / "target_alignment.json", target_alignment)
        if not args.disable_target_centering and not target_alignment.get("success", False):
            raise RuntimeError(
                "Target was not visible in the required center range before MoMa inference. "
                f"See {episode_dir / 'target_alignment.json'}"
            )
        save_observation_artifacts(episode_dir, moma_obs)

        pnib_instruction = (
            build_pnib_instruction(
                args,
                moma_obs,
                target_class=effective_target["target_class"],
                instruction_target=effective_target["instruction_target"],
            )
            if args.moma_model_name == "PNIB"
            else None
        )
        prediction = moma_predictor.predict(
            global_points_cam=moma_obs.global_points_cam,
            floor_points_cam=moma_obs.floor_points_cam,
            target_points_cam=moma_obs.target_points_cam,
            T_camera_to_world=moma_obs.T_camera_to_world,
            seed=episode_seed,
            rgb=moma_obs.rgb,
            global_colors=moma_obs.global_colors,
            floor_colors=moma_obs.floor_colors,
            instruction=pnib_instruction,
        )
        save_prediction(episode_dir / "pred_result_world.txt", prediction)
        prediction_dict = prediction.to_dict()
        predictor_debug = getattr(moma_predictor, "last_debug", None)
        if predictor_debug:
            prediction_dict["predictor_debug"] = predictor_debug
        prediction_dict["effective_target"] = effective_target
        write_json(episode_dir / "prediction.json", prediction_dict)
        save_prediction_topdown(
            episode_dir / "prediction_topdown.png",
            global_points_cam=moma_obs.global_points_cam,
            floor_points_cam=moma_obs.floor_points_cam,
            T_camera_to_world=moma_obs.T_camera_to_world,
            target_pose=prediction.target_pose,
            score=prediction.score,
            global_colors=moma_obs.global_colors,
            floor_colors=moma_obs.floor_colors,
        )
        print(
            f"[Episode {episode_index}] {args.moma_model_name} world target="
            f"({prediction.x_world:.3f}, {prediction.y_world:.3f}, "
            f"{math.degrees(prediction.yaw_world):.1f} deg), score={prediction.score:.4f}"
        )

        nav_info = navigate_to_world_pose(
            env,
            prediction.target_pose,
            render=not args.no_nav_video,
            verbose=False,
        )
        if not args.no_nav_video:
            save_video(episode_dir / "navigation.mp4", nav_info.get("images", []), fps=20)
        nav_info_for_json = dict(nav_info)
        nav_info_for_json.pop("images", None)
        write_json(episode_dir / "navigation.json", nav_info_for_json)

        final_views = save_final_policy_views(env, episode_dir)
        if args.clear_frame_stack_before_manip:
            clear_frame_stack_with_current_obs(env)

        atomic_result = None
        if args.atomic_task_call_json is not None:
            if pi05_client is None:
                raise RuntimeError("atomic pi0.5 task selected but no policy client is available")
            atomic_result = execute_atomic_task_from_json(
                env=env,
                client=pi05_client,
                args=args,
                episode_dir=episode_dir,
                episode_id=episode_index_in_layout,
            )
            success = bool(atomic_result["success"])
            rollout_logs = atomic_result["rollout_logs"]
        elif args.manipulation_policy == "pi05":
            if pi05_client is None:
                raise RuntimeError("pi05 manipulation policy selected but no policy client is available")
            from pi05_rollout import execute_pi05_manipulation_policy

            horizon = args.pi05_horizon or int(config.experiment.rollout.horizon)
            success, rollout_logs = execute_pi05_manipulation_policy(
                env=env,
                client=pi05_client,
                log_dir=str(episode_dir),
                episode_id=episode_index_in_layout,
                reset_before_rollout=False,
                horizon=horizon,
                replan_steps=args.pi05_replan_steps,
                resize_size=args.pi05_resize_size,
                render=not args.no_pi05_video,
                video_skip=args.pi05_video_skip,
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )
        else:
            from manipulation_rollout import execute_manipulation_policy

            success, rollout_logs = execute_manipulation_policy(
                env=env,
                rollout_model=rollout_model,
                config=config,
                log_dir=str(episode_dir),
                episode_id=episode_index_in_layout,
                reset_before_rollout=False,
            )
        success_bool = bool(success)
        marker = episode_dir / f"ep{episode_index_in_layout:04d}_{'s' if success_bool else 'f'}.txt"
        marker.touch()
        print(
            f"[Episode {episode_index}] ended with "
            f"[{'success' if success_bool else 'failure'}]. success={success:.3f}"
        )

        summary = {
            "episode_index": episode_index,
            "episode_index_in_layout": episode_index_in_layout,
            "layout_id": layout_id,
            "style_id": style_id,
            "episode_seed": episode_seed,
            "success": success,
            "success_bool": success_bool,
            "prediction": prediction_dict,
            "effective_target": effective_target,
            "target_alignment": target_alignment,
            "navigation": nav_info_for_json,
            "manipulation_policy": args.manipulation_policy,
            "atomic_task_result": atomic_result,
            "rollout_logs": rollout_logs,
            "final_policy_views": final_views,
            "frame_stack_cleared_before_manip": bool(args.clear_frame_stack_before_manip),
            "robot_hidden_for_moma_render": not bool(args.show_robot_in_moma_render),
        }
        write_json(episode_dir / "episode_summary.json", summary)
        return summary

    except Exception as exc:
        error = {
            "episode_index": episode_index,
            "episode_index_in_layout": episode_index_in_layout,
            "layout_id": layout_id,
            "style_id": style_id,
            "episode_seed": episode_seed,
            "manipulation_policy": args.manipulation_policy,
            "success": 0.0,
            "success_bool": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(episode_dir / "episode_error.json", error)
        print(f"[Episode {episode_index}] ERROR: {exc}")
        print(traceback.format_exc())
        return error
    finally:
        if env is not None:
            close_env(env)


def main():
    args = parse_args()
    task_name = f"{args.env_name}_{args.moma_model_name}_{args.manipulation_policy}"
    run_dir = make_run_dir(args.output_root, env_name=task_name)
    logged_args = dict(vars(args))
    if logged_args.get("vlm_api_key"):
        logged_args["vlm_api_key"] = "***"
    write_json(run_dir / "config.json", logged_args)

    config, ckpt_path = get_policy_config(
        ckpt_root_dir=args.ckpt_root_dir,
        data_root_dir=args.data_root_dir,
        env_name=args.env_name,
        method_name=args.policy_name,
        seed=args.policy_seed,
        dataset_name=args.dataset_name,
    )
    rollout_model = None
    pi05_client = None
    pairs = layout_style_pairs(args.layout_id, args.style_id)
    summaries = []
    episode_index = 0
    try:
        if args.manipulation_policy == "pi05":
            from openpi_client import OpenPIWebsocketClient

            pi05_client = OpenPIWebsocketClient(
                host=args.pi05_host,
                port=args.pi05_port,
                connect_timeout_s=args.pi05_connect_timeout_s,
                infer_timeout_s=args.pi05_infer_timeout_s,
                max_retries=args.pi05_max_retries,
            )
            server_metadata = pi05_client.server_metadata
            write_json(
                run_dir / "pi05_server.json",
                {
                    "uri": pi05_client.uri,
                    "server_metadata": server_metadata,
                    "server_commit": args.pi05_server_commit,
                    "policy_config": args.pi05_policy_config,
                    "checkpoint": args.pi05_checkpoint,
                },
            )
            print(
                f"[Pi05] Connected to {pi05_client.uri} "
                f"config={args.pi05_policy_config} metadata={server_metadata}"
            )
        else:
            _, rollout_model, _ = load_policy(config, ckpt_path)

        skip_moma = bool(
            args.skip_navigation or args.long_horizon_task or args.vlm_plan_json
        )
        moma_predictor = None if skip_moma else build_moma_predictor(args)
        for layout_id, style_id in pairs:
            for episode_index_in_layout in range(args.num_episodes_per_layout):
                summaries.append(
                    run_episode(
                        args=args,
                        config=config,
                        rollout_model=rollout_model,
                        pi05_client=pi05_client,
                        moma_predictor=moma_predictor,
                        layout_id=layout_id,
                        style_id=style_id,
                        episode_index=episode_index,
                        episode_index_in_layout=episode_index_in_layout,
                        run_dir=run_dir,
                    )
                )
                episode_index += 1
    finally:
        if pi05_client is not None:
            pi05_client.close()

    successes = [float(s.get("success", 0.0)) for s in summaries]
    layout_stats = {}
    for layout_id, _ in pairs:
        layout_successes = [
            float(s.get("success", 0.0))
            for s in summaries
            if int(s.get("layout_id", -1)) == int(layout_id)
        ]
        layout_stats[layout_dir_name(layout_id)] = {
            "layout_id": layout_id,
            "num_episodes": len(layout_successes),
            "num_success": float(np.sum(layout_successes)) if layout_successes else 0.0,
            "success_rate": float(np.mean(layout_successes)) if layout_successes else 0.0,
        }
    summary = {
        "run_dir": str(run_dir),
        "num_episodes": len(summaries),
        "num_success": float(np.sum(successes)) if successes else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "layout_stats": layout_stats,
        "episodes": summaries,
    }
    write_json(run_dir / "summary.json", summary)
    print(f"[Done] success_rate={summary['success_rate']:.3f} output={run_dir}")


if __name__ == "__main__":
    main()
