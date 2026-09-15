#!/usr/bin/env python3
"""Minimal RoboCasa VLM-plan -> shared remote pi0.5 execution entrypoint."""

from __future__ import annotations

import argparse
import inspect
import json
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .atomic_task_policy_adapter import RemoteAtomicTaskPolicyAdapter
    from .atomic_task_verifier import RuntimeAtomicTaskVerifier
    from .grounding import RoboCasaGrounder
    from .io_utils import make_run_dir, save_obs_image, write_json
    from .local_work_pose_refiner import (
        LocalWorkPoseRefiner,
        OpenAICompatibleLocalPoseVLM,
    )
    from .multi_view_perception import MultiViewObservationProvider
    from .navigation import TargetAwareNavigationPipeline, TargetAwarePolicyAdapter
    from .navigation.navigation_pipeline import (
        DEFAULT_CONFIG_PATH as DEFAULT_NAVIGATION_CONFIG,
        load_navigation_config,
    )
    from .openpi_client import OpenPIWebsocketClient
    from .oracle_logging import OracleLoggingGrounder, OracleLoggingVerifier
    from .orchestrator import RoboCasaOrchestrator
    from .pi05_env import POLICY_CAMERA_NAMES, create_pi05_env
    from .pi05_rollout import _image_to_hwc_uint8
    from .robust_vlm_task_planner import (
        RobustOpenAICompatibleVLMPlanner,
        prepare_execution_plan,
    )
    from .view_scheduler import (
        DEFAULT_VIEW_SELECTION_CONFIG,
        FixedViewScheduler,
    )
    from .vlm_atomic_task_verifier import VLMAtomicTaskVerifier
    from .vlm_client import OpenAICompatibleVisionClient
    from .vlm_grounding import VLMRoboCasaGrounder
    from .vlm_task_planner import load_vlm_task_plan
except ImportError:  # Support ``python robocasa/eval_vlm_pi05.py``.
    from atomic_task_policy_adapter import RemoteAtomicTaskPolicyAdapter
    from atomic_task_verifier import RuntimeAtomicTaskVerifier
    from grounding import RoboCasaGrounder
    from io_utils import make_run_dir, save_obs_image, write_json
    from local_work_pose_refiner import (
        LocalWorkPoseRefiner,
        OpenAICompatibleLocalPoseVLM,
    )
    from multi_view_perception import MultiViewObservationProvider
    from navigation import TargetAwareNavigationPipeline, TargetAwarePolicyAdapter
    from navigation.navigation_pipeline import (
        DEFAULT_CONFIG_PATH as DEFAULT_NAVIGATION_CONFIG,
        load_navigation_config,
    )
    from openpi_client import OpenPIWebsocketClient
    from oracle_logging import OracleLoggingGrounder, OracleLoggingVerifier
    from orchestrator import RoboCasaOrchestrator
    from pi05_env import POLICY_CAMERA_NAMES, create_pi05_env
    from pi05_rollout import _image_to_hwc_uint8
    from robust_vlm_task_planner import (
        RobustOpenAICompatibleVLMPlanner,
        prepare_execution_plan,
    )
    from view_scheduler import (
        DEFAULT_VIEW_SELECTION_CONFIG,
        FixedViewScheduler,
    )
    from vlm_atomic_task_verifier import VLMAtomicTaskVerifier
    from vlm_client import OpenAICompatibleVisionClient
    from vlm_grounding import VLMRoboCasaGrounder
    from vlm_task_planner import load_vlm_task_plan


THIS_DIR = Path(__file__).resolve().parent
LEGACY_PRIVILEGED_MAPPED_NAVIGATION_CONFIG = (
    DEFAULT_NAVIGATION_CONFIG.with_name(
        "navigation_legacy_privileged_mapped.yaml"
    )
)
LEGACY_PRIVILEGED_MAPPED_PROFILE = "steam_in_microwave_4_mapped_navigation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create RoboCasa directly, decompose a long task with a VLM, and execute "
            "the atomic plan through one shared remote pi0.5 connection."
        )
    )
    parser.add_argument("--env_name", default="PickPlaceCounterToMicrowave")
    parser.add_argument(
        "--system_profile",
        choices=("current", LEGACY_PRIVILEGED_MAPPED_PROFILE),
        default="current",
        help=(
            "Select current behavior or reproduce the privileged "
            "SteamInMicrowave_4 runtime while replacing only navigation with "
            "occupancy-map candidate planning and A* execution."
        ),
    )
    parser.add_argument("--layout_id", type=int, default=1)
    parser.add_argument("--style_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--camera_size", type=int, default=256)
    parser.add_argument("--long_horizon_task", required=True)
    parser.add_argument("--vlm_plan_json", type=Path)
    parser.add_argument("--vlm_base_url", default="http://172.16.11.115:11434/v1")
    parser.add_argument("--vlm_model", default="qwen2.5vl:3b")
    parser.add_argument("--vlm_api_key", default="ollama")
    parser.add_argument("--vlm_timeout_s", type=float, default=120.0)
    parser.add_argument("--vlm_max_tokens", type=int, default=4096)
    parser.add_argument("--vlm_enable_thinking", action="store_true")
    parser.add_argument("--vlm_validation_retries", type=int, default=2)
    parser.add_argument("--vlm_text_only", action="store_true")
    parser.add_argument("--grounding_vlm_base_url")
    parser.add_argument("--grounding_vlm_model")
    parser.add_argument("--grounding_vlm_api_key")
    parser.add_argument("--grounding_vlm_timeout_s", type=float)
    parser.add_argument("--grounding_vlm_max_tokens", type=int, default=4096)
    parser.add_argument("--grounding_vlm_enable_thinking", action="store_true")
    parser.add_argument(
        "--grounding_vlm_validation_retries",
        type=int,
        default=1,
        help="Retry a grounding response that fails strict schema validation.",
    )
    parser.add_argument("--verifier_vlm_base_url")
    parser.add_argument("--verifier_vlm_model")
    parser.add_argument("--verifier_vlm_api_key")
    parser.add_argument("--verifier_vlm_timeout_s", type=float)
    parser.add_argument("--verifier_vlm_max_tokens", type=int, default=4096)
    parser.add_argument("--verifier_vlm_enable_thinking", action="store_true")
    parser.add_argument(
        "--reasoning_mode",
        choices=(
            "simulator",
            "vlm_critic",
            "vlm",
            "vlm_with_oracle_logging",
        ),
        default="simulator",
        help=(
            "Reasoning backend. vlm_critic uses a final task-level visual critic; "
            "legacy vlm modes remain available for periodic baseline comparison."
        ),
    )
    parser.add_argument(
        "--view_selection_config",
        type=Path,
        default=DEFAULT_VIEW_SELECTION_CONFIG,
        help="Task-conditioned fixed VLM camera-selection YAML.",
    )
    parser.add_argument(
        "--navigation_config",
        type=Path,
        default=DEFAULT_NAVIGATION_CONFIG,
        help="Target-aware global-navigation YAML.",
    )
    parser.add_argument(
        "--target_aware_navigation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable target-aware geometric navigation before manipulation skills.",
    )
    parser.add_argument(
        "--enable_local_refiner",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the existing VLM local work-pose refiner before π0.5.",
    )
    parser.add_argument(
        "--local_refiner_failure_policy",
        choices=("use_coarse_pose", "abort"),
        default=None,
    )
    parser.add_argument("--local_pose_base_url")
    parser.add_argument("--local_pose_model")
    parser.add_argument("--local_pose_api_key")
    parser.add_argument("--local_pose_timeout_s", type=float)
    parser.add_argument("--local_pose_max_tokens", type=int, default=8192)
    parser.add_argument(
        "--local_pose_enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--local_pose_cameras",
        nargs="+",
        default=(
            "robot0_topview",
            "robot0_frontview",
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ),
    )
    parser.add_argument("--local_pose_max_decisions", type=int, default=8)
    parser.add_argument(
        "--plan_only",
        action="store_true",
        help="Create/reset the environment and save the VLM plan, but do not connect to pi0.5.",
    )
    parser.add_argument(
        "--no_env",
        action="store_true",
        help="VLM API smoke test without scene images/context; requires --plan_only.",
    )
    parser.add_argument("--pi05_host", default="172.16.36.10")
    parser.add_argument("--pi05_port", type=int, default=8000)
    parser.add_argument("--pi05_api_key")
    parser.add_argument("--pi05_connect_timeout_s", type=float, default=15.0)
    parser.add_argument("--pi05_infer_timeout_s", type=float, default=120.0)
    parser.add_argument("--pi05_max_retries", type=int, default=1)
    parser.add_argument(
        "--pi05_task_retries",
        type=int,
        default=0,
        help="Retry a verifier-marked retryable atomic task without resetting the environment.",
    )
    parser.add_argument(
        "--retry_recovery_mode",
        choices=("same_skill", "dependency_recovery"),
        default="same_skill",
        help=(
            "same_skill preserves the original retry behavior; "
            "dependency_recovery re-runs the producer PickObject after "
            "OBJECT_DROPPED, then re-grounds and retries the interrupted task."
        ),
    )
    parser.add_argument("--pi05_resize_size", type=int, default=224)
    parser.add_argument("--pi05_replan_steps", type=int, default=5)
    parser.add_argument("--pi05_atomic_task_horizon", type=int, default=300)
    parser.add_argument("--pi05_success_handoff_steps", type=int, default=30)
    parser.add_argument(
        "--atomic_prompt_style",
        choices=("concise", "legacy_handoff"),
        default="concise",
    )
    parser.add_argument(
        "--pi05_use_registry_horizons",
        action="store_true",
        help="Use each atomic task's horizon from RoboCasa dataset_registry.py.",
    )
    parser.add_argument(
        "--verification_mode",
        choices=("periodic", "periodic_early_stop", "skill_end"),
        default=None,
        help=(
            "periodic preserves the handoff baseline; periodic_early_stop "
            "checks every interval and stops after two consecutive successes "
            "(or accepts one success exactly at the horizon); skill_end "
            "checks exactly once after the full rollout. Existing defaults "
            "remain periodic for vlm_critic and skill_end otherwise."
        ),
    )
    parser.add_argument(
        "--pi05_verify_interval",
        type=int,
        help="Periodic verifier interval; ignored in skill_end mode.",
    )
    parser.add_argument(
        "--pi05_min_steps_before_verify",
        type=int,
        help="Earliest periodic verification step; ignored in skill_end mode.",
    )
    parser.add_argument(
        "--pi05_base_action_mode",
        choices=("full", "residual", "frozen"),
        default="residual",
    )
    parser.add_argument("--pi05_base_residual_limit", type=float, default=0.15)
    parser.add_argument(
        "--pi05_disable_held_object_guard",
        action="store_true",
        help="Disable grasp latching, gripper protection, and OBJECT_DROPPED detection.",
    )
    parser.add_argument(
        "--pi05_held_object_guard_action_override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep the gripper closed after a grasp is latched until the "
            "destination condition permits release."
        ),
    )
    parser.add_argument(
        "--pi05_held_object_hold_confirmation_steps", type=int, default=2
    )
    parser.add_argument(
        "--pi05_held_object_guard_grasp_threshold", type=float
    )
    parser.add_argument(
        "--pi05_held_object_drop_stops_rollout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop an atomic rollout immediately after a confirmed guarded-object drop.",
    )
    parser.add_argument(
        "--simulator_verifier_holding_threshold", type=float
    )
    parser.add_argument(
        "--simulator_grounding_position_threshold_m", type=float
    )
    parser.add_argument(
        "--simulator_grounding_orientation_cosine_threshold", type=float
    )
    parser.add_argument(
        "--pi05_held_object_drop_confirmation_steps", type=int, default=2
    )
    parser.add_argument("--pi05_video_skip", type=int, default=2)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--continue_on_unsuccessful", action="store_true")
    parser.add_argument("--output_root", type=Path, default=THIS_DIR / "outputs_vlm_pi05")
    args = parser.parse_args()

    if args.system_profile == LEGACY_PRIVILEGED_MAPPED_PROFILE:
        args.reasoning_mode = "simulator"
        args.navigation_config = LEGACY_PRIVILEGED_MAPPED_NAVIGATION_CONFIG
        args.target_aware_navigation = True
        args.enable_local_refiner = False
        args.local_refiner_failure_policy = "use_coarse_pose"
        args.verification_mode = "periodic"
        args.pi05_verify_interval = 5
        args.pi05_min_steps_before_verify = 10
        args.pi05_atomic_task_horizon = 600
        args.pi05_success_handoff_steps = 30
        args.pi05_task_retries = 3
        args.retry_recovery_mode = "dependency_recovery"
        args.pi05_replan_steps = 5
        args.pi05_base_action_mode = "full"
        args.pi05_disable_held_object_guard = False
        args.pi05_held_object_guard_action_override = True
        args.pi05_held_object_hold_confirmation_steps = 2
        args.pi05_held_object_drop_confirmation_steps = 2
        args.pi05_held_object_guard_grasp_threshold = 0.05
        args.pi05_held_object_drop_stops_rollout = True
        args.simulator_verifier_holding_threshold = 0.035
        args.simulator_grounding_position_threshold_m = 0.7
        args.simulator_grounding_orientation_cosine_threshold = 0.5
        args.atomic_prompt_style = "legacy_handoff"
        args.continue_on_unsuccessful = False

    positive = (
        "camera_size",
        "vlm_timeout_s",
        "vlm_max_tokens",
        "grounding_vlm_max_tokens",
        "verifier_vlm_max_tokens",
        "pi05_port",
        "pi05_connect_timeout_s",
        "pi05_infer_timeout_s",
        "pi05_resize_size",
        "pi05_replan_steps",
        "pi05_atomic_task_horizon",
        "pi05_held_object_hold_confirmation_steps",
        "pi05_held_object_drop_confirmation_steps",
        "pi05_video_skip",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    for module in ("grounding", "verifier"):
        for field in ("base_url", "model", "api_key", "timeout_s"):
            name = f"{module}_vlm_{field}"
            if getattr(args, name) is None:
                setattr(args, name, getattr(args, f"vlm_{field}"))
        if getattr(args, f"{module}_vlm_timeout_s") <= 0:
            parser.error(f"--{module}_vlm_timeout_s must be positive")
    if (
        args.vlm_validation_retries < 0
        or args.grounding_vlm_validation_retries < 0
        or args.pi05_max_retries < 0
        or args.pi05_task_retries < 0
    ):
        parser.error("retry counts must be non-negative")
    if args.verification_mode is None:
        args.verification_mode = (
            "periodic" if args.reasoning_mode == "vlm_critic" else "skill_end"
        )
    if args.pi05_verify_interval is None:
        args.pi05_verify_interval = 60 if args.reasoning_mode != "simulator" else 5
    if args.pi05_min_steps_before_verify is None:
        args.pi05_min_steps_before_verify = (
            60 if args.reasoning_mode != "simulator" else 10
        )
    if args.pi05_verify_interval <= 0:
        parser.error("--pi05_verify_interval must be positive")
    if args.pi05_min_steps_before_verify < 0:
        parser.error("--pi05_min_steps_before_verify must be non-negative")
    if args.pi05_base_residual_limit < 0:
        parser.error("--pi05_base_residual_limit must be non-negative")
    if args.pi05_success_handoff_steps < 0:
        parser.error("--pi05_success_handoff_steps must be non-negative")
    for name in (
        "pi05_held_object_guard_grasp_threshold",
        "simulator_verifier_holding_threshold",
        "simulator_grounding_position_threshold_m",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name} must be positive when provided")
    orientation_threshold = args.simulator_grounding_orientation_cosine_threshold
    if orientation_threshold is not None and not -1.0 <= orientation_threshold <= 1.0:
        parser.error(
            "--simulator_grounding_orientation_cosine_threshold must be in [-1, 1]"
        )
    if args.no_env and not args.plan_only:
        parser.error("--no_env requires --plan_only")
    if not args.no_env and (args.layout_id <= 0 or args.style_id <= 0):
        parser.error("this RoboCasa version uses 1-based --layout_id and --style_id")
    if args.vlm_plan_json is not None and not args.vlm_plan_json.is_file():
        parser.error(f"--vlm_plan_json does not exist: {args.vlm_plan_json}")
    if not args.view_selection_config.is_file():
        parser.error(
            f"--view_selection_config does not exist: {args.view_selection_config}"
        )
    if not args.navigation_config.is_file():
        parser.error(f"--navigation_config does not exist: {args.navigation_config}")
    try:
        navigation_options = load_navigation_config(args.navigation_config)["navigation"]
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(f"invalid --navigation_config: {exc}")
    if args.target_aware_navigation is None:
        args.target_aware_navigation = bool(navigation_options.get("enabled", True))
    if args.enable_local_refiner is None:
        args.enable_local_refiner = bool(
            navigation_options.get("enable_local_refiner", True)
        )
    if args.local_refiner_failure_policy is None:
        args.local_refiner_failure_policy = str(
            navigation_options.get(
                "local_refiner_failure_policy", "use_coarse_pose"
            )
        )
    if args.local_refiner_failure_policy not in {"use_coarse_pose", "abort"}:
        parser.error("navigation local_refiner_failure_policy must be use_coarse_pose or abort")
    if args.local_pose_max_tokens <= 0 or args.local_pose_max_decisions <= 0:
        parser.error("local-pose token and decision limits must be positive")
    args.local_pose_base_url = args.local_pose_base_url or args.vlm_base_url
    args.local_pose_model = args.local_pose_model or args.vlm_model
    args.local_pose_api_key = (
        args.local_pose_api_key
        if args.local_pose_api_key is not None
        else args.vlm_api_key
    )
    args.local_pose_timeout_s = args.local_pose_timeout_s or args.vlm_timeout_s
    if args.local_pose_timeout_s <= 0:
        parser.error("--local_pose_timeout_s must be positive")
    if not args.local_pose_cameras:
        parser.error("--local_pose_cameras must not be empty")
    return args


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _entity_state(entity: Any, task_env: Any) -> Mapping[str, Any] | None:
    get_state = getattr(entity, "get_state", None)
    if not callable(get_state):
        return None
    parameters = [
        parameter
        for parameter in inspect.signature(get_state).parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is parameter.empty
    ]
    if not parameters:
        state = get_state()
    elif parameters[0].name == "sim":
        state = get_state(task_env.sim)
    else:
        state = get_state(task_env)
    return _jsonable(state) if isinstance(state, Mapping) else None


def _scene_context(env: Any, args: argparse.Namespace) -> dict[str, Any]:
    task_env = env.env
    task_fixtures = getattr(task_env, "fixtures", {})
    task_objects = getattr(task_env, "objects", {})
    fixtures = []
    for alias, fixture in task_fixtures.items():
        fixtures.append(
            {
                "alias": str(alias),
                "name": str(getattr(fixture, "name", alias)),
                "natural_name": str(getattr(fixture, "nat_lang", alias)),
                "type": type(fixture).__name__,
                "state": _entity_state(fixture, task_env),
            }
        )
    objects = []
    for alias, obj in task_objects.items():
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
    episode_meta = task_env.get_ep_meta() if hasattr(task_env, "get_ep_meta") else {}
    fixture_refs = {}
    raw_fixture_refs = episode_meta.get("fixture_refs", {})
    if isinstance(raw_fixture_refs, Mapping):
        aliases_by_name = {
            str(getattr(fixture, "name", alias)): str(alias)
            for alias, fixture in task_fixtures.items()
        }
        for role, identifier in raw_fixture_refs.items():
            identifier = str(identifier)
            alias = (
                identifier
                if identifier in getattr(task_env, "fixtures", {})
                else aliases_by_name.get(identifier)
            )
            if alias is not None:
                fixture_refs[str(role)] = alias

    # RoboCasa can deliberately initialize the base beside an object. Exposing that
    # handoff prevents the planner from inserting a redundant navigation call to the
    # center of the same (potentially long) counter.
    robot_initial_state: dict[str, str] = {}
    initial_ref = getattr(task_env, "init_robot_base_ref", None)
    if initial_ref is not None:
        try:
            initial_fixture = task_env.get_fixture(initial_ref)
        except (AttributeError, KeyError, TypeError, ValueError):
            initial_fixture = initial_ref
        for alias, fixture in task_fixtures.items():
            if fixture is initial_fixture or getattr(fixture, "name", None) == getattr(
                initial_fixture, "name", initial_fixture
            ):
                robot_initial_state["fixture"] = str(alias)
                break
    for cfg in getattr(task_env, "object_cfgs", []) or []:
        if not isinstance(cfg, Mapping) or cfg.get("init_robot_here") is not True:
            continue
        object_name = str(cfg.get("name", ""))
        if object_name in task_objects:
            robot_initial_state["object"] = object_name
        break
    return {
        "env_name": args.env_name,
        "layout_id": args.layout_id,
        "style_id": args.style_id,
        "long_horizon_task": args.long_horizon_task,
        "base_episode_instruction": episode_meta.get("lang"),
        "fixture_refs": fixture_refs,
        "robot_initial_state": robot_initial_state,
        "fixtures": fixtures,
        "objects": objects,
    }


def _vlm_images(observation: Mapping[str, Any]) -> list[np.ndarray]:
    images = []
    for camera_name in POLICY_CAMERA_NAMES:
        key = f"{camera_name}_image"
        if key not in observation:
            raise KeyError(f"missing VLM/π0.5 camera observation {key!r}")
        images.append(_image_to_hwc_uint8(observation[key], key=key))
    return images


def _save_initial_views(run_dir: Path, observation: Mapping[str, Any]) -> None:
    for camera_name in POLICY_CAMERA_NAMES:
        key = f"{camera_name}_image"
        if key in observation:
            save_obs_image(run_dir / "initial_views" / f"{camera_name}.png", observation[key])


def _evaluation_only_privileged_oracle(env: Any) -> dict[str, Any]:
    """Score the final task state without influencing runtime execution.

    RoboCasa task environments expose their native success predicate through
    _check_success. The pi0.5 wrapper keeps that environment in env. This
    check intentionally runs only after the orchestrator has returned so
    privileged state cannot change grounding, verification, retries, or policy
    actions in a VLM reasoning run.
    """

    task_env = getattr(env, "env", env)
    check_success = getattr(task_env, "_check_success", None)
    source = f"{type(task_env).__name__}._check_success"
    if not callable(check_success):
        return {
            "available": False,
            "success": None,
            "source": source,
            "error": "task environment does not expose _check_success",
            "affects_runtime": False,
        }
    try:
        success = bool(check_success())
    except Exception as exc:  # Evaluation must preserve the completed rollout.
        return {
            "available": False,
            "success": None,
            "source": source,
            "error": f"{type(exc).__name__}: {exc}",
            "affects_runtime": False,
        }
    return {
        "available": True,
        "success": success,
        "source": source,
        "error": None,
        "affects_runtime": False,
    }


def main() -> int:
    args = parse_args()
    run_name = f"{args.env_name}_vlm_pi05"
    run_dir = make_run_dir(args.output_root, env_name=run_name)
    config = vars(args).copy()
    for key in (
        "vlm_api_key",
        "grounding_vlm_api_key",
        "verifier_vlm_api_key",
        "local_pose_api_key",
    ):
        config[key] = "***" if getattr(args, key) else None
    config["pi05_api_key"] = "***" if args.pi05_api_key else None
    write_json(run_dir / "config.json", config)

    env = None
    client = None
    planner = None
    full_episode_video: dict[str, Any] | None = None
    try:
        observation = None
        context = None
        if not args.no_env:
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
            context = _scene_context(env, args)
            write_json(run_dir / "scene_context.json", context)

        if args.vlm_plan_json is not None:
            calls = load_vlm_task_plan(args.vlm_plan_json)
            calls, normalizations = prepare_execution_plan(
                calls, context, prompt_style=args.atomic_prompt_style
            )
            planner_provenance = {
                "source": "saved_plan",
                "path": str(args.vlm_plan_json.resolve()),
            }
            if normalizations:
                planner_provenance["plan_normalizations"] = normalizations
        else:
            planner = RobustOpenAICompatibleVLMPlanner(
                base_url=args.vlm_base_url,
                model=args.vlm_model,
                api_key=args.vlm_api_key,
                timeout_s=args.vlm_timeout_s,
                max_tokens=args.vlm_max_tokens,
                enable_thinking=args.vlm_enable_thinking,
                include_images=not args.vlm_text_only and observation is not None,
                max_validation_retries=args.vlm_validation_retries,
                prompt_style=args.atomic_prompt_style,
            )
            calls, planner_provenance = planner.plan(
                task=args.long_horizon_task,
                images=(
                    _vlm_images(observation)
                    if observation is not None and not args.vlm_text_only
                    else None
                ),
                scene_context=context,
            )

        task_plan = {
            "long_horizon_task": args.long_horizon_task,
            "planner": planner_provenance,
            "atomic_task_calls": [call.to_dict() for call in calls],
        }
        write_json(run_dir / "task_plan.json", task_plan)
        print("[Planner] Atomic task plan:")
        for index, call in enumerate(calls, start=1):
            print(f"  {index}. {call.atomic_task}: {call.policy_prompt}")

        if args.plan_only:
            print(f"[Done] Plan saved to {run_dir / 'task_plan.json'}")
            return 0
        if env is None:
            raise RuntimeError("execution requires a RoboCasa environment")
        if not args.no_video:
            full_episode_video = env.start_episode_video_recording(
                run_dir / "episode_full.mp4",
                camera_name="robot0_agentview_left",
                fps=20,
                frame_stride=args.pi05_video_skip,
            )

        client = OpenPIWebsocketClient(
            host=args.pi05_host,
            port=args.pi05_port,
            api_key=args.pi05_api_key,
            connect_timeout_s=args.pi05_connect_timeout_s,
            infer_timeout_s=args.pi05_infer_timeout_s,
            max_retries=args.pi05_max_retries,
        )
        write_json(
            run_dir / "pi05_server.json",
            {"uri": client.uri, "server_metadata": client.server_metadata},
        )
        print(f"[Pi05] Connected to {client.uri}")
        alias_resolver_client = planner
        if alias_resolver_client is None:
            # Saved plans stay offline unless an identifier cannot be grounded locally.
            alias_resolver_client = RobustOpenAICompatibleVLMPlanner(
                base_url=args.vlm_base_url,
                model=args.vlm_model,
                api_key=args.vlm_api_key,
                timeout_s=args.vlm_timeout_s,
                max_tokens=args.vlm_max_tokens,
                enable_thinking=args.vlm_enable_thinking,
                include_images=False,
                max_validation_retries=args.vlm_validation_retries,
                prompt_style=args.atomic_prompt_style,
            )
        entity_alias_resolver = alias_resolver_client.resolve_entity_alias
        if args.reasoning_mode == "simulator":
            runtime_verifier: Any = RuntimeAtomicTaskVerifier(
                entity_alias_resolver=entity_alias_resolver,
                holding_threshold=args.simulator_verifier_holding_threshold,
            )
            grounder_options = {
                "scene_context": context,
                "entity_alias_resolver": entity_alias_resolver,
            }
            if args.simulator_grounding_position_threshold_m is not None:
                grounder_options["position_threshold_m"] = (
                    args.simulator_grounding_position_threshold_m
                )
            if args.simulator_grounding_orientation_cosine_threshold is not None:
                grounder_options["orientation_cosine_threshold"] = (
                    args.simulator_grounding_orientation_cosine_threshold
                )
            runtime_grounder: Any = RoboCasaGrounder(**grounder_options)
        else:
            grounding_vlm_client = OpenAICompatibleVisionClient(
                base_url=args.grounding_vlm_base_url,
                model=args.grounding_vlm_model,
                api_key=args.grounding_vlm_api_key,
                timeout_s=args.grounding_vlm_timeout_s,
                max_tokens=args.grounding_vlm_max_tokens,
                enable_thinking=args.grounding_vlm_enable_thinking,
            )
            verifier_vlm_client = OpenAICompatibleVisionClient(
                base_url=args.verifier_vlm_base_url,
                model=args.verifier_vlm_model,
                api_key=args.verifier_vlm_api_key,
                timeout_s=args.verifier_vlm_timeout_s,
                max_tokens=args.verifier_vlm_max_tokens,
                enable_thinking=args.verifier_vlm_enable_thinking,
            )
            observation_provider = MultiViewObservationProvider(
                image_size=args.camera_size
            )
            view_scheduler = FixedViewScheduler(
                config_path=args.view_selection_config
            )
            vlm_verifier = VLMAtomicTaskVerifier(
                client=verifier_vlm_client,
                observation_provider=observation_provider,
                view_scheduler=view_scheduler,
                scene_context=context,
                critic_only=args.reasoning_mode == "vlm_critic",
                log_dir=run_dir,
            )
            vlm_grounder = VLMRoboCasaGrounder(
                client=grounding_vlm_client,
                observation_provider=observation_provider,
                view_scheduler=view_scheduler,
                scene_context=context,
                max_validation_retries=args.grounding_vlm_validation_retries,
                log_dir=run_dir,
            )
            runtime_verifier = vlm_verifier
            runtime_grounder = vlm_grounder
            if args.reasoning_mode == "vlm_with_oracle_logging":
                simulator_verifier = RuntimeAtomicTaskVerifier(
                    entity_alias_resolver=entity_alias_resolver
                )
                simulator_grounder = RoboCasaGrounder(
                    scene_context=context,
                    entity_alias_resolver=entity_alias_resolver,
                )
                runtime_verifier = OracleLoggingVerifier(
                    primary=vlm_verifier,
                    oracle=simulator_verifier,
                    log_dir=run_dir,
                )
                runtime_grounder = OracleLoggingGrounder(
                    primary=vlm_grounder,
                    oracle=simulator_grounder,
                    log_dir=run_dir,
                )
        # The rollout reads this mode from the callable so the existing policy
        # adapter and its camera/action contract remain unchanged.
        runtime_verifier.verification_mode = args.verification_mode
        write_json(
            run_dir / "reasoning_backend.json",
            {
                "system_profile": args.system_profile,
                "reasoning_mode": args.reasoning_mode,
                "atomic_prompt_style": args.atomic_prompt_style,
                "simulator_grounding_position_threshold_m": (
                    args.simulator_grounding_position_threshold_m
                ),
                "simulator_grounding_orientation_cosine_threshold": (
                    args.simulator_grounding_orientation_cosine_threshold
                ),
                "simulator_verifier_holding_threshold": (
                    args.simulator_verifier_holding_threshold
                ),
                "runtime_grounder": type(runtime_grounder).__name__,
                "runtime_verifier": type(runtime_verifier).__name__,
                "verification_mode": args.verification_mode,
                "retry_recovery_mode": args.retry_recovery_mode,
                "verifier_strategy": (
                    (
                        f"vlm_critic_{args.verification_mode}"
                    )
                    if args.reasoning_mode == "vlm_critic"
                    else (
                        "periodic_baseline"
                        if args.verification_mode == "periodic"
                        else "periodic_two_check_early_stop"
                        if args.verification_mode == "periodic_early_stop"
                        else type(runtime_verifier).__name__
                    )
                ),
                "verify_interval": args.pi05_verify_interval,
                "min_steps_before_verify": args.pi05_min_steps_before_verify,
                "view_scheduler": (
                    view_scheduler.to_dict()
                    if args.reasoning_mode != "simulator"
                    else None
                ),
                "policy_camera_names": list(POLICY_CAMERA_NAMES),
                "target_aware_navigation": args.target_aware_navigation,
                "navigation_config": str(args.navigation_config),
                "post_grasp_transport": load_navigation_config(
                    args.navigation_config
                ).get("post_grasp_transport"),
                "post_place_retreat": load_navigation_config(
                    args.navigation_config
                ).get("post_place_retreat"),
                "post_skill_return": load_navigation_config(
                    args.navigation_config
                ).get("post_skill_return"),
                "local_refiner_enabled": args.enable_local_refiner,
                "local_refiner_failure_policy": args.local_refiner_failure_policy,
                "local_pose_vlm": (
                    {
                        "base_url": args.local_pose_base_url,
                        "model": args.local_pose_model,
                        "timeout_s": args.local_pose_timeout_s,
                        "max_tokens": args.local_pose_max_tokens,
                        "enable_thinking": args.local_pose_enable_thinking,
                        "camera_names": list(args.local_pose_cameras),
                    }
                    if args.enable_local_refiner
                    else None
                ),
                "simulator_held_object_guard_enabled": bool(
                    args.reasoning_mode == "simulator"
                    and not args.pi05_disable_held_object_guard
                ),
                "simulator_held_object_guard_action_override_enabled": bool(
                    args.reasoning_mode == "simulator"
                    and not args.pi05_disable_held_object_guard
                    and args.pi05_held_object_guard_action_override
                ),
            },
        )
        adapter = RemoteAtomicTaskPolicyAdapter(
            client=client,
            verifier=runtime_verifier,
            log_dir=run_dir,
            resize_size=args.pi05_resize_size,
            replan_steps=args.pi05_replan_steps,
            atomic_task_horizon=args.pi05_atomic_task_horizon,
            use_registry_horizons=args.pi05_use_registry_horizons,
            verify_interval=args.pi05_verify_interval,
            min_steps_before_verify=args.pi05_min_steps_before_verify,
            base_action_mode=args.pi05_base_action_mode,
            base_residual_limit=args.pi05_base_residual_limit,
            # This legacy guard reads privileged simulator grasp / relation state.
            # Keep it for the simulator baseline, but never let it influence a VLM
            # runtime mode (including oracle-shadow logging).
            held_object_guard=(
                args.reasoning_mode == "simulator"
                and not args.pi05_disable_held_object_guard
            ),
            held_object_guard_action_override=(
                args.pi05_held_object_guard_action_override
            ),
            held_object_hold_confirmation_steps=(
                args.pi05_held_object_hold_confirmation_steps
            ),
            held_object_drop_confirmation_steps=(
                args.pi05_held_object_drop_confirmation_steps
            ),
            success_handoff_steps=args.pi05_success_handoff_steps,
            verification_mode=args.verification_mode,
            prompt_style=args.atomic_prompt_style,
            held_object_drop_stops_rollout=(
                args.pi05_held_object_drop_stops_rollout
            ),
            held_object_guard_grasp_threshold=(
                args.pi05_held_object_guard_grasp_threshold
            ),
            render=not args.no_video,
            video_skip=args.pi05_video_skip,
        )
        navigation_runtime_config = load_navigation_config(
            args.navigation_config
        )
        post_skill_return_enabled = bool(
            navigation_runtime_config.get("post_skill_return", {}).get(
                "enabled", False
            )
        )
        if (
            args.target_aware_navigation
            or args.enable_local_refiner
            or post_skill_return_enabled
        ):
            navigation_pipeline = TargetAwareNavigationPipeline(
                config_path=args.navigation_config
            )
            local_refiner = None
            if args.enable_local_refiner:
                local_refiner = LocalWorkPoseRefiner(
                    decision_maker=OpenAICompatibleLocalPoseVLM(
                        base_url=args.local_pose_base_url,
                        model=args.local_pose_model,
                        api_key=args.local_pose_api_key,
                        timeout_s=args.local_pose_timeout_s,
                        max_tokens=args.local_pose_max_tokens,
                        enable_thinking=args.local_pose_enable_thinking,
                    ),
                    log_dir=run_dir,
                    camera_names=args.local_pose_cameras,
                    view_scheduler=FixedViewScheduler(
                        config_path=args.view_selection_config
                    ),
                    image_size=args.camera_size,
                    max_decisions=args.local_pose_max_decisions,
                    held_object_guard=True,
                )
            adapter = TargetAwarePolicyAdapter(
                policy_adapter=adapter,
                navigation_pipeline=navigation_pipeline,
                refiner=local_refiner,
                navigation_enabled=args.target_aware_navigation,
                local_refiner_enabled=args.enable_local_refiner,
                local_refiner_failure_policy=args.local_refiner_failure_policy,
                context={
                    "layout_id": args.layout_id,
                    "style_id": args.style_id,
                    "seed": args.seed,
                },
            )
        result = RoboCasaOrchestrator(
            atomic_task_policy_adapter=adapter,
            grounder=runtime_grounder,
        ).run_task_plan(
            env=env,
            task_plan=calls,
            episode_id=0,
            stop_on_unsuccessful=not args.continue_on_unsuccessful,
            max_task_retries=args.pi05_task_retries,
            retry_recovery_mode=args.retry_recovery_mode,
        )
        full_episode_video = env.finalize_episode_video_recording()
        write_json(run_dir / "full_episode_video.json", full_episode_video)
        result["full_episode_video"] = full_episode_video
        result["runtime_reported_success"] = bool(result.get("success", False))
        result["evaluation_only_privileged_oracle"] = (
            _evaluation_only_privileged_oracle(env)
        )
        result["task_plan_path"] = str(run_dir / "task_plan.json")
        write_json(run_dir / "long_horizon_result.json", result)
        oracle = result["evaluation_only_privileged_oracle"]
        print(
            f"[Done] status={result['status']} "
            f"runtime_success={result['runtime_reported_success']} "
            f"oracle_success={oracle['success']} output={run_dir}"
        )
        return 0 if result["success"] else 2
    except Exception as exc:
        error = {"error": str(exc), "traceback": traceback.format_exc()}
        raw_response = getattr(planner, "last_response_text", None)
        if isinstance(raw_response, str) and raw_response:
            (run_dir / "vlm_raw_response.txt").write_text(
                raw_response + "\n", encoding="utf-8"
            )
        raw_http_response = getattr(planner, "last_raw_http_response", None)
        if isinstance(raw_http_response, str):
            (run_dir / "vlm_raw_http_response.json").write_text(
                raw_http_response, encoding="utf-8"
            )
        if planner is not None:
            write_json(
                run_dir / "vlm_diagnostics.json",
                {
                    "http_status": getattr(planner, "last_http_status", None),
                    "finish_reason": getattr(planner, "last_finish_reason", None),
                    "usage": getattr(planner, "last_usage", None),
                    "parsed_response": getattr(
                        planner, "last_parsed_response", None
                    ),
                },
            )
        write_json(run_dir / "run_error.json", error)
        print(f"[Error] {exc}")
        print(error["traceback"])
        print(f"[Error] Details saved to {run_dir / 'run_error.json'}")
        return 1
    finally:
        if env is not None:
            full_episode_video = env.finalize_episode_video_recording()
            if full_episode_video.get("enabled", False):
                write_json(
                    run_dir / "full_episode_video.json",
                    full_episode_video,
                )
        if client is not None:
            client.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
