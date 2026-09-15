#!/usr/bin/env python3
"""Probe whether pi0.5 can pick, retract, and keep holding an object.

This demo bypasses the planner and long-horizon scheduler.  The policy receives
only the normal pi0.5 camera/state observation plus one explicit language
instruction.  Privileged simulator state is used for evaluation, and (in the
default ``after_grasp_handoff`` mode) only to decide when the post-grasp handoff
window starts.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .atomic_task_policy_adapter import RemoteAtomicTaskPolicyAdapter
    from .atomic_task_prompt_builder import clean_entity_name
    from .atomic_task_schemas import AtomicTaskCall
    from .atomic_task_verifier import RuntimeAtomicTaskVerifier
    from .grounding import RoboCasaGrounder
    from .io_utils import make_run_dir, save_obs_image, write_json
    from .openpi_client import OpenPIWebsocketClient
    from .pi05_env import POLICY_CAMERA_NAMES, create_pi05_env
    from .post_grasp_transport import PostGraspTransportController
    from .navigation.navigation_pipeline import (
        DEFAULT_CONFIG_PATH as DEFAULT_NAVIGATION_CONFIG,
        load_navigation_config,
    )
    from .utils import object_utils
except ImportError:  # Support ``python robocasa/demo_pick_retract.py``.
    from atomic_task_policy_adapter import RemoteAtomicTaskPolicyAdapter
    from atomic_task_prompt_builder import clean_entity_name
    from atomic_task_schemas import AtomicTaskCall
    from atomic_task_verifier import RuntimeAtomicTaskVerifier
    from grounding import RoboCasaGrounder
    from io_utils import make_run_dir, save_obs_image, write_json
    from openpi_client import OpenPIWebsocketClient
    from pi05_env import POLICY_CAMERA_NAMES, create_pi05_env
    from post_grasp_transport import PostGraspTransportController
    from navigation.navigation_pipeline import (
        DEFAULT_CONFIG_PATH as DEFAULT_NAVIGATION_CONFIG,
        load_navigation_config,
    )
    from utils import object_utils


THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one direct pi0.5 PickObject rollout and evaluate whether the "
            "object is lifted, the arm clears the source, and the gripper keeps "
            "holding the object."
        )
    )
    parser.add_argument("--env_name", default="SteamInMicrowave")
    parser.add_argument("--target_object", default="vegetable")
    parser.add_argument(
        "--source_fixture",
        help=(
            "Optional source fixture alias. If omitted, privileged grounding "
            "resolves the target object's current supporting fixture."
        ),
    )
    parser.add_argument("--source_name", help="Natural-language source name override.")
    parser.add_argument("--layout_id", type=int, default=1)
    parser.add_argument("--style_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--camera_size", type=int, default=256)
    parser.add_argument(
        "--policy_prompt",
        help="Exact VLA prompt. If omitted, a pick/lift/retract/hold prompt is generated.",
    )
    parser.add_argument(
        "--stop_mode",
        choices=("after_grasp_handoff", "fixed_horizon"),
        default="after_grasp_handoff",
        help=(
            "after_grasp_handoff: detect the first grasp periodically, then run "
            "--success_handoff_steps more steps; fixed_horizon: no intermediate "
            "success checks and always run the full horizon."
        ),
    )
    parser.add_argument(
        "--success_handoff_steps",
        type=int,
        default=60,
        help="Post-grasp steps used by after_grasp_handoff mode.",
    )
    parser.add_argument(
        "--min_lift_m",
        type=float,
        default=0.08,
        help="Minimum final object height increase counted as a clear lift.",
    )
    parser.add_argument(
        "--source_clearance_margin_m",
        type=float,
        default=0.02,
        help="Expansion margin for the final EEF-outside-source check.",
    )
    parser.add_argument(
        "--held_object_guard",
        action="store_true",
        help=(
            "Enable the current pipeline's gripper guard. It is disabled by "
            "default so the demo measures the policy's own keep-holding behavior."
        ),
    )
    parser.add_argument(
        "--post-grasp-only",
        "--post_grasp_only",
        dest="post_grasp_only",
        action="store_true",
        help=(
            "After the Pick verifier succeeds, run only the deterministic "
            "post-grasp lift/retract/hold controller; do not navigate."
        ),
    )
    parser.add_argument(
        "--navigation_config",
        type=Path,
        default=DEFAULT_NAVIGATION_CONFIG,
        help="YAML containing the post_grasp_transport section.",
    )
    parser.add_argument("--pi05_host", default="172.16.11.115")
    parser.add_argument("--pi05_port", type=int, default=8000)
    parser.add_argument("--pi05_api_key")
    parser.add_argument("--pi05_connect_timeout_s", type=float, default=15.0)
    parser.add_argument("--pi05_infer_timeout_s", type=float, default=120.0)
    parser.add_argument("--pi05_max_retries", type=int, default=1)
    parser.add_argument("--pi05_horizon", type=int, default=600)
    parser.add_argument("--pi05_replan_steps", type=int, default=5)
    parser.add_argument("--pi05_resize_size", type=int, default=224)
    parser.add_argument("--pi05_verify_interval", type=int, default=5)
    parser.add_argument("--pi05_min_steps_before_verify", type=int, default=10)
    parser.add_argument(
        "--pi05_base_action_mode",
        choices=("full", "residual", "frozen"),
        default="frozen",
        help="frozen isolates arm/gripper behavior when the reset pose is already usable.",
    )
    parser.add_argument("--pi05_base_residual_limit", type=float, default=0.15)
    parser.add_argument("--video_camera", default="robot0_agentview_left")
    parser.add_argument("--video_skip", type=int, default=1)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=THIS_DIR / "outputs_pick_retract_demo",
    )
    args = parser.parse_args()

    positive = (
        "layout_id",
        "style_id",
        "camera_size",
        "pi05_port",
        "pi05_connect_timeout_s",
        "pi05_infer_timeout_s",
        "pi05_horizon",
        "pi05_replan_steps",
        "pi05_resize_size",
        "pi05_verify_interval",
        "video_skip",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.pi05_min_steps_before_verify < 0:
        parser.error("--pi05_min_steps_before_verify must be non-negative")
    if args.pi05_max_retries < 0:
        parser.error("--pi05_max_retries must be non-negative")
    if args.success_handoff_steps < 0:
        parser.error("--success_handoff_steps must be non-negative")
    if args.min_lift_m < 0:
        parser.error("--min_lift_m must be non-negative")
    if args.source_clearance_margin_m < 0:
        parser.error("--source_clearance_margin_m must be non-negative")
    if not args.target_object.strip():
        parser.error("--target_object must be non-empty")
    if not args.navigation_config.is_file():
        parser.error(
            f"--navigation_config does not exist: {args.navigation_config}"
        )
    return args


def _task_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if all(hasattr(current, name) for name in ("fixtures", "objects", "sim")):
            return current
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is not None and unwrapped is not current:
            current = unwrapped
        else:
            current = getattr(current, "env", None)
    raise ValueError("could not find the underlying RoboCasa task environment")


def _object_natural_name(raw_env: Any, alias: str) -> str:
    get_obj_lang = getattr(raw_env, "get_obj_lang", None)
    if callable(get_obj_lang):
        try:
            value = str(get_obj_lang(alias)).strip()
            if value:
                return value
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    obj = raw_env.objects[alias]
    return clean_entity_name(str(getattr(obj, "name", alias)))


def _fixture_natural_name(raw_env: Any, alias: str) -> str:
    fixture = raw_env.fixtures[alias]
    value = str(getattr(fixture, "nat_lang", "")).strip()
    return value or clean_entity_name(str(getattr(fixture, "name", alias)))


def _scene_context(env: Any, args: argparse.Namespace) -> dict[str, Any]:
    raw_env = _task_env(env)
    fixture_refs: dict[str, str] = {}
    get_ep_meta = getattr(raw_env, "get_ep_meta", None)
    episode_meta = get_ep_meta() if callable(get_ep_meta) else {}
    raw_fixture_refs = episode_meta.get("fixture_refs", {})
    if isinstance(raw_fixture_refs, Mapping):
        aliases_by_name = {
            str(getattr(fixture, "name", alias)): str(alias)
            for alias, fixture in raw_env.fixtures.items()
        }
        for role, identifier in raw_fixture_refs.items():
            identifier = str(identifier)
            alias = identifier if identifier in raw_env.fixtures else aliases_by_name.get(identifier)
            if alias is not None:
                fixture_refs[str(role)] = alias
    return {
        "env_name": args.env_name,
        "layout_id": args.layout_id,
        "style_id": args.style_id,
        "fixture_refs": fixture_refs,
        "fixtures": [
            {
                "alias": str(alias),
                "name": str(getattr(fixture, "name", alias)),
                "natural_name": str(getattr(fixture, "nat_lang", alias)),
                "type": type(fixture).__name__,
            }
            for alias, fixture in raw_env.fixtures.items()
        ],
        "objects": [
            {
                "alias": str(alias),
                "name": str(getattr(obj, "name", alias)),
                "natural_name": _object_natural_name(raw_env, str(alias)),
                "type": type(obj).__name__,
            }
            for alias, obj in raw_env.objects.items()
        ],
    }


def _save_policy_views(output_dir: Path, observation: Mapping[str, Any]) -> None:
    for camera_name in POLICY_CAMERA_NAMES:
        key = f"{camera_name}_image"
        if key in observation:
            save_obs_image(output_dir / f"{camera_name}.png", observation[key])


def _object_position(raw_env: Any, alias: str) -> np.ndarray:
    body_id = raw_env.obj_body_id[alias]
    return np.asarray(raw_env.sim.data.body_xpos[body_id], dtype=float).copy()


def _eef_position(raw_env: Any) -> np.ndarray:
    site_id = raw_env.robots[0].eef_site_id
    if isinstance(site_id, Mapping):
        site_id = site_id.get("right", next(iter(site_id.values()), None))
    if site_id is None:
        raise KeyError("robot does not expose a right end-effector site")
    return np.asarray(raw_env.sim.data.site_xpos[site_id], dtype=float).copy()


def _probe_call(object_alias: str, object_name: str) -> AtomicTaskCall:
    return AtomicTaskCall.from_mapping(
        {
            "subgoal_id": f"probe_{object_alias}",
            "atomic_task": "PickObject",
            "arguments": {
                "object_id": object_alias,
                "object_name": object_name,
            },
            "termination_condition": {
                "predicate": "holding",
                "subject": object_alias,
                "desired_value": True,
            },
            "policy_prompt": f"Pick up the {object_name} and keep holding it.",
            "metadata": {"demo_only": True},
        }
    )


def _resolve_source_fixture(
    *,
    env: Any,
    raw_env: Any,
    scene_context: Mapping[str, Any],
    probe_call: AtomicTaskCall,
    explicit_source: str | None,
) -> tuple[str | None, dict[str, Any] | None]:
    if explicit_source is not None and explicit_source.strip():
        alias = explicit_source.strip()
        if alias not in raw_env.fixtures:
            raise ValueError(
                f"source fixture alias {alias!r} does not exist; "
                f"available aliases={sorted(str(item) for item in raw_env.fixtures)}"
            )
        return alias, None
    grounder = RoboCasaGrounder(scene_context=scene_context)
    grounding = grounder.ground(env=env, atomic_task_call=probe_call).to_dict()
    alias = str(grounding.get("target_fixture_alias") or "").strip() or None
    return alias, grounding


def _default_prompt(object_name: str, source_name: str | None) -> str:
    source_clause = f" from the {source_name}" if source_name else ""
    return (
        f"Pick up the {object_name}{source_clause}. Lift it clear of the source, "
        "retract the arm toward the robot to a stable carrying pose in front of "
        "the robot, and keep the gripper closed while holding the object. Stop "
        "after reaching the carrying pose. Do not place or release the object."
    )


def _pick_only_prompt(object_name: str, source_name: str | None) -> str:
    source_clause = f" from the {source_name}" if source_name else ""
    return (
        f"Pick up the {object_name}{source_clause} and keep holding it. "
        "Do not place or release the object."
    )


def _pick_call(
    *,
    object_alias: str,
    object_name: str,
    source_alias: str | None,
    source_name: str | None,
    prompt: str,
) -> AtomicTaskCall:
    arguments: dict[str, Any] = {
        "object_id": object_alias,
        "object_name": object_name,
    }
    if source_alias:
        arguments["source_id"] = source_alias
    if source_name:
        arguments["source_name"] = source_name
    return AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "pick_retract_probe",
            "atomic_task": "PickObject",
            "arguments": arguments,
            # The rollout detects the first grasp, then gives the VLA a fixed
            # post-success window in which to execute the requested retraction.
            "termination_condition": {
                "predicate": "holding",
                "subject": object_alias,
                "desired_value": True,
            },
            "policy_prompt": prompt,
            "metadata": {"demo_only": True, "probe": "pick_retract_hold"},
        }
    )


def _clearance_verification(
    *,
    env: Any,
    observation: Mapping[str, Any],
    object_alias: str,
    object_name: str,
    source_alias: str | None,
    source_name: str | None,
    margin: float,
    step_index: int,
) -> dict[str, Any] | None:
    if source_alias is None:
        return None
    call = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "posthoc_source_clearance",
            "atomic_task": "PickObject",
            "arguments": {
                "object_id": object_alias,
                "object_name": object_name,
                "source_id": source_alias,
                "source_name": source_name or clean_entity_name(source_alias),
            },
            "termination_condition": {
                "predicate": "eef_outside_fixture",
                "subject": source_alias,
                "desired_value": True,
                "margin": margin,
                "only_2d": False,
            },
            "policy_prompt": "Post-hoc evaluator only.",
            "metadata": {"demo_only": True},
        }
    )
    verifier = RuntimeAtomicTaskVerifier(default_required_consecutive_successes=1)
    return verifier(
        env=env,
        atomic_task_call=call,
        observation=observation,
        step_index=step_index,
        info={"verification_phase": "posthoc_demo_evaluation"},
    )


def main() -> int:
    args = parse_args()
    run_dir = make_run_dir(
        args.output_root,
        env_name=f"{args.env_name}_{args.target_object}_pick_retract",
    )
    config = vars(args).copy()
    config["pi05_api_key"] = "***" if args.pi05_api_key else None
    write_json(run_dir / "config.json", config)

    env = None
    client = None
    full_video: dict[str, Any] | None = None
    try:
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
        # Kitchen fixtures and movable objects are created lazily by the first
        # robosuite reset in this RoboCasa version.
        observation = env.reset()
        raw_env = _task_env(env)
        if not getattr(raw_env, "layout_and_style_ids", None):
            excluded = sorted(int(value) for value in getattr(raw_env, "EXCLUDE_LAYOUTS", []))
            raise ValueError(
                f"{args.env_name} does not support layout {args.layout_id}; "
                f"excluded layouts={excluded}"
            )
        _save_policy_views(run_dir / "initial_views", observation)
        scene_context = _scene_context(env, args)
        write_json(run_dir / "scene_context.json", scene_context)

        object_alias = args.target_object.strip()
        if object_alias not in raw_env.objects:
            raise ValueError(
                f"target object alias {object_alias!r} does not exist; "
                f"available aliases={sorted(str(item) for item in raw_env.objects)}"
            )
        object_name = _object_natural_name(raw_env, object_alias)
        probe_call = _probe_call(object_alias, object_name)
        source_alias, grounding = _resolve_source_fixture(
            env=env,
            raw_env=raw_env,
            scene_context=scene_context,
            probe_call=probe_call,
            explicit_source=args.source_fixture,
        )
        if grounding is not None:
            write_json(run_dir / "source_grounding.json", grounding)
        source_name = (
            args.source_name.strip()
            if args.source_name is not None and args.source_name.strip()
            else _fixture_natural_name(raw_env, source_alias)
            if source_alias is not None
            else None
        )
        prompt = (
            args.policy_prompt.strip()
            if args.policy_prompt is not None and args.policy_prompt.strip()
            else (
                _pick_only_prompt(object_name, source_name)
                if args.post_grasp_only
                else _default_prompt(object_name, source_name)
            )
        )
        call = _pick_call(
            object_alias=object_alias,
            object_name=object_name,
            source_alias=source_alias,
            source_name=source_name,
            prompt=prompt,
        )
        write_json(run_dir / "pick_call.json", call.to_dict())

        initial_object_position = _object_position(raw_env, object_alias)
        initial_eef_position = _eef_position(raw_env)
        print(
            f"[Target] object={object_alias} ({object_name}) "
            f"source={source_alias or 'unresolved'}"
        )
        print(f"[Prompt] {prompt}")
        print(
            f"[Mode] stop_mode={args.stop_mode} "
            f"held_object_guard={args.held_object_guard} "
            f"post_grasp_only={args.post_grasp_only}"
        )

        if not args.no_video:
            full_video = env.start_episode_video_recording(
                run_dir / "pick_retract_full.mp4",
                camera_name=args.video_camera,
                fps=20,
                frame_stride=args.video_skip,
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
        verifier = RuntimeAtomicTaskVerifier(default_required_consecutive_successes=1)
        verifier.verification_mode = (
            "periodic" if args.stop_mode == "after_grasp_handoff" else "skill_end"
        )
        adapter = RemoteAtomicTaskPolicyAdapter(
            client=client,
            verifier=verifier,
            log_dir=run_dir,
            resize_size=args.pi05_resize_size,
            replan_steps=args.pi05_replan_steps,
            atomic_task_horizon=args.pi05_horizon,
            verify_interval=args.pi05_verify_interval,
            min_steps_before_verify=args.pi05_min_steps_before_verify,
            base_action_mode=args.pi05_base_action_mode,
            base_residual_limit=args.pi05_base_residual_limit,
            held_object_guard=args.held_object_guard,
            success_handoff_steps=(
                args.success_handoff_steps
                if args.stop_mode == "after_grasp_handoff"
                else 0
            ),
            render=not args.no_video,
            video_skip=args.video_skip,
        )
        policy_result = adapter.execute(
            env=env,
            scheduler_query={"atomic_task_call": call.to_dict()},
            episode_id=0,
        )
        post_grasp_transport_result = None
        if args.post_grasp_only:
            transport_config = load_navigation_config(
                args.navigation_config
            ).get("post_grasp_transport", {})
            controller = PostGraspTransportController(
                transport_config,
                log_dir=run_dir,
            )
            post_grasp_transport_result = controller.prepare(
                env=env,
                atomic_task_call=call,
                episode_id=0,
                pick_verified=bool(policy_result.get("success", False)),
            )

        final_observation = env.get_observation()
        _save_policy_views(run_dir / "final_views", final_observation)
        final_object_position = _object_position(raw_env, object_alias)
        final_eef_position = _eef_position(raw_env)
        holding = bool(object_utils.check_obj_grasped(raw_env, object_alias))
        lift_m = float(final_object_position[2] - initial_object_position[2])
        lifted = lift_m >= args.min_lift_m
        completed_steps = int(
            policy_result.get("rollout_logs", {}).get("Horizon", 0)
        )
        clearance_verification = _clearance_verification(
            env=env,
            observation=final_observation,
            object_alias=object_alias,
            object_name=object_name,
            source_alias=source_alias,
            source_name=source_name,
            margin=args.source_clearance_margin_m,
            step_index=completed_steps,
        )
        source_cleared = (
            clearance_verification.get("status") == "success"
            if clearance_verification is not None
            else None
        )
        overall_success = bool(
            holding
            and lifted
            and (source_cleared is not False)
            and (
                not args.post_grasp_only
                or bool(
                    post_grasp_transport_result
                    and post_grasp_transport_result.get(
                        "transport_ready", False
                    )
                )
            )
        )
        metrics = {
            "holding_final": holding,
            "object_lift_m": lift_m,
            "min_lift_m": args.min_lift_m,
            "lifted": lifted,
            "source_fixture_alias": source_alias,
            "source_clearance_margin_m": args.source_clearance_margin_m,
            "source_cleared": source_cleared,
            "source_clearance_verification": clearance_verification,
            "initial_object_position": initial_object_position,
            "final_object_position": final_object_position,
            "initial_eef_position": initial_eef_position,
            "final_eef_position": final_eef_position,
            "eef_displacement_m": float(
                np.linalg.norm(final_eef_position - initial_eef_position)
            ),
            "overall_success": overall_success,
            "post_grasp_transport_ready": (
                post_grasp_transport_result.get("transport_ready")
                if post_grasp_transport_result is not None
                else None
            ),
            "overall_rule": (
                "holding_final AND lifted AND source_cleared; source_cleared is "
                "ignored only when no source fixture can be resolved"
            ),
        }
        write_json(run_dir / "pose_metrics.json", metrics)
        write_json(run_dir / "policy_result.json", policy_result)
        if post_grasp_transport_result is not None:
            write_json(
                run_dir / "post_grasp_transport_result.json",
                post_grasp_transport_result,
            )

        if not args.no_video:
            full_video = env.finalize_episode_video_recording()
            write_json(run_dir / "full_video.json", full_video)
        result = {
            "success": overall_success,
            "policy_reported_success": bool(policy_result.get("success", False)),
            "target_object_alias": object_alias,
            "target_object_name": object_name,
            "source_fixture_alias": source_alias,
            "source_name": source_name,
            "policy_prompt": prompt,
            "stop_mode": args.stop_mode,
            "held_object_guard": args.held_object_guard,
            "post_grasp_only": args.post_grasp_only,
            "post_grasp_transport_result": post_grasp_transport_result,
            "metrics": metrics,
            "policy_result": policy_result,
            "full_video": full_video,
            "run_dir": str(run_dir),
        }
        write_json(run_dir / "result.json", result)
        print(
            f"[Done] success={overall_success} holding={holding} "
            f"lift_m={lift_m:.3f} source_cleared={source_cleared}"
        )
        print(f"[Output] {run_dir}")
        return 0 if overall_success else 2
    except Exception as exc:
        error = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "run_dir": str(run_dir),
        }
        write_json(run_dir / "run_error.json", error)
        print(f"[Error] {type(exc).__name__}: {exc}")
        print(f"[Error] Details saved to {run_dir / 'run_error.json'}")
        return 1
    finally:
        if env is not None and not args.no_video:
            try:
                metadata = env.finalize_episode_video_recording()
                if metadata.get("enabled", False):
                    write_json(run_dir / "full_video.json", metadata)
            except Exception:
                pass
        if client is not None:
            client.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
