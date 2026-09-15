#!/usr/bin/env python3
"""Run one object-conditioned RoboCasa navigation probe through remote pi0.5.

The demo intentionally bypasses the task planner and the long-horizon
orchestrator.  It uses privileged grounding only to resolve a movable object to
its current supporting fixture and to compute the same fixture + reference
object navigation condition used by the main pipeline.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any, Mapping

try:
    from .atomic_task_policy_adapter import RemoteAtomicTaskPolicyAdapter
    from .atomic_task_prompt_builder import clean_entity_name
    from .atomic_task_schemas import AtomicTaskCall
    from .atomic_task_verifier import RuntimeAtomicTaskVerifier
    from .grounding import RoboCasaGrounder
    from .io_utils import make_run_dir, save_obs_image, write_json
    from .openpi_client import OpenPIWebsocketClient
    from .pi05_env import POLICY_CAMERA_NAMES, create_pi05_env
except ImportError:  # Support ``python robocasa/demo_navigate_to_object.py``.
    from atomic_task_policy_adapter import RemoteAtomicTaskPolicyAdapter
    from atomic_task_prompt_builder import clean_entity_name
    from atomic_task_schemas import AtomicTaskCall
    from atomic_task_verifier import RuntimeAtomicTaskVerifier
    from grounding import RoboCasaGrounder
    from io_utils import make_run_dir, save_obs_image, write_json
    from openpi_client import OpenPIWebsocketClient
    from pi05_env import POLICY_CAMERA_NAMES, create_pi05_env


THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Navigate the RoboCasa mobile base to the work pose associated with "
            "one live movable object and save diagnostics plus video."
        )
    )
    parser.add_argument("--env_name", default="DeliverStraw")
    parser.add_argument("--target_object", default="glass_cup")
    parser.add_argument("--layout_id", type=int, default=2)
    parser.add_argument("--style_id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--camera_size", type=int, default=256)
    parser.add_argument("--position_threshold_m", type=float, default=0.4)
    parser.add_argument(
        "--orientation_cosine_threshold", type=float, default=0.90
    )
    parser.add_argument(
        "--policy_prompt",
        help=(
            "Optional exact VLA prompt. By default the demo generates "
            "'Navigate to the <fixture> near the <object>.'"
        ),
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
        default="full",
    )
    parser.add_argument("--pi05_base_residual_limit", type=float, default=0.15)
    parser.add_argument("--video_camera", default="robot0_agentview_left")
    parser.add_argument("--video_skip", type=int, default=2)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=THIS_DIR / "outputs_navigation_object_demo",
    )
    args = parser.parse_args()

    positive = (
        "layout_id",
        "style_id",
        "camera_size",
        "position_threshold_m",
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
    if not -1.0 <= args.orientation_cosine_threshold <= 1.0:
        parser.error("--orientation_cosine_threshold must be in [-1, 1]")
    if not args.target_object.strip():
        parser.error("--target_object must be non-empty")
    return args


def _raw_env(env: Any) -> Any:
    return getattr(env, "env", env)


def _object_natural_name(raw_env: Any, alias: str) -> str:
    get_obj_lang = getattr(raw_env, "get_obj_lang", None)
    if callable(get_obj_lang):
        try:
            name = str(get_obj_lang(alias)).strip()
            if name:
                return name
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    obj = raw_env.objects[alias]
    return str(getattr(obj, "name", clean_entity_name(alias)))


def _scene_context(env: Any, args: argparse.Namespace) -> dict[str, Any]:
    raw_env = _raw_env(env)
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
            alias = (
                identifier
                if identifier in raw_env.fixtures
                else aliases_by_name.get(identifier)
            )
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


def _save_policy_views(
    output_dir: Path, observation: Mapping[str, Any]
) -> None:
    for camera_name in POLICY_CAMERA_NAMES:
        key = f"{camera_name}_image"
        if key in observation:
            save_obs_image(output_dir / f"{camera_name}.png", observation[key])


def _probe_operation_call(
    *, object_alias: str, object_name: str, args: argparse.Namespace
) -> AtomicTaskCall:
    """Create a non-executed operation whose target the grounder can resolve."""

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
            "metadata": {
                "work_pose": {
                    "mode": "standard_pick",
                    "reference_object_id": object_alias,
                    "position_threshold_m": args.position_threshold_m,
                    "orientation_cosine_threshold": (
                        args.orientation_cosine_threshold
                    ),
                },
                "demo_only": True,
            },
        }
    )


def _object_conditioned_navigation_call(
    *,
    grounder: RoboCasaGrounder,
    probe_call: AtomicTaskCall,
    grounding: Mapping[str, Any],
    object_alias: str,
    object_name: str,
    explicit_prompt: str | None,
) -> AtomicTaskCall:
    call = grounder.build_navigation_call(
        operation_call=probe_call,
        grounding_result=grounding,
    )
    fixture_alias = str(grounding["target_fixture_alias"])
    fixture_role = next(
        (
            str(role)
            for role, alias in grounder.scene_context.get("fixture_refs", {}).items()
            if str(alias) == fixture_alias
        ),
        None,
    )
    fixture_entity_name = next(
        (
            str(entity.get("natural_name") or entity.get("name"))
            for entity in grounder.scene_context.get("fixtures", [])
            if isinstance(entity, Mapping)
            and str(entity.get("alias")) == fixture_alias
            and (entity.get("natural_name") or entity.get("name"))
        ),
        clean_entity_name(fixture_alias),
    )
    fixture_name = (
        clean_entity_name(fixture_role)
        if fixture_role is not None
        else fixture_entity_name
    )
    prompt = (
        explicit_prompt.strip()
        if explicit_prompt is not None and explicit_prompt.strip()
        else f"Navigate to the {fixture_name} near the {object_name}."
    )
    payload = call.to_dict()
    payload["subgoal_id"] = f"navigate_near_{object_alias}"
    payload["policy_prompt"] = prompt
    payload["arguments"]["reference_object_id"] = object_alias
    payload["metadata"].update(
        {
            "demo_only": True,
            "target_object_alias": object_alias,
            "target_object_name": object_name,
        }
    )
    return AtomicTaskCall.from_mapping(payload)


def main() -> int:
    args = parse_args()
    run_dir = make_run_dir(
        args.output_root,
        env_name=f"{args.env_name}_{args.target_object}_navigation",
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
        raw_env = _raw_env(env)
        if not getattr(raw_env, "layout_and_style_ids", None):
            excluded_layouts = sorted(
                int(layout_id)
                for layout_id in getattr(raw_env, "EXCLUDE_LAYOUTS", [])
            )
            raise ValueError(
                f"{args.env_name} does not support layout {args.layout_id}; "
                f"excluded layouts={excluded_layouts}. Choose a layout that "
                "contains every fixture required by the task."
            )
        observation = env.reset()
        _save_policy_views(run_dir / "initial_views", observation)
        scene_context = _scene_context(env, args)
        write_json(run_dir / "scene_context.json", scene_context)

        object_alias = args.target_object.strip()
        if object_alias not in raw_env.objects:
            available = sorted(str(alias) for alias in raw_env.objects)
            raise ValueError(
                f"target object alias {object_alias!r} does not exist; "
                f"available object aliases={available}"
            )
        object_name = _object_natural_name(raw_env, object_alias)

        grounder = RoboCasaGrounder(
            scene_context=scene_context,
            position_threshold_m=args.position_threshold_m,
            orientation_cosine_threshold=args.orientation_cosine_threshold,
        )
        probe_call = _probe_operation_call(
            object_alias=object_alias,
            object_name=object_name,
            args=args,
        )
        initial_grounding = grounder.ground(
            env=env, atomic_task_call=probe_call
        ).to_dict()
        write_json(run_dir / "initial_grounding.json", initial_grounding)
        if not initial_grounding.get("target_fixture_alias"):
            raise RuntimeError(
                "grounder could not resolve the target object to a supporting fixture; "
                f"see {run_dir / 'initial_grounding.json'}"
            )

        navigation_call = _object_conditioned_navigation_call(
            grounder=grounder,
            probe_call=probe_call,
            grounding=initial_grounding,
            object_alias=object_alias,
            object_name=object_name,
            explicit_prompt=args.policy_prompt,
        )
        write_json(run_dir / "navigation_call.json", navigation_call.to_dict())
        print(
            f"[Target] object={object_alias} "
            f"fixture={initial_grounding['target_fixture_alias']}"
        )
        print(f"[Prompt] {navigation_call.policy_prompt}")

        if not args.no_video:
            full_video = env.start_episode_video_recording(
                run_dir / "navigation_full.mp4",
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
        verifier = RuntimeAtomicTaskVerifier()
        verifier.verification_mode = "periodic"
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
            held_object_guard=False,
            success_handoff_steps=0,
            render=not args.no_video,
            video_skip=args.video_skip,
        )
        policy_result = adapter.execute(
            env=env,
            scheduler_query={"atomic_task_call": navigation_call.to_dict()},
            episode_id=0,
        )

        final_observation = env.get_observation()
        _save_policy_views(run_dir / "final_views", final_observation)
        final_grounding = grounder.ground(
            env=env, atomic_task_call=probe_call
        ).to_dict()
        write_json(run_dir / "final_grounding.json", final_grounding)
        if not args.no_video:
            full_video = env.finalize_episode_video_recording()
            write_json(run_dir / "full_video.json", full_video)

        result = {
            "success": bool(policy_result.get("success", False)),
            "status": policy_result.get("status"),
            "target_object_alias": object_alias,
            "target_object_name": object_name,
            "target_fixture_alias": initial_grounding.get("target_fixture_alias"),
            "policy_prompt": navigation_call.policy_prompt,
            "initial_grounding": initial_grounding,
            "final_grounding": final_grounding,
            "policy_result": policy_result,
            "full_video": full_video,
            "run_dir": str(run_dir),
        }
        write_json(run_dir / "result.json", result)
        print(
            f"[Done] success={result['success']} "
            f"final_grounded={final_grounding.get('grounded')} output={run_dir}"
        )
        return 0 if result["success"] else 2
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
