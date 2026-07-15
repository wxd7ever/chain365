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
    from .io_utils import make_run_dir, save_obs_image, write_json
    from .openpi_client import OpenPIWebsocketClient
    from .orchestrator import RoboCasaOrchestrator
    from .pi05_env import POLICY_CAMERA_NAMES, create_pi05_env
    from .pi05_rollout import _image_to_hwc_uint8
    from .robust_vlm_task_planner import (
        RobustOpenAICompatibleVLMPlanner,
        prepare_execution_plan,
    )
    from .vlm_task_planner import load_vlm_task_plan
except ImportError:  # Support ``python robocasa/eval_vlm_pi05.py``.
    from atomic_task_policy_adapter import RemoteAtomicTaskPolicyAdapter
    from atomic_task_verifier import RuntimeAtomicTaskVerifier
    from io_utils import make_run_dir, save_obs_image, write_json
    from openpi_client import OpenPIWebsocketClient
    from orchestrator import RoboCasaOrchestrator
    from pi05_env import POLICY_CAMERA_NAMES, create_pi05_env
    from pi05_rollout import _image_to_hwc_uint8
    from robust_vlm_task_planner import (
        RobustOpenAICompatibleVLMPlanner,
        prepare_execution_plan,
    )
    from vlm_task_planner import load_vlm_task_plan


THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create RoboCasa directly, decompose a long task with a VLM, and execute "
            "the atomic plan through one shared remote pi0.5 connection."
        )
    )
    parser.add_argument("--env_name", default="PickPlaceCounterToMicrowave")
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
    parser.add_argument("--vlm_validation_retries", type=int, default=2)
    parser.add_argument("--vlm_text_only", action="store_true")
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
    parser.add_argument("--pi05_resize_size", type=int, default=224)
    parser.add_argument("--pi05_replan_steps", type=int, default=5)
    parser.add_argument("--pi05_atomic_task_horizon", type=int, default=300)
    parser.add_argument(
        "--pi05_use_registry_horizons",
        action="store_true",
        help="Use each atomic task's horizon from RoboCasa dataset_registry.py.",
    )
    parser.add_argument("--pi05_verify_interval", type=int, default=5)
    parser.add_argument("--pi05_min_steps_before_verify", type=int, default=10)
    parser.add_argument(
        "--pi05_base_action_mode",
        choices=("full", "residual", "frozen"),
        default="residual",
    )
    parser.add_argument("--pi05_base_residual_limit", type=float, default=0.15)
    parser.add_argument("--pi05_video_skip", type=int, default=2)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--continue_on_unsuccessful", action="store_true")
    parser.add_argument("--output_root", type=Path, default=THIS_DIR / "outputs_vlm_pi05")
    args = parser.parse_args()

    positive = (
        "camera_size",
        "vlm_timeout_s",
        "pi05_port",
        "pi05_connect_timeout_s",
        "pi05_infer_timeout_s",
        "pi05_resize_size",
        "pi05_replan_steps",
        "pi05_atomic_task_horizon",
        "pi05_verify_interval",
        "pi05_video_skip",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if (
        args.vlm_validation_retries < 0
        or args.pi05_max_retries < 0
        or args.pi05_task_retries < 0
    ):
        parser.error("retry counts must be non-negative")
    if args.pi05_min_steps_before_verify < 0:
        parser.error("--pi05_min_steps_before_verify must be non-negative")
    if args.pi05_base_residual_limit < 0:
        parser.error("--pi05_base_residual_limit must be non-negative")
    if args.no_env and not args.plan_only:
        parser.error("--no_env requires --plan_only")
    if not args.no_env and (args.layout_id <= 0 or args.style_id <= 0):
        parser.error("this RoboCasa version uses 1-based --layout_id and --style_id")
    if args.vlm_plan_json is not None and not args.vlm_plan_json.is_file():
        parser.error(f"--vlm_plan_json does not exist: {args.vlm_plan_json}")
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
    fixtures = []
    for alias, fixture in getattr(task_env, "fixtures", {}).items():
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
    episode_meta = task_env.get_ep_meta() if hasattr(task_env, "get_ep_meta") else {}
    return {
        "env_name": args.env_name,
        "layout_id": args.layout_id,
        "style_id": args.style_id,
        "long_horizon_task": args.long_horizon_task,
        "base_episode_instruction": episode_meta.get("lang"),
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


def main() -> int:
    args = parse_args()
    run_name = f"{args.env_name}_vlm_pi05"
    run_dir = make_run_dir(args.output_root, env_name=run_name)
    config = vars(args).copy()
    config["vlm_api_key"] = "***" if args.vlm_api_key else None
    config["pi05_api_key"] = "***" if args.pi05_api_key else None
    write_json(run_dir / "config.json", config)

    env = None
    client = None
    planner = None
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
            calls, normalizations = prepare_execution_plan(calls, context)
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
                include_images=not args.vlm_text_only and observation is not None,
                max_validation_retries=args.vlm_validation_retries,
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
        adapter = RemoteAtomicTaskPolicyAdapter(
            client=client,
            verifier=RuntimeAtomicTaskVerifier(),
            log_dir=run_dir,
            resize_size=args.pi05_resize_size,
            replan_steps=args.pi05_replan_steps,
            atomic_task_horizon=args.pi05_atomic_task_horizon,
            use_registry_horizons=args.pi05_use_registry_horizons,
            verify_interval=args.pi05_verify_interval,
            min_steps_before_verify=args.pi05_min_steps_before_verify,
            base_action_mode=args.pi05_base_action_mode,
            base_residual_limit=args.pi05_base_residual_limit,
            render=not args.no_video,
            video_skip=args.pi05_video_skip,
        )
        result = RoboCasaOrchestrator(
            atomic_task_policy_adapter=adapter
        ).run_task_plan(
            env=env,
            task_plan=calls,
            episode_id=0,
            stop_on_unsuccessful=not args.continue_on_unsuccessful,
            max_task_retries=args.pi05_task_retries,
        )
        result["task_plan_path"] = str(run_dir / "task_plan.json")
        write_json(run_dir / "long_horizon_result.json", result)
        print(f"[Done] status={result['status']} output={run_dir}")
        return 0 if result["success"] else 2
    except Exception as exc:
        error = {"error": str(exc), "traceback": traceback.format_exc()}
        raw_response = getattr(planner, "last_response_text", None)
        if isinstance(raw_response, str) and raw_response:
            (run_dir / "vlm_raw_response.txt").write_text(
                raw_response + "\n", encoding="utf-8"
            )
        write_json(run_dir / "run_error.json", error)
        print(f"[Error] {exc}")
        print(error["traceback"])
        print(f"[Error] Details saved to {run_dir / 'run_error.json'}")
        return 1
    finally:
        if client is not None:
            client.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
