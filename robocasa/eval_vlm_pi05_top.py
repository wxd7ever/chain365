#!/usr/bin/env python3
"""Run eval_vlm_pi05 with top-camera local work-pose refinement enabled.

Use the original eval_vlm_pi05.py as the no-refinement baseline. This separate
entrypoint keeps the A/B experiment explicit and leaves the trained pi0.5 visual
input contract unchanged.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

try:
    from . import eval_vlm_pi05 as base_eval
    from .local_pose_policy_adapter import LocalPoseRefiningPolicyAdapter
    from .local_work_pose_refiner import (
        LocalWorkPoseRefiner,
        OpenAICompatibleLocalPoseVLM,
    )
except ImportError:
    import eval_vlm_pi05 as base_eval
    from local_pose_policy_adapter import LocalPoseRefiningPolicyAdapter
    from local_work_pose_refiner import (
        LocalWorkPoseRefiner,
        OpenAICompatibleLocalPoseVLM,
    )


@dataclass(frozen=True)
class _LocalPoseOptions:
    base_url: str
    model: str
    api_key: str | None
    timeout_s: float
    camera_names: tuple[str, ...]
    image_size: int
    max_decisions: int
    action_steps: int
    settle_steps: int
    translation_command: float
    rotation_command: float
    translation_distance_m: float
    rotation_angle_deg: float
    held_translation_distance_m: float
    held_rotation_angle_deg: float
    motion_max_steps: int
    translation_tolerance_m: float
    rotation_tolerance_deg: float
    max_total_translation_m: float
    max_total_rotation_deg: float
    max_invalid_stops: int
    min_confidence: float
    held_object_guard: bool

    def safe_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "base_url": self.base_url,
            "model": self.model,
            "camera_names": list(self.camera_names),
            "uses_top_camera": "robot0_topview" in self.camera_names,
            "image_size": self.image_size,
            "max_decisions": self.max_decisions,
            "action_steps": self.action_steps,
            "settle_steps": self.settle_steps,
            "translation_command": self.translation_command,
            "rotation_command": self.rotation_command,
            "translation_distance_m": self.translation_distance_m,
            "rotation_angle_deg": self.rotation_angle_deg,
            "held_translation_distance_m": self.held_translation_distance_m,
            "held_rotation_angle_deg": self.held_rotation_angle_deg,
            "motion_max_steps": self.motion_max_steps,
            "translation_tolerance_m": self.translation_tolerance_m,
            "rotation_tolerance_deg": self.rotation_tolerance_deg,
            "max_total_translation_m": self.max_total_translation_m,
            "max_total_rotation_deg": self.max_total_rotation_deg,
            "max_invalid_stops": self.max_invalid_stops,
            "min_confidence": self.min_confidence,
            "held_object_guard": self.held_object_guard,
        }


def _extra_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--local_pose_base_url")
    parser.add_argument("--local_pose_model")
    parser.add_argument("--local_pose_api_key")
    parser.add_argument("--local_pose_timeout_s", type=float)
    parser.add_argument(
        "--local_pose_cameras",
        nargs="+",
        default=None,
        help=(
            "Named MuJoCo views sent to the local-pose VLM. The default includes "
            "robot0_topview for the experiment."
        ),
    )
    parser.add_argument(
        "--local_pose_without_top_camera",
        action="store_true",
        help="Ablation: run the same local refiner without robot0_topview.",
    )
    parser.add_argument("--local_pose_image_size", type=int, default=256)
    parser.add_argument("--local_pose_max_decisions", type=int, default=8)
    parser.add_argument("--local_pose_action_steps", type=int, default=5)
    parser.add_argument("--local_pose_settle_steps", type=int, default=2)
    parser.add_argument("--local_pose_translation_command", type=float, default=0.20)
    parser.add_argument("--local_pose_rotation_command", type=float, default=0.25)
    parser.add_argument("--local_pose_translation_distance_m", type=float, default=0.10)
    parser.add_argument("--local_pose_rotation_angle_deg", type=float, default=8.0)
    parser.add_argument(
        "--local_pose_held_translation_distance_m", type=float, default=0.01
    )
    parser.add_argument("--local_pose_held_rotation_angle_deg", type=float, default=1.0)
    parser.add_argument("--local_pose_motion_max_steps", type=int, default=2000)
    parser.add_argument(
        "--local_pose_translation_tolerance_m", type=float, default=0.005
    )
    parser.add_argument("--local_pose_rotation_tolerance_deg", type=float, default=0.5)
    parser.add_argument(
        "--local_pose_max_total_translation_m", type=float, default=0.30
    )
    parser.add_argument("--local_pose_max_total_rotation_deg", type=float, default=24.0)
    parser.add_argument("--local_pose_max_invalid_stops", type=int, default=2)
    parser.add_argument("--local_pose_min_confidence", type=float, default=0.55)
    parser.add_argument("--local_pose_disable_held_object_guard", action="store_true")
    return parser


def _validate_extra(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in (
        "local_pose_image_size",
        "local_pose_max_decisions",
        "local_pose_action_steps",
        "local_pose_motion_max_steps",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.local_pose_settle_steps < 0:
        parser.error("--local_pose_settle_steps must be non-negative")
    for name in ("local_pose_translation_command", "local_pose_rotation_command"):
        value = getattr(args, name)
        if not 0.0 < value <= 1.0:
            parser.error(f"--{name} must be in (0, 1]")
    for name in (
        "local_pose_translation_distance_m",
        "local_pose_rotation_angle_deg",
        "local_pose_held_translation_distance_m",
        "local_pose_held_rotation_angle_deg",
        "local_pose_translation_tolerance_m",
        "local_pose_rotation_tolerance_deg",
        "local_pose_max_total_translation_m",
        "local_pose_max_total_rotation_deg",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.local_pose_max_invalid_stops < 0:
        parser.error("--local_pose_max_invalid_stops must be non-negative")
    if not 0.0 <= args.local_pose_min_confidence <= 1.0:
        parser.error("--local_pose_min_confidence must be in [0, 1]")
    if args.local_pose_timeout_s is not None and args.local_pose_timeout_s <= 0:
        parser.error("--local_pose_timeout_s must be positive")


def main() -> int:
    extra_parser = _extra_parser()
    extra_args, remaining = extra_parser.parse_known_args()
    _validate_extra(extra_args, extra_parser)
    original_argv = sys.argv
    original_parse_args = base_eval.parse_args
    original_orchestrator = base_eval.RoboCasaOrchestrator
    options_holder: dict[str, _LocalPoseOptions] = {}

    def parse_base_args() -> argparse.Namespace:
        args = original_parse_args()
        cameras = tuple(
            extra_args.local_pose_cameras
            or (
                "robot0_topview",
                "robot0_frontview",
                "robot0_agentview_left",
                "robot0_agentview_right",
            )
        )
        if extra_args.local_pose_without_top_camera:
            cameras = tuple(name for name in cameras if name != "robot0_topview")
        if not cameras:
            extra_parser.error("local-pose camera list must not be empty")
        options = _LocalPoseOptions(
            base_url=extra_args.local_pose_base_url or args.vlm_base_url,
            model=extra_args.local_pose_model or args.vlm_model,
            api_key=(
                extra_args.local_pose_api_key
                if extra_args.local_pose_api_key is not None
                else args.vlm_api_key
            ),
            timeout_s=(
                extra_args.local_pose_timeout_s
                if extra_args.local_pose_timeout_s is not None
                else args.vlm_timeout_s
            ),
            camera_names=cameras,
            image_size=extra_args.local_pose_image_size,
            max_decisions=extra_args.local_pose_max_decisions,
            action_steps=extra_args.local_pose_action_steps,
            settle_steps=extra_args.local_pose_settle_steps,
            translation_command=extra_args.local_pose_translation_command,
            rotation_command=extra_args.local_pose_rotation_command,
            translation_distance_m=extra_args.local_pose_translation_distance_m,
            rotation_angle_deg=extra_args.local_pose_rotation_angle_deg,
            held_translation_distance_m=extra_args.local_pose_held_translation_distance_m,
            held_rotation_angle_deg=extra_args.local_pose_held_rotation_angle_deg,
            motion_max_steps=extra_args.local_pose_motion_max_steps,
            translation_tolerance_m=extra_args.local_pose_translation_tolerance_m,
            rotation_tolerance_deg=extra_args.local_pose_rotation_tolerance_deg,
            max_total_translation_m=extra_args.local_pose_max_total_translation_m,
            max_total_rotation_deg=extra_args.local_pose_max_total_rotation_deg,
            max_invalid_stops=extra_args.local_pose_max_invalid_stops,
            min_confidence=extra_args.local_pose_min_confidence,
            held_object_guard=not extra_args.local_pose_disable_held_object_guard,
        )
        options_holder["options"] = options
        # Safe, non-secret settings are persisted by the baseline config writer.
        for key, value in options.safe_dict().items():
            setattr(args, f"local_pose_{key}", value)
        return args

    class TopCameraExperimentOrchestrator(original_orchestrator):
        def __init__(self, *, atomic_task_policy_adapter: Any, grounder: Any | None = None):
            if grounder is None:
                raise ValueError("top-camera refinement requires a grounder")
            options = options_holder["options"]
            refiner = LocalWorkPoseRefiner(
                decision_maker=OpenAICompatibleLocalPoseVLM(
                    base_url=options.base_url,
                    model=options.model,
                    api_key=options.api_key,
                    timeout_s=options.timeout_s,
                ),
                log_dir=atomic_task_policy_adapter.log_dir,
                camera_names=options.camera_names,
                image_size=options.image_size,
                max_decisions=options.max_decisions,
                action_steps=options.action_steps,
                settle_steps=options.settle_steps,
                translation_command=options.translation_command,
                rotation_command=options.rotation_command,
                translation_distance_m=options.translation_distance_m,
                rotation_angle_deg=options.rotation_angle_deg,
                held_translation_distance_m=options.held_translation_distance_m,
                held_rotation_angle_deg=options.held_rotation_angle_deg,
                motion_max_steps=options.motion_max_steps,
                translation_tolerance_m=options.translation_tolerance_m,
                rotation_tolerance_deg=options.rotation_tolerance_deg,
                max_total_translation_m=options.max_total_translation_m,
                max_total_rotation_deg=options.max_total_rotation_deg,
                max_invalid_stops=options.max_invalid_stops,
                min_confidence=options.min_confidence,
                held_object_guard=options.held_object_guard,
            )
            self.local_pose_adapter = LocalPoseRefiningPolicyAdapter(
                policy_adapter=atomic_task_policy_adapter,
                refiner=refiner,
                grounder=grounder,
            )
            super().__init__(
                atomic_task_policy_adapter=self.local_pose_adapter,
                grounder=grounder,
            )

        def run_task_plan(self, **kwargs: Any) -> dict[str, Any]:
            result = dict(super().run_task_plan(**kwargs))
            refinements = list(self.local_pose_adapter.refinement_results)
            result["local_pose_experiment"] = options_holder["options"].safe_dict()
            result["num_local_pose_refinements"] = len(refinements)
            result["local_pose_refinement_results"] = refinements
            return result

    try:
        sys.argv = [original_argv[0], *remaining]
        base_eval.parse_args = parse_base_args
        base_eval.RoboCasaOrchestrator = TopCameraExperimentOrchestrator
        return int(base_eval.main())
    finally:
        sys.argv = original_argv
        base_eval.parse_args = original_parse_args
        base_eval.RoboCasaOrchestrator = original_orchestrator


if __name__ == "__main__":
    raise SystemExit(main())

