#!/usr/bin/env python3
"""Automate paired RoboCasa local-pose camera ablations across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
BASELINE_SCRIPT = THIS_DIR / "eval_vlm_pi05.py"
MULTIVIEW_SCRIPT = THIS_DIR / "eval_vlm_pi05_multiview.py"

VARIANTS = (
    "baseline",
    "refiner_no_top",
    "refiner_top_no_wrist",
    "refiner_top",
)

VARIANT_CAMERAS = {
    "baseline": (),
    "refiner_no_top": (
        "robot0_frontview",
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ),
    "refiner_top_no_wrist": (
        "robot0_topview",
        "robot0_frontview",
        "robot0_agentview_left",
        "robot0_agentview_right",
    ),
    "refiner_top": (
        "robot0_topview",
        "robot0_frontview",
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one fixed task plan per seed, run paired baseline / camera "
            "ablation evaluations, and aggregate success and retry metrics. Put "
            "the ordinary eval_vlm_pi05 arguments after --."
        )
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=THIS_DIR / "outputs_top_camera_ablation",
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--mujoco_gl", default="egl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--local_pose_base_url")
    parser.add_argument("--local_pose_model", default="qwen2.5vl:3b")
    parser.add_argument("--local_pose_api_key")
    parser.add_argument("--local_pose_timeout_s", type=float)
    parser.add_argument("--local_pose_image_size", type=int, default=256)
    parser.add_argument("--local_pose_max_decisions", type=int, default=8)
    parser.add_argument("--local_pose_action_steps", type=int, default=5)
    parser.add_argument("--local_pose_settle_steps", type=int, default=2)
    parser.add_argument("--local_pose_translation_command", type=float, default=1.0)
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
    parser.add_argument("common_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.common_args and args.common_args[0] == "--":
        args.common_args = args.common_args[1:]
    if not args.common_args:
        parser.error("ordinary eval arguments are required after --")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain unique non-negative integers")
    positive = (
        "local_pose_image_size",
        "local_pose_max_decisions",
        "local_pose_action_steps",
        "local_pose_motion_max_steps",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("local-pose image size, decisions, and action steps must be positive")
    if args.local_pose_settle_steps < 0:
        parser.error("--local_pose_settle_steps must be non-negative")
    for name in ("local_pose_translation_command", "local_pose_rotation_command"):
        if not 0.0 < getattr(args, name) <= 1.0:
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
    controlled = (
        "--seed",
        "--output_root",
        "--vlm_plan_json",
        "--plan_only",
        "--no_env",
        "--local_pose_cameras",
        "--local_pose_without_top_camera",
    )
    for option in controlled:
        if any(arg == option or arg.startswith(f"{option}=") for arg in args.common_args):
            parser.error(f"{option} is controlled by this automation script")
    if not any(
        arg == "--long_horizon_task" or arg.startswith("--long_horizon_task=")
        for arg in args.common_args
    ):
        parser.error("common eval arguments must include --long_horizon_task")
    return args


def _latest(path: Path, name: str) -> Path | None:
    candidates = list(path.rglob(name)) if path.is_dir() else []
    return max(candidates, key=lambda item: item.stat().st_mtime_ns, default=None)


def _redacted_command(command: Sequence[str]) -> list[str]:
    result: list[str] = []
    hide_next = False
    for item in command:
        if hide_next:
            result.append("***")
            hide_next = False
            continue
        if "api_key=" in item:
            result.append(item.split("=", 1)[0] + "=***")
            continue
        result.append(item)
        if item.endswith("api_key"):
            hide_next = True
    return result


def _run_command(
    command: Sequence[str],
    *,
    log_path: Path,
    mujoco_gl: str,
    dry_run: bool,
) -> int:
    printable = " ".join(_redacted_command(command))
    print(f"[Ablation] {printable}")
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MUJOCO_GL"] = mujoco_gl
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"COMMAND: {printable}\n\n")
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return int(process.wait())


def _local_pose_args(args: argparse.Namespace, variant: str) -> list[str]:
    cameras = VARIANT_CAMERAS[variant]
    values = [
        "--local_pose_model",
        args.local_pose_model,
        "--local_pose_image_size",
        str(args.local_pose_image_size),
        "--local_pose_max_decisions",
        str(args.local_pose_max_decisions),
        "--local_pose_action_steps",
        str(args.local_pose_action_steps),
        "--local_pose_settle_steps",
        str(args.local_pose_settle_steps),
        "--local_pose_translation_command",
        str(args.local_pose_translation_command),
        "--local_pose_rotation_command",
        str(args.local_pose_rotation_command),
        "--local_pose_min_confidence",
        str(args.local_pose_min_confidence),
        "--local_pose_translation_distance_m",
        str(args.local_pose_translation_distance_m),
        "--local_pose_rotation_angle_deg",
        str(args.local_pose_rotation_angle_deg),
        "--local_pose_held_translation_distance_m",
        str(args.local_pose_held_translation_distance_m),
        "--local_pose_held_rotation_angle_deg",
        str(args.local_pose_held_rotation_angle_deg),
        "--local_pose_motion_max_steps",
        str(args.local_pose_motion_max_steps),
        "--local_pose_translation_tolerance_m",
        str(args.local_pose_translation_tolerance_m),
        "--local_pose_rotation_tolerance_deg",
        str(args.local_pose_rotation_tolerance_deg),
        "--local_pose_max_total_translation_m",
        str(args.local_pose_max_total_translation_m),
        "--local_pose_max_total_rotation_deg",
        str(args.local_pose_max_total_rotation_deg),
        "--local_pose_max_invalid_stops",
        str(args.local_pose_max_invalid_stops),
        "--local_pose_cameras",
        *cameras,
    ]
    if args.local_pose_base_url:
        values.extend(("--local_pose_base_url", args.local_pose_base_url))
    if args.local_pose_api_key is not None:
        values.extend(("--local_pose_api_key", args.local_pose_api_key))
    if args.local_pose_timeout_s is not None:
        values.extend(("--local_pose_timeout_s", str(args.local_pose_timeout_s)))
    return values


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object in {path}")
    return dict(value)


def _result_row(
    *,
    variant: str,
    seed: int,
    result_path: Path | None,
    return_code: int,
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "completed": False,
        "return_code": return_code,
        "result_path": str(result_path) if result_path is not None else None,
        "error": error,
        "success": False,
        "status": "error",
        "num_atomic_attempts": 0,
        "num_operation_attempts": 0,
        "num_successful_steps": 0,
        "num_executed_steps": 0,
        "num_local_pose_refinements": 0,
        "num_successful_refinements": 0,
        "num_refinement_actions": 0,
        "num_refinement_env_steps": 0,
        "uses_top_camera": "robot0_topview" in VARIANT_CAMERAS[variant],
        "uses_wrist_camera": "robot0_eye_in_hand" in VARIANT_CAMERAS[variant],
    }
    if result_path is None or not result_path.is_file():
        return row
    result = _load_json(result_path)
    steps = result.get("step_results", [])
    if not isinstance(steps, list):
        steps = []
    refinements = result.get("local_pose_refinement_results", [])
    if not isinstance(refinements, list):
        refinements = []
    row.update(
        {
            "completed": True,
            "success": bool(result.get("success", False)),
            "status": str(result.get("status", "unknown")),
            "num_atomic_attempts": int(result.get("num_atomic_attempts", 0)),
            "num_operation_attempts": sum(
                int(step.get("num_attempts", 0))
                for step in steps
                if isinstance(step, Mapping)
            ),
            "num_successful_steps": sum(
                bool(step.get("success", False))
                for step in steps
                if isinstance(step, Mapping)
            ),
            "num_executed_steps": len(steps),
            "num_local_pose_refinements": len(refinements),
            "num_successful_refinements": sum(
                bool(item.get("success", False))
                for item in refinements
                if isinstance(item, Mapping)
            ),
            "num_refinement_actions": sum(
                int(item.get("num_executed_actions", 0))
                for item in refinements
                if isinstance(item, Mapping)
            ),
            "num_refinement_env_steps": sum(
                int(item.get("total_env_steps", 0))
                for item in refinements
                if isinstance(item, Mapping)
            ),
        }
    )
    return row


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [
            row for row in rows if row["variant"] == variant and row["completed"]
        ]
        if not selected:
            continue
        result[variant] = {
            "num_completed": len(selected),
            "num_successes": sum(bool(row["success"]) for row in selected),
            "success_rate": statistics.fmean(
                float(bool(row["success"])) for row in selected
            ),
            "mean_atomic_attempts": _mean(selected, "num_atomic_attempts"),
            "mean_operation_attempts": _mean(selected, "num_operation_attempts"),
            "mean_refinement_actions": _mean(selected, "num_refinement_actions"),
            "mean_refinement_env_steps": _mean(
                selected, "num_refinement_env_steps"
            ),
        }
    return result


def _paired_comparison(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> dict[str, Any] | None:
    left_by_seed = {
        int(row["seed"]): row
        for row in rows
        if row["variant"] == left and row["completed"]
    }
    right_by_seed = {
        int(row["seed"]): row
        for row in rows
        if row["variant"] == right and row["completed"]
    }
    seeds = sorted(set(left_by_seed).intersection(right_by_seed))
    if not seeds:
        return None
    deltas = [
        int(bool(right_by_seed[seed]["success"]))
        - int(bool(left_by_seed[seed]["success"]))
        for seed in seeds
    ]
    return {
        "left": left,
        "right": right,
        "interpretation": f"positive means {right} is better",
        "num_paired_seeds": len(seeds),
        "paired_seeds": seeds,
        "success_rate_delta": statistics.fmean(deltas),
        "wins": sum(delta > 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
    }


def _write_outputs(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    aggregate = _aggregate(rows)
    comparisons = []
    for left, right in (
        ("baseline", "refiner_top"),
        ("refiner_no_top", "refiner_top"),
        ("refiner_top_no_wrist", "refiner_top"),
    ):
        comparison = _paired_comparison(rows, left, right)
        if comparison is not None:
            comparisons.append(comparison)
    summary = {
        "variants": {
            name: {"camera_names": list(VARIANT_CAMERAS[name])}
            for name in VARIANTS
        },
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "runs": list(rows),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "automation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if rows:
        with (output_root / "automation_runs.csv").open(
            "w", encoding="utf-8", newline=""
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("\n[Ablation] Aggregate results")
    for variant, metrics in aggregate.items():
        print(
            f"  {variant}: {metrics['num_successes']}/{metrics['num_completed']} "
            f"success_rate={metrics['success_rate']:.3f}"
        )
    for comparison in comparisons:
        print(
            f"  {comparison['left']} -> {comparison['right']}: "
            f"delta={comparison['success_rate_delta']:+.3f} "
            f"wins/losses/ties={comparison['wins']}/"
            f"{comparison['losses']}/{comparison['ties']}"
        )
    print(f"[Ablation] Summary: {output_root / 'automation_summary.json'}")


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    python = str(args.python_executable)
    rows: list[dict[str, Any]] = []
    had_infrastructure_error = False

    for seed in args.seeds:
        plan_root = output_root / "plans" / f"seed_{seed}"
        plan_path = _latest(plan_root, "task_plan.json") if args.resume else None
        if plan_path is None:
            plan_command = [
                python,
                str(BASELINE_SCRIPT),
                *args.common_args,
                "--seed",
                str(seed),
                "--plan_only",
                "--output_root",
                str(plan_root),
            ]
            return_code = _run_command(
                plan_command,
                log_path=plan_root / "plan.log",
                mujoco_gl=args.mujoco_gl,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                plan_path = Path("<generated-task-plan.json>")
            else:
                plan_path = _latest(plan_root, "task_plan.json")
            if return_code != 0 or plan_path is None:
                had_infrastructure_error = True
                error = f"plan generation failed with return code {return_code}"
                print(f"[Ablation] seed={seed}: {error}")
                for variant in args.variants:
                    rows.append(
                        _result_row(
                            variant=variant,
                            seed=seed,
                            result_path=None,
                            return_code=return_code,
                            error=error,
                        )
                    )
                if args.stop_on_error:
                    break
                continue

        for variant in args.variants:
            run_root = output_root / "runs" / variant / f"seed_{seed}"
            existing_result = (
                _latest(run_root, "long_horizon_result.json") if args.resume else None
            )
            if existing_result is not None:
                print(f"[Ablation] Resume {variant} seed={seed}: {existing_result}")
                rows.append(
                    _result_row(
                        variant=variant,
                        seed=seed,
                        result_path=existing_result,
                        return_code=0,
                    )
                )
                continue
            script = BASELINE_SCRIPT if variant == "baseline" else MULTIVIEW_SCRIPT
            command = [
                python,
                str(script),
                *args.common_args,
                "--seed",
                str(seed),
                "--vlm_plan_json",
                str(plan_path),
                "--output_root",
                str(run_root),
            ]
            if variant != "baseline":
                command.extend(_local_pose_args(args, variant))
            return_code = _run_command(
                command,
                log_path=run_root / "run.log",
                mujoco_gl=args.mujoco_gl,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                continue
            result_path = _latest(run_root, "long_horizon_result.json")
            # eval_vlm_pi05 returns 2 for a completed but unsuccessful task.
            infrastructure_error = return_code not in (0, 2) or result_path is None
            if infrastructure_error:
                had_infrastructure_error = True
            rows.append(
                _result_row(
                    variant=variant,
                    seed=seed,
                    result_path=result_path,
                    return_code=return_code,
                    error=(
                        f"evaluation failed with return code {return_code}"
                        if infrastructure_error
                        else None
                    ),
                )
            )
            if infrastructure_error and args.stop_on_error:
                break
        if had_infrastructure_error and args.stop_on_error:
            break

    if not args.dry_run:
        _write_outputs(output_root, rows)
    return 1 if had_infrastructure_error else 0


if __name__ == "__main__":
    raise SystemExit(main())

