#!/usr/bin/env python3
"""Run paired privileged-vs-VLM RoboCasa reasoning evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
EVAL_SCRIPT = THIS_DIR / "eval_vlm_pi05.py"

VARIANTS = {
    "privileged": {
        "reasoning_mode": "simulator",
        "verification_mode": "periodic",
        "verify_interval": 5,
        "min_steps_before_verify": 10,
        "description": (
            "Privileged RoboCasa grounding, runtime verifier, and held-object guard"
        ),
    },
    "vlm": {
        "reasoning_mode": "vlm_critic",
        "verification_mode": "periodic",
        "verify_interval": 60,
        "min_steps_before_verify": 60,
        "description": "Image-only VLM grounding and simple task-level VLM critic",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one fixed task plan per seed, run paired privileged and "
            "VLM reasoning evaluations, and aggregate end-to-end and atomic-task "
            "success metrics. Put ordinary eval_vlm_pi05 arguments after --."
        )
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 21)))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=THIS_DIR / "outputs_reasoning_backend_ablation",
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--mujoco_gl", default="egl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument(
        "--verification_mode",
        choices=("periodic", "periodic_early_stop", "skill_end"),
        help="Optional shared mode override for both variants.",
    )
    parser.add_argument(
        "--verify_interval",
        type=int,
        help="Optional shared interval override for both variants.",
    )
    parser.add_argument(
        "--min_steps_before_verify",
        type=int,
        help="Optional shared minimum-step override for both variants.",
    )
    parser.add_argument(
        "--privileged_verification_mode",
        choices=("periodic", "periodic_early_stop", "skill_end"),
        default="periodic",
    )
    parser.add_argument("--privileged_verify_interval", type=int, default=5)
    parser.add_argument(
        "--privileged_min_steps_before_verify", type=int, default=10
    )
    parser.add_argument(
        "--vlm_verification_mode",
        choices=("periodic", "periodic_early_stop", "skill_end"),
        default="periodic",
    )
    parser.add_argument("--vlm_verify_interval", type=int, default=60)
    parser.add_argument("--vlm_min_steps_before_verify", type=int, default=60)
    parser.add_argument("common_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.common_args and args.common_args[0] == "--":
        args.common_args = args.common_args[1:]
    if not args.common_args:
        parser.error("ordinary eval arguments are required after --")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain unique non-negative integers")
    intervals = (
        args.verify_interval,
        args.privileged_verify_interval,
        args.vlm_verify_interval,
    )
    if any(value is not None and value <= 0 for value in intervals):
        parser.error("verification intervals must be positive")
    minimum_steps = (
        args.min_steps_before_verify,
        args.privileged_min_steps_before_verify,
        args.vlm_min_steps_before_verify,
    )
    if any(value is not None and value < 0 for value in minimum_steps):
        parser.error("minimum verification steps must be non-negative")
    controlled = (
        "--seed",
        "--output_root",
        "--vlm_plan_json",
        "--plan_only",
        "--no_env",
        "--reasoning_mode",
        "--verification_mode",
        "--pi05_verify_interval",
        "--pi05_min_steps_before_verify",
    )
    for option in controlled:
        if any(
            value == option or value.startswith(f"{option}=")
            for value in args.common_args
        ):
            parser.error(f"{option} is controlled by this benchmark")
    if not any(
        value == "--long_horizon_task"
        or value.startswith("--long_horizon_task=")
        for value in args.common_args
    ):
        parser.error("common eval arguments must include --long_horizon_task")
    return args


def _variant_verification_schedule(
    args: argparse.Namespace, variant: str
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    mode = args.verification_mode or getattr(
        args, f"{variant}_verification_mode"
    )
    interval = (
        args.verify_interval
        if args.verify_interval is not None
        else getattr(args, f"{variant}_verify_interval")
    )
    minimum_steps = (
        args.min_steps_before_verify
        if args.min_steps_before_verify is not None
        else getattr(args, f"{variant}_min_steps_before_verify")
    )
    return {
        "mode": str(mode),
        "interval": int(interval),
        "min_steps_before_verify": int(minimum_steps),
    }


def _latest(path: Path, name: str) -> Path | None:
    candidates = list(path.rglob(name)) if path.is_dir() else []
    return max(candidates, key=lambda item: item.stat().st_mtime_ns, default=None)


def _has_available_oracle(result_path: Path) -> bool:
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(result, Mapping):
        return False
    oracle = result.get("evaluation_only_privileged_oracle")
    return bool(
        isinstance(oracle, Mapping) and oracle.get("available", False)
    )


def _result_matches_configuration(
    result_path: Path,
    *,
    variant: str,
    schedule: Mapping[str, Any],
) -> bool:
    config_path = result_path.parent / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return bool(
        isinstance(config, Mapping)
        and config.get("reasoning_mode") == VARIANTS[variant]["reasoning_mode"]
        and config.get("verification_mode") == schedule["mode"]
        and config.get("pi05_verify_interval") == schedule["interval"]
        and config.get("pi05_min_steps_before_verify")
        == schedule["min_steps_before_verify"]
    )


def _redacted_command(command: Sequence[str]) -> list[str]:
    result: list[str] = []
    hide_next = False
    for value in command:
        if hide_next:
            result.append("***")
            hide_next = False
            continue
        if "api_key=" in value:
            result.append(value.split("=", 1)[0] + "=***")
            continue
        result.append(value)
        if value.endswith("api_key"):
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
    print(f"[Reasoning Ablation] {printable}")
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


def _failure_stage(
    result: Mapping[str, Any], step_results: Sequence[Mapping[str, Any]]
) -> str | None:
    if bool(result.get("success", False)):
        return None
    if not step_results:
        return "planning_or_runtime"
    final = step_results[-1]
    failure_code = str(final.get("failure_code") or "")
    if failure_code.startswith("GROUNDING_"):
        return "grounding"
    if final.get("atomic_task") == "NavigateKitchen":
        return "navigation"
    logs = final.get("rollout_logs")
    if isinstance(logs, Mapping):
        verification = logs.get("Final_Verification")
        if isinstance(verification, Mapping):
            verifier_code = str(verification.get("failure_code") or "")
            if verifier_code.endswith("_ERROR"):
                return "verifier_error"
        return "atomic_policy_or_verifier"
    return "orchestration"


def _result_row(
    *,
    variant: str,
    seed: int,
    verification_schedule: Mapping[str, Any],
    result_path: Path | None,
    return_code: int,
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": variant,
        "seed": int(seed),
        "verification_mode": verification_schedule["mode"],
        "retry_recovery_mode": None,
        "verify_interval": verification_schedule["interval"],
        "min_steps_before_verify": verification_schedule[
            "min_steps_before_verify"
        ],
        "completed": False,
        "success": False,
        "oracle_task_success": None,
        "oracle_available": False,
        "oracle_source": None,
        "oracle_error": None,
        "runtime_success": None,
        "runtime_oracle_agreement": None,
        "status": "missing",
        "return_code": int(return_code),
        "result_path": str(result_path) if result_path is not None else None,
        "error": error,
        "num_planned_steps": None,
        "num_executed_steps": None,
        "num_successful_atomic_steps": None,
        "runtime_atomic_step_success_rate": None,
        "plan_completion_rate": None,
        "num_atomic_attempts": None,
        "num_retries": None,
        "retry_dependency_recoveries": None,
        "num_inserted_navigation_steps": None,
        "num_grounding_events": None,
        "runtime_grounding_resolution_rate": None,
        "num_verifier_calls": None,
        "total_policy_steps": None,
        "failure_stage": None,
    }
    if result_path is None or not result_path.is_file():
        return row
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        row["error"] = f"could not parse result: {exc}"
        return row
    if not isinstance(result, Mapping):
        row["error"] = "result JSON is not an object"
        return row
    status = str(result.get("status", "unknown"))
    execution_completed = status in {"success", "failed", "uncertain"}
    runtime_success = bool(
        result.get("runtime_reported_success", result.get("success", False))
    )
    oracle = result.get("evaluation_only_privileged_oracle")
    oracle_available = bool(
        isinstance(oracle, Mapping) and oracle.get("available", False)
    )
    oracle_success = (
        bool(oracle.get("success")) if oracle_available else None
    )
    step_results = [
        item
        for item in result.get("step_results", [])
        if isinstance(item, Mapping)
    ]
    successful_steps = sum(bool(item.get("success", False)) for item in step_results)
    planned_steps = int(result.get("num_planned_steps", 0))
    executed_steps = int(result.get("num_executed_steps", len(step_results)))
    attempts = int(result.get("num_atomic_attempts", 0))
    verifier_calls = 0
    total_policy_steps = 0
    for item in step_results:
        logs = item.get("rollout_logs")
        if not isinstance(logs, Mapping):
            continue
        total_policy_steps += int(logs.get("Horizon", 0))
        history = logs.get("Verification_History", [])
        if isinstance(history, list):
            verifier_calls += len(history)
    grounding_events = [
        item
        for item in result.get("grounding_events", [])
        if isinstance(item, Mapping)
    ]
    resolved_groundings = sum(
        str(item.get("status")) in {"grounded", "navigation_required"}
        for item in grounding_events
    )
    row.update(
        {
            "completed": execution_completed and oracle_available,
            # success is intentionally the common post-rollout environment
            # oracle, not either backend's own runtime verdict.
            "success": bool(oracle_success) if oracle_available else False,
            "oracle_task_success": oracle_success,
            "oracle_available": oracle_available,
            "oracle_source": oracle.get("source") if isinstance(oracle, Mapping) else None,
            "oracle_error": oracle.get("error") if isinstance(oracle, Mapping) else None,
            "runtime_success": runtime_success,
            "runtime_oracle_agreement": (
                runtime_success == oracle_success if oracle_available else None
            ),
            "status": status,
            "num_planned_steps": planned_steps,
            "num_executed_steps": executed_steps,
            "num_successful_atomic_steps": successful_steps,
            "runtime_atomic_step_success_rate": (
                successful_steps / len(step_results) if step_results else None
            ),
            "plan_completion_rate": (
                executed_steps / planned_steps if planned_steps else None
            ),
            "num_atomic_attempts": attempts,
            "num_retries": max(0, attempts - len(step_results)),
            "retry_recovery_mode": str(
                result.get("retry_recovery_mode", "same_skill")
            ),
            "retry_dependency_recoveries": int(
                result.get("num_dependency_recoveries", 0)
            ),
            "num_inserted_navigation_steps": int(
                result.get("num_inserted_navigation_steps", 0)
            ),
            "num_grounding_events": len(grounding_events),
            "runtime_grounding_resolution_rate": (
                resolved_groundings / len(grounding_events)
                if grounding_events
                else None
            ),
            "num_verifier_calls": verifier_calls,
            "total_policy_steps": total_policy_steps,
            "failure_stage": _failure_stage(result, step_results),
        }
    )
    if not oracle_available and row["error"] is None:
        row["error"] = (
            "result has no available evaluation-only privileged oracle; "
            "rerun with the current eval_vlm_pi05.py"
        )
    elif runtime_success and oracle_success is False:
        row["failure_stage"] = "runtime_false_positive"
    elif not runtime_success and oracle_success is True:
        row["failure_stage"] = "runtime_false_negative"
    return row


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [
            row for row in rows if row["variant"] == variant and row["completed"]
        ]
        if not selected:
            continue
        successes = sum(bool(row["success"]) for row in selected)
        failure_stages = Counter(
            str(row["failure_stage"])
            for row in selected
            if row.get("failure_stage") is not None
        )
        result[variant] = {
            "num_completed": len(selected),
            "num_successes": successes,
            "oracle_task_success_rate": successes / len(selected),
            "oracle_task_success_rate_wilson_95": _wilson_interval(
                successes, len(selected)
            ),
            "runtime_reported_success_rate": (
                sum(bool(row["runtime_success"]) for row in selected)
                / len(selected)
            ),
            "runtime_oracle_agreement_rate": (
                sum(bool(row["runtime_oracle_agreement"]) for row in selected)
                / len(selected)
            ),
            "runtime_false_positive_count": sum(
                bool(row["runtime_success"]) and not bool(row["success"])
                for row in selected
            ),
            "runtime_false_negative_count": sum(
                not bool(row["runtime_success"]) and bool(row["success"])
                for row in selected
            ),
            "mean_runtime_atomic_step_success_rate": _mean(
                selected, "runtime_atomic_step_success_rate"
            ),
            "mean_plan_completion_rate": _mean(
                selected, "plan_completion_rate"
            ),
            "mean_runtime_grounding_resolution_rate": _mean(
                selected, "runtime_grounding_resolution_rate"
            ),
            "mean_atomic_attempts": _mean(selected, "num_atomic_attempts"),
            "mean_retries": _mean(selected, "num_retries"),
            "mean_dependency_recoveries": _mean(
                selected, "retry_dependency_recoveries"
            ),
            "mean_inserted_navigation_steps": _mean(
                selected, "num_inserted_navigation_steps"
            ),
            "mean_verifier_calls": _mean(selected, "num_verifier_calls"),
            "mean_total_policy_steps": _mean(selected, "total_policy_steps"),
            "failure_stage_counts": dict(sorted(failure_stages.items())),
        }
    return result


def _exact_mcnemar_p(privileged_only: int, vlm_only: int) -> float:
    discordant = privileged_only + vlm_only
    if discordant == 0:
        return 1.0
    lower = min(privileged_only, vlm_only)
    probability = sum(
        math.comb(discordant, index) for index in range(lower + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * probability)


def _paired_comparison(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    privileged = {
        int(row["seed"]): row
        for row in rows
        if row["variant"] == "privileged" and row["completed"]
    }
    vlm = {
        int(row["seed"]): row
        for row in rows
        if row["variant"] == "vlm" and row["completed"]
    }
    seeds = sorted(set(privileged).intersection(vlm))
    if not seeds:
        return None
    privileged_only = sum(
        bool(privileged[seed]["success"]) and not bool(vlm[seed]["success"])
        for seed in seeds
    )
    vlm_only = sum(
        bool(vlm[seed]["success"]) and not bool(privileged[seed]["success"])
        for seed in seeds
    )
    both_success = sum(
        bool(privileged[seed]["success"]) and bool(vlm[seed]["success"])
        for seed in seeds
    )
    both_failed = len(seeds) - privileged_only - vlm_only - both_success
    deltas = [
        int(bool(vlm[seed]["success"]))
        - int(bool(privileged[seed]["success"]))
        for seed in seeds
    ]
    return {
        "num_paired_seeds": len(seeds),
        "paired_seeds": seeds,
        "oracle_success_rate_delta_vlm_minus_privileged": statistics.fmean(
            deltas
        ),
        "both_success": both_success,
        "privileged_only_success": privileged_only,
        "vlm_only_success": vlm_only,
        "both_failed": both_failed,
        "mcnemar_exact_two_sided_p": _exact_mcnemar_p(
            privileged_only, vlm_only
        ),
    }


def _atomic_task_metrics(
    records: Sequence[tuple[str, int, Path]]
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for variant, _seed, result_path in records:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(result, Mapping):
            continue
        for item in result.get("step_results", []):
            if not isinstance(item, Mapping):
                continue
            task = str(item.get("atomic_task") or "unknown")
            groups[(variant, task)].append(item)
    metrics: dict[str, Any] = {}
    for (variant, task), items in sorted(groups.items()):
        successes = sum(bool(item.get("success", False)) for item in items)
        attempts = [
            int(item.get("num_attempts", 0))
            for item in items
            if item.get("num_attempts") is not None
        ]
        metrics.setdefault(variant, {})[task] = {
            "num_steps": len(items),
            "num_successes": successes,
            "runtime_reported_success_rate": successes / len(items),
            "runtime_reported_success_rate_wilson_95": _wilson_interval(
                successes, len(items)
            ),
            "mean_attempts": statistics.fmean(attempts) if attempts else None,
        }
    return metrics


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[tuple[str, int, Path]],
    *,
    requested_seeds: Sequence[int],
    requested_variants: Sequence[str],
    verification_schedules: Mapping[str, Mapping[str, Any]],
    common_args: Sequence[str],
    mujoco_gl: str,
) -> None:
    aggregate = _aggregate(rows)
    summary = {
        "comparison_unit": "paired_seed_with_shared_task_plan",
        "experiment_config": {
            "requested_seeds": list(requested_seeds),
            "requested_variants": list(requested_variants),
            "common_eval_args": _redacted_command(common_args),
            "mujoco_gl": mujoco_gl,
        },
        "primary_metric": "evaluation_only_privileged_oracle_task_success",
        "scoring_protocol": {
            "timing": "once_after_complete_orchestrator_rollout",
            "source": "native RoboCasa task environment _check_success",
            "same_scorer_for_both_variants": True,
            "affects_runtime": False,
            "note": (
                "Atomic-step and grounding rates are runtime-reported "
                "diagnostics, not common-oracle accuracy metrics."
            ),
        },
        "verification_schedules": {
            name: dict(verification_schedules[name])
            for name in requested_variants
        },
        "variants": {name: VARIANTS[name] for name in requested_variants},
        "aggregate": aggregate,
        "paired_comparison": _paired_comparison(rows),
        "runtime_atomic_task_metrics": _atomic_task_metrics(records),
        "runs": list(rows),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_root / "runs.csv", rows)
    print("\n[Reasoning Ablation] Aggregate results")
    for variant, metrics in aggregate.items():
        print(
            f"  {variant}: {metrics['num_successes']}/"
            f"{metrics['num_completed']} "
            "oracle_task_success_rate="
            f"{metrics['oracle_task_success_rate']:.3f}; "
            "runtime_reported_success_rate="
            f"{metrics['runtime_reported_success_rate']:.3f}"
        )
    comparison = summary["paired_comparison"]
    if comparison is not None:
        print(
            "  VLM - privileged paired success delta="
            f"{comparison['oracle_success_rate_delta_vlm_minus_privileged']:+.3f}; "
            "privileged-only/vlm-only="
            f"{comparison['privileged_only_success']}/"
            f"{comparison['vlm_only_success']}; "
            f"McNemar p={comparison['mcnemar_exact_two_sided_p']:.4f}"
        )
    print(f"[Reasoning Ablation] Summary: {output_root / 'summary.json'}")


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    python = str(args.python_executable)
    rows: list[dict[str, Any]] = []
    records: list[tuple[str, int, Path]] = []
    had_infrastructure_error = False
    verification_schedules = {
        variant: _variant_verification_schedule(args, variant)
        for variant in VARIANTS
    }

    for seed in args.seeds:
        plan_root = output_root / "plans" / f"seed_{seed}"
        plan_path = _latest(plan_root, "task_plan.json") if args.resume else None
        if plan_path is None:
            plan_command = [
                python,
                str(EVAL_SCRIPT),
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
            plan_path = (
                Path("<generated-task-plan.json>")
                if args.dry_run
                else _latest(plan_root, "task_plan.json")
            )
            if return_code != 0 or plan_path is None:
                had_infrastructure_error = True
                error = f"plan generation failed with return code {return_code}"
                for variant in args.variants:
                    rows.append(
                        _result_row(
                            variant=variant,
                            seed=seed,
                            verification_schedule=verification_schedules[variant],
                            result_path=None,
                            return_code=return_code,
                            error=error,
                        )
                    )
                if args.stop_on_error:
                    break
                continue

        for variant in args.variants:
            schedule = verification_schedules[variant]
            run_root = output_root / "runs" / variant / f"seed_{seed}"
            result_path = (
                _latest(run_root, "long_horizon_result.json")
                if args.resume
                else None
            )
            if result_path is not None and not _has_available_oracle(result_path):
                print(
                    f"[Reasoning Ablation] Ignore unscored legacy result: "
                    f"{result_path}"
                )
                result_path = None
            if result_path is not None and not _result_matches_configuration(
                result_path, variant=variant, schedule=schedule
            ):
                print(
                    "[Reasoning Ablation] Ignore result with mismatched "
                    f"configuration: {result_path}"
                )
                result_path = None
            return_code = 0
            if result_path is None:
                command = [
                    python,
                    str(EVAL_SCRIPT),
                    *args.common_args,
                    "--seed",
                    str(seed),
                    "--vlm_plan_json",
                    str(plan_path),
                    "--reasoning_mode",
                    str(VARIANTS[variant]["reasoning_mode"]),
                    "--verification_mode",
                    str(schedule["mode"]),
                    "--pi05_verify_interval",
                    str(schedule["interval"]),
                    "--pi05_min_steps_before_verify",
                    str(schedule["min_steps_before_verify"]),
                    "--output_root",
                    str(run_root),
                ]
                return_code = _run_command(
                    command,
                    log_path=run_root / "run.log",
                    mujoco_gl=args.mujoco_gl,
                    dry_run=args.dry_run,
                )
                if args.dry_run:
                    continue
                result_path = _latest(run_root, "long_horizon_result.json")
            else:
                print(
                    f"[Reasoning Ablation] Resume {variant} seed={seed}: "
                    f"{result_path}"
                )
            infrastructure_error = return_code not in (0, 2) or result_path is None
            if infrastructure_error:
                had_infrastructure_error = True
            row = _result_row(
                variant=variant,
                seed=seed,
                verification_schedule=schedule,
                result_path=result_path,
                return_code=return_code,
                error=(
                    f"evaluation failed with return code {return_code}"
                    if infrastructure_error
                    else None
                ),
            )
            rows.append(row)
            if not row["completed"]:
                infrastructure_error = True
                had_infrastructure_error = True
            if result_path is not None:
                records.append((variant, seed, result_path))
            if infrastructure_error and args.stop_on_error:
                break
        if had_infrastructure_error and args.stop_on_error:
            break

    if not args.dry_run:
        _write_outputs(
            output_root,
            rows,
            records,
            requested_seeds=args.seeds,
            requested_variants=args.variants,
            verification_schedules=verification_schedules,
            common_args=args.common_args,
            mujoco_gl=args.mujoco_gl,
        )
    return 1 if had_infrastructure_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
