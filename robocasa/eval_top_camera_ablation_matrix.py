#!/usr/bin/env python3
"""Run the top-camera ablation over a layout × style scene matrix."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

try:
    from .eval_top_camera_ablation import (
        VARIANT_CAMERAS,
        _aggregate,
        _redacted_command,
    )
except ImportError:
    from eval_top_camera_ablation import (
        VARIANT_CAMERAS,
        _aggregate,
        _redacted_command,
    )


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SINGLE_SCENE_SCRIPT = THIS_DIR / "eval_top_camera_ablation.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Traverse layout_ids × style_ids and invoke the paired seed/camera "
            "ablation for every scene. All eval_top_camera_ablation.py arguments "
            "that follow are forwarded unchanged."
        )
    )
    parser.add_argument("--layout_ids", nargs="+", type=int, default=[1])
    parser.add_argument("--style_ids", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--matrix_output_root",
        type=Path,
        default=THIS_DIR / "outputs_top_camera_ablation_matrix",
    )
    parser.add_argument(
        "--matrix_python",
        default=sys.executable,
        help="Python executable used to start each single-scene ablation.",
    )
    args, forwarded = parser.parse_known_args()
    for name in ("layout_ids", "style_ids"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"--{name} must not contain duplicates")
        if any(value < 1 or value > 60 for value in values):
            parser.error(f"--{name} values must be RoboCasa IDs in [1, 60]")
    if "--" not in forwarded:
        parser.error(
            "forwarded arguments must contain -- before ordinary RoboCasa eval arguments"
        )
    separator = forwarded.index("--")
    automation_args = forwarded[:separator]
    common_args = forwarded[separator + 1 :]
    if not common_args:
        parser.error("ordinary RoboCasa eval arguments are required after --")
    for option in ("--output_root", "--layout_id", "--style_id"):
        values = automation_args if option == "--output_root" else common_args
        if any(item == option or item.startswith(f"{option}=") for item in values):
            parser.error(f"{option} is controlled by the matrix runner")
    return args, [*automation_args, "--", *common_args]


def _split_forwarded(forwarded: Sequence[str]) -> tuple[list[str], list[str]]:
    separator = forwarded.index("--")
    return list(forwarded[:separator]), list(forwarded[separator + 1 :])


def _run(command: Sequence[str]) -> int:
    printable = " ".join(_redacted_command(command))
    print(f"[SceneMatrix] {printable}")
    return int(subprocess.run(list(command), cwd=REPO_ROOT, check=False).returncode)


def _load_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object in {path}")
    return dict(value)


def _paired_comparison(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> dict[str, Any] | None:
    def key(row: Mapping[str, Any]) -> tuple[int, int, int]:
        return int(row["layout_id"]), int(row["style_id"]), int(row["seed"])

    left_by_scene = {
        key(row): row
        for row in rows
        if row.get("variant") == left and row.get("completed")
    }
    right_by_scene = {
        key(row): row
        for row in rows
        if row.get("variant") == right and row.get("completed")
    }
    paired_keys = sorted(set(left_by_scene).intersection(right_by_scene))
    if not paired_keys:
        return None
    deltas = [
        int(bool(right_by_scene[item]["success"]))
        - int(bool(left_by_scene[item]["success"]))
        for item in paired_keys
    ]
    return {
        "left": left,
        "right": right,
        "interpretation": f"positive means {right} is better",
        "num_paired_runs": len(paired_keys),
        "paired_scene_seeds": [
            {"layout_id": layout, "style_id": style, "seed": seed}
            for layout, style, seed in paired_keys
        ],
        "success_rate_delta": fmean(deltas),
        "wins": sum(delta > 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
    }


def _write_matrix_summary(
    *,
    output_root: Path,
    layout_ids: Sequence[int],
    style_ids: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    scene_summaries: Sequence[Mapping[str, Any]],
) -> None:
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
        "layout_ids": list(layout_ids),
        "style_ids": list(style_ids),
        "num_scene_combinations": len(layout_ids) * len(style_ids),
        "pairing_key": ["layout_id", "style_id", "seed"],
        "variants": {
            name: {"camera_names": list(cameras)}
            for name, cameras in VARIANT_CAMERAS.items()
        },
        "aggregate": _aggregate(rows),
        "paired_comparisons": comparisons,
        "scene_summaries": list(scene_summaries),
        "runs": list(rows),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "matrix_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if rows:
        with (output_root / "matrix_runs.csv").open(
            "w", encoding="utf-8", newline=""
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print("\n[SceneMatrix] Aggregate results")
    for variant, metrics in summary["aggregate"].items():
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
    print(f"[SceneMatrix] Summary: {output_root / 'matrix_summary.json'}")


def main() -> int:
    args, forwarded = parse_args()
    automation_args, common_args = _split_forwarded(forwarded)
    output_root = args.matrix_output_root.resolve()
    is_dry_run = "--dry_run" in automation_args
    stop_on_error = "--stop_on_error" in automation_args
    rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    had_error = False
    combinations = [
        (layout_id, style_id)
        for layout_id in args.layout_ids
        for style_id in args.style_ids
    ]
    print(
        f"[SceneMatrix] scenes={len(combinations)} "
        f"layouts={args.layout_ids} styles={args.style_ids}"
    )

    for layout_id, style_id in combinations:
        scene_root = (
            output_root / f"layout_{layout_id:03d}_style_{style_id:03d}"
        )
        command = [
            str(args.matrix_python),
            str(SINGLE_SCENE_SCRIPT),
            *automation_args,
            "--output_root",
            str(scene_root),
            "--",
            *common_args,
            "--layout_id",
            str(layout_id),
            "--style_id",
            str(style_id),
        ]
        return_code = _run(command)
        if is_dry_run:
            continue
        summary_path = scene_root / "automation_summary.json"
        if return_code != 0 or not summary_path.is_file():
            had_error = True
            print(
                f"[SceneMatrix] layout={layout_id} style={style_id} failed: "
                f"return_code={return_code}"
            )
            if stop_on_error:
                break
            continue
        scene_summary = _load_summary(summary_path)
        scene_summaries.append(
            {
                "layout_id": layout_id,
                "style_id": style_id,
                "summary_path": str(summary_path),
                "aggregate": scene_summary.get("aggregate", {}),
                "paired_comparisons": scene_summary.get(
                    "paired_comparisons", []
                ),
            }
        )
        for raw_row in scene_summary.get("runs", []):
            if not isinstance(raw_row, Mapping):
                continue
            row = {
                "layout_id": layout_id,
                "style_id": style_id,
                **dict(raw_row),
            }
            rows.append(row)

    if not is_dry_run:
        _write_matrix_summary(
            output_root=output_root,
            layout_ids=args.layout_ids,
            style_ids=args.style_ids,
            rows=rows,
            scene_summaries=scene_summaries,
        )
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())

