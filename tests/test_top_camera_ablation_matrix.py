from __future__ import annotations

import sys
from pathlib import Path

from robocasa import eval_top_camera_ablation_matrix as matrix


def test_parse_args_splits_matrix_automation_and_eval_args(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_top_camera_ablation_matrix.py",
            "--layout_ids",
            "1",
            "2",
            "--style_ids",
            "5",
            "10",
            "--dry_run",
            "--seeds",
            "7",
            "--",
            "--env_name",
            "SteamInMicrowave",
            "--long_horizon_task",
            "steam a vegetable",
        ],
    )

    args, forwarded = matrix.parse_args()
    automation_args, common_args = matrix._split_forwarded(forwarded)

    assert args.layout_ids == [1, 2]
    assert args.style_ids == [5, 10]
    assert automation_args == ["--dry_run", "--seeds", "7"]
    assert common_args == [
        "--env_name",
        "SteamInMicrowave",
        "--long_horizon_task",
        "steam a vegetable",
    ]


def test_paired_comparison_uses_layout_style_and_seed():
    rows = [
        {
            "layout_id": 1,
            "style_id": 1,
            "seed": 1,
            "variant": "baseline",
            "completed": True,
            "success": False,
        },
        {
            "layout_id": 1,
            "style_id": 1,
            "seed": 1,
            "variant": "refiner_top",
            "completed": True,
            "success": True,
        },
        {
            "layout_id": 2,
            "style_id": 1,
            "seed": 1,
            "variant": "baseline",
            "completed": True,
            "success": True,
        },
        {
            "layout_id": 2,
            "style_id": 1,
            "seed": 1,
            "variant": "refiner_top",
            "completed": True,
            "success": False,
        },
    ]

    comparison = matrix._paired_comparison(rows, "baseline", "refiner_top")

    assert comparison is not None
    assert comparison["num_paired_runs"] == 2
    assert comparison["success_rate_delta"] == 0.0
    assert comparison["wins"] == 1
    assert comparison["losses"] == 1


def test_dry_run_traverses_cartesian_product(monkeypatch, tmp_path: Path):
    commands: list[list[str]] = []
    monkeypatch.setattr(matrix, "_run", lambda command: commands.append(list(command)) or 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_top_camera_ablation_matrix.py",
            "--layout_ids",
            "1",
            "2",
            "--style_ids",
            "5",
            "10",
            "--matrix_output_root",
            str(tmp_path),
            "--dry_run",
            "--seeds",
            "3",
            "--",
            "--env_name",
            "SteamInMicrowave",
            "--long_horizon_task",
            "steam a vegetable",
        ],
    )

    assert matrix.main() == 0
    assert len(commands) == 4
    scene_pairs = {
        (
            command[command.index("--layout_id") + 1],
            command[command.index("--style_id") + 1],
        )
        for command in commands
    }
    assert scene_pairs == {("1", "5"), ("1", "10"), ("2", "5"), ("2", "10")}
    assert all("--dry_run" in command for command in commands)

