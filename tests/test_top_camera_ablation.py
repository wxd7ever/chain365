from __future__ import annotations

import json
import sys
from pathlib import Path

from robocasa.eval_top_camera_ablation import (
    VARIANT_CAMERAS,
    _aggregate,
    _paired_comparison,
    _result_row,
    main,
)
from robocasa.eval_vlm_pi05_multiview import DEFAULT_LOCAL_POSE_CAMERAS


def test_multiview_defaults_include_top_and_wrist():
    assert "robot0_topview" in DEFAULT_LOCAL_POSE_CAMERAS
    assert "robot0_eye_in_hand" in DEFAULT_LOCAL_POSE_CAMERAS
    assert "robot0_eye_in_hand" in VARIANT_CAMERAS["refiner_no_top"]
    assert "robot0_topview" not in VARIANT_CAMERAS["refiner_no_top"]
    assert "robot0_eye_in_hand" not in VARIANT_CAMERAS["refiner_top_no_wrist"]


def test_result_metrics_and_paired_comparison(tmp_path: Path):
    baseline_path = tmp_path / "baseline.json"
    top_path = tmp_path / "top.json"
    baseline_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "success": False,
                "num_atomic_attempts": 4,
                "step_results": [{"success": False, "num_attempts": 3}],
            }
        ),
        encoding="utf-8",
    )
    top_path.write_text(
        json.dumps(
            {
                "status": "success",
                "success": True,
                "num_atomic_attempts": 2,
                "step_results": [{"success": True, "num_attempts": 1}],
                "local_pose_refinement_results": [
                    {
                        "success": True,
                        "num_executed_actions": 2,
                        "total_env_steps": 14,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        _result_row(
            variant="baseline",
            seed=1,
            result_path=baseline_path,
            return_code=2,
        ),
        _result_row(
            variant="refiner_top",
            seed=1,
            result_path=top_path,
            return_code=0,
        ),
    ]
    aggregate = _aggregate(rows)
    comparison = _paired_comparison(rows, "baseline", "refiner_top")

    assert aggregate["baseline"]["success_rate"] == 0.0
    assert aggregate["refiner_top"]["success_rate"] == 1.0
    assert rows[1]["num_refinement_actions"] == 2
    assert comparison is not None
    assert comparison["success_rate_delta"] == 1.0
    assert comparison["wins"] == 1


def test_dry_run_builds_all_variant_commands(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_top_camera_ablation.py",
            "--dry_run",
            "--seeds",
            "7",
            "--output_root",
            str(tmp_path),
            "--",
            "--env_name",
            "SteamInMicrowave",
            "--long_horizon_task",
            "steam a vegetable",
        ],
    )
    assert main() == 0
    output = capsys.readouterr().out
    assert "eval_vlm_pi05.py" in output
    assert "eval_vlm_pi05_multiview.py" in output
    assert "robot0_topview" in output
    assert "robot0_eye_in_hand" in output

