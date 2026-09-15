from __future__ import annotations

import json
import sys
from pathlib import Path

from robocasa.eval_reasoning_backend_ablation import (
    _aggregate,
    _atomic_task_metrics,
    _paired_comparison,
    _result_row,
    main,
)
from robocasa.eval_vlm_pi05 import _evaluation_only_privileged_oracle


PRIVILEGED_SCHEDULE = {
    "mode": "periodic",
    "interval": 5,
    "min_steps_before_verify": 10,
}
VLM_SCHEDULE = {
    "mode": "periodic",
    "interval": 60,
    "min_steps_before_verify": 60,
}


def _write_result(
    path: Path,
    *,
    success: bool,
    atomic_successes: list[bool],
    oracle_success: bool | None = None,
) -> None:
    if oracle_success is None:
        oracle_success = success
    steps = []
    for index, atomic_success in enumerate(atomic_successes):
        steps.append(
            {
                "subgoal_id": f"step_{index}",
                "atomic_task": "PickObject" if index == 0 else "PlaceObject",
                "success": atomic_success,
                "num_attempts": 1 if atomic_success else 3,
                "rollout_logs": {
                    "Horizon": 60 if atomic_success else 600,
                    "Verification_History": [{}, {}],
                    "Final_Verification": {
                        "status": "success" if atomic_success else "failed",
                        "failure_code": (
                            None if atomic_success else "VLM_CRITIC_FAILURE"
                        ),
                    },
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "status": "success" if success else "failed",
                "success": success,
                "runtime_reported_success": success,
                "evaluation_only_privileged_oracle": {
                    "available": True,
                    "success": oracle_success,
                    "source": "FakeTask._check_success",
                    "error": None,
                    "affects_runtime": False,
                },
                "num_planned_steps": 2,
                "num_executed_steps": len(steps),
                "num_atomic_attempts": sum(
                    int(item["num_attempts"]) for item in steps
                ),
                "num_inserted_navigation_steps": 0,
                "grounding_events": [
                    {"status": "grounded"} for _ in steps
                ],
                "step_results": steps,
            }
        ),
        encoding="utf-8",
    )


def test_result_aggregation_and_paired_statistics(tmp_path: Path):
    privileged_path = tmp_path / "privileged.json"
    vlm_path = tmp_path / "vlm.json"
    _write_result(
        privileged_path,
        success=True,
        atomic_successes=[True, True],
    )
    _write_result(
        vlm_path,
        success=False,
        atomic_successes=[True, False],
    )
    rows = [
        _result_row(
            variant="privileged",
            seed=7,
            verification_schedule=PRIVILEGED_SCHEDULE,
            result_path=privileged_path,
            return_code=0,
        ),
        _result_row(
            variant="vlm",
            seed=7,
            verification_schedule=VLM_SCHEDULE,
            result_path=vlm_path,
            return_code=2,
        ),
    ]

    aggregate = _aggregate(rows)
    paired = _paired_comparison(rows)
    atomic = _atomic_task_metrics(
        [
            ("privileged", 7, privileged_path),
            ("vlm", 7, vlm_path),
        ]
    )

    assert aggregate["privileged"]["oracle_task_success_rate"] == 1.0
    assert aggregate["vlm"]["oracle_task_success_rate"] == 0.0
    assert rows[1]["runtime_atomic_step_success_rate"] == 0.5
    assert rows[1]["num_retries"] == 2
    assert rows[1]["failure_stage"] == "atomic_policy_or_verifier"
    assert paired is not None
    assert paired["oracle_success_rate_delta_vlm_minus_privileged"] == -1.0
    assert paired["privileged_only_success"] == 1
    assert paired["mcnemar_exact_two_sided_p"] == 1.0
    assert (
        atomic["vlm"]["PlaceObject"]["runtime_reported_success_rate"]
        == 0.0
    )


def test_primary_success_uses_common_oracle_and_tracks_runtime_false_positive(
    tmp_path: Path,
):
    result_path = tmp_path / "false_positive.json"
    _write_result(
        result_path,
        success=True,
        oracle_success=False,
        atomic_successes=[True, True],
    )

    row = _result_row(
        variant="vlm",
        seed=9,
        verification_schedule=VLM_SCHEDULE,
        result_path=result_path,
        return_code=0,
    )

    assert row["completed"] is True
    assert row["success"] is False
    assert row["oracle_task_success"] is False
    assert row["runtime_success"] is True
    assert row["runtime_oracle_agreement"] is False
    assert row["failure_stage"] == "runtime_false_positive"


def test_evaluation_oracle_unwraps_env_and_is_read_only():
    class TaskEnv:
        def _check_success(self):
            return True

    class Wrapper:
        env = TaskEnv()

    result = _evaluation_only_privileged_oracle(Wrapper())

    assert result == {
        "available": True,
        "success": True,
        "source": "TaskEnv._check_success",
        "error": None,
        "affects_runtime": False,
    }


def test_dry_run_uses_shared_plan_and_both_reasoning_modes(
    monkeypatch, tmp_path: Path, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_reasoning_backend_ablation.py",
            "--dry_run",
            "--seeds",
            "3",
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
    assert output.count("--vlm_plan_json <generated-task-plan.json>") == 2
    assert "--reasoning_mode simulator" in output
    assert "--reasoning_mode vlm_critic" in output
    assert "--reasoning_mode simulator --verification_mode periodic" in output
    assert "--reasoning_mode vlm_critic --verification_mode periodic" in output
    assert output.count("--pi05_verify_interval 5") == 1
    assert output.count("--pi05_verify_interval 60") == 1
