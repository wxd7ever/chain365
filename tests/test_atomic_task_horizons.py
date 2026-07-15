from __future__ import annotations

import sys
from pathlib import Path


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from atomic_task_horizons import load_atomic_task_horizons  # noqa: E402
import pi05_rollout  # noqa: E402
from atomic_task_policy_adapter import RemoteAtomicTaskPolicyAdapter  # noqa: E402


def test_registry_horizons_match_current_atomic_tasks():
    horizons = load_atomic_task_horizons()
    assert horizons["PickPlaceCounterToMicrowave"] == 1050
    assert horizons["CloseMicrowave"] == 450
    assert horizons["TurnOnMicrowave"] == 450


class Client:
    def infer(self, payload):
        raise AssertionError("patched rollout should be used")


def test_adapter_uses_registry_horizon(monkeypatch, tmp_path):
    captured = {}

    def fake_rollout(**kwargs):
        captured.update(kwargs)
        return True, {"Final_Verification": {"status": "success"}}

    monkeypatch.setattr(pi05_rollout, "execute_pi05_atomic_task_policy", fake_rollout)
    adapter = RemoteAtomicTaskPolicyAdapter(
        client=Client(),
        verifier=lambda **kwargs: {},
        log_dir=tmp_path,
        atomic_task_horizon=300,
        use_registry_horizons=True,
    )
    result = adapter.execute(
        env=object(),
        scheduler_query={
            "atomic_task_call": {
                "subgoal_id": "g1",
                "atomic_task": "PickPlaceCounterToMicrowave",
                "policy_prompt": "Put the food in the microwave.",
                "arguments": {},
                "termination_condition": {
                    "predicate": "inside",
                    "subject": "obj",
                    "object": "microwave",
                },
            }
        },
        episode_id=0,
    )
    assert captured["horizon"] == 1050
    assert result["configured_horizon"] == 1050
    assert result["horizon_source"] == "robocasa_dataset_registry"
