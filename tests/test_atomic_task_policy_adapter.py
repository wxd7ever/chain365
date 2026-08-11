from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

import pi05_rollout  # noqa: E402
from atomic_task_policy_adapter import RemoteAtomicTaskPolicyAdapter  # noqa: E402


class Client:
    def infer(self, payload):
        raise AssertionError("patched rollout should be used")


def query():
    return {
        "atomic_task_call": {
            "subgoal_id": "g1",
            "atomic_task": "OpenMicrowave",
            "arguments": {"fixture_id": "microwave_1"},
            "termination_condition": {"predicate": "open", "subject": "microwave_1"},
        }
    }


def test_adapter_builds_prompt_calls_rollout_and_returns_module_result(monkeypatch, tmp_path):
    captured = {}

    def fake_rollout(**kwargs):
        captured.update(kwargs)
        return True, {
            "Final_Verification": {"status": "success"},
            "Prompt": "Open the microwave door.",
        }

    monkeypatch.setattr(pi05_rollout, "execute_pi05_atomic_task_policy", fake_rollout)
    adapter = RemoteAtomicTaskPolicyAdapter(
        client=Client(), verifier=lambda **kwargs: {}, log_dir=tmp_path
    )
    result = adapter.execute(env=object(), scheduler_query=query(), episode_id=3)
    assert captured["episode_id"] == 3
    assert captured["held_object_guard"] is True
    assert captured["held_object_hold_confirmation_steps"] == 2
    assert captured["held_object_drop_confirmation_steps"] == 2
    assert result["module"] == "ATOMIC_TASK_POLICY"
    assert result["atomic_task"] == "OpenMicrowave"
    assert result["prompt"] == "Open the microwave door."
    assert result["status"] == "success"


def test_adapter_requires_atomic_task_call(tmp_path):
    adapter = RemoteAtomicTaskPolicyAdapter(
        client=Client(), verifier=lambda **kwargs: {}, log_dir=tmp_path
    )
    with pytest.raises(ValueError, match="atomic_task_call"):
        adapter.execute(env=object(), scheduler_query={}, episode_id=0)
