from __future__ import annotations

import numpy as np

from robocasa.eval_xr1_end_to_end import run_direct_rollout


class _Client:
    def __init__(self) -> None:
        self.instructions: list[str] = []
        self.actions = 0

    def set_instruction(self, instruction: str) -> None:
        self.instructions.append(instruction)

    def act(self) -> np.ndarray:
        self.actions += 1
        return np.zeros(12, dtype=np.float32)


class _Env:
    def __init__(self, *, done_at: int | None = None) -> None:
        self.steps = 0
        self.done_at = done_at

    def step(self, action):
        assert set(action) == {
            "action.end_effector_position",
            "action.end_effector_rotation",
            "action.gripper_close",
            "action.base_motion",
            "action.control_mode",
        }
        self.steps += 1
        done = self.done_at is not None and self.steps >= self.done_at
        return {}, float(self.steps), done, {"step": self.steps}


def test_direct_rollout_uses_one_full_instruction_and_no_early_oracle() -> None:
    client = _Client()
    env = _Env()

    result = run_direct_rollout(
        env=env,
        client=client,
        instruction="complete the full task",
        horizon=5,
    )

    assert client.instructions == ["complete the full task"]
    assert client.actions == 5
    assert env.steps == 5
    assert result["completed_steps"] == 5
    assert result["environment_done"] is False
    assert result["reward_sum"] == 15.0


def test_direct_rollout_stops_only_when_environment_ends() -> None:
    client = _Client()
    env = _Env(done_at=3)

    result = run_direct_rollout(
        env=env,
        client=client,
        instruction="complete the full task",
        horizon=10,
    )

    assert client.actions == 3
    assert result["completed_steps"] == 3
    assert result["environment_done"] is True
