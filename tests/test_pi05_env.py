from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from pi05_env import RawRoboCasaPi05Env  # noqa: E402


def observation():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    return {"robot0_agentview_left_image": image, "robot0_base_pos": np.zeros(3)}


class RawEnv:
    action_spec = (np.full(12, -1.0), np.full(12, 1.0))

    def __init__(self):
        self.actions = []

    def reset(self):
        return observation()

    def step(self, action):
        self.actions.append(action)
        return observation(), 0.0, False, {}

    def close(self):
        pass


def test_raw_env_adapter_flips_images_and_preserves_flat_actions():
    raw = RawEnv()
    env = RawRoboCasaPi05Env(raw)
    initial = env.reset()
    assert env.action_dimension == 12
    assert np.array_equal(
        initial["robot0_agentview_left_image"],
        observation()["robot0_agentview_left_image"][::-1],
    )
    action = np.arange(12, dtype=np.float32)
    current, _, _, _ = env.step(action)
    assert np.array_equal(raw.actions[0], action)
    assert current is env.get_observation()
