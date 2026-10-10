from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from pi05_env import RawRoboCasaPi05Env  # noqa: E402
from xr1_policy_client import (  # noqa: E402
    CAMERA_KEYS,
    XR1RoboCasaPolicy,
    collect_images,
    observation_to_state,
    sample_history,
)


def observation(value: float = 0.0):
    image = np.full((8, 8, 3), value, dtype=np.uint8)
    return {
        "robot0_agentview_left_image": image.copy(),
        "robot0_agentview_right_image": image.copy(),
        "robot0_eye_in_hand_image": image.copy(),
        "robot0_base_to_eef_pos": np.array([value, 0.0, 0.0]),
        "robot0_base_to_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.1, 0.2]),
        "robot0_base_pos": np.array([0.0, value, 0.0]),
        "robot0_base_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    }


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def list_robot_types(self):
        return ["robocasa365"]

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"input_ids": np.zeros((1, 2), dtype=np.int64)}


class FakeTransport:
    def __init__(self):
        self.requests = []
        self.closed = False

    def infer(self, request, *, robot_type):
        self.requests.append((request, robot_type))
        return np.zeros((1, 32, 12), dtype=np.float32)

    def close(self):
        self.closed = True


def test_official_observation_schema_state_and_history():
    obs = observation(3.0)
    assert observation_to_state(obs).shape == (14,)
    images = collect_images(obs)
    assert tuple(images) == CAMERA_KEYS
    assert all(image.shape == (8, 8, 3) for image in images.values())

    from collections import deque

    history = deque([np.array([0]), np.array([1]), np.array([2])], maxlen=7)
    sampled = sample_history(history, length=4, interval=2)
    assert sampled[:, 0].tolist() == [0, 0, 0, 2]


def test_policy_preserves_history_across_skill_switch_and_clears_actions():
    processor = FakeProcessor()
    transport = FakeTransport()
    policy = XR1RoboCasaPolicy(
        processor=processor,
        transport=transport,
        replan_steps=16,
    )
    policy.reset(observation(0.0))
    policy.observe(observation(1.0))
    policy.set_instruction("Pick up the bowl.")
    action = policy.act()
    assert action.shape == (12,)
    assert len(policy.action_queue) == 15
    assert processor.calls[-1][1]["state"].shape == (1, 4, 60)
    assert transport.requests[-1][1] == "robocasa365"

    history_size = len(policy.state_queue)
    policy.set_instruction("Place the bowl in the microwave.")
    assert len(policy.action_queue) == 0
    assert len(policy.state_queue) == history_size
    messages = processor.calls[-1][0]
    prompt = messages[0]["content"][-1]["text"]
    assert "Pick up the bowl." in prompt
    policy.close()
    assert transport.closed is True


class RawEnv:
    def __init__(self):
        self.action = None

    def step(self, action):
        self.action = np.asarray(action)
        return observation(), 0.0, False, {}


def test_raw_env_accepts_official_converted_action_dict():
    raw = RawEnv()
    env = RawRoboCasaPi05Env(raw)
    action = np.arange(12, dtype=np.float32)
    converted = {
        "action.end_effector_position": action[0:3],
        "action.end_effector_rotation": action[3:6],
        "action.gripper_close": action[6:7],
        "action.base_motion": action[7:11],
        "action.control_mode": action[11:12],
    }
    seen = []
    env.add_observation_listener(seen.append)
    env.step(converted)
    assert np.array_equal(raw.action, action)
    assert len(seen) == 1
