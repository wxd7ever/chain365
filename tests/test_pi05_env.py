from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from pi05_env import RawRoboCasaPi05Env  # noqa: E402
import pi05_env  # noqa: E402


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


def test_full_episode_video_streams_across_all_environment_steps(
    monkeypatch, tmp_path: Path
):
    class Writer:
        def __init__(self):
            self.frames = []
            self.closed = False

        def append_data(self, frame):
            self.frames.append(np.asarray(frame).copy())

        def close(self):
            self.closed = True

    writer = Writer()
    monkeypatch.setattr(
        pi05_env.imageio,
        "get_writer",
        lambda *args, **kwargs: writer,
    )
    env = RawRoboCasaPi05Env(RawEnv())
    initial = env.reset()
    video_path = tmp_path / "episode_full.mp4"

    started = env.start_episode_video_recording(
        video_path, fps=20, frame_stride=2
    )
    for _ in range(3):
        env.step(np.zeros(12, dtype=np.float32))
    finalized = env.finalize_episode_video_recording()
    finalized_again = env.finalize_episode_video_recording()

    assert started["frame_count"] == 1
    assert len(writer.frames) == 2
    assert np.array_equal(
        writer.frames[0], initial["robot0_agentview_left_image"]
    )
    assert writer.closed is True
    assert finalized["enabled"] is True
    assert finalized["path"] == str(video_path)
    assert finalized["frame_count"] == 2
    assert finalized["env_steps_seen"] == 3
    assert finalized["finalized"] is True
    assert finalized["error"] is None
    assert finalized_again == finalized
