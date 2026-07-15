"""Checkpoint-free RoboCasa environment adapter for remote pi0.5 rollouts."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


POLICY_CAMERA_NAMES = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)


def _process_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Flip raw MuJoCo camera images into the policy's upright HWC convention."""

    processed: dict[str, Any] = {}
    for key, value in observation.items():
        array = np.asarray(value)
        if key.endswith("_image"):
            if array.ndim != 3:
                raise ValueError(f"camera observation {key!r} must be 3-D, got {array.shape}")
            array = np.ascontiguousarray(array[::-1])
        processed[key] = array
    return processed


class RawRoboCasaPi05Env:
    """Expose raw RoboSuite observations/actions through the pi0.5 rollout API.

    The adapter deliberately keeps the native flat 12-D PandaOmron action vector.
    It only flips renderer images once and caches the latest observation so atomic
    tasks can continue without resetting the underlying simulator.
    """

    def __init__(self, env: Any):
        self.env = env
        self._observation: dict[str, Any] | None = None

    @property
    def unwrapped(self) -> "RawRoboCasaPi05Env":
        return self

    @property
    def action_dimension(self) -> int:
        action_spec = getattr(self.env, "action_spec", None)
        if action_spec is None:
            return int(getattr(self.env, "action_dim"))
        return int(np.asarray(action_spec[0]).size)

    def reset(self) -> dict[str, Any]:
        self._observation = _process_observation(self.env.reset())
        return self._observation

    def get_observation(self) -> dict[str, Any]:
        if self._observation is None:
            raise RuntimeError("RoboCasa environment must be reset before reading observation")
        return self._observation

    def step(self, action: np.ndarray):
        result = self.env.step(np.asarray(action, dtype=np.float32))
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise RuntimeError("raw RoboCasa env.step must return (obs, reward, done, info)")
        observation, reward, done, info = result
        self._observation = _process_observation(observation)
        return self._observation, reward, done, info

    def render(
        self,
        mode: str = "rgb_array",
        height: int = 512,
        width: int = 512,
    ) -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError(f"only rgb_array rendering is supported, got {mode!r}")
        frame = self.env.sim.render(
            height=height,
            width=width,
            camera_name="robot0_agentview_left",
        )
        return np.ascontiguousarray(frame[::-1])

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def create_pi05_env(
    *,
    env_name: str,
    layout_id: int,
    style_id: int,
    seed: int,
    camera_size: int = 256,
) -> RawRoboCasaPi05Env:
    """Create a native RoboCasa task without any local policy checkpoint."""

    if camera_size <= 0:
        raise ValueError(f"camera_size must be positive, got {camera_size}")
    if layout_id <= 0 or style_id <= 0:
        raise ValueError("RoboCasa layout_id and style_id must be 1-based positive integers")
    try:
        from .utils.env_utils import create_env
    except ImportError:
        from robocasa.utils.env_utils import create_env

    env = create_env(
        env_name=env_name,
        robots="PandaOmron",
        camera_names=list(POLICY_CAMERA_NAMES),
        camera_widths=camera_size,
        camera_heights=camera_size,
        seed=seed,
        layout_and_style_ids=[(layout_id, style_id)],
    )
    return RawRoboCasaPi05Env(env)
