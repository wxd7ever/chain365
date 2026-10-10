"""Checkpoint-free RoboCasa environment adapter for remote pi0.5 rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import imageio.v2 as imageio
import numpy as np


POLICY_CAMERA_NAMES = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)

# These cameras are intentionally separate from POLICY_CAMERA_NAMES. The remote
# pi0.5 checkpoint was trained with exactly the three policy views above, while
# local work-pose refinement may render additional cameras on demand.
LOCAL_POSE_CAMERA_NAMES = (
    "robot0_topview",
    "robot0_frontview",
    "robot0_agentview_left",
    "robot0_agentview_right",
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
        self._observation_listeners: list[
            Callable[[Mapping[str, Any]], None]
        ] = []
        self._episode_video_writer: Any | None = None
        self._episode_video_step = 0
        self._episode_video_metadata: dict[str, Any] = {
            "enabled": False,
            "path": None,
            "camera": None,
            "fps": None,
            "frame_stride": None,
            "frame_count": 0,
            "env_steps_seen": 0,
            "finalized": True,
            "error": None,
        }

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

    def add_observation_listener(
        self, listener: Callable[[Mapping[str, Any]], None]
    ) -> None:
        """Register one callback invoked exactly once after every env.step()."""
        if listener not in self._observation_listeners:
            self._observation_listeners.append(listener)

    def step(self, action: np.ndarray):
        if isinstance(action, Mapping):
            required = (
                "action.end_effector_position",
                "action.end_effector_rotation",
                "action.gripper_close",
                "action.base_motion",
                "action.control_mode",
            )
            missing = [key for key in required if key not in action]
            if missing:
                raise KeyError(
                    f"RoboCasa365 action dictionary is missing keys {missing}"
                )
            action = np.concatenate(
                [
                    np.asarray(action["action.end_effector_position"]).reshape(-1),
                    np.asarray(action["action.end_effector_rotation"]).reshape(-1),
                    np.asarray(action["action.gripper_close"]).reshape(-1),
                    np.asarray(action["action.base_motion"]).reshape(-1),
                    np.asarray(action["action.control_mode"]).reshape(-1),
                ]
            )
        flat_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if flat_action.shape != (12,):
            raise ValueError(f"PandaOmron policy action must resolve to 12 values, got {flat_action.shape}")
        result = self.env.step(flat_action)
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise RuntimeError("raw RoboCasa env.step must return (obs, reward, done, info)")
        observation, reward, done, info = result
        self._observation = _process_observation(observation)
        for listener in tuple(self._observation_listeners):
            listener(self._observation)
        self._episode_video_step += 1
        self._episode_video_metadata["env_steps_seen"] = self._episode_video_step
        frame_stride = self._episode_video_metadata.get("frame_stride")
        if (
            self._episode_video_writer is not None
            and isinstance(frame_stride, int)
            and self._episode_video_step % frame_stride == 0
        ):
            self._append_episode_video_frame()
        return self._observation, reward, done, info

    def start_episode_video_recording(
        self,
        path: str | Path,
        *,
        camera_name: str = "robot0_agentview_left",
        fps: int = 20,
        frame_stride: int = 2,
    ) -> dict[str, Any]:
        """Start one streaming video spanning all subsequent environment steps."""

        if self._observation is None:
            raise RuntimeError(
                "RoboCasa environment must be reset before recording video"
            )
        if fps <= 0:
            raise ValueError("episode video fps must be positive")
        if frame_stride <= 0:
            raise ValueError("episode video frame_stride must be positive")
        if not isinstance(camera_name, str) or not camera_name.strip():
            raise ValueError("episode video camera_name must be non-empty")
        if self._episode_video_writer is not None:
            self.finalize_episode_video_recording()

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._episode_video_step = 0
        self._episode_video_metadata = {
            "enabled": True,
            "path": str(output_path),
            "camera": camera_name.strip(),
            "fps": int(fps),
            "frame_stride": int(frame_stride),
            "frame_count": 0,
            "env_steps_seen": 0,
            "finalized": False,
            "error": None,
        }
        try:
            self._episode_video_writer = imageio.get_writer(
                str(output_path),
                fps=int(fps),
                macro_block_size=None,
            )
            # Include the state immediately before the first grounding/policy call.
            self._append_episode_video_frame()
        except Exception as exc:
            self._episode_video_metadata["error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            self.finalize_episode_video_recording()
        return dict(self._episode_video_metadata)

    def _append_episode_video_frame(self) -> None:
        if self._episode_video_writer is None or self._observation is None:
            return
        camera_name = str(self._episode_video_metadata["camera"])
        key = f"{camera_name}_image"
        try:
            frame = np.asarray(self._observation[key])
            if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
                raise ValueError(f"invalid video frame shape {frame.shape}")
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            self._episode_video_writer.append_data(
                np.ascontiguousarray(frame)
            )
            self._episode_video_metadata["frame_count"] += 1
        except Exception as exc:
            self._episode_video_metadata["error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            self.finalize_episode_video_recording()

    def finalize_episode_video_recording(self) -> dict[str, Any]:
        """Close the episode writer; safe to call repeatedly and on failures."""

        writer = self._episode_video_writer
        self._episode_video_writer = None
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:
                if self._episode_video_metadata.get("error") is None:
                    self._episode_video_metadata["error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
        self._episode_video_metadata["finalized"] = True
        return dict(self._episode_video_metadata)

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

    def render_camera(
        self,
        camera_name: str,
        *,
        height: int = 256,
        width: int = 256,
    ) -> np.ndarray:
        """Render one named MuJoCo camera without changing policy observations."""

        if not isinstance(camera_name, str) or not camera_name.strip():
            raise ValueError("camera_name must be a non-empty string")
        if height <= 0 or width <= 0:
            raise ValueError("camera render dimensions must be positive")
        frame = self.env.sim.render(
            height=int(height),
            width=int(width),
            camera_name=camera_name.strip(),
        )
        return np.ascontiguousarray(np.asarray(frame)[::-1])

    def render_camera_views(
        self,
        camera_names: tuple[str, ...] | list[str],
        *,
        height: int = 256,
        width: int = 256,
    ) -> dict[str, np.ndarray]:
        """Render a deterministic, labelled set of local-refinement views."""

        if not camera_names:
            raise ValueError("camera_names must not be empty")
        return {
            str(camera_name): self.render_camera(
                str(camera_name), height=height, width=width
            )
            for camera_name in camera_names
        }

    def close(self) -> None:
        self.finalize_episode_video_recording()
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
