"""Xiaomi-Robotics-1 RoboCasa365 policy client.

The wire protocol, temporal sampling, processor request construction, state
layout, and action decoding in this module follow Xiaomi's official
``deploy/client.py`` and ``eval_robocasa365/entry.py`` implementations:
https://github.com/XiaomiRobotics/Xiaomi-Robotics-1
"""

from __future__ import annotations

import collections
import os
import pickle
import socket
import struct
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image


DEFAULT_XR1_HOST = os.getenv("XR1_HOST", "172.16.11.115")
DEFAULT_XR1_PORT = int(os.getenv("XR1_PORT", "10086"))
DEFAULT_XR1_MODEL_PATH = os.getenv(
    "XR1_MODEL_PATH", "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365"
)
ROBOT_TYPE = "robocasa365"
STATE_DIM = 60
ACTION_DIM = 12
OBS_HISTORY = 4
OBS_INTERVAL = 2
REPLAN_STEPS = 16
CROP_RATIO = 0.95
CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)

_LOCAL_CAMERA_KEYS = {
    "video.robot0_agentview_left": "robot0_agentview_left_image",
    "video.robot0_agentview_right": "robot0_agentview_right_image",
    "video.robot0_eye_in_hand": "robot0_eye_in_hand_image",
}
_STATE_KEYS = {
    "state.end_effector_position_relative": "robot0_base_to_eef_pos",
    "state.end_effector_rotation_relative": "robot0_base_to_eef_quat",
    "state.gripper_qpos": "robot0_gripper_qpos",
    "state.base_position": "robot0_base_pos",
    "state.base_rotation": "robot0_base_quat",
}


def _latest(value: Any, *, expected_ndim: int, key: str) -> np.ndarray:
    """Read the newest item from either a raw or frame-stacked observation."""

    array = np.asarray(value)
    while array.ndim > expected_ndim:
        array = array[-1]
    if array.ndim != expected_ndim:
        raise ValueError(
            f"XR-1 observation {key!r} must resolve to {expected_ndim}D, "
            f"got {np.asarray(value).shape}"
        )
    return array


def _observation_value(observation: Mapping[str, Any], official_key: str) -> Any:
    if official_key in observation:
        return observation[official_key]
    local_key = _STATE_KEYS.get(official_key) or _LOCAL_CAMERA_KEYS.get(official_key)
    if local_key is not None and local_key in observation:
        return observation[local_key]
    raise KeyError(
        f"Missing XR-1 RoboCasa365 observation {official_key!r}; "
        f"available_keys={sorted(observation.keys())}"
    )


def quat_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Official RoboCasa365 quaternion-to-axis-angle conversion."""

    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    xyz = quaternion[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(
        sin_half, np.clip(quaternion[3], -1.0, 1.0)
    )
    return (xyz / sin_half * angle).astype(np.float32)


def observation_to_state(observation: Mapping[str, Any]) -> np.ndarray:
    """Build the official EE-first 14D RoboCasa365 state."""

    def vector(key: str, size: int) -> np.ndarray:
        value = _latest(
            _observation_value(observation, key), expected_ndim=1, key=key
        ).astype(np.float32, copy=False).reshape(-1)
        if value.shape != (size,):
            raise ValueError(f"XR-1 state {key!r} must be {size}D, got {value.shape}")
        return value

    state = np.concatenate(
        [
            vector("state.end_effector_position_relative", 3),
            quat_xyzw_to_axis_angle(
                vector("state.end_effector_rotation_relative", 4)
            ),
            vector("state.gripper_qpos", 2),
            vector("state.base_position", 3),
            quat_xyzw_to_axis_angle(vector("state.base_rotation", 4)),
        ]
    ).astype(np.float32)
    if state.shape != (14,):
        raise ValueError(f"Expected a 14D RoboCasa365 state, got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("XR-1 RoboCasa365 state contains NaN or Inf")
    return state


def collect_images(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Collect the official three RoboCasa365 views as HWC uint8 RGB."""

    images: dict[str, np.ndarray] = {}
    for key in CAMERA_KEYS:
        image = _latest(
            _observation_value(observation, key), expected_ndim=3, key=key
        )
        if image.shape[-1] not in (3, 4) and image.shape[0] in (3, 4):
            image = np.moveaxis(image, 0, -1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"XR-1 camera {key!r} must be HWC RGB, got {image.shape}")
        if image.dtype != np.uint8:
            image = image.astype(np.float32, copy=False)
            if image.max(initial=0.0) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        images[key] = np.ascontiguousarray(image)
    return images


def sample_history(
    history: collections.deque[np.ndarray], length: int, interval: int
) -> np.ndarray:
    """Official history sampling, including reset-time prefix repetition."""

    items = list(history)
    if not items:
        raise ValueError("Observation history cannot be empty")
    indices = [
        max(0, len(items) - 1 - (length - 1 - index) * interval)
        for index in range(length)
    ]
    return np.ascontiguousarray(
        np.stack([items[index] for index in indices], axis=0)
    )


def center_crop(image: np.ndarray, crop_ratio: float) -> Image.Image:
    """Official center crop followed by resize back to the original size."""

    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if crop_ratio >= 1.0:
        return pil_image
    width, height = pil_image.size
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = pil_image.crop((left, top, left + crop_width, top + crop_height))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return cropped.resize((width, height), resampling)


def check_xr1_connectivity(host: str, port: int, timeout_s: float = 5.0) -> None:
    """Fail once with a clear error instead of silently retrying forever."""

    try:
        connection = socket.create_connection((host, port), timeout=timeout_s)
    except OSError as exc:
        raise ConnectionError(
            f"Cannot connect to XR-1 server at {host}:{port}: {exc}"
        ) from exc
    connection.close()


class _XR1LengthPrefixedClient:
    """Official pickle + uint32 big-endian length-prefixed socket protocol."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        processor: Any,
        connect_timeout_s: float,
        infer_timeout_s: float,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.processor = processor
        try:
            self.client_socket = socket.create_connection(
                (self.host, self.port), timeout=float(connect_timeout_s)
            )
        except OSError as exc:
            raise ConnectionError(
                f"Cannot connect to XR-1 server at {self.host}:{self.port}: {exc}"
            ) from exc
        self.client_socket.settimeout(float(infer_timeout_s))

    def _send_with_length_prefix(self, data: Mapping[str, Any]) -> None:
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        self.client_socket.sendall(struct.pack(">I", len(serialized)) + serialized)

    def _recv_exact(self, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            packet = self.client_socket.recv(size - len(chunks))
            if not packet:
                raise ConnectionError("XR-1 connection closed while receiving response")
            chunks.extend(packet)
        return bytes(chunks)

    def infer(self, request: Mapping[str, Any], *, robot_type: str) -> Any:
        self._send_with_length_prefix(request)
        data_len = struct.unpack(">I", self._recv_exact(4))[0]
        raw_actions = pickle.loads(self._recv_exact(data_len))
        return self.processor.decode_action(raw_actions, robot_type=robot_type)

    def close(self) -> None:
        sock = getattr(self, "client_socket", None)
        if sock is not None:
            try:
                sock.close()
            finally:
                self.client_socket = None


class XR1RoboCasaPolicy:
    """Stateful atomic-skill policy using the official XR-1 client protocol."""

    manages_temporal_observations = True
    requires_robocasa_action_conversion = True
    policy_backend = "xr1"

    def __init__(
        self,
        *,
        host: str = DEFAULT_XR1_HOST,
        port: int = DEFAULT_XR1_PORT,
        model_path: str = DEFAULT_XR1_MODEL_PATH,
        connect_timeout_s: float = 15.0,
        infer_timeout_s: float = 120.0,
        obs_history: int = OBS_HISTORY,
        obs_interval: int = OBS_INTERVAL,
        replan_steps: int = REPLAN_STEPS,
        crop_ratio: float = CROP_RATIO,
        robot_type: str = ROBOT_TYPE,
        processor: Any | None = None,
        transport: Any | None = None,
    ) -> None:
        if not host:
            raise ValueError("XR-1 host must be non-empty")
        if port <= 0 or obs_history <= 0 or obs_interval <= 0 or replan_steps <= 0:
            raise ValueError("XR-1 port/history/interval/replan values must be positive")
        if not 0 < crop_ratio <= 1:
            raise ValueError("XR-1 crop_ratio must be in (0, 1]")
        if processor is None:
            try:
                from transformers import AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "XR-1 requires transformers==4.57.1 in the RoboCasa client environment"
                ) from exc
            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
            )
        self.processor = processor
        self.host = str(host)
        self.port = int(port)
        self.model_path = str(model_path)
        self.robot_type = str(robot_type)
        self.obs_history = int(obs_history)
        self.obs_interval = int(obs_interval)
        self.replan_steps = int(replan_steps)
        self.crop_ratio = float(crop_ratio)
        robot_types = self.processor.list_robot_types()
        if self.robot_type not in robot_types:
            raise ValueError(
                f"Robot type {self.robot_type!r} is missing from processor; "
                f"available={robot_types}"
            )
        print(f"[XR1] connecting to {self.host}:{self.port}")
        self._transport = transport or _XR1LengthPrefixedClient(
            host=self.host,
            port=self.port,
            processor=self.processor,
            connect_timeout_s=connect_timeout_s,
            infer_timeout_s=infer_timeout_s,
        )
        queue_length = (self.obs_history - 1) * self.obs_interval + 1
        self.image_queues = {
            key: collections.deque(maxlen=queue_length) for key in CAMERA_KEYS
        }
        self.state_queue: collections.deque[np.ndarray] = collections.deque(
            maxlen=queue_length
        )
        self.action_queue: collections.deque[np.ndarray] = collections.deque()
        self.current_instruction: str | None = None
        self.episode_initialized = False
        self.num_queries = 0
        self.action_chunk_lengths: list[int] = []
        self.infer_ms: list[float] = []
        self.environment_observer_attached = False
        print("[XR1] connected")

    def reset(self, obs: Mapping[str, Any]) -> None:
        """Reset an episode: clear history, queued actions, and instruction."""

        for queue in self.image_queues.values():
            queue.clear()
        self.state_queue.clear()
        self.action_queue.clear()
        self.current_instruction = None
        self.num_queries = 0
        self.action_chunk_lengths.clear()
        self.infer_ms.clear()
        self.episode_initialized = True
        self.observe(obs)

    def set_instruction(self, instruction: str) -> None:
        """Switch atomic skills while preserving temporal observations."""

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("XR-1 atomic instruction must be non-empty")
        self.cancel_current_chunk()
        self.current_instruction = instruction.strip()
        print(f"[XR1] instruction: {self.current_instruction}")

    def observe(self, obs: Mapping[str, Any]) -> None:
        if not isinstance(obs, Mapping):
            raise TypeError("XR-1 observation must be a mapping")
        for key, image in collect_images(obs).items():
            self.image_queues[key].append(image)
        self.state_queue.append(observation_to_state(obs))

    def _build_messages(
        self, image_history: Mapping[str, np.ndarray]
    ) -> list[dict[str, Any]]:
        if self.current_instruction is None:
            raise RuntimeError("XR-1 instruction must be set before inference")
        videos = {
            key: [center_crop(frame, self.crop_ratio) for frame in image_history[key]]
            for key in CAMERA_KEYS
        }
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Left camera: "},
                    {"type": "video", "video": videos[CAMERA_KEYS[0]]},
                    {"type": "text", "text": "\nRight camera: "},
                    {"type": "video", "video": videos[CAMERA_KEYS[1]]},
                    {"type": "text", "text": "\nWrist camera: "},
                    {"type": "video", "video": videos[CAMERA_KEYS[2]]},
                    {
                        "type": "text",
                        "text": (
                            "\n\nGenerate robot actions for the task:\n"
                            f"{self.current_instruction} /no_cot"
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<cot></cot>"}],
            },
        ]

    def _infer_actions(self) -> np.ndarray:
        if not self.episode_initialized or not self.state_queue:
            raise RuntimeError("XR-1 policy must be reset with an initial observation")
        if self.current_instruction is None:
            raise RuntimeError("XR-1 instruction must be set before act()")
        states = sample_history(
            self.state_queue, self.obs_history, self.obs_interval
        )
        images = {
            key: sample_history(queue, self.obs_history, self.obs_interval)
            for key, queue in self.image_queues.items()
        }
        padded_state = np.zeros(
            (1, states.shape[0], STATE_DIM), dtype=np.float32
        )
        padded_state[0, :, : states.shape[-1]] = states
        inputs = self.processor.apply_chat_template(
            self._build_messages(images),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            do_resize=False,
            state=padded_state,
            robot_type=self.robot_type,
        )
        request = dict(inputs)
        request["task_id"] = self.robot_type
        print("[XR1] query model")
        started = time.perf_counter()
        decoded = self._transport.infer(request, robot_type=self.robot_type)
        self.infer_ms.append((time.perf_counter() - started) * 1000.0)
        if hasattr(decoded, "float") and hasattr(decoded, "cpu"):
            decoded = decoded[0, :, :ACTION_DIM].float().cpu().numpy()
        else:
            decoded = np.asarray(decoded)[0, :, :ACTION_DIM]
        actions = np.asarray(decoded, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise RuntimeError(
                f"XR-1 decoded actions must have shape (T, {ACTION_DIM}), got {actions.shape}"
            )
        if len(actions) < self.replan_steps:
            raise RuntimeError(
                f"XR-1 returned {len(actions)} actions, fewer than "
                f"replan_steps={self.replan_steps}"
            )
        if not np.isfinite(actions).all():
            raise RuntimeError("XR-1 decoded actions contain NaN or Inf")
        self.num_queries += 1
        self.action_chunk_lengths.append(int(len(actions)))
        print(f"[XR1] received actions shape: {actions.shape}")
        return actions

    def infer(self, _payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Use the existing remote-policy rollout interface."""
        return {"actions": self._infer_actions()}

    def _query_model(self) -> None:
        actions = self._infer_actions()
        self.action_queue.extend(actions[: self.replan_steps])

    def act(self) -> np.ndarray:
        if not self.action_queue:
            self._query_model()
        action = np.asarray(self.action_queue.popleft(), dtype=np.float32)
        print(f"[XR1] remaining queued actions: {len(self.action_queue)}")
        return action

    def cancel_current_chunk(self) -> None:
        discarded = len(self.action_queue)
        self.action_queue.clear()
        if discarded:
            print(f"[XR1] cancelled {discarded} queued action(s)")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "policy_backend": self.policy_backend,
            "host": self.host,
            "port": self.port,
            "model_path": self.model_path,
            "robot_type": self.robot_type,
            "state_dimension": 14,
            "padded_state_dimension": STATE_DIM,
            "camera_keys": list(CAMERA_KEYS),
            "obs_history": self.obs_history,
            "obs_interval": self.obs_interval,
            "replan_steps": self.replan_steps,
            "crop_ratio": self.crop_ratio,
            "num_queries": self.num_queries,
            "action_chunk_lengths": list(self.action_chunk_lengths),
            "queued_actions": len(self.action_queue),
            "history_size": len(self.state_queue),
        }

    def close(self) -> None:
        self.cancel_current_chunk()
        self._transport.close()
        print("[XR1] connection closed")
