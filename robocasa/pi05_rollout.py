"""Remote π0.5 manipulation rollout for the existing RoboCasa environment.

The OpenPI RoboCasa policy is served remotely and does not implement
``robomimic.algo.RolloutPolicy``. This module therefore owns the small rollout
loop required after the MoMa navigation stage while preserving the existing
environment, success criterion, action interface, and episode artifacts.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

try:
    from io_utils import save_video
except ImportError:  # Standalone package/test fallback; online deployment provides io_utils.
    import imageio.v2 as imageio

    def save_video(path: str | Path, frames: list[np.ndarray], fps: int = 20) -> None:
        """Write video when the online-evaluation helper module is unavailable."""

        imageio.mimsave(str(path), frames, fps=fps)


_IMAGE_KEYS = {
    "observation/image": "robot0_agentview_left_image",
    "observation/wrist_image": "robot0_eye_in_hand_image",
    "observation/right_image": "robot0_agentview_right_image",
}
_STATE_KEYS = (
    ("robot0_base_to_eef_pos", 3),
    ("robot0_base_to_eef_quat", 4),
    ("robot0_base_pos", 3),
    ("robot0_base_quat", 4),
    ("robot0_gripper_qpos", 2),
)
_ACTION_DIM = 12
_EXPECTED_CONTROLLER_SPLITS = {
    "right": (0, 6),
    "right_gripper": (6, 7),
    "base": (7, 10),
    "torso": (10, 11),
}


def _latest_value(value: Any, *, key: str) -> np.ndarray:
    """Get the current item from a FrameStack value or a single-frame value."""

    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(f"Observation {key!r} must be an array, got a scalar")
    # FrameStack prepends exactly one temporal dimension. Keep this loop
    # tolerant of a leading singleton batch dimension in diagnostic callers.
    while array.ndim > 3:
        array = array[-1]
    return array


def _image_to_hwc_uint8(value: Any, *, key: str) -> np.ndarray:
    """Convert the current RoboMimic image to contiguous HWC uint8.

    The local environment has already vertically flipped the raw renderer
    output and converted it to CHW float in [0, 1]. This function reverses only
    the layout/range conversion; it intentionally performs no further flip or
    rotation.
    """

    image = _latest_value(value, key=key)
    if image.ndim != 3:
        raise ValueError(f"Observation {key!r} must resolve to a 3-D image, got shape {image.shape}")

    if image.shape[-1] in (1, 3, 4):
        pass  # HWC already.
    elif image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    else:
        raise ValueError(
            f"Observation {key!r} must be HWC or CHW with 1/3/4 channels, got shape {image.shape}"
        )

    if image.shape[-1] != 3:
        raise ValueError(f"Observation {key!r} must have three RGB channels, got shape {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError(f"Observation {key!r} contains non-finite pixels")

    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        if image.max(initial=0.0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(image)


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Match OpenPI's PIL-based ``resize_with_pad`` for one HWC RGB image."""

    if height <= 0 or width <= 0:
        raise ValueError(f"resize dimensions must be positive, got {(height, width)}")
    if image.shape == (height, width, 3):
        return np.ascontiguousarray(image)

    source_height, source_width = image.shape[:2]
    ratio = max(source_width / width, source_height / height)
    resized_height = int(source_height / ratio)
    resized_width = int(source_width / ratio)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = Image.fromarray(image).resize((resized_width, resized_height), resample=resampling)
    canvas = Image.new("RGB", (width, height), 0)
    canvas.paste(
        resized,
        ((width - resized_width) // 2, (height - resized_height) // 2),
    )
    return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8))


def _state_component(value: Any, *, key: str, expected_dim: int) -> np.ndarray:
    """Extract one current low-dimensional state component as float32."""

    component = np.asarray(value)
    if component.ndim >= 2:
        component = component[-1]
    component = component.reshape(-1)
    if component.size != expected_dim:
        raise ValueError(
            f"Observation {key!r} must have {expected_dim} values, got shape {np.asarray(value).shape}"
        )
    if not np.isfinite(component).all():
        raise ValueError(f"Observation {key!r} contains non-finite values")
    return component.astype(np.float32, copy=False)


def build_pi05_observation(
    observation: Mapping[str, Any],
    prompt: str,
    *,
    resize_size: int = 224,
) -> dict[str, Any]:
    """Build the exact RoboCasa π0.5 request mapping from local observations."""

    if not isinstance(observation, Mapping):
        raise TypeError("observation must be a mapping")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("π0.5 requires a non-empty RoboCasa task-description prompt")

    payload: dict[str, Any] = {}
    for payload_key, local_key in _IMAGE_KEYS.items():
        if local_key not in observation:
            raise KeyError(f"Missing required π0.5 camera observation {local_key!r}")
        image = _image_to_hwc_uint8(observation[local_key], key=local_key)
        payload[payload_key] = resize_with_pad(image, resize_size, resize_size)

    state_parts = []
    for local_key, dim in _STATE_KEYS:
        if local_key not in observation:
            raise KeyError(f"Missing required π0.5 state observation {local_key!r}")
        state_parts.append(_state_component(observation[local_key], key=local_key, expected_dim=dim))
    payload["observation/state"] = np.ascontiguousarray(np.concatenate(state_parts), dtype=np.float32)
    payload["prompt"] = prompt.strip()
    return payload


def _current_observation(env: Any) -> Mapping[str, Any]:
    """Read the current observation without resetting after navigation."""

    if hasattr(env, "_get_stacked_obs_from_history"):
        return env._get_stacked_obs_from_history()
    return env.get_observation()


def _task_prompt(env: Any) -> str:
    """Return the episode's native RoboCasa task description."""

    prompt = getattr(env, "_ep_lang_str", None)
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()

    unwrapped = getattr(env, "unwrapped", env)
    prompt = getattr(unwrapped, "_ep_lang_str", None)
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()

    raw_env = getattr(unwrapped, "env", None)
    if raw_env is not None and hasattr(raw_env, "get_ep_meta"):
        prompt = raw_env.get_ep_meta().get("lang", "")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
    raise RuntimeError("Could not obtain a non-empty RoboCasa episode language prompt")


def _validate_action_contract(env: Any) -> bool:
    """Verify that the local flat vector matches OpenPI's 12-D RoboCasa action.

    The official Gym wrapper calls ``convert_action`` because it accepts a
    dictionary action. The current RoboMimic wrapper accepts the underlying
    RoboSuite vector directly. For PandaOmron, the vector is exactly the same
    layout: arm ``0:6``, gripper ``6``, base ``7:10``, torso ``10``, then the
    HybridMobileBase control-mode/base-mode value at ``11``.

    A fake environment used in unit tests may not expose RoboSuite internals;
    in that case this returns ``False`` after any available dimension check.
    """

    action_dimension = getattr(env, "action_dimension", None)
    if action_dimension is not None and int(action_dimension) != _ACTION_DIM:
        raise RuntimeError(
            f"Local environment action_dimension must be {_ACTION_DIM} for π0.5, got {action_dimension}"
        )

    unwrapped = getattr(env, "unwrapped", env)
    raw_env = getattr(unwrapped, "env", None)
    robots = getattr(raw_env, "robots", None)
    if not robots:
        return False
    controller = getattr(robots[0], "composite_controller", None)
    splits = getattr(controller, "_action_split_indexes", None)
    if splits is None:
        return False
    actual = {name: tuple(indices) for name, indices in dict(splits).items()}
    if actual != _EXPECTED_CONTROLLER_SPLITS:
        raise RuntimeError(
            "Local PandaOmron action layout differs from the OpenPI RoboCasa 12-D contract: "
            f"expected {_EXPECTED_CONTROLLER_SPLITS}, got {actual}"
        )
    limits = getattr(controller, "action_limits", None)
    if limits is not None and len(limits[0]) != _ACTION_DIM:
        raise RuntimeError(
            f"Local controller action vector must have {_ACTION_DIM} dimensions, got {len(limits[0])}"
        )
    return True


def _success_from_info(info: Mapping[str, Any]) -> bool:
    """Extract RoboMimic/RoboCasa task completion from a step info mapping."""

    value = info.get("is_success", info.get("success", False))
    if isinstance(value, Mapping):
        if "task" in value:
            return bool(value["task"])
        return any(bool(item) for item in value.values())
    return bool(value)


def _step_env(env: Any, action: np.ndarray) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
    """Support both current RoboMimic four-tuple and Gymnasium five-tuple steps."""

    result = env.step(action)
    if not isinstance(result, (tuple, list)):
        raise RuntimeError(
            f"env.step must return a tuple/list, got {type(result).__name__}"
        )
    if len(result) == 4:
        observation, reward, done, info = result
    elif len(result) == 5:
        observation, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    else:
        raise RuntimeError(f"Unexpected env.step return length {len(result)}")
    if not isinstance(info, Mapping):
        raise RuntimeError(f"Expected mapping info from env.step, got {type(info).__name__}")
    return observation, float(reward), bool(done), info


def _capture_video_frame(env: Any, observation: Mapping[str, Any]) -> np.ndarray | None:
    """Render a diagnostic frame, with left-camera fallback if render is unavailable."""

    try:
        frame = env.render(mode="rgb_array", height=512, width=512)
        if frame is not None:
            return _image_to_hwc_uint8(frame, key="render")
    except Exception:
        pass
    try:
        return _image_to_hwc_uint8(observation["robot0_agentview_left_image"], key="robot0_agentview_left_image")
    except Exception:
        return None


def _timing_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _apply_base_action_mode(
    action: np.ndarray,
    mode: str,
    residual_limit: float,
) -> np.ndarray:
    """Apply the scheduler-selected base constraint without touching other axes."""

    if mode not in {"full", "frozen", "residual"}:
        raise ValueError(
            f"base_action_mode must be one of full/residual/frozen, got {mode!r}"
        )
    if residual_limit < 0 or not math.isfinite(residual_limit):
        raise ValueError(
            f"base_residual_limit must be finite and non-negative, got {residual_limit}"
        )
    array = np.asarray(action)
    if array.shape != (_ACTION_DIM,):
        raise ValueError(f"action must have shape ({_ACTION_DIM},), got {array.shape}")
    constrained = array.copy()
    if mode == "frozen":
        constrained[7:10] = 0
    elif mode == "residual":
        constrained[7:10] = np.clip(
            constrained[7:10], -residual_limit, residual_limit
        )
    return constrained


def _validate_verifier_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(
            f"Atomic task verifier returned {type(value).__name__}, expected a mapping"
        )
    result = dict(value)
    status = result.get("status")
    if status not in {"success", "failed", "uncertain"}:
        raise RuntimeError(f"Atomic task verifier returned invalid status {status!r}")
    result.setdefault("goal_satisfied", status == "success")
    result.setdefault("failure_code", None)
    result.setdefault("retryable", status != "success")
    result.setdefault("state_evidence", [])
    if not isinstance(result["state_evidence"], list):
        raise RuntimeError("Atomic task verifier state_evidence must be a list")
    return result


def execute_pi05_atomic_task_policy(
    *,
    env: Any,
    client: Any,
    atomic_task_call: Any,
    verifier: Any,
    log_dir: str | Path,
    episode_id: int | str,
    horizon: int = 300,
    replan_steps: int = 5,
    resize_size: int = 224,
    verify_interval: int = 5,
    min_steps_before_verify: int = 10,
    render: bool = True,
    video_skip: int = 2,
    base_action_mode: str = "residual",
    base_residual_limit: float = 0.15,
    held_object_guard: bool = True,
    held_object_hold_confirmation_steps: int = 2,
    held_object_drop_confirmation_steps: int = 2,
    success_handoff_steps: int = 0,
) -> tuple[bool, dict[str, Any]]:
    """Execute one atomic subgoal without resetting the RoboCasa environment.

    The passed client and environment are caller-owned and intentionally reused.
    ``client.reset()`` clears policy session state only; this function never calls
    ``env.reset()`` and never uses the episode's global success signal.
    """

    try:
        from .atomic_task_prompt_builder import build_atomic_task_prompt
        from .atomic_task_schemas import (
            AtomicTaskCall,
            validate_atomic_task_call,
        )
        from .held_object_guard import build_held_object_guard
    except ImportError:
        from atomic_task_prompt_builder import build_atomic_task_prompt
        from atomic_task_schemas import AtomicTaskCall, validate_atomic_task_call
        from held_object_guard import build_held_object_guard

    call = (
        atomic_task_call
        if isinstance(atomic_task_call, AtomicTaskCall)
        else AtomicTaskCall.from_mapping(atomic_task_call)
    )
    validate_atomic_task_call(call)
    prompt = build_atomic_task_prompt(call)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if replan_steps <= 0:
        raise ValueError(f"replan_steps must be positive, got {replan_steps}")
    if verify_interval <= 0:
        raise ValueError(f"verify_interval must be positive, got {verify_interval}")
    if min_steps_before_verify < 0:
        raise ValueError(
            f"min_steps_before_verify must be non-negative, got {min_steps_before_verify}"
        )
    if video_skip <= 0:
        raise ValueError(f"video_skip must be positive, got {video_skip}")
    if held_object_hold_confirmation_steps <= 0:
        raise ValueError("held_object_hold_confirmation_steps must be positive")
    if held_object_drop_confirmation_steps <= 0:
        raise ValueError("held_object_drop_confirmation_steps must be positive")
    if success_handoff_steps < 0:
        raise ValueError("success_handoff_steps must be non-negative")
    if not callable(getattr(client, "infer", None)):
        raise TypeError("client must provide infer(observation) -> mapping")
    if not callable(verifier):
        raise TypeError("verifier must be callable")
    # Validate mode and limit even if no environment step is eventually taken.
    _apply_base_action_mode(
        np.zeros(_ACTION_DIM, dtype=np.float32),
        base_action_mode,
        base_residual_limit,
    )

    action_contract_validated = _validate_action_contract(env)
    observation = _current_observation(env)
    if hasattr(client, "reset"):
        client.reset()
    object_guard = build_held_object_guard(
        env=env,
        atomic_task_call=call,
        enabled=held_object_guard,
        hold_confirmation_steps=held_object_hold_confirmation_steps,
        drop_confirmation_steps=held_object_drop_confirmation_steps,
    )
    object_guard.start()

    action_plan: deque[np.ndarray] = deque()
    rewards: list[float] = []
    action_chunk_lengths: list[int] = []
    server_infer_ms: list[float] = []
    policy_infer_ms: list[float] = []
    video_frames: list[np.ndarray] = []
    verification_history: list[dict[str, Any]] = []
    action_min = math.inf
    action_max = -math.inf
    atomic_task_success = False
    terminal_failure = False
    done = False
    info: Mapping[str, Any] = {}
    last_verified_step: int | None = None
    final_verification: dict[str, Any] | None = None
    first_success_step: int | None = None
    handoff_target_step: int | None = None
    maximum_total_steps = horizon + success_handoff_steps

    for step_index in range(maximum_total_steps):
        if not action_plan:
            payload = build_pi05_observation(
                observation, prompt, resize_size=resize_size
            )
            response = client.infer(payload)
            if not isinstance(response, Mapping):
                raise RuntimeError(
                    f"pi0.5 client returned {type(response).__name__}, expected a mapping"
                )
            if "actions" not in response:
                raise RuntimeError(
                    f"pi0.5 response is missing 'actions'; keys={sorted(response.keys())}"
                )
            action_chunk = np.asarray(response["actions"])
            if action_chunk.ndim != 2:
                raise RuntimeError(
                    f"pi0.5 action chunk must be two-dimensional, got {action_chunk.shape}"
                )
            if action_chunk.shape[1] != _ACTION_DIM:
                raise RuntimeError(
                    f"pi0.5 action dimension must be {_ACTION_DIM}, got {action_chunk.shape[1]}"
                )
            if action_chunk.shape[0] < replan_steps:
                raise RuntimeError(
                    f"pi0.5 returned {action_chunk.shape[0]} actions, "
                    f"fewer than replan_steps={replan_steps}"
                )
            if not np.isfinite(action_chunk).all():
                raise RuntimeError("pi0.5 returned actions containing NaN or Inf")
            action_chunk_lengths.append(int(action_chunk.shape[0]))
            action_plan.extend(
                np.asarray(action, dtype=np.float32)
                for action in action_chunk[:replan_steps]
            )
            for timing_key, target in (
                ("server_timing", server_infer_ms),
                ("policy_timing", policy_infer_ms),
            ):
                timing = response.get(timing_key)
                if isinstance(timing, Mapping) and "infer_ms" in timing:
                    target.append(float(timing["infer_ms"]))

        raw_action = action_plan.popleft()
        action_min = min(action_min, float(raw_action.min()))
        action_max = max(action_max, float(raw_action.max()))
        action = _apply_base_action_mode(
            raw_action, base_action_mode, base_residual_limit
        )
        action = object_guard.apply_action(action, step_index=step_index + 1)
        observation, reward, done, info = _step_env(env, action)
        rewards.append(reward)
        completed_steps = step_index + 1

        if render and step_index % video_skip == 0:
            frame = _capture_video_frame(env, observation)
            if frame is not None:
                video_frames.append(frame)

        guard_failure = object_guard.observe(step_index=completed_steps)
        if guard_failure is not None:
            final_verification = _validate_verifier_result(guard_failure)
            verification_history.append(final_verification)
            last_verified_step = completed_steps
            terminal_failure = False
            break

        should_verify = completed_steps >= min_steps_before_verify and (
            completed_steps % verify_interval == 0 or done
        )
        if should_verify:
            final_verification = _validate_verifier_result(
                verifier(
                    env=env,
                    atomic_task_call=call,
                    observation=observation,
                    step_index=completed_steps,
                    info=info,
                )
            )
            verification_history.append(final_verification)
            last_verified_step = completed_steps
            if final_verification["status"] == "success":
                if first_success_step is None:
                    first_success_step = completed_steps
                    handoff_target_step = (
                        completed_steps + success_handoff_steps
                    )
                if completed_steps >= handoff_target_step:
                    atomic_task_success = True
                    break
            if final_verification["status"] == "failed":
                terminal_failure = not bool(final_verification.get("retryable", False))
                break
        if done:
            break
        if first_success_step is None and completed_steps >= horizon:
            break

    completed_steps = len(rewards)
    if last_verified_step != completed_steps or final_verification is None:
        final_verification = _validate_verifier_result(
            verifier(
                env=env,
                atomic_task_call=call,
                observation=observation,
                step_index=completed_steps,
                info=info,
            )
        )
        verification_history.append(final_verification)
        if final_verification["status"] == "success":
            if first_success_step is None:
                first_success_step = completed_steps
                handoff_target_step = completed_steps + success_handoff_steps
            atomic_task_success = completed_steps >= handoff_target_step
        elif final_verification["status"] == "failed":
            terminal_failure = not bool(final_verification.get("retryable", False))

    video_path = None
    if render and video_frames:
        output_dir = Path(log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"ep{episode_id}_{call.subgoal_id}_pi05.mp4"
        save_video(video_path, video_frames, fps=20)

    logs: dict[str, Any] = {
        "Atomic_Task_Call": call.to_dict(),
        "Atomic_Task": call.atomic_task,
        "Prompt": prompt,
        "Success": bool(atomic_task_success),
        "Terminal_Failure": bool(terminal_failure),
        "Final_Verification": final_verification,
        "Verification_History": verification_history,
        "Return": float(np.sum(rewards)),
        "Horizon": completed_steps,
        "Configured_Pre_Success_Horizon": int(horizon),
        "Success_Handoff_Steps": int(success_handoff_steps),
        "First_Success_Step": first_success_step,
        "Handoff_Target_Step": handoff_target_step,
        "Post_Success_Steps_Executed": (
            max(0, completed_steps - first_success_step)
            if first_success_step is not None
            else 0
        ),
        "Handoff_Completed": bool(
            first_success_step is not None
            and handoff_target_step is not None
            and completed_steps >= handoff_target_step
            and atomic_task_success
        ),
        "Num_Policy_Queries": len(action_chunk_lengths),
        "Action_Chunk_Lengths": action_chunk_lengths,
        "Replan_Steps": int(replan_steps),
        "Base_Action_Mode": base_action_mode,
        "Base_Residual_Limit": float(base_residual_limit),
        "Input_Image_Size": int(resize_size),
        "Action_Interface_Validated": action_contract_validated,
        "Held_Object_Guard": object_guard.to_dict(),
        "Action_Raw_Min": float(action_min) if rewards else None,
        "Action_Raw_Max": float(action_max) if rewards else None,
        "Server_Infer_Ms": _timing_summary(server_infer_ms),
        "Policy_Infer_Ms": _timing_summary(policy_infer_ms),
        "Video_Path": str(video_path) if video_path is not None else None,
    }
    return bool(atomic_task_success), logs


def execute_pi05_manipulation_policy(
    *,
    env: Any,
    client: Any,
    log_dir: str,
    episode_id: int,
    reset_before_rollout: bool = False,
    horizon: int = 500,
    replan_steps: int = 5,
    resize_size: int = 224,
    render: bool = True,
    video_skip: int = 2,
    terminate_on_success: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Run remote π0.5 action chunks in the current post-navigation environment.

    ``client`` only needs an ``infer(observation) -> mapping`` method. Returned
    actions are deliberately sent directly as flat 12-D vectors to the local
    RoboMimic wrapper; OpenPI's Gym ``convert_action`` must not be used here.
    """

    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if replan_steps <= 0:
        raise ValueError(f"replan_steps must be positive, got {replan_steps}")
    if video_skip <= 0:
        raise ValueError(f"video_skip must be positive, got {video_skip}")
    if not callable(getattr(client, "infer", None)):
        raise TypeError("client must provide infer(observation) -> mapping")

    action_contract_validated = _validate_action_contract(env)
    observation = env.reset() if reset_before_rollout else _current_observation(env)
    prompt = _task_prompt(env)
    if hasattr(client, "reset"):
        client.reset()

    action_plan: deque[np.ndarray] = deque()
    rewards: list[float] = []
    action_chunk_lengths: list[int] = []
    server_infer_ms: list[float] = []
    policy_infer_ms: list[float] = []
    video_frames: list[np.ndarray] = []
    action_min = math.inf
    action_max = -math.inf
    success = False
    done = False

    for step_index in range(horizon):
        if not action_plan:
            payload = build_pi05_observation(observation, prompt, resize_size=resize_size)
            response = client.infer(payload)
            if not isinstance(response, Mapping):
                raise RuntimeError(f"π0.5 client returned {type(response).__name__}, expected a mapping")
            if "actions" not in response:
                raise RuntimeError(f"π0.5 response is missing 'actions'; keys={sorted(response.keys())}")

            action_chunk = np.asarray(response["actions"])
            if action_chunk.ndim != 2 or action_chunk.shape[1] != _ACTION_DIM:
                raise RuntimeError(
                    f"π0.5 actions must have shape (T, {_ACTION_DIM}), got {action_chunk.shape}"
                )
            if action_chunk.shape[0] < replan_steps:
                raise RuntimeError(
                    f"π0.5 returned {action_chunk.shape[0]} actions, fewer than replan_steps={replan_steps}"
                )
            if not np.isfinite(action_chunk).all():
                raise RuntimeError("π0.5 returned non-finite actions")

            action_chunk_lengths.append(int(action_chunk.shape[0]))
            action_plan.extend(np.asarray(action, dtype=np.float32) for action in action_chunk[:replan_steps])
            for timing_key, target in (("server_timing", server_infer_ms), ("policy_timing", policy_infer_ms)):
                timing = response.get(timing_key)
                if isinstance(timing, Mapping) and "infer_ms" in timing:
                    target.append(float(timing["infer_ms"]))

        action = action_plan.popleft()
        action_min = min(action_min, float(action.min()))
        action_max = max(action_max, float(action.max()))
        observation, reward, done, info = _step_env(env, action)
        rewards.append(reward)
        success = success or _success_from_info(info)

        if render and step_index % video_skip == 0:
            frame = _capture_video_frame(env, observation)
            if frame is not None:
                video_frames.append(frame)

        if done or (terminate_on_success and success):
            break

    video_path = None
    if render and video_frames:
        video_path = Path(log_dir) / f"ep{episode_id}_pi05_rollout.mp4"
        save_video(video_path, video_frames, fps=20)

    logs: dict[str, Any] = {
        "Return": float(np.sum(rewards)),
        "Horizon": len(rewards),
        "Success_Rate": float(success),
        "Num_Policy_Queries": len(action_chunk_lengths),
        "Action_Chunk_Lengths": action_chunk_lengths,
        "Replan_Steps": int(replan_steps),
        "Prompt": prompt,
        "Input_Image_Size": int(resize_size),
        "Action_Interface_Validated": action_contract_validated,
        "Action_Raw_Min": float(action_min) if rewards else None,
        "Action_Raw_Max": float(action_max) if rewards else None,
        "Server_Infer_Ms": _timing_summary(server_infer_ms),
        "Policy_Infer_Ms": _timing_summary(policy_infer_ms),
        "Video_Path": str(video_path) if video_path is not None else None,
    }
    return float(success), logs
