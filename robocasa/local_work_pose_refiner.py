"""Top-camera VLM refinement of a mobile base before manipulation.

The refiner is deliberately separate from the remote pi0.5 policy. It renders
extra MuJoCo cameras on demand, requests one discrete base command, executes a
short velocity pulse, and closes the loop with a new set of images.
"""

from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

try:
    from .atomic_task_schemas import AtomicTaskCall
    from .held_object_guard import build_held_object_guard
except ImportError:
    from atomic_task_schemas import AtomicTaskCall
    from held_object_guard import build_held_object_guard


DISCRETE_BASE_ACTIONS = frozenset(
    {
        "FORWARD_SMALL",
        "BACKWARD_SMALL",
        "STRAFE_LEFT_SMALL",
        "STRAFE_RIGHT_SMALL",
        "TURN_LEFT_SMALL",
        "TURN_RIGHT_SMALL",
        "STOP",
        "UNRESOLVED",
    }
)

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_ACTION_DIM = 12
_INVERSE_ACTION = {
    "FORWARD_SMALL": "BACKWARD_SMALL",
    "BACKWARD_SMALL": "FORWARD_SMALL",
    "STRAFE_LEFT_SMALL": "STRAFE_RIGHT_SMALL",
    "STRAFE_RIGHT_SMALL": "STRAFE_LEFT_SMALL",
    "TURN_LEFT_SMALL": "TURN_RIGHT_SMALL",
    "TURN_RIGHT_SMALL": "TURN_LEFT_SMALL",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _as_rgb_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"camera image must be three-dimensional, got {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"camera image must have three RGB channels, got {array.shape}")
    if array.dtype != np.uint8:
        array = array.astype(np.float32, copy=False)
        if array.max(initial=0.0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def _encode_png_data_url(image: Any) -> str:
    buffer = io.BytesIO()
    Image.fromarray(_as_rgb_uint8(image)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_response_text(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("local-pose VLM response has no choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise RuntimeError("local-pose VLM choice must be an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("local-pose VLM choice has no message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        if any(parts):
            return "".join(parts)
    raise RuntimeError("local-pose VLM message content is not text")


def _validate_decision(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("local-pose decision must be a JSON object")
    action = str(value.get("action", "")).strip().upper()
    if action not in DISCRETE_BASE_ACTIONS:
        raise RuntimeError(
            f"unsupported local-pose action {action!r}; expected {sorted(DISCRETE_BASE_ACTIONS)}"
        )
    confidence_value = value.get("confidence", 0.0)
    if isinstance(confidence_value, bool):
        raise RuntimeError("local-pose confidence must be numeric")
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("local-pose confidence must be numeric") from exc
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise RuntimeError("local-pose confidence must be in [0, 1]")
    target_visible = value.get("target_visible", False)
    operation_ready = value.get("operation_ready", False)
    if not isinstance(target_visible, bool) or not isinstance(operation_ready, bool):
        raise RuntimeError("target_visible and operation_ready must be booleans")
    reason = str(value.get("reason", "")).strip()
    return {
        "action": action,
        "confidence": confidence,
        "target_visible": target_visible,
        "operation_ready": operation_ready,
        "reason": reason,
    }


class OpenAICompatibleLocalPoseVLM:
    """Request one robot-frame discrete base command from labelled camera views."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_s: float = 120.0,
    ):
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("local-pose VLM base_url must be non-empty")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("local-pose VLM model must be non-empty")
        if timeout_s <= 0:
            raise ValueError("local-pose VLM timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.last_response_text: str | None = None

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def decide(
        self,
        *,
        images: Mapping[str, np.ndarray],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not images:
            raise ValueError("local-pose VLM requires at least one camera image")
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Decide one small mobile-base correction for the operation below. "
                    "All action directions are in the ROBOT BASE frame, not image "
                    "coordinates. Positive forward is the robot's facing direction; "
                    "left and right are the robot's sides. Use STOP only when the "
                    "target is visible and the arm appears close enough and aligned "
                    "to manipulate it. Context:\n"
                    + json.dumps(_jsonable(context), ensure_ascii=False)
                ),
            }
        ]
        for camera_name, image in images.items():
            content.extend(
                (
                    {
                        "type": "text",
                        "text": f"Camera view: {camera_name}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": _encode_png_data_url(image)},
                    },
                )
            )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a conservative RoboCasa local work-pose controller. "
                        "Return exactly one JSON object with keys action, confidence, "
                        "target_visible, operation_ready, reason. action must be one "
                        f"of {sorted(DISCRETE_BASE_ACTIONS)}. Choose only one short "
                        "correction per call. Do not output arm commands."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"local-pose VLM HTTP {exc.code} from {self.endpoint}: {detail}"
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(
                f"could not reach local-pose VLM {self.endpoint}: {reason}"
            ) from exc
        if not isinstance(response_data, Mapping):
            raise RuntimeError("local-pose VLM returned non-object JSON")
        response_text = _extract_response_text(response_data)
        self.last_response_text = response_text
        cleaned = _JSON_FENCE.sub("", response_text.strip())
        try:
            decision = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"local-pose VLM did not return valid JSON: {exc}"
            ) from exc
        return _validate_decision(decision)


def _render_views(
    env: Any,
    camera_names: Sequence[str],
    *,
    image_size: int,
) -> dict[str, np.ndarray]:
    renderer = getattr(env, "render_camera_views", None)
    if callable(renderer):
        rendered = renderer(
            list(camera_names), height=image_size, width=image_size
        )
        return {
            str(name): _as_rgb_uint8(image) for name, image in rendered.items()
        }
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        simulator = getattr(current, "sim", None)
        if simulator is not None and callable(getattr(simulator, "render", None)):
            return {
                name: np.ascontiguousarray(
                    _as_rgb_uint8(
                        simulator.render(
                            height=image_size,
                            width=image_size,
                            camera_name=name,
                        )
                    )[::-1]
                )
                for name in camera_names
            }
        current = getattr(current, "env", None)
    raise ValueError("environment does not expose named MuJoCo camera rendering")


def _base_pose(observation: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(observation, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in ("robot0_base_pos", "robot0_base_quat"):
        if key in observation:
            result[key] = _jsonable(np.asarray(observation[key]))
    return result


def _discrete_action_vector(
    action_name: str,
    *,
    translation_command: float,
    rotation_command: float,
) -> np.ndarray:
    action = np.zeros(_ACTION_DIM, dtype=np.float32)
    if action_name == "FORWARD_SMALL":
        action[7] = translation_command
    elif action_name == "BACKWARD_SMALL":
        action[7] = -translation_command
    elif action_name == "STRAFE_LEFT_SMALL":
        action[8] = translation_command
    elif action_name == "STRAFE_RIGHT_SMALL":
        action[8] = -translation_command
    elif action_name == "TURN_LEFT_SMALL":
        action[9] = rotation_command
    elif action_name == "TURN_RIGHT_SMALL":
        action[9] = -rotation_command
    else:
        raise ValueError(f"{action_name!r} is not an executable base action")
    action[11] = 1.0  # HybridMobileBase base mode; arm tracks its desired pose.
    return action


def _oscillating(actions: Sequence[str]) -> bool:
    if len(actions) < 4:
        return False
    a, b, c, d = actions[-4:]
    return a == c and b == d and _INVERSE_ACTION.get(a) == b


class LocalWorkPoseRefiner:
    """Closed-loop local base correction immediately before a manipulation skill."""

    def __init__(
        self,
        *,
        decision_maker: Any,
        log_dir: str | Path,
        camera_names: Sequence[str],
        image_size: int = 256,
        max_decisions: int = 8,
        action_steps: int = 5,
        settle_steps: int = 2,
        translation_command: float = 0.20,
        rotation_command: float = 0.25,
        min_confidence: float = 0.55,
        held_object_guard: bool = True,
    ):
        if not callable(getattr(decision_maker, "decide", None)):
            raise TypeError("decision_maker must provide decide(images=..., context=...)")
        names = tuple(str(name).strip() for name in camera_names if str(name).strip())
        if not names:
            raise ValueError("local-pose camera_names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("local-pose camera_names must not contain duplicates")
        if image_size <= 0 or max_decisions <= 0 or action_steps <= 0:
            raise ValueError("image_size, max_decisions, and action_steps must be positive")
        if settle_steps < 0:
            raise ValueError("settle_steps must be non-negative")
        if not 0.0 < translation_command <= 1.0:
            raise ValueError("translation_command must be in (0, 1]")
        if not 0.0 < rotation_command <= 1.0:
            raise ValueError("rotation_command must be in (0, 1]")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self.decision_maker = decision_maker
        self.log_dir = Path(log_dir)
        self.camera_names = names
        self.image_size = int(image_size)
        self.max_decisions = int(max_decisions)
        self.action_steps = int(action_steps)
        self.settle_steps = int(settle_steps)
        self.translation_command = float(translation_command)
        self.rotation_command = float(rotation_command)
        self.min_confidence = float(min_confidence)
        self.held_object_guard = bool(held_object_guard)

    def refine(
        self,
        *,
        env: Any,
        atomic_task_call: AtomicTaskCall | Mapping[str, Any],
        grounding_result: Mapping[str, Any] | None,
        episode_id: int | str,
    ) -> dict[str, Any]:
        call = (
            atomic_task_call
            if isinstance(atomic_task_call, AtomicTaskCall)
            else AtomicTaskCall.from_mapping(atomic_task_call)
        )
        safe_episode = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(episode_id))
        output_dir = self.log_dir / "local_pose_refinement" / safe_episode
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            observation = env.get_observation()
        except (AttributeError, RuntimeError):
            observation = None
        guard = build_held_object_guard(
            env=env,
            atomic_task_call=call,
            enabled=self.held_object_guard,
        )
        guard.start()
        decisions: list[dict[str, Any]] = []
        executed_actions: list[str] = []
        total_env_steps = 0
        status = "uncertain"
        success = False
        failure_code: str | None = None

        for decision_index in range(self.max_decisions):
            images = _render_views(
                env, self.camera_names, image_size=self.image_size
            )
            for camera_name, image in images.items():
                Image.fromarray(image).save(
                    output_dir / f"decision_{decision_index:02d}_{camera_name}.png"
                )
            context = {
                "atomic_task": call.atomic_task,
                "policy_prompt": call.policy_prompt,
                "arguments": call.arguments,
                "grounding_result": dict(grounding_result or {}),
                "decision_index": decision_index,
                "previous_actions": executed_actions,
            }
            decision = _validate_decision(
                self.decision_maker.decide(images=images, context=context)
            )
            record: dict[str, Any] = {
                "decision_index": decision_index,
                **decision,
                "camera_names": list(images),
                "base_pose_before": _base_pose(observation),
                "executed_env_steps": 0,
            }
            decisions.append(record)

            if decision["confidence"] < self.min_confidence:
                failure_code = "LOCAL_POSE_LOW_CONFIDENCE"
                break
            if decision["action"] == "UNRESOLVED":
                failure_code = "LOCAL_POSE_UNRESOLVED"
                break
            if decision["action"] == "STOP":
                if decision["operation_ready"] and decision["target_visible"]:
                    status = "success"
                    success = True
                else:
                    failure_code = "LOCAL_POSE_INVALID_STOP"
                break

            action = _discrete_action_vector(
                decision["action"],
                translation_command=self.translation_command,
                rotation_command=self.rotation_command,
            )
            guard_failure: dict[str, Any] | None = None
            for _ in range(self.action_steps):
                applied = guard.apply_action(action, step_index=total_env_steps + 1)
                observation, _, done, _ = env.step(applied)
                total_env_steps += 1
                record["executed_env_steps"] += 1
                guard_failure = guard.observe(step_index=total_env_steps)
                if guard_failure is not None or done:
                    break
            if guard_failure is None:
                settle_action = np.zeros(_ACTION_DIM, dtype=np.float32)
                settle_action[11] = 1.0
                for _ in range(self.settle_steps):
                    applied = guard.apply_action(
                        settle_action, step_index=total_env_steps + 1
                    )
                    observation, _, done, _ = env.step(applied)
                    total_env_steps += 1
                    record["executed_env_steps"] += 1
                    guard_failure = guard.observe(step_index=total_env_steps)
                    if guard_failure is not None or done:
                        break
            record["base_pose_after"] = _base_pose(observation)
            executed_actions.append(decision["action"])
            if guard_failure is not None:
                status = "failed"
                failure_code = str(
                    guard_failure.get("failure_code", "OBJECT_DROPPED")
                )
                record["guard_failure"] = guard_failure
                break
            if _oscillating(executed_actions):
                failure_code = "LOCAL_POSE_OSCILLATION"
                break

        if not success and failure_code is None:
            failure_code = "LOCAL_POSE_MAX_DECISIONS"
        result = {
            "module": "LOCAL_WORK_POSE_REFINER",
            "status": status,
            "success": success,
            "failure_code": failure_code,
            "atomic_task": call.atomic_task,
            "subgoal_id": call.subgoal_id,
            "camera_names": list(self.camera_names),
            "uses_top_camera": "robot0_topview" in self.camera_names,
            "num_decisions": len(decisions),
            "num_executed_actions": len(executed_actions),
            "total_env_steps": total_env_steps,
            "decisions": decisions,
            "held_object_guard": guard.to_dict(),
            "artifact_dir": str(output_dir),
        }
        (output_dir / "result.json").write_text(
            json.dumps(_jsonable(result), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return result

