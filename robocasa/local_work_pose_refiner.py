"""Top-camera VLM refinement of a mobile base before manipulation.

The refiner is deliberately separate from the remote pi0.5 policy. It renders
extra MuJoCo cameras on demand, requests one discrete base command, executes a
short velocity pulse, and closes the loop with a new set of images.
"""

from __future__ import annotations

import base64
import io
import json
import math
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
    from .work_pose_dataset import (
        current_base_pose,
        object_eef_diagnostics,
        wrap_angle,
    )
except ImportError:
    from atomic_task_schemas import AtomicTaskCall
    from held_object_guard import build_held_object_guard
    from work_pose_dataset import (
        current_base_pose,
        object_eef_diagnostics,
        wrap_angle,
    )


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
_TRANSLATION_ACTIONS = frozenset(
    {
        "FORWARD_SMALL",
        "BACKWARD_SMALL",
        "STRAFE_LEFT_SMALL",
        "STRAFE_RIGHT_SMALL",
    }
)
_ROTATION_ACTIONS = frozenset({"TURN_LEFT_SMALL", "TURN_RIGHT_SMALL"})
_DISTANCE_ALIGNMENTS = frozenset(
    {"TOO_FAR", "ALIGNED", "TOO_CLOSE", "UNRESOLVED"}
)
_LATERAL_ALIGNMENTS = frozenset(
    {"TARGET_LEFT", "ALIGNED", "TARGET_RIGHT", "UNRESOLVED"}
)
_YAW_ALIGNMENTS = frozenset(
    {"TURN_LEFT", "ALIGNED", "TURN_RIGHT", "UNRESOLVED"}
)
_INVERSE_ACTION = {
    "FORWARD_SMALL": "BACKWARD_SMALL",
    "BACKWARD_SMALL": "FORWARD_SMALL",
    "STRAFE_LEFT_SMALL": "STRAFE_RIGHT_SMALL",
    "STRAFE_RIGHT_SMALL": "STRAFE_LEFT_SMALL",
    "TURN_LEFT_SMALL": "TURN_RIGHT_SMALL",
    "TURN_RIGHT_SMALL": "TURN_LEFT_SMALL",
}

_ACTION_AXIS = {
    "FORWARD_SMALL": "distance",
    "BACKWARD_SMALL": "distance",
    "STRAFE_LEFT_SMALL": "lateral",
    "STRAFE_RIGHT_SMALL": "lateral",
    "TURN_LEFT_SMALL": "yaw",
    "TURN_RIGHT_SMALL": "yaw",
}
_SCALE_NAMES = ("coarse", "medium", "fine")


class LocalPoseVLMOutputTruncated(RuntimeError):
    """Raised when the VLM exhausts its generation budget before final output."""


def _action_axis(action_name: str) -> str | None:
    return _ACTION_AXIS.get(action_name)


def _sanitise_grounding_for_vlm(grounding_result: Mapping[str, Any] | None) -> dict[str, Any]:
    """Remove evaluation-only/world-frame pose data before building the VLM prompt.

    The VLM is intended to reason from camera views in the robot base frame. Expert
    poses and world-frame coordinates are valid evaluator labels but would leak ground
    truth and can also tempt the model to interpret world x/y as forward/lateral.
    """
    if not isinstance(grounding_result, Mapping):
        return {}
    blocked_tokens = (
        "expert",
        "ground_truth",
        "groundtruth",
        "world_pose",
        "base_pose",
        "position",
        "quaternion",
        "yaw",
    )

    def scrub(value: Any) -> Any:
        if isinstance(value, Mapping):
            cleaned: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                lowered = key_text.lower()
                if any(token in lowered for token in blocked_tokens):
                    continue
                cleaned[key_text] = scrub(item)
            return cleaned
        if isinstance(value, (list, tuple)):
            return [scrub(item) for item in value]
        return value

    return scrub(grounding_result)


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


def _nonempty_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _content_parts(value: Any) -> list[str]:
    """Extract textual payloads from OpenAI-compatible structured content.

    Different servers / model backends return assistant content in slightly
    different schemas.  Accept plain strings, structured text objects, and
    lists of either without requiring a specific ``type`` label.
    """
    direct = _nonempty_text(value)
    if direct is not None:
        return [direct]

    if isinstance(value, Mapping):
        # If the server already returned the requested JSON object directly as
        # message.content, preserve it losslessly as JSON text.
        decision_keys = {
            "distance_alignment",
            "lateral_alignment",
            "yaw_alignment",
            "action",
        }
        if decision_keys.intersection(value.keys()):
            return [json.dumps(_jsonable(value), ensure_ascii=False)]

        parts: list[str] = []
        # Common variants include {text: ...}, {value: ...},
        # {output_text: ...}, or nested {text: {value: ...}}.
        for key in ("text", "output_text", "content", "value"):
            if key not in value:
                continue
            nested_parts = _content_parts(value.get(key))
            if nested_parts:
                parts.extend(nested_parts)
        return parts

    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            parts.extend(_content_parts(item))
        return parts

    return []


def _json_object_from_text(text: str) -> str | None:
    """Return the outermost JSON object embedded in text, if present."""
    stripped = _JSON_FENCE.sub("", text.strip()).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidate = stripped[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, Mapping):
            return candidate
    return None


def _extract_response_text(response: Mapping[str, Any]) -> str:
    """Extract assistant text from several OpenAI-compatible schemas.

    Supported examples include:
      * choices[0].message.content = "..."
      * content = [{"type": "text", "text": "..."}]
      * content = [{"type": "output_text", "text": "..."}]
      * content = {"text": "..."} or a direct decision object
      * choices[0].text = "..."
      * message.reasoning_content containing the requested JSON object
    """
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            "local-pose VLM response has no choices; "
            f"top_level_keys={list(response.keys())}"
        )

    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise RuntimeError(
            "local-pose VLM choice must be an object; "
            f"choice_type={type(choice).__name__}"
        )

    message = choice.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        parts = _content_parts(content)
        if parts:
            return "".join(parts)

        # Some reasoning-capable OpenAI-compatible servers may put the final
        # JSON in reasoning_content while leaving content null / structured.
        reasoning = _nonempty_text(message.get("reasoning_content"))
        if reasoning is not None:
            candidate = _json_object_from_text(reasoning)
            if candidate is not None:
                return candidate

        raise RuntimeError(
            "local-pose VLM message content has no extractable text; "
            f"content_type={type(content).__name__}; "
            f"message_keys={list(message.keys())}; "
            f"content_preview={repr(content)[:500]}"
        )

    # Legacy / completion-style OpenAI-compatible response.
    choice_text = _nonempty_text(choice.get("text"))
    if choice_text is not None:
        return choice_text

    raise RuntimeError(
        "local-pose VLM choice has neither a valid message nor text; "
        f"choice_keys={list(choice.keys())}"
    )



def _normalise_alignment(value: Any, aliases: Mapping[str, str]) -> str:
    text = str(value or "").strip().upper().replace(" ", "_").replace("-", "_")
    return aliases.get(text, text)

def _normalise_action(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "FORWARD": "FORWARD_SMALL",
        "MOVE_FORWARD": "FORWARD_SMALL",
        "BACKWARD": "BACKWARD_SMALL",
        "MOVE_BACKWARD": "BACKWARD_SMALL",
        "STRAFE_LEFT": "STRAFE_LEFT_SMALL",
        "MOVE_LEFT": "STRAFE_LEFT_SMALL",
        "STRAFE_RIGHT": "STRAFE_RIGHT_SMALL",
        "MOVE_RIGHT": "STRAFE_RIGHT_SMALL",
        "TURN_LEFT": "TURN_LEFT_SMALL",
        "ROTATE_LEFT": "TURN_LEFT_SMALL",
        "TURN_RIGHT": "TURN_RIGHT_SMALL",
        "ROTATE_RIGHT": "TURN_RIGHT_SMALL",
        "READY": "STOP",
        "ALIGNED": "STOP",
    }
    return aliases.get(text, text)


def _fallback_action_from_alignment(
    distance_alignment: str,
    lateral_alignment: str,
    yaw_alignment: str,
) -> str:
    """Best-effort fallback used only when the VLM omitted an action field."""
    if "UNRESOLVED" in {distance_alignment, lateral_alignment, yaw_alignment}:
        return "UNRESOLVED"
    if yaw_alignment != "ALIGNED":
        return "TURN_LEFT_SMALL" if yaw_alignment == "TURN_LEFT" else "TURN_RIGHT_SMALL"
    if lateral_alignment != "ALIGNED":
        return (
            "STRAFE_LEFT_SMALL"
            if lateral_alignment == "TARGET_LEFT"
            else "STRAFE_RIGHT_SMALL"
        )
    if distance_alignment != "ALIGNED":
        return "FORWARD_SMALL" if distance_alignment == "TOO_FAR" else "BACKWARD_SMALL"
    return "STOP"

def _validate_decision(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("local-pose decision must be a JSON object")
    distance_alignment = _normalise_alignment(
        value.get("distance_alignment"),
        {
            "FAR": "TOO_FAR",
            "TOO_DISTANT": "TOO_FAR",
            "NEEDS_FORWARD": "TOO_FAR",
            "FORWARD_SMALL": "TOO_FAR",
            "CLOSE": "TOO_CLOSE",
            "NEEDS_BACKWARD": "TOO_CLOSE",
            "BACKWARD_SMALL": "TOO_CLOSE",
            "READY": "ALIGNED",
            "CENTERED": "ALIGNED",
        },
    )
    lateral_alignment = _normalise_alignment(
        value.get("lateral_alignment"),
        {
            "LEFT": "TARGET_LEFT",
            "RIGHT": "TARGET_RIGHT",
            "NEEDS_LEFT": "TARGET_LEFT",
            "NEEDS_RIGHT": "TARGET_RIGHT",
            "CENTERED": "ALIGNED",
            "STRAFE_LEFT_SMALL": "TARGET_LEFT",
            "READY": "ALIGNED",
            "STRAFE_RIGHT_SMALL": "TARGET_RIGHT",
        },
    )
    yaw_alignment = _normalise_alignment(
        value.get("yaw_alignment"),
        {
            "LEFT": "TURN_LEFT",
            "RIGHT": "TURN_RIGHT",
            "ROTATE_LEFT": "TURN_LEFT",
            "ROTATE_RIGHT": "TURN_RIGHT",
            "NEEDS_LEFT": "TURN_LEFT",
            "NEEDS_RIGHT": "TURN_RIGHT",
            "STRAIGHT": "ALIGNED",
            "CENTERED": "ALIGNED",
            "READY": "ALIGNED",
            "TURN_LEFT_SMALL": "TURN_LEFT",
            "TURN_RIGHT_SMALL": "TURN_RIGHT",
        },
    )
    structured = bool(distance_alignment or lateral_alignment or yaw_alignment)
    if structured:
        if distance_alignment not in _DISTANCE_ALIGNMENTS:
            raise RuntimeError(
                "distance_alignment must be one of "
                f"{sorted(_DISTANCE_ALIGNMENTS)}"
            )
        if lateral_alignment not in _LATERAL_ALIGNMENTS:
            raise RuntimeError(
                "lateral_alignment must be one of "
                f"{sorted(_LATERAL_ALIGNMENTS)}"
            )
        if yaw_alignment not in _YAW_ALIGNMENTS:
            raise RuntimeError(
                f"yaw_alignment {yaw_alignment!r} must be one of "
                f"{sorted(_YAW_ALIGNMENTS)}"
            )

        # Keep the VLM action only as a diagnostic proposal.  The physical
        # command is selected deterministically from the three alignment fields
        # using the fixed work-pose refinement order:
        #
        #   1) yaw/orientation
        #   2) lateral alignment
        #   3) forward/backward manipulation distance
        #
        # This prevents a visually plausible FORWARD/STRAFE proposal from being
        # executed while the base is still facing the wrong direction.
        raw_action = value.get("action", value.get("recommended_action"))
        if raw_action is None or not str(raw_action).strip():
            vlm_proposed_action = None
        else:
            vlm_proposed_action = _normalise_action(raw_action)
            if vlm_proposed_action not in DISCRETE_BASE_ACTIONS:
                raise RuntimeError(
                    f"unsupported local-pose action {vlm_proposed_action!r}; "
                    f"expected {sorted(DISCRETE_BASE_ACTIONS)}"
                )

        # UNRESOLVED is conservative: no physical correction is trusted if any
        # alignment axis cannot be judged from the current views.
        if "UNRESOLVED" in {
            distance_alignment,
            lateral_alignment,
            yaw_alignment,
        }:
            action = "UNRESOLVED"
            action_source = "forced_unresolved"
        else:
            action = _fallback_action_from_alignment(
                distance_alignment, lateral_alignment, yaw_alignment
            )
            if action == "STOP":
                action_source = "fixed_priority_all_aligned"
            elif action in _ROTATION_ACTIONS:
                action_source = "fixed_priority_yaw"
            elif action.startswith("STRAFE_"):
                action_source = "fixed_priority_lateral"
            else:
                action_source = "fixed_priority_distance"
    else:
        action = _normalise_action(value.get("action", value.get("recommended_action")))
        action_source = "vlm"
        if action not in DISCRETE_BASE_ACTIONS:
            raise RuntimeError(
                f"unsupported local-pose action {action!r}; "
                f"expected {sorted(DISCRETE_BASE_ACTIONS)}"
            )

    confidence_value = value.get("confidence", 0.0)
    if isinstance(confidence_value, bool):
        raise RuntimeError("local-pose confidence must be numeric or a confidence label")
    if isinstance(confidence_value, str):
        confidence_key = confidence_value.strip().lower().replace("_", " ").replace("-", " ")
        confidence_aliases = {
            "very high": 0.95,
            "high": 0.90,
            "medium high": 0.80,
            "medium": 0.65,
            "moderate": 0.65,
            "medium low": 0.50,
            "low": 0.35,
            "very low": 0.15,
        }
        if confidence_key in confidence_aliases:
            confidence_value = confidence_aliases[confidence_key]
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "local-pose confidence must be numeric or one of high/medium/low"
        ) from exc
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise RuntimeError("local-pose confidence must be in [0, 1]")
    target_visible = value.get("target_visible", False)
    operation_ready = value.get("operation_ready", False)
    if not isinstance(target_visible, bool) or not isinstance(operation_ready, bool):
        raise RuntimeError("target_visible and operation_ready must be booleans")
    reason = str(value.get("reason", "")).strip()
    return {
        "action": action,
        "action_source": action_source,
        "vlm_proposed_action": vlm_proposed_action if structured else action,
        "confidence": confidence,
        "target_visible": target_visible,
        "operation_ready": operation_ready,
        "reason": reason,
        "distance_alignment": distance_alignment or None,
        "lateral_alignment": lateral_alignment or None,
        "yaw_alignment": yaw_alignment or None,
        "structured_alignment": structured,
    }


def _alternative_alignment_action(
    decision: Mapping[str, Any],
    repeated_action: str,
    blocked_actions: set[str],
) -> str | None:
    candidates: list[tuple[str, str]] = []
    lateral = decision.get("lateral_alignment")
    distance = decision.get("distance_alignment")
    yaw = decision.get("yaw_alignment")
    if yaw == "TURN_LEFT":
        candidates.append(("yaw", "TURN_LEFT_SMALL"))
    elif yaw == "TURN_RIGHT":
        candidates.append(("yaw", "TURN_RIGHT_SMALL"))
    if lateral == "TARGET_LEFT":
        candidates.append(("lateral", "STRAFE_LEFT_SMALL"))
    elif lateral == "TARGET_RIGHT":
        candidates.append(("lateral", "STRAFE_RIGHT_SMALL"))
    if distance == "TOO_FAR":
        candidates.append(("distance", "FORWARD_SMALL"))
    elif distance == "TOO_CLOSE":
        candidates.append(("distance", "BACKWARD_SMALL"))
    repeated_axis = (
        "yaw"
        if repeated_action in _ROTATION_ACTIONS
        else "lateral"
        if repeated_action.startswith("STRAFE_")
        else "distance"
    )
    for axis, action in candidates:
        if axis != repeated_axis and action not in blocked_actions:
            return action
    return None


class OpenAICompatibleLocalPoseVLM:
    """Request one robot-frame discrete base command from labelled camera views."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_s: float = 120.0,
        max_tokens: int = 8192,
        enable_thinking: bool = True,
    ):
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("local-pose VLM base_url must be non-empty")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("local-pose VLM model must be non-empty")
        if timeout_s <= 0:
            raise ValueError("local-pose VLM timeout_s must be positive")
        if max_tokens <= 0:
            raise ValueError("local-pose VLM max_tokens must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)
        self.enable_thinking = bool(enable_thinking)
        # Per-request diagnostics.  These are reset at the beginning of every
        # decide() call so stale responses can never leak into the next record.
        self.last_response_text: str | None = None
        self.last_raw_http_response: str | None = None
        self.last_http_status: int | None = None
        self.last_finish_reason: str | None = None
        self.last_usage: dict[str, Any] | None = None

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def decide(
        self,
        *,
        images: Mapping[str, np.ndarray],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        # Clear all response diagnostics before issuing a new request.  Without
        # this, a parse / timeout failure could incorrectly expose the previous
        # successful response as if it belonged to the current decision.
        self.last_response_text = None
        self.last_raw_http_response = None
        self.last_http_status = None
        self.last_finish_reason = None
        self.last_usage = None

        if not images:
            raise ValueError("local-pose VLM requires at least one camera image")
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Judge the robot work pose for the operation below along three "
                    "independent axes. All directions are in the ROBOT BASE frame, "
                    "not image coordinates. "
                    "IMPORTANT: distance_alignment describes MANIPULATION REACHABILITY, "
                    "not visual framing, image centeredness, or apparent image proximity. "
                    "Use distance_alignment=TOO_FAR when the target may be visible or even "
                    "centered, but the robot base still appears too far away for the arm to "
                    "comfortably reach and manipulate the target without additional base motion. "
                    "Use distance_alignment=ALIGNED only when there is reasonably strong visual "
                    "evidence that the base is close enough for comfortable arm reach. A target "
                    "being centered in frontview or eye-in-hand, clearly visible, or visually "
                    "near the gripper is NOT sufficient evidence for ALIGNED. "
                    "Use distance_alignment=TOO_CLOSE when the base is so close that arm workspace, "
                    "collision clearance, or manipulation posture is likely compromised. "
                    "Use distance_alignment=UNRESOLVED when the available views do not provide "
                    "enough evidence to judge manipulation reachability. When uncertain between "
                    "ALIGNED and TOO_FAR, prefer TOO_FAR and request a conservative forward "
                    "correction rather than prematurely declaring the pose ready. "
                    "distance_alignment is therefore one of TOO_FAR, ALIGNED, TOO_CLOSE, or "
                    "UNRESOLVED. lateral_alignment is TARGET_LEFT, ALIGNED, TARGET_RIGHT, or "
                    "UNRESOLVED. yaw_alignment is TURN_LEFT, ALIGNED, TURN_RIGHT, or UNRESOLVED. "
                    "Also choose exactly one action from FORWARD_SMALL, BACKWARD_SMALL, "
                    "STRAFE_LEFT_SMALL, STRAFE_RIGHT_SMALL, TURN_LEFT_SMALL, TURN_RIGHT_SMALL, "
                    "STOP, or UNRESOLVED. Use the STRICT refinement order yaw -> lateral -> "
                    "distance. If yaw_alignment is TURN_LEFT or TURN_RIGHT, propose the matching "
                    "TURN action regardless of lateral or distance error. Only after yaw_alignment "
                    "is ALIGNED may you propose a STRAFE action for lateral error. Only after both "
                    "yaw_alignment and lateral_alignment are ALIGNED may you propose FORWARD or "
                    "BACKWARD for distance error. Explain the current stage in reason. The runtime "
                    "controller will enforce this same priority from the alignment fields. Visual "
                    "alignment does NOT imply distance alignment. "
                    "Eye-in-hand alignment is useful evidence for direction and target visibility, "
                    "but is weak evidence for metric arm reachability. Do not infer reachability "
                    "solely because the gripper appears over, near, or centered on the target. "
                    "Mark operation_ready true only if all three axes are aligned AND the target "
                    "appears comfortably reachable by the arm. If reachability is uncertain, set "
                    "operation_ready=false and distance_alignment to TOO_FAR or UNRESOLVED. "
                    "STOP is valid only when all three axes are ALIGNED, target_visible is true, "
                    "and operation_ready is true. Return confidence as a numeric value in [0, 1]. "
                    "The controller executes one correction "
                    "and then requests fresh images. Corrections listed in "
                    "previous_corrections have already been physically executed; reassess "
                    "all axes from the new views before choosing the next action. Action names "
                    "encode direction/axis only; the controller selects an adaptive coarse, "
                    "medium, or fine magnitude. If temporal_guidance reports repeated corrections "
                    "on one axis, a direction reversal, or an unresolved different axis, explicitly "
                    "reassess whether continuing the same axis is still justified by fresh visual "
                    "evidence. Never estimate lateral/distance error from world x/y coordinates or "
                    "from evaluator poses; judge directions in the ROBOT BASE frame from images. "
                    "Context:\n"
                    + json.dumps(_jsonable(context), ensure_ascii=False)
                ),
            }
        ]
        for camera_name, image in images.items():
            camera_label = f"Camera view: {camera_name}"
            if camera_name == "robot0_topview":
                camera_label += (
                    ". CALIBRATED IMAGE AXES: image RIGHT = robot FORWARD; "
                    "image UP = robot LEFT; image LEFT = robot BACKWARD; "
                    "image DOWN = robot RIGHT. Use this mapping exactly when "
                    "judging lateral and distance corrections."
                )
            content.extend(
                (
                    {
                        "type": "text",
                        "text": camera_label,
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
                        "Return exactly one JSON object with keys distance_alignment, "
                        "lateral_alignment, yaw_alignment, action, confidence, target_visible, "
                        "operation_ready, reason. Never output arm commands. "
                        "Interpret distance_alignment strictly as MANIPULATION REACHABILITY: "
                        "ALIGNED means the base appears close enough for the arm to comfortably "
                        "reach the target, not merely that the target is visible, centered, or "
                        "aligned with the gripper in an image. A centered target in frontview or "
                        "eye-in-hand is not proof of reachability. If distance/reachability is "
                        "ambiguous, prefer TOO_FAR over ALIGNED and keep operation_ready=false. "
                        "Use UNRESOLVED when the views truly cannot support a reachability or "
                        "direction judgment. Follow the STRICT action priority yaw -> lateral -> "
                        "distance. Any non-ALIGNED yaw must be corrected first with TURN_LEFT or "
                        "TURN_RIGHT. Only when yaw is ALIGNED may lateral error be corrected with "
                        "STRAFE_LEFT or STRAFE_RIGHT. Only when both yaw and lateral are ALIGNED may "
                        "distance be corrected with FORWARD or BACKWARD. The runtime controller also "
                        "enforces this priority, so use reason to explain the current refinement "
                        "stage. Return confidence as a number from 0 to 1, "
                        "not a word such as high. Treat entries in previous_corrections as completed "
                        "physical motion and reassess from the fresh images. operation_ready=true "
                        "requires all three axes ALIGNED and strong visual evidence of comfortable "
                        "arm reachability. STOP is valid only when all three axes are ALIGNED and "
                        "target_visible and operation_ready are true."
                    ),
                },
                {"role": "user", "content": content},
            ],
            # Thinking mode: use sampling instead of greedy decoding
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,

            "max_tokens": self.max_tokens,

            "chat_template_kwargs": {
                "enable_thinking": self.enable_thinking,
            },
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
                status = getattr(response, "status", None)
                if isinstance(status, int):
                    self.last_http_status = status
                raw_http_response = response.read().decode("utf-8", errors="replace")
                # Save the exact body BEFORE JSON / message parsing.  This is the
                # authoritative response for debugging parser failures.
                self.last_raw_http_response = raw_http_response
                try:
                    response_data = json.loads(raw_http_response)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "local-pose VLM returned invalid HTTP JSON: "
                        f"{exc}; body_preview={raw_http_response[:1000]!r}"
                    ) from exc
        except urllib.error.HTTPError as exc:
            self.last_http_status = int(exc.code)
            detail = exc.read().decode("utf-8", errors="replace")
            self.last_raw_http_response = detail
            raise RuntimeError(
                f"local-pose VLM HTTP {exc.code} from {self.endpoint}: {detail}"
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(
                f"could not reach local-pose VLM {self.endpoint}: {reason}"
            ) from exc
        if not isinstance(response_data, Mapping):
            raise RuntimeError(
                "local-pose VLM returned non-object JSON; "
                f"response_type={type(response_data).__name__}"
            )

        usage = response_data.get("usage")
        if isinstance(usage, Mapping):
            self.last_usage = dict(usage)

        choices = response_data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            finish_reason = choices[0].get("finish_reason")
            if finish_reason is not None:
                self.last_finish_reason = str(finish_reason)

        # A length stop means the generation budget was exhausted.  Never try
        # to execute a possibly incomplete decision, even if some partial text
        # happens to be present.  In the failure mode observed with Qwen, the
        # model spent the entire budget in reasoning and returned content=null.
        if self.last_finish_reason == "length":
            raise LocalPoseVLMOutputTruncated(
                "local-pose VLM exhausted max_tokens before producing a complete "
                f"decision; max_tokens={self.max_tokens}; usage={self.last_usage}"
            )

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


def _read_base_pose(
    env: Any,
    observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        return current_base_pose(env)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        if not isinstance(observation, Mapping):
            raise RuntimeError("environment does not expose a valid base pose")
        position = np.asarray(observation["robot0_base_pos"], dtype=np.float64)
        quaternion = np.asarray(observation["robot0_base_quat"], dtype=np.float64)
        while position.ndim > 1:
            position = position[-1]
        while quaternion.ndim > 1:
            quaternion = quaternion[-1]
        position = position.reshape(-1)
        quaternion = quaternion.reshape(-1)
        if position.size != 3 or quaternion.size != 4:
            raise RuntimeError("base pose observation has an invalid shape")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12:
            raise RuntimeError("base quaternion must be non-zero")
        x, y, z, w = quaternion / norm
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        return {
            "position": position.tolist(),
            "quaternion_xyzw": quaternion.tolist(),
            "yaw_rad": yaw,
        }


def _target_base_pose(
    pose: Mapping[str, Any],
    action_name: str,
    *,
    translation_distance_m: float,
    rotation_angle_rad: float,
) -> dict[str, Any]:
    position = np.asarray(pose["position"], dtype=np.float64).copy()
    yaw = float(pose["yaw_rad"])
    forward_m = 0.0
    left_m = 0.0
    yaw_delta = 0.0
    if action_name == "FORWARD_SMALL":
        forward_m = translation_distance_m
    elif action_name == "BACKWARD_SMALL":
        forward_m = -translation_distance_m
    elif action_name == "STRAFE_LEFT_SMALL":
        left_m = translation_distance_m
    elif action_name == "STRAFE_RIGHT_SMALL":
        left_m = -translation_distance_m
    elif action_name == "TURN_LEFT_SMALL":
        yaw_delta = rotation_angle_rad
    elif action_name == "TURN_RIGHT_SMALL":
        yaw_delta = -rotation_angle_rad
    else:
        raise ValueError(f"{action_name!r} is not an executable base action")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    position[:2] += (
        cosine * forward_m - sine * left_m,
        sine * forward_m + cosine * left_m,
    )
    return {
        "position": position.tolist(),
        "yaw_rad": wrap_angle(yaw + yaw_delta),
    }


def _pose_delta(
    pose: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> tuple[float, float]:
    position = np.asarray(pose["position"], dtype=np.float64)
    reference_position = np.asarray(reference["position"], dtype=np.float64)
    translation = float(np.linalg.norm(position[:2] - reference_position[:2]))
    rotation = abs(
        wrap_angle(float(pose["yaw_rad"]) - float(reference["yaw_rad"]))
    )
    return translation, rotation


def _target_error_robot_frame(
    pose: Mapping[str, Any],
    target: Mapping[str, Any],
) -> tuple[float, float, float]:
    position = np.asarray(pose["position"], dtype=np.float64)
    target_position = np.asarray(target["position"], dtype=np.float64)
    error = target_position[:2] - position[:2]
    yaw = float(pose["yaw_rad"])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    forward_error = cosine * error[0] + sine * error[1]
    left_error = -sine * error[0] + cosine * error[1]
    yaw_error = wrap_angle(float(target["yaw_rad"]) - yaw)
    return float(forward_error), float(left_error), float(yaw_error)


def _closed_loop_action_vector(
    action_name: str,
    *,
    forward_error_m: float,
    left_error_m: float,
    yaw_error_rad: float,
    translation_command: float,
    rotation_command: float,
) -> np.ndarray:
    action = np.zeros(_ACTION_DIM, dtype=np.float32)
    if action_name in _TRANSLATION_ACTIONS:
        action[7] = np.clip(
            4.0 * forward_error_m,
            -translation_command,
            translation_command,
        )
        action[8] = np.clip(
            4.0 * left_error_m,
            -translation_command,
            translation_command,
        )
        action[9] = np.clip(
            2.0 * yaw_error_rad,
            -rotation_command,
            rotation_command,
        )
    elif action_name in _ROTATION_ACTIONS:
        action[7] = np.clip(
            4.0 * forward_error_m,
            -translation_command,
            translation_command,
        )
        action[8] = np.clip(
            4.0 * left_error_m,
            -translation_command,
            translation_command,
        )
        action[9] = np.clip(
            2.0 * yaw_error_rad,
            -rotation_command,
            rotation_command,
        )
    else:
        raise ValueError(f"{action_name!r} is not an executable base action")
    action[11] = 1.0
    return action


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
        max_decisions: int = 10,
        action_steps: int = 5,
        settle_steps: int = 2,
        translation_command: float = 0.20,
        rotation_command: float = 0.25,
        translation_distance_m: float = 0.30,
        rotation_angle_deg: float = 25.0,
        medium_translation_distance_m: float = 0.12,
        fine_translation_distance_m: float = 0.05,
        medium_rotation_angle_deg: float = 10.0,
        fine_rotation_angle_deg: float = 5.0,
        held_translation_distance_m: float = 0.01,
        held_rotation_angle_deg: float = 1.0,
        motion_max_steps: int = 2000,
        translation_tolerance_m: float = 0.015,
        rotation_tolerance_deg: float = 1.0,
        max_total_translation_m: float = 1.20,
        max_total_rotation_deg: float = 125.0,
        max_invalid_stops: int = 2,
        rotation_translation_drift_limit_m: float = 0.08,
        translation_rotation_drift_limit_deg: float = 5.0,
        stall_window_steps: int = 100,
        stall_translation_epsilon_m: float = 0.001,
        stall_rotation_epsilon_deg: float = 0.2,
        partial_progress_ratio: float = 0.75,
        max_motion_recoveries: int = 2,
        max_fine_axis_reversals: int = 3,
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
        positive_values = {
            "translation_distance_m": translation_distance_m,
            "rotation_angle_deg": rotation_angle_deg,
            "medium_translation_distance_m": medium_translation_distance_m,
            "fine_translation_distance_m": fine_translation_distance_m,
            "medium_rotation_angle_deg": medium_rotation_angle_deg,
            "fine_rotation_angle_deg": fine_rotation_angle_deg,
            "held_translation_distance_m": held_translation_distance_m,
            "held_rotation_angle_deg": held_rotation_angle_deg,
            "translation_tolerance_m": translation_tolerance_m,
            "rotation_tolerance_deg": rotation_tolerance_deg,
            "max_total_translation_m": max_total_translation_m,
            "max_total_rotation_deg": max_total_rotation_deg,
            "rotation_translation_drift_limit_m": rotation_translation_drift_limit_m,
            "translation_rotation_drift_limit_deg": translation_rotation_drift_limit_deg,
            "stall_translation_epsilon_m": stall_translation_epsilon_m,
            "stall_rotation_epsilon_deg": stall_rotation_epsilon_deg,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"{', '.join(invalid)} must be positive")
        if motion_max_steps <= 0:
            raise ValueError("motion_max_steps must be positive")
        if stall_window_steps <= 0:
            raise ValueError("stall_window_steps must be positive")
        if not 0.0 < partial_progress_ratio <= 1.0:
            raise ValueError("partial_progress_ratio must be in (0, 1]")
        if max_motion_recoveries < 0:
            raise ValueError("max_motion_recoveries must be non-negative")
        if max_fine_axis_reversals < 0:
            raise ValueError("max_fine_axis_reversals must be non-negative")
        if not (
            translation_distance_m >= medium_translation_distance_m >= fine_translation_distance_m
        ):
            raise ValueError(
                "translation scales must satisfy coarse >= medium >= fine"
            )
        if not (rotation_angle_deg >= medium_rotation_angle_deg >= fine_rotation_angle_deg):
            raise ValueError("rotation scales must satisfy coarse >= medium >= fine")
        if max_invalid_stops < 0:
            raise ValueError("max_invalid_stops must be non-negative")
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
        self.translation_distance_m = float(translation_distance_m)
        self.rotation_angle_deg = float(rotation_angle_deg)
        self.medium_translation_distance_m = float(medium_translation_distance_m)
        self.fine_translation_distance_m = float(fine_translation_distance_m)
        self.medium_rotation_angle_deg = float(medium_rotation_angle_deg)
        self.fine_rotation_angle_deg = float(fine_rotation_angle_deg)
        self.held_translation_distance_m = float(held_translation_distance_m)
        self.held_rotation_angle_deg = float(held_rotation_angle_deg)
        self.motion_max_steps = int(motion_max_steps)
        self.translation_tolerance_m = float(translation_tolerance_m)
        self.rotation_tolerance_deg = float(rotation_tolerance_deg)
        self.max_total_translation_m = float(max_total_translation_m)
        self.max_total_rotation_deg = float(max_total_rotation_deg)
        self.max_invalid_stops = int(max_invalid_stops)
        self.rotation_translation_drift_limit_m = float(
            rotation_translation_drift_limit_m
        )
        self.translation_rotation_drift_limit_deg = float(
            translation_rotation_drift_limit_deg
        )
        self.stall_window_steps = int(stall_window_steps)
        self.stall_translation_epsilon_m = float(stall_translation_epsilon_m)
        self.stall_rotation_epsilon_deg = float(stall_rotation_epsilon_deg)
        self.partial_progress_ratio = float(partial_progress_ratio)
        self.max_motion_recoveries = int(max_motion_recoveries)
        self.max_fine_axis_reversals = int(max_fine_axis_reversals)
        self.min_confidence = float(min_confidence)
        self.held_object_guard = bool(held_object_guard)

    def _step_profile(self, axis: str, level: int, held_mode: bool) -> dict[str, Any]:
        level = max(0, min(2, int(level)))
        if held_mode:
            return {
                "scale": "held_fine",
                "level": 2,
                "translation_m": self.held_translation_distance_m,
                "rotation_deg": self.held_rotation_angle_deg,
            }
        translation_scales = (
            self.translation_distance_m,
            self.medium_translation_distance_m,
            self.fine_translation_distance_m,
        )
        rotation_scales = (
            self.rotation_angle_deg,
            self.medium_rotation_angle_deg,
            self.fine_rotation_angle_deg,
        )
        return {
            "scale": _SCALE_NAMES[level],
            "level": level,
            "translation_m": float(translation_scales[level]),
            "rotation_deg": float(rotation_scales[level]),
            "axis": axis,
        }

    def _adaptive_step_for_action(
        self,
        *,
        action_name: str,
        held_mode: bool,
        axis_scale_levels: Mapping[str, int],
        last_axis_actions: Mapping[str, str],
        consecutive_axis_count: int,
    ) -> tuple[dict[str, Any], bool]:
        axis = _action_axis(action_name)
        if axis is None:
            raise ValueError(f"{action_name!r} has no correction axis")
        level = int(axis_scale_levels.get(axis, 0))
        previous_axis_action = last_axis_actions.get(axis)
        reversed_direction = bool(
            previous_axis_action and _INVERSE_ACTION.get(previous_axis_action) == action_name
        )
        if reversed_direction:
            level = min(2, level + 1)
        elif consecutive_axis_count >= 2:
            # A repeated correction on one axis becomes more cautious even before
            # a visible sign flip occurs. The first step can be coarse; the second
            # consecutive step is at most medium.
            level = max(level, 1)
        return self._step_profile(axis, level, held_mode), reversed_direction

    def _execute_correction(
        self,
        *,
        env: Any,
        observation: Mapping[str, Any] | None,
        action_name: str,
        guard: Any,
        held_mode: bool,
        start_step: int,
        translation_distance_override_m: float | None = None,
        rotation_angle_override_deg: float | None = None,
        step_scale: str | None = None,
    ) -> tuple[Mapping[str, Any] | None, dict[str, Any], dict[str, Any] | None]:
        """Execute one VLM-requested base correction with action-aware convergence.

        The requested axis is the convergence axis. Motion on the orthogonal axis is
        treated as bounded drift, not as a reason to keep chasing a full SE(2) target.
        This prevents a successful yaw correction from being rejected solely because
        the mobile base translated slightly while turning (and vice versa).

        ``LOCAL_POSE_MOTION_STALLED`` is reserved for a controller that makes
        negligible *physical pose progress* over a complete stall window.
        """
        if action_name not in _TRANSLATION_ACTIONS | _ROTATION_ACTIONS:
            raise ValueError(f"{action_name!r} is not an executable base action")

        initial_pose = _read_base_pose(env, observation)
        translation_distance = (
            self.held_translation_distance_m
            if held_mode
            else float(
                self.translation_distance_m
                if translation_distance_override_m is None
                else translation_distance_override_m
            )
        )
        rotation_angle_deg = (
            self.held_rotation_angle_deg
            if held_mode
            else float(
                self.rotation_angle_deg
                if rotation_angle_override_deg is None
                else rotation_angle_override_deg
            )
        )
        target_pose = _target_base_pose(
            initial_pose,
            action_name,
            translation_distance_m=translation_distance,
            rotation_angle_rad=math.radians(rotation_angle_deg),
        )
        requested_translation = (
            translation_distance if action_name in _TRANSLATION_ACTIONS else 0.0
        )
        requested_rotation_deg = (
            rotation_angle_deg if action_name in _ROTATION_ACTIONS else 0.0
        )
        requested_primary = (
            requested_translation
            if action_name in _TRANSLATION_ACTIONS
            else math.radians(requested_rotation_deg)
        )
        effective_translation_tolerance_m = min(
            self.translation_tolerance_m,
            max(0.005, 0.10 * translation_distance),
        )
        effective_rotation_tolerance_deg = min(
            self.rotation_tolerance_deg,
            max(0.5, 0.10 * rotation_angle_deg),
        )

        object_alias = guard.to_dict().get("object_alias") if held_mode else None
        initial_object_eef_distance: float | None = None
        if object_alias:
            initial_diagnostics = object_eef_diagnostics(env, object_alias)
            distance = initial_diagnostics.get("object_eef_distance_m")
            if isinstance(distance, (int, float)) and np.isfinite(distance):
                initial_object_eef_distance = float(distance)

        trace: list[dict[str, Any]] = []
        guard_failure: dict[str, Any] | None = None
        failure_code: str | None = None
        stable_count = 0
        converged = False
        convergence_mode: str | None = None
        done = False

        # Stall detection compares actual base poses at the start/end of each window.
        # This is deliberately independent of target-error improvement.
        window_start_pose = dict(initial_pose)

        for motion_step in range(1, self.motion_max_steps + 1):
            pose = _read_base_pose(env, observation)
            forward_error, left_error, yaw_error = _target_error_robot_frame(
                pose, target_pose
            )
            translation_error = math.hypot(forward_error, left_error)
            yaw_error_deg = abs(math.degrees(yaw_error))

            if action_name in _TRANSLATION_ACTIONS:
                primary_error = translation_error
                within_primary = primary_error <= effective_translation_tolerance_m
                cross_axis_error = yaw_error_deg
                cross_axis_safe = (
                    cross_axis_error <= self.translation_rotation_drift_limit_deg
                )
            else:
                primary_error = abs(yaw_error)
                within_primary = yaw_error_deg <= effective_rotation_tolerance_deg
                # For a pure turn target_pose has the same x/y as the start pose.
                cross_axis_error = translation_error
                cross_axis_safe = (
                    cross_axis_error <= self.rotation_translation_drift_limit_m
                )

            # Primary-axis convergence is enough to finish this discrete correction.
            # Cross-axis error is a safety bound rather than a precision requirement.
            within = within_primary and cross_axis_safe
            stable_count = stable_count + 1 if within else 0
            if stable_count >= 3:
                converged = True
                convergence_mode = "primary_axis_tolerance"
                break

            if within_primary:
                # Do not chase harmless cross-axis drift after the requested axis has
                # already converged. If the drift is unsafe it will be rejected below.
                action = np.zeros(_ACTION_DIM, dtype=np.float32)
                action[11] = 1.0
            else:
                action = _closed_loop_action_vector(
                    action_name,
                    forward_error_m=forward_error,
                    left_error_m=left_error,
                    yaw_error_rad=yaw_error,
                    translation_command=self.translation_command,
                    rotation_command=self.rotation_command,
                )

            applied = guard.apply_action(
                action, step_index=start_step + motion_step
            )
            observation, _, done, _ = env.step(applied)
            guard_failure = guard.observe(step_index=start_step + motion_step)

            current_pose = _read_base_pose(env, observation)
            actual_translation, actual_rotation = _pose_delta(
                current_pose, initial_pose
            )
            actual_rotation_deg = math.degrees(actual_rotation)

            trace_entry: dict[str, Any] = {
                "motion_step": motion_step,
                "translation_error_m": translation_error,
                "yaw_error_deg": math.degrees(yaw_error),
                "primary_error": primary_error,
                "cross_axis_error": cross_axis_error,
                "within_primary_tolerance": within_primary,
                "cross_axis_safe": cross_axis_safe,
                "within_tolerance": within,
                "applied_base_action": np.asarray(applied[7:10], dtype=float).tolist(),
            }
            if object_alias:
                diagnostics = object_eef_diagnostics(env, object_alias)
                trace_entry.update(diagnostics)
                distance = diagnostics.get("object_eef_distance_m")
                if (
                    initial_object_eef_distance is not None
                    and isinstance(distance, (int, float))
                    and np.isfinite(distance)
                    and float(distance)
                    > max(0.03, initial_object_eef_distance + 0.01)
                ):
                    failure_code = "HELD_OBJECT_EEF_DISTANCE_INCREASED"
            trace.append(trace_entry)

            # Requested-axis overshoot stays a hard safety failure.
            if (
                requested_translation > 0
                and actual_translation > requested_translation * 1.15
            ) or (
                requested_rotation_deg > 0
                and actual_rotation_deg > requested_rotation_deg * 1.15
            ):
                failure_code = "LOCAL_POSE_MOTION_OVERSHOOT"

            # Large orthogonal drift is a different failure from being stalled.
            if failure_code is None:
                if (
                    action_name in _ROTATION_ACTIONS
                    and actual_translation > self.rotation_translation_drift_limit_m
                ) or (
                    action_name in _TRANSLATION_ACTIONS
                    and actual_rotation_deg > self.translation_rotation_drift_limit_deg
                ):
                    failure_code = "LOCAL_POSE_CROSS_AXIS_DRIFT"

            # A true stall means the base barely moved over an entire window.
            if (
                failure_code is None
                and motion_step % self.stall_window_steps == 0
                and not within_primary
            ):
                window_translation, window_rotation = _pose_delta(
                    current_pose, window_start_pose
                )
                window_rotation_deg = math.degrees(window_rotation)
                physically_stalled = (
                    window_translation <= self.stall_translation_epsilon_m
                    and window_rotation_deg <= self.stall_rotation_epsilon_deg
                )

                completion_ratio = 0.0
                if requested_primary > 1e-12:
                    completion_ratio = max(
                        0.0, min(1.0, 1.0 - primary_error / requested_primary)
                    )

                if physically_stalled:
                    # Near-target actuator deadzones are accepted as partial completion;
                    # the next VLM observation closes the remaining error.
                    if completion_ratio >= self.partial_progress_ratio:
                        converged = True
                        convergence_mode = "partial_deadzone"
                    else:
                        failure_code = "LOCAL_POSE_MOTION_STALLED"
                window_start_pose = dict(current_pose)

            if guard_failure is not None:
                failure_code = str(
                    guard_failure.get("failure_code", "OBJECT_DROPPED")
                )
            if done and failure_code is None:
                failure_code = "LOCAL_POSE_ENVIRONMENT_DONE"
            if converged:
                break
            if failure_code is not None:
                break

        if not converged and failure_code is None:
            final_pose_before_timeout = _read_base_pose(env, observation)
            final_forward, final_left, final_yaw = _target_error_robot_frame(
                final_pose_before_timeout, target_pose
            )
            final_primary_error = (
                math.hypot(final_forward, final_left)
                if action_name in _TRANSLATION_ACTIONS
                else abs(final_yaw)
            )
            completion_ratio = 0.0
            if requested_primary > 1e-12:
                completion_ratio = max(
                    0.0, min(1.0, 1.0 - final_primary_error / requested_primary)
                )
            if completion_ratio >= self.partial_progress_ratio:
                converged = True
                convergence_mode = "partial_max_steps"
            else:
                failure_code = "LOCAL_POSE_MOTION_MAX_STEPS"

        if converged and guard_failure is None:
            settle_action = np.zeros(_ACTION_DIM, dtype=np.float32)
            settle_action[11] = 1.0
            for settle_index in range(self.settle_steps):
                absolute_step = start_step + len(trace) + 1
                applied = guard.apply_action(
                    settle_action, step_index=absolute_step
                )
                observation, _, done, _ = env.step(applied)
                guard_failure = guard.observe(step_index=absolute_step)
                trace.append(
                    {
                        "motion_step": len(trace) + 1,
                        "settle_step": settle_index + 1,
                        "applied_base_action": np.asarray(
                            applied[7:10], dtype=float
                        ).tolist(),
                    }
                )
                if guard_failure is not None or done:
                    failure_code = (
                        str(guard_failure.get("failure_code", "OBJECT_DROPPED"))
                        if guard_failure is not None
                        else "LOCAL_POSE_ENVIRONMENT_DONE"
                    )
                    break

        final_pose = _read_base_pose(env, observation)
        actual_translation, actual_rotation = _pose_delta(
            final_pose, initial_pose
        )
        final_forward, final_left, final_yaw = _target_error_robot_frame(
            final_pose, target_pose
        )
        final_translation_error = math.hypot(final_forward, final_left)
        final_yaw_error_deg = math.degrees(final_yaw)
        final_primary_error = (
            final_translation_error
            if action_name in _TRANSLATION_ACTIONS
            else abs(final_yaw)
        )
        completion_ratio = 0.0
        if requested_primary > 1e-12:
            completion_ratio = max(
                0.0, min(1.0, 1.0 - final_primary_error / requested_primary)
            )

        motion = {
            "held_mode": held_mode,
            "action": action_name,
            "initial_pose": initial_pose,
            "target_pose": target_pose,
            "final_pose": final_pose,
            "requested_translation_m": requested_translation,
            "requested_rotation_deg": requested_rotation_deg,
            "translation_tolerance_m": effective_translation_tolerance_m,
            "rotation_tolerance_deg": effective_rotation_tolerance_deg,
            "step_scale": step_scale or ("held_fine" if held_mode else "coarse"),
            "rotation_translation_drift_limit_m": (
                self.rotation_translation_drift_limit_m
            ),
            "translation_rotation_drift_limit_deg": (
                self.translation_rotation_drift_limit_deg
            ),
            "actual_translation_m": actual_translation,
            "actual_rotation_deg": math.degrees(actual_rotation),
            "final_translation_error_m": final_translation_error,
            "final_yaw_error_deg": final_yaw_error_deg,
            "primary_completion_ratio": completion_ratio,
            "steps": len(trace),
            "converged": converged and failure_code is None,
            "convergence_mode": convergence_mode,
            "failure_code": failure_code,
            "trace": trace,
        }
        return observation, motion, guard_failure

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
        guard_start_state = guard.to_dict()
        held_mode = bool(
            guard_start_state.get("latched")
            and not guard_start_state.get("release_allowed")
        )
        decisions: list[dict[str, Any]] = []
        executed_actions: list[str] = []
        axis_scale_levels = {"distance": 0, "lateral": 0, "yaw": 0}
        axis_reversal_counts = {"distance": 0, "lateral": 0, "yaw": 0}
        last_axis_actions: dict[str, str] = {}
        consecutive_axis: str | None = None
        consecutive_axis_count = 0
        total_env_steps = 0
        total_translation_m = 0.0
        total_rotation_deg = 0.0
        invalid_stop_count = 0
        invalid_decision_count = 0
        motion_recovery_count = 0
        last_validation_error: str | None = None
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
                "grounding_result": _sanitise_grounding_for_vlm(grounding_result),
                "decision_index": decision_index,
                "previous_actions": executed_actions,
                "held_mode": held_mode,
                "correction_scales": {
                    "translation_m": {
                        "coarse": self.translation_distance_m,
                        "medium": self.medium_translation_distance_m,
                        "fine": self.fine_translation_distance_m,
                    },
                    "rotation_deg": {
                        "coarse": self.rotation_angle_deg,
                        "medium": self.medium_rotation_angle_deg,
                        "fine": self.fine_rotation_angle_deg,
                    },
                    "controller_selects_scale": True,
                },
                "temporal_guidance": {
                    "consecutive_axis": consecutive_axis,
                    "consecutive_axis_count": consecutive_axis_count,
                    "axis_scale_levels": {
                        axis: _SCALE_NAMES[min(2, max(0, level))]
                        for axis, level in axis_scale_levels.items()
                    },
                    "axis_reversal_counts": dict(axis_reversal_counts),
                    "instruction": (
                        "Reassess all axes from images. If one axis has been corrected "
                        "repeatedly while another remains misaligned, do not keep choosing "
                        "the repeated axis without strong fresh visual evidence. Direction "
                        "reversals indicate overshoot; the controller will automatically "
                        "reduce the step size."
                    ),
                },
                "previous_corrections": [
                    {
                        "action": item.get("action"),
                        "reason": item.get("reason"),
                        "distance_alignment": item.get("distance_alignment"),
                        "lateral_alignment": item.get("lateral_alignment"),
                        "yaw_alignment": item.get("yaw_alignment"),
                        "step_scale": item.get("step_scale"),
                        "requested_translation_m": item.get("motion", {}).get(
                            "requested_translation_m"
                        ),
                        "requested_rotation_deg": item.get("motion", {}).get(
                            "requested_rotation_deg"
                        ),
                        "actual_translation_m": item.get("motion", {}).get(
                            "actual_translation_m"
                        ),
                        "actual_rotation_deg": item.get("motion", {}).get(
                            "actual_rotation_deg"
                        ),
                        "converged": item.get("motion", {}).get("converged"),
                        "convergence_mode": item.get("motion", {}).get(
                            "convergence_mode"
                        ),
                        "failure_code": item.get("motion", {}).get("failure_code"),
                        "final_translation_error_m": item.get("motion", {}).get(
                            "final_translation_error_m"
                        ),
                        "final_yaw_error_deg": item.get("motion", {}).get(
                            "final_yaw_error_deg"
                        ),
                    }
                    for item in decisions
                    if item.get("motion")
                ],
                "last_validation_error": last_validation_error,
                "invalid_stop_count": invalid_stop_count,
                "invalid_decision_count": invalid_decision_count,
                "motion_recovery_count": motion_recovery_count,
            }
            try:
                decision = _validate_decision(
                    self.decision_maker.decide(images=images, context=context)
                )
            except LocalPoseVLMOutputTruncated as exc:
                raw_response = getattr(
                    self.decision_maker, "last_response_text", None
                )
                raw_http_response = getattr(
                    self.decision_maker, "last_raw_http_response", None
                )
                http_status = getattr(
                    self.decision_maker, "last_http_status", None
                )
                finish_reason = getattr(
                    self.decision_maker, "last_finish_reason", None
                )
                usage = getattr(self.decision_maker, "last_usage", None)
                record = {
                    "decision_index": decision_index,
                    "failure_code": "LOCAL_POSE_VLM_OUTPUT_TRUNCATED",
                    "validation_error": str(exc),
                    "raw_response": raw_response,
                    "raw_http_response": raw_http_response,
                    "http_status": http_status,
                    "vlm_finish_reason": finish_reason,
                    "vlm_usage": usage,
                    "camera_names": list(images),
                    "base_pose_before": _read_base_pose(env, observation),
                    "executed_env_steps": 0,
                }
                decisions.append(record)
                if isinstance(raw_response, str):
                    (output_dir / f"decision_{decision_index:02d}_parsed.txt").write_text(
                        raw_response + "\n", encoding="utf-8"
                    )
                if isinstance(raw_http_response, str):
                    (
                        output_dir
                        / f"decision_{decision_index:02d}_http_response.json"
                    ).write_text(raw_http_response + "\n", encoding="utf-8")
                failure_code = "LOCAL_POSE_VLM_OUTPUT_TRUNCATED"
                break
            except (RuntimeError, ValueError) as exc:
                invalid_decision_count += 1
                raw_response = getattr(
                    self.decision_maker, "last_response_text", None
                )
                raw_http_response = getattr(
                    self.decision_maker, "last_raw_http_response", None
                )
                http_status = getattr(
                    self.decision_maker, "last_http_status", None
                )
                finish_reason = getattr(
                    self.decision_maker, "last_finish_reason", None
                )
                usage = getattr(self.decision_maker, "last_usage", None)
                record = {
                    "decision_index": decision_index,
                    "validation_error": str(exc),
                    # Parsed assistant text, only when THIS request reached that
                    # stage successfully. It is reset before every request.
                    "raw_response": raw_response,
                    # Exact HTTP body for THIS request, captured before parser
                    # validation. This is the primary parser-debug artifact.
                    "raw_http_response": raw_http_response,
                    "http_status": http_status,
                    "vlm_finish_reason": finish_reason,
                    "vlm_usage": usage,
                    "camera_names": list(images),
                    "base_pose_before": _read_base_pose(env, observation),
                    "executed_env_steps": 0,
                }
                decisions.append(record)
                if isinstance(raw_response, str):
                    (output_dir / f"decision_{decision_index:02d}_parsed.txt").write_text(
                        raw_response + "\n", encoding="utf-8"
                    )
                if isinstance(raw_http_response, str):
                    (
                        output_dir
                        / f"decision_{decision_index:02d}_http_response.json"
                    ).write_text(raw_http_response + "\n", encoding="utf-8")
                last_validation_error = (
                    f"Previous response was invalid: {exc}. Return exactly the "
                    "requested JSON keys and allowed enum values."
                )
                if invalid_decision_count <= self.max_invalid_stops:
                    continue
                failure_code = "LOCAL_POSE_INVALID_DECISION"
                break
            raw_http_response = getattr(
                self.decision_maker, "last_raw_http_response", None
            )
            http_status = getattr(self.decision_maker, "last_http_status", None)
            finish_reason = getattr(
                self.decision_maker, "last_finish_reason", None
            )
            usage = getattr(self.decision_maker, "last_usage", None)
            record: dict[str, Any] = {
                "decision_index": decision_index,
                **decision,
                "http_status": http_status,
                "vlm_finish_reason": finish_reason,
                "vlm_usage": usage,
                "camera_names": list(images),
                "base_pose_before": _read_base_pose(env, observation),
                "executed_env_steps": 0,
            }
            decisions.append(record)
            # Preserve every successful request body as well, so parser behavior
            # can be compared across decision 0/1/2/... without bloating the
            # main result JSON with duplicate transport metadata.
            if isinstance(raw_http_response, str):
                (
                    output_dir / f"decision_{decision_index:02d}_http_response.json"
                ).write_text(raw_http_response + "\n", encoding="utf-8")

            if decision["confidence"] < self.min_confidence:
                failure_code = "LOCAL_POSE_LOW_CONFIDENCE"
                break
            if decision["action"] == "UNRESOLVED":
                failure_code = "LOCAL_POSE_UNRESOLVED"
                break
            # Repeating the same discrete direction is valid after a fresh visual
            # observation. A 30-degree correction may legitimately require three
            # consecutive 10-degree TURN_LEFT_SMALL actions. Safety is enforced by
            # motion budgets, cross-axis drift limits, and oscillation detection.
            record["same_as_previous_action"] = bool(
                executed_actions and decision["action"] == executed_actions[-1]
            )
            if decision["action"] == "STOP":
                valid_stop = bool(
                    decision["operation_ready"] and decision["target_visible"]
                )
                if valid_stop:
                    status = "success"
                    success = True
                    record["stop_validation"] = "accepted"
                else:
                    invalid_stop_count += 1
                    last_validation_error = (
                        "STOP rejected: target_visible and operation_ready must "
                        "both be true. Reassess all three alignment axes and return "
                        "a corrective direction."
                    )
                    record["stop_validation"] = "rejected"
                    record["validation_error"] = last_validation_error
                    if invalid_stop_count <= self.max_invalid_stops:
                        continue
                    failure_code = "LOCAL_POSE_INVALID_STOP"
                break

            action_axis = _action_axis(decision["action"])
            if action_axis is None:
                failure_code = "LOCAL_POSE_INVALID_ACTION_AXIS"
                break
            prospective_consecutive_count = (
                consecutive_axis_count + 1 if consecutive_axis == action_axis else 1
            )
            step_profile, reversed_direction = self._adaptive_step_for_action(
                action_name=decision["action"],
                held_mode=held_mode,
                axis_scale_levels=axis_scale_levels,
                last_axis_actions=last_axis_actions,
                consecutive_axis_count=prospective_consecutive_count,
            )
            selected_level = int(step_profile["level"])
            if reversed_direction:
                axis_reversal_counts[action_axis] += 1
                axis_scale_levels[action_axis] = max(
                    axis_scale_levels[action_axis], selected_level
                )
            elif prospective_consecutive_count >= 2:
                axis_scale_levels[action_axis] = max(
                    axis_scale_levels[action_axis], selected_level
                )

            if (
                reversed_direction
                and selected_level >= 2
                and axis_reversal_counts[action_axis] > self.max_fine_axis_reversals
            ):
                failure_code = "LOCAL_POSE_FINE_OSCILLATION"
                record["oscillation_axis"] = action_axis
                record["axis_reversal_count"] = axis_reversal_counts[action_axis]
                break

            record["step_scale"] = step_profile["scale"]
            record["adaptive_step"] = dict(step_profile)
            record["direction_reversal"] = reversed_direction
            record["axis_reversal_count"] = axis_reversal_counts[action_axis]

            requested_translation = (
                float(step_profile["translation_m"])
                if decision["action"] in _TRANSLATION_ACTIONS
                else 0.0
            )
            requested_rotation = (
                float(step_profile["rotation_deg"])
                if decision["action"] in _ROTATION_ACTIONS
                else 0.0
            )
            if (
                decision["action"] in _TRANSLATION_ACTIONS
                and total_translation_m + requested_translation
                > self.max_total_translation_m + 1e-9
            ) or (
                decision["action"] in _ROTATION_ACTIONS
                and total_rotation_deg + requested_rotation
                > self.max_total_rotation_deg + 1e-9
            ):
                failure_code = "LOCAL_POSE_MOTION_BUDGET_EXCEEDED"
                record["budget_before"] = {
                    "translation_m": total_translation_m,
                    "rotation_deg": total_rotation_deg,
                }
                record["requested_step"] = dict(step_profile)
                break

            observation, motion, guard_failure = self._execute_correction(
                env=env,
                observation=observation,
                action_name=decision["action"],
                guard=guard,
                held_mode=held_mode,
                start_step=total_env_steps,
                translation_distance_override_m=float(step_profile["translation_m"]),
                rotation_angle_override_deg=float(step_profile["rotation_deg"]),
                step_scale=str(step_profile["scale"]),
            )
            record["motion"] = motion
            record["executed_env_steps"] = int(motion["steps"])
            record["base_pose_after"] = motion["final_pose"]
            total_env_steps += int(motion["steps"])
            total_translation_m += float(motion["actual_translation_m"])
            total_rotation_deg += float(motion["actual_rotation_deg"])
            executed_actions.append(decision["action"])
            if consecutive_axis == action_axis:
                consecutive_axis_count += 1
            else:
                consecutive_axis = action_axis
                consecutive_axis_count = 1
            last_axis_actions[action_axis] = decision["action"]
            if guard_failure is not None:
                status = "failed"
                failure_code = str(
                    guard_failure.get("failure_code", "OBJECT_DROPPED")
                )
                record["guard_failure"] = guard_failure
                break
            motion_failure = motion.get("failure_code")
            if motion_failure is not None:
                motion_failure = str(motion_failure)
                recoverable_motion_failures = {
                    "LOCAL_POSE_MOTION_STALLED",
                    "LOCAL_POSE_MOTION_MAX_STEPS",
                }
                if (
                    motion_failure in recoverable_motion_failures
                    and motion_recovery_count < self.max_motion_recoveries
                ):
                    motion_recovery_count += 1
                    record["motion_recovery"] = "reobserve"
                    last_validation_error = (
                        f"Previous correction {decision['action']} ended with "
                        f"{motion_failure}. The base is now at the reported final pose. "
                        "Reassess the fresh camera views; choose the correction supported "
                        "by the current images rather than assuming the prior command "
                        "completed exactly."
                    )
                    continue
                failure_code = motion_failure
                break
            last_validation_error = None
            # Inverse cycles are handled by coarse-to-fine scale reduction above.
            # Only repeated reversals at the fine scale become a hard failure.

        if not success and failure_code is None:
            failure_code = "LOCAL_POSE_MAX_DECISIONS"
        if not success and failure_code is not None:
            status = "failed"
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
            "held_mode": held_mode,
            "invalid_stop_count": invalid_stop_count,
            "invalid_decision_count": invalid_decision_count,
            "motion_recovery_count": motion_recovery_count,
            "total_translation_m": total_translation_m,
            "total_rotation_deg": total_rotation_deg,
            "action_distribution": {
                name: executed_actions.count(name)
                for name in sorted(set(executed_actions))
            },
            "motion_budget": {
                "max_translation_m": self.max_total_translation_m,
                "max_rotation_deg": self.max_total_rotation_deg,
            },
            "adaptive_refinement": {
                "translation_scales_m": {
                    "coarse": self.translation_distance_m,
                    "medium": self.medium_translation_distance_m,
                    "fine": self.fine_translation_distance_m,
                },
                "rotation_scales_deg": {
                    "coarse": self.rotation_angle_deg,
                    "medium": self.medium_rotation_angle_deg,
                    "fine": self.fine_rotation_angle_deg,
                },
                "axis_scale_levels": {
                    axis: _SCALE_NAMES[min(2, max(0, level))]
                    for axis, level in axis_scale_levels.items()
                },
                "axis_reversal_counts": dict(axis_reversal_counts),
            },
            "decisions": decisions,
            "held_object_guard": guard.to_dict(),
            "artifact_dir": str(output_dir),
        }
        (output_dir / "result.json").write_text(
            json.dumps(_jsonable(result), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return result