"""Shared OpenAI-compatible client for text or multi-view JSON decisions."""

from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class VisionVLMOutputTruncated(RuntimeError):
    """Raised when a backend exhausts its token budget before returning JSON."""


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


def as_rgb_uint8(image: Any) -> np.ndarray:
    """Return a contiguous HWC RGB uint8 image."""

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


def image_to_data_url(image: Any) -> str:
    """Encode one image as a PNG data URL accepted by OpenAI-compatible APIs."""

    buffer = io.BytesIO()
    Image.fromarray(as_rgb_uint8(image)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _nonempty_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _content_parts(value: Any) -> list[str]:
    direct = _nonempty_text(value)
    if direct is not None:
        return [direct]
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key in ("text", "output_text", "content", "value"):
            if key in value:
                parts.extend(_content_parts(value[key]))
        if parts:
            return parts
        # Some servers place the requested JSON object directly in content.
        if value:
            try:
                return [json.dumps(_jsonable(value), ensure_ascii=False)]
            except (TypeError, ValueError):
                pass
        return []
    if isinstance(value, (list, tuple)):
        return [part for item in value for part in _content_parts(item)]
    return []


def _json_object_from_text(text: str) -> str | None:
    stripped = _JSON_FENCE.sub("", text.strip()).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start, end = stripped.find("{"), stripped.rfind("}")
    if start >= 0 and end > start:
        candidate = stripped[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, Mapping):
            return candidate
    return None


def extract_assistant_response(response: Mapping[str, Any]) -> str:
    """Extract assistant JSON text from common OpenAI/vLLM response variants."""

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            "vision VLM response has no choices; "
            f"top_level_keys={list(response.keys())}"
        )
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise RuntimeError("vision VLM choice must be an object")
    message = choice.get("message")
    if isinstance(message, Mapping):
        parts = _content_parts(message.get("content"))
        if parts:
            return "".join(parts)
        for key in ("reasoning_content", "reasoning"):
            reasoning = _nonempty_text(message.get(key))
            if reasoning:
                candidate = _json_object_from_text(reasoning)
                if candidate is not None:
                    return candidate
    parts = _content_parts(choice.get("text"))
    if parts:
        return "".join(parts)
    raise RuntimeError("vision VLM response contains no assistant JSON text")


class OpenAICompatibleVisionClient:
    """Issue text or multi-view requests and return one validated JSON object."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        max_tokens: int = 16384,
        enable_thinking: bool = False,
        temperature: float = 0.6,
        top_p: float | None = 0.95,
        top_k: int | None = 20,
        min_p: float | None = 0.0,
        presence_penalty: float | None = 1.5,
    ):
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("vision VLM base_url must be non-empty")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("vision VLM model must be non-empty")
        if timeout_s <= 0:
            raise ValueError("vision VLM timeout_s must be positive")
        if max_tokens <= 0:
            raise ValueError("vision VLM max_tokens must be positive")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("vision VLM temperature must be numeric")
        if top_p is not None and (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not 0.0 <= float(top_p) <= 1.0
        ):
            raise ValueError("vision VLM top_p must be in [0, 1] or null")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
        ):
            raise ValueError("vision VLM top_k must be a positive integer or null")
        if min_p is not None and (
            isinstance(min_p, bool)
            or not isinstance(min_p, (int, float))
            or not 0.0 <= float(min_p) <= 1.0
        ):
            raise ValueError("vision VLM min_p must be in [0, 1] or null")
        if presence_penalty is not None and (
            isinstance(presence_penalty, bool)
            or not isinstance(presence_penalty, (int, float))
        ):
            raise ValueError("vision VLM presence_penalty must be numeric or null")
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)
        self.enable_thinking = bool(enable_thinking)
        self.temperature = float(temperature)
        self.top_p = None if top_p is None else float(top_p)
        self.top_k = top_k
        self.min_p = None if min_p is None else float(min_p)
        self.presence_penalty = (
            None if presence_penalty is None else float(presence_penalty)
        )
        self.last_response_text: str | None = None
        self.last_raw_http_response: str | None = None
        self.last_parsed_response: dict[str, Any] | None = None
        self.last_http_status: int | None = None
        self.last_finish_reason: str | None = None
        self.last_usage: dict[str, Any] | None = None

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _reset_diagnostics(self) -> None:
        self.last_response_text = None
        self.last_raw_http_response = None
        self.last_parsed_response = None
        self.last_http_status = None
        self.last_finish_reason = None
        self.last_usage = None

    def query_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> dict[str, Any]:
        """Query one JSON decision while retaining complete response diagnostics."""

        self._reset_diagnostics()
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty")
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user_prompt must be non-empty")
        if images is None:
            items: list[tuple[str, Any]] = []
        else:
            items = list(images.items()) if isinstance(images, Mapping) else list(images)
        request_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if isinstance(request_max_tokens, bool) or request_max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        thinking = self.enable_thinking if enable_thinking is None else bool(enable_thinking)

        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for camera_name, image in items:
            label = str(camera_name).strip()
            if not label:
                raise ValueError("camera names must be non-empty")
            content.extend(
                [
                    {"type": "text", "text": f"Camera view: {label}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image)},
                    },
                ]
            )
        user_content: str | list[dict[str, Any]] = content if items else user_prompt
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": int(request_max_tokens),
            "chat_template_kwargs": {"enable_thinking": thinking},
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        for key, value in (
            ("top_p", self.top_p),
            ("top_k", self.top_k),
            ("min_p", self.min_p),
            ("presence_penalty", self.presence_penalty),
        ):
            if value is not None:
                payload[key] = value
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
                raw = response.read().decode("utf-8", errors="replace")
                self.last_raw_http_response = raw
        except urllib.error.HTTPError as exc:
            self.last_http_status = int(exc.code)
            detail = exc.read().decode("utf-8", errors="replace")
            self.last_raw_http_response = detail
            raise RuntimeError(
                f"vision VLM HTTP {exc.code} from {self.endpoint}: {detail}"
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(
                f"could not reach vision VLM {self.endpoint}: {reason}"
            ) from exc
        try:
            response_data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "vision VLM returned invalid HTTP JSON: "
                f"{exc}; body_preview={raw[:1000]!r}"
            ) from exc
        if not isinstance(response_data, Mapping):
            raise RuntimeError("vision VLM returned non-object HTTP JSON")
        usage = response_data.get("usage")
        if isinstance(usage, Mapping):
            self.last_usage = dict(usage)
        choices = response_data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            finish_reason = choices[0].get("finish_reason")
            if finish_reason is not None:
                self.last_finish_reason = str(finish_reason)
        if self.last_finish_reason == "length":
            raise VisionVLMOutputTruncated(
                "vision VLM exhausted max_tokens before producing a complete "
                f"decision; max_tokens={request_max_tokens}; usage={self.last_usage}"
            )
        response_text = extract_assistant_response(response_data)
        self.last_response_text = response_text
        cleaned = _JSON_FENCE.sub("", response_text.strip()).strip()
        candidate = _json_object_from_text(cleaned) or cleaned
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"vision VLM did not return valid JSON: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise RuntimeError("vision VLM assistant response must be a JSON object")
        self.last_parsed_response = dict(parsed)
        return dict(parsed)
