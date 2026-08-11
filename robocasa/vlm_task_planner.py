"""OpenAI-compatible VLM planner for RoboCasa long-horizon task decomposition."""

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
    from .atomic_task_prompt_builder import load_atomic_task_prompts
    from .atomic_task_schemas import (
        AtomicTaskCall,
        load_available_atomic_tasks,
        validate_atomic_task_call,
    )
except ImportError:  # Direct execution from the robocasa script directory.
    from atomic_task_prompt_builder import load_atomic_task_prompts
    from atomic_task_schemas import (
        AtomicTaskCall,
        load_available_atomic_tasks,
        validate_atomic_task_call,
    )


_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _planner_catalog() -> dict[str, str]:
    available = load_available_atomic_tasks()
    templates = load_atomic_task_prompts()
    return {
        task: templates[task][0] if task in templates else ""
        for task in sorted(available)
    }


def _encode_rgb_image(image: np.ndarray) -> str:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"VLM image must have HWC RGB shape, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_response_text(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("VLM response is missing a non-empty 'choices' list")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise RuntimeError("VLM response choices[0] must be a mapping")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("VLM response choices[0].message must be a mapping")
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping) and item.get("type") in {"text", "output_text"}
        ]
        text = "".join(parts).strip()
        if text:
            return text
    raise RuntimeError("VLM response message content does not contain text")


def parse_vlm_task_plan(value: Any) -> list[AtomicTaskCall]:
    """Parse and validate a VLM response or saved plan using the real allowlist."""

    if isinstance(value, str):
        text = _JSON_FENCE.sub("", value.strip())
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"VLM did not return valid JSON: {exc}") from exc
    if isinstance(value, Mapping):
        for key in ("atomic_task_calls", "task_plan", "steps"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, list) or not value:
        raise ValueError(
            "VLM task plan must be a non-empty list or contain atomic_task_calls/task_plan/steps"
        )
    calls: list[AtomicTaskCall] = []
    seen_subgoals: set[str] = set()
    for index, item in enumerate(value):
        if isinstance(item, Mapping) and "atomic_task_call" in item:
            item = item["atomic_task_call"]
        call = AtomicTaskCall.from_mapping(item)
        validate_atomic_task_call(call)
        if call.subgoal_id in seen_subgoals:
            raise ValueError(f"Duplicate VLM subgoal_id {call.subgoal_id!r}")
        seen_subgoals.add(call.subgoal_id)
        calls.append(call)
    return calls


class OpenAICompatibleVLMPlanner:
    """Decompose one language task through an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_s: float = 120.0,
        include_images: bool = True,
    ):
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("VLM base_url must be non-empty")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("VLM model must be non-empty")
        if timeout_s <= 0:
            raise ValueError("VLM timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.include_images = bool(include_images)
        self.last_response_text: str | None = None

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def resolve_entity_alias(
        self,
        *,
        identifier: str,
        field: str,
        candidates: Sequence[Mapping[str, Any]],
        condition: Mapping[str, Any],
        atomic_task_call: Mapping[str, Any],
    ) -> str | None:
        """Ask the VLM to select one candidate, without trusting invented aliases."""

        compact_call = {
            key: atomic_task_call.get(key)
            for key in ("atomic_task", "policy_prompt", "arguments")
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You ground a RoboCasa entity reference to an exact simulator "
                        "alias. Select only an alias copied verbatim from candidates. "
                        "If the information is insufficient or ambiguous, return null. "
                        "Return JSON only as {\"alias\": <string-or-null>}."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "identifier": identifier,
                            "field": field,
                            "condition": dict(condition),
                            "atomic_task_call": compact_call,
                            "candidates": list(candidates),
                        },
                        ensure_ascii=False,
                    ),
                },
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
                f"VLM alias resolver HTTP {exc.code} from {self.endpoint}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach VLM alias resolver {self.endpoint}: {exc.reason}"
            ) from exc
        if not isinstance(response_data, Mapping):
            raise RuntimeError("VLM alias resolver returned non-object JSON")
        response_text = _extract_response_text(response_data)
        text = _JSON_FENCE.sub("", response_text.strip())
        try:
            resolved = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"VLM alias resolver did not return valid JSON: {exc}"
            ) from exc
        if not isinstance(resolved, Mapping) or set(resolved) != {"alias"}:
            raise RuntimeError(
                "VLM alias resolver response must contain exactly an alias field"
            )
        alias = resolved["alias"]
        if alias is None:
            return None
        if not isinstance(alias, str) or not alias.strip():
            raise RuntimeError("VLM alias resolver alias must be a string or null")
        return alias.strip()


    def _system_prompt(self) -> str:
        catalog = _planner_catalog()
        return (
            "You are a RoboCasa task planner. Decompose the user's long-horizon task "
            "into the smallest necessary ordered atomic tasks. Use ONLY atomic_task names "
            "from the catalog below; never invent aliases such as OpenSingleDoor or "
            "PickPlaceObject. Return JSON only with shape "
            "{\"atomic_task_calls\": [AtomicTaskCall, ...]}. Each AtomicTaskCall must "
            "contain subgoal_id, atomic_task, policy_prompt, arguments, and "
            "termination_condition. policy_prompt must be a direct natural-language "
            "instruction for only that step. Use simulator-readable entity names such "
            "as apple and microwave in termination conditions. When scene context lists "
            "entity aliases, use those aliases exactly for subject/object while using "
            "natural names in policy_prompt. Supported predicates are "
            "open, closed, powered, inside, on, holding, inserted, pressed, turned, and "
            "fixture_state. For relations use subject for the movable object and object "
            "for the destination. NEVER emit NavigateKitchen or navigation_pose. Plan "
            "only manipulation / fixture-operation skills. The runtime scheduler grounds "
            "each operation target against the current scene and inserts NavigateKitchen "
            "only when the robot is not already at that target. Do not reset the scene.\n"
            f"Atomic task catalog (task: episode-language template):\n"
            f"{json.dumps(catalog, ensure_ascii=False, indent=2)}"
        )

    def _user_content(
        self,
        *,
        task: str,
        image_data_urls: Sequence[str] | None,
        scene_context: Mapping[str, Any] | None,
    ) -> str | list[dict[str, Any]]:
        context = json.dumps(scene_context or {}, ensure_ascii=False)
        text = (
            f"Long-horizon task: {task.strip()}\n"
            f"Scene context: {context}\n"
            "Produce the ordered atomic_task_calls now."
        )
        if not self.include_images or not image_data_urls:
            return text
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        content.extend(
            {"type": "image_url", "image_url": {"url": url}}
            for url in image_data_urls
        )
        return content

    def plan(
        self,
        *,
        task: str,
        images: Sequence[np.ndarray] | None = None,
        scene_context: Mapping[str, Any] | None = None,
    ) -> tuple[list[AtomicTaskCall], dict[str, Any]]:
        """Call the VLM once and return validated calls plus request provenance."""

        if not isinstance(task, str) or not task.strip():
            raise ValueError("long-horizon task must be a non-empty string")
        image_urls = [_encode_rgb_image(image) for image in (images or [])]
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": self._user_content(
                        task=task,
                        image_data_urls=image_urls,
                        scene_context=scene_context,
                    ),
                },
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
            raise RuntimeError(f"VLM HTTP {exc.code} from {self.endpoint}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach VLM endpoint {self.endpoint}: {exc.reason}") from exc
        if not isinstance(response_data, Mapping):
            raise RuntimeError("VLM endpoint returned a non-object JSON response")
        response_text = _extract_response_text(response_data)
        self.last_response_text = response_text
        calls = parse_vlm_task_plan(response_text)
        provenance = {
            "base_url": self.base_url,
            "endpoint": self.endpoint,
            "model": self.model,
            "num_images": len(image_urls),
            "raw_response_text": response_text,
            "usage": response_data.get("usage"),
        }
        return calls, provenance


def load_vlm_task_plan(path: str | Path) -> list[AtomicTaskCall]:
    """Load a previously saved VLM plan for deterministic replay."""

    plan_path = Path(path)
    return parse_vlm_task_plan(json.loads(plan_path.read_text(encoding="utf-8")))
