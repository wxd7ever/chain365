"""OpenAI-compatible VLM planner for RoboCasa long-horizon task decomposition."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .atomic_task_prompt_builder import load_atomic_task_prompts
    from .atomic_task_schemas import (
        AtomicTaskCall,
        load_available_atomic_tasks,
        validate_atomic_task_call,
    )
    from .vlm_client import (
        OpenAICompatibleVisionClient,
        extract_assistant_response,
        image_to_data_url,
    )
except ImportError:  # Direct execution from the robocasa script directory.
    from atomic_task_prompt_builder import load_atomic_task_prompts
    from atomic_task_schemas import (
        AtomicTaskCall,
        load_available_atomic_tasks,
        validate_atomic_task_call,
    )
    from vlm_client import (
        OpenAICompatibleVisionClient,
        extract_assistant_response,
        image_to_data_url,
    )


_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_LEGACY_PICK_PLACE_PREFIX = "PickPlace"


def _is_legacy_pick_place_task(task: str) -> bool:
    """Return whether ``task`` is a legacy combined pick-and-place skill."""

    return str(task).startswith(_LEGACY_PICK_PLACE_PREFIX)


def _planner_available_atomic_tasks() -> set[str]:
    """Expose decomposed transfers to new plans while retaining legacy runtime tasks."""

    return {
        task
        for task in load_available_atomic_tasks()
        if not _is_legacy_pick_place_task(task)
    }


def _planner_catalog() -> dict[str, str]:
    available = _planner_available_atomic_tasks()
    templates = load_atomic_task_prompts()
    return {
        task: templates[task][0] if task in templates else ""
        for task in sorted(available)
    }


def _encode_rgb_image(image: np.ndarray) -> str:
    """Backward-compatible wrapper around the shared image encoder."""

    return image_to_data_url(image)


def _extract_response_text(response: Mapping[str, Any]) -> str:
    """Backward-compatible alias for the shared response parser."""

    return extract_assistant_response(response)


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
        api_key: str | None,
        timeout_s: float = 120.0,
        include_images: bool = True,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
    ):
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("VLM base_url must be non-empty")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("VLM model must be non-empty")
        if timeout_s <= 0:
            raise ValueError("VLM timeout_s must be positive")
        if max_tokens <= 0:
            raise ValueError("VLM max_tokens must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.include_images = bool(include_images)
        self.max_tokens = int(max_tokens)
        self.enable_thinking = bool(enable_thinking)
        self.last_response_text: str | None = None
        self.last_raw_http_response: str | None = None
        self.last_parsed_response: dict[str, Any] | None = None
        self.last_http_status: int | None = None
        self.last_finish_reason: str | None = None
        self.last_usage: dict[str, Any] | None = None
        self._vlm_client = OpenAICompatibleVisionClient(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            timeout_s=self.timeout_s,
            max_tokens=self.max_tokens,
            enable_thinking=self.enable_thinking,
            temperature=0.0,
            top_p=None,
            top_k=None,
            min_p=None,
            presence_penalty=None,
        )

    @property
    def endpoint(self) -> str:
        return self._vlm_client.endpoint

    def _sync_client_diagnostics(self) -> None:
        self.last_response_text = self._vlm_client.last_response_text
        self.last_raw_http_response = self._vlm_client.last_raw_http_response
        self.last_parsed_response = self._vlm_client.last_parsed_response
        self.last_http_status = self._vlm_client.last_http_status
        self.last_finish_reason = self._vlm_client.last_finish_reason
        self.last_usage = self._vlm_client.last_usage

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
        system_prompt = (
            "You ground a RoboCasa entity reference to an exact simulator alias. "
            "Select only an alias copied verbatim from candidates. If the information "
            "is insufficient or ambiguous, return null. Return JSON only as "
            '{"alias": <string-or-null>}.'
        )
        user_prompt = json.dumps(
            {
                "identifier": identifier,
                "field": field,
                "condition": dict(condition),
                "atomic_task_call": compact_call,
                "candidates": list(candidates),
            },
            ensure_ascii=False,
        )
        try:
            resolved = self._vlm_client.query_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=None,
                max_tokens=min(self.max_tokens, 256),
                enable_thinking=False,
            )
        finally:
            self._sync_client_diagnostics()
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
            "for the destination. For every movable-object transfer, emit a PickObject "
            "call followed by a PlaceObject call. NEVER emit a legacy combined atomic "
            "task whose name starts with PickPlace, and do not add an intermediate "
            "placement unless the user explicitly requests it. Do not restore the scene "
            "to its initial state. Opening a cabinet, drawer, refrigerator, or other "
            "fixture to access an object does NOT imply closing it afterward. Emit a "
            "Close task only when the user explicitly requests closure or when a later "
            "requested operation requires closure as a functional or safety precondition "
            "(for example, close a microwave before turning it on). Otherwise leave the "
            "opened fixture open and omit the unnecessary Close task. NEVER emit "
            "NavigateKitchen or navigation_pose. Plan "
            "only manipulation / fixture-operation skills. The runtime scheduler grounds "
            "each operation target against the current scene and inserts NavigateKitchen "
            "only when the robot is not already at that target. Do not reset the scene.\n"
            f"Atomic task catalog (task: episode-language template):\n"
            f"{json.dumps(catalog, ensure_ascii=False, indent=2)}"
        )

    def _user_prompt(
        self,
        *,
        task: str,
        scene_context: Mapping[str, Any] | None,
    ) -> str:
        context = json.dumps(scene_context or {}, ensure_ascii=False)
        text = (
            f"Long-horizon task: {task.strip()}\n"
            f"Scene context: {context}\n"
            "Produce the ordered atomic_task_calls now."
        )
        return text

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
        image_items = (
            [
                (f"planner_view_{index}", image)
                for index, image in enumerate(images or [])
            ]
            if self.include_images
            else []
        )
        try:
            response_value = self._vlm_client.query_json(
                system_prompt=self._system_prompt(),
                user_prompt=self._user_prompt(
                    task=task,
                    scene_context=scene_context,
                ),
                images=image_items,
                max_tokens=self.max_tokens,
                enable_thinking=self.enable_thinking,
            )
        finally:
            self._sync_client_diagnostics()
        calls = parse_vlm_task_plan(response_value)
        legacy_tasks = sorted(
            {
                call.atomic_task
                for call in calls
                if _is_legacy_pick_place_task(call.atomic_task)
            }
        )
        if legacy_tasks:
            raise ValueError(
                "VLM plan used legacy combined PickPlace task(s): "
                f"{', '.join(legacy_tasks)}. Use separate PickObject and "
                "PlaceObject calls instead."
            )
        provenance = {
            "base_url": self.base_url,
            "endpoint": self.endpoint,
            "model": self.model,
            "num_images": len(image_items),
            "raw_response_text": self.last_response_text,
            "raw_http_response": self.last_raw_http_response,
            "finish_reason": self.last_finish_reason,
            "usage": self.last_usage,
        }
        return calls, provenance


def load_vlm_task_plan(path: str | Path) -> list[AtomicTaskCall]:
    """Load a previously saved VLM plan for deterministic replay."""

    plan_path = Path(path)
    return parse_vlm_task_plan(json.loads(plan_path.read_text(encoding="utf-8")))
