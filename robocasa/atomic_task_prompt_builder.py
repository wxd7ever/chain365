"""Build policy language from explicit prompts or versioned episode templates."""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any, Mapping

try:
    from .atomic_task_schemas import AtomicTaskCall, validate_atomic_task_call
except ImportError:  # Direct execution from the robocasa script directory.
    from atomic_task_schemas import AtomicTaskCall, validate_atomic_task_call


_PROMPT_REGISTRY_PATH = (
    Path(__file__).resolve().parent / "configs" / "atomic_task_prompts.json"
)
_INSTANCE_SUFFIX = re.compile(r"_\d+$")


def clean_entity_name(value: str) -> str:
    """Convert simulator instance identifiers into natural-language names."""

    if not isinstance(value, str):
        raise TypeError("entity name must be a string")
    return _INSTANCE_SUFFIX.sub("", value.strip()).replace("_", " ")


def load_atomic_task_prompts(
    path: str | Path | None = None,
) -> dict[str, list[str]]:
    """Load templates transcribed from this version's atomic episode metadata."""

    prompt_path = Path(path) if path is not None else _PROMPT_REGISTRY_PATH
    value = json.loads(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Prompt registry {prompt_path} must be a JSON object")
    result: dict[str, list[str]] = {}
    for task, templates in value.items():
        if isinstance(templates, str):
            templates = [templates]
        if not isinstance(templates, list) or not templates or not all(
            isinstance(item, str) and item.strip() for item in templates
        ):
            raise ValueError(f"Prompt templates for {task!r} must be non-empty strings")
        result[str(task)] = [item.strip() for item in templates]
    return result


def _template_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, Mapping):
        raise TypeError("atomic_task_call.arguments must be a mapping")
    values = dict(arguments)
    for key, value in list(values.items()):
        if isinstance(value, str) and (key.endswith("_name") or key.endswith("_id")):
            values[key] = clean_entity_name(value)
        if key.endswith("_id") and isinstance(value, str):
            values.setdefault(f"{key[:-3]}_name", clean_entity_name(value))
    if "object_name" not in values and isinstance(values.get("subject"), str):
        values["object_name"] = clean_entity_name(values["subject"])
    return values


def build_atomic_task_prompt(
    atomic_task_call: AtomicTaskCall | Mapping[str, Any],
    *,
    prompt_registry: Mapping[str, list[str] | str] | None = None,
) -> str:
    """Build a non-empty natural-language prompt for one atomic task."""

    call = (
        atomic_task_call
        if isinstance(atomic_task_call, AtomicTaskCall)
        else AtomicTaskCall.from_mapping(atomic_task_call)
    )
    validate_atomic_task_call(call)
    if call.policy_prompt is not None:
        prompt = call.policy_prompt.strip()
        if not prompt:
            raise ValueError("atomic task policy_prompt must not be empty")
        return prompt

    registry = dict(
        load_atomic_task_prompts() if prompt_registry is None else prompt_registry
    )
    if call.atomic_task not in registry:
        raise ValueError(f"No prompt template registered for atomic task {call.atomic_task!r}")
    templates = registry[call.atomic_task]
    if isinstance(templates, str):
        templates = [templates]
    if not templates:
        raise ValueError(f"No prompt template registered for atomic task {call.atomic_task!r}")
    template = templates[0]
    values = _template_arguments(call.arguments)
    required = {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(template)
        if field_name
    }
    missing = sorted(name for name in required if name not in values)
    if missing:
        raise ValueError(
            f"Prompt template for {call.atomic_task!r} is missing arguments: {missing}"
        )
    try:
        prompt = template.format_map(values).strip()
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(
            f"Could not format prompt template for {call.atomic_task!r}: {exc}"
        ) from exc
    if not prompt:
        raise ValueError(f"Prompt for atomic task {call.atomic_task!r} is empty")
    return prompt
