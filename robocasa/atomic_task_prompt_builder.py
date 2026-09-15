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
_PROMPT_STYLES = {"concise", "legacy_handoff"}


def simplify_pick_place_prompt(atomic_task: str, prompt: str) -> str:
    """Remove rollout-behavior suffixes from PickObject / PlaceObject prompts."""

    simplified = prompt.strip()
    if atomic_task == "PickObject":
        suffix_markers = (
            ", lift it clear",
            ". lift it clear",
            " and keep holding it",
            ". keep holding it",
        )
    elif atomic_task == "PlaceObject":
        simplified = re.sub(
            r"^Place\s+the\s+held\s+",
            "Place the ",
            simplified,
            flags=re.IGNORECASE,
        )
        suffix_markers = (
            ", release it",
            ". release it",
            " and release it",
        )
    else:
        return simplified

    lowered = simplified.lower()
    suffix_positions = [
        position
        for marker in suffix_markers
        if (position := lowered.find(marker)) >= 0
    ]
    if suffix_positions:
        simplified = simplified[: min(suffix_positions)]
    return simplified.rstrip(" ,.!?") + "."


def legacy_pick_place_prompt(atomic_task: str, prompt: str) -> str:
    """Restore the Pick/Place rollout language used by the August baseline."""

    legacy = prompt.strip()
    lowered = legacy.lower()
    if atomic_task == "PickObject":
        if "keep holding" not in lowered:
            legacy = legacy.rstrip(" ,.!?") + " and keep holding it."
        return legacy
    if atomic_task != "PlaceObject":
        return legacy
    if re.match(r"^Place\s+the\s+", legacy, flags=re.IGNORECASE) and not re.match(
        r"^Place\s+the\s+held\s+", legacy, flags=re.IGNORECASE
    ):
        legacy = re.sub(
            r"^Place\s+the\s+",
            "Place the held ",
            legacy,
            count=1,
            flags=re.IGNORECASE,
        )
    if "release it" not in legacy.lower():
        legacy = legacy.rstrip(" ,.!?") + ", release it, and move the gripper away."
    return legacy


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
    prompt_style: str = "concise",
) -> str:
    """Build a non-empty natural-language prompt for one atomic task."""

    call = (
        atomic_task_call
        if isinstance(atomic_task_call, AtomicTaskCall)
        else AtomicTaskCall.from_mapping(atomic_task_call)
    )
    validate_atomic_task_call(call)
    if prompt_style not in _PROMPT_STYLES:
        raise ValueError(
            f"prompt_style must be one of {sorted(_PROMPT_STYLES)}, "
            f"got {prompt_style!r}"
        )
    if call.policy_prompt is not None:
        prompt = call.policy_prompt.strip()
        if not prompt:
            raise ValueError("atomic task policy_prompt must not be empty")
        return (
            simplify_pick_place_prompt(call.atomic_task, prompt)
            if prompt_style == "concise"
            else legacy_pick_place_prompt(call.atomic_task, prompt)
        )

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
    return (
        simplify_pick_place_prompt(call.atomic_task, prompt)
        if prompt_style == "concise"
        else legacy_pick_place_prompt(call.atomic_task, prompt)
    )
