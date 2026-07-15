"""Schemas and validation for scheduler-issued RoboCasa atomic tasks."""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
_ALLOWLIST_PATH = _CONFIG_DIR / "robocasa_atomic_tasks.json"
_DATASET_REGISTRY_PATH = (
    Path(__file__).resolve().parent / "utils" / "dataset_registry.py"
)


@dataclass(frozen=True)
class AtomicTaskCall:
    """One atomic policy invocation produced by a planner or scheduler."""

    subgoal_id: str
    atomic_task: str
    arguments: dict[str, Any]
    termination_condition: dict[str, Any] | list[dict[str, Any]]
    policy_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AtomicTaskCall":
        """Parse a call without silently discarding unknown or malformed fields."""

        if not isinstance(value, Mapping):
            raise TypeError("atomic_task_call must be a mapping")
        required = {"subgoal_id", "atomic_task", "arguments", "termination_condition"}
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"atomic_task_call is missing required fields: {missing}")
        unknown = sorted(
            set(value).difference(required | {"policy_prompt", "metadata"})
        )
        if unknown:
            raise ValueError(f"atomic_task_call contains unknown fields: {unknown}")
        arguments = value["arguments"]
        metadata = value.get("metadata", {})
        condition = value["termination_condition"]
        if not isinstance(arguments, dict):
            raise TypeError("atomic_task_call.arguments must be a dict")
        if not isinstance(metadata, dict):
            raise TypeError("atomic_task_call.metadata must be a dict")
        if isinstance(condition, list):
            if not all(isinstance(item, dict) for item in condition):
                raise TypeError("every termination_condition item must be a dict")
            parsed_condition: dict[str, Any] | list[dict[str, Any]] = [
                dict(item) for item in condition
            ]
        elif isinstance(condition, dict):
            parsed_condition = dict(condition)
        else:
            raise TypeError("termination_condition must be a dict or list of dicts")
        return cls(
            subgoal_id=value["subgoal_id"],
            atomic_task=value["atomic_task"],
            policy_prompt=value.get("policy_prompt"),
            arguments=dict(arguments),
            termination_condition=parsed_condition,
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable shallow-data representation."""

        return asdict(self)


def _tasks_from_loaded_metadata() -> set[str]:
    module = sys.modules.get("robocasa.utils.dataset_registry")
    datasets = getattr(module, "ATOMIC_TASK_DATASETS", None)
    if isinstance(datasets, Mapping):
        return {str(name) for name in datasets if str(name).strip()}
    return set()


def _tasks_from_dataset_source() -> set[str]:
    """Read ATOMIC_TASK_DATASETS keys without importing RoboCasa/MuJoCo."""

    if not _DATASET_REGISTRY_PATH.is_file():
        return set()
    tree = ast.parse(
        _DATASET_REGISTRY_PATH.read_text(encoding="utf-8"),
        filename=str(_DATASET_REGISTRY_PATH),
    )
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "ATOMIC_TASK_DATASETS"
            for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.Call):
            return {
                keyword.arg
                for keyword in node.value.keywords
                if keyword.arg is not None and keyword.arg.strip()
            }
    return set()


def _tasks_from_loaded_env_registry() -> set[str]:
    module = sys.modules.get("robocasa.environments.kitchen.kitchen")
    registry = getattr(module, "REGISTERED_KITCHEN_ENVS", None)
    if not isinstance(registry, Mapping):
        return set()
    return {
        str(name)
        for name, task_class in registry.items()
        if ".environments.kitchen.atomic." in getattr(task_class, "__module__", "")
    }


def _tasks_from_config(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        value = value.get("atomic_tasks")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Atomic task allowlist {path} must contain a string list")
    return {item.strip() for item in value if item.strip()}


def load_available_atomic_tasks(config_path: str | Path | None = None) -> set[str]:
    """Load the current version's atomic tasks from metadata, registry, or JSON.

    The checked-in dataset registry is authoritative for the multitask policy: it
    includes tasks that were moved from composite to atomic while retaining their
    original Python module. The explicit JSON is therefore only a portable fallback.
    """

    if config_path is not None:
        tasks = _tasks_from_config(Path(config_path))
        if not tasks:
            raise RuntimeError(f"No atomic tasks found in {config_path}")
        return tasks
    for loader in (
        _tasks_from_loaded_metadata,
        _tasks_from_dataset_source,
        _tasks_from_loaded_env_registry,
        lambda: _tasks_from_config(_ALLOWLIST_PATH),
    ):
        tasks = loader()
        if tasks:
            return tasks
    raise RuntimeError(
        "Could not discover RoboCasa atomic tasks from dataset metadata, "
        "the task registry, or configs/robocasa_atomic_tasks.json"
    )


def validate_atomic_task_call(
    atomic_task_call: AtomicTaskCall | Mapping[str, Any],
) -> None:
    """Raise a precise exception if an atomic task call violates the contract."""

    call = (
        atomic_task_call
        if isinstance(atomic_task_call, AtomicTaskCall)
        else AtomicTaskCall.from_mapping(atomic_task_call)
    )
    if not isinstance(call.subgoal_id, str) or not call.subgoal_id.strip():
        raise ValueError("atomic_task_call.subgoal_id must be a non-empty string")
    if not isinstance(call.atomic_task, str) or not call.atomic_task.strip():
        raise ValueError("atomic_task_call.atomic_task must be a non-empty string")
    if not isinstance(call.arguments, dict):
        raise TypeError("atomic_task_call.arguments must be a dict")
    if not call.termination_condition:
        raise ValueError("atomic_task_call.termination_condition must not be empty")
    if call.policy_prompt is not None and (
        not isinstance(call.policy_prompt, str) or not call.policy_prompt.strip()
    ):
        raise ValueError("atomic_task_call.policy_prompt must be non-empty when provided")
    available = load_available_atomic_tasks()
    if call.atomic_task not in available:
        raise ValueError(
            f"Unknown RoboCasa atomic task {call.atomic_task!r}; "
            f"available tasks: {sorted(available)}"
        )
