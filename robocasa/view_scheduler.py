"""Task-conditioned fixed camera selection for VLM reasoning modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


DEFAULT_VIEW_SELECTION_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "view_selection.yaml"
)

VIEW_SELECTION_PURPOSES = frozenset(
    {
        "grounding",
        "critic",
        "holding_verification",
        "placement_verification",
        "navigation_verification",
        "work_pose_refinement",
    }
)


def _atomic_task_name(atomic_task_call: Any) -> str:
    if isinstance(atomic_task_call, Mapping):
        value = atomic_task_call.get("atomic_task")
    else:
        value = getattr(atomic_task_call, "atomic_task", None)
    return str(value or "").strip()


def _unique_names(values: Sequence[Any], *, field: str) -> tuple[str, ...]:
    names = tuple(str(value).strip() for value in values if str(value).strip())
    if not names:
        raise ValueError(f"{field} must contain at least one camera name")
    if len(names) != len(set(names)):
        raise ValueError(f"{field} must not contain duplicate camera names")
    return names


class FixedViewScheduler:
    """Select a configured subset of currently available reasoning cameras.

    The YAML owns all task-to-view mappings. Runtime logic only validates the
    configuration, filters unavailable cameras, and applies deterministic
    fallback rules. It never changes the pi0.5 policy observation contract.
    """

    def __init__(self, config_path: str | Path = DEFAULT_VIEW_SELECTION_CONFIG):
        path = Path(config_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"view-selection config does not exist: {path}")
        with path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("view-selection config must be a non-empty mapping")

        config: dict[str, dict[str, tuple[str, ...]]] = {}
        allowed_config_purposes = VIEW_SELECTION_PURPOSES | {"verification"}
        for task_name, task_value in raw.items():
            name = str(task_name).strip()
            if not name or not isinstance(task_value, Mapping):
                raise ValueError(
                    "each view-selection task entry must be a named mapping"
                )
            purpose_map: dict[str, tuple[str, ...]] = {}
            for purpose, views in task_value.items():
                purpose_name = str(purpose).strip()
                if purpose_name not in allowed_config_purposes:
                    raise ValueError(
                        f"unsupported view-selection purpose {purpose_name!r} "
                        f"for task {name!r}"
                    )
                if not isinstance(views, (list, tuple)):
                    raise ValueError(
                        f"view selection {name}.{purpose_name} must be a list"
                    )
                purpose_map[purpose_name] = _unique_names(
                    views, field=f"view selection {name}.{purpose_name}"
                )
            config[name] = purpose_map
        self.config_path = path
        self._config = config

    @staticmethod
    def _purpose_candidates(purpose: str) -> tuple[str, ...]:
        if purpose.endswith("_verification"):
            return purpose, "verification"
        return (purpose,)

    def _configured_views(self, *, task_name: str, purpose: str) -> tuple[str, ...] | None:
        purpose_candidates = self._purpose_candidates(purpose)
        for section_name in (task_name, "default"):
            section = self._config.get(section_name, {})
            for candidate in purpose_candidates:
                views = section.get(candidate)
                if views is not None:
                    return views
        return None

    def select_views(
        self,
        *,
        atomic_task_call: Any,
        purpose: str,
        available_views: Sequence[str],
    ) -> list[str]:
        """Return configured cameras filtered to ``available_views``.

        If a task-specific selection has no available camera, the scheduler
        retries the matching default selection. If that is also unavailable or
        no mapping exists, it returns every available view in caller order.
        """

        purpose_name = str(purpose).strip()
        if purpose_name not in VIEW_SELECTION_PURPOSES:
            raise ValueError(
                f"unsupported view-selection purpose {purpose_name!r}; "
                f"expected one of {sorted(VIEW_SELECTION_PURPOSES)}"
            )
        available = _unique_names(available_views, field="available_views")
        available_set = set(available)
        task_name = _atomic_task_name(atomic_task_call)

        configured = self._configured_views(
            task_name=task_name,
            purpose=purpose_name,
        )
        if configured is not None:
            selected = [name for name in configured if name in available_set]
            if selected:
                return selected

        if task_name != "default":
            default_configured = self._configured_views(
                task_name="default",
                purpose=purpose_name,
            )
            if default_configured is not None:
                selected = [
                    name for name in default_configured if name in available_set
                ]
                if selected:
                    return selected
        return list(available)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "config_path": str(self.config_path),
            "tasks": sorted(self._config),
        }
