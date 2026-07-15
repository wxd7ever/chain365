"""Read per-task rollout horizons from this RoboCasa version's dataset registry."""

from __future__ import annotations

import ast
from pathlib import Path


_REGISTRY_PATH = Path(__file__).resolve().parent / "utils" / "dataset_registry.py"


def load_atomic_task_horizons(path: str | Path | None = None) -> dict[str, int]:
    """Return positive ``ATOMIC_TASK_DATASETS`` horizons without importing MuJoCo."""

    registry_path = Path(path) if path is not None else _REGISTRY_PATH
    tree = ast.parse(
        registry_path.read_text(encoding="utf-8"),
        filename=str(registry_path),
    )
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "ATOMIC_TASK_DATASETS"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Call):
            break
        result: dict[str, int] = {}
        for task_keyword in node.value.keywords:
            if task_keyword.arg is None or not isinstance(task_keyword.value, ast.Call):
                continue
            horizon_keyword = next(
                (
                    keyword
                    for keyword in task_keyword.value.keywords
                    if keyword.arg == "horizon"
                ),
                None,
            )
            if horizon_keyword is None:
                continue
            horizon = ast.literal_eval(horizon_keyword.value)
            if not isinstance(horizon, int) or horizon <= 0:
                raise ValueError(
                    f"Invalid horizon for {task_keyword.arg!r} in {registry_path}: {horizon!r}"
                )
            result[task_keyword.arg] = horizon
        if result:
            return result
    raise RuntimeError(f"Could not read atomic task horizons from {registry_path}")
