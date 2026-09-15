"""Persist inspectable occupancy-map artifacts for target-aware navigation."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


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
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    return str(value)


def _candidate(candidate: Any) -> dict[str, Any]:
    if candidate is None:
        return {}
    return {
        "xy": _jsonable(getattr(candidate, "xy", None)),
        "yaw": _jsonable(getattr(candidate, "yaw", None)),
        "operation_distance": _jsonable(
            getattr(candidate, "operation_distance", None)
        ),
        "clearance_m": _jsonable(getattr(candidate, "clearance_m", None)),
        "path_cost_m": _jsonable(getattr(candidate, "path_cost_m", None)),
        "score": _jsonable(getattr(candidate, "score", None)),
        "valid": _jsonable(getattr(candidate, "valid", None)),
        "rejection_reason": _jsonable(
            getattr(candidate, "rejection_reason", None)
        ),
    }


def _safe_name(value: str) -> str:
    return _SAFE_NAME.sub("_", value.strip()).strip("_") or "navigation"


def save_navigation_plan_artifacts(
    *,
    output_root: str | Path,
    artifact_index: int,
    subgoal_id: str,
    pipeline: Any,
    navigation_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Save grids, planning metadata, and one four-panel diagnostic figure."""

    output_dir = Path(output_root) / (
        f"{int(artifact_index):06d}_{_safe_name(str(subgoal_id))}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    debug = dict(getattr(pipeline, "last_debug", {}) or {})
    occupancy_map = debug.get("map")
    metadata: dict[str, Any] = {
        "artifact_index": int(artifact_index),
        "subgoal_id": str(subgoal_id),
        "navigation_result": _jsonable(navigation_result),
        "map_available": occupancy_map is not None,
    }
    if occupancy_map is None:
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "directory": str(output_dir),
            "metadata": str(metadata_path),
            "overview": None,
            "map_available": False,
        }

    static_grid = np.asarray(
        getattr(occupancy_map, "static_grid", occupancy_map.raw_grid),
        dtype=np.uint8,
    )
    articulated_grid = np.asarray(
        getattr(
            occupancy_map,
            "articulated_grid",
            np.zeros_like(occupancy_map.raw_grid),
        ),
        dtype=np.uint8,
    )
    raw_grid = np.asarray(occupancy_map.raw_grid, dtype=np.uint8)
    inflated_grid = np.asarray(occupancy_map.grid, dtype=np.uint8)
    grid_paths = {
        "static_occupancy": output_dir / "static_occupancy.npy",
        "articulated_occupancy": output_dir / "articulated_occupancy.npy",
        "raw_occupancy": output_dir / "raw_occupancy.npy",
        "inflated_occupancy": output_dir / "inflated_occupancy.npy",
    }
    for path, grid in zip(
        grid_paths.values(),
        (static_grid, articulated_grid, raw_grid, inflated_grid),
    ):
        np.save(path, grid)

    bounds = tuple(float(value) for value in occupancy_map.world_bounds)
    extent = [bounds[0], bounds[1], bounds[2], bounds[3]]
    start = np.asarray(debug.get("start_xy", ()), dtype=float)
    collision_start = np.asarray(
        debug.get("start_collision_xy", debug.get("start_xy", ())), dtype=float
    )
    target = debug.get("target")
    target_position = np.asarray(
        getattr(target, "world_position", ()), dtype=float
    )
    raw_candidates = list(debug.get("raw_candidates", ()))
    valid_candidates = list(debug.get("valid_candidates", ()))
    selected = debug.get("selected")
    path_xy = np.asarray(
        debug.get("collision_center_path_xy", debug.get("path_xy", ())),
        dtype=float,
    )

    # Matplotlib is intentionally lazy: navigation remains usable in minimal
    # runtime environments when artifact saving is disabled.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 4, figsize=(28, 7), constrained_layout=True)
    for axis, grid, title in (
        (axes[0], static_grid, "Static occupancy"),
        (axes[1], articulated_grid, "Current articulated occupancy"),
        (axes[2], raw_grid, "Static + articulated occupancy"),
        (axes[3], inflated_grid, "Inflated occupancy + navigation plan"),
    ):
        axis.imshow(
            grid,
            origin="lower",
            extent=extent,
            cmap="Greys",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        axis.set_aspect("equal")
        axis.set_title(title)
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world y [m]")

    for item in getattr(occupancy_map, "articulated_geometries", ()):
        polygon = np.asarray(item.get("polygon_xy", ()), dtype=float)
        if polygon.ndim != 2 or len(polygon) < 3:
            continue
        closed = np.vstack((polygon, polygon[0]))
        for axis in (axes[1], axes[2], axes[3]):
            axis.plot(closed[:, 0], closed[:, 1], color="red", linewidth=1.2)

    if start.shape == (2,):
        axes[3].scatter(
            start[0], start[1], c="lime", s=80, marker="o", label="base origin"
        )
    if collision_start.shape == (2,):
        axes[3].scatter(
            collision_start[0],
            collision_start[1],
            c="blue",
            s=85,
            marker="s",
            label="footprint center",
        )
    if target_position.size >= 2:
        axes[3].scatter(
            target_position[0],
            target_position[1],
            c="orange",
            s=90,
            marker="*",
            label="target",
        )
    if raw_candidates:
        candidate_xy = np.asarray(
            [getattr(candidate, "xy", (np.nan, np.nan)) for candidate in raw_candidates],
            dtype=float,
        )
        axes[3].scatter(
            candidate_xy[:, 0],
            candidate_xy[:, 1],
            c="gold",
            s=18,
            alpha=0.55,
            label="raw candidates",
        )
    if valid_candidates:
        valid_xy = np.asarray(
            [getattr(candidate, "xy", (np.nan, np.nan)) for candidate in valid_candidates],
            dtype=float,
        )
        axes[3].scatter(
            valid_xy[:, 0],
            valid_xy[:, 1],
            c="green",
            s=28,
            alpha=0.8,
            label="valid candidates",
        )
    if selected is not None:
        selected_center = pipeline._footprint_center(selected.xy, selected.yaw)
        axes[3].scatter(
            selected_center[0],
            selected_center[1],
            c="magenta",
            s=110,
            marker="X",
            label="selected goal center",
        )
    if path_xy.ndim == 2 and len(path_xy):
        axes[3].plot(
            path_xy[:, 0],
            path_xy[:, 1],
            "c-o",
            linewidth=2,
            markersize=4,
            label="A* path",
        )
    handles, labels = axes[3].get_legend_handles_labels()
    if handles:
        axes[3].legend(handles, labels, loc="best", fontsize=8)

    reason = str(navigation_result.get("reason", "UNKNOWN"))
    figure.suptitle(f"{subgoal_id}: {reason}")
    overview_path = output_dir / "overview.png"
    figure.savefig(overview_path, dpi=180)
    plt.close(figure)

    metadata.update(
        {
            "map": {
                "resolution": float(occupancy_map.resolution),
                "origin_xy": [
                    float(occupancy_map.origin_x),
                    float(occupancy_map.origin_y),
                ],
                "world_bounds": list(bounds),
                "grid_shape": list(inflated_grid.shape),
                "inflation_radius": float(occupancy_map.inflation_radius),
                "map_debug": _jsonable(getattr(occupancy_map, "map_debug", {})),
            },
            "start_xy": _jsonable(debug.get("start_xy")),
            "start_collision_xy": _jsonable(debug.get("start_collision_xy")),
            "skip_evidence": _jsonable(debug.get("skip_evidence")),
            "target": {
                "name": _jsonable(getattr(target, "target_name", None)),
                "alias": _jsonable(getattr(target, "target_alias", None)),
                "type": _jsonable(getattr(target, "target_type", None)),
                "world_position": _jsonable(
                    getattr(target, "world_position", None)
                ),
            },
            "raw_candidates": [_candidate(value) for value in raw_candidates],
            "valid_candidates": [_candidate(value) for value in valid_candidates],
            "selected_candidate": _candidate(selected) if selected is not None else None,
            "path_xy": _jsonable(debug.get("path_xy", ())),
            "collision_center_path_xy": _jsonable(
                debug.get("collision_center_path_xy", ())
            ),
            "files": {key: str(path) for key, path in grid_paths.items()},
        }
    )
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "directory": str(output_dir),
        "metadata": str(metadata_path),
        "overview": str(overview_path),
        "map_available": True,
    }
