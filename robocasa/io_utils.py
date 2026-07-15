"""Small filesystem helpers shared by online RoboCasa evaluation entrypoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def make_run_dir(output_root: str | Path, *, env_name: str) -> Path:
    """Create a unique UTC-timestamped directory for one evaluation run."""

    root = Path(output_root).expanduser()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / env_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: str | Path, value: Any) -> None:
    """Write human-readable JSON, including common NumPy scalar values."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def default(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    output_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=default) + "\n",
        encoding="utf-8",
    )


def _as_hwc_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    while array.ndim > 3:
        array = array[-1]
    if array.ndim != 3:
        raise ValueError(f"image must be three-dimensional, got {array.shape}")
    if array.shape[-1] not in (1, 3, 4) and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        array = array.astype(np.float32, copy=False)
        if array.max(initial=0.0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def save_obs_image(path: str | Path, image: Any) -> None:
    """Save one observation image after normalizing dtype and channel layout."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_as_hwc_uint8(image)).save(output_path)


def save_video(
    path: str | Path,
    frames: Sequence[np.ndarray],
    *,
    fps: int = 20,
) -> None:
    """Save RGB frames to a video; an empty frame sequence is a no-op."""

    if not frames:
        return
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        str(output_path),
        [_as_hwc_uint8(frame) for frame in frames],
        fps=fps,
        macro_block_size=None,
    )
