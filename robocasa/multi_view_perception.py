"""Fresh, labelled multi-view observations for high-level visual reasoning."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .vlm_client import as_rgb_uint8
except ImportError:
    from vlm_client import as_rgb_uint8


DEFAULT_REASONING_CAMERA_NAMES = (
    "robot0_topview",
    "robot0_frontview",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)


@dataclass(frozen=True)
class ObservationBundle:
    images: Mapping[str, np.ndarray]
    camera_names: tuple[str, ...]
    purpose: str
    timestamp: float
    step_index: int | None = None


class MultiViewObservationProvider:
    """Render fresh reasoning views without touching the pi0.5 camera contract."""

    def __init__(
        self,
        *,
        camera_names: Sequence[str] = DEFAULT_REASONING_CAMERA_NAMES,
        image_size: int = 256,
    ):
        names = tuple(str(name).strip() for name in camera_names if str(name).strip())
        if not names or len(names) != len(set(names)):
            raise ValueError("reasoning camera_names must be non-empty and unique")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.camera_names = names
        self.image_size = int(image_size)

    @staticmethod
    def _step_index(env: Any) -> int | None:
        for name in ("env_steps", "step_count", "timestep", "_elapsed_steps"):
            value = getattr(env, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    def observe(
        self,
        *,
        env: Any,
        camera_names: Sequence[str] | None = None,
        purpose: str,
    ) -> ObservationBundle:
        names = self.camera_names if camera_names is None else tuple(camera_names)
        names = tuple(str(name).strip() for name in names if str(name).strip())
        if not names or len(names) != len(set(names)):
            raise ValueError("camera_names must be non-empty and unique")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("purpose must be non-empty")
        renderer = getattr(env, "render_camera_views", None)
        if not callable(renderer):
            raise ValueError("environment does not provide render_camera_views()")
        rendered = renderer(
            list(names), height=self.image_size, width=self.image_size
        )
        if not isinstance(rendered, Mapping):
            raise TypeError("render_camera_views() must return a mapping")
        missing = [name for name in names if name not in rendered]
        if missing:
            raise KeyError(f"render_camera_views() omitted cameras: {missing}")
        images = {name: as_rgb_uint8(rendered[name]) for name in names}
        return ObservationBundle(
            images=images,
            camera_names=names,
            purpose=purpose.strip(),
            timestamp=time.time(),
            step_index=self._step_index(env),
        )
