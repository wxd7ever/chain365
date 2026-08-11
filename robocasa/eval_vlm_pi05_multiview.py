#!/usr/bin/env python3
"""Top-camera local-pose evaluation with wrist view enabled by default."""

from __future__ import annotations

import sys

try:
    from .eval_vlm_pi05_top import main
except ImportError:
    from eval_vlm_pi05_top import main


DEFAULT_LOCAL_POSE_CAMERAS = (
    "robot0_topview",
    "robot0_frontview",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)


def _has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


if __name__ == "__main__":
    if not _has_option("--local_pose_cameras"):
        sys.argv.extend(("--local_pose_cameras", *DEFAULT_LOCAL_POSE_CAMERAS))
    raise SystemExit(main())

