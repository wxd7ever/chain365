"""Artifact persistence shared by visual grounding and verification."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

try:
    from .multi_view_perception import ObservationBundle
    from .vlm_client import as_rgb_uint8
except ImportError:
    from multi_view_perception import ObservationBundle
    from vlm_client import as_rgb_uint8


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "request")).strip("_")
    return text or "request"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_vlm_artifacts(
    *,
    root: str | Path | None,
    category: str,
    request_id: str,
    bundle: ObservationBundle,
    system_prompt: str,
    user_prompt: str,
    client: Any,
    parsed_json: Mapping[str, Any] | None,
    runtime_decision: Mapping[str, Any],
    evidence: Mapping[str, Any] | None = None,
    rejected_evidence: Any = None,
    final_decision: Mapping[str, Any] | None = None,
) -> tuple[str | None, list[str]]:
    """Persist prompts, all fresh images, raw/parsed responses and decision."""

    if root is None:
        return None, []
    artifact_dir = Path(root) / category / safe_name(request_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[str] = []
    for camera_name in bundle.camera_names:
        path = artifact_dir / f"{safe_name(camera_name)}.png"
        Image.fromarray(as_rgb_uint8(bundle.images[camera_name])).save(path)
        image_paths.append(str(path))
    write_json(
        artifact_dir / "request.json",
        {
            "camera_names": bundle.camera_names,
            "purpose": bundle.purpose,
            "timestamp": bundle.timestamp,
            "step_index": bundle.step_index,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
    )
    raw_http = getattr(client, "last_raw_http_response", None)
    if isinstance(raw_http, str):
        (artifact_dir / "raw_http_response.json").write_text(
            raw_http, encoding="utf-8"
        )
    raw_response = getattr(client, "last_response_text", None)
    if isinstance(raw_response, str):
        (artifact_dir / "assistant_response.txt").write_text(
            raw_response + "\n", encoding="utf-8"
        )
    write_json(artifact_dir / "parsed_response.json", parsed_json)
    write_json(
        artifact_dir / "diagnostics.json",
        {
            "http_status": getattr(client, "last_http_status", None),
            "finish_reason": getattr(client, "last_finish_reason", None),
            "usage": getattr(client, "last_usage", None),
        },
    )
    write_json(artifact_dir / "runtime_decision.json", runtime_decision)
    if evidence is not None:
        write_json(artifact_dir / "evidence.json", evidence)
    if rejected_evidence is not None:
        write_json(artifact_dir / "rejected_evidence.json", rejected_evidence)
    if final_decision is not None:
        write_json(artifact_dir / "final_decision.json", final_decision)
    return str(artifact_dir), image_paths
