"""Shadow simulator comparisons that can never affect VLM runtime decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

try:
    from .grounding import GroundingResult, grounding_result_to_dict
    from .vlm_artifacts import write_json
except ImportError:
    from grounding import GroundingResult, grounding_result_to_dict
    from vlm_artifacts import write_json


class OracleLoggingGrounder:
    """Return only the primary result and record a best-effort oracle comparison."""

    def __init__(self, *, primary: Any, oracle: Any, log_dir: str | Path):
        self.primary = primary
        self.oracle = oracle
        self.log_dir = Path(log_dir)
        self._count = 0

    def build_navigation_call(self, **kwargs: Any) -> Any:
        return self.primary.build_navigation_call(**kwargs)

    def ground(self, **kwargs: Any) -> GroundingResult | Mapping[str, Any]:
        vlm_result = self.primary.ground(**kwargs)
        self._count += 1
        simulator_gt: dict[str, Any] | None = None
        oracle_error: str | None = None
        try:
            simulator_gt = grounding_result_to_dict(self.oracle.ground(**kwargs))
        except Exception as exc:  # Oracle is strictly non-authoritative.
            oracle_error = str(exc)
        vlm_dict = grounding_result_to_dict(vlm_result)
        agreement = bool(
            simulator_gt is not None
            and vlm_dict.get("grounded") == simulator_gt.get("grounded")
            and vlm_dict.get("status") == simulator_gt.get("status")
        )
        write_json(
            self.log_dir / "oracle_grounding" / f"{self._count:06d}.json",
            {
                "vlm_result": vlm_dict,
                "simulator_gt": simulator_gt,
                "agreement": agreement,
                "oracle_error": oracle_error,
                "runtime_source": "vlm_result",
            },
        )
        return vlm_result


class OracleLoggingVerifier:
    """Return only VLM verification while logging simulator shadow output."""

    def __init__(self, *, primary: Any, oracle: Any, log_dir: str | Path):
        if not callable(primary) or not callable(oracle):
            raise TypeError("primary and oracle verifiers must be callable")
        self.primary = primary
        self.oracle = oracle
        self.log_dir = Path(log_dir)
        self._count = 0

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        vlm_result = self.primary(**kwargs)
        self._count += 1
        simulator_gt: dict[str, Any] | None = None
        oracle_error: str | None = None
        try:
            simulator_gt = dict(self.oracle(**kwargs))
        except Exception as exc:  # Oracle is strictly non-authoritative.
            oracle_error = str(exc)
        vlm_prediction = bool(vlm_result.get("goal_satisfied", False))
        simulator_prediction = (
            bool(simulator_gt.get("goal_satisfied", False))
            if simulator_gt is not None
            else None
        )
        agreement = bool(
            simulator_gt is not None
            and vlm_prediction == simulator_prediction
            and vlm_result.get("status") == simulator_gt.get("status")
        )
        contradicting_views: list[str] = []
        rejected_evidence: list[dict[str, Any]] = []
        for item in vlm_result.get("state_evidence", []):
            if not isinstance(item, Mapping):
                continue
            for rejected in item.get("rejected_evidence", []) or []:
                if isinstance(rejected, Mapping):
                    rejected_evidence.append(dict(rejected))
                    camera = rejected.get("camera_name")
                    if isinstance(camera, str) and camera:
                        contradicting_views.append(camera)
            evidence = item.get("evidence")
            extracted = (
                evidence.get("extracted")
                if isinstance(evidence, Mapping)
                else item.get("extracted_evidence")
            )
            if isinstance(extracted, Mapping):
                for camera in extracted.get("contradicting_views", []) or []:
                    if isinstance(camera, str) and camera:
                        contradicting_views.append(camera)
        write_json(
            self.log_dir / "oracle_verification" / f"{self._count:06d}.json",
            {
                "vlm_result": vlm_result,
                "vlm_prediction": vlm_prediction,
                "simulator_gt": simulator_gt,
                "simulator_prediction": simulator_prediction,
                "agreement": agreement,
                "false_positive": bool(
                    simulator_prediction is not None
                    and vlm_prediction
                    and not simulator_prediction
                ),
                "false_negative": bool(
                    simulator_prediction is not None
                    and not vlm_prediction
                    and simulator_prediction
                ),
                "contradicting_views": list(dict.fromkeys(contradicting_views)),
                "rejected_evidence": rejected_evidence,
                "oracle_error": oracle_error,
                "runtime_source": "vlm_result",
            },
        )
        return vlm_result
