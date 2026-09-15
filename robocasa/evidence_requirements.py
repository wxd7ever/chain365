"""Deterministic visual-evidence requirements for VLM predicates."""

from __future__ import annotations

from typing import Sequence


HOLDING_SIDE_VIEWS = ("robot0_agentview_left", "robot0_agentview_right")

PREDICATE_REQUIRED_VIEWS: dict[str, tuple[str, ...]] = {
    "holding": ("robot0_eye_in_hand", "robot0_frontview"),
}


def required_views(predicate: str) -> tuple[str, ...]:
    """Return the views required to accept a positive predicate decision."""

    return PREDICATE_REQUIRED_VIEWS.get(str(predicate).strip().lower(), ())


def missing_required_views(
    predicate: str, supporting_views: Sequence[str]
) -> list[str]:
    supplied = {str(view) for view in supporting_views}
    predicate_name = str(predicate).strip().lower()
    missing = [
        view for view in required_views(predicate_name) if view not in supplied
    ]
    if predicate_name == "holding" and not supplied.intersection(
        HOLDING_SIDE_VIEWS
    ):
        missing.append("one_of:" + "|".join(HOLDING_SIDE_VIEWS))
    return missing


