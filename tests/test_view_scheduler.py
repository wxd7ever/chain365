from __future__ import annotations

from pathlib import Path

import pytest

from robocasa.view_scheduler import FixedViewScheduler


AVAILABLE_VIEWS = (
    "robot0_topview",
    "robot0_frontview",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_agentview_front",
    "robot0_eye_in_hand",
)


def call(atomic_task: str) -> dict[str, str]:
    return {"atomic_task": atomic_task}


@pytest.mark.parametrize(
    ("atomic_task", "purpose", "expected"),
    [
        (
            "PickObject",
            "critic",
            [
                "robot0_eye_in_hand",
                "robot0_agentview_left",
                "robot0_agentview_right",
            ],
        ),
        (
            "PlaceObject",
            "critic",
            ["robot0_agentview_front", "robot0_agentview_right"],
        ),
        (
            "NavigateKitchen",
            "critic",
            ["robot0_frontview", "robot0_agentview_left"],
        ),
        (
            "PickObject",
            "grounding",
            ["robot0_eye_in_hand", "robot0_frontview"],
        ),
        (
            "PickObject",
            "holding_verification",
            [
                "robot0_eye_in_hand",
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_frontview",
            ],
        ),
        (
            "PlaceObject",
            "placement_verification",
            ["robot0_agentview_right", "robot0_frontview"],
        ),
        (
            "NavigateKitchen",
            "navigation_verification",
            ["robot0_frontview", "robot0_agentview_left"],
        ),
        (
            "CloseMicrowave",
            "placement_verification",
            ["robot0_agentview_right", "robot0_frontview"],
        ),
        (
            "PressButton",
            "placement_verification",
            ["robot0_eye_in_hand", "robot0_frontview"],
        ),
    ],
)
def test_task_conditioned_view_selection(atomic_task, purpose, expected):
    selected = FixedViewScheduler().select_views(
        atomic_task_call=call(atomic_task),
        purpose=purpose,
        available_views=AVAILABLE_VIEWS,
    )

    assert selected == expected


def test_unknown_task_uses_default_grounding_views():
    selected = FixedViewScheduler().select_views(
        atomic_task_call=call("UnknownTask"),
        purpose="grounding",
        available_views=AVAILABLE_VIEWS,
    )

    assert selected == [
        "robot0_frontview",
        "robot0_agentview_left",
        "robot0_agentview_right",
    ]


def test_unavailable_task_views_fall_back_to_default(tmp_path: Path):
    config = tmp_path / "views.yaml"
    config.write_text(
        """
default:
  grounding: [robot0_frontview]
PickObject:
  grounding: [missing_camera]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    scheduler = FixedViewScheduler(config)

    assert scheduler.select_views(
        atomic_task_call=call("PickObject"),
        purpose="grounding",
        available_views=("robot0_frontview", "robot0_eye_in_hand"),
    ) == ["robot0_frontview"]


def test_work_pose_selection_is_filtered_by_available_views():
    selected = FixedViewScheduler().select_views(
        atomic_task_call=call("PickObject"),
        purpose="work_pose_refinement",
        available_views=("robot0_frontview", "robot0_eye_in_hand"),
    )

    assert selected == ["robot0_frontview", "robot0_eye_in_hand"]


def test_scheduler_rejects_unknown_runtime_purpose():
    with pytest.raises(ValueError, match="unsupported view-selection purpose"):
        FixedViewScheduler().select_views(
            atomic_task_call=call("PickObject"),
            purpose="verification",
            available_views=AVAILABLE_VIEWS,
        )
