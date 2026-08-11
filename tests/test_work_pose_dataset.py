from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from robocasa.atomic_task_schemas import (
    AtomicTaskCall,
    load_available_atomic_tasks,
    validate_atomic_task_call,
)
from robocasa.skill_contract_registry import apply_skill_contract
from robocasa.work_pose_dataset import (
    ANNOTATION_COLUMNS,
    AnnotationSegment,
    annotation_segments,
    apply_local_pose_delta,
    ensure_model_cameras,
    generate_pose_perturbations,
    pose_error,
    steam_operation_spec,
    yaw_quaternion_xyzw,
)


def test_annotation_segments_decodes_official_run_length_encoding():
    rows = [
        (10, 20, 30, 0),
        (10, 20, 30, 0),
        (11, 21, 31, 1),
        (11, 21, 31, 1),
        (11, 21, 31, 1),
    ]
    frame = pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)
    labels = {
        10: "Pick the potato.",
        20: "PickPlaceSinkToCounter",
        30: "Pick",
        11: "Place the potato.",
        21: "PickPlaceSinkToCounter",
        31: "Place",
    }

    segments = annotation_segments(frame, labels)

    assert [(value.start_frame, value.end_frame) for value in segments] == [
        (0, 1),
        (2, 4),
    ]
    assert segments[0].length == 2
    assert segments[0].stage == "pick"
    assert segments[1].subtask_idx == 1
    assert segments[1].stage == "place"


@pytest.mark.parametrize(
    ("subtask_idx", "stage", "text", "operation", "task", "object_id", "target_id"),
    [
        (0, "pick", "Pick the potato from the sink.", "pick", "PickObject", "vegetable", "vegetable"),
        (1, "place", "Place the potato in the bowl.", "place", "PlaceObject", "vegetable", "bowl"),
        (2, "pick", "Pick up the bowl.", "pick", "PickObject", "bowl", "bowl"),
        (4, "place", "Place the bowl in the microwave.", "place", "PlaceObject", "bowl", "microwave"),
    ],
)
def test_steam_annotations_map_to_four_policy_only_skills(
    subtask_idx,
    stage,
    text,
    operation,
    task,
    object_id,
    target_id,
):
    segment = AnnotationSegment(
        subtask_idx=subtask_idx,
        start_frame=10,
        end_frame=20,
        subtask=text,
        source_atomic_task="PickPlaceSinkToCounter",
        stage=stage,
    )

    result = steam_operation_spec(
        segment,
        {"object_cfgs": [{"name": "vegetable", "info": {"cat": "potato"}}]},
    )

    assert result is not None
    assert result["operation"] == operation
    assert result["atomic_task_call"]["atomic_task"] == task
    assert result["object_id"] == object_id
    assert result["target_id"] == target_id
    validate_atomic_task_call(result["atomic_task_call"])
    if operation == "pick":
        assert result["atomic_task_call"]["termination_condition"]["predicate"] == "holding"
    else:
        predicates = {
            value["predicate"]
            for value in result["atomic_task_call"]["termination_condition"]
        }
        assert {"inside", "released", "gripper_far"} <= predicates


def test_local_pose_delta_and_error_use_robot_frame():
    expert = {
        "position": [1.0, 2.0, 0.7],
        "quaternion_xyzw": yaw_quaternion_xyzw(math.pi / 2),
        "yaw_rad": math.pi / 2,
    }

    moved = apply_local_pose_delta(
        expert,
        forward_m=0.20,
        left_m=0.10,
        yaw_rad=math.radians(-15),
    )
    error = pose_error(moved, expert)

    assert moved["position"][:2] == pytest.approx([0.9, 2.2])
    assert error["translation_m"] == pytest.approx(math.hypot(0.2, 0.1))
    assert error["yaw_deg"] == pytest.approx(15.0)


def test_pose_perturbations_are_seeded_and_within_difficulty_ranges():
    record = {
        "episode_index": 7,
        "operation": "pick",
        "object_id": "vegetable",
        "snapshot_id": "snapshot",
        "segment": {"subtask_idx": 0},
        "expert_base_pose": {
            "position": [0.0, 0.0, 0.7],
            "quaternion_xyzw": yaw_quaternion_xyzw(0.0),
            "yaw_rad": 0.0,
        },
    }

    first = generate_pose_perturbations(
        [record], difficulties=["mild"], samples_per_stage=8, seed=123
    )
    second = generate_pose_perturbations(
        [record], difficulties=["mild"], samples_per_stage=8, seed=123
    )

    assert first == second
    assert len(first) == 8
    for sample in first:
        distance = sample["initial_pose_error"]["translation_m"]
        yaw = sample["initial_pose_error"]["yaw_deg"]
        assert 0.10 <= distance <= 0.20
        assert 5.0 <= yaw <= 10.0


def test_camera_injection_adds_missing_views_without_duplicates():
    xml = """
    <mujoco>
      <worldbody>
        <body name="mobilebase0_support">
          <camera name="robot0_frontview" pos="0 0 0" quat="1 0 0 0"/>
        </body>
        <body name="robot0_right_hand"/>
      </worldbody>
    </mujoco>
    """

    updated = ensure_model_cameras(
        xml,
        ("robot0_frontview", "robot0_topview", "robot0_eye_in_hand"),
    )
    names = [camera.get("name") for camera in ET.fromstring(updated).iter("camera")]

    assert names.count("robot0_frontview") == 1
    assert names.count("robot0_topview") == 1
    assert names.count("robot0_eye_in_hand") == 1


def test_virtual_pick_place_skills_are_valid_and_have_distinct_contracts(tmp_path):
    config = tmp_path / "tasks.json"
    config.write_text(json.dumps(["OpenMicrowave"]))
    assert {"PickObject", "PlaceObject"} <= load_available_atomic_tasks(config)

    pick = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "pick",
            "atomic_task": "PickObject",
            "arguments": {"object_id": "vegetable"},
            "termination_condition": {
                "predicate": "holding",
                "subject": "vegetable",
                "desired_value": True,
                "threshold": 0.05,
            },
            "policy_prompt": "Pick the potato and keep holding it.",
        }
    )
    enriched_pick, _ = apply_skill_contract(
        pick, {"objects": [{"alias": "vegetable"}], "fixtures": []}
    )
    assert enriched_pick.metadata["skill_contract"]["family"] == "object_pick"
    assert enriched_pick.termination_condition == pick.termination_condition

    place = steam_operation_spec(
        AnnotationSegment(
            subtask_idx=1,
            start_frame=1,
            end_frame=2,
            subtask="Place the potato in the bowl.",
            source_atomic_task="PickPlaceSinkToCounter",
            stage="place",
        ),
        {"object_cfgs": [{"name": "vegetable", "info": {"cat": "potato"}}]},
    )
    enriched_place, _ = apply_skill_contract(
        AtomicTaskCall.from_mapping(place["atomic_task_call"]),
        {
            "objects": [{"alias": "vegetable"}, {"alias": "bowl"}],
            "fixtures": [],
        },
    )
    predicates = [
        condition["predicate"]
        for condition in enriched_place.termination_condition
    ]
    assert enriched_place.metadata["skill_contract"]["family"] == "object_transfer"
    assert predicates == ["inside", "released", "gripper_far"]
