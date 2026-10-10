from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

import robocasa.work_pose_dataset as work_pose_dataset

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
    eef_pose_error,
    eef_pose_from_state,
    ensure_model_cameras,
    generate_pose_perturbations,
    move_base_to_pose,
    move_eef_to_pose,
    move_eef_to_pose_smooth,
    pose_error,
    steam_operation_spec,
    terminal_eef_pose,
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
        assert predicates == {
            "inside",
            "released",
            "eef_outside_receptacle",
        }


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


def test_stress_pose_perturbations_use_large_translation_and_yaw_ranges():
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

    samples = generate_pose_perturbations(
        [record], difficulties=["stress"], samples_per_stage=16, seed=456
    )

    assert len(samples) == 16
    assert all(sample["difficulty"] == "stress" for sample in samples)
    for sample in samples:
        distance = sample["initial_pose_error"]["translation_m"]
        yaw = sample["initial_pose_error"]["yaw_deg"]
        assert 0.60 <= distance <= 0.90
        assert 40.0 <= yaw <= 60.0


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
    assert predicates == ["inside", "released", "eef_outside_receptacle"]


def test_move_base_to_pose_slow_start_records_diagnostics_and_video(monkeypatch):
    class FakeEnv:
        def __init__(self):
            self.position = np.array([0.0, 0.0, 0.7])
            self.actions = []

        def get_observation(self):
            return {
                "robot0_base_pos": self.position.copy(),
                "robot0_base_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            }

        def step(self, action):
            self.actions.append(np.asarray(action).copy())
            self.position[0] += float(action[7]) * 0.001
            return self.get_observation(), 0.0, False, {}

        def render_camera(self, camera_name, *, height, width):
            assert camera_name == "robot0_agentview_left"
            return np.zeros((height, width, 3), dtype=np.uint8)

    monkeypatch.setattr(
        work_pose_dataset,
        "object_eef_diagnostics",
        lambda env, object_id: {
            "holding": True,
            "object_eef_distance_m": 0.04,
        },
    )
    env = FakeEnv()
    result = move_base_to_pose(
        env,
        {
            "position": [1.0, 0.0, 0.7],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "yaw_rad": 0.0,
        },
        max_steps=3,
        settle_steps=0,
        max_translation_command=0.20,
        slow_start_steps=2,
        slow_start_translation_command=0.08,
        diagnostic_object_id="bowl",
        capture_video=True,
        video_stride=1,
        video_height=8,
        video_width=8,
    )

    assert [entry["translation_command_limit"] for entry in result["trace"]] == [
        0.08,
        0.08,
        0.20,
    ]
    assert abs(env.actions[0][7]) == pytest.approx(0.08)
    assert abs(env.actions[1][7]) == pytest.approx(0.08)
    assert abs(env.actions[2][7]) == pytest.approx(0.20)
    assert all(
        entry["object_eef_distance_m"] == 0.04 for entry in result["trace"]
    )
    assert len(result["video_frames"]) == 5

def test_terminal_eef_pose_uses_official_end_window_and_quaternion_sign_alignment():
    observations = np.zeros((4, 16), dtype=np.float64)
    observations[:, 7:10] = [
        [0.40, -0.10, 0.70],
        [0.50, -0.10, 0.80],
        [0.52, -0.08, 0.82],
        [0.54, -0.06, 0.84],
    ]
    observations[:, 10:14] = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, -1.0],
    ]
    segment = AnnotationSegment(
        subtask_idx=0,
        start_frame=0,
        end_frame=3,
        subtask="pick",
        source_atomic_task="PickPlaceSinkToCounter",
        stage="pick",
    )

    pose = terminal_eef_pose(observations, segment, window=3)

    assert pose["position"] == pytest.approx([0.54, -0.06, 0.84])
    assert pose["aggregation"] == "terminal_frame"
    assert abs(pose["quaternion_xyzw"][3]) == pytest.approx(1.0)
    assert pose["window_start"] == 1
    assert pose["window_end"] == 3
    decoded = eef_pose_from_state(observations[-1])
    assert eef_pose_error(decoded, pose)["orientation_deg"] == pytest.approx(0.0)


def test_move_eef_to_expert_pose_freezes_base_and_converges():
    class FakeEnv:
        def __init__(self):
            self.position = np.zeros(3, dtype=np.float64)
            self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])
            self.actions = []

        def get_observation(self):
            return {
                "robot0_base_to_eef_pos": self.position.copy(),
                "robot0_base_to_eef_quat": self.quaternion.copy(),
            }

        def step(self, action):
            action = np.asarray(action, dtype=np.float64)
            self.actions.append(action.copy())
            self.position += action[:3] * 0.05
            return self.get_observation(), 0.0, False, {}

    env = FakeEnv()
    result = move_eef_to_pose(
        env,
        {
            "position": [0.05, -0.02, 0.03],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "frame": "robot_base",
        },
        max_steps=40,
        translation_tolerance_m=0.003,
        orientation_tolerance_deg=1.0,
        max_translation_command=0.20,
        max_rotation_command=0.10,
        stable_steps=3,
        settle_steps=2,
        gripper_action=0.5,
    )

    assert result["success"] is True
    assert result["final_error"]["translation_m"] <= 0.003
    assert env.actions
    assert all(np.allclose(action[7:10], 0.0) for action in env.actions)
    assert all(action[11] == pytest.approx(1.0) for action in env.actions)
    assert all(action[6] == pytest.approx(0.5) for action in env.actions)


def test_move_eef_to_pose_smooth_bounds_action_slew_and_converges():
    class FakeEnv:
        def __init__(self):
            self.position = np.zeros(3, dtype=np.float64)
            self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])
            self.actions = []

        def get_observation(self):
            return {
                "robot0_base_to_eef_pos": self.position.copy(),
                "robot0_base_to_eef_quat": self.quaternion.copy(),
            }

        def step(self, action):
            action = np.asarray(action, dtype=np.float64)
            self.actions.append(action.copy())
            self.position += action[:3] * 0.20
            return self.get_observation(), 0.0, False, {}

    env = FakeEnv()
    result = move_eef_to_pose_smooth(
        env,
        {
            "position": [0.06, -0.03, 0.04],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "frame": "robot_base",
        },
        max_steps=100,
        trajectory_steps=40,
        translation_tolerance_m=0.003,
        orientation_tolerance_deg=1.0,
        max_translation_command=0.10,
        max_rotation_command=0.08,
        max_translation_command_delta=0.006,
        max_rotation_command_delta=0.004,
        stable_steps=3,
    )

    arm_actions = np.asarray([action[:6] for action in env.actions])
    action_changes = np.diff(
        np.vstack([np.zeros((1, 6)), arm_actions]), axis=0
    )
    trajectory_trace = [
        item for item in result["trace"] if item["phase"] == "smooth_trajectory"
    ]

    assert result["success"] is True
    assert result["trajectory_type"] == "quintic_scurve_slerp"
    assert len(trajectory_trace) == 40
    assert trajectory_trace[0]["trajectory_progress"] < 0.001
    assert trajectory_trace[-1]["trajectory_progress"] == pytest.approx(1.0)
    assert np.max(np.abs(action_changes[:, :3])) <= 0.006001
    assert np.max(np.abs(action_changes[:, 3:6])) <= 0.004001


def test_move_eef_to_pose_zero_delta_settle_holds_arm_stationary():
    class FakeEnv:
        def __init__(self):
            self.position = np.zeros(3, dtype=np.float64)
            self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])
            self.actions = []

        def get_observation(self):
            return {
                "robot0_base_to_eef_pos": self.position.copy(),
                "robot0_base_to_eef_quat": self.quaternion.copy(),
            }

        def step(self, action):
            action = np.asarray(action, dtype=np.float64)
            self.actions.append(action.copy())
            self.position += action[:3] * 0.05
            return self.get_observation(), 0.0, False, {}

    env = FakeEnv()
    result = move_eef_to_pose(
        env,
        {
            "position": [0.05, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "frame": "robot_base",
        },
        max_steps=40,
        translation_tolerance_m=0.003,
        orientation_tolerance_deg=1.0,
        stable_steps=3,
        settle_steps=3,
        settle_mode="zero_delta",
        gripper_action=1.0,
    )

    settle_actions = env.actions[-3:]
    assert result["success"] is True
    assert result["settle_mode"] == "zero_delta"
    assert len(settle_actions) == 3
    assert all(np.allclose(action[:6], 0.0) for action in settle_actions)
    assert all(action[6] == pytest.approx(1.0) for action in settle_actions)
    assert all(action[11] == pytest.approx(1.0) for action in settle_actions)


class _InertialEefEnv:
    """Small second-order plant used to exercise simulated handoff dynamics."""

    control_timestep = 0.05

    def __init__(self, *, damping=0.65, action_gain=0.35):
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        self.actions = []
        self.damping = float(damping)
        self.action_gain = float(action_gain)

    def get_observation(self):
        return {
            "robot0_base_to_eef_pos": self.position.copy(),
            "robot0_base_to_eef_quat": self.quaternion.copy(),
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.actions.append(action.copy())
        self.velocity = (
            self.damping * self.velocity
            + self.action_gain * action[:3]
        )
        self.position += self.velocity * self.control_timestep
        return self.get_observation(), 0.0, False, {}


def _identity_target(position):
    return {
        "position": list(position),
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "frame": "robot_base",
    }


def test_smooth_handoff_bounds_first_action_from_nonzero_previous_action():
    env = _InertialEefEnv()
    previous = np.zeros(12, dtype=np.float64)
    previous[0] = 0.03
    previous[6] = 1.0
    previous[11] = 1.0

    result = move_eef_to_pose_smooth(
        env,
        _identity_target([0.025, 0.0, 0.0]),
        initial_applied_action=previous,
        max_steps=120,
        trajectory_steps=30,
        max_translation_command_delta=0.005,
        max_rotation_command_delta=0.004,
        handoff_max_steps=30,
        handoff_stable_steps=2,
        handoff_action_tolerance=0.005,
        linear_velocity_threshold_m_s=0.05,
        angular_velocity_threshold_deg_s=5.0,
        stable_steps=3,
        translation_tolerance_m=0.003,
        orientation_tolerance_deg=1.0,
        gripper_action=1.0,
    )

    first = result["trace"][0]
    assert first["phase"] == "bounded_action_handoff"
    assert first["adjacent_action_comparable"] is True
    assert max(abs(value) for value in first["adjacent_action_delta"][:3]) <= 0.005001
    assert result["handoff"]["settled"] is True
    assert result["success"] is True
    assert all(action[6] == pytest.approx(1.0) for action in env.actions)


def test_smooth_handoff_does_not_compare_actions_across_controller_modes():
    env = _InertialEefEnv()
    previous = np.zeros(12, dtype=np.float64)
    previous[0] = 0.5
    previous[11] = -1.0

    result = move_eef_to_pose_smooth(
        env,
        _identity_target([0.0, 0.0, 0.0]),
        initial_applied_action=previous,
        max_steps=20,
        trajectory_steps=2,
        stable_steps=1,
        handoff_max_steps=4,
        handoff_stable_steps=1,
    )

    assert result["handoff"]["action_modes_compatible"] is False
    assert result["trace"][0]["adjacent_action_comparable"] is False
    assert result["trace"][0]["adjacent_action_delta"] is None


def test_pose_in_tolerance_but_high_simulated_velocity_is_not_success():
    env = _InertialEefEnv(damping=1.0, action_gain=0.0)
    env.velocity[0] = 0.20

    result = move_eef_to_pose_smooth(
        env,
        _identity_target([0.0, 0.0, 0.0]),
        max_steps=3,
        trajectory_steps=1,
        stable_steps=1,
        translation_tolerance_m=0.10,
        orientation_tolerance_deg=1.0,
        linear_velocity_threshold_m_s=0.01,
        angular_velocity_threshold_deg_s=5.0,
    )

    assert result["success"] is False
    assert result["final_error"]["translation_m"] <= 0.10
    assert result["final_velocity"]["linear_m_s"] > 0.01
    assert result["final_consecutive"] == 0


def test_one_stable_step_at_budget_does_not_satisfy_eight_step_window():
    env = _InertialEefEnv(damping=0.0, action_gain=0.0)

    result = move_eef_to_pose_smooth(
        env,
        _identity_target([0.0, 0.0, 0.0]),
        max_steps=1,
        trajectory_steps=1,
        stable_steps=8,
        translation_tolerance_m=0.01,
        orientation_tolerance_deg=1.0,
    )

    assert result["success"] is False
    assert result["final_consecutive"] == 1
    assert result["end_reason"] == "STEP_BUDGET_EXHAUSTED"


def test_zero_settle_drift_outside_tolerance_is_not_success():
    class DriftEnv(_InertialEefEnv):
        def step(self, action):
            self.actions.append(np.asarray(action, dtype=np.float64).copy())
            if len(self.actions) == 2:
                self.position[0] += 0.02
            return self.get_observation(), 0.0, False, {}

    env = DriftEnv(damping=0.0, action_gain=0.0)
    result = move_eef_to_pose_smooth(
        env,
        _identity_target([0.0, 0.0, 0.0]),
        max_steps=2,
        trajectory_steps=1,
        stable_steps=1,
        settle_steps=1,
        settle_mode="zero_delta",
        translation_tolerance_m=0.005,
        orientation_tolerance_deg=1.0,
    )

    assert result["success"] is False
    assert result["final_error"]["translation_m"] > 0.005
    assert result["final_consecutive"] == 0


def test_confirmed_drop_stops_trajectory_without_normal_settle():
    class DropGuard:
        def __init__(self):
            self.started = 0

        def start(self):
            self.started += 1

        def apply_action(self, action, *, step_index):
            return action

        def observe(self, *, step_index):
            if step_index == 2:
                return {
                    "failure_code": "OBJECT_DROPPED",
                    "state_evidence": [
                        {"step_index": step_index}, {"holding": False}
                    ],
                }
            return None

        def to_dict(self):
            return {"started": self.started}

    env = _InertialEefEnv()
    result = move_eef_to_pose_smooth(
        env,
        _identity_target([0.10, 0.0, 0.0]),
        guard=DropGuard(),
        max_steps=50,
        trajectory_steps=20,
        stable_steps=3,
        settle_steps=5,
        gripper_action=1.0,
    )

    assert result["success"] is False
    assert result["end_reason"] == "OBJECT_DROPPED"
    assert result["first_drop_step"] == 2
    assert len(env.actions) == 2
    assert not any("settle" in item["phase"] for item in result["trace"])
