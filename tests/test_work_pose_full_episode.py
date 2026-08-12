from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import robocasa.scripts.work_pose.eval_work_pose_full_episode as full_episode
from robocasa.atomic_task_schemas import AtomicTaskCall, validate_atomic_task_call
from robocasa.scripts.work_pose.eval_work_pose_full_episode import (
    aggregate,
    execute_current_skill_with_retries,
    fixture_call,
    microwave_alias,
    parse_args,
    prepare_vla_completion_call,
    select_episode_workflows,
    stabilize_held_object,
    station_move_step_budget,
    target_degraded_pose,
)


def test_parse_args_rejects_translation_command_above_controller_limit(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--condition",
            "baseline",
            "--held_max_translation_command",
            "2.0",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()
    assert "controller limit 1.0" in capsys.readouterr().err


def test_station_move_budget_expands_for_observed_held_navigation_distance():
    budget = station_move_step_budget(
        initial_translation_m=0.278586,
        translation_tolerance_m=0.03,
        max_translation_command=0.20,
        minimum_steps=360,
        maximum_steps=25000,
    )

    assert budget == 1363
    far_budget = station_move_step_budget(
        initial_translation_m=4.103,
        translation_tolerance_m=0.03,
        max_translation_command=0.20,
        minimum_steps=360,
        maximum_steps=25000,
    )
    assert far_budget == 20485


def test_station_move_budget_keeps_floor_for_short_move_and_caps_long_move():
    assert station_move_step_budget(
        initial_translation_m=0.05,
        translation_tolerance_m=0.03,
        max_translation_command=0.20,
        minimum_steps=360,
        maximum_steps=25000,
    ) == 360
    assert station_move_step_budget(
        initial_translation_m=20.0,
        translation_tolerance_m=0.03,
        max_translation_command=0.20,
        minimum_steps=360,
        maximum_steps=25000,
    ) == 25000


def workflow_data(num_episodes: int = 2):
    records = []
    samples = []
    for episode_index in range(num_episodes):
        for stage_index in (0, 1, 2, 4):
            record_index = len(records)
            records.append(
                {
                    "episode_index": episode_index,
                    "segment": {"subtask_idx": stage_index},
                }
            )
            for rank in range(2):
                samples.append(
                    {
                        "sample_id": (
                            f"ep{episode_index:06d}_s{stage_index:02d}_"
                            f"mild_{rank:02d}"
                        ),
                        "record_index": record_index,
                        "difficulty": "mild",
                        "valid": True,
                    }
                )
    return records, samples


def test_select_episode_workflows_counts_complete_episodes_not_samples():
    records, samples = workflow_data(3)

    workflows = select_episode_workflows(
        samples=samples,
        records=records,
        difficulty="mild",
        sample_rank=1,
        episode_start=1,
        episode_count=2,
    )

    assert [value["episode_index"] for value in workflows] == [1, 2]
    assert all(
        set(value["stages"]) == {0, 1, 2, 4} for value in workflows
    )
    assert workflows[0]["stages"][0]["sample"]["sample_id"].endswith("_01")


def test_select_episode_workflows_rejects_incomplete_manifest():
    records, samples = workflow_data(1)
    samples = [
        sample
        for sample in samples
        if "_s04_" not in sample["sample_id"]
    ]

    with pytest.raises(ValueError, match="complete episodes"):
        select_episode_workflows(
            samples=samples,
            records=records,
            difficulty="mild",
            sample_rank=0,
            episode_start=0,
            episode_count=1,
        )


@pytest.mark.parametrize(
    ("stage_index", "task", "predicate"),
    [
        (5, "CloseMicrowave", "closed"),
        (6, "TurnOnMicrowave", "powered"),
    ],
)
def test_fixture_calls_are_valid(stage_index, task, predicate):
    call = fixture_call(stage_index, "microwave_left_group_1")

    validate_atomic_task_call(call)
    assert call.atomic_task == task
    assert call.termination_condition["predicate"] == predicate
    assert call.termination_condition["subject"] == "microwave_left_group_1"


def test_microwave_alias_prefers_exact_alias():
    context = {
        "fixtures": [
            {
                "alias": "microwave_left_group_1",
                "name": "microwave",
                "natural_name": "microwave",
            },
            {
                "alias": "microwave",
                "name": "microwave",
                "natural_name": "microwave",
            },
        ]
    }

    assert microwave_alias(context) == "microwave"


def test_target_degraded_pose_prefers_physically_reached_pose():
    sample = {
        "target_degraded_base_pose": {"position": [1, 2, 3]},
        "base_movement": {
            "final_pose": {
                "position": [4, 5, 6],
                "quaternion_xyzw": [0, 0, 0, 1],
            }
        },
    }

    assert target_degraded_pose(sample)["position"] == [4, 5, 6]


def test_full_episode_aggregate_reports_stage_and_task_success():
    results = [
        {
            "success": True,
            "operation_results": [
                {"stage_index": 0, "atomic_task": "PickObject", "success": True},
                {"stage_index": 1, "atomic_task": "PlaceObject", "success": True},
            ],
        },
        {
            "success": False,
            "operation_results": [
                {"stage_index": 0, "atomic_task": "PickObject", "success": False}
            ],
        },
    ]

    summary = aggregate(results)

    assert summary["num_episodes"] == 2
    assert summary["num_success"] == 1
    assert summary["full_task_success_rate"] == 0.5
    assert summary["stage_metrics"]["stage_00_PickObject"] == {
        "attempts": 2,
        "successes": 1,
        "success_rate": 0.5,
    }


def test_stabilization_runs_full_post_pick_window_and_precheck_stops_at_eight(
    monkeypatch,
):
    class FakeEnv:
        def __init__(self):
            self.steps = 0

        def step(self, action):
            self.steps += 1
            assert np.asarray(action).shape == (12,)
            return {}, 0.0, False, {}

    class FakeGuard:
        def start(self):
            pass

        def apply_action(self, action, *, step_index):
            return action

        def observe(self, *, step_index):
            return None

        def to_dict(self):
            return {"enabled": True}

    monkeypatch.setattr(
        full_episode,
        "build_held_object_guard",
        lambda **kwargs: FakeGuard(),
    )
    monkeypatch.setattr(
        full_episode,
        "object_eef_diagnostics",
        lambda env, object_id: {
            "holding": True,
            "object_eef_distance_m": 0.03,
        },
    )
    monkeypatch.setattr(full_episode, "holding_state", lambda env, object_id: True)
    call = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "pick_bowl",
            "atomic_task": "PickObject",
            "arguments": {"object_id": "bowl"},
            "termination_condition": {
                "predicate": "holding",
                "subject": "bowl",
                "desired_value": True,
            },
            "policy_prompt": "Pick up the bowl and keep holding it.",
        }
    )

    post_pick = stabilize_held_object(
        env=FakeEnv(),
        atomic_task_call=call,
        object_id="bowl",
        total_steps=40,
        required_consecutive=8,
        drop_confirmation_steps=5,
    )
    pre_navigation = stabilize_held_object(
        env=FakeEnv(),
        atomic_task_call=call,
        object_id="bowl",
        total_steps=16,
        required_consecutive=8,
        drop_confirmation_steps=5,
        stop_when_confirmed=True,
    )

    assert post_pick["success"] is True
    assert post_pick["steps"] == 40
    assert post_pick["final_consecutive"] == 40
    assert pre_navigation["success"] is True
    assert pre_navigation["steps"] == 8
    assert pre_navigation["final_consecutive"] == 8


def retry_call(task="OpenMicrowave"):
    return AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "retry_skill",
            "atomic_task": task,
            "arguments": (
                {"object_id": "bowl"}
                if task == "PickObject"
                else {"fixture_id": "microwave"}
            ),
            "termination_condition": (
                {
                    "predicate": "holding",
                    "subject": "bowl",
                    "desired_value": True,
                }
                if task == "PickObject"
                else {
                    "predicate": "open",
                    "subject": "microwave",
                    "desired_value": True,
                }
            ),
            "policy_prompt": f"Execute {task}.",
        }
    )


def retry_args(max_retries=1):
    return SimpleNamespace(
        pi05_task_retries=max_retries,
        operation_completion_mode="expert",
        post_pick_stabilization_steps=40,
        expert_handoff_max_steps=20,
        expert_handoff_translation_tolerance_m=0.025,
        expert_handoff_orientation_tolerance_deg=12.0,
        expert_handoff_max_translation_command=0.20,
        expert_handoff_max_rotation_command=0.15,
        expert_handoff_stable_steps=8,
        expert_handoff_settle_steps=10,
        navigation_hold_confirmation_steps=8,
        held_drop_confirmation_steps=5,
    )


def install_successful_expert_handoff(monkeypatch):
    monkeypatch.setattr(
        full_episode,
        "execute_expert_handoff_with_retries",
        lambda **kwargs: {
            "success": True,
            "failure_code": None,
            "num_attempts": 1,
        },
    )


def test_vla_completion_prompt_requires_pick_clearance():
    call = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "pick_vegetable",
            "atomic_task": "PickObject",
            "arguments": {
                "object_id": "vegetable",
                "object_name": "potato",
                "source_id": "sink",
                "source_name": "sink",
            },
            "termination_condition": {
                "predicate": "holding",
                "subject": "vegetable",
                "desired_value": True,
            },
            "policy_prompt": "Pick up the potato from the sink.",
        }
    )

    prepared, changes = prepare_vla_completion_call(call)

    assert "stable carrying pose" in prepared.policy_prompt
    assert "keep the gripper closed" in prepared.policy_prompt
    conditions = prepared.termination_condition
    assert isinstance(conditions, list)
    assert {
        "predicate": "eef_outside_fixture",
        "subject": "sink",
        "margin": 0.02,
        "desired_value": True,
    } in conditions
    assert changes[0]["added_predicates"] == ["eef_outside_fixture"]


def test_vla_completion_skips_expert_handoff(monkeypatch, tmp_path):
    monkeypatch.setattr(
        full_episode,
        "execute_current_skill",
        lambda **kwargs: {
            "success": True,
            "status": "success",
            "policy_result": {"verifier_result": {"retryable": False}},
        },
    )
    monkeypatch.setattr(
        full_episode,
        "execute_expert_handoff_with_retries",
        lambda **kwargs: pytest.fail("expert handoff must not run in VLA mode"),
    )
    args = retry_args()
    args.operation_completion_mode = "vla"
    args.post_pick_stabilization_steps = 0

    result = execute_current_skill_with_retries(
        env=object(),
        client=object(),
        call=retry_call("PickObject"),
        target_id="bowl",
        expert_pose={"position": [0, 0, 0]},
        expert_handoff_pose=None,
        step_dir=tmp_path,
        step_id="ep0_pick",
        decision_maker=None,
        allow_refinement=False,
        args=args,
    )

    assert result["success"] is True
    assert "expert_handoff" not in result


def test_split_operation_retries_retryable_result_without_restoring_state(
    monkeypatch, tmp_path
):
    env = object()
    install_successful_expert_handoff(monkeypatch)
    calls = []
    results = iter(
        [
            {
                "success": False,
                "status": "uncertain",
                "policy_result": {"verifier_result": {"retryable": True}},
            },
            {
                "success": True,
                "status": "success",
                "policy_result": {"verifier_result": {"retryable": False}},
            },
        ]
    )

    def fake_execute_current_skill(**kwargs):
        calls.append(kwargs)
        return next(results)

    monkeypatch.setattr(
        full_episode,
        "execute_current_skill",
        fake_execute_current_skill,
    )
    result = execute_current_skill_with_retries(
        env=env,
        client=object(),
        call=retry_call(),
        target_id="microwave",
        expert_pose={"position": [0, 0, 0]},
        expert_handoff_pose={
            "position": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        step_dir=tmp_path,
        step_id="ep0_s0",
        decision_maker=object(),
        allow_refinement=True,
        args=retry_args(),
    )

    assert result["success"] is True
    assert result["num_attempts"] == 2
    assert [item["attempt_index"] for item in result["attempt_results"]] == [0, 1]
    assert [item["env"] for item in calls] == [env, env]
    assert [item["allow_refinement"] for item in calls] == [True, False]
    assert calls[1]["step_dir"] == tmp_path / "retry_1"


def test_pick_stabilization_failure_is_retryable(monkeypatch, tmp_path):
    install_successful_expert_handoff(monkeypatch)
    monkeypatch.setattr(
        full_episode,
        "execute_current_skill",
        lambda **kwargs: {
            "success": True,
            "status": "success",
            "policy_result": {"verifier_result": {"retryable": False}},
        },
    )
    stabilizations = iter(
        [
            {"success": False, "failure_code": "OBJECT_DROPPED"},
            {"success": True, "failure_code": None},
        ]
    )
    monkeypatch.setattr(
        full_episode,
        "stabilize_held_object",
        lambda **kwargs: next(stabilizations),
    )

    result = execute_current_skill_with_retries(
        env=object(),
        client=object(),
        call=retry_call("PickObject"),
        target_id="bowl",
        expert_pose={"position": [0, 0, 0]},
        expert_handoff_pose={
            "position": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        step_dir=tmp_path,
        step_id="ep0_pick",
        decision_maker=None,
        allow_refinement=False,
        args=retry_args(),
    )

    assert result["success"] is True
    assert result["num_attempts"] == 2
    assert result["attempt_results"][0]["failure_code"] == (
        "POST_PICK_STABILIZATION_FAILED"
    )
    assert result["post_pick_stabilization"]["success"] is True

def test_expert_handoff_nonconvergence_does_not_repeat_completed_operation(
    monkeypatch, tmp_path
):
    policy_calls = []

    def fake_policy(**kwargs):
        policy_calls.append(kwargs)
        return {
            "success": True,
            "status": "success",
            "policy_result": {"verifier_result": {"retryable": False}},
        }

    monkeypatch.setattr(full_episode, "execute_current_skill", fake_policy)
    monkeypatch.setattr(
        full_episode,
        "execute_expert_handoff_with_retries",
        lambda **kwargs: {
            "success": False,
            "failure_code": "EXPERT_EEF_POSE_NOT_CONVERGED",
            "num_attempts": 2,
        },
    )
    result = execute_current_skill_with_retries(
        env=object(),
        client=object(),
        call=retry_call(),
        target_id="microwave",
        expert_pose={"position": [0, 0, 0]},
        expert_handoff_pose={
            "position": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        step_dir=tmp_path,
        step_id="ep0_open",
        decision_maker=None,
        allow_refinement=False,
        args=retry_args(max_retries=2),
    )

    assert result["success"] is False
    assert result["failure_code"] == "EXPERT_EEF_POSE_NOT_CONVERGED"
    assert result["num_attempts"] == 1
    assert len(policy_calls) == 1


def test_pick_drop_during_expert_handoff_retries_pick(monkeypatch, tmp_path):
    policy_calls = []
    handoffs = iter(
        [
            {
                "success": False,
                "failure_code": "OBJECT_DROPPED_DURING_EXPERT_HANDOFF",
                "num_attempts": 1,
            },
            {"success": True, "failure_code": None, "num_attempts": 1},
        ]
    )

    def fake_policy(**kwargs):
        policy_calls.append(kwargs)
        return {
            "success": True,
            "status": "success",
            "policy_result": {"verifier_result": {"retryable": False}},
        }

    monkeypatch.setattr(full_episode, "execute_current_skill", fake_policy)
    monkeypatch.setattr(
        full_episode,
        "execute_expert_handoff_with_retries",
        lambda **kwargs: next(handoffs),
    )
    args = retry_args(max_retries=1)
    args.post_pick_stabilization_steps = 0
    result = execute_current_skill_with_retries(
        env=object(),
        client=object(),
        call=retry_call("PickObject"),
        target_id="bowl",
        expert_pose={"position": [0, 0, 0]},
        expert_handoff_pose={
            "position": [0, 0, 0],
            "quaternion_xyzw": [0, 0, 0, 1],
        },
        step_dir=tmp_path,
        step_id="ep0_pick",
        decision_maker=None,
        allow_refinement=False,
        args=args,
    )

    assert result["success"] is True
    assert result["num_attempts"] == 2
    assert len(policy_calls) == 2

def test_official_stage_handoff_poses_executes_terminal_pose_extraction(
    monkeypatch,
):
    states = []
    for position in ([0.40, 0.10, 0.20], [0.45, 0.12, 0.25]):
        state = np.zeros(16, dtype=np.float64)
        state[7:10] = position
        state[10:14] = [0.0, 0.0, 0.0, 1.0]
        states.append(state)

    class ObservationColumn:
        def to_numpy(self):
            return np.asarray(states, dtype=object)

    class FakeDataFrame:
        def __getitem__(self, key):
            assert key == "observation.state"
            return ObservationColumn()

    segment = SimpleNamespace(
        subtask_idx=0,
        start_frame=0,
        end_frame=1,
        subtask="pick the object",
        source_atomic_task="PickPlaceSinkToCounter",
        stage="pick",
    )
    monkeypatch.setattr(
        full_episode,
        "load_episode_dataframe",
        lambda dataset, episode_index: FakeDataFrame(),
    )
    monkeypatch.setattr(
        full_episode,
        "annotation_segments",
        lambda dataframe, task_labels: [segment],
    )

    result = full_episode.official_stage_handoff_poses(
        "dataset",
        0,
        {},
        window=2,
    )

    assert set(result) == {0}
    assert result[0]["subtask"] == "pick the object"
    assert result[0]["frame"] == "robot_base"
