from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from pi05_rollout import execute_pi05_atomic_task_policy  # noqa: E402
import held_object_guard  # noqa: E402


def observation():
    return {
        "robot0_agentview_left_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_agentview_right_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "robot0_base_to_eef_pos": np.zeros(3),
        "robot0_base_to_eef_quat": np.zeros(4),
        "robot0_base_pos": np.zeros(3),
        "robot0_base_quat": np.zeros(4),
        "robot0_gripper_qpos": np.zeros(2),
    }


class FakeEnv:
    action_dimension = 12

    def __init__(self):
        self.actions = []
        self.reset_calls = 0

    def get_observation(self):
        return observation()

    def reset(self):
        self.reset_calls += 1
        raise AssertionError("atomic rollout must not reset the environment")

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return observation(), 1.0, False, {}


class FakeClient:
    def __init__(self):
        self.payloads = []
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def infer(self, payload):
        self.payloads.append(payload)
        actions = np.zeros((10, 12), dtype=np.float32)
        actions[:, 7:10] = 0.5
        return {"actions": actions}


class FakeVerifier:
    def __init__(self):
        self.calls = 0
        self.steps = []
        self.phases = []

    def __call__(self, **kwargs):
        self.calls += 1
        self.steps.append(kwargs["step_index"])
        self.phases.append(kwargs["info"].get("verification_phase"))
        success = self.calls >= 2
        return {
            "status": "success" if success else "uncertain",
            "goal_satisfied": success,
            "failure_code": None,
            "retryable": not success,
            "state_evidence": [{"step": kwargs["step_index"]}],
        }


class SimplePeriodicCritic:
    critic_only = True

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        step_index = kwargs["step_index"]
        self.calls.append(
            (step_index, kwargs["info"].get("verification_phase"))
        )
        success = step_index >= 120
        return {
            "status": "success" if success else "uncertain",
            "goal_satisfied": success,
            "failure_code": None if success else "VLM_CRITIC_NOT_COMPLETE",
            "retryable": not success,
            "state_evidence": [{"step": step_index}],
        }


def test_simple_periodic_critic_checks_every_60_steps_without_step_zero(
    tmp_path,
):
    env = FakeEnv()
    verifier = SimplePeriodicCritic()
    success, logs = execute_pi05_atomic_task_policy(
        env=env,
        client=FakeClient(),
        atomic_task_call={
            "subgoal_id": "pick_object",
            "atomic_task": "PickObject",
            "policy_prompt": "Pick up the vegetable and keep holding it.",
            "arguments": {"object_id": "vegetable"},
            "termination_condition": {
                "predicate": "holding",
                "subject": "vegetable",
                "desired_value": True,
            },
        },
        verifier=verifier,
        log_dir=tmp_path,
        episode_id=0,
        horizon=300,
        replan_steps=5,
        verify_interval=60,
        min_steps_before_verify=60,
        render=False,
        held_object_guard=False,
        success_handoff_steps=30,
        verification_mode="periodic",
    )

    assert success is True
    assert len(env.actions) == 120
    assert verifier.calls == [(60, "periodic"), (120, "periodic")]
    assert logs["Initial_Verification"] is None
    assert logs["Simple_Periodic_Critic"] is True
    assert logs["Success_Handoff_Steps"] == 30
    assert logs["Effective_Success_Handoff_Steps"] == 0
    assert logs["First_Success_Step"] == 120
    assert logs["Handoff_Target_Step"] == 120


def test_atomic_rollout_reuses_env_client_and_stops_on_verifier(tmp_path):
    env = FakeEnv()
    client = FakeClient()
    verifier = FakeVerifier()
    success, logs = execute_pi05_atomic_task_policy(
        env=env,
        client=client,
        atomic_task_call={
            "subgoal_id": "g1",
            "atomic_task": "OpenMicrowave",
            "policy_prompt": "Open the microwave door.",
            "arguments": {"fixture_id": "microwave_1"},
            "termination_condition": {"predicate": "open", "subject": "microwave_1"},
        },
        verifier=verifier,
        log_dir=tmp_path,
        episode_id=0,
        horizon=20,
        replan_steps=5,
        verify_interval=2,
        min_steps_before_verify=0,
        render=False,
        base_action_mode="residual",
        base_residual_limit=0.15,
        verification_mode="periodic",
    )
    assert success is True
    assert env.reset_calls == 0
    assert client.reset_calls == 1
    assert client.payloads[0]["prompt"] == "Open the microwave door."
    assert len(env.actions) == 2
    assert np.allclose(env.actions[0][7:10], 0.15)
    assert logs["Atomic_Task_Call"]["subgoal_id"] == "g1"
    assert logs["Final_Verification"]["status"] == "success"
    assert logs["Initial_Verification"]["status"] == "uncertain"
    assert verifier.steps == [0, 2]
    assert verifier.phases == ["task_start", "periodic"]
    assert logs["Prompt"] == "Open the microwave door."
    assert logs["Held_Object_Guard"]["enabled"] is False


def test_periodic_early_stop_requires_two_interval_separated_successes(tmp_path):
    class ImmediateVerifier:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(
                (kwargs["step_index"], kwargs["info"]["verification_phase"])
            )
            return {
                "status": "success",
                "goal_satisfied": True,
                "failure_code": None,
                "retryable": False,
                "state_evidence": [],
            }

    env = FakeEnv()
    verifier = ImmediateVerifier()
    success, logs = execute_pi05_atomic_task_policy(
        env=env,
        client=FakeClient(),
        atomic_task_call={
            "subgoal_id": "pick",
            "atomic_task": "PickObject",
            "policy_prompt": "Pick up the vegetable.",
            "arguments": {"object_id": "vegetable"},
            "termination_condition": {
                "predicate": "holding",
                "subject": "vegetable",
            },
        },
        verifier=verifier,
        log_dir=tmp_path,
        episode_id=0,
        horizon=300,
        verify_interval=60,
        min_steps_before_verify=60,
        success_handoff_steps=30,
        verification_mode="periodic_early_stop",
        held_object_guard=False,
        render=False,
    )

    assert success is True
    assert len(env.actions) == 120
    assert verifier.calls == [(60, "periodic"), (120, "periodic")]
    assert logs["Initial_Verification"] is None
    assert logs["Effective_Success_Handoff_Steps"] == 0
    assert logs["Immediate_Periodic_Early_Stop"] is False
    assert logs["First_Success_Step"] == 60
    assert logs["Handoff_Target_Step"] == 120
    assert logs["Periodic_Early_Stop_Required_Consecutive_Successes"] == 2
    assert logs["Periodic_Early_Stop_Consecutive_Successes"] == 2
    assert logs["Periodic_Early_Stop_Success_Steps"] == [60, 120]
    stability = logs["Final_Verification"]["state_evidence"][-1]
    assert stability["source"] == "rollout_consecutive_stability"
    assert stability["verification_interval_steps"] == 60
    assert stability["stable"] is True


def test_periodic_early_stop_resets_stability_after_unsuccessful_check(tmp_path):
    class SequenceVerifier:
        def __init__(self):
            self.calls = []
            self.results = iter((True, False, True, True))

        def __call__(self, **kwargs):
            self.calls.append(kwargs["step_index"])
            success = next(self.results)
            return {
                "status": "success" if success else "uncertain",
                "goal_satisfied": success,
                "failure_code": None,
                "retryable": not success,
                "state_evidence": [],
            }

    env = FakeEnv()
    verifier = SequenceVerifier()
    success, logs = execute_pi05_atomic_task_policy(
        env=env,
        client=FakeClient(),
        atomic_task_call={
            "subgoal_id": "pick",
            "atomic_task": "PickObject",
            "policy_prompt": "Pick up the vegetable.",
            "arguments": {"object_id": "vegetable"},
            "termination_condition": {
                "predicate": "holding",
                "subject": "vegetable",
            },
        },
        verifier=verifier,
        log_dir=tmp_path,
        episode_id=0,
        horizon=300,
        verify_interval=60,
        min_steps_before_verify=60,
        verification_mode="periodic_early_stop",
        held_object_guard=False,
        render=False,
    )

    assert success is True
    assert len(env.actions) == 240
    assert verifier.calls == [60, 120, 180, 240]
    assert logs["First_Success_Step"] == 180
    assert logs["Periodic_Early_Stop_Success_Steps"] == [180, 240]


def test_periodic_early_stop_accepts_first_success_exactly_at_horizon(tmp_path):
    class HorizonOnlyVerifier:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            step = kwargs["step_index"]
            self.calls.append(step)
            success = step == 300
            return {
                "status": "success" if success else "uncertain",
                "goal_satisfied": success,
                "failure_code": None,
                "retryable": not success,
                "state_evidence": [],
            }

    env = FakeEnv()
    verifier = HorizonOnlyVerifier()
    success, logs = execute_pi05_atomic_task_policy(
        env=env,
        client=FakeClient(),
        atomic_task_call={
            "subgoal_id": "place",
            "atomic_task": "PlaceObject",
            "policy_prompt": "Place the vegetable in the bowl.",
            "arguments": {
                "object_id": "vegetable",
                "destination_id": "bowl",
            },
            "termination_condition": {
                "predicate": "inside",
                "subject": "vegetable",
                "object": "bowl",
            },
        },
        verifier=verifier,
        log_dir=tmp_path,
        episode_id=0,
        horizon=300,
        verify_interval=60,
        min_steps_before_verify=60,
        verification_mode="periodic_early_stop",
        held_object_guard=False,
        render=False,
    )

    assert success is True
    assert len(env.actions) == 300
    assert verifier.calls == [60, 120, 180, 240, 300]
    assert logs["First_Success_Step"] == 300
    assert logs["Handoff_Target_Step"] == 300
    assert logs["Periodic_Early_Stop_Consecutive_Successes"] == 1
    assert logs["Periodic_Early_Stop_Horizon_Single_Success_Accepted"] is True
    stability = logs["Final_Verification"]["state_evidence"][-1]
    assert stability["consecutive_stable"] is False
    assert stability["horizon_single_success_accepted"] is True
    assert stability["stable"] is True


def test_navigation_can_freeze_manipulator_without_clipping_base(tmp_path):
    class WholeBodyClient(FakeClient):
        def infer(self, payload):
            self.payloads.append(payload)
            return {"actions": np.ones((10, 12), dtype=np.float32)}

    env = FakeEnv()
    success, logs = execute_pi05_atomic_task_policy(
        env=env,
        client=WholeBodyClient(),
        atomic_task_call={
            "subgoal_id": "navigate",
            "atomic_task": "NavigateKitchen",
            "policy_prompt": "Navigate to the sink.",
            "arguments": {"fixture_id": "sink_1"},
            "termination_condition": {
                "predicate": "navigation_pose",
                "subject": "sink_1",
            },
        },
        verifier=FakeVerifier(),
        log_dir=tmp_path,
        episode_id=0,
        horizon=20,
        replan_steps=5,
        verify_interval=2,
        min_steps_before_verify=0,
        render=False,
        base_action_mode="full",
        freeze_manipulator=True,
        verification_mode="periodic",
    )

    assert success is True
    assert np.allclose(env.actions[0][:7], 0)
    assert np.allclose(env.actions[0][7:10], 1)
    assert env.actions[0][10] == 0
    assert env.actions[0][11] == 1
    assert logs["Manipulator_Frozen"] is True


def test_success_handoff_continues_policy_with_extra_budget(tmp_path):
    env = FakeEnv()
    success, logs = execute_pi05_atomic_task_policy(
        env=env,
        client=FakeClient(),
        atomic_task_call={
            "subgoal_id": "g1",
            "atomic_task": "OpenMicrowave",
            "policy_prompt": "Open the microwave door and retract.",
            "arguments": {"fixture_id": "microwave_1"},
            "termination_condition": {
                "predicate": "open",
                "subject": "microwave_1",
            },
        },
        verifier=FakeVerifier(),
        log_dir=tmp_path,
        episode_id=0,
        horizon=4,
        replan_steps=5,
        verify_interval=2,
        min_steps_before_verify=0,
        render=False,
        success_handoff_steps=3,
        verification_mode="periodic",
    )

    assert success is True
    assert len(env.actions) == 5
    assert logs["Configured_Pre_Success_Horizon"] == 4
    assert logs["First_Success_Step"] == 2
    assert logs["Handoff_Target_Step"] == 5
    assert logs["Post_Success_Steps_Executed"] == 3
    assert logs["Handoff_Completed"] is True
    assert logs["Final_Verification"]["status"] == "success"


def test_task_start_success_skips_policy_and_is_also_final_verification(tmp_path):
    class AlreadySatisfiedVerifier:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(
                (
                    kwargs["step_index"],
                    kwargs["info"].get("verification_phase"),
                )
            )
            return {
                "status": "success",
                "goal_satisfied": True,
                "failure_code": None,
                "retryable": False,
                "state_evidence": [],
            }

    env = FakeEnv()
    client = FakeClient()
    verifier = AlreadySatisfiedVerifier()
    success, logs = execute_pi05_atomic_task_policy(
        env=env,
        client=client,
        atomic_task_call={
            "subgoal_id": "already_open",
            "atomic_task": "OpenMicrowave",
            "policy_prompt": "Open the microwave door.",
            "arguments": {"fixture_id": "microwave_1"},
            "termination_condition": {
                "predicate": "open",
                "subject": "microwave_1",
            },
        },
        verifier=verifier,
        log_dir=tmp_path,
        episode_id=0,
        horizon=20,
        render=False,
        verification_mode="periodic",
    )

    assert success is True
    assert env.actions == []
    assert client.payloads == []
    assert verifier.calls == [(0, "task_start")]
    assert logs["Initial_Verification"] == logs["Final_Verification"]
    assert logs["First_Success_Step"] == 0


def test_guard_failure_does_not_stop_skill_end_rollout(monkeypatch, tmp_path):
    class DroppingGuard:
        def start(self):
            pass

        def apply_action(self, action, *, step_index):
            return action

        def observe(self, *, step_index):
            if step_index != 2:
                return None
            return {
                "status": "failed",
                "goal_satisfied": False,
                "failure_code": "OBJECT_DROPPED",
                "retryable": True,
                "state_evidence": [{"step_index": step_index}],
            }

        def to_dict(self):
            return {"enabled": True, "dropped_step": 2}

    monkeypatch.setattr(
        held_object_guard,
        "build_held_object_guard",
        lambda **kwargs: DroppingGuard(),
    )
    env = FakeEnv()
    success, logs = execute_pi05_atomic_task_policy(
        env=env,
        client=FakeClient(),
        atomic_task_call={
            "subgoal_id": "g1",
            "atomic_task": "OpenMicrowave",
            "policy_prompt": "Open the microwave door.",
            "arguments": {"fixture_id": "microwave_1"},
            "termination_condition": {"predicate": "open", "subject": "microwave_1"},
        },
        verifier=SkillEndVerifier(success=True),
        log_dir=tmp_path,
        episode_id=0,
        horizon=6,
        replan_steps=5,
        verify_interval=5,
        min_steps_before_verify=0,
        render=False,
        verification_mode="skill_end",
    )

    assert success is True
    assert len(env.actions) == 6
    assert logs["Horizon"] == 6
    assert logs["Final_Verification"]["status"] == "success"
    assert len(logs["Verification_History"]) == 1
    assert len(logs["Guard_Failures"]) == 1
    assert logs["First_Guard_Failure"]["failure_code"] == "OBJECT_DROPPED"
    assert logs["Held_Object_Guard"]["dropped_step"] == 2


class SkillEndVerifier:
    def __init__(self, *, success):
        self.success = success
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "status": "success" if self.success else "failed",
            "goal_satisfied": self.success,
            "failure_code": None if self.success else "VLM_CRITIC_FAILURE",
            "retryable": not self.success,
            "state_evidence": [{"step": kwargs["step_index"]}],
        }


def test_skill_end_runs_full_horizon_and_verifies_exactly_once(tmp_path):
    env = FakeEnv()
    verifier = SkillEndVerifier(success=True)
    success, logs = execute_pi05_atomic_task_policy(
        env=env,
        client=FakeClient(),
        atomic_task_call={
            "subgoal_id": "pick_object",
            "atomic_task": "PickObject",
            "policy_prompt": "Pick up the vegetable and keep holding it.",
            "arguments": {"object_id": "vegetable"},
            "termination_condition": {
                "predicate": "holding",
                "subject": "vegetable",
                "desired_value": True,
            },
        },
        verifier=verifier,
        log_dir=tmp_path,
        episode_id=0,
        horizon=600,
        replan_steps=5,
        verify_interval=1,
        min_steps_before_verify=0,
        render=False,
        verification_mode="skill_end",
    )

    assert success is True
    assert len(env.actions) == 600
    assert len(verifier.calls) == 1
    final_call = verifier.calls[0]
    assert final_call["step_index"] == 600
    assert final_call["info"]["verification_phase"] == "skill_end"
    assert final_call["info"]["verification_mode"] == "skill_end"
    assert final_call["info"]["verification_step"] == 600
    assert "verification_observation_buffer" not in final_call["info"]
    assert logs["Initial_Verification"] is None
    assert len(logs["Verification_History"]) == 1
    assert logs["verification_mode"] == "skill_end"
    assert logs["verification_step"] == 600


def test_skill_end_failure_remains_retryable_after_full_horizon(tmp_path):
    env = FakeEnv()
    verifier = SkillEndVerifier(success=False)
    success, logs = execute_pi05_atomic_task_policy(
        env=env,
        client=FakeClient(),
        atomic_task_call={
            "subgoal_id": "pick_object",
            "atomic_task": "PickObject",
            "policy_prompt": "Pick up the vegetable and keep holding it.",
            "arguments": {"object_id": "vegetable"},
            "termination_condition": {
                "predicate": "holding",
                "subject": "vegetable",
                "desired_value": True,
            },
        },
        verifier=verifier,
        log_dir=tmp_path,
        episode_id=0,
        horizon=6,
        replan_steps=5,
        render=False,
        verification_mode="skill_end",
    )

    assert success is False
    assert len(env.actions) == 6
    assert len(verifier.calls) == 1
    assert logs["Final_Verification"]["failure_code"] == "VLM_CRITIC_FAILURE"
    assert logs["Final_Verification"]["retryable"] is True
    assert logs["Terminal_Failure"] is False
