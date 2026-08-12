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

    def __call__(self, **kwargs):
        self.calls += 1
        success = self.calls >= 2
        return {
            "status": "success" if success else "uncertain",
            "goal_satisfied": success,
            "failure_code": None,
            "retryable": not success,
            "state_evidence": [{"step": kwargs["step_index"]}],
        }


def test_atomic_rollout_reuses_env_client_and_stops_on_verifier(tmp_path):
    env = FakeEnv()
    client = FakeClient()
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
        verifier=FakeVerifier(),
        log_dir=tmp_path,
        episode_id=0,
        horizon=20,
        replan_steps=5,
        verify_interval=2,
        min_steps_before_verify=0,
        render=False,
        base_action_mode="residual",
        base_residual_limit=0.15,
    )
    assert success is True
    assert env.reset_calls == 0
    assert client.reset_calls == 1
    assert client.payloads[0]["prompt"] == "Open the microwave door."
    assert len(env.actions) == 4
    assert np.allclose(env.actions[0][7:10], 0.15)
    assert logs["Atomic_Task_Call"]["subgoal_id"] == "g1"
    assert logs["Final_Verification"]["status"] == "success"
    assert logs["Prompt"] == "Open the microwave door."
    assert logs["Held_Object_Guard"]["enabled"] is False


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
    )

    assert success is True
    assert len(env.actions) == 7
    assert logs["Configured_Pre_Success_Horizon"] == 4
    assert logs["First_Success_Step"] == 4
    assert logs["Handoff_Target_Step"] == 7
    assert logs["Post_Success_Steps_Executed"] == 3
    assert logs["Handoff_Completed"] is True
    assert logs["Final_Verification"]["status"] == "success"


def test_guard_failure_stops_rollout_and_returns_retryable_drop(monkeypatch, tmp_path):
    class DroppingGuard:
        def start(self):
            pass

        def apply_action(self, action, *, step_index):
            return action

        def observe(self, *, step_index):
            if step_index < 2:
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
        verifier=FakeVerifier(),
        log_dir=tmp_path,
        episode_id=0,
        horizon=20,
        replan_steps=5,
        verify_interval=5,
        min_steps_before_verify=0,
        render=False,
    )

    assert success is False
    assert len(env.actions) == 2
    assert logs["Horizon"] == 2
    assert logs["Final_Verification"]["failure_code"] == "OBJECT_DROPPED"
    assert logs["Final_Verification"]["retryable"] is True
    assert logs["Held_Object_Guard"]["dropped_step"] == 2
