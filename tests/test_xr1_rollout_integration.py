from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from pi05_rollout import execute_pi05_atomic_task_policy  # noqa: E402


def observation():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    return {
        "robot0_agentview_left_image": image.copy(),
        "robot0_agentview_right_image": image.copy(),
        "robot0_eye_in_hand_image": image.copy(),
        "robot0_base_to_eef_pos": np.zeros(3),
        "robot0_base_to_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
        "robot0_base_pos": np.zeros(3),
        "robot0_base_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    }


class Env:
    action_dimension = 12

    def __init__(self):
        self.actions = []

    def get_observation(self):
        return observation()

    def step(self, action):
        assert isinstance(action, dict)
        self.actions.append(action)
        return observation(), 0.0, False, {}


class TemporalClient:
    manages_temporal_observations = True
    requires_robocasa_action_conversion = True
    policy_backend = "xr1"

    def __init__(self):
        self.episode_initialized = False
        self.reset_calls = []
        self.instructions = []
        self.observations = []
        self.cancel_calls = 0

    def reset(self, obs):
        self.reset_calls.append(obs)
        self.episode_initialized = True

    def set_instruction(self, instruction):
        self.instructions.append(instruction)

    def infer(self, payload):
        assert payload == {}
        return {"actions": np.zeros((4, 12), dtype=np.float32)}

    def observe(self, obs):
        self.observations.append(obs)

    def cancel_current_chunk(self):
        self.cancel_calls += 1

    def diagnostics(self):
        return {"history_size": len(self.observations) + 1}


class Verifier:
    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        success = kwargs["step_index"] == 1
        return {
            "status": "success" if success else "uncertain",
            "goal_satisfied": success,
            "failure_code": None,
            "retryable": not success,
            "state_evidence": [],
        }


def test_xr1_rollout_observes_once_and_cancels_remaining_chunk(tmp_path):
    env = Env()
    client = TemporalClient()
    success, logs = execute_pi05_atomic_task_policy(
        env=env,
        client=client,
        atomic_task_call={
            "subgoal_id": "open",
            "atomic_task": "OpenMicrowave",
            "policy_prompt": "Open the microwave door.",
            "arguments": {"fixture_id": "microwave"},
            "termination_condition": {
                "predicate": "open",
                "subject": "microwave",
            },
        },
        verifier=Verifier(),
        log_dir=tmp_path,
        episode_id=0,
        horizon=5,
        replan_steps=4,
        verify_interval=1,
        min_steps_before_verify=0,
        verification_mode="periodic",
        held_object_guard=False,
        render=False,
    )
    assert success is True
    assert len(client.reset_calls) == 1
    assert client.instructions == ["Open the microwave door."]
    assert len(client.observations) == 1
    assert client.cancel_calls == 1
    assert logs["Queued_Actions_Discarded_At_Skill_End"] == 3
    assert logs["Policy_Backend"] == "xr1"
    assert len(env.actions) == 1
