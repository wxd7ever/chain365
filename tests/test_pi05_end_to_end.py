import numpy as np

from robocasa.pi05_rollout import execute_pi05_manipulation_policy


def _observation():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    return {
        "robot0_agentview_left_image": image,
        "robot0_agentview_right_image": image,
        "robot0_eye_in_hand_image": image,
        "robot0_base_to_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_base_to_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_base_pos": np.zeros(3, dtype=np.float32),
        "robot0_base_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
    }


class _Client:
    def reset(self):
        return None

    def infer(self, observation):
        assert observation["prompt"] == "Complete the whole task."
        return {"actions": np.zeros((5, 12), dtype=np.float32)}


class _SparseRewardEnv:
    action_dimension = 12

    def __init__(self, success_step):
        self.success_step = success_step
        self.steps = 0
        self.observation = _observation()

    @property
    def unwrapped(self):
        return self

    def get_observation(self):
        return self.observation

    def step(self, action):
        assert np.asarray(action).shape == (12,)
        self.steps += 1
        reward = float(self.steps >= self.success_step)
        return self.observation, reward, False, {}


def test_direct_pi05_stops_on_native_robocasa_sparse_reward(tmp_path):
    env = _SparseRewardEnv(success_step=3)

    success, logs = execute_pi05_manipulation_policy(
        env=env,
        client=_Client(),
        log_dir=str(tmp_path),
        episode_id=0,
        horizon=20,
        replan_steps=5,
        render=False,
        terminate_on_success=True,
        prompt_override="Complete the whole task.",
    )

    assert success == 1.0
    assert env.steps == 3
    assert logs["Horizon"] == 3
    assert logs["First_Success_Step"] == 3
    assert logs["Success_Source"] == "robocasa_sparse_reward"


def test_direct_pi05_can_continue_after_success_when_requested(tmp_path):
    env = _SparseRewardEnv(success_step=2)

    success, logs = execute_pi05_manipulation_policy(
        env=env,
        client=_Client(),
        log_dir=str(tmp_path),
        episode_id=0,
        horizon=6,
        replan_steps=5,
        render=False,
        terminate_on_success=False,
        prompt_override="Complete the whole task.",
    )

    assert success == 1.0
    assert env.steps == 6
    assert logs["First_Success_Step"] == 2
