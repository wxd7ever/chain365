from __future__ import annotations

from robocasa.post_skill_return import PostSkillReturnController


def _call(task: str) -> dict:
    return {
        "subgoal_id": task.lower(),
        "atomic_task": task,
        "arguments": {"object_id": "vegetable"},
        "termination_condition": {
            "predicate": "holding" if task == "PickObject" else "open",
            "subject": "vegetable" if task == "PickObject" else "microwave",
        },
        "policy_prompt": task,
    }


def _controller() -> PostSkillReturnController:
    common = {
        "target_position": [0.32, 0.0, 0.50],
        "orientation_mode": "preserve_skill_end",
        "pre_lift_height": 0.10,
        "pre_lift_minimum_z": 0.22,
    }
    return PostSkillReturnController(
        {
            "failure_policy": "block",
            "default_preset": "neutral",
            "task_presets": {
                "PickObject": "carry",
                "CloseMicrowave": "fixture_neutral",
            },
            "presets": {
                "carry": dict(common),
                "fixture_neutral": {
                    **common,
                    "target_position": [0.38, 0.0, 0.52],
                    "gripper_action": -1.0,
                },
                "neutral": dict(common, gripper_action=-1.0),
            },
        }
    )


def test_pick_moves_to_fixed_carry_pose_with_closed_gripper(monkeypatch):
    pose = {
        "position": [0.65, 0.30, 0.05],
        "quaternion_xyzw": [0.1, 0.2, 0.3, 0.9],
        "frame": "robot_base",
    }
    motions = []
    guard = object()

    monkeypatch.setattr(
        "robocasa.post_skill_return.current_eef_pose", lambda env: dict(pose)
    )
    monkeypatch.setattr(
        "robocasa.post_skill_return.resolve_object_alias",
        lambda env, call: "vegetable",
    )
    monkeypatch.setattr(
        "robocasa.post_skill_return._raw_task_env", lambda env: object()
    )
    monkeypatch.setattr(
        "robocasa.post_skill_return._holding", lambda raw, alias: True
    )
    monkeypatch.setattr(
        "robocasa.post_skill_return._object_snapshot",
        lambda raw, alias: {"alias": alias},
    )
    monkeypatch.setattr(
        "robocasa.post_skill_return.build_held_object_guard",
        lambda **kwargs: guard,
    )

    def move(env, target, **kwargs):
        motions.append((target, kwargs))
        pose.update(target)
        return {"success": True, "guard_failure": None}

    monkeypatch.setattr("robocasa.post_skill_return.move_eef_to_pose", move)
    result = _controller().prepare(
        env=object(), atomic_task_call=_call("PickObject")
    )

    assert result["success"] is True
    assert result["transport_ready"] is True
    assert result["pose_strategy"] == "task_conditioned_smooth_staged_eef_trajectory"
    assert result["smooth_trajectory"] is True
    assert motions[0][0]["position"] == [0.65, 0.30, 0.22]
    assert motions[1][0]["position"] == [0.32, 0.0, 0.50]
    assert motions[1][0]["quaternion_xyzw"] == [0.1, 0.2, 0.3, 0.9]
    assert all(kwargs["gripper_action"] == 1.0 for _, kwargs in motions)
    assert all(kwargs["guard"] is guard for _, kwargs in motions)


def test_fixture_skill_uses_its_fixed_open_gripper_pose(monkeypatch):
    pose = {
        "position": [0.70, -0.20, 0.20],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "frame": "robot_base",
    }
    motions = []
    monkeypatch.setattr(
        "robocasa.post_skill_return.current_eef_pose", lambda env: dict(pose)
    )

    def move(env, target, **kwargs):
        motions.append((target, kwargs))
        pose.update(target)
        return {"success": True, "guard_failure": None}

    monkeypatch.setattr("robocasa.post_skill_return.move_eef_to_pose", move)
    result = _controller().prepare(
        env=object(), atomic_task_call=_call("CloseMicrowave")
    )

    assert result["success"] is True
    assert result["preset"] == "fixture_neutral"
    assert result["retreat_ready"] is True
    assert motions[1][0]["position"] == [0.38, 0.0, 0.52]
    assert all(kwargs["gripper_action"] == -1.0 for _, kwargs in motions)
    assert all(kwargs["guard"] is None for _, kwargs in motions)



def test_fixed_orientation_return_translates_before_rotating(monkeypatch):
    pose = {
        "position": [0.70, -0.20, 0.20],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "frame": "robot_base",
    }
    fixed_quaternion = [0.0, 0.0, 0.70710678, 0.70710678]
    motions = []
    controller = PostSkillReturnController(
        {
            "failure_policy": "block",
            "default_preset": "released",
            "task_presets": {"CloseMicrowave": "released"},
            "presets": {
                "released": {
                    "target_position": [0.38, 0.0, 0.52],
                    "orientation_mode": "fixed",
                    "target_quaternion_xyzw": fixed_quaternion,
                    "gripper_action": -1.0,
                    "pre_lift_height": 0.08,
                    "pre_lift_minimum_z": 0.35,
                }
            },
        }
    )
    monkeypatch.setattr(
        "robocasa.post_skill_return.current_eef_pose", lambda env: dict(pose)
    )

    def move(env, target, **kwargs):
        motions.append((target, kwargs))
        pose.update(target)
        return {"success": True, "guard_failure": None, "steps": 1}

    monkeypatch.setattr("robocasa.post_skill_return.move_eef_to_pose", move)
    result = controller.prepare(
        env=object(), atomic_task_call=_call("CloseMicrowave")
    )

    assert result["success"] is True
    assert len(motions) == 3
    assert motions[1][0]["quaternion_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert motions[2][0]["quaternion_xyzw"] == fixed_quaternion
    assert motions[1][1]["settle_steps"] == 0
    assert motions[2][1]["settle_steps"] == 10
    assert result["return_orientation"] is not None
    assert result["handoff"]["trajectory_type"] == (
        "staged_translation_then_orientation"
    )



def test_pick_return_passes_actual_action_across_policy_lift_and_retract(monkeypatch):
    pose = {
        "position": [0.55, 0.10, 0.20],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "frame": "robot_base",
    }
    policy_action = [0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 1.0]
    lift_action = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 1.0]
    retract_action = [0.0] * 12
    retract_action[6] = 1.0
    retract_action[11] = 1.0
    motions = []

    monkeypatch.setattr(
        "robocasa.post_skill_return.current_eef_pose", lambda env: dict(pose)
    )
    monkeypatch.setattr(
        "robocasa.post_skill_return.resolve_object_alias",
        lambda env, call: "vegetable",
    )
    monkeypatch.setattr(
        "robocasa.post_skill_return._raw_task_env", lambda env: object()
    )
    monkeypatch.setattr(
        "robocasa.post_skill_return._holding", lambda raw, alias: True
    )
    monkeypatch.setattr(
        "robocasa.post_skill_return._object_snapshot",
        lambda raw, alias: {"alias": alias},
    )
    monkeypatch.setattr(
        "robocasa.post_skill_return.build_held_object_guard",
        lambda **kwargs: object(),
    )

    def move(env, target, **kwargs):
        motions.append(kwargs)
        pose.update(target)
        last = lift_action if len(motions) == 1 else retract_action
        return {
            "success": True,
            "guard_failure": None,
            "steps": 4,
            "last_applied_action": last,
        }

    monkeypatch.setattr("robocasa.post_skill_return.move_eef_to_pose", move)
    result = _controller().prepare(
        env=object(),
        atomic_task_call=_call("PickObject"),
        initial_applied_action=policy_action,
        simulation_step_offset=600,
    )

    assert result["success"] is True
    assert motions[0]["initial_applied_action"] == policy_action
    assert motions[0]["simulation_step_offset"] == 600
    assert motions[1]["initial_applied_action"] == lift_action
    assert motions[1]["simulation_step_offset"] == 604
    assert motions[1]["start_guard"] is False
    assert result["total_motion_steps"] == 8


def test_confirmed_drop_during_lift_skips_retract_and_transport_ready(monkeypatch):
    pose = {
        "position": [0.55, 0.10, 0.20],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "frame": "robot_base",
    }
    motions = []
    monkeypatch.setattr(
        "robocasa.post_skill_return.current_eef_pose", lambda env: dict(pose)
    )
    monkeypatch.setattr(
        "robocasa.post_skill_return.resolve_object_alias",
        lambda env, call: "vegetable",
    )
    monkeypatch.setattr(
        "robocasa.post_skill_return._raw_task_env", lambda env: object()
    )
    monkeypatch.setattr(
        "robocasa.post_skill_return._holding", lambda raw, alias: True
    )
    monkeypatch.setattr(
        "robocasa.post_skill_return._object_snapshot",
        lambda raw, alias: {"alias": alias},
    )
    monkeypatch.setattr(
        "robocasa.post_skill_return.build_held_object_guard",
        lambda **kwargs: object(),
    )

    def move(env, target, **kwargs):
        motions.append(kwargs)
        return {
            "success": False,
            "failure_code": "OBJECT_DROPPED_DURING_EXPERT_HANDOFF",
            "end_reason": "OBJECT_DROPPED",
            "guard_failure": {
                "failure_code": "OBJECT_DROPPED",
                "step_index": 603,
                "evidence": {"holding": False},
            },
            "first_drop_step": 603,
            "steps": 3,
            "last_applied_action": [0.0] * 12,
        }

    monkeypatch.setattr("robocasa.post_skill_return.move_eef_to_pose", move)
    result = _controller().prepare(
        env=object(), atomic_task_call=_call("PickObject")
    )

    assert len(motions) == 1
    assert result["success"] is False
    assert result["transport_ready"] is False
    assert result["navigation_safe"] is False
    assert result["failure_code"] == "OBJECT_DROPPED_DURING_PREP"
    assert result["pre_lift"]["first_drop_step"] == 603
