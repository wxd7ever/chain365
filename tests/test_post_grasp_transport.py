from __future__ import annotations

import math
from types import SimpleNamespace

from robocasa.atomic_task_schemas import AtomicTaskCall
from robocasa.post_grasp_transport import (
    PostGraspTransportController,
    resolve_object_alias,
)


def _call() -> AtomicTaskCall:
    return AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "pick_bowl",
            "atomic_task": "PickObject",
            "arguments": {"object_id": "bowl"},
            "termination_condition": {
                "predicate": "holding",
                "subject": "bowl",
                "desired_value": True,
            },
        }
    )


def _alias_env(*, natural_name: str = "bell pepper"):
    return SimpleNamespace(
        fixtures={},
        objects={"vegetable": SimpleNamespace(name="vegetable")},
        sim=object(),
        get_obj_lang=lambda alias: natural_name,
    )


def test_resolve_object_alias_prefers_subject_from_list_condition():
    call = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "pick_vegetable_from_sink",
            "atomic_task": "PickObject",
            "arguments": {
                "object_name": "bell pepper",
                "source_name": "sink",
            },
            "termination_condition": [
                {
                    "predicate": "holding",
                    "subject": "vegetable",
                    "desired_value": True,
                }
            ],
        }
    )

    assert resolve_object_alias(_alias_env(), call) == "vegetable"


def test_resolve_object_alias_uses_environment_natural_name():
    call = AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "pick_pepper",
            "atomic_task": "PickObject",
            "arguments": {"object_name": "bell pepper"},
            "termination_condition": {
                "predicate": "holding",
                "subject": "generic_target",
                "desired_value": True,
            },
        }
    )

    assert resolve_object_alias(_alias_env(), call) == "vegetable"


def _patch_observation_helpers(monkeypatch):
    raw_env = SimpleNamespace()
    pose = {
        "position": [0.60, 0.20, 0.50],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "frame": "robot_base",
    }
    monkeypatch.setattr(
        "robocasa.post_grasp_transport._raw_task_env",
        lambda env: raw_env,
    )
    monkeypatch.setattr(
        "robocasa.post_grasp_transport.resolve_object_alias",
        lambda env, call: "bowl",
    )
    monkeypatch.setattr(
        "robocasa.post_grasp_transport.current_eef_pose",
        lambda env: {
            "position": list(pose["position"]),
            "quaternion_xyzw": list(pose["quaternion_xyzw"]),
            "frame": "robot_base",
        },
    )
    monkeypatch.setattr(
        "robocasa.post_grasp_transport.current_base_pose",
        lambda env: {"position": [1.0, 2.0, 0.7], "yaw_rad": 0.5},
    )
    monkeypatch.setattr(
        "robocasa.post_grasp_transport._object_snapshot",
        lambda env, alias: {
            "alias": alias,
            "position": [0.7, 0.2, pose["position"][2] - 0.1],
            "lowest_z": pose["position"][2] - 0.15,
        },
    )
    monkeypatch.setattr(
        "robocasa.post_grasp_transport._holding",
        lambda env, alias: True,
    )
    guard = SimpleNamespace(name="guard")
    monkeypatch.setattr(
        "robocasa.post_grasp_transport.build_held_object_guard",
        lambda **kwargs: guard,
    )
    return pose, guard


def test_controller_lifts_then_radially_retracts_and_holds(monkeypatch):
    pose, guard = _patch_observation_helpers(monkeypatch)
    motions = []

    def move(env, target, **kwargs):
        motions.append({"target": target, **kwargs})
        pose["position"] = list(target["position"])
        pose["quaternion_xyzw"] = list(target["quaternion_xyzw"])
        return {
            "success": True,
            "failure_code": None,
            "guard_failure": None,
            "steps": 12,
        }

    monkeypatch.setattr(
        "robocasa.post_grasp_transport.move_eef_to_pose", move
    )
    controller = PostGraspTransportController(
        {
            "lift_height": 0.10,
            "retract_distance": 0.08,
            "minimum_planar_radius": 0.35,
            "hold_steps": 7,
        }
    )
    result = controller.prepare(
        env=object(), atomic_task_call=_call(), episode_id=1
    )

    assert result["success"]
    assert result["transport_ready"]
    assert result["fixed_carry_pose"] is False
    assert len(motions) == 2
    assert motions[0]["target"]["position"] == [0.60, 0.20, 0.60]
    initial_radius = math.hypot(0.60, 0.20)
    retract_radius = math.hypot(*motions[1]["target"]["position"][:2])
    assert abs((initial_radius - retract_radius) - 0.08) < 1e-9
    assert motions[0]["guard"] is guard
    assert motions[1]["guard"] is guard
    assert motions[0]["gripper_action"] == 1.0
    assert motions[1]["gripper_action"] == 1.0
    assert motions[1]["settle_steps"] == 7
    assert motions[1]["settle_mode"] == "zero_delta"


def test_controller_reports_drop_during_lift(monkeypatch):
    _, _ = _patch_observation_helpers(monkeypatch)
    monkeypatch.setattr(
        "robocasa.post_grasp_transport.move_eef_to_pose",
        lambda *args, **kwargs: {
            "success": False,
            "failure_code": "OBJECT_DROPPED_DURING_EXPERT_HANDOFF",
            "guard_failure": {"failure_code": "OBJECT_DROPPED"},
            "steps": 2,
        },
    )
    result = PostGraspTransportController().prepare(
        env=object(), atomic_task_call=_call()
    )
    assert not result["success"]
    assert not result["transport_ready"]
    assert result["failure_code"] == "OBJECT_DROPPED_DURING_PREP"
    assert result["reason"] == "OBJECT_DROPPED_DURING_LIFT"
    assert "retract" not in result


def test_controller_refuses_unheld_object_without_motion(monkeypatch):
    _patch_observation_helpers(monkeypatch)
    monkeypatch.setattr(
        "robocasa.post_grasp_transport._holding",
        lambda env, alias: False,
    )
    monkeypatch.setattr(
        "robocasa.post_grasp_transport.move_eef_to_pose",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("motion must not start")
        ),
    )
    result = PostGraspTransportController().prepare(
        env=object(), atomic_task_call=_call()
    )
    assert result["failure_code"] == "OBJECT_DROPPED_DURING_PREP"
    assert result["reason"] == "OBJECT_DROPPED_BEFORE_LIFT"


def test_nonconverged_motion_is_accepted_when_transport_is_safe(monkeypatch):
    pose, _ = _patch_observation_helpers(monkeypatch)

    def move(env, target, **kwargs):
        pose["position"] = list(target["position"])
        return {
            "success": False,
            "failure_code": "FINAL_POSE_NOT_REACHED",
            "guard_failure": None,
            "steps": kwargs["max_steps"],
        }

    monkeypatch.setattr(
        "robocasa.post_grasp_transport.move_eef_to_pose", move
    )
    result = PostGraspTransportController().prepare(
        env=object(), atomic_task_call=_call()
    )

    assert result["success"]
    assert result["transport_ready"]
    assert result["failure_code"] is None
    assert result["reason"] == "TRANSPORT_READY_WITH_MOTION_WARNING"
    assert result["transport_ready_check"]["clearance"]["safe"]
    assert result["transport_ready_check"]["compactness"]["safe"]
    assert len(result["motion_warnings"]) == 2


def test_transport_rejects_arm_that_remains_too_extended(monkeypatch):
    pose, _ = _patch_observation_helpers(monkeypatch)

    def move(env, target, **kwargs):
        pose["position"] = list(target["position"])
        return {
            "success": True,
            "failure_code": None,
            "guard_failure": None,
            "steps": 12,
        }

    monkeypatch.setattr(
        "robocasa.post_grasp_transport.move_eef_to_pose", move
    )
    result = PostGraspTransportController(
        {"maximum_transport_planar_radius": 0.45}
    ).prepare(env=object(), atomic_task_call=_call())

    assert not result["success"]
    assert not result["transport_ready"]
    assert result["failure_code"] == "TRANSPORT_READY_FAILED"
    assert result["reason"] == "FINAL_ARM_NOT_COMPACT"
