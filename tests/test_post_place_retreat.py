from __future__ import annotations

import math
from types import SimpleNamespace

from robocasa.atomic_task_schemas import AtomicTaskCall
from robocasa.post_place_retreat import PostPlaceRetreatController


def _call() -> AtomicTaskCall:
    return AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "place_bowl",
            "atomic_task": "PlaceObject",
            "arguments": {
                "object_id": "bowl",
                "destination_id": "microwave_main_group",
            },
            "termination_condition": {
                "predicate": "inside",
                "subject": "bowl",
                "object": "microwave_main_group",
                "desired_value": True,
            },
        }
    )


def _patch_observation_helpers(monkeypatch):
    raw_env = SimpleNamespace()
    pose = {
        "position": [0.60, 0.20, 0.45],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "frame": "robot_base",
    }
    monkeypatch.setattr(
        "robocasa.post_place_retreat._raw_task_env", lambda env: raw_env
    )
    monkeypatch.setattr(
        "robocasa.post_place_retreat.resolve_object_alias",
        lambda env, call: "bowl",
    )
    monkeypatch.setattr(
        "robocasa.post_place_retreat.current_eef_pose",
        lambda env: {
            "position": list(pose["position"]),
            "quaternion_xyzw": list(pose["quaternion_xyzw"]),
            "frame": "robot_base",
        },
    )
    monkeypatch.setattr(
        "robocasa.post_place_retreat.current_base_pose",
        lambda env: {"position": [1.0, 2.0, 0.7], "yaw_rad": 0.5},
    )
    monkeypatch.setattr(
        "robocasa.post_place_retreat._object_snapshot",
        lambda env, alias: {
            "alias": alias,
            "position": [1.2, 2.2, 1.0],
            "lowest_z": 0.94,
        },
    )
    return pose


def test_controller_opens_lifts_then_radially_retracts(monkeypatch):
    pose = _patch_observation_helpers(monkeypatch)
    motions = []

    def move(env, target, **kwargs):
        motions.append({"target": target, **kwargs})
        pose["position"] = list(target["position"])
        pose["quaternion_xyzw"] = list(target["quaternion_xyzw"])
        return {
            "success": True,
            "failure_code": None,
            "steps": 12,
        }

    monkeypatch.setattr(
        "robocasa.post_place_retreat.move_eef_to_pose", move
    )
    controller = PostPlaceRetreatController(
        {
            "lift_height": 0.10,
            "retract_distance": 0.12,
            "minimum_planar_radius": 0.35,
            "hold_steps": 7,
        }
    )
    result = controller.prepare(
        env=object(),
        atomic_task_call=_call(),
        episode_id=1,
        place_verified=True,
    )

    assert result["success"]
    assert result["retreat_ready"]
    assert result["fixed_safe_pose"] is False
    assert len(motions) == 2
    assert motions[0]["target"]["position"] == [0.60, 0.20, 0.55]
    initial_radius = math.hypot(0.60, 0.20)
    retract_radius = math.hypot(*motions[1]["target"]["position"][:2])
    assert abs((initial_radius - retract_radius) - 0.12) < 1e-9
    assert motions[0]["gripper_action"] == -1.0
    assert motions[1]["gripper_action"] == -1.0
    assert motions[1]["settle_steps"] == 7
    assert motions[1]["settle_mode"] == "zero_delta"


def test_controller_stops_after_failed_lift(monkeypatch):
    _patch_observation_helpers(monkeypatch)
    motions = []

    def move(*args, **kwargs):
        motions.append(kwargs)
        return {"success": False, "failure_code": None, "steps": 160}

    monkeypatch.setattr(
        "robocasa.post_place_retreat.move_eef_to_pose", move
    )
    result = PostPlaceRetreatController().prepare(
        env=object(), atomic_task_call=_call(), place_verified=True
    )
    assert len(motions) == 1
    assert result["failure_code"] == "POST_PLACE_LIFT_FAILED"
    assert not result["retreat_ready"]


def test_controller_refuses_unverified_place_without_motion(monkeypatch):
    monkeypatch.setattr(
        "robocasa.post_place_retreat.resolve_object_alias",
        lambda env, call: "bowl",
    )
    monkeypatch.setattr(
        "robocasa.post_place_retreat.move_eef_to_pose",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("motion must not start")
        ),
    )
    result = PostPlaceRetreatController().prepare(
        env=object(), atomic_task_call=_call(), place_verified=False
    )
    assert result["failure_code"] == "POST_PLACE_RETREAT_FAILED"
    assert result["reason"] == "PLACE_NOT_VERIFIED"


def test_nonconverged_motion_is_accepted_when_retreat_is_safe(monkeypatch):
    pose = _patch_observation_helpers(monkeypatch)

    def move(env, target, **kwargs):
        pose["position"] = list(target["position"])
        return {
            "success": False,
            "failure_code": "FINAL_POSE_NOT_REACHED",
            "steps": kwargs["max_steps"],
        }

    monkeypatch.setattr(
        "robocasa.post_place_retreat.move_eef_to_pose", move
    )
    result = PostPlaceRetreatController().prepare(
        env=object(), atomic_task_call=_call(), place_verified=True
    )

    assert result["success"]
    assert result["retreat_ready"]
    assert result["navigation_safe"]
    assert result["failure_code"] is None
    assert result["reason"] == (
        "POST_PLACE_RETREAT_READY_WITH_MOTION_WARNING"
    )
    assert len(result["motion_warnings"]) == 2


def test_unresolved_object_alias_does_not_block_arm_retreat(monkeypatch):
    pose = _patch_observation_helpers(monkeypatch)
    monkeypatch.setattr(
        "robocasa.post_place_retreat.resolve_object_alias",
        lambda env, call: None,
    )

    def move(env, target, **kwargs):
        pose["position"] = list(target["position"])
        return {"success": True, "failure_code": None, "steps": 12}

    monkeypatch.setattr(
        "robocasa.post_place_retreat.move_eef_to_pose", move
    )
    result = PostPlaceRetreatController().prepare(
        env=object(), atomic_task_call=_call(), place_verified=True
    )

    assert result["success"]
    assert result["retreat_ready"]
    assert result["object_alias"] is None
    assert result["before"]["object_pose"]["available"] is False
