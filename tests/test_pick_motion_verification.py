from __future__ import annotations

import numpy as np

from robocasa.atomic_task_schemas import AtomicTaskCall
from robocasa.pick_motion_verification import (
    capture_pick_motion_baseline,
    evaluate_pick_motion,
)


class _Entity:
    def __init__(self, name: str, *, horizontal_radius: float = 0.2):
        self.name = name
        self.nat_lang = name
        self.horizontal_radius = horizontal_radius


class _PickEnv:
    def __init__(
        self,
        *,
        object_positions: dict[str, tuple[float, float, float]],
        fixture_names: tuple[str, ...] = (),
        contacts: set[frozenset[str]] | None = None,
    ):
        self.objects = {
            alias: _Entity(alias, horizontal_radius=0.2 if "bowl" in alias else 0.06)
            for alias in object_positions
        }
        self.fixtures = {alias: _Entity(alias) for alias in fixture_names}
        self.obj_body_id = {
            alias: index for index, alias in enumerate(object_positions)
        }
        body_xpos = np.asarray(list(object_positions.values()), dtype=float)
        self.sim = type(
            "PickSim",
            (),
            {"data": type("PickData", (), {"body_xpos": body_xpos})()},
        )()
        self.contacts = set(contacts or set())
        self._aliases = {
            id(entity): alias
            for alias, entity in {**self.objects, **self.fixtures}.items()
        }
        self.gripper = object()
        self.robots = [
            type("PickRobot", (), {"gripper": {"right": self.gripper}})()
        ]

    def get_fixture(self, fixture):
        if isinstance(fixture, str):
            return self.fixtures[fixture]
        return fixture

    def check_contact(self, first, second):
        first_alias = self._aliases.get(id(first))
        second_alias = self._aliases.get(id(second))
        if first_alias is None or second_alias is None:
            return False
        return frozenset((first_alias, second_alias)) in self.contacts

    def _check_grasp(self, gripper, obj):
        return False


def _pick_call(target: str, source: str | None = None) -> AtomicTaskCall:
    arguments = {"object_name": target}
    if source is not None:
        arguments["source_name"] = source
    return AtomicTaskCall.from_mapping(
        {
            "subgoal_id": f"pick_{target}",
            "atomic_task": "PickObject",
            "arguments": arguments,
            "termination_condition": {
                "predicate": "holding",
                "subject": target,
                "desired_value": True,
            },
            "policy_prompt": f"Pick up the {target}.",
        }
    )


def _contact(first: str, second: str) -> frozenset[str]:
    return frozenset((first, second))


def test_object_inside_bowl_uses_bowl_as_direct_support():
    env = _PickEnv(
        object_positions={
            "food": (0.0, 0.0, 0.86),
            "bowl": (0.0, 0.0, 0.80),
        },
        fixture_names=("counter",),
        contacts={_contact("food", "bowl"), _contact("bowl", "counter")},
    )

    baseline = capture_pick_motion_baseline(
        raw_env=env,
        object_alias="food",
        atomic_task_call=_pick_call("food", "bowl"),
    )

    assert baseline is not None
    assert baseline["source_object_alias"] == "bowl"
    assert baseline["initially_inside_source_object"] is True
    assert baseline["initial_support_contacts"] == [
        {
            "kind": "object",
            "alias": "bowl",
            "relation": "target_in_receptacle",
        }
    ]

    env.contacts.remove(_contact("food", "bowl"))
    env.sim.data.body_xpos[env.obj_body_id["food"], 2] += 0.05
    ready, evidence = evaluate_pick_motion(
        raw_env=env,
        object_alias="food",
        baseline=baseline,
        minimum_delta_m=0.04,
    )

    assert ready is True
    assert evidence["readiness_basis"] == "direct_support_departure"
    assert evidence["fallback_used"] is False
    assert evidence["current_support_contacts"] == [
        {"kind": "object", "alias": "bowl", "contact": False}
    ]


def test_loaded_bowl_ignores_contents_and_uses_fixture_support():
    env = _PickEnv(
        object_positions={
            "bowl": (0.0, 0.0, 0.80),
            "food": (0.0, 0.0, 0.86),
        },
        fixture_names=("counter",),
        contacts={_contact("bowl", "food"), _contact("bowl", "counter")},
    )

    baseline = capture_pick_motion_baseline(
        raw_env=env,
        object_alias="bowl",
        atomic_task_call=_pick_call("bowl", "counter"),
    )

    assert baseline is not None
    assert baseline["initial_support_contacts"] == [
        {"kind": "fixture", "alias": "counter", "relation": "contact"}
    ]
    assert all(
        item["alias"] != "food"
        for item in baseline["initial_support_contacts"]
    )


def test_unresolved_support_falls_back_to_relative_vertical_lift():
    env = _PickEnv(object_positions={"food": (0.0, 0.0, 0.80)})
    baseline = capture_pick_motion_baseline(
        raw_env=env,
        object_alias="food",
        atomic_task_call=_pick_call("food"),
    )

    assert baseline is not None
    assert baseline["initial_support_contacts"] == []

    env.sim.data.body_xpos[env.obj_body_id["food"], 2] += 0.05
    ready, evidence = evaluate_pick_motion(
        raw_env=env,
        object_alias="food",
        baseline=baseline,
        minimum_delta_m=0.04,
    )

    assert ready is True
    assert evidence["support_contacts_available"] is False
    assert evidence["fallback_used"] is True
    assert evidence["readiness_basis"] == "vertical_lift_fallback"


def test_unresolved_support_can_fall_back_to_source_object_departure():
    env = _PickEnv(
        object_positions={
            "food": (0.0, 0.0, 0.86),
            "bowl": (0.0, 0.0, 0.80),
        },
        contacts={_contact("food", "bowl")},
    )
    baseline = capture_pick_motion_baseline(
        raw_env=env,
        object_alias="food",
        atomic_task_call=_pick_call("food", "bowl"),
    )

    assert baseline is not None
    assert baseline["initially_inside_source_object"] is True
    baseline["_support_entities"] = ()
    baseline["initial_support_contacts"] = []
    env.contacts.clear()
    env.sim.data.body_xpos[env.obj_body_id["food"], 0] += 0.30

    ready, evidence = evaluate_pick_motion(
        raw_env=env,
        object_alias="food",
        baseline=baseline,
        minimum_delta_m=0.04,
    )

    assert ready is True
    assert evidence["vertical_lift_satisfied"] is False
    assert evidence["object_left_source_object"] is True
    assert evidence["readiness_basis"] == "source_object_departure_fallback"
