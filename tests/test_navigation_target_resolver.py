from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robocasa.atomic_task_schemas import AtomicTaskCall
from robocasa.navigation.navigation_types import TargetType
from robocasa.navigation.target_resolver import TargetResolver


class FakeFixture:
    def __init__(self, name):
        self.name = name
        self.nat_lang = name.replace("_", " ")
        self.pos = np.array([2.0, 3.0, 0.8])
        self.rot = 0.5


class FakeObject:
    def __init__(self, name):
        self.name = name
        self.nat_lang = name


class FakeEnv:
    def __init__(self):
        self.fixtures = {
            "counter": FakeFixture("counter"),
            "microwave": FakeFixture("microwave"),
        }
        self.objects = {
            "vegetable": FakeObject("vegetable"),
            "bowl": FakeObject("bowl"),
        }
        self.obj_body_id = {"vegetable": 0, "bowl": 1}
        self.sim = SimpleNamespace(
            data=SimpleNamespace(
                body_xpos=np.array([[0.2, 0.3, 0.9], [1.1, 1.2, 0.9]]),
                body_xquat=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            )
        )


def call(task, arguments, condition):
    return AtomicTaskCall.from_mapping(
        {
            "subgoal_id": "test",
            "atomic_task": task,
            "arguments": arguments,
            "termination_condition": condition,
        }
    )


def test_fixture_target_uses_live_fixture_pose():
    target = TargetResolver().resolve(
        FakeEnv(),
        call(
            "CloseMicrowave",
            {"fixture_id": "microwave"},
            {"predicate": "closed", "subject": "microwave", "desired_value": True},
        ),
    )
    assert target.target_type == TargetType.FIXTURE
    assert target.target_alias == "microwave"
    assert target.world_position == (2.0, 3.0, 0.8)


def test_pick_resolves_exposed_object_at_current_simulator_position(monkeypatch):
    resolver = TargetResolver()
    monkeypatch.setattr(
        resolver, "locate_support_fixture", lambda raw, alias: ("counter", ())
    )
    monkeypatch.setattr(resolver, "_is_enclosed_fixture", lambda fixture: False)
    target = resolver.resolve(
        FakeEnv(),
        call(
            "PickObject",
            {"object_id": "vegetable"},
            {"predicate": "holding", "subject": "vegetable", "desired_value": True},
        ),
    )
    assert target.target_type == TargetType.EXPOSED_OBJECT
    assert target.world_position == (0.2, 0.3, 0.9)
    assert target.support_fixture == "counter"


def test_object_in_fixture_keeps_object_pose_but_uses_fixture_access_side(monkeypatch):
    resolver = TargetResolver()
    monkeypatch.setattr(
        resolver, "locate_support_fixture", lambda raw, alias: ("microwave", ())
    )
    monkeypatch.setattr(resolver, "_is_enclosed_fixture", lambda fixture: True)
    target = resolver.resolve(
        FakeEnv(),
        call(
            "PickObject",
            {"object_id": "vegetable"},
            {"predicate": "holding", "subject": "vegetable", "desired_value": True},
        ),
    )
    assert target.target_type == TargetType.OBJECT_IN_FIXTURE
    assert target.support_fixture == "microwave"
    assert target.world_position == (0.2, 0.3, 0.9)


def test_place_targets_destination_object_instead_of_held_source(monkeypatch):
    resolver = TargetResolver()
    monkeypatch.setattr(
        resolver, "locate_support_fixture", lambda raw, alias: ("counter", ())
    )
    monkeypatch.setattr(resolver, "_is_enclosed_fixture", lambda fixture: False)
    target = resolver.resolve(
        FakeEnv(),
        call(
            "PlaceObject",
            {"object_id": "vegetable", "destination_id": "bowl"},
            {
                "predicate": "inside",
                "subject": "vegetable",
                "object": "bowl",
                "desired_value": True,
            },
        ),
    )
    assert target.target_alias == "bowl"
    assert target.world_position == (1.1, 1.2, 0.9)


def test_combined_pick_place_defaults_to_live_source_object(monkeypatch):
    resolver = TargetResolver()
    monkeypatch.setattr(
        resolver, "locate_support_fixture", lambda raw, alias: ("counter", ())
    )
    monkeypatch.setattr(resolver, "_is_enclosed_fixture", lambda fixture: False)
    target = resolver.resolve(
        FakeEnv(),
        call(
            "PickPlaceObject",
            {"object_id": "vegetable", "destination_id": "bowl"},
            {
                "predicate": "inside",
                "subject": "vegetable",
                "object": "bowl",
                "desired_value": True,
            },
        ),
    )
    assert target.target_alias == "vegetable"
    assert target.world_position == (0.2, 0.3, 0.9)
