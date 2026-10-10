from __future__ import annotations

import sys
from pathlib import Path


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

from orchestrator import RoboCasaOrchestrator  # noqa: E402


def call(subgoal_id, task):
    return {
        "subgoal_id": subgoal_id,
        "atomic_task": task,
        "policy_prompt": f"Do {task}.",
        "arguments": {},
        "termination_condition": {"predicate": "open", "subject": "microwave"},
    }


class Adapter:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.env_ids = []
        self.atomic_tasks = []

    def execute(self, *, env, scheduler_query, episode_id):
        self.env_ids.append(id(env))
        success = next(self.statuses)
        atomic_call = scheduler_query["atomic_task_call"]
        self.atomic_tasks.append(atomic_call["atomic_task"])
        return {
            "subgoal_id": atomic_call["subgoal_id"],
            "status": "success" if success else "uncertain",
            "success": success,
            "verifier_result": {"retryable": not success},
        }


def test_plan_reuses_env_and_executes_in_order():
    adapter = Adapter([True, True])
    orchestrator = RoboCasaOrchestrator(atomic_task_policy_adapter=adapter)
    env = object()
    result = orchestrator.run_task_plan(
        env=env,
        task_plan=[call("g1", "OpenMicrowave"), call("g2", "TurnOnMicrowave")],
        episode_id=0,
    )
    assert result["success"] is True
    assert result["num_executed_steps"] == 2
    assert adapter.env_ids == [id(env), id(env)]


def test_plan_stops_after_uncertain_step():
    adapter = Adapter([False, True])
    orchestrator = RoboCasaOrchestrator(atomic_task_policy_adapter=adapter)
    result = orchestrator.run_task_plan(
        env=object(),
        task_plan=[call("g1", "OpenMicrowave"), call("g2", "TurnOnMicrowave")],
        episode_id=0,
    )
    assert result["success"] is False
    assert result["num_executed_steps"] == 1


def test_plan_retries_retryable_step_without_resetting_env():
    adapter = Adapter([False, True, True])
    orchestrator = RoboCasaOrchestrator(atomic_task_policy_adapter=adapter)
    env = object()
    result = orchestrator.run_task_plan(
        env=env,
        task_plan=[call("g1", "CloseMicrowave"), call("g2", "TurnOnMicrowave")],
        episode_id=0,
        max_task_retries=1,
    )
    assert result["success"] is True
    assert result["num_executed_steps"] == 2
    assert result["num_atomic_attempts"] == 3
    assert result["step_results"][0]["num_attempts"] == 2
    assert len(result["step_results"][0]["attempt_results"]) == 2
    assert adapter.env_ids == [id(env), id(env), id(env)]


def test_plan_does_not_retry_non_retryable_failure():
    class NonRetryableAdapter(Adapter):
        def execute(self, **kwargs):
            result = super().execute(**kwargs)
            result["verifier_result"]["retryable"] = False
            return result

    adapter = NonRetryableAdapter([False, True])
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter
    ).run_task_plan(
        env=object(),
        task_plan=[call("g1", "CloseMicrowave"), call("g2", "TurnOnMicrowave")],
        episode_id=0,
        max_task_retries=1,
    )
    assert result["num_executed_steps"] == 1
    assert result["num_atomic_attempts"] == 1


class Grounder:
    def __init__(self, results):
        self.results = iter(results)
        self.num_ground_calls = 0

    def ground(self, *, env, atomic_task_call):
        self.num_ground_calls += 1
        return next(self.results)

    def build_navigation_call(self, *, operation_call, grounding_result):
        return {
            "subgoal_id": f"navigate_before_{operation_call.subgoal_id}",
            "atomic_task": "NavigateKitchen",
            "policy_prompt": "Navigate to the sink.",
            "arguments": {
                "fixture_id": grounding_result["target_fixture_alias"],
                "fixture_name": "sink",
            },
            "termination_condition": {
                "predicate": "navigation_pose",
                "subject": grounding_result["target_fixture_alias"],
                "desired_value": True,
            },
        }


def grounding(grounded, *, fixture="sink_main_group", status=None, **extra):
    result = {
        "grounded": grounded,
        "status": status or ("grounded" if grounded else "navigation_required"),
        "target_fixture_alias": fixture,
        "target_entity_alias": "blender_jug",
        "target_entity_kind": "object",
        "target_mode": "operation_object_current_location",
        "evidence": [],
    }
    result.update(extra)
    return result


def pick_object_call():
    return {
        "subgoal_id": "pick_vegetable",
        "atomic_task": "PickObject",
        "policy_prompt": "Pick up the vegetable and keep holding it.",
        "arguments": {"object_id": "vegetable"},
        "termination_condition": {
            "predicate": "holding",
            "subject": "vegetable",
            "desired_value": True,
        },
    }


def place_object_call():
    return {
        "subgoal_id": "place_vegetable_in_bowl",
        "atomic_task": "PlaceObject",
        "policy_prompt": "Put the vegetable in the bowl.",
        "arguments": {"object_id": "vegetable", "destination_id": "bowl"},
        "termination_condition": {
            "predicate": "inside",
            "subject": "vegetable",
            "object": "bowl",
            "desired_value": True,
        },
    }


def test_grounded_operation_does_not_insert_navigation():
    adapter = Adapter([True])
    grounder = Grounder([grounding(True)])
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter,
        grounder=grounder,
    ).run_task_plan(
        env=object(),
        task_plan=[call("g1", "PickPlaceCounterToSink")],
        episode_id=0,
    )

    assert result["success"] is True
    assert result["num_inserted_navigation_steps"] == 0
    assert adapter.atomic_tasks == ["PickPlaceCounterToSink"]
    assert grounder.num_ground_calls == 1


def test_target_aware_adapter_receives_grounding_decision():
    class TargetAwareAdapter(Adapter):
        handles_target_aware_navigation = True

        def __init__(self):
            super().__init__([True])
            self.scheduler_queries = []

        def execute(self, *, env, scheduler_query, episode_id):
            self.scheduler_queries.append(dict(scheduler_query))
            return super().execute(
                env=env,
                scheduler_query=scheduler_query,
                episode_id=episode_id,
            )

    adapter = TargetAwareAdapter()
    grounder = Grounder([grounding(False)])
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter,
        grounder=grounder,
    ).run_task_plan(
        env=object(),
        task_plan=[call("g1", "PickPlaceCounterToSink")],
        episode_id=0,
    )

    assert result["success"] is True
    assert result["num_inserted_navigation_steps"] == 0
    assert adapter.scheduler_queries[0]["grounding_result"]["grounded"] is False
    assert adapter.scheduler_queries[0]["grounding_result"]["status"] == "navigation_required"


def test_scheduler_inserts_navigation_then_regrounds_before_operation():
    adapter = Adapter([True, True])
    grounder = Grounder([grounding(False), grounding(True)])
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter,
        grounder=grounder,
    ).run_task_plan(
        env=object(),
        task_plan=[call("g1", "PickPlaceCounterToSink")],
        episode_id=0,
    )

    assert result["success"] is True
    assert result["num_inserted_navigation_steps"] == 1
    assert result["num_atomic_attempts"] == 2
    assert adapter.atomic_tasks == ["NavigateKitchen", "PickPlaceCounterToSink"]
    assert [event["phase"] for event in result["grounding_events"]] == [
        "before_operation",
        "after_inserted_navigation",
    ]


def test_operation_is_blocked_when_regrounding_after_navigation_fails():
    adapter = Adapter([True])
    grounder = Grounder([grounding(False), grounding(False)])
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter,
        grounder=grounder,
    ).run_task_plan(
        env=object(),
        task_plan=[call("g1", "PickPlaceCounterToSink")],
        episode_id=0,
    )

    assert result["success"] is False
    assert result["status"] == "failed"
    assert result["step_results"][0]["failure_code"] == (
        "GROUNDING_FAILED_AFTER_NAVIGATION"
    )
    assert adapter.atomic_tasks == ["NavigateKitchen"]


def test_place_uses_pick_handoff_and_grounding_only_controls_navigation():
    adapter = Adapter([True, True, True])
    grounder = Grounder(
        [
            grounding(True, holding_source_object=False),
            grounding(
                False,
                fixture="counter_right_main_group",
                status="navigation_required",
                holding_source_object=False,
                target_entity_alias="bowl",
                entity_role="object_receptacle",
                destination_entity_alias="bowl",
            ),
            grounding(
                True,
                fixture="counter_right_main_group",
                holding_source_object=False,
                target_entity_alias="bowl",
                entity_role="object_receptacle",
                destination_entity_alias="bowl",
            ),
        ]
    )
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter,
        grounder=grounder,
    ).run_task_plan(
        env=object(),
        task_plan=[pick_object_call(), place_object_call()],
        episode_id=0,
        max_task_retries=1,
    )

    assert result["success"] is True
    assert adapter.atomic_tasks == ["PickObject", "NavigateKitchen", "PlaceObject"]
    assert result["num_atomic_attempts"] == 3
    assert result["num_inserted_navigation_steps"] == 1
    assert result["transition_events"] == []
    assert result["recovery_events"] == []
    place_groundings = result["grounding_events"][1:]
    assert [item["status"] for item in place_groundings] == [
        "navigation_required",
        "grounded",
    ]
    assert all(
        item["scheduler_handoff"]["holding_object_alias"] == "vegetable"
        for item in place_groundings
    )

def test_place_retry_receives_successful_navigation_handoff():
    class TargetAwareRetryAdapter:
        handles_target_aware_navigation = True

        def __init__(self):
            self.scheduler_queries = []
            self.place_attempts = 0

        def execute(self, *, env, scheduler_query, episode_id):
            del env, episode_id
            self.scheduler_queries.append(dict(scheduler_query))
            atomic_call = scheduler_query["atomic_task_call"]
            if atomic_call["atomic_task"] == "PickObject":
                return {
                    "subgoal_id": atomic_call["subgoal_id"],
                    "atomic_task": "PickObject",
                    "status": "success",
                    "success": True,
                    "verifier_result": {"retryable": False},
                }
            self.place_attempts += 1
            success = self.place_attempts == 2
            return {
                "subgoal_id": atomic_call["subgoal_id"],
                "atomic_task": "PlaceObject",
                "status": "success" if success else "uncertain",
                "success": success,
                "verifier_result": {"retryable": not success},
                "target_aware_stage": {
                    "navigation_required": self.place_attempts == 1,
                    "navigation_success": True,
                    "navigation_result": {
                        "success": True,
                        "reason": "SUCCESS",
                        "target_name": "bowl",
                        "target_type": "EXPOSED_OBJECT",
                    },
                },
            }

    adapter = TargetAwareRetryAdapter()
    grounder = Grounder(
        [
            grounding(True),
            grounding(
                False,
                fixture="counter_right_main_group",
                status="navigation_required",
                target_entity_alias="bowl",
            ),
        ]
    )
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter, grounder=grounder
    ).run_task_plan(
        env=object(),
        task_plan=[pick_object_call(), place_object_call()],
        episode_id=0,
        max_task_retries=1,
    )

    assert result["success"] is True
    first_place_query = adapter.scheduler_queries[1]
    retry_place_query = adapter.scheduler_queries[2]
    assert "reuse_successful_navigation" not in first_place_query
    handoff = retry_place_query["reuse_successful_navigation"]
    assert handoff["subgoal_id"] == "place_vegetable_in_bowl"
    assert handoff["source_attempt_index"] == 0
    assert handoff["navigation_result"]["reason"] == "SUCCESS"


class ScriptedAdapter:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.atomic_tasks = []
        self.scheduler_queries = []

    def execute(self, *, env, scheduler_query, episode_id):
        self.scheduler_queries.append(dict(scheduler_query))
        atomic_call = scheduler_query["atomic_task_call"]
        self.atomic_tasks.append(atomic_call["atomic_task"])
        outcome_value = dict(next(self.outcomes))
        success = bool(outcome_value.get("success", False))
        failure_code = outcome_value.get("failure_code")
        return {
            "subgoal_id": atomic_call["subgoal_id"],
            "atomic_task": atomic_call["atomic_task"],
            "status": "success" if success else "failed",
            "success": success,
            "verifier_result": {
                "retryable": bool(outcome_value.get("retryable", not success)),
                "failure_code": failure_code,
            },
        }


def outcome(success, failure_code=None):
    return {
        "success": success,
        "failure_code": failure_code,
        "retryable": not success,
    }


def test_same_skill_mode_preserves_legacy_place_retry_after_drop():
    adapter = ScriptedAdapter(
        [
            outcome(True),
            outcome(False, "OBJECT_DROPPED"),
            outcome(True),
        ]
    )
    grounder = Grounder([grounding(True), grounding(True)])
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter,
        grounder=grounder,
    ).run_task_plan(
        env=object(),
        task_plan=[pick_object_call(), place_object_call()],
        episode_id=0,
        max_task_retries=1,
        retry_recovery_mode="same_skill",
    )

    assert result["success"] is True
    assert result["retry_recovery_mode"] == "same_skill"
    assert result["num_dependency_recoveries"] == 0
    assert adapter.atomic_tasks == ["PickObject", "PlaceObject", "PlaceObject"]
    assert result["recovery_events"] == []


def test_same_skill_no_drop_retry_stops_place_after_confirmed_drop():
    adapter = ScriptedAdapter(
        [
            outcome(True),
            outcome(False, "OBJECT_DROPPED"),
            outcome(True),
        ]
    )
    grounder = Grounder([grounding(True), grounding(True)])
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter,
        grounder=grounder,
    ).run_task_plan(
        env=object(),
        task_plan=[pick_object_call(), place_object_call()],
        episode_id=0,
        max_task_retries=3,
        retry_recovery_mode="same_skill_no_drop_retry",
    )

    assert result["success"] is False
    assert result["retry_recovery_mode"] == "same_skill_no_drop_retry"
    assert result["num_dependency_recoveries"] == 0
    assert adapter.atomic_tasks == ["PickObject", "PlaceObject"]
    place_result = result["step_results"][1]
    assert place_result["num_attempts"] == 1
    assert place_result["retry_suppressed"] is True
    assert (
        place_result["retry_suppressed_reason"]
        == "OBJECT_DROPPED_WITHOUT_ROLLBACK"
    )
    assert result["recovery_events"] == []


def test_same_skill_no_drop_retry_never_rolls_back_missing_precondition():
    adapter = ScriptedAdapter(
        [
            outcome(False, "OBJECT_NOT_GRASPED"),
            outcome(False, "OBJECT_NOT_GRASPED"),
        ]
    )
    grounder = Grounder([grounding(True), grounding(True)])
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter,
        grounder=grounder,
    ).run_task_plan(
        env=object(),
        task_plan=[pick_object_call(), place_object_call()],
        episode_id=0,
        stop_on_unsuccessful=False,
        max_task_retries=1,
        retry_recovery_mode="same_skill_no_drop_retry",
    )

    assert result["success"] is False
    assert adapter.atomic_tasks == ["PickObject", "PickObject"]
    assert "retry_from_current_pose" not in adapter.scheduler_queries[0]
    retry_handoff = adapter.scheduler_queries[1]["retry_from_current_pose"]
    assert retry_handoff["subgoal_id"] == "pick_vegetable"
    assert retry_handoff["source_attempt_index"] == 0
    place_result = result["step_results"][1]
    assert place_result["failure_code"] == "PRECONDITION_NOT_SATISFIED"
    assert place_result["rollback_subgoal_id"] is None
    assert place_result["num_attempts"] == 0
    assert result["recovery_events"] == []


def test_dependency_recovery_regrabs_and_restarts_navigation_after_drop():
    adapter = ScriptedAdapter(
        [
            outcome(True),
            outcome(False, "OBJECT_DROPPED"),
            outcome(True),
            outcome(True),
            outcome(True),
        ]
    )
    grounder = Grounder(
        [
            grounding(True),
            grounding(False, fixture="counter_right_main_group"),
            grounding(False, fixture="counter_right_main_group"),
            grounding(True, fixture="counter_right_main_group"),
        ]
    )
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter,
        grounder=grounder,
    ).run_task_plan(
        env=object(),
        task_plan=[pick_object_call(), place_object_call()],
        episode_id=0,
        max_task_retries=1,
        retry_recovery_mode="dependency_recovery",
    )

    assert result["success"] is True
    assert result["retry_recovery_mode"] == "dependency_recovery"
    assert result["num_dependency_recoveries"] == 1
    assert adapter.atomic_tasks == [
        "PickObject",
        "NavigateKitchen",
        "PickObject",
        "NavigateKitchen",
        "PlaceObject",
    ]
    assert len(result["recovery_events"]) == 1
    assert result["recovery_events"][0]["recovery_phase"] == "navigation"
    assert result["recovery_events"][0]["trigger_failure_code"] == "OBJECT_DROPPED"
    assert any(
        event["type"] == "HOLDING_DEPENDENCY_INVALIDATED"
        for event in result["transition_events"]
    )


def test_dependency_recovery_regrabs_and_regrounds_after_place_drop():
    adapter = ScriptedAdapter(
        [
            outcome(True),
            outcome(False, "OBJECT_DROPPED"),
            outcome(True),
            outcome(True),
            outcome(True),
        ]
    )
    grounder = Grounder(
        [
            grounding(True),
            grounding(True, fixture="counter_right_main_group"),
            grounding(False, fixture="counter_right_main_group"),
            grounding(True, fixture="counter_right_main_group"),
        ]
    )
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter,
        grounder=grounder,
    ).run_task_plan(
        env=object(),
        task_plan=[pick_object_call(), place_object_call()],
        episode_id=0,
        max_task_retries=1,
        retry_recovery_mode="dependency_recovery",
    )

    assert result["success"] is True
    assert result["num_dependency_recoveries"] == 1
    assert adapter.atomic_tasks == [
        "PickObject",
        "PlaceObject",
        "PickObject",
        "NavigateKitchen",
        "PlaceObject",
    ]
    assert result["recovery_events"][0]["recovery_phase"] == "operation"
    assert [
        item["phase"] for item in result["grounding_events"]
    ] == [
        "before_operation",
        "before_operation",
        "after_drop_recovery",
        "after_inserted_navigation",
    ]


def test_dependency_recovery_preserves_post_grasp_drop_failure_code():
    adapter = ScriptedAdapter(
        [
            outcome(True),
            outcome(False, "OBJECT_DROPPED_DURING_PREP"),
            outcome(True),
            outcome(True),
            outcome(True),
        ]
    )
    grounder = Grounder(
        [
            grounding(True),
            grounding(False, fixture="counter_right_main_group"),
            grounding(False, fixture="counter_right_main_group"),
            grounding(True, fixture="counter_right_main_group"),
        ]
    )
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter,
        grounder=grounder,
    ).run_task_plan(
        env=object(),
        task_plan=[pick_object_call(), place_object_call()],
        episode_id=0,
        max_task_retries=1,
        retry_recovery_mode="dependency_recovery",
    )

    assert result["success"] is True
    assert result["num_dependency_recoveries"] == 1
    assert result["recovery_events"][0]["trigger_failure_code"] == (
        "OBJECT_DROPPED_DURING_PREP"
    )
    assert adapter.atomic_tasks == [
        "PickObject",
        "NavigateKitchen",
        "PickObject",
        "NavigateKitchen",
        "PlaceObject",
    ]


def test_retry_recovery_mode_is_validated():
    try:
        RoboCasaOrchestrator(
            atomic_task_policy_adapter=Adapter([True])
        ).run_task_plan(
            env=object(),
            task_plan=[call("g1", "OpenMicrowave")],
            episode_id=0,
            retry_recovery_mode="unknown",
        )
    except ValueError as exc:
        assert "retry_recovery_mode" in str(exc)
    else:
        raise AssertionError("invalid retry recovery mode should fail")

def test_dependency_mode_keeps_same_skill_retry_for_pick_failure():
    adapter = ScriptedAdapter(
        [
            outcome(False, "OBJECT_DROPPED"),
            outcome(True),
        ]
    )
    grounder = Grounder([grounding(True)])
    result = RoboCasaOrchestrator(
        atomic_task_policy_adapter=adapter,
        grounder=grounder,
    ).run_task_plan(
        env=object(),
        task_plan=[pick_object_call()],
        episode_id=0,
        max_task_retries=1,
        retry_recovery_mode="dependency_recovery",
    )

    assert result["success"] is True
    assert adapter.atomic_tasks == ["PickObject", "PickObject"]
    assert result["num_dependency_recoveries"] == 0
    assert result["recovery_events"] == []
