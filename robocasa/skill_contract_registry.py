"""Skill contracts shared by the VLM planner, scheduler, and verifier.

The registry deliberately separates the policy goal from the state required for a
safe handoff.  A placement is therefore not complete merely because an object was
observed inside a receptacle for one frame: the policy must release it and withdraw
before the next skill is allowed to start.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

try:
    from .atomic_task_schemas import AtomicTaskCall, load_available_atomic_tasks
except ImportError:  # Direct execution from the robocasa script directory.
    from atomic_task_schemas import AtomicTaskCall, load_available_atomic_tasks


CONTRACT_VERSION = 1
DEFAULT_CONSECUTIVE_SUCCESSES = 2


@dataclass(frozen=True)
class SkillContract:
    """Declarative task-family contract before call-specific alias resolution."""

    atomic_task: str
    family: str
    precondition: tuple[dict[str, Any], ...]
    goal: dict[str, Any]
    handoff: tuple[dict[str, Any], ...]
    failure: tuple[dict[str, Any], ...]
    recovery: tuple[dict[str, Any], ...]
    required_consecutive_successes: int = DEFAULT_CONSECUTIVE_SUCCESSES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_BASE_PRECONDITIONS = (
    {
        "name": "environment_ready",
        "check": "environment_has_current_observation",
        "on_failure": "Replan",
    },
    {
        "name": "aliases_resolvable",
        "check": "all_call_aliases_exist_in_scene",
        "on_failure": "Replan",
    },
)

_BASE_FAILURES = (
    {
        "code": "PRECONDITION_FAILED",
        "trigger": "a declared precondition is false or unverifiable",
        "retryable": False,
        "recovery": "Replan",
    },
    {
        "code": "SKILL_TIMEOUT",
        "trigger": "the atomic horizon is exhausted before stable verification",
        "retryable": True,
        "recovery": "RetrySkill",
    },
    {
        "code": "INVALID_TERMINATION_CONDITION",
        "trigger": "the goal or handoff condition cannot be evaluated",
        "retryable": False,
        "recovery": "Replan",
    },
)

_BASE_RECOVERIES = (
    {
        "name": "RetrySkill",
        "action": "retry_the_same_atomic_task_from_current_state",
    },
    {
        "name": "Replan",
        "action": "refresh_scene_state_and_request_a_new_remaining_plan",
    },
)


def _task_family(atomic_task: str) -> str:
    transfer_tasks = {
        "CheesyBread",
        "CoffeeServeMug",
        "CoffeeSetupMug",
        "MakeIcedCoffee",
        "PackDessert",
    }
    if atomic_task == "NavigateKitchen":
        return "navigation"
    if atomic_task == "PickObject":
        return "object_pick"
    if (
        atomic_task == "PlaceObject"
        or atomic_task.startswith("PickPlace")
        or atomic_task in transfer_tasks
    ):
        return "object_transfer"
    if atomic_task.startswith(("Open", "Close", "Slide", "Lower")):
        return "fixture_articulation"
    if atomic_task.startswith(("TurnOn", "TurnOff", "Start", "Preheat", "Adjust")):
        return "fixture_control"
    return "manipulation"


def _family_definition(atomic_task: str) -> SkillContract:
    family = _task_family(atomic_task)
    preconditions = list(deepcopy(_BASE_PRECONDITIONS))
    handoff: list[dict[str, Any]] = []
    failures = list(deepcopy(_BASE_FAILURES))
    recoveries = list(deepcopy(_BASE_RECOVERIES))

    if family == "navigation":
        preconditions.append(
            {
                "name": "navigation_target_exists",
                "check": "arguments.fixture_id_resolves_to_fixture",
                "on_failure": "Replan",
            }
        )
        failures.append(
            {
                "code": "NAVIGATION_NOT_CONVERGED",
                "trigger": "base pose remains outside navigation tolerances",
                "retryable": True,
                "recovery": "RetrySkill",
            }
        )
    elif family == "object_pick":
        preconditions.extend(
            (
                {
                    "name": "pick_object_exists",
                    "check": "goal_subject_resolves_to_object",
                    "on_failure": "Replan",
                },
                {
                    "name": "gripper_available",
                    "check": "gripper_is_empty_or_holds_goal_object",
                    "on_failure": "ReleaseAndRetract",
                },
            )
        )
        handoff.append({"rule": "keep_goal_object_held_for_next_skill"})
        failures.extend(
            (
                {
                    "code": "OBJECT_NOT_GRASPED",
                    "trigger": "the requested object is not held stably",
                    "retryable": True,
                    "recovery": "RetrySkill",
                },
                {
                    "code": "OBJECT_DROPPED",
                    "trigger": "a confirmed grasp is lost before handoff",
                    "retryable": True,
                    "recovery": "RetrySkill",
                },
            )
        )
    elif family == "object_transfer":
        preconditions.extend(
            (
                {
                    "name": "transfer_entities_exist",
                    "check": "goal_subject_and_destination_are_resolvable",
                    "on_failure": "Replan",
                },
                {
                    "name": "gripper_available",
                    "check": "gripper_is_empty_or_holds_goal_object",
                    "on_failure": "ReleaseAndRetract",
                },
            )
        )
        handoff.extend(
            (
                {"rule": "released_goal_object"},
                {
                    "rule": "eef_outside_destination_receptacle",
                    "margin": 0.02,
                },
            )
        )
        failures.extend(
            (
                {
                    "code": "OBJECT_DROPPED",
                    "trigger": (
                        "a confirmed grasp is lost before the destination relation "
                        "is satisfied"
                    ),
                    "retryable": True,
                    "recovery": "RegraspObject",
                },
                {
                    "code": "OBJECT_NOT_RELEASED",
                    "trigger": "goal relation is true but the gripper still holds the object",
                    "retryable": True,
                    "recovery": "ReleaseAndRetract",
                },
                {
                    "code": "UNSAFE_HANDOFF_POSE",
                    "trigger": "the gripper or end effector has not cleared the handoff region",
                    "retryable": True,
                    "recovery": "RetractFromFixture",
                },
                {
                    "code": "GOAL_DESTABILIZED",
                    "trigger": "the goal becomes false during consecutive verification",
                    "retryable": True,
                    "recovery": "RetrySkill",
                },
            )
        )
        recoveries[0:0] = [
            {
                "name": "RegraspObject",
                "action": "retry_the_transfer_from_current_state_to_regrasp_object",
            },
            {
                "name": "ReleaseAndRetract",
                "action": "open_gripper_then_move_eef_away_from_the_goal_object",
            },
            {
                "name": "RetractFromFixture",
                "action": "move_eef_outside_the_destination_fixture_without_disturbing_goal",
            },
        ]
    else:
        preconditions.append(
            {
                "name": "controlled_entity_exists",
                "check": "goal_subject_resolves_in_scene",
                "on_failure": "Replan",
            }
        )
        failures.append(
            {
                "code": "GOAL_NOT_REACHED",
                "trigger": "the requested fixture or manipulation state is not stable",
                "retryable": True,
                "recovery": "RetrySkill",
            }
        )

    return SkillContract(
        atomic_task=atomic_task,
        family=family,
        precondition=tuple(preconditions),
        goal={"source": "atomic_task_call.termination_condition"},
        handoff=tuple(handoff),
        failure=tuple(failures),
        recovery=tuple(recoveries),
    )


def load_skill_contract_registry() -> dict[str, SkillContract]:
    """Return a contract for every atomic task known to this checkout."""

    return {
        atomic_task: _family_definition(atomic_task)
        for atomic_task in sorted(load_available_atomic_tasks())
    }


def get_skill_contract(atomic_task: str) -> SkillContract:
    """Look up one contract and fail early for an unknown policy skill."""

    registry = load_skill_contract_registry()
    try:
        return registry[atomic_task]
    except KeyError as exc:
        raise ValueError(f"No skill contract registered for {atomic_task!r}") from exc


def _condition_key(condition: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        condition.get("predicate"),
        condition.get("subject"),
        condition.get("object", condition.get("destination")),
        condition.get("object_prefix"),
    )


def _as_conditions(
    condition: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(condition, Mapping):
        return [dict(condition)]
    return [dict(item) for item in condition]


def _resolved_handoff_conditions(
    contract: SkillContract,
    goal_conditions: Sequence[Mapping[str, Any]],
    scene_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resolve family handoff rules against the concrete goal aliases."""

    if contract.family != "object_transfer":
        return []
    relation_predicates = {"inside", "inserted", "on", "object_fixture_relation"}
    relation = next(
        (
            condition
            for condition in goal_conditions
            if str(condition.get("predicate", "")).lower() in relation_predicates
        ),
        None,
    )
    handoff: list[dict[str, Any]] = []
    if relation is not None:
        subject = str(relation.get("subject", "")).strip()
        destination = str(
            relation.get("object", relation.get("destination", ""))
        ).strip()
        if subject:
            handoff.append(
                {
                    "predicate": "released",
                    "subject": subject,
                    "desired_value": True,
                }
            )
        if destination:
            handoff.append(
                {
                    "predicate": "eef_outside_receptacle",
                    "subject": destination,
                    "margin": 0.02,
                    "desired_value": True,
                }
            )
    elif contract.atomic_task == "MakeIcedCoffee":
        count_goal = next(
            (
                condition
                for condition in goal_conditions
                if condition.get("predicate") == "receptacle_count"
            ),
            None,
        )
        if count_goal is not None:
            object_prefix = str(count_goal.get("object_prefix", "")).strip()
            receptacle = str(count_goal.get("subject", "")).strip()
            if object_prefix:
                handoff.append(
                    {
                        "predicate": "released",
                        "subject": object_prefix,
                        "object_prefix": object_prefix,
                        "desired_value": True,
                    }
                )
            if receptacle:
                handoff.append(
                    {
                        "predicate": "gripper_far",
                        "subject": receptacle,
                        "threshold": 0.25,
                        "desired_value": True,
                    }
                )
    return handoff


def apply_skill_contract(
    atomic_task_call: AtomicTaskCall | Mapping[str, Any],
    scene_context: Mapping[str, Any] | None = None,
) -> tuple[AtomicTaskCall, list[dict[str, Any]]]:
    """Attach a resolved contract and add enforceable handoff predicates.

    The operation is idempotent so saved, already-normalized plans can safely pass
    through the same planner/scheduler preparation path again.
    """

    call = (
        atomic_task_call
        if isinstance(atomic_task_call, AtomicTaskCall)
        else AtomicTaskCall.from_mapping(atomic_task_call)
    )
    contract = get_skill_contract(call.atomic_task)
    value = call.to_dict()
    metadata = value["metadata"]
    existing_contract = metadata.get("skill_contract")
    existing_goal = (
        existing_contract.get("goal", {}).get("conditions")
        if isinstance(existing_contract, Mapping)
        and isinstance(existing_contract.get("goal"), Mapping)
        else None
    )
    goal_conditions = (
        _as_conditions(existing_goal)
        if isinstance(existing_goal, (Mapping, list))
        else _as_conditions(value["termination_condition"])
    )
    all_conditions = _as_conditions(value["termination_condition"])
    removed: list[dict[str, Any]] = []
    if contract.family == "object_transfer":
        relation_predicates = {
            "inside",
            "inserted",
            "on",
            "object_fixture_relation",
        }
        has_relation = any(
            str(condition.get("predicate", "")).lower()
            in relation_predicates
            for condition in goal_conditions
        )
        if has_relation:
            handoff_predicates = {
                "released",
                "gripper_far",
                "eef_outside_fixture",
                "eef_outside_receptacle",
            }
            primary_goal_conditions = [
                condition
                for condition in goal_conditions
                if str(condition.get("predicate", "")).lower()
                not in handoff_predicates
            ]
            if primary_goal_conditions:
                goal_conditions = primary_goal_conditions
            legacy_predicates = {"gripper_far", "eef_outside_fixture"}
            removed = [
                condition
                for condition in all_conditions
                if str(condition.get("predicate", "")).lower()
                in legacy_predicates
            ]
            all_conditions = [
                condition
                for condition in all_conditions
                if str(condition.get("predicate", "")).lower()
                not in legacy_predicates
            ]
    handoff_conditions = _resolved_handoff_conditions(
        contract, goal_conditions, dict(scene_context or {})
    )
    existing_keys = {_condition_key(condition) for condition in all_conditions}
    added: list[dict[str, Any]] = []
    for condition in handoff_conditions:
        if _condition_key(condition) not in existing_keys:
            all_conditions.append(condition)
            existing_keys.add(_condition_key(condition))
            added.append(condition)

    if len(all_conditions) == 1 and not isinstance(value["termination_condition"], list):
        value["termination_condition"] = all_conditions[0]
    else:
        value["termination_condition"] = all_conditions
    metadata["skill_contract"] = {
        "contract_version": CONTRACT_VERSION,
        "atomic_task": contract.atomic_task,
        "family": contract.family,
        "precondition": deepcopy(list(contract.precondition)),
        "goal": {"conditions": deepcopy(goal_conditions)},
        "handoff": {"conditions": deepcopy(handoff_conditions)},
        "failure": deepcopy(list(contract.failure)),
        "recovery": deepcopy(list(contract.recovery)),
        "verification": {
            "required_consecutive_successes": contract.required_consecutive_successes,
        },
    }
    changes = []
    if added or removed or not isinstance(existing_contract, Mapping):
        changes.append(
            {
                "type": "apply_skill_contract",
                "subgoal_id": call.subgoal_id,
                "atomic_task": call.atomic_task,
                "family": contract.family,
                "added_handoff_predicates": [item["predicate"] for item in added],
                "removed_handoff_predicates": [
                    item["predicate"] for item in removed
                ],
                "required_consecutive_successes": (
                    contract.required_consecutive_successes
                ),
            }
        )
    return AtomicTaskCall.from_mapping(value), changes
