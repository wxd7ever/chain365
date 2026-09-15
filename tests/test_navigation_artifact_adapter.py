from __future__ import annotations

from robocasa.navigation.navigation_types import NavigationResult
from robocasa.navigation.skill_adapter import TargetAwarePolicyAdapter


class FakePolicy:
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def execute(self, **kwargs):
        return {
            "success": True,
            "status": "success",
            "verifier_result": {"goal_satisfied": True, "retryable": False},
        }


class FakePipeline:
    def __init__(self):
        self.config = {
            "navigation": {
                "save_debug_artifacts": True,
                "debug_artifact_subdir": "navigation_planning",
            }
        }
        self.last_debug = {}

    def navigate_to_manipulation_target(self, *args, **kwargs):
        return NavigationResult(
            True,
            "ALREADY_AT_WORK_REGION",
            target_name="bowl",
            target_type="EXPOSED_OBJECT",
            skipped=True,
        )


def test_target_aware_adapter_links_saved_navigation_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "robocasa.navigation.skill_adapter.current_base_pose",
        lambda env: {"position": [0.0, 0.0, 0.7], "yaw_rad": 0.0},
    )
    captured = {}

    def save_artifacts(**kwargs):
        captured.update(kwargs)
        return {
            "directory": str(tmp_path / "navigation_planning" / "000001_pick_bowl"),
            "metadata": str(tmp_path / "metadata.json"),
            "overview": str(tmp_path / "overview.png"),
            "map_available": True,
        }

    monkeypatch.setattr(
        "robocasa.navigation.skill_adapter.save_navigation_plan_artifacts",
        save_artifacts,
    )
    adapter = TargetAwarePolicyAdapter(
        policy_adapter=FakePolicy(tmp_path),
        navigation_pipeline=FakePipeline(),
        navigation_enabled=True,
        local_refiner_enabled=False,
    )
    call = {
        "subgoal_id": "pick_bowl",
        "atomic_task": "PickObject",
        "arguments": {"object_id": "bowl"},
        "termination_condition": {
            "predicate": "holding",
            "subject": "bowl",
            "desired_value": True,
        },
        "policy_prompt": "Pick up the bowl.",
    }

    result = adapter.execute(
        env=object(),
        scheduler_query={"atomic_task_call": call},
        episode_id=0,
    )

    assert captured["artifact_index"] == 1
    assert captured["subgoal_id"] == "pick_bowl"
    assert captured["output_root"] == tmp_path / "navigation_planning"
    artifacts = result["target_aware_stage"]["navigation_artifacts"]
    assert artifacts["saved"] is True
    assert result["navigation_result"]["artifacts"] == artifacts
