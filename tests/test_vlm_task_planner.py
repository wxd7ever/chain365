from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))

import vlm_task_planner  # noqa: E402
import vlm_client  # noqa: E402
from vlm_task_planner import (  # noqa: E402
    OpenAICompatibleVLMPlanner,
    parse_vlm_task_plan,
)


def plan_value():
    return {
        "atomic_task_calls": [
            {
                "subgoal_id": "g1",
                "atomic_task": "OpenMicrowave",
                "policy_prompt": "Open the microwave door.",
                "arguments": {"fixture_name": "microwave"},
                "termination_condition": {
                    "predicate": "open",
                    "subject": "microwave",
                },
            },
            {
                "subgoal_id": "g2",
                "atomic_task": "TurnOnMicrowave",
                "policy_prompt": "Press the start button on the microwave.",
                "arguments": {"fixture_name": "microwave"},
                "termination_condition": {
                    "predicate": "powered",
                    "subject": "microwave",
                },
            },
        ]
    }


def test_parse_vlm_plan_validates_real_atomic_tasks():
    calls = parse_vlm_task_plan(json.dumps(plan_value()))
    assert [call.atomic_task for call in calls] == [
        "OpenMicrowave",
        "TurnOnMicrowave",
    ]


def test_parse_vlm_plan_rejects_invented_alias():
    value = plan_value()
    value["atomic_task_calls"][0]["atomic_task"] = "OpenSingleDoor"
    with pytest.raises(ValueError, match="Unknown RoboCasa atomic task"):
        parse_vlm_task_plan(value)


def test_openai_compatible_request_and_response(monkeypatch):
    response_body = {
        "choices": [
            {"message": {"content": json.dumps(plan_value())}}
        ],
        "usage": {"total_tokens": 10},
    }
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(response_body).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.headers.get("Authorization")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(vlm_client.urllib.request, "urlopen", fake_urlopen)
    planner = OpenAICompatibleVLMPlanner(
        base_url="http://172.16.11.115:11434/v1",
        model="qwen2.5vl:3b",
        api_key="ollama",
        include_images=False,
    )
    calls, provenance = planner.plan(task="Open and turn on the microwave.")
    assert len(calls) == 2
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["authorization"] == "Bearer ollama"
    assert captured["payload"]["model"] == "qwen2.5vl:3b"
    assert captured["payload"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert provenance["num_images"] == 0


def test_planner_uses_shared_client_and_clears_failed_response_diagnostics():
    planner = OpenAICompatibleVLMPlanner(
        base_url="http://vlm.invalid/v1",
        model="qwen-vl",
        api_key=None,
        include_images=False,
    )

    class FakeSharedClient:
        endpoint = "http://vlm.invalid/v1/chat/completions"
        last_response_text = json.dumps(plan_value())
        last_raw_http_response = "raw-http"
        last_parsed_response = plan_value()
        last_http_status = 200
        last_finish_reason = "stop"
        last_usage = {"total_tokens": 10}

        def query_json(self, **kwargs):
            assert kwargs["images"] == []
            return plan_value()

    planner._vlm_client = FakeSharedClient()
    calls, provenance = planner.plan(task="Open and turn on the microwave.")
    assert len(calls) == 2
    assert provenance["raw_http_response"] == "raw-http"
    assert planner.last_finish_reason == "stop"
