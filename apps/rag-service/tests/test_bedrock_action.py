from __future__ import annotations

import json

from app.bedrock_action import BedrockActionEvent, create_action_response, handle_rag_action


class FakeWorkflow:
    def __init__(self) -> None:
        self.calls = []

    def run(self, query: str, top_k: int, filters: dict):
        self.calls.append((query, top_k, filters))
        return {
            "answer": "workflow-answer",
            "citations": [{"sourceId": "c1", "title": "Doc", "url": "https://example.com"}],
        }


def _decode_body(response: dict) -> dict:
    body = response["response"]["responseBody"]["application/json"]["body"]
    return json.loads(body)


def test_event_from_raw_parses_fields() -> None:
    raw = {
        "actionGroup": "rag_search",
        "apiPath": "/rag_search",
        "httpMethod": "POST",
        "messageVersion": "1.0",
        "requestBody": {
            "content": {
                "application/json": {
                    "properties": [
                        {"name": "query", "type": "string", "value": "hello"},
                        {"name": "topK", "type": "integer", "value": "3"},
                    ],
                }
            }
        },
    }
    event = BedrockActionEvent.from_raw(raw)
    assert event.action_group == "rag_search"
    assert event.properties["query"] == "hello"
    assert event.properties["topK"] == 3


def test_create_action_response_wraps_body() -> None:
    event = BedrockActionEvent(
        action_group="rag_search",
        api_path="/rag_search",
        http_method="POST",
        message_version="1.0",
        session_attributes=None,
        prompt_session_attributes=None,
        properties={},
    )
    response = create_action_response(event, {"answer": "ok"}, 200)
    assert response["response"]["httpStatusCode"] == 200
    assert _decode_body(response)["answer"] == "ok"


def test_handle_rag_action_missing_query_returns_400() -> None:
    workflow = FakeWorkflow()
    response = handle_rag_action({"requestBody": {}}, workflow)
    assert response["response"]["httpStatusCode"] == 400
    assert _decode_body(response)["citations"] == []


def test_handle_rag_action_success() -> None:
    workflow = FakeWorkflow()
    raw_event = {
        "actionGroup": "rag_search",
        "apiPath": "/rag_search",
        "httpMethod": "POST",
        "messageVersion": "1.0",
        "requestBody": {
            "content": {
                "application/json": {
                    "properties": [
                        {"name": "query", "type": "string", "value": "policy question"},
                        {"name": "topK", "type": "integer", "value": "7"},
                        {"name": "filters", "type": "object", "value": '{"source_type": "crawler"}'},
                    ]
                }
            }
        },
    }

    response = handle_rag_action(raw_event, workflow)
    assert response["response"]["httpStatusCode"] == 200
    body = _decode_body(response)
    assert body["answer"] == "workflow-answer"
    assert body["citations"][0]["sourceId"] == "c1"
    assert workflow.calls[0] == ("policy question", 7, {"source_type": "crawler"})
