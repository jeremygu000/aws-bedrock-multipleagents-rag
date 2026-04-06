"""Bedrock Action Group adapter for the Python RAG retrieval service.

This module converts Bedrock action payloads into internal retrieval requests
and converts retrieval results back into Bedrock-compatible response envelopes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .workflow import RagWorkflow


@dataclass(frozen=True)
class BedrockActionEvent:
    """Normalized view of a Bedrock action event payload."""

    action_group: str
    api_path: str
    http_method: str
    message_version: str
    session_attributes: dict[str, str] | None
    prompt_session_attributes: dict[str, str] | None
    properties: dict[str, Any]

    @classmethod
    def from_raw(cls, event: dict[str, Any]) -> "BedrockActionEvent":
        """Parse a raw Bedrock action event into a stable internal object."""

        json_body = (
            event.get("requestBody", {})
            .get("content", {})
            .get("application/json", {})
            .get("properties", {})
        )
        properties = _normalize_properties(json_body)
        return cls(
            action_group=str(event.get("actionGroup", "rag_search")),
            api_path=str(event.get("apiPath", "/rag_search")),
            http_method=str(event.get("httpMethod", "POST")),
            message_version=str(event.get("messageVersion", "1.0")),
            session_attributes=_safe_str_dict(event.get("sessionAttributes")),
            prompt_session_attributes=_safe_str_dict(event.get("promptSessionAttributes")),
            properties=properties,
        )


def create_action_response(
    event: BedrockActionEvent,
    body: dict[str, Any],
    status_code: int = 200,
) -> dict[str, Any]:
    """Build a Bedrock action response envelope with JSON body serialization."""

    return {
        "messageVersion": event.message_version,
        "response": {
            "actionGroup": event.action_group,
            "apiPath": event.api_path,
            "httpMethod": event.http_method,
            "httpStatusCode": status_code,
            "responseBody": {
                "application/json": {
                    "body": json.dumps(body, ensure_ascii=False),
                }
            },
        },
        "sessionAttributes": event.session_attributes,
        "promptSessionAttributes": event.prompt_session_attributes,
    }


def handle_rag_action(raw_event: dict[str, Any], workflow: RagWorkflow) -> dict[str, Any]:
    """Handle Bedrock `rag_search` action end-to-end.

    Flow:
    1. Parse and validate action input.
    2. Execute LangGraph workflow (request build -> retrieve -> generate).
    3. Return concise answer text with structured citations.
    """

    event = BedrockActionEvent.from_raw(raw_event)
    query = str(event.properties.get("query", "")).strip()
    if not query:
        return create_action_response(
            event,
            {"answer": "Missing required field: query", "citations": []},
            status_code=400,
        )

    top_k = _coerce_int(event.properties.get("topK"), default=5, min_value=1, max_value=20)
    filters = event.properties.get("filters")
    normalized_filters = filters if isinstance(filters, dict) else {}

    state = workflow.run(
        query=query,
        top_k=top_k,
        filters=normalized_filters,
    )
    citations = state.get("citations", [])
    answer = state.get(
        "answer",
        "I could not produce a grounded answer from the current evidence.",
    )

    return create_action_response(
        event,
        {
            "answer": answer,
            "citations": citations,
        },
        status_code=200,
    )


def _normalize_properties(raw: Any) -> dict[str, Any]:
    """Convert Bedrock properties to a flat dict.

    Bedrock sends properties as a list of ``{name, type, value}`` objects.
    Legacy tests may pass a plain dict.  Accept both.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        out: dict[str, Any] = {}
        for item in raw:
            if isinstance(item, dict) and "name" in item:
                value = item.get("value", "")
                prop_type = str(item.get("type", "string")).lower()
                if prop_type == "integer":
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        pass
                elif prop_type == "object":
                    if isinstance(value, str):
                        try:
                            import json as _json

                            value = _json.loads(value)
                        except (ValueError, TypeError):
                            pass
                elif prop_type == "boolean":
                    value = str(value).lower() in ("true", "1", "yes")
                out[item["name"]] = value
        return out
    return {}


def _coerce_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    """Safely coerce unknown input to bounded integer."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, parsed))


def _safe_str_dict(value: Any) -> dict[str, str] | None:
    """Normalize a dictionary to `dict[str, str]`, otherwise return `None`."""

    if not isinstance(value, dict):
        return None
    normalized: dict[str, str] = {}
    for key, item in value.items():
        normalized[str(key)] = str(item)
    return normalized
