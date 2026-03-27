"""AWS Lambda entrypoint for the Bedrock `rag_search` action group."""

from __future__ import annotations

from typing import Any

from app.answer_generator import (
    BedrockConverseAnswerGenerator,
    QwenAnswerGenerator,
    RoutedAnswerGenerator,
)
from app.bedrock_action import handle_rag_action
from app.config import get_settings
from app.query_processing import QueryProcessor
from app.qwen_client import QwenClient
from app.repository import PostgresRepository
from app.secrets import resolve_db_password
from app.workflow import RagWorkflow

settings = get_settings()
repository = PostgresRepository(settings)
qwen_client = QwenClient(settings)
query_processor = QueryProcessor(settings=settings, qwen_client=qwen_client)
answer_generator = RoutedAnswerGenerator(
    bedrock_generator=BedrockConverseAnswerGenerator(settings),
    qwen_generator=QwenAnswerGenerator(qwen_client),
)
workflow = RagWorkflow(
    settings=settings,
    repository=repository,
    query_processor=query_processor,
    answer_generator=answer_generator,
)


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    """Handle Bedrock action requests in Lambda runtime.

    This function performs a quick configuration guard before executing retrieval.
    It always returns a Bedrock-compatible response envelope.
    """

    try:
        # Fail soft when DB credentials are not configured in this environment.
        if not resolve_db_password(settings):
            return {
                "messageVersion": str(event.get("messageVersion", "1.0")),
                "response": {
                    "actionGroup": str(event.get("actionGroup", "rag_search")),
                    "apiPath": str(event.get("apiPath", "/rag_search")),
                    "httpMethod": str(event.get("httpMethod", "POST")),
                    "httpStatusCode": 200,
                    "responseBody": {
                        "application/json": {
                            "body": (
                                '{"answer":"RAG database is not configured yet. '
                                'Set RAG_DB_PASSWORD_SECRET_ARN or RAG_DB_PASSWORD for RagSearchFn.","citations":[]}'
                            ),
                        }
                    },
                },
                "sessionAttributes": event.get("sessionAttributes"),
                "promptSessionAttributes": event.get("promptSessionAttributes"),
            }

        # Delegate core business logic to LangGraph workflow adapter.
        return handle_rag_action(event, workflow)
    except Exception:
        # Do not leak internal details to Bedrock action clients.
        return {
            "messageVersion": str(event.get("messageVersion", "1.0")),
            "response": {
                "actionGroup": str(event.get("actionGroup", "rag_search")),
                "apiPath": str(event.get("apiPath", "/rag_search")),
                "httpMethod": str(event.get("httpMethod", "POST")),
                "httpStatusCode": 500,
                "responseBody": {
                    "application/json": {
                        "body": '{"answer":"rag_search failed","citations":[]}',
                    }
                },
            },
            "sessionAttributes": event.get("sessionAttributes"),
            "promptSessionAttributes": event.get("promptSessionAttributes"),
        }
