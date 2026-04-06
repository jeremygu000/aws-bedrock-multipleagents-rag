"""AWS Lambda entrypoint for the Bedrock `rag_search` action group."""

from __future__ import annotations

import logging
from typing import Any

from app.answer_generator import (
    BedrockConverseAnswerGenerator,
    QwenAnswerGenerator,
    RoutedAnswerGenerator,
)
from app.bedrock_action import handle_rag_action
from app.config import get_settings
from app.crag import CragQueryRewriter, CragWebSearcher, RetrievalGrader
from app.embedding_factory import get_embedding_client
from app.graph_retriever import GraphRetriever
from app.hyde_retriever import HyDEConfig, HyDERetriever
from app.query_processing import QueryProcessor
from app.qwen_client import QwenClient
from app.repository import PostgresRepository
from app.reranker import LLMReranker
from app.secrets import resolve_db_password, resolve_neo4j_password
from app.workflow import RagWorkflow

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Ensure all app.* loggers emit INFO+ in Lambda (root default is WARNING)
logging.getLogger("app").setLevel(logging.INFO)

settings = get_settings()
repository = PostgresRepository(settings)
qwen_client = QwenClient(settings)
embedding_client = get_embedding_client(settings)
query_processor = QueryProcessor(
    settings=settings, qwen_client=qwen_client, embedding_client=embedding_client
)
answer_generator = RoutedAnswerGenerator(
    bedrock_generator=BedrockConverseAnswerGenerator(settings),
    qwen_generator=QwenAnswerGenerator(qwen_client, settings),
)
reranker = LLMReranker(settings=settings, qwen_client=qwen_client)


def _build_graph_retriever() -> GraphRetriever | None:
    """Build GraphRetriever when graph retrieval is enabled, or return None."""
    if not settings.enable_graph_retrieval:
        return None

    from app.entity_vector_store import EntityVectorStore

    vector_store = EntityVectorStore(settings)
    neo4j_repo = None

    if settings.enable_neo4j:
        from app.graph_repository import Neo4jRepository

        neo4j_password = resolve_neo4j_password(settings)
        neo4j_repo = Neo4jRepository(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=neo4j_password,
            database=settings.neo4j_database,
        )

    return GraphRetriever(
        qwen_client=qwen_client,
        vector_store=vector_store,
        neo4j_repo=neo4j_repo,
        settings=settings,
        embedding_client=embedding_client,
    )


graph_retriever = _build_graph_retriever()

retrieval_grader = None
crag_query_rewriter = None
crag_web_searcher = None
if settings.enable_crag:
    retrieval_grader = RetrievalGrader(settings)
    crag_query_rewriter = CragQueryRewriter(settings)
    crag_web_searcher = CragWebSearcher(settings)

hyde_retriever = None
if settings.enable_hyde:
    hyde_config = HyDEConfig(
        enabled=True,
        temperature=settings.hyde_temperature,
        max_hypothesis_tokens=settings.hyde_max_tokens,
        include_original=settings.hyde_include_original,
        num_hypotheses=settings.hyde_num_hypotheses,
    )
    from app.bedrock_embedding_client import BedrockEmbeddingClient

    bedrock_client = BedrockEmbeddingClient(settings)
    hyde_retriever = HyDERetriever(
        llm_client=qwen_client,
        embedding_model=bedrock_client,
        config=hyde_config,
    )
    logger.info(
        "HyDE retriever initialized: model=%s, num_hypotheses=%d, "
        "temperature=%.2f, max_tokens=%d, include_original=%s, aggregation=%s",
        settings.hyde_model_id,
        settings.hyde_num_hypotheses,
        settings.hyde_temperature,
        settings.hyde_max_tokens,
        settings.hyde_include_original,
        settings.hyde_aggregation,
    )
else:
    logger.info("HyDE disabled (RAG_ENABLE_HYDE=%s)", settings.enable_hyde)

workflow = RagWorkflow(
    settings=settings,
    repository=repository,
    query_processor=query_processor,
    answer_generator=answer_generator,
    reranker=reranker,
    graph_retriever=graph_retriever,
    retrieval_grader=retrieval_grader,
    crag_query_rewriter=crag_query_rewriter,
    crag_web_searcher=crag_web_searcher,
    hyde_retriever=hyde_retriever,
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
        logger.exception("rag_search handler failed")
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
