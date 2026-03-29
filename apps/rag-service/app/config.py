"""Application configuration for the Python Hybrid RAG service.

This module defines a single settings object sourced from environment variables.
It supports both local development defaults and cloud runtime fallbacks.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import quote_plus

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed runtime settings used by API and Lambda paths.

    Notes:
    - `RAG_*` variables are preferred.
    - Some fields fall back to existing `RDS_*` variables for compatibility.
    - Secret ARN based password resolution is handled separately in `secrets.py`.
    """

    app_name: str = Field(default="hybrid-rag-service", validation_alias="RAG_APP_NAME")
    app_host: str = Field(default="0.0.0.0", validation_alias="RAG_APP_HOST")
    app_port: int = Field(default=8080, validation_alias="RAG_APP_PORT")
    app_log_level: str = Field(default="info", validation_alias="RAG_APP_LOG_LEVEL")

    db_host: str = Field(
        default="localhost",
        validation_alias=AliasChoices("RAG_DB_HOST", "RDS_HOST"),
    )
    db_port: int = Field(
        default=5432,
        validation_alias=AliasChoices("RAG_DB_PORT", "RDS_PORT"),
    )
    db_name: str = Field(
        default="postgres",
        validation_alias=AliasChoices("RAG_DB_NAME", "RDS_DB_NAME"),
    )
    db_user: str = Field(
        default="postgres",
        validation_alias=AliasChoices("RAG_DB_USER", "RDS_MASTER_USERNAME"),
    )
    db_password: str = Field(
        default="",
        validation_alias=AliasChoices("RAG_DB_PASSWORD", "RDS_MASTER_PASSWORD"),
    )
    db_password_secret_arn: str = Field(
        default="",
        validation_alias=AliasChoices("RAG_DB_PASSWORD_SECRET_ARN", "RDS_MASTER_SECRET_ARN"),
    )
    db_password_secret_json_key: str = Field(
        default="password",
        validation_alias="RAG_DB_PASSWORD_SECRET_JSON_KEY",
    )
    db_ssl_mode: str = Field(default="require", validation_alias="RAG_DB_SSLMODE")
    db_connect_timeout_s: int = Field(default=10, validation_alias="RAG_DB_CONNECT_TIMEOUT_S")
    aws_region: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RAG_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"),
    )

    embedding_dimensions: int = Field(default=1024, validation_alias="RAG_EMBED_DIM")
    sparse_backend: Literal["opensearch", "postgres"] = Field(
        default="opensearch",
        validation_alias="RAG_SPARSE_BACKEND",
    )
    opensearch_endpoint: str = Field(default="", validation_alias="RAG_OPENSEARCH_ENDPOINT")
    opensearch_index: str = Field(default="kb_chunks", validation_alias="RAG_OPENSEARCH_INDEX")
    opensearch_timeout_s: int = Field(default=10, validation_alias="RAG_OPENSEARCH_TIMEOUT_S")
    opensearch_use_sigv4: bool = Field(default=True, validation_alias="RAG_OPENSEARCH_USE_SIGV4")
    default_rrf_k: int = Field(default=60, validation_alias="RAG_RRF_K")
    answer_model_id: str = Field(
        default="amazon.nova-lite-v1:0",
        validation_alias=AliasChoices("RAG_ANSWER_MODEL_ID", "FOUNDATION_MODEL_ID"),
    )
    answer_max_tokens: int = Field(default=500, validation_alias="RAG_ANSWER_MAX_TOKENS")
    answer_temperature: float = Field(default=0.05, validation_alias="RAG_ANSWER_TEMPERATURE")

    qwen_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
    )
    qwen_api_key_secret_arn: str = Field(
        default="",
        validation_alias="QWEN_API_KEY_SECRET_ARN",
    )
    qwen_api_key_secret_key: str = Field(
        default="DASHSCOPE_API_KEY",
        validation_alias="QWEN_API_KEY_SECRET_KEY",
    )
    qwen_model_id: str = Field(
        default="qwen-plus",
        validation_alias=AliasChoices("QWEN_MODEL_ID", "LLM_MODEL"),
    )
    qwen_embedding_model_id: str = Field(
        default="text-embedding-v3",
        validation_alias="QWEN_EMBEDDING_MODEL_ID",
    )
    qwen_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias="QWEN_BASE_URL",
    )
    qwen_max_tokens: int = Field(default=500, validation_alias="QWEN_MAX_TOKENS")
    qwen_temperature: float = Field(default=0.0, validation_alias="QWEN_TEMPERATURE")

    route_min_hits: int = Field(default=3, validation_alias="RAG_ROUTE_MIN_HITS")
    route_top_score_threshold: float = Field(
        default=0.015,
        validation_alias="RAG_ROUTE_TOP_SCORE_THRESHOLD",
    )
    route_complex_query_token_threshold: int = Field(
        default=18,
        validation_alias="RAG_ROUTE_COMPLEX_QUERY_TOKEN_THRESHOLD",
    )
    enable_query_rewrite: bool = Field(
        default=True,
        validation_alias="RAG_ENABLE_QUERY_REWRITE",
    )
    enable_hybrid_retrieval: bool = Field(
        default=True,
        validation_alias="RAG_ENABLE_HYBRID_RETRIEVAL",
    )
    enable_keyword_extraction: bool = Field(
        default=True,
        validation_alias="RAG_ENABLE_KEYWORD_EXTRACTION",
    )
    enable_reranking: bool = Field(
        default=True,
        validation_alias="RAG_ENABLE_RERANKING",
    )
    enable_intent_aware_prompts: bool = Field(
        default=True,
        validation_alias="RAG_ENABLE_INTENT_AWARE_PROMPTS",
    )
    enable_relevance_scores_in_evidence: bool = Field(
        default=True,
        validation_alias="RAG_ENABLE_RELEVANCE_SCORES_IN_EVIDENCE",
    )
    answer_evidence_max_chars: int = Field(
        default=1200,
        ge=200,
        le=5000,
        validation_alias="RAG_ANSWER_EVIDENCE_MAX_CHARS",
    )
    rerank_candidate_count: int = Field(
        default=20,
        ge=1,
        le=100,
        validation_alias="RAG_RERANK_CANDIDATE_COUNT",
    )
    rerank_max_tokens: int = Field(
        default=30000,
        ge=1000,
        validation_alias="RAG_RERANK_MAX_TOKENS",
    )

    # --- Entity extraction settings ---
    enable_entity_extraction: bool = Field(
        default=False,
        validation_alias="RAG_ENABLE_ENTITY_EXTRACTION",
    )
    entity_extraction_max_retries: int = Field(
        default=1,
        ge=0,
        le=3,
        validation_alias="RAG_ENTITY_EXTRACTION_MAX_RETRIES",
    )
    entity_summary_max_tokens: int = Field(
        default=500,
        ge=50,
        le=5000,
        validation_alias="RAG_ENTITY_SUMMARY_MAX_TOKENS",
        description="Token threshold above which merged entity descriptions are LLM-summarized",
    )
    entity_extraction_embed_batch_size: int = Field(
        default=20,
        ge=1,
        le=100,
        validation_alias="RAG_ENTITY_EXTRACTION_EMBED_BATCH_SIZE",
        description="Batch size for embedding entity/relation descriptions during ingestion",
    )

    # --- Neo4j graph database settings ---
    enable_neo4j: bool = Field(
        default=False,
        validation_alias="RAG_ENABLE_NEO4J",
        description="Feature flag to enable Neo4j graph storage (Phase 2.3+)",
    )
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        validation_alias="RAG_NEO4J_URI",
    )
    neo4j_username: str = Field(
        default="neo4j",
        validation_alias="RAG_NEO4J_USERNAME",
    )
    neo4j_password: str = Field(
        default="",
        validation_alias="RAG_NEO4J_PASSWORD",
    )
    neo4j_password_secret_arn: str = Field(
        default="",
        validation_alias="RAG_NEO4J_PASSWORD_SECRET_ARN",
    )
    neo4j_database: str = Field(
        default="neo4j",
        validation_alias="RAG_NEO4J_DATABASE",
    )

    # --- Graph retrieval settings (Phase 3) ---
    enable_graph_retrieval: bool = Field(
        default=False,
        validation_alias="RAG_ENABLE_GRAPH_RETRIEVAL",
        description="Feature flag to enable graph-enhanced retrieval (Phase 3+)",
    )
    retrieval_mode: str = Field(
        default="mix",
        validation_alias="RAG_RETRIEVAL_MODE",
        description="Graph retrieval mode: chunks_only, graph_only, mix",
    )
    graph_top_k_entities: int = Field(
        default=10,
        ge=1,
        le=50,
        validation_alias="RAG_GRAPH_TOP_K_ENTITIES",
    )
    graph_top_k_relations: int = Field(
        default=10,
        ge=1,
        le=50,
        validation_alias="RAG_GRAPH_TOP_K_RELATIONS",
    )
    graph_neighbor_depth: int = Field(
        default=1,
        ge=0,
        le=3,
        validation_alias="RAG_GRAPH_NEIGHBOR_DEPTH",
        description="How many hops to traverse for entity neighbors in local retrieval",
    )
    graph_retrieval_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        validation_alias="RAG_GRAPH_RETRIEVAL_WEIGHT",
        description="Weight for graph-derived hits during RRF fusion (traditional weight = 1 - this)",
    )

    # --- Query result cache settings (L2) ---
    enable_query_cache: bool = Field(
        default=False,
        validation_alias="RAG_ENABLE_QUERY_CACHE",
        description="Feature flag to enable L2 query result caching",
    )
    query_cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        validation_alias="RAG_QUERY_CACHE_TTL_HOURS",
        description="TTL in hours for cached query results",
    )
    query_cache_similarity_threshold: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        validation_alias="RAG_QUERY_CACHE_SIMILARITY_THRESHOLD",
        description="Cosine similarity threshold for cache hit (>= this value)",
    )

    # --- Ingestion settings ---
    s3_bucket: str = Field(default="", validation_alias="RAG_S3_BUCKET")
    max_upload_size_mb: int = Field(default=50, ge=1, validation_alias="RAG_MAX_UPLOAD_SIZE_MB")
    ingestion_chunk_size: int = Field(default=512, ge=50, validation_alias="RAG_CHUNK_SIZE")
    ingestion_chunk_overlap: int = Field(default=64, ge=0, validation_alias="RAG_CHUNK_OVERLAP")
    ingestion_chunk_min_size: int = Field(default=50, ge=1, validation_alias="RAG_CHUNK_MIN_SIZE")
    ingestion_embed_batch_size: int = Field(
        default=20, ge=1, le=100, validation_alias="RAG_EMBED_BATCH_SIZE"
    )
    ingestion_queue_url: str = Field(default="", validation_alias="RAG_INGESTION_QUEUE_URL")

    # --- Evaluation framework settings (Phase 4) ---
    enable_eval_tracing: bool = Field(
        default=False,
        validation_alias="RAG_ENABLE_EVAL_TRACING",
        description="Enable evaluation tracing for production monitoring",
    )
    eval_faithfulness_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        validation_alias="RAG_EVAL_FAITHFULNESS_THRESHOLD",
    )
    eval_answer_relevancy_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        validation_alias="RAG_EVAL_ANSWER_RELEVANCY_THRESHOLD",
    )
    eval_context_precision_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        validation_alias="RAG_EVAL_CONTEXT_PRECISION_THRESHOLD",
    )
    eval_context_recall_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        validation_alias="RAG_EVAL_CONTEXT_RECALL_THRESHOLD",
    )
    eval_hallucination_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        validation_alias="RAG_EVAL_HALLUCINATION_THRESHOLD",
        description="DeepEval hallucination metric threshold (lower = stricter)",
    )

    # --- Streaming SSE settings (Phase 6) ---
    enable_streaming: bool = Field(
        default=True,
        validation_alias="RAG_ENABLE_STREAMING",
        description="Feature flag to enable GET /retrieve/stream SSE endpoint",
    )

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False,
    )

    def build_db_dsn(self, password: str) -> str:
        """Build a PostgreSQL DSN string with encoded credentials.

        Args:
            password: Plaintext database password resolved at runtime.

        Returns:
            SQLAlchemy/psycopg compatible PostgreSQL DSN.
        """

        # Encode user/password to safely handle special URL characters.
        user = quote_plus(self.db_user)
        encoded_password = quote_plus(password)
        return (
            f"postgresql+psycopg://{user}:{encoded_password}@{self.db_host}:{self.db_port}/{self.db_name}"
            f"?sslmode={self.db_ssl_mode}&connect_timeout={self.db_connect_timeout_s}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached settings instance."""

    return Settings()
