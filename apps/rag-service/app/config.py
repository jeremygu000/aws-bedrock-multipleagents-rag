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
            f"postgresql://{user}:{encoded_password}@{self.db_host}:{self.db_port}/{self.db_name}"
            f"?sslmode={self.db_ssl_mode}&connect_timeout={self.db_connect_timeout_s}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached settings instance."""

    return Settings()
