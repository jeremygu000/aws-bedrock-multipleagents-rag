"""Database repository for sparse/hybrid retrieval using SQLAlchemy Core.

Design choice:
- Use SQLAlchemy Core to make complex SQL (CTE + RRF fusion) readable and composable.
- Keep execution close to SQL semantics for predictable PostgreSQL behavior.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from sqlalchemy import (
    JSON,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    bindparam,
    create_engine,
    func,
    literal,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.engine import Engine

from .config import Settings
from .models import RetrieveRequest
from .secrets import resolve_db_password

# Lightweight table metadata used by SQLAlchemy Core query builders.
# We only define columns required by retrieval queries to keep the model concise.
metadata = MetaData()

kb_documents = Table(
    "kb_documents",
    metadata,
    Column("doc_id", PGUUID(as_uuid=True)),
    Column("category", String),
    Column("lang", String),
    Column("source_type", String),
)

kb_chunks = Table(
    "kb_chunks",
    metadata,
    Column("chunk_id", PGUUID(as_uuid=True)),
    Column("doc_id", PGUUID(as_uuid=True)),
    Column("chunk_text", Text),
    Column("tsv", Text),
    Column("citation_url", Text),
    Column("citation_title", Text),
    Column("citation_year", Integer),
    Column("citation_month", Integer),
    Column("page_start", Integer),
    Column("page_end", Integer),
    Column("section_id", Text),
    Column("anchor_id", Text),
    Column("metadata", JSON),
    Column("embedding", Text),
)


def _to_vector_literal(values: list[float]) -> str:
    """Convert a Python float list to PostgreSQL pgvector literal format."""

    return "[" + ",".join(f"{value:.9f}" for value in values) + "]"


def _jsonb_to_dict(value: Any) -> dict[str, Any]:
    """Normalize JSON-like DB values into a dictionary.

    This accepts dict/json-string/None and always returns a dictionary
    so response mapping code stays simple and safe.
    """

    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _build_filters(request: RetrieveRequest) -> tuple[list[Any], dict[str, Any]]:
    """Build reusable SQLAlchemy filter clauses and named bind parameters.

    Returns:
        A tuple of:
        - SQLAlchemy boolean expressions for `WHERE` clauses.
        - Named parameters dictionary used by SQLAlchemy execution.
    """

    clauses: list[Any] = []
    params: dict[str, Any] = {}
    filters = request.filters

    if request.require_strict_citation:
        # Enforce hard citation requirements before answer synthesis.
        clauses.extend(
            [
                kb_chunks.c.citation_url.is_not(None),
                kb_chunks.c.citation_title.is_not(None),
                kb_chunks.c.citation_year.is_not(None),
                kb_chunks.c.citation_month.is_not(None),
                (
                    kb_chunks.c.page_start.is_not(None)
                    | kb_chunks.c.section_id.is_not(None)
                    | kb_chunks.c.anchor_id.is_not(None)
                ),
            ]
        )

    if filters.category:
        params["filter_category"] = filters.category
        clauses.append(kb_documents.c.category == bindparam("filter_category"))

    if filters.lang:
        params["filter_lang"] = filters.lang
        clauses.append(kb_documents.c.lang == bindparam("filter_lang"))

    if filters.source_type:
        params["filter_source_type"] = filters.source_type
        clauses.append(kb_documents.c.source_type == bindparam("filter_source_type"))

    if filters.citation_year_from is not None:
        params["filter_citation_year_from"] = filters.citation_year_from
        clauses.append(kb_chunks.c.citation_year >= bindparam("filter_citation_year_from"))

    if filters.citation_year_to is not None:
        params["filter_citation_year_to"] = filters.citation_year_to
        clauses.append(kb_chunks.c.citation_year <= bindparam("filter_citation_year_to"))

    if filters.citation_month is not None:
        params["filter_citation_month"] = filters.citation_month
        clauses.append(kb_chunks.c.citation_month == bindparam("filter_citation_month"))

    return clauses, params


class PostgresRepository:
    """Repository that executes retrieval queries against PostgreSQL.

    The class lazily creates an SQLAlchemy engine and caches it for reuse.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize repository with immutable runtime settings."""

        self._settings = settings
        self._engine: Engine | None = None

    def retrieve(self, request: RetrieveRequest) -> list[dict[str, Any]]:
        """Dispatch retrieval to sparse-only or hybrid mode based on request payload."""

        filter_clauses, filter_params = _build_filters(request)
        if request.query_embedding:
            return self._retrieve_hybrid(request, filter_clauses, filter_params)
        return self._retrieve_sparse(request, filter_clauses, filter_params)

    def _retrieve_sparse(
        self,
        request: RetrieveRequest,
        filter_clauses: list[Any],
        filter_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run sparse retrieval using PostgreSQL full-text ranking and RRF scoring."""

        # Build a reusable tsquery expression from the user query.
        ts_query = func.plainto_tsquery("simple", bindparam("query_text"))
        sparse_rank = func.row_number().over(
            order_by=func.ts_rank_cd(kb_chunks.c.tsv, ts_query).desc()
        )

        # Candidate CTE: find top sparse matches before final projection.
        sparse_candidates = (
            select(
                kb_chunks.c.chunk_id,
                sparse_rank.label("sparse_rank"),
            )
            .select_from(kb_chunks.join(kb_documents, kb_documents.c.doc_id == kb_chunks.c.doc_id))
            .where(kb_chunks.c.tsv.op("@@")(ts_query), *filter_clauses)
            .limit(bindparam("k_sparse"))
            .cte("sparse_candidates")
        )

        # Apply reciprocal-rank-style score to sparse candidate rank.
        score = (literal(1.0) / (bindparam("rrf_k") + sparse_candidates.c.sparse_rank)).label(
            "score"
        )
        statement = (
            select(
                kb_chunks.c.chunk_id,
                kb_chunks.c.doc_id,
                kb_chunks.c.chunk_text,
                score,
                kb_documents.c.category,
                kb_documents.c.lang,
                kb_documents.c.source_type,
                kb_chunks.c.metadata,
                kb_chunks.c.citation_url,
                kb_chunks.c.citation_title,
                kb_chunks.c.citation_year,
                kb_chunks.c.citation_month,
                kb_chunks.c.page_start,
                kb_chunks.c.page_end,
                kb_chunks.c.section_id,
                kb_chunks.c.anchor_id,
            )
            .select_from(
                sparse_candidates.join(
                    kb_chunks, kb_chunks.c.chunk_id == sparse_candidates.c.chunk_id
                ).join(kb_documents, kb_documents.c.doc_id == kb_chunks.c.doc_id)
            )
            .order_by(score.desc())
            .limit(bindparam("k_final"))
        )

        rrf_k = request.rrf_k or self._settings.default_rrf_k
        # Bind values separately from SQL construction for safety and readability.
        params: dict[str, Any] = {
            "query_text": request.query,
            "k_sparse": request.k_sparse,
            "rrf_k": rrf_k,
            "k_final": request.k_final,
            **filter_params,
        }
        return self._run_statement(statement, params)

    def _retrieve_hybrid(
        self,
        request: RetrieveRequest,
        filter_clauses: list[Any],
        filter_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run hybrid retrieval by fusing sparse and dense candidates via RRF."""

        if not request.query_embedding:
            raise ValueError("query_embedding is required for hybrid retrieval")
        if len(request.query_embedding) != self._settings.embedding_dimensions:
            raise ValueError(
                "query_embedding length mismatch: "
                f"expected {self._settings.embedding_dimensions}, got {len(request.query_embedding)}"
            )

        # Convert embedding to pgvector literal consumed by `<=>` distance operator.
        vector_literal = _to_vector_literal(request.query_embedding)
        ts_query = func.plainto_tsquery("simple", bindparam("query_text"))
        sparse_rank = func.row_number().over(
            order_by=func.ts_rank_cd(kb_chunks.c.tsv, ts_query).desc()
        )

        # Sparse candidate set.
        sparse_candidates = (
            select(
                kb_chunks.c.chunk_id,
                sparse_rank.label("sparse_rank"),
            )
            .select_from(kb_chunks.join(kb_documents, kb_documents.c.doc_id == kb_chunks.c.doc_id))
            .where(kb_chunks.c.tsv.op("@@")(ts_query), *filter_clauses)
            .limit(bindparam("k_sparse"))
            .cte("sparse_candidates")
        )

        # Dense candidate set ordered by vector distance.
        query_vector = text("(:query_vector)::vector")
        dense_distance = kb_chunks.c.embedding.op("<=>")(query_vector)
        dense_rank = func.row_number().over(order_by=dense_distance)

        dense_candidates = (
            select(
                kb_chunks.c.chunk_id,
                dense_rank.label("dense_rank"),
            )
            .select_from(kb_chunks.join(kb_documents, kb_documents.c.doc_id == kb_chunks.c.doc_id))
            .where(*filter_clauses)
            .order_by(dense_distance)
            .limit(bindparam("k_dense"))
            .cte("dense_candidates")
        )

        # Full outer join keeps hits that appear in either sparse or dense set.
        fused = (
            select(
                func.coalesce(sparse_candidates.c.chunk_id, dense_candidates.c.chunk_id).label(
                    "chunk_id"
                ),
                (
                    func.coalesce(
                        literal(1.0) / (bindparam("rrf_k") + sparse_candidates.c.sparse_rank),
                        literal(0.0),
                    )
                    + func.coalesce(
                        literal(1.0) / (bindparam("rrf_k") + dense_candidates.c.dense_rank),
                        literal(0.0),
                    )
                ).label("rrf_score"),
            )
            .select_from(
                sparse_candidates.join(
                    dense_candidates,
                    dense_candidates.c.chunk_id == sparse_candidates.c.chunk_id,
                    isouter=True,
                    full=True,
                )
            )
            .cte("fused")
        )

        # Final projection joins fused ids back to chunk/document payload fields.
        statement = (
            select(
                kb_chunks.c.chunk_id,
                kb_chunks.c.doc_id,
                kb_chunks.c.chunk_text,
                fused.c.rrf_score.label("score"),
                kb_documents.c.category,
                kb_documents.c.lang,
                kb_documents.c.source_type,
                kb_chunks.c.metadata,
                kb_chunks.c.citation_url,
                kb_chunks.c.citation_title,
                kb_chunks.c.citation_year,
                kb_chunks.c.citation_month,
                kb_chunks.c.page_start,
                kb_chunks.c.page_end,
                kb_chunks.c.section_id,
                kb_chunks.c.anchor_id,
            )
            .select_from(
                fused.join(kb_chunks, kb_chunks.c.chunk_id == fused.c.chunk_id).join(
                    kb_documents, kb_documents.c.doc_id == kb_chunks.c.doc_id
                )
            )
            .order_by(fused.c.rrf_score.desc())
            .limit(bindparam("k_final"))
        )

        rrf_k = request.rrf_k or self._settings.default_rrf_k
        params: dict[str, Any] = {
            "query_text": request.query,
            "query_vector": vector_literal,
            "k_sparse": request.k_sparse,
            "k_dense": request.k_dense,
            "rrf_k": rrf_k,
            "k_final": request.k_final,
            **filter_params,
        }
        return self._run_statement(statement, params)

    def _run_statement(self, statement: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute a SQLAlchemy statement and map rows to API response structure."""

        engine = self._get_engine()
        with engine.connect() as conn:
            rows = conn.execute(statement, params).mappings().all()

        # Normalize DB types and ensure citation structure is always complete.
        normalized: list[dict[str, Any]] = []
        for row in rows:
            metadata = _jsonb_to_dict(row.get("metadata"))
            normalized.append(
                {
                    "chunk_id": str(_normalize_uuid(row["chunk_id"])),
                    "doc_id": str(_normalize_uuid(row["doc_id"])),
                    "chunk_text": row["chunk_text"],
                    "score": float(row["score"]),
                    "category": row["category"],
                    "lang": row["lang"],
                    "source_type": row["source_type"],
                    "metadata": metadata,
                    "citation": {
                        "url": row["citation_url"],
                        "title": row["citation_title"],
                        "year": int(row["citation_year"]),
                        "month": int(row["citation_month"]),
                        "page_start": row["page_start"],
                        "page_end": row["page_end"],
                        "section_id": row["section_id"],
                        "anchor_id": row["anchor_id"],
                    },
                }
            )
        return normalized

    def _get_engine(self) -> Engine:
        """Lazily construct and cache SQLAlchemy engine for connection reuse."""

        if self._engine is not None:
            return self._engine

        # Resolve password with secret-first strategy.
        db_password = resolve_db_password(self._settings)
        if not db_password:
            raise ValueError(
                "DB password is not configured. Set RAG_DB_PASSWORD or "
                "RAG_DB_PASSWORD_SECRET_ARN."
            )

        # `pool_pre_ping=True` helps recover stale Lambda execution environment connections.
        self._engine = create_engine(
            self._settings.build_db_dsn(db_password),
            pool_pre_ping=True,
        )
        return self._engine


def _normalize_uuid(value: Any) -> UUID:
    """Normalize UUID-like DB values into a standard `UUID` object."""

    if isinstance(value, UUID):
        return value
    return UUID(str(value))
