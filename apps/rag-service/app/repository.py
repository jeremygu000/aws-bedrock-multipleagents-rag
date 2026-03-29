"""Database repository for sparse/hybrid retrieval using SQLAlchemy Core.

Design choice:
- Use SQLAlchemy Core to make complex SQL (CTE + RRF fusion) readable and composable.
- Keep execution close to SQL semantics for predictable PostgreSQL behavior.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection
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
        self._opensearch_client: OpenSearch | None = None

    def retrieve(self, request: RetrieveRequest) -> list[dict[str, Any]]:
        """Dispatch retrieval to sparse-only or hybrid mode based on request payload."""

        filter_clauses, filter_params = _build_filters(request)
        if self._should_use_opensearch():
            try:
                if request.query_embedding:
                    return self._retrieve_hybrid_with_opensearch(
                        request, filter_clauses, filter_params
                    )
                sparse_hits = self._retrieve_sparse_opensearch(request)
                if sparse_hits:
                    return sparse_hits
            except Exception:
                # Keep service resilient by falling back to PostgreSQL retrieval path.
                pass

        if request.query_embedding:
            return self._retrieve_hybrid(request, filter_clauses, filter_params)
        return self._retrieve_sparse(request, filter_clauses, filter_params)

    def _should_use_opensearch(self) -> bool:
        """Return whether OpenSearch sparse retrieval should be used."""

        return self._settings.sparse_backend == "opensearch" and bool(
            self._settings.opensearch_endpoint.strip()
        )

    def _retrieve_sparse_opensearch(self, request: RetrieveRequest) -> list[dict[str, Any]]:
        """Run sparse retrieval on OpenSearch (BM25) with strict-citation normalization."""

        candidates = self._retrieve_sparse_opensearch_candidates(request)
        return candidates[: request.k_final]

    def _retrieve_sparse_opensearch_candidates(
        self, request: RetrieveRequest
    ) -> list[dict[str, Any]]:
        """Fetch sparse candidate hits from OpenSearch with rank-based scoring."""

        client = self._get_opensearch_client()
        body = self._build_opensearch_query(request, size=request.k_sparse)
        response = client.search(index=self._settings.opensearch_index, body=body)
        response_hits = response.get("hits", {}).get("hits", [])
        if not isinstance(response_hits, list):
            return []

        rrf_k = request.rrf_k or self._settings.default_rrf_k
        normalized: list[dict[str, Any]] = []
        for rank, hit in enumerate(response_hits, start=1):
            mapped = _normalize_opensearch_hit(hit, request.require_strict_citation)
            if not mapped:
                continue
            mapped["score"] = 1.0 / (rrf_k + rank)
            normalized.append(mapped)
        return normalized

    def _retrieve_hybrid_with_opensearch(
        self,
        request: RetrieveRequest,
        filter_clauses: list[Any],
        filter_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Fuse OpenSearch sparse ranks with PostgreSQL dense ranks using RRF."""

        sparse_hits = self._retrieve_sparse_opensearch_candidates(request)
        dense_hits = self._retrieve_dense_only(request, filter_clauses, filter_params)
        fused_hits = self._fuse_ranked_hits(
            sparse_hits=sparse_hits,
            dense_hits=dense_hits,
            k_final=request.k_final,
            rrf_k=request.rrf_k or self._settings.default_rrf_k,
        )
        if fused_hits:
            return fused_hits
        return sparse_hits[: request.k_final]

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
        query_vector = text("CAST(:query_vector AS vector)")
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

    def _retrieve_dense_only(
        self,
        request: RetrieveRequest,
        filter_clauses: list[Any],
        filter_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run dense-only retrieval on PostgreSQL for OpenSearch hybrid fusion."""

        if not request.query_embedding:
            raise ValueError("query_embedding is required for dense retrieval")
        if len(request.query_embedding) != self._settings.embedding_dimensions:
            raise ValueError(
                "query_embedding length mismatch: "
                f"expected {self._settings.embedding_dimensions}, got {len(request.query_embedding)}"
            )

        vector_literal = _to_vector_literal(request.query_embedding)
        query_vector = text("CAST(:query_vector AS vector)")
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

        score = (literal(1.0) / (bindparam("rrf_k") + dense_candidates.c.dense_rank)).label("score")
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
                dense_candidates.join(
                    kb_chunks, kb_chunks.c.chunk_id == dense_candidates.c.chunk_id
                ).join(kb_documents, kb_documents.c.doc_id == kb_chunks.c.doc_id)
            )
            .order_by(score.desc())
            .limit(bindparam("k_dense"))
        )

        rrf_k = request.rrf_k or self._settings.default_rrf_k
        params: dict[str, Any] = {
            "query_vector": vector_literal,
            "k_dense": request.k_dense,
            "rrf_k": rrf_k,
            **filter_params,
        }
        return self._run_statement(statement, params)

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch full chunk payloads by their IDs for graph→chunk fusion.

        Used by hybrid fusion to convert graph-derived ``source_chunk_ids``
        into concrete retrieval hits that can participate in RRF scoring.

        Args:
            chunk_ids: List of chunk UUID strings to look up.

        Returns:
            List of normalized retrieval hit dicts (same schema as ``retrieve``).
            Missing IDs are silently skipped.
        """

        if not chunk_ids:
            return []

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_ids: list[str] = []
        for cid in chunk_ids:
            if cid not in seen:
                seen.add(cid)
                unique_ids.append(cid)

        statement = (
            select(
                kb_chunks.c.chunk_id,
                kb_documents.c.doc_id,
                kb_chunks.c.chunk_text,
                literal(0.0).label("score"),
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
            .select_from(kb_chunks.join(kb_documents, kb_chunks.c.doc_id == kb_documents.c.doc_id))
            .where(kb_chunks.c.chunk_id.in_(bindparam("chunk_ids", expanding=True)))
        )

        return self._run_statement(statement, {"chunk_ids": unique_ids})

    def _fuse_ranked_hits(
        self,
        sparse_hits: list[dict[str, Any]],
        dense_hits: list[dict[str, Any]],
        k_final: int,
        rrf_k: int,
    ) -> list[dict[str, Any]]:
        """Fuse sparse and dense ranked hits with reciprocal-rank fusion in Python."""

        sparse_rank = {hit["chunk_id"]: index for index, hit in enumerate(sparse_hits, start=1)}
        dense_rank = {hit["chunk_id"]: index for index, hit in enumerate(dense_hits, start=1)}

        payload_by_chunk: dict[str, dict[str, Any]] = {}
        for hit in dense_hits:
            payload_by_chunk[hit["chunk_id"]] = _copy_hit(hit)
        for hit in sparse_hits:
            existing = payload_by_chunk.get(hit["chunk_id"])
            if existing is None:
                payload_by_chunk[hit["chunk_id"]] = _copy_hit(hit)
                continue
            if not existing.get("chunk_text"):
                existing["chunk_text"] = hit.get("chunk_text", "")

        scored: list[dict[str, Any]] = []
        for chunk_id, payload in payload_by_chunk.items():
            score = 0.0
            sparse_position = sparse_rank.get(chunk_id)
            dense_position = dense_rank.get(chunk_id)
            if sparse_position is not None:
                score += 1.0 / (rrf_k + sparse_position)
            if dense_position is not None:
                score += 1.0 / (rrf_k + dense_position)
            payload["score"] = score
            scored.append(payload)

        scored.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return scored[:k_final]

    def _build_opensearch_query(self, request: RetrieveRequest, size: int) -> dict[str, Any]:
        """Build OpenSearch BM25 query with metadata filters and strict citation constraints."""

        filters = _build_opensearch_filters(request)
        bool_query: dict[str, Any] = {
            "must": [
                {
                    "multi_match": {
                        "query": request.query,
                        "fields": [
                            "chunk_text^3",
                            "citation.title^2",
                            "citation_title^2",
                            "metadata.title",
                        ],
                        "operator": "or",
                    }
                }
            ]
        }
        if filters:
            bool_query["filter"] = filters

        return {
            "size": size,
            "query": {"bool": bool_query},
        }

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

    def _get_opensearch_client(self) -> OpenSearch:
        """Lazily construct and cache OpenSearch/Elasticsearch client.

        When ``opensearch_use_sigv4`` is *True* (default) the client authenticates
        with AWS IAM SigV4 — suitable for managed AWS OpenSearch domains.
        When *False* the client connects without authentication — suitable for
        self-hosted Elasticsearch / OpenSearch (e.g. Docker on EC2).
        """

        if self._opensearch_client is not None:
            return self._opensearch_client
        if not self._settings.opensearch_endpoint.strip():
            raise ValueError("RAG_OPENSEARCH_ENDPOINT is not configured.")

        endpoint = self._settings.opensearch_endpoint.strip()
        parsed = urlparse(endpoint if "://" in endpoint else f"https://{endpoint}")
        if not parsed.hostname:
            raise ValueError("Invalid OpenSearch endpoint.")

        use_ssl = parsed.scheme != "http"
        default_port = 443 if use_ssl else 9200

        kwargs: dict[str, object] = {
            "hosts": [{"host": parsed.hostname, "port": parsed.port or default_port}],
            "use_ssl": use_ssl,
            "verify_certs": use_ssl,
            "timeout": self._settings.opensearch_timeout_s,
            "connection_class": RequestsHttpConnection,
        }

        if self._settings.opensearch_use_sigv4:
            if not self._settings.aws_region:
                raise ValueError(
                    "AWS region is not configured. "
                    "Set RAG_AWS_REGION, AWS_REGION, or AWS_DEFAULT_REGION."
                )
            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials is None:
                raise ValueError("AWS credentials are not available for OpenSearch access.")
            kwargs["http_auth"] = AWSV4SignerAuth(credentials, self._settings.aws_region, "es")

        self._opensearch_client = OpenSearch(**kwargs)  # type: ignore[arg-type]
        return self._opensearch_client

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


def _normalize_opensearch_hit(
    hit: Any,
    require_strict_citation: bool,
) -> dict[str, Any] | None:
    """Normalize one OpenSearch hit into retrieval payload schema."""

    if not isinstance(hit, dict):
        return None
    source = hit.get("_source", {})
    if not isinstance(source, dict):
        return None

    citation_object = source.get("citation")
    citation_nested = citation_object if isinstance(citation_object, dict) else {}

    citation_url = citation_nested.get("url") or source.get("citation_url")
    citation_title = citation_nested.get("title") or source.get("citation_title")
    citation_year = _to_int(citation_nested.get("year") or source.get("citation_year"))
    citation_month = _to_int(citation_nested.get("month") or source.get("citation_month"))
    page_start = _to_int(citation_nested.get("page_start") or source.get("page_start"))
    page_end = _to_int(citation_nested.get("page_end") or source.get("page_end"))
    section_id = citation_nested.get("section_id") or source.get("section_id")
    anchor_id = citation_nested.get("anchor_id") or source.get("anchor_id")

    if require_strict_citation:
        if (
            not citation_url
            or not citation_title
            or citation_year is None
            or citation_month is None
        ):
            return None
        if page_start is None and section_id is None and anchor_id is None:
            return None

    metadata = source.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    chunk_id = str(source.get("chunk_id") or hit.get("_id") or "")
    doc_id = str(source.get("doc_id") or metadata_dict.get("doc_id") or chunk_id)
    if not chunk_id:
        return None

    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "chunk_text": str(source.get("chunk_text") or ""),
        "score": float(hit.get("_score") or 0.0),
        "category": str(source.get("category") or metadata_dict.get("category") or "unknown"),
        "lang": str(source.get("lang") or metadata_dict.get("lang") or "unknown"),
        "source_type": str(
            source.get("source_type") or metadata_dict.get("source_type") or "crawler"
        ),
        "metadata": metadata_dict,
        "citation": {
            "url": citation_url,
            "title": citation_title,
            "year": citation_year if citation_year is not None else 1970,
            "month": citation_month if citation_month is not None else 1,
            "page_start": page_start,
            "page_end": page_end,
            "section_id": section_id,
            "anchor_id": anchor_id,
        },
    }


def _build_opensearch_filters(request: RetrieveRequest) -> list[dict[str, Any]]:
    """Build OpenSearch bool filters aligned with retrieval request filters."""

    filters: list[dict[str, Any]] = []
    request_filters = request.filters

    if request.require_strict_citation:
        filters.extend(
            [
                {
                    "bool": {
                        "should": [
                            {"exists": {"field": "citation.url"}},
                            {"exists": {"field": "citation_url"}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
                {
                    "bool": {
                        "should": [
                            {"exists": {"field": "citation.title"}},
                            {"exists": {"field": "citation_title"}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
                {
                    "bool": {
                        "should": [
                            {"exists": {"field": "citation.year"}},
                            {"exists": {"field": "citation_year"}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
                {
                    "bool": {
                        "should": [
                            {"exists": {"field": "citation.month"}},
                            {"exists": {"field": "citation_month"}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
            ]
        )

    if request_filters.category:
        filters.append(
            {
                "bool": {
                    "should": [
                        {"term": {"category.keyword": request_filters.category}},
                        {"term": {"category": request_filters.category}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )

    if request_filters.lang:
        filters.append(
            {
                "bool": {
                    "should": [
                        {"term": {"lang.keyword": request_filters.lang}},
                        {"term": {"lang": request_filters.lang}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )

    if request_filters.source_type:
        filters.append(
            {
                "bool": {
                    "should": [
                        {"term": {"source_type.keyword": request_filters.source_type}},
                        {"term": {"source_type": request_filters.source_type}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )

    if request_filters.citation_year_from is not None:
        filters.append(
            {
                "bool": {
                    "should": [
                        {"range": {"citation.year": {"gte": request_filters.citation_year_from}}},
                        {"range": {"citation_year": {"gte": request_filters.citation_year_from}}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )

    if request_filters.citation_year_to is not None:
        filters.append(
            {
                "bool": {
                    "should": [
                        {"range": {"citation.year": {"lte": request_filters.citation_year_to}}},
                        {"range": {"citation_year": {"lte": request_filters.citation_year_to}}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )

    if request_filters.citation_month is not None:
        filters.append(
            {
                "bool": {
                    "should": [
                        {"term": {"citation.month": request_filters.citation_month}},
                        {"term": {"citation_month": request_filters.citation_month}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )

    return filters


def _to_int(value: Any) -> int | None:
    """Safely coerce unknown value to integer when possible."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _copy_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Create a safe mutable copy of retrieval hit payload."""

    citation = hit.get("citation")
    metadata = hit.get("metadata")
    return {
        **hit,
        "citation": dict(citation) if isinstance(citation, dict) else {},
        "metadata": dict(metadata) if isinstance(metadata, dict) else {},
    }
