"""Write-path repository for the document ingestion pipeline.

Handles INSERT/UPDATE/DELETE operations for ingestion runs, documents, and chunks.
Uses SQLAlchemy Core (not ORM) and mirrors the lazy engine/client pattern from repository.py.
"""

from __future__ import annotations

import uuid
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection, helpers
from sqlalchemy import (
    ARRAY,
    JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.engine import Engine

from .config import Settings
from .ingestion_models import ChunkRecord, DocumentRecord
from .secrets import resolve_db_password

# Separate metadata object to avoid table name conflicts with repository.py
metadata_obj = MetaData()

ingestion_runs = Table(
    "ingestion_runs",
    metadata_obj,
    Column("run_id", PGUUID(as_uuid=True), primary_key=True),
    Column("source_type", String, nullable=False),
    Column("status", String, nullable=False),
    Column("config", JSON, nullable=False),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    Column("notes", Text),
)

kb_documents_w = Table(
    "kb_documents",
    metadata_obj,
    Column("doc_id", PGUUID(as_uuid=True), primary_key=True),
    Column("source_type", String, nullable=False),
    Column("source_uri", String, nullable=False),
    Column("title", String, nullable=False),
    Column("lang", String, nullable=False),
    Column("category", String, nullable=False),
    Column("mime_type", String, nullable=False),
    Column("content_hash", String, nullable=False),
    Column("doc_version", String, nullable=False),
    Column("published_year", Integer, nullable=False),
    Column("published_month", Integer, nullable=False),
    Column("author", String),
    Column("tags", ARRAY(String)),
    Column("metadata", JSON),
    Column("run_id", PGUUID(as_uuid=True)),
)

kb_chunks_w = Table(
    "kb_chunks",
    metadata_obj,
    Column("chunk_id", PGUUID(as_uuid=True), primary_key=True),
    Column("doc_id", PGUUID(as_uuid=True), nullable=False),
    Column("doc_version", String, nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("chunk_text", Text, nullable=False),
    Column("token_count", Integer, nullable=False),
    Column("citation_url", Text, nullable=False),
    Column("citation_title", Text, nullable=False),
    Column("citation_year", Integer, nullable=False),
    Column("citation_month", Integer, nullable=False),
    Column("page_start", Integer),
    Column("page_end", Integer),
    Column("section_id", Text),
    Column("anchor_id", Text),
    Column("embedding", Text, nullable=False),  # Inserted as pgvector literal string
    Column("metadata", JSON),
    Column("run_id", PGUUID(as_uuid=True)),
)

_VALID_COMPLETION_STATUSES = {"succeeded", "failed"}


def _to_vector_literal(values: list[float]) -> str:
    """Convert a Python float list to PostgreSQL pgvector literal format."""

    return "[" + ",".join(f"{v:.9f}" for v in values) + "]"


class IngestionRepository:
    """Repository for write-path ingestion operations against PostgreSQL and OpenSearch.

    The class lazily creates an SQLAlchemy engine and OpenSearch client, caching them for reuse.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize repository with immutable runtime settings."""

        self._settings = settings
        self._engine: Engine | None = None
        self._opensearch_client: OpenSearch | None = None

    def create_ingestion_run(self, source_type: str, config: dict[str, Any]) -> UUID:
        """INSERT a new ingestion run with status='running'. Return the generated run_id."""

        run_id = uuid.uuid4()
        stmt = insert(ingestion_runs).values(
            run_id=run_id,
            source_type=source_type,
            status="running",
            config=config,
            started_at=text("now()"),
        )
        engine = self._get_engine()
        with engine.begin() as conn:
            conn.execute(stmt)
        return run_id

    def complete_ingestion_run(self, run_id: UUID, status: str, notes: str | None = None) -> None:
        """UPDATE ingestion_runs with final status and finished_at timestamp.

        Args:
            run_id: The run to update.
            status: Must be 'succeeded' or 'failed'. Raises ValueError otherwise.
            notes: Optional notes to attach (e.g. error message on failure).
        """

        if status not in _VALID_COMPLETION_STATUSES:
            raise ValueError(
                f"Invalid completion status '{status}'. "
                f"Must be one of: {sorted(_VALID_COMPLETION_STATUSES)}"
            )
        values: dict[str, Any] = {
            "status": status,
            "finished_at": text("now()"),
        }
        if notes is not None:
            values["notes"] = notes
        stmt = update(ingestion_runs).where(ingestion_runs.c.run_id == run_id).values(**values)
        engine = self._get_engine()
        with engine.begin() as conn:
            conn.execute(stmt)

    def upsert_document(self, doc: DocumentRecord) -> UUID:
        """INSERT or UPDATE a document record in kb_documents.

        On (source_uri, doc_version) conflict, updates title, content_hash, metadata,
        run_id, category, and lang. Returns the doc_id (generated on INSERT, fetched on UPDATE).

        The metadata dict is enriched with 'year' and 'month' keys to satisfy the DDL
        CHECK constraint requiring metadata->>'year' = published_year.
        """

        meta = {**doc.metadata, "year": str(doc.published_year), "month": str(doc.published_month)}
        doc_id = uuid.uuid4()

        # Build raw SQL for INSERT ... ON CONFLICT ... DO UPDATE RETURNING doc_id.
        # SQLAlchemy Core's insert().on_conflict_do_update() requires the dialect extension.
        stmt = text(
            """
            INSERT INTO kb_documents (
                doc_id, source_type, source_uri, title, lang, category,
                mime_type, content_hash, doc_version, published_year, published_month,
                author, tags, metadata, run_id
            )
            VALUES (
                :doc_id, :source_type, :source_uri, :title, :lang, :category,
                :mime_type, :content_hash, :doc_version, :published_year, :published_month,
                :author, :tags, :metadata::jsonb, :run_id
            )
            ON CONFLICT (source_uri, doc_version) DO UPDATE
                SET title = EXCLUDED.title,
                    content_hash = EXCLUDED.content_hash,
                    metadata = EXCLUDED.metadata,
                    run_id = EXCLUDED.run_id,
                    category = EXCLUDED.category,
                    lang = EXCLUDED.lang
            RETURNING doc_id
            """
        )
        import json

        params: dict[str, Any] = {
            "doc_id": str(doc_id),
            "source_type": doc.source_type,
            "source_uri": doc.source_uri,
            "title": doc.title,
            "lang": doc.lang,
            "category": doc.category,
            "mime_type": doc.mime_type,
            "content_hash": doc.content_hash,
            "doc_version": doc.doc_version,
            "published_year": doc.published_year,
            "published_month": doc.published_month,
            "author": doc.author,
            "tags": doc.tags,
            "metadata": json.dumps(meta),
            "run_id": str(doc.run_id) if doc.run_id else None,
        }
        engine = self._get_engine()
        with engine.begin() as conn:
            row = conn.execute(stmt, params).fetchone()
        if row is None:
            raise RuntimeError("upsert_document: RETURNING doc_id returned no rows")
        return UUID(str(row[0]))

    def delete_chunks_for_doc(self, doc_id: UUID, doc_version: str) -> int:
        """DELETE chunks matching doc_id and doc_version.

        Returns the number of rows deleted.
        """

        stmt = delete(kb_chunks_w).where(
            kb_chunks_w.c.doc_id == doc_id,
            kb_chunks_w.c.doc_version == doc_version,
        )
        engine = self._get_engine()
        with engine.begin() as conn:
            result = conn.execute(stmt)
        return result.rowcount

    def batch_insert_chunks(self, chunks: list[ChunkRecord]) -> int:
        """INSERT multiple chunks in a single statement.

        Converts embedding list[float] to pgvector literal '[0.1,0.2,...]'.
        Enriches metadata with 'year' and 'month' to satisfy the DDL CHECK constraint.
        Returns the number of rows inserted.
        """

        if not chunks:
            return 0

        import json

        rows: list[dict[str, Any]] = []
        for chunk in chunks:
            meta = {
                **chunk.metadata,
                "year": str(chunk.citation_year),
                "month": str(chunk.citation_month),
            }
            rows.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": str(chunk.doc_id),
                    "doc_version": chunk.doc_version,
                    "chunk_index": chunk.chunk_index,
                    "chunk_text": chunk.chunk_text,
                    "token_count": chunk.token_count,
                    "citation_url": chunk.citation_url,
                    "citation_title": chunk.citation_title,
                    "citation_year": chunk.citation_year,
                    "citation_month": chunk.citation_month,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_id": chunk.section_id,
                    "anchor_id": chunk.anchor_id,
                    "embedding": _to_vector_literal(chunk.embedding),
                    "metadata": json.dumps(meta),
                    "run_id": str(chunk.run_id) if chunk.run_id else None,
                }
            )

        stmt = text(
            """
            INSERT INTO kb_chunks (
                chunk_id, doc_id, doc_version, chunk_index, chunk_text, token_count,
                citation_url, citation_title, citation_year, citation_month,
                page_start, page_end, section_id, anchor_id,
                embedding, metadata, run_id
            )
            VALUES (
                :chunk_id::uuid, :doc_id::uuid, :doc_version, :chunk_index, :chunk_text,
                :token_count, :citation_url, :citation_title, :citation_year, :citation_month,
                :page_start, :page_end, :section_id, :anchor_id,
                (:embedding)::vector, :metadata::jsonb, :run_id::uuid
            )
            """
        )
        engine = self._get_engine()
        with engine.begin() as conn:
            for row in rows:
                conn.execute(stmt, row)
        return len(rows)

    def bulk_index_opensearch(
        self,
        chunks: list[ChunkRecord],
        doc_ids: dict[int, UUID] | None = None,
    ) -> None:
        """Bulk index chunks to OpenSearch.

        Each chunk becomes a document with chunk_text, citation fields, and metadata.
        If OpenSearch is not configured (no endpoint), this method silently skips.

        Args:
            chunks: List of ChunkRecord objects to index.
            doc_ids: Optional mapping of chunk_index to doc_id override. Unused if None.
        """

        if not self._settings.opensearch_endpoint.strip():
            return
        if not chunks:
            return

        client = self._get_opensearch_client()
        actions: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            doc_id = chunk.doc_id
            if doc_ids is not None and chunk.chunk_index in doc_ids:
                doc_id = doc_ids[chunk.chunk_index]
            actions.append(
                {
                    "_index": self._settings.opensearch_index,
                    "_id": chunk_id,
                    "_source": {
                        "chunk_text": chunk.chunk_text,
                        "doc_id": str(doc_id),
                        "citation_url": chunk.citation_url,
                        "citation_title": chunk.citation_title,
                        "citation_year": chunk.citation_year,
                        "citation_month": chunk.citation_month,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "section_id": chunk.section_id,
                        "anchor_id": chunk.anchor_id,
                        "metadata": chunk.metadata,
                    },
                }
            )
        helpers.bulk(client, actions)

    def get_ingestion_run(self, run_id: UUID) -> dict[str, Any] | None:
        """SELECT a single ingestion run by run_id.

        Returns a dict with keys matching IngestionStatusResponse, or None if not found.
        """

        stmt = select(ingestion_runs).where(ingestion_runs.c.run_id == run_id)
        engine = self._get_engine()
        with engine.begin() as conn:
            row = conn.execute(stmt).fetchone()
        if row is None:
            return None
        return {
            "run_id": str(row.run_id),
            "status": row.status,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
            "notes": row.notes,
        }

    def _get_engine(self) -> Engine:
        """Lazily construct and cache SQLAlchemy engine for connection reuse."""

        if self._engine is not None:
            return self._engine

        db_password = resolve_db_password(self._settings)
        if not db_password:
            raise ValueError(
                "DB password is not configured. Set RAG_DB_PASSWORD or "
                "RAG_DB_PASSWORD_SECRET_ARN."
            )

        self._engine = create_engine(
            self._settings.build_db_dsn(db_password),
            pool_pre_ping=True,
        )
        return self._engine

    def _get_opensearch_client(self) -> OpenSearch:
        """Lazily construct and cache OpenSearch/Elasticsearch client."""

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
