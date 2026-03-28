"""Data models for the document ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class UploadMetadata(BaseModel):
    """Metadata provided by the upload endpoint, persisted as S3 object metadata."""

    title: str
    source_uri: str = ""
    lang: str = "en"
    category: str = "general"
    published_year: int = Field(ge=1000, le=2100)
    published_month: int = Field(ge=1, le=12)
    author: str | None = None
    tags: list[str] = Field(default_factory=list)
    doc_version: str = "1.0"


class UploadResponse(BaseModel):
    """Response returned immediately after a successful upload request."""

    run_id: str
    s3_key: str
    filename: str
    status: str = "accepted"
    message: str = "Document queued for ingestion"


class IngestionStatusResponse(BaseModel):
    """Response for GET /ingestion/{run_id}."""

    run_id: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    notes: str | None = None


@dataclass
class DocumentRecord:
    """Intermediate record for inserting into kb_documents."""

    source_type: str
    source_uri: str
    title: str
    lang: str
    category: str
    mime_type: str
    content_hash: str
    doc_version: str
    published_year: int
    published_month: int
    author: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: UUID | None = None


@dataclass
class ChunkRecord:
    """Intermediate record for inserting into kb_chunks."""

    doc_id: UUID
    doc_version: str
    chunk_index: int
    chunk_text: str
    token_count: int
    citation_url: str
    citation_title: str
    citation_year: int
    citation_month: int
    embedding: list[float]
    page_start: int | None = None
    page_end: int | None = None
    section_id: str | None = None
    anchor_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: UUID | None = None


@dataclass
class IngestionResult:
    """Summary of a completed ingestion run."""

    run_id: UUID
    doc_id: UUID
    status: str
    chunks_created: int
    error: str | None = None
