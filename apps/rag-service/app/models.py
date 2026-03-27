"""Pydantic request/response models for retrieval APIs and action handlers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class RetrievalFilters(BaseModel):
    """Optional retrieval filters applied to both sparse and dense candidate queries."""

    category: str | None = None
    lang: str | None = None
    source_type: Literal["crawler", "file"] | None = None
    citation_year_from: int | None = Field(default=None, ge=1000, le=2100)
    citation_year_to: int | None = Field(default=None, ge=1000, le=2100)
    citation_month: int | None = Field(default=None, ge=1, le=12)

    @field_validator("citation_year_to")
    @classmethod
    def validate_year_range(cls, value: int | None, info: Any) -> int | None:
        """Validate that `citation_year_to` is not less than `citation_year_from`."""

        year_from = info.data.get("citation_year_from")
        if value is not None and year_from is not None and value < year_from:
            raise ValueError("citation_year_to must be greater than or equal to citation_year_from")
        return value


class RetrieveRequest(BaseModel):
    """Input payload for retrieval operations.

    `query_embedding` is optional:
    - missing -> sparse mode
    - present -> hybrid sparse+dense mode
    """

    query: str = Field(min_length=1)
    query_embedding: list[float] | None = None
    k_sparse: int = Field(default=40, ge=1, le=200)
    k_dense: int = Field(default=40, ge=1, le=200)
    k_final: int = Field(default=8, ge=1, le=50)
    rrf_k: int | None = Field(default=None, ge=1, le=500)
    require_strict_citation: bool = True
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)


class Citation(BaseModel):
    """Strict citation contract returned for each retrieval hit."""

    url: str
    title: str
    year: int = Field(ge=1000, le=2100)
    month: int = Field(ge=1, le=12)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_id: str | None = None
    anchor_id: str | None = None


class RetrievalHit(BaseModel):
    """Normalized retrieval hit payload consumed by API and Bedrock action responses."""

    chunk_id: str
    doc_id: str
    chunk_text: str
    score: float
    category: str
    lang: str
    source_type: Literal["crawler", "file"]
    metadata: dict[str, Any]
    citation: Citation


class RetrieveResponse(BaseModel):
    """Top-level retrieval API response."""

    query: str
    mode: Literal["sparse", "hybrid"]
    hit_count: int
    hits: list[RetrievalHit]
