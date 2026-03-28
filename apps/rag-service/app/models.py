"""Pydantic request/response models for retrieval APIs and action handlers."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Graph-enhanced retrieval types (Phase 3)
# ---------------------------------------------------------------------------


class RetrievalMode(str, Enum):
    """Controls how graph context is combined with traditional retrieval.

    - ``chunks_only``: Legacy mode — no graph retrieval.
    - ``graph_only``: Return only graph-derived context (debugging).
    - ``mix``: Merge graph context into the standard chunk pipeline (default).
    """

    CHUNKS_ONLY = "chunks_only"
    GRAPH_ONLY = "graph_only"
    MIX = "mix"


class GraphEntity(BaseModel):
    """A single entity surfaced by graph retrieval."""

    entity_id: str
    name: str
    type: str
    description: str
    confidence: float = 0.0
    score: float = 0.0  # retrieval similarity / relevance score


class GraphRelation(BaseModel):
    """A single relation surfaced by graph retrieval."""

    source_entity: str
    target_entity: str
    relation_type: str
    evidence: str
    confidence: float = 0.0
    weight: float = 1.0
    score: float = 0.0  # retrieval similarity / relevance score


class GraphContext(BaseModel):
    """Aggregated graph retrieval result injected into the RAG workflow state.

    Produced by ``GraphRetriever`` and consumed by downstream nodes
    (reranker, answer generator) to enrich evidence.
    """

    entities: list[GraphEntity] = Field(default_factory=list)
    relations: list[GraphRelation] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs referenced by retrieved entities/relations — used for boosting",
    )

    @property
    def is_empty(self) -> bool:
        """Return True when no graph evidence was found."""
        return not self.entities and not self.relations

    def to_evidence_text(self, max_entities: int = 10, max_relations: int = 10) -> str:
        """Serialize graph context into a compact text block for prompt injection.

        Args:
            max_entities: Cap on entities to include.
            max_relations: Cap on relations to include.

        Returns:
            Human-readable evidence string, or empty string if nothing found.
        """
        if self.is_empty:
            return ""

        parts: list[str] = []

        if self.entities:
            entity_lines = []
            for ent in sorted(self.entities, key=lambda e: e.score, reverse=True)[:max_entities]:
                entity_lines.append(f"- {ent.name} ({ent.type}): {ent.description}")
            parts.append("### Entities\n" + "\n".join(entity_lines))

        if self.relations:
            rel_lines = []
            for rel in sorted(self.relations, key=lambda r: r.score, reverse=True)[:max_relations]:
                rel_lines.append(
                    f"- {rel.source_entity} --[{rel.relation_type}]--> {rel.target_entity}: {rel.evidence}"
                )
            parts.append("### Relations\n" + "\n".join(rel_lines))

        return "\n\n".join(parts)


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


class KeywordResult(BaseModel):
    """Dual-level keyword extraction result inspired by LightRAG."""

    hl_keywords: list[str] = Field(default_factory=list, description="High-level thematic keywords")
    ll_keywords: list[str] = Field(
        default_factory=list, description="Low-level specific entities/terms"
    )
