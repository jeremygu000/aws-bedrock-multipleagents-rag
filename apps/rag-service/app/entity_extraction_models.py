"""Pydantic models for entity and relation extraction.

These models define the schema for LLM-extracted entities and relations from document chunks,
following the Qwen-Plus Entity Extraction Spec (docs/qwen-plus-entity-extraction-spec.md).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class EntityType(str, Enum):
    """Supported entity types for extraction."""

    WORK = "Work"
    PERSON = "Person"
    ORGANIZATION = "Organization"
    IDENTIFIER = "Identifier"
    TERRITORY = "Territory"
    LICENSE_TERM = "LicenseTerm"
    DATE = "Date"


class RelationType(str, Enum):
    """Supported relation types between entities."""

    WROTE = "WROTE"
    PERFORMED_BY = "PERFORMED_BY"
    PUBLISHED_BY = "PUBLISHED_BY"
    HAS_IDENTIFIER = "HAS_IDENTIFIER"
    VALID_IN_TERRITORY = "VALID_IN_TERRITORY"
    HAS_TERM = "HAS_TERM"
    REFERENCES = "REFERENCES"


class Mention(BaseModel):
    """A surface mention of an entity in the source text."""

    text: str = Field(description="Original surface form in the source text")
    start: int = Field(default=0, ge=0, description="Start character offset in chunk text")
    end: int = Field(default=0, ge=0, description="End character offset in chunk text")

    @model_validator(mode="after")
    def validate_span(self) -> Mention:
        if self.end < self.start:
            raise ValueError(f"Mention end ({self.end}) must be >= start ({self.start})")
        return self


class ExtractedEntity(BaseModel):
    """An entity extracted from a document chunk."""

    entity_id: str = Field(description="Unique identifier for the entity within this extraction")
    type: EntityType = Field(description="Entity type classification")
    name: str = Field(min_length=1, description="Canonical entity name")
    canonical_key: str | None = Field(
        default=None,
        description="Deduplication key (e.g., normalized name or ID)",
    )
    description: str = Field(default="", description="Entity description from extraction context")
    aliases: list[str] = Field(default_factory=list, description="Alternative names or spellings")
    mentions: list[Mention] = Field(
        default_factory=list,
        description="Surface mentions with character offsets",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence score")
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="IDs of chunks this entity was extracted from (for provenance tracking)",
    )


class ExtractedRelation(BaseModel):
    """A relation between two extracted entities."""

    type: RelationType = Field(description="Relation type classification")
    source_entity_id: str = Field(description="entity_id of the source entity")
    target_entity_id: str = Field(description="entity_id of the target entity")
    evidence: str = Field(default="", description="Supporting text snippet from the source")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence score")
    weight: float = Field(default=1.0, ge=0.0, description="Accumulated weight from merges")
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="IDs of chunks this relation was extracted from (for provenance tracking)",
    )


class ChunkExtractionResult(BaseModel):
    """Complete extraction result for a single chunk, matching the LLM JSON schema."""

    chunk_id: str = Field(description="ID of the source chunk")
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)

    @field_validator("relations")
    @classmethod
    def validate_relation_endpoints(
        cls, relations: list[ExtractedRelation], info: object
    ) -> list[ExtractedRelation]:
        """Validate that relation endpoints reference existing entity IDs or names.

        Allows name-based references (case-insensitive) in addition to entity_id matches.
        Name→ID resolution happens in post-processing (EntityExtractor._resolve_relation_endpoints).
        """
        data = getattr(info, "data", {})
        entities = data.get("entities")
        if entities is None:
            return relations

        entity_ids = {e.entity_id for e in entities}
        entity_names = {e.name.lower() for e in entities}

        invalid = []
        for rel in relations:
            if (
                rel.source_entity_id not in entity_ids
                and rel.source_entity_id.lower() not in entity_names
            ):
                invalid.append(f"source '{rel.source_entity_id}' not in entities or names")
            if (
                rel.target_entity_id not in entity_ids
                and rel.target_entity_id.lower() not in entity_names
            ):
                invalid.append(f"target '{rel.target_entity_id}' not in entities or names")

        if invalid:
            raise ValueError(f"Invalid relation endpoints: {'; '.join(invalid)}")
        return relations


class ExtractionTrace(BaseModel):
    """Trace/logging metadata for an extraction job, per spec section 8."""

    model_provider: str = "qwen"
    model_name: str = "qwen-plus"
    model_version: str | None = None
    prompt_version: str = "1.0"
    schema_version: str = "1.0"
    chunk_id: str = ""
    doc_id: str = ""
    run_id: str = ""
    validation_status: str = "valid"
    failure_reason: str | None = None
