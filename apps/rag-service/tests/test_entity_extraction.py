import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.entity_extraction import (
    EntityDeduplicator,
    EntityExtractor,
    _entity_dedup_key,
    _estimate_token_count,
)
from app.entity_extraction_models import (
    ChunkExtractionResult,
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionTrace,
    Mention,
    RelationType,
)

VALID_LLM_RESPONSE = json.dumps(
    {
        "chunk_id": "chunk-001",
        "entities": [
            {
                "entity_id": "e1",
                "type": "Person",
                "name": "John Smith",
                "canonical_key": "john_smith",
                "aliases": ["J. Smith"],
                "mentions": [{"text": "John Smith", "start": 0, "end": 10}],
                "confidence": 0.95,
            },
            {
                "entity_id": "e2",
                "type": "Work",
                "name": "Yesterday",
                "canonical_key": "yesterday",
                "aliases": [],
                "mentions": [{"text": "Yesterday", "start": 20, "end": 29}],
                "confidence": 0.90,
            },
        ],
        "relations": [
            {
                "type": "WROTE",
                "source_entity_id": "e1",
                "target_entity_id": "e2",
                "evidence": "John Smith wrote Yesterday",
                "confidence": 0.85,
            }
        ],
    }
)


def _make_qwen_mock(response: str = VALID_LLM_RESPONSE) -> MagicMock:
    mock = MagicMock()
    mock.chat.return_value = response
    return mock


class TestEntityExtractionModels:
    def test_valid_entity(self) -> None:
        entity = ExtractedEntity(
            entity_id="e1",
            type=EntityType.PERSON,
            name="John",
            confidence=0.9,
        )
        assert entity.name == "John"
        assert entity.type == EntityType.PERSON
        assert entity.aliases == []
        assert entity.mentions == []

    def test_entity_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ExtractedEntity(entity_id="e1", type=EntityType.PERSON, name="X", confidence=1.5)
        with pytest.raises(ValidationError):
            ExtractedEntity(entity_id="e1", type=EntityType.PERSON, name="X", confidence=-0.1)

    def test_entity_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExtractedEntity(entity_id="e1", type=EntityType.PERSON, name="", confidence=0.9)

    def test_mention_span_validation(self) -> None:
        valid = Mention(text="hello", start=0, end=5)
        assert valid.start == 0
        assert valid.end == 5

        with pytest.raises(ValidationError, match="must be >= start"):
            Mention(text="hello", start=10, end=5)

    def test_valid_relation(self) -> None:
        rel = ExtractedRelation(
            type=RelationType.WROTE,
            source_entity_id="e1",
            target_entity_id="e2",
            confidence=0.8,
        )
        assert rel.type == RelationType.WROTE
        assert rel.evidence == ""

    def test_chunk_extraction_result_validates_endpoints(self) -> None:
        with pytest.raises(ValidationError, match="not in entities"):
            ChunkExtractionResult(
                chunk_id="c1",
                entities=[
                    ExtractedEntity(
                        entity_id="e1", type=EntityType.PERSON, name="A", confidence=0.9
                    )
                ],
                relations=[
                    ExtractedRelation(
                        type=RelationType.WROTE,
                        source_entity_id="e1",
                        target_entity_id="e999",
                        confidence=0.8,
                    )
                ],
            )

    def test_chunk_extraction_result_valid(self) -> None:
        result = ChunkExtractionResult(
            chunk_id="c1",
            entities=[
                ExtractedEntity(entity_id="e1", type=EntityType.PERSON, name="A", confidence=0.9),
                ExtractedEntity(entity_id="e2", type=EntityType.WORK, name="B", confidence=0.8),
            ],
            relations=[
                ExtractedRelation(
                    type=RelationType.WROTE,
                    source_entity_id="e1",
                    target_entity_id="e2",
                    confidence=0.85,
                )
            ],
        )
        assert len(result.entities) == 2
        assert len(result.relations) == 1

    def test_extraction_trace_defaults(self) -> None:
        trace = ExtractionTrace(chunk_id="c1", doc_id="d1")
        assert trace.model_provider == "qwen"
        assert trace.model_name == "qwen-plus"
        assert trace.validation_status == "valid"
        assert trace.failure_reason is None


class TestRuleBasedExtraction:
    def test_iswc_extraction(self) -> None:
        text = "The work has ISWC T-345.678.901-2 registered."
        extractor = EntityExtractor(_make_qwen_mock())
        entities = extractor._rule_based_extraction(text, "c1")
        assert len(entities) >= 1
        iswc_entities = [e for e in entities if e.name.startswith("T-") or "345" in e.name]
        assert len(iswc_entities) >= 1
        assert iswc_entities[0].type == EntityType.IDENTIFIER
        assert iswc_entities[0].confidence == 1.0

    def test_isrc_extraction(self) -> None:
        text = "Track identified as ISRC: US-ABC-12-34567 in the catalog."
        extractor = EntityExtractor(_make_qwen_mock())
        entities = extractor._rule_based_extraction(text, "c1")
        isrc_entities = [e for e in entities if "ABC" in e.name or "34567" in e.name]
        assert len(isrc_entities) >= 1
        assert isrc_entities[0].type == EntityType.IDENTIFIER

    def test_date_extraction_iso(self) -> None:
        text = "Agreement signed on 2024-03-15 in London."
        extractor = EntityExtractor(_make_qwen_mock())
        entities = extractor._rule_based_extraction(text, "c1")
        date_entities = [e for e in entities if e.type == EntityType.DATE]
        assert len(date_entities) >= 1
        assert "2024-03-15" in date_entities[0].name

    def test_date_extraction_written(self) -> None:
        text = "Published on January 15, 2024 by the committee."
        extractor = EntityExtractor(_make_qwen_mock())
        entities = extractor._rule_based_extraction(text, "c1")
        date_entities = [e for e in entities if e.type == EntityType.DATE]
        assert len(date_entities) >= 1

    def test_no_duplicates(self) -> None:
        text = "ISWC T-345.678.901-2 and again T-345.678.901-2."
        extractor = EntityExtractor(_make_qwen_mock())
        entities = extractor._rule_based_extraction(text, "c1")
        names = [e.name for e in entities if e.type == EntityType.IDENTIFIER]
        unique_names = set(names)
        assert len(names) == len(unique_names)

    def test_empty_text(self) -> None:
        extractor = EntityExtractor(_make_qwen_mock())
        entities = extractor._rule_based_extraction("", "c1")
        assert entities == []


class TestLLMExtraction:
    def test_happy_path(self) -> None:
        mock = _make_qwen_mock(VALID_LLM_RESPONSE)
        extractor = EntityExtractor(mock)
        result, trace = extractor.extract("chunk-001", "doc-001", "John Smith wrote Yesterday")

        assert trace.validation_status == "valid"
        assert len(result.entities) >= 2
        assert len(result.relations) >= 1
        assert result.chunk_id == "chunk-001"

        person_entities = [e for e in result.entities if e.type == EntityType.PERSON]
        assert len(person_entities) >= 1
        assert person_entities[0].name == "John Smith"

    def test_markdown_fenced_response(self) -> None:
        fenced = f"```json\n{VALID_LLM_RESPONSE}\n```"
        mock = _make_qwen_mock(fenced)
        extractor = EntityExtractor(mock)
        result, trace = extractor.extract("chunk-001", "doc-001", "text")

        assert trace.validation_status == "valid"
        assert len(result.entities) >= 2

    def test_invalid_json_triggers_repair(self) -> None:
        mock = MagicMock()
        mock.chat.side_effect = [
            '{"chunk_id": "c1", "entities": [INVALID',
            VALID_LLM_RESPONSE,
        ]
        extractor = EntityExtractor(mock)
        result, trace = extractor.extract("c1", "d1", "text")

        assert mock.chat.call_count == 2
        assert trace.validation_status == "valid"

    def test_both_attempts_fail_returns_rule_entities_only(self) -> None:
        mock = MagicMock()
        mock.chat.side_effect = ValueError("API error")
        extractor = EntityExtractor(mock)

        text = "Agreement date 2024-03-15"
        result, trace = extractor.extract("c1", "d1", text)

        assert trace.validation_status == "extraction_failed"
        assert trace.failure_reason is not None
        assert result.relations == []
        date_entities = [e for e in result.entities if e.type == EntityType.DATE]
        assert len(date_entities) >= 1

    def test_merge_deduplicates_rule_and_llm_entities(self) -> None:
        llm_response = json.dumps(
            {
                "chunk_id": "c1",
                "entities": [
                    {
                        "entity_id": "e1",
                        "type": "Identifier",
                        "name": "T3456789012",
                        "canonical_key": "T3456789012",
                        "aliases": [],
                        "mentions": [{"text": "T-345.678.901-2", "start": 10, "end": 25}],
                        "confidence": 0.95,
                    }
                ],
                "relations": [],
            }
        )
        mock = _make_qwen_mock(llm_response)
        extractor = EntityExtractor(mock)
        text = "The code is T-345.678.901-2 as noted."
        result, trace = extractor.extract("c1", "d1", text)

        assert trace.validation_status == "valid"
        id_entities = [e for e in result.entities if e.type == EntityType.IDENTIFIER]
        canonical_keys = [e.canonical_key for e in id_entities if e.canonical_key]
        assert len(set(canonical_keys)) == len(canonical_keys)

    def test_empty_chunk_text(self) -> None:
        empty_response = json.dumps({"chunk_id": "c1", "entities": [], "relations": []})
        mock = _make_qwen_mock(empty_response)
        extractor = EntityExtractor(mock)
        result, trace = extractor.extract("c1", "d1", "")

        assert trace.validation_status == "valid"
        assert result.entities == []
        assert result.relations == []


class TestStripMarkdownFences:
    def test_no_fences(self) -> None:
        assert EntityExtractor._strip_markdown_fences('{"a": 1}') == '{"a": 1}'

    def test_json_fences(self) -> None:
        assert EntityExtractor._strip_markdown_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_plain_fences(self) -> None:
        assert EntityExtractor._strip_markdown_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_whitespace_handling(self) -> None:
        assert EntityExtractor._strip_markdown_fences('  ```json\n{"a": 1}\n```  ') == '{"a": 1}'


class TestConfigFlag:
    def test_entity_extraction_disabled_by_default(self, monkeypatch) -> None:
        from app.config import Settings

        monkeypatch.delenv("RAG_ENABLE_ENTITY_EXTRACTION", raising=False)
        settings = Settings()
        assert settings.enable_entity_extraction is False

    def test_entity_extraction_can_be_enabled(self) -> None:
        from app.config import Settings

        settings = Settings(RAG_ENABLE_ENTITY_EXTRACTION="true")
        assert settings.enable_entity_extraction is True

    def test_entity_extraction_max_retries_default(self) -> None:
        from app.config import Settings

        settings = Settings()
        assert settings.entity_extraction_max_retries == 1

    def test_entity_summary_max_tokens_default(self) -> None:
        from app.config import Settings

        settings = Settings()
        assert settings.entity_summary_max_tokens == 500

    def test_extraction_gleaning_rounds_default(self) -> None:
        from app.config import Settings

        settings = Settings()
        assert settings.extraction_gleaning_rounds == 0

    def test_extraction_gleaning_rounds_can_be_set(self) -> None:
        from app.config import Settings

        settings = Settings(RAG_EXTRACTION_GLEANING_ROUNDS="2")
        assert settings.extraction_gleaning_rounds == 2


# ── Helpers for Phase 2.2 tests ──────────────────────────────────────────


def _make_entity(
    entity_id: str = "e1",
    name: str = "John Smith",
    entity_type: EntityType = EntityType.PERSON,
    canonical_key: str | None = "john_smith",
    aliases: list[str] | None = None,
    mentions: list[Mention] | None = None,
    confidence: float = 0.9,
    description: str = "",
    source_chunk_ids: list[str] | None = None,
) -> ExtractedEntity:
    """Create an ExtractedEntity with sensible defaults for testing."""
    return ExtractedEntity(
        entity_id=entity_id,
        type=entity_type,
        name=name,
        canonical_key=canonical_key,
        aliases=aliases or [],
        mentions=mentions or [],
        confidence=confidence,
        description=description,
        source_chunk_ids=source_chunk_ids or [],
    )


def _make_relation(
    rel_type: RelationType = RelationType.WROTE,
    source_entity_id: str = "e1",
    target_entity_id: str = "e2",
    evidence: str = "evidence text",
    confidence: float = 0.85,
    weight: float = 1.0,
    source_chunk_ids: list[str] | None = None,
) -> ExtractedRelation:
    """Create an ExtractedRelation with sensible defaults for testing."""
    return ExtractedRelation(
        type=rel_type,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        evidence=evidence,
        confidence=confidence,
        weight=weight,
        source_chunk_ids=source_chunk_ids or [],
    )


def _make_chunk_result(
    chunk_id: str,
    entities: list[ExtractedEntity],
    relations: list[ExtractedRelation] | None = None,
) -> ChunkExtractionResult:
    """Create a ChunkExtractionResult, bypassing the relation endpoint validator."""
    # Use model_construct to skip relation endpoint validation since we may
    # reference entity IDs from other chunks
    return ChunkExtractionResult.model_construct(
        chunk_id=chunk_id,
        entities=entities,
        relations=relations or [],
    )


# ── Helper function unit tests ───────────────────────────────────────────


class TestEntityDedupKey:
    """Tests for _entity_dedup_key helper."""

    def test_uses_canonical_key_when_present(self) -> None:
        entity = _make_entity(canonical_key="john_smith")
        assert _entity_dedup_key(entity) == "john_smith"

    def test_falls_back_to_lowercase_name(self) -> None:
        entity = _make_entity(name="John Smith", canonical_key=None)
        assert _entity_dedup_key(entity) == "john smith"

    def test_empty_canonical_key_falls_back(self) -> None:
        """Empty string canonical_key is falsy, should fall back to name.lower()."""
        entity = _make_entity(name="ACME Corp", canonical_key="")
        assert _entity_dedup_key(entity) == "acme corp"


class TestEstimateTokenCount:
    """Tests for _estimate_token_count helper."""

    def test_empty_string(self) -> None:
        assert _estimate_token_count("") == 0

    def test_short_text(self) -> None:
        # "hello" = 5 chars → 5 // 4 = 1
        assert _estimate_token_count("hello") == 1

    def test_longer_text(self) -> None:
        text = "a" * 400
        assert _estimate_token_count(text) == 100


# ── EntityDeduplicator.merge_entities tests ──────────────────────────────


class TestMergeEntities:
    """Tests for EntityDeduplicator.merge_entities()."""

    def test_single_chunk_no_dedup(self) -> None:
        """Entities from a single chunk should pass through unchanged."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)
        e1 = _make_entity(entity_id="e1", name="Alice", canonical_key="alice")
        e2 = _make_entity(entity_id="e2", name="Bob", canonical_key="bob")
        chunk = _make_chunk_result("c1", [e1, e2])

        merged = dedup.merge_entities([chunk])

        assert len(merged) == 2
        names = {e.name for e in merged}
        assert names == {"Alice", "Bob"}

    def test_dedup_by_canonical_key(self) -> None:
        """Same canonical_key across chunks → merged into one entity."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        e1 = _make_entity(
            entity_id="e1",
            name="John Smith",
            canonical_key="john_smith",
            confidence=0.8,
            description="A songwriter",
        )
        e2 = _make_entity(
            entity_id="e2",
            name="J. Smith",
            canonical_key="john_smith",
            confidence=0.95,
            description="An artist",
        )
        chunk1 = _make_chunk_result("c1", [e1])
        chunk2 = _make_chunk_result("c2", [e2])

        merged = dedup.merge_entities([chunk1, chunk2])

        assert len(merged) == 1
        entity = merged[0]
        # Should keep the base name (first seen)
        assert entity.name == "John Smith"
        # Should take the best confidence
        assert entity.confidence == 0.95
        # "J. Smith" should be an alias (different from base name)
        assert "J. Smith" in entity.aliases
        # Both chunk IDs tracked
        assert "c1" in entity.source_chunk_ids
        assert "c2" in entity.source_chunk_ids

    def test_dedup_by_lowercase_name_fallback(self) -> None:
        """When canonical_key is None, dedup falls back to name.lower()."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        e1 = _make_entity(entity_id="e1", name="APRA AMCOS", canonical_key=None, confidence=0.7)
        e2 = _make_entity(entity_id="e2", name="apra amcos", canonical_key=None, confidence=0.9)
        chunk1 = _make_chunk_result("c1", [e1])
        chunk2 = _make_chunk_result("c2", [e2])

        merged = dedup.merge_entities([chunk1, chunk2])

        assert len(merged) == 1
        assert merged[0].confidence == 0.9

    def test_no_duplicate_aliases(self) -> None:
        """Base entity name should not appear in aliases."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        e1 = _make_entity(
            entity_id="e1", name="John Smith", canonical_key="john_smith", aliases=["Johnny"]
        )
        e2 = _make_entity(
            entity_id="e2", name="John Smith", canonical_key="john_smith", aliases=["JS"]
        )
        chunk1 = _make_chunk_result("c1", [e1])
        chunk2 = _make_chunk_result("c2", [e2])

        merged = dedup.merge_entities([chunk1, chunk2])

        entity = merged[0]
        # "John Smith" should not be in aliases (it's the name)
        assert "John Smith" not in entity.aliases
        # Both alias sets should be merged
        assert "Johnny" in entity.aliases
        assert "JS" in entity.aliases

    def test_mentions_aggregated(self) -> None:
        """All mentions from duplicate entities are combined."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        m1 = Mention(text="John Smith", start=0, end=10)
        m2 = Mention(text="J. Smith", start=5, end=13)
        e1 = _make_entity(
            entity_id="e1", name="John Smith", canonical_key="john_smith", mentions=[m1]
        )
        e2 = _make_entity(
            entity_id="e2", name="J. Smith", canonical_key="john_smith", mentions=[m2]
        )
        chunk1 = _make_chunk_result("c1", [e1])
        chunk2 = _make_chunk_result("c2", [e2])

        merged = dedup.merge_entities([chunk1, chunk2])

        assert len(merged[0].mentions) == 2

    def test_source_chunk_ids_stamped_from_chunk_result(self) -> None:
        """When entity has no source_chunk_ids, they are stamped from chunk_result.chunk_id."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        e1 = _make_entity(entity_id="e1", name="Alice", canonical_key="alice", source_chunk_ids=[])
        chunk1 = _make_chunk_result("c1", [e1])

        merged = dedup.merge_entities([chunk1])

        assert merged[0].source_chunk_ids == ["c1"]

    def test_source_chunk_ids_preserved_if_already_set(self) -> None:
        """When entity already has source_chunk_ids, they are preserved (not overwritten)."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        e1 = _make_entity(
            entity_id="e1",
            name="Alice",
            canonical_key="alice",
            source_chunk_ids=["original-chunk"],
        )
        chunk1 = _make_chunk_result("c1", [e1])

        merged = dedup.merge_entities([chunk1])

        assert merged[0].source_chunk_ids == ["original-chunk"]

    def test_empty_input(self) -> None:
        """No chunks → no merged entities."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        merged = dedup.merge_entities([])

        assert merged == []

    def test_distinct_entities_not_merged(self) -> None:
        """Entities with different keys remain separate."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        e1 = _make_entity(entity_id="e1", name="Alice", canonical_key="alice")
        e2 = _make_entity(entity_id="e2", name="Bob", canonical_key="bob")
        chunk1 = _make_chunk_result("c1", [e1])
        chunk2 = _make_chunk_result("c2", [e2])

        merged = dedup.merge_entities([chunk1, chunk2])

        assert len(merged) == 2

    def test_three_way_merge(self) -> None:
        """Three occurrences of the same entity across three chunks merge correctly."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        e1 = _make_entity(entity_id="e1", name="APRA", canonical_key="apra", confidence=0.7)
        e2 = _make_entity(entity_id="e2", name="APRA Corp", canonical_key="apra", confidence=0.8)
        e3 = _make_entity(entity_id="e3", name="APRA Ltd", canonical_key="apra", confidence=0.9)
        c1 = _make_chunk_result("c1", [e1])
        c2 = _make_chunk_result("c2", [e2])
        c3 = _make_chunk_result("c3", [e3])

        merged = dedup.merge_entities([c1, c2, c3])

        assert len(merged) == 1
        entity = merged[0]
        assert entity.confidence == 0.9
        assert len(entity.source_chunk_ids) == 3
        # "APRA Corp" and "APRA Ltd" should be aliases (different from base "APRA")
        assert "APRA Corp" in entity.aliases
        assert "APRA Ltd" in entity.aliases


# ── EntityDeduplicator._merge_descriptions tests ────────────────────────


class TestMergeDescriptions:
    """Tests for EntityDeduplicator._merge_descriptions()."""

    def test_empty_descriptions(self) -> None:
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)
        assert dedup._merge_descriptions([], "entity", "Person") == ""

    def test_single_description(self) -> None:
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)
        assert dedup._merge_descriptions(["A songwriter"], "John", "Person") == "A songwriter"

    def test_short_descriptions_returns_longest(self) -> None:
        """When combined is under token threshold, return the longest description."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock, summary_max_tokens=500)

        descriptions = ["Short desc", "A longer description with more detail"]
        result = dedup._merge_descriptions(descriptions, "Entity", "Person")

        # Should return the longest
        assert result == "A longer description with more detail"
        # LLM should NOT be called
        mock.chat.assert_not_called()

    def test_long_descriptions_triggers_llm_summarization(self) -> None:
        """When combined descriptions exceed token threshold, LLM summarization is triggered."""
        mock = _make_qwen_mock("Consolidated summary of the entity.")
        # Set very low threshold to force LLM path
        dedup = EntityDeduplicator(mock, summary_max_tokens=5)

        descriptions = [
            "A famous songwriter from England known for many hits.",
            "An award-winning musician with decades of experience in the industry.",
        ]
        result = dedup._merge_descriptions(descriptions, "John Smith", "Person")

        assert result == "Consolidated summary of the entity."
        mock.chat.assert_called_once()
        # Verify the prompt includes entity name and type
        call_args = mock.chat.call_args
        assert "John Smith" in call_args[0][1]  # user prompt
        assert "Person" in call_args[0][1]

    def test_llm_failure_falls_back_to_longest(self) -> None:
        """When LLM summarization fails, fallback to the longest description."""
        mock = MagicMock()
        mock.chat.side_effect = RuntimeError("API error")
        dedup = EntityDeduplicator(mock, summary_max_tokens=5)

        descriptions = [
            "Short.",
            "A much longer description that should be returned as fallback.",
        ]
        result = dedup._merge_descriptions(descriptions, "Entity", "Person")

        assert result == "A much longer description that should be returned as fallback."

    def test_duplicate_descriptions_not_repeated(self) -> None:
        """Duplicate descriptions from merge_entities flow are filtered before merge."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock, summary_max_tokens=500)

        descriptions = ["Same desc", "Different desc"]
        result = dedup._merge_descriptions(descriptions, "E", "Person")

        assert result == "Different desc"  # longer one


# ── EntityDeduplicator.merge_relations tests ─────────────────────────────


class TestMergeRelations:
    """Tests for EntityDeduplicator.merge_relations()."""

    def test_single_relation_no_dedup(self) -> None:
        """A single relation passes through with entity ID remapping."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        e1 = _make_entity(entity_id="e1", name="John", canonical_key="john")
        e2 = _make_entity(
            entity_id="e2", name="Yesterday", canonical_key="yesterday", entity_type=EntityType.WORK
        )
        rel = _make_relation(
            source_entity_id="e1", target_entity_id="e2", evidence="John wrote Yesterday"
        )
        chunk = _make_chunk_result("c1", [e1, e2], [rel])

        # Merged entities have new IDs
        merged_e1 = _make_entity(entity_id="merged_e1", name="John", canonical_key="john")
        merged_e2 = _make_entity(
            entity_id="merged_e2",
            name="Yesterday",
            canonical_key="yesterday",
            entity_type=EntityType.WORK,
        )

        merged_rels = dedup.merge_relations([chunk], [merged_e1, merged_e2])

        assert len(merged_rels) == 1
        # Entity IDs should be remapped to merged entity IDs
        assert merged_rels[0].source_entity_id == "merged_e1"
        assert merged_rels[0].target_entity_id == "merged_e2"

    def test_duplicate_relations_merged(self) -> None:
        """Same relation across two chunks → merged with accumulated weight and evidence."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        e1 = _make_entity(entity_id="e1", name="John", canonical_key="john")
        e2 = _make_entity(
            entity_id="e2", name="Song A", canonical_key="song_a", entity_type=EntityType.WORK
        )
        rel1 = _make_relation(
            source_entity_id="e1",
            target_entity_id="e2",
            evidence="Evidence from chunk 1",
            confidence=0.8,
        )

        e3 = _make_entity(entity_id="e3", name="John", canonical_key="john")
        e4 = _make_entity(
            entity_id="e4", name="Song A", canonical_key="song_a", entity_type=EntityType.WORK
        )
        rel2 = _make_relation(
            source_entity_id="e3",
            target_entity_id="e4",
            evidence="Evidence from chunk 2",
            confidence=0.95,
        )

        chunk1 = _make_chunk_result("c1", [e1, e2], [rel1])
        chunk2 = _make_chunk_result("c2", [e3, e4], [rel2])

        merged_e1 = _make_entity(entity_id="m_e1", name="John", canonical_key="john")
        merged_e2 = _make_entity(
            entity_id="m_e2", name="Song A", canonical_key="song_a", entity_type=EntityType.WORK
        )

        merged_rels = dedup.merge_relations([chunk1, chunk2], [merged_e1, merged_e2])

        assert len(merged_rels) == 1
        rel = merged_rels[0]
        # Weight = number of occurrences
        assert rel.weight == 2.0
        # Best confidence kept
        assert rel.confidence == 0.95
        # Evidence joined with pipe separator
        assert "Evidence from chunk 1" in rel.evidence
        assert "Evidence from chunk 2" in rel.evidence
        assert " | " in rel.evidence
        # Both chunks tracked
        assert "c1" in rel.source_chunk_ids
        assert "c2" in rel.source_chunk_ids

    def test_distinct_relations_not_merged(self) -> None:
        """Relations with different (source, target, type) remain separate."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        e1 = _make_entity(entity_id="e1", name="John", canonical_key="john")
        e2 = _make_entity(
            entity_id="e2", name="Song A", canonical_key="song_a", entity_type=EntityType.WORK
        )
        e3 = _make_entity(
            entity_id="e3", name="Song B", canonical_key="song_b", entity_type=EntityType.WORK
        )

        rel1 = _make_relation(source_entity_id="e1", target_entity_id="e2")
        rel2 = _make_relation(source_entity_id="e1", target_entity_id="e3")

        chunk = _make_chunk_result("c1", [e1, e2, e3], [rel1, rel2])

        merged_rels = dedup.merge_relations([chunk], [e1, e2, e3])

        assert len(merged_rels) == 2

    def test_different_relation_types_not_merged(self) -> None:
        """Same endpoints but different types → separate relations."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        e1 = _make_entity(entity_id="e1", name="John", canonical_key="john")
        e2 = _make_entity(
            entity_id="e2", name="Song A", canonical_key="song_a", entity_type=EntityType.WORK
        )

        rel1 = _make_relation(
            source_entity_id="e1", target_entity_id="e2", rel_type=RelationType.WROTE
        )
        rel2 = _make_relation(
            source_entity_id="e1", target_entity_id="e2", rel_type=RelationType.PERFORMED_BY
        )

        chunk = _make_chunk_result("c1", [e1, e2], [rel1, rel2])

        merged_rels = dedup.merge_relations([chunk], [e1, e2])

        assert len(merged_rels) == 2

    def test_empty_input(self) -> None:
        """No chunks → no merged relations."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        merged_rels = dedup.merge_relations([], [])

        assert merged_rels == []

    def test_entity_id_remapping_across_chunks(self) -> None:
        """Entity IDs in relations should be remapped to the merged entity set."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        # Chunk 1: e1 → e2 (WROTE)
        e1_c1 = _make_entity(entity_id="c1_e1", name="Alice", canonical_key="alice")
        e2_c1 = _make_entity(
            entity_id="c1_e2", name="Work X", canonical_key="work_x", entity_type=EntityType.WORK
        )
        rel_c1 = _make_relation(
            source_entity_id="c1_e1", target_entity_id="c1_e2", evidence="Alice wrote Work X"
        )

        # Chunk 2: e1 → e2 (WROTE) — same entities, different IDs
        e1_c2 = _make_entity(entity_id="c2_e1", name="Alice", canonical_key="alice")
        e2_c2 = _make_entity(
            entity_id="c2_e2", name="Work X", canonical_key="work_x", entity_type=EntityType.WORK
        )
        rel_c2 = _make_relation(
            source_entity_id="c2_e1", target_entity_id="c2_e2", evidence="Alice authored Work X"
        )

        chunk1 = _make_chunk_result("c1", [e1_c1, e2_c1], [rel_c1])
        chunk2 = _make_chunk_result("c2", [e1_c2, e2_c2], [rel_c2])

        # Merged entities (from merge_entities step)
        merged_alice = _make_entity(entity_id="merged_alice", name="Alice", canonical_key="alice")
        merged_work = _make_entity(
            entity_id="merged_work",
            name="Work X",
            canonical_key="work_x",
            entity_type=EntityType.WORK,
        )

        merged_rels = dedup.merge_relations([chunk1, chunk2], [merged_alice, merged_work])

        assert len(merged_rels) == 1
        rel = merged_rels[0]
        assert rel.source_entity_id == "merged_alice"
        assert rel.target_entity_id == "merged_work"
        assert rel.weight == 2.0

    def test_duplicate_evidence_not_repeated(self) -> None:
        """Identical evidence strings should not be duplicated in merged evidence."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        e1 = _make_entity(entity_id="e1", name="John", canonical_key="john")
        e2 = _make_entity(
            entity_id="e2", name="Song", canonical_key="song", entity_type=EntityType.WORK
        )

        # Same evidence string from two chunks
        rel1 = _make_relation(
            source_entity_id="e1", target_entity_id="e2", evidence="John wrote Song"
        )
        rel2 = _make_relation(
            source_entity_id="e1", target_entity_id="e2", evidence="John wrote Song"
        )

        chunk1 = _make_chunk_result("c1", [e1, e2], [rel1])
        chunk2 = _make_chunk_result("c2", [e1, e2], [rel2])

        merged_rels = dedup.merge_relations([chunk1, chunk2], [e1, e2])

        assert len(merged_rels) == 1
        # Evidence should appear only once (deduped)
        assert merged_rels[0].evidence == "John wrote Song"
        assert " | " not in merged_rels[0].evidence

    def test_source_chunk_ids_on_single_relation(self) -> None:
        """Single relation should get chunk_id stamped if source_chunk_ids is empty."""
        mock = _make_qwen_mock()
        dedup = EntityDeduplicator(mock)

        e1 = _make_entity(entity_id="e1", name="A", canonical_key="a")
        e2 = _make_entity(entity_id="e2", name="B", canonical_key="b", entity_type=EntityType.WORK)
        rel = _make_relation(source_entity_id="e1", target_entity_id="e2", source_chunk_ids=[])

        chunk = _make_chunk_result("c1", [e1, e2], [rel])
        merged_rels = dedup.merge_relations([chunk], [e1, e2])

        assert merged_rels[0].source_chunk_ids == ["c1"]


# ── Gleaning tests (Phase 7) ────────────────────────────────────────────


class TestGleaning:
    """Tests for gleaning (multi-round entity extraction) — Phase 7."""

    def test_gleaning_disabled_noop(self):
        """rounds=0 produces identical output — chat called exactly once (for initial extraction)."""
        mock = _make_qwen_mock(VALID_LLM_RESPONSE)
        extractor = EntityExtractor(mock, gleaning_rounds=0)
        result, trace = extractor.extract("chunk-001", "doc-001", "John Smith wrote Yesterday")

        assert trace.validation_status == "valid"
        # Only 1 chat call = initial extraction, no gleaning
        assert mock.chat.call_count == 1
        assert len(result.entities) >= 2  # John Smith + Yesterday from VALID_LLM_RESPONSE

    def test_gleaning_one_round_finds_more(self):
        """Round 1 extracts additional entities that get merged into result."""
        # Initial extraction returns John Smith + Yesterday
        # Gleaning round returns a new entity (Organization: "BMI")
        gleaning_response = json.dumps(
            {
                "entities": [
                    {
                        "entity_id": "g1",
                        "type": "Organization",
                        "name": "BMI",
                        "canonical_key": "bmi",
                        "aliases": [],
                        "mentions": [{"text": "BMI", "start": 30, "end": 33}],
                        "confidence": 0.85,
                    }
                ],
                "relations": [],
            }
        )
        mock = MagicMock()
        mock.chat.side_effect = [VALID_LLM_RESPONSE, gleaning_response]

        extractor = EntityExtractor(mock, gleaning_rounds=1)
        result, trace = extractor.extract(
            "chunk-001", "doc-001", "John Smith wrote Yesterday for BMI"
        )

        assert trace.validation_status == "valid"
        assert mock.chat.call_count == 2  # initial + 1 gleaning round
        names = {e.name for e in result.entities}
        assert "BMI" in names
        assert "John Smith" in names
        assert "Yesterday" in names

    def test_merge_deduplicates_by_key(self):
        """Same entity returned in gleaning round is merged, not duplicated."""
        # Gleaning returns "John Smith" again (same canonical_key) with extra description
        gleaning_response = json.dumps(
            {
                "entities": [
                    {
                        "entity_id": "g1",
                        "type": "Person",
                        "name": "John Smith",
                        "canonical_key": "john_smith",
                        "aliases": ["Johnny"],
                        "mentions": [{"text": "John Smith", "start": 0, "end": 10}],
                        "confidence": 0.98,
                        "description": "A prolific songwriter",
                    }
                ],
                "relations": [],
            }
        )
        mock = MagicMock()
        mock.chat.side_effect = [VALID_LLM_RESPONSE, gleaning_response]

        extractor = EntityExtractor(mock, gleaning_rounds=1)
        result, trace = extractor.extract("chunk-001", "doc-001", "John Smith wrote Yesterday")

        assert trace.validation_status == "valid"
        # Count Person entities named John Smith — should be exactly 1 (deduped)
        john_entities = [e for e in result.entities if e.canonical_key == "john_smith"]
        assert len(john_entities) == 1
        # Should have higher confidence from merge (max)
        assert john_entities[0].confidence == 0.98
        # Should have "Johnny" in aliases from gleaning
        assert "Johnny" in john_entities[0].aliases

    def test_merge_combines_descriptions(self):
        """Descriptions from initial and gleaning rounds are combined."""
        # Create a response where initial entity has description "Original desc"
        initial_response = json.dumps(
            {
                "chunk_id": "c1",
                "entities": [
                    {
                        "entity_id": "e1",
                        "type": "Person",
                        "name": "Alice",
                        "canonical_key": "alice",
                        "aliases": [],
                        "mentions": [{"text": "Alice", "start": 0, "end": 5}],
                        "confidence": 0.9,
                        "description": "Original desc",
                    }
                ],
                "relations": [],
            }
        )
        gleaning_response = json.dumps(
            {
                "entities": [
                    {
                        "entity_id": "g1",
                        "type": "Person",
                        "name": "Alice",
                        "canonical_key": "alice",
                        "aliases": [],
                        "mentions": [],
                        "confidence": 0.8,
                        "description": "Additional info",
                    }
                ],
                "relations": [],
            }
        )
        mock = MagicMock()
        mock.chat.side_effect = [initial_response, gleaning_response]

        extractor = EntityExtractor(mock, gleaning_rounds=1)
        result, _ = extractor.extract("c1", "d1", "Alice is here")

        alice_entities = [e for e in result.entities if e.canonical_key == "alice"]
        assert len(alice_entities) == 1
        assert "Original desc" in alice_entities[0].description
        assert "Additional info" in alice_entities[0].description

    def test_merge_accumulates_aliases(self):
        """Aliases from initial and gleaning rounds are unioned."""
        initial_response = json.dumps(
            {
                "chunk_id": "c1",
                "entities": [
                    {
                        "entity_id": "e1",
                        "type": "Person",
                        "name": "Robert",
                        "canonical_key": "robert",
                        "aliases": ["Rob"],
                        "mentions": [{"text": "Robert", "start": 0, "end": 6}],
                        "confidence": 0.9,
                    }
                ],
                "relations": [],
            }
        )
        gleaning_response = json.dumps(
            {
                "entities": [
                    {
                        "entity_id": "g1",
                        "type": "Person",
                        "name": "Robert",
                        "canonical_key": "robert",
                        "aliases": ["Bob", "Bobby"],
                        "mentions": [],
                        "confidence": 0.85,
                    }
                ],
                "relations": [],
            }
        )
        mock = MagicMock()
        mock.chat.side_effect = [initial_response, gleaning_response]

        extractor = EntityExtractor(mock, gleaning_rounds=1)
        result, _ = extractor.extract("c1", "d1", "Robert aka Rob and Bob")

        robert = [e for e in result.entities if e.canonical_key == "robert"][0]
        assert "Rob" in robert.aliases
        assert "Bob" in robert.aliases
        assert "Bobby" in robert.aliases
        # Name itself should not be in aliases
        assert "Robert" not in robert.aliases

    def test_gleaning_respects_max_rounds(self):
        """Stops at configured limit even if every round finds something."""
        # Each gleaning round returns a new entity
        initial = VALID_LLM_RESPONSE
        round1 = json.dumps(
            {
                "entities": [
                    {
                        "entity_id": "g1",
                        "type": "Organization",
                        "name": "BMI",
                        "canonical_key": "bmi",
                        "aliases": [],
                        "mentions": [{"text": "BMI", "start": 30, "end": 33}],
                        "confidence": 0.8,
                    }
                ],
                "relations": [],
            }
        )
        round2 = json.dumps(
            {
                "entities": [
                    {
                        "entity_id": "g2",
                        "type": "Territory",
                        "name": "USA",
                        "canonical_key": "usa",
                        "aliases": [],
                        "mentions": [{"text": "USA", "start": 40, "end": 43}],
                        "confidence": 0.7,
                    }
                ],
                "relations": [],
            }
        )
        mock = MagicMock()
        mock.chat.side_effect = [initial, round1, round2]

        extractor = EntityExtractor(mock, gleaning_rounds=2)
        result, _ = extractor.extract("c1", "d1", "John Smith wrote Yesterday for BMI in the USA")

        # 1 initial + 2 gleaning rounds = 3 calls
        assert mock.chat.call_count == 3
        names = {e.name for e in result.entities}
        assert "BMI" in names
        assert "USA" in names

    def test_gleaning_empty_round_stops(self):
        """If a round finds nothing, no more rounds are attempted."""
        # Round 1 returns empty → should stop, never reach round 2
        empty_response = json.dumps({"entities": [], "relations": []})
        mock = MagicMock()
        mock.chat.side_effect = [VALID_LLM_RESPONSE, empty_response]

        extractor = EntityExtractor(
            mock, gleaning_rounds=3
        )  # configured for 3, but should stop after 1
        result, trace = extractor.extract("c1", "d1", "John Smith wrote Yesterday")

        assert trace.validation_status == "valid"
        # 1 initial + 1 gleaning (empty) = 2 calls total (NOT 4)
        assert mock.chat.call_count == 2
