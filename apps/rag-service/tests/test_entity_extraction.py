import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.entity_extraction import EntityExtractor
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
    def test_entity_extraction_disabled_by_default(self) -> None:
        from app.config import Settings

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
