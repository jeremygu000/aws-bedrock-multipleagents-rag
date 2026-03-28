"""Entity and relation extraction from document chunks.

Two-stage pipeline per spec (section 2):
1. Rule-first extraction — regex for deterministic fields (ISWC, ISRC, dates, identifiers)
2. LLM structured extraction — Qwen-Plus for entities, aliases, relations, and evidence spans

Retry logic: on schema validation failure, attempt JSON repair once.
If still invalid, mark as extraction_failed and skip.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from pydantic import ValidationError

from .entity_extraction_models import (
    ChunkExtractionResult,
    EntityType,
    ExtractedEntity,
    ExtractionTrace,
    Mention,
)
from .prompts import (
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_USER_PROMPT_TEMPLATE,
    JSON_REPAIR_SYSTEM_PROMPT,
    JSON_REPAIR_USER_PROMPT_TEMPLATE,
)

if TYPE_CHECKING:
    from .qwen_client import QwenClient

logger = logging.getLogger(__name__)

# Rule-based patterns for deterministic identifier extraction (spec section 2.1)
_IDENTIFIER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ISWC", re.compile(r"\bT-?\d{3}\.?\d{3}\.?\d{3}-?\d\b")),
    ("ISRC", re.compile(r"\b[A-Z]{2}-?\w{3}-?\d{2}-?\d{5}\b")),
    ("ISWC", re.compile(r"\bISWC\s*[:.]?\s*(T-?\d{3}\.?\d{3}\.?\d{3}-?\d)\b", re.IGNORECASE)),
    ("ISRC", re.compile(r"\bISRC\s*[:.]?\s*([A-Z]{2}-?\w{3}-?\d{2}-?\d{5})\b", re.IGNORECASE)),
]

# Date patterns: YYYY-MM-DD, DD/MM/YYYY, Month DD YYYY
_DATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"),
    re.compile(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b"),
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
]


class EntityExtractor:
    """Two-stage entity/relation extractor: rule-first + LLM structured extraction."""

    def __init__(self, qwen_client: QwenClient) -> None:
        self._qwen = qwen_client

    def extract(
        self,
        chunk_id: str,
        doc_id: str,
        chunk_text: str,
    ) -> tuple[ChunkExtractionResult, ExtractionTrace]:
        """Run two-stage extraction on a single chunk.

        Returns (extraction_result, trace) tuple. On failure, returns empty result
        with trace.validation_status set to 'extraction_failed'.
        """
        trace = ExtractionTrace(chunk_id=chunk_id, doc_id=doc_id)

        rule_entities = self._rule_based_extraction(chunk_text, chunk_id)

        try:
            llm_result = self._llm_extraction(chunk_id, doc_id, chunk_text)
        except Exception as exc:
            logger.warning("LLM extraction failed for chunk %s: %s", chunk_id, exc)
            trace.validation_status = "extraction_failed"
            trace.failure_reason = str(exc)
            return (
                ChunkExtractionResult(
                    chunk_id=chunk_id,
                    entities=rule_entities,
                    relations=[],
                ),
                trace,
            )

        merged_entities = self._merge_rule_entities(rule_entities, llm_result.entities)
        llm_result = llm_result.model_copy(update={"entities": merged_entities})

        trace.validation_status = "valid"
        return llm_result, trace

    def _rule_based_extraction(self, text: str, chunk_id: str) -> list[ExtractedEntity]:
        """Stage 1: Extract deterministic identifiers and dates via regex."""
        entities: list[ExtractedEntity] = []
        seen: set[str] = set()

        for label, pattern in _IDENTIFIER_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(1) if match.lastindex else match.group(0)
                dedup_key = f"{label}:{value}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                entity_id = f"rule_{label.lower()}_{len(entities)}"
                entities.append(
                    ExtractedEntity(
                        entity_id=entity_id,
                        type=EntityType.IDENTIFIER,
                        name=value,
                        canonical_key=value.replace("-", "").replace(".", ""),
                        aliases=[],
                        mentions=[
                            Mention(text=match.group(0), start=match.start(), end=match.end())
                        ],
                        confidence=1.0,
                    )
                )

        for pattern in _DATE_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(1) if match.lastindex else match.group(0)
                dedup_key = f"DATE:{value}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                entity_id = f"rule_date_{len(entities)}"
                entities.append(
                    ExtractedEntity(
                        entity_id=entity_id,
                        type=EntityType.DATE,
                        name=value,
                        canonical_key=None,
                        aliases=[],
                        mentions=[
                            Mention(text=match.group(0), start=match.start(), end=match.end())
                        ],
                        confidence=1.0,
                    )
                )

        return entities

    def _llm_extraction(self, chunk_id: str, doc_id: str, chunk_text: str) -> ChunkExtractionResult:
        """Stage 2: LLM structured extraction with retry-on-failure."""
        user_prompt = ENTITY_EXTRACTION_USER_PROMPT_TEMPLATE.format(
            chunk_id=chunk_id, doc_id=doc_id, chunk_text=chunk_text
        )

        raw_response = self._qwen.chat(ENTITY_EXTRACTION_SYSTEM_PROMPT, user_prompt)
        raw_json = self._strip_markdown_fences(raw_response)

        try:
            return self._parse_extraction_json(raw_json)
        except (json.JSONDecodeError, ValidationError) as first_error:
            logger.info(
                "First parse failed for chunk %s, attempting repair: %s", chunk_id, first_error
            )
            return self._repair_and_parse(raw_json, str(first_error))

    def _repair_and_parse(self, raw_json: str, validation_error: str) -> ChunkExtractionResult:
        """Retry once with a JSON repair prompt (spec section 2)."""
        repair_prompt = JSON_REPAIR_USER_PROMPT_TEMPLATE.format(
            raw_json=raw_json, validation_error=validation_error
        )
        repaired_response = self._qwen.chat(JSON_REPAIR_SYSTEM_PROMPT, repair_prompt)
        repaired_json = self._strip_markdown_fences(repaired_response)
        return self._parse_extraction_json(repaired_json)

    def _parse_extraction_json(self, raw_json: str) -> ChunkExtractionResult:
        data = json.loads(raw_json)
        return ChunkExtractionResult.model_validate(data)

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Strip ```json ... ``` fences that LLMs sometimes wrap around JSON output."""
        stripped = text.strip()
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1 :]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
        return stripped.strip()

    @staticmethod
    def _merge_rule_entities(
        rule_entities: list[ExtractedEntity],
        llm_entities: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        """Merge rule-based entities into LLM results, deduplicating by canonical_key."""
        llm_keys: set[str] = set()
        for entity in llm_entities:
            if entity.canonical_key:
                llm_keys.add(entity.canonical_key)
            llm_keys.add(entity.name.lower())

        merged = list(llm_entities)
        for rule_entity in rule_entities:
            key = rule_entity.canonical_key or rule_entity.name.lower()
            if key not in llm_keys:
                merged.append(rule_entity)

        return merged
