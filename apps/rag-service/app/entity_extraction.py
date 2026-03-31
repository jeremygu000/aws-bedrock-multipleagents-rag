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

from json_repair import repair_json
from pydantic import ValidationError

from .entity_extraction_models import (
    ChunkExtractionResult,
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionTrace,
    Mention,
)
from .prompts import (
    ENTITY_DESCRIPTION_SUMMARIZE_SYSTEM_PROMPT,
    ENTITY_DESCRIPTION_SUMMARIZE_USER_PROMPT_TEMPLATE,
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_USER_PROMPT_TEMPLATE,
    GLEANING_SYSTEM_PROMPT,
    GLEANING_USER_PROMPT_TEMPLATE,
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

    def __init__(self, qwen_client: QwenClient, gleaning_rounds: int = 0) -> None:
        self._qwen = qwen_client
        self._gleaning_rounds = gleaning_rounds

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

        if self._gleaning_rounds > 0:
            current_entities = list(llm_result.entities)
            current_relations = list(llm_result.relations)

            for round_num in range(1, self._gleaning_rounds + 1):
                new_entities, new_relations = self._gleaning_round(
                    chunk_id, chunk_text, current_entities, current_relations, round_num
                )
                if not new_entities and not new_relations:
                    break
                current_entities = self._merge_gleaned_entities(current_entities, new_entities)
                current_relations = self._merge_gleaned_relations(current_relations, new_relations)

            llm_result = llm_result.model_copy(
                update={
                    "entities": current_entities,
                    "relations": current_relations,
                }
            )

        merged_entities = self._merge_rule_entities(rule_entities, llm_result.entities)
        llm_result = llm_result.model_copy(update={"entities": merged_entities})

        logger.info(
            "Extraction complete chunk %s: entities=%d (rule=%d, llm=%d), relations=%d",
            chunk_id,
            len(merged_entities),
            len(rule_entities),
            len(llm_result.entities) - len(rule_entities),
            len(llm_result.relations),
        )

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
        """Stage 2: LLM structured extraction with multi-layer repair.

        Parsing pipeline:
        1. Extract JSON text from LLM response (strip fences, find object boundaries)
        2. Try direct json.loads + Pydantic validation
        3. On JSON syntax error: deterministic repair via json-repair library
        4. On Pydantic validation error: lenient parse (filter bad relations)
        5. Last resort: LLM repair call (slow but handles semantic issues)
        """
        user_prompt = ENTITY_EXTRACTION_USER_PROMPT_TEMPLATE.format(
            chunk_id=chunk_id, doc_id=doc_id, chunk_text=chunk_text
        )

        raw_response = self._qwen.chat(ENTITY_EXTRACTION_SYSTEM_PROMPT, user_prompt)
        extracted_text = self._extract_json_text(raw_response)

        try:
            raw_data = json.loads(extracted_text)
            raw_entity_count = len(raw_data.get("entities", []))
            raw_relation_count = len(raw_data.get("relations", []))
        except Exception:
            raw_entity_count = -1
            raw_relation_count = -1

        # Layer 1: Direct parse
        try:
            result = self._parse_extraction_json(extracted_text)
            result = self._resolve_relation_endpoints(result)
            logger.info(
                "Extraction chunk %s: path=direct, raw_entities=%d, raw_relations=%d, "
                "parsed_entities=%d, parsed_relations=%d",
                chunk_id,
                raw_entity_count,
                raw_relation_count,
                len(result.entities),
                len(result.relations),
            )
            return result
        except json.JSONDecodeError as json_err:
            logger.info("Direct JSON parse failed for chunk %s: %s", chunk_id, json_err)
            repaired = repair_json(extracted_text, return_objects=False)
            if not isinstance(repaired, str):
                repaired = json.dumps(repaired)
            try:
                result = self._parse_extraction_json(repaired)
                result = self._resolve_relation_endpoints(result)
                logger.info(
                    "Extraction chunk %s: path=json_repair, raw_entities=%d, raw_relations=%d, "
                    "parsed_entities=%d, parsed_relations=%d",
                    chunk_id,
                    raw_entity_count,
                    raw_relation_count,
                    len(result.entities),
                    len(result.relations),
                )
                return result
            except json.JSONDecodeError as repair_err:
                logger.info(
                    "json-repair also failed for chunk %s: %s, trying LLM repair",
                    chunk_id,
                    repair_err,
                )
                result = self._llm_repair_and_parse(extracted_text, str(json_err))
                result = self._resolve_relation_endpoints(result)
                logger.info(
                    "Extraction chunk %s: path=llm_repair(json), raw_entities=%d, raw_relations=%d, "
                    "parsed_entities=%d, parsed_relations=%d",
                    chunk_id,
                    raw_entity_count,
                    raw_relation_count,
                    len(result.entities),
                    len(result.relations),
                )
                return result
            except ValidationError:
                result = self._lenient_parse(repaired)
                logger.info(
                    "Extraction chunk %s: path=lenient(after_repair), raw_entities=%d, raw_relations=%d, "
                    "parsed_entities=%d, parsed_relations=%d",
                    chunk_id,
                    raw_entity_count,
                    raw_relation_count,
                    len(result.entities),
                    len(result.relations),
                )
                return result
        except ValidationError as val_err:
            logger.info("Pydantic validation failed for chunk %s: %s", chunk_id, val_err)
            try:
                result = self._lenient_parse(extracted_text)
                logger.info(
                    "Extraction chunk %s: path=lenient(validation), raw_entities=%d, raw_relations=%d, "
                    "parsed_entities=%d, parsed_relations=%d",
                    chunk_id,
                    raw_entity_count,
                    raw_relation_count,
                    len(result.entities),
                    len(result.relations),
                )
                return result
            except Exception as lenient_err:
                logger.info(
                    "Lenient parse also failed for chunk %s: %s, trying LLM repair",
                    chunk_id,
                    lenient_err,
                )
                result = self._llm_repair_and_parse(extracted_text, str(val_err))
                result = self._resolve_relation_endpoints(result)
                logger.info(
                    "Extraction chunk %s: path=llm_repair(validation), raw_entities=%d, raw_relations=%d, "
                    "parsed_entities=%d, parsed_relations=%d",
                    chunk_id,
                    raw_entity_count,
                    raw_relation_count,
                    len(result.entities),
                    len(result.relations),
                )
                return result

    def _llm_repair_and_parse(self, raw_json: str, validation_error: str) -> ChunkExtractionResult:
        """Last resort: ask the LLM to fix the JSON (spec section 2)."""
        repair_prompt = JSON_REPAIR_USER_PROMPT_TEMPLATE.format(
            raw_json=raw_json, validation_error=validation_error
        )
        repaired_response = self._qwen.chat(JSON_REPAIR_SYSTEM_PROMPT, repair_prompt)
        repaired_text = self._extract_json_text(repaired_response)

        try:
            return self._parse_extraction_json(repaired_text)
        except json.JSONDecodeError:
            repaired = repair_json(repaired_text, return_objects=False)
            return self._parse_extraction_json(repaired)

    def _parse_extraction_json(self, raw_json: str) -> ChunkExtractionResult:
        data = json.loads(raw_json)
        return ChunkExtractionResult.model_validate(data)

    def _lenient_parse(self, raw_json: str) -> ChunkExtractionResult:
        """Parse JSON leniently: keep valid entities, filter invalid relations.

        When Pydantic strict validation fails (e.g., bad relation endpoint IDs,
        invalid enum values), this extracts what it can instead of failing entirely.
        """
        data = json.loads(raw_json)

        raw_entity_count = len(data.get("entities", []))
        raw_relation_count = len(data.get("relations", []))
        skipped_entities = 0
        dangling_relations = 0
        invalid_relations = 0

        entities: list[ExtractedEntity] = []
        for raw_entity in data.get("entities", []):
            if not isinstance(raw_entity, dict):
                skipped_entities += 1
                continue
            try:
                entities.append(ExtractedEntity.model_validate(raw_entity))
            except (ValidationError, Exception) as exc:
                skipped_entities += 1
                name = raw_entity.get("name", "?") if isinstance(raw_entity, dict) else "?"
                logger.debug("Skipping invalid entity: %s — %s", name, exc)

        entity_ids = {e.entity_id for e in entities}
        entity_name_to_id: dict[str, str] = {}
        for e in entities:
            name_lower = e.name.lower()
            if name_lower not in entity_name_to_id:
                entity_name_to_id[name_lower] = e.entity_id
            for alias in e.aliases:
                alias_lower = alias.lower()
                if alias_lower not in entity_name_to_id:
                    entity_name_to_id[alias_lower] = e.entity_id

        name_resolved_count = 0
        relations: list[ExtractedRelation] = []
        for raw_rel in data.get("relations", []):
            if not isinstance(raw_rel, dict):
                invalid_relations += 1
                continue
            try:
                rel = ExtractedRelation.model_validate(raw_rel)
                src_id = rel.source_entity_id
                tgt_id = rel.target_entity_id

                # Fall back to entity-name matching when entity_id doesn't resolve
                if src_id not in entity_ids:
                    resolved = entity_name_to_id.get(src_id.lower())
                    if resolved:
                        src_id = resolved
                if tgt_id not in entity_ids:
                    resolved = entity_name_to_id.get(tgt_id.lower())
                    if resolved:
                        tgt_id = resolved

                if src_id in entity_ids and tgt_id in entity_ids:
                    if src_id != rel.source_entity_id or tgt_id != rel.target_entity_id:
                        name_resolved_count += 1
                        rel = rel.model_copy(
                            update={
                                "source_entity_id": src_id,
                                "target_entity_id": tgt_id,
                            }
                        )
                    relations.append(rel)
                else:
                    dangling_relations += 1
                    logger.debug(
                        "Skipping relation with dangling ref: %s -> %s (valid IDs: %s)",
                        raw_rel.get("source_entity_id", "?"),
                        raw_rel.get("target_entity_id", "?"),
                        entity_ids,
                    )
            except (ValidationError, Exception) as exc:
                invalid_relations += 1
                logger.debug("Skipping invalid relation: %s", exc)

        chunk_id = data.get("chunk_id", "")
        dropped_relations = dangling_relations + invalid_relations
        if skipped_entities > 0 or dropped_relations > 0 or name_resolved_count > 0:
            logger.info(
                "Lenient parse chunk %s: entities=%d/%d (dropped %d), "
                "relations=%d/%d (dangling=%d, invalid=%d, name_resolved=%d)",
                chunk_id,
                len(entities),
                raw_entity_count,
                skipped_entities,
                len(relations),
                raw_relation_count,
                dangling_relations,
                invalid_relations,
                name_resolved_count,
            )
        return ChunkExtractionResult(chunk_id=chunk_id, entities=entities, relations=relations)

    @staticmethod
    def _extract_json_text(text: str) -> str:
        """Extract JSON from LLM response, handling markdown fences and surrounding prose.

        Handles:
        - Clean JSON (no wrapping)
        - ```json ... ``` fences (with or without language tag)
        - Partial fences (opening ``` without closing, or vice versa)
        - Prose before/after JSON object: "Here is the result:\n{...}\nDone"
        """
        stripped = text.strip()

        fence_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)
        fence_match = fence_pattern.search(stripped)
        if fence_match:
            stripped = fence_match.group(1).strip()

        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            return stripped

        first_brace = stripped.find("{")
        last_brace = stripped.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            return stripped[first_brace : last_brace + 1]

        return stripped

    def _gleaning_round(
        self,
        chunk_id: str,
        chunk_text: str,
        existing_entities: list[ExtractedEntity],
        existing_relations: list[ExtractedRelation],
        round_num: int,
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        existing_entity_summary = "\n".join(
            (
                f"- {e.name} ({e.type.value}): {e.description}"
                if e.description
                else f"- {e.name} ({e.type.value})"
            )
            for e in existing_entities
        )
        existing_relation_summary = (
            "\n".join(
                f"- {r.source_entity_id} --[{r.type.value}]--> {r.target_entity_id}: {r.evidence}"
                for r in existing_relations
            )
            or "(none)"
        )

        user_prompt = GLEANING_USER_PROMPT_TEMPLATE.format(
            existing_entities=existing_entity_summary or "(none)",
            existing_relations=existing_relation_summary,
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            round=round_num,
        )

        try:
            raw_response = self._qwen.chat(GLEANING_SYSTEM_PROMPT, user_prompt)
            raw_json = self._extract_json_text(raw_response)
            data = json.loads(raw_json)

            new_entities = [ExtractedEntity.model_validate(e) for e in data.get("entities", [])]
            new_relations = [ExtractedRelation.model_validate(r) for r in data.get("relations", [])]

            logger.info(
                "Gleaning round %d for chunk %s: found %d new entities, %d new relations",
                round_num,
                chunk_id,
                len(new_entities),
                len(new_relations),
            )
            return new_entities, new_relations
        except Exception as exc:
            logger.warning("Gleaning round %d failed for chunk %s: %s", round_num, chunk_id, exc)
            return [], []

    @staticmethod
    def _merge_gleaned_entities(
        existing: list[ExtractedEntity],
        new: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        by_key: dict[str, ExtractedEntity] = {}
        for e in existing:
            key = e.canonical_key if e.canonical_key else e.name.lower()
            by_key[key] = e

        for e in new:
            key = e.canonical_key if e.canonical_key else e.name.lower()
            if key in by_key:
                old = by_key[key]
                merged_desc = (
                    f"{old.description} {e.description}".strip()
                    if e.description
                    else old.description
                )
                merged_aliases = sorted(set(old.aliases + e.aliases) - {old.name})
                merged_mentions = old.mentions + e.mentions
                by_key[key] = old.model_copy(
                    update={
                        "description": merged_desc,
                        "confidence": max(old.confidence, e.confidence),
                        "aliases": merged_aliases,
                        "mentions": merged_mentions,
                    }
                )
            else:
                by_key[key] = e

        return list(by_key.values())

    @staticmethod
    def _merge_gleaned_relations(
        existing: list[ExtractedRelation],
        new: list[ExtractedRelation],
    ) -> list[ExtractedRelation]:
        seen: set[tuple[str, str, str]] = set()
        for r in existing:
            seen.add((r.source_entity_id, r.target_entity_id, r.type.value))

        merged = list(existing)
        for r in new:
            key = (r.source_entity_id, r.target_entity_id, r.type.value)
            if key not in seen:
                seen.add(key)
                merged.append(r)

        return merged

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

    @staticmethod
    def _resolve_relation_endpoints(result: ChunkExtractionResult) -> ChunkExtractionResult:
        """Resolve name-based relation endpoints to entity_id values."""
        entity_ids = {e.entity_id for e in result.entities}
        entity_name_to_id: dict[str, str] = {}
        for e in result.entities:
            name_lower = e.name.lower()
            if name_lower not in entity_name_to_id:
                entity_name_to_id[name_lower] = e.entity_id
            for alias in e.aliases:
                alias_lower = alias.lower()
                if alias_lower not in entity_name_to_id:
                    entity_name_to_id[alias_lower] = e.entity_id

        resolved_relations: list[ExtractedRelation] = []
        resolved_count = 0
        dropped_count = 0
        for rel in result.relations:
            src_id = rel.source_entity_id
            tgt_id = rel.target_entity_id

            if src_id not in entity_ids:
                resolved = entity_name_to_id.get(src_id.lower())
                if resolved:
                    src_id = resolved
            if tgt_id not in entity_ids:
                resolved = entity_name_to_id.get(tgt_id.lower())
                if resolved:
                    tgt_id = resolved

            if src_id in entity_ids and tgt_id in entity_ids:
                if src_id != rel.source_entity_id or tgt_id != rel.target_entity_id:
                    resolved_count += 1
                    rel = rel.model_copy(
                        update={"source_entity_id": src_id, "target_entity_id": tgt_id}
                    )
                resolved_relations.append(rel)
            else:
                dropped_count += 1

        if resolved_count > 0 or dropped_count > 0:
            logger.info(
                "Relation endpoint resolution chunk %s: kept=%d, name_resolved=%d, dropped=%d",
                result.chunk_id,
                len(resolved_relations),
                resolved_count,
                dropped_count,
            )

        return result.model_copy(update={"relations": resolved_relations})


def _entity_dedup_key(entity: ExtractedEntity) -> str:
    """Compute the deduplication key for an entity.

    Uses canonical_key if available, otherwise falls back to lowercase name.
    """
    return entity.canonical_key if entity.canonical_key else entity.name.lower()


def _estimate_token_count(text: str) -> int:
    """Rough token count estimation (~4 chars per token for English text)."""
    return len(text) // 4


class EntityDeduplicator:
    """Cross-chunk entity and relation deduplication with LLM-assisted description merging.

    Merges entities from multiple chunk extraction results into a unified set,
    deduplicating by canonical_key (or normalized name). When merged descriptions
    exceed a token threshold, uses LLM to summarize them.
    """

    def __init__(self, qwen_client: QwenClient, summary_max_tokens: int = 500) -> None:
        """Initialize the deduplicator.

        Args:
            qwen_client: Qwen client for LLM description summarization.
            summary_max_tokens: Token threshold above which merged descriptions
                are summarized by LLM instead of simple concatenation.
        """
        self._qwen = qwen_client
        self._summary_max_tokens = summary_max_tokens

    def merge_entities(
        self,
        chunk_results: list[ChunkExtractionResult],
    ) -> list[ExtractedEntity]:
        """Merge entities across multiple chunk extraction results.

        Deduplication strategy:
        1. Group entities by dedup key (canonical_key or name.lower())
        2. For each group, merge into a single entity:
           - Keep the highest confidence score
           - Union all aliases and mentions
           - Track all source chunk IDs
           - Merge descriptions: keep longer if under token limit, else LLM summarize

        Args:
            chunk_results: Extraction results from multiple chunks of the same document.

        Returns:
            Deduplicated list of merged entities.
        """
        groups: dict[str, list[ExtractedEntity]] = {}

        for result in chunk_results:
            for entity in result.entities:
                key = _entity_dedup_key(entity)
                if key not in groups:
                    groups[key] = []
                # Stamp source_chunk_ids if not already set
                if not entity.source_chunk_ids:
                    entity = entity.model_copy(update={"source_chunk_ids": [result.chunk_id]})
                groups[key].append(entity)

        merged: list[ExtractedEntity] = []
        for _key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
                continue
            merged.append(self._merge_entity_group(group))

        return merged

    def _merge_entity_group(self, group: list[ExtractedEntity]) -> ExtractedEntity:
        """Merge a group of duplicate entities into one consolidated entity.

        Uses the first entity as the base, then folds in data from subsequent entities.
        """
        base = group[0]
        all_aliases: set[str] = set(base.aliases)
        all_mentions: list[Mention] = list(base.mentions)
        all_chunk_ids: list[str] = list(base.source_chunk_ids)
        descriptions: list[str] = [base.description] if base.description else []
        best_confidence = base.confidence

        for entity in group[1:]:
            all_aliases.update(entity.aliases)
            # Add name as alias if different from base name
            if entity.name.lower() != base.name.lower():
                all_aliases.add(entity.name)
            all_mentions.extend(entity.mentions)
            for cid in entity.source_chunk_ids:
                if cid not in all_chunk_ids:
                    all_chunk_ids.append(cid)
            if entity.description and entity.description not in descriptions:
                descriptions.append(entity.description)
            best_confidence = max(best_confidence, entity.confidence)

        all_aliases.discard(base.name)

        merged_description = self._merge_descriptions(
            descriptions, entity_name=base.name, entity_type=base.type.value
        )

        return base.model_copy(
            update={
                "aliases": sorted(all_aliases),
                "mentions": all_mentions,
                "source_chunk_ids": all_chunk_ids,
                "description": merged_description,
                "confidence": best_confidence,
            }
        )

    def _merge_descriptions(
        self,
        descriptions: list[str],
        entity_name: str,
        entity_type: str,
    ) -> str:
        """Merge multiple descriptions, using LLM summarization if needed.

        If combined descriptions are under the token threshold, returns the longest one.
        Otherwise, asks the LLM to consolidate them into a single coherent summary.
        """
        if not descriptions:
            return ""
        if len(descriptions) == 1:
            return descriptions[0]

        combined = "\n".join(descriptions)
        if _estimate_token_count(combined) <= self._summary_max_tokens:
            return max(descriptions, key=len)

        try:
            numbered = "\n".join(f"- {d}" for d in descriptions)
            user_prompt = ENTITY_DESCRIPTION_SUMMARIZE_USER_PROMPT_TEMPLATE.format(
                entity_name=entity_name,
                entity_type=entity_type,
                descriptions=numbered,
            )
            summary = self._qwen.chat(ENTITY_DESCRIPTION_SUMMARIZE_SYSTEM_PROMPT, user_prompt)
            return summary.strip()
        except Exception as exc:
            logger.warning(
                "LLM description summarization failed for entity '%s': %s",
                entity_name,
                exc,
            )
            return max(descriptions, key=len)

    def merge_relations(
        self,
        chunk_results: list[ChunkExtractionResult],
        merged_entities: list[ExtractedEntity],
    ) -> list[ExtractedRelation]:
        """Merge relations across multiple chunk extraction results.

        Deduplication strategy:
        1. Resolve entity IDs to entity names (for stable cross-chunk matching)
        2. Group relations by (source_name, target_name, type)
        3. For each group, merge into a single relation:
           - Accumulate weight (count of occurrences)
           - Keep the highest confidence score
           - Merge evidence strings
           - Track all source chunk IDs

        Args:
            chunk_results: Extraction results from multiple chunks.
            merged_entities: The deduplicated entity list (for ID→name resolution).

        Returns:
            Deduplicated list of merged relations.
        """
        # Build entity_id → name lookup per chunk for ID resolution
        entity_id_to_name: dict[str, str] = {}
        for result in chunk_results:
            for entity in result.entities:
                entity_id_to_name[entity.entity_id] = entity.name.lower()

        # Build merged entity name → entity_id lookup for re-mapping
        merged_name_to_id: dict[str, str] = {}
        for entity in merged_entities:
            merged_name_to_id[entity.name.lower()] = entity.entity_id
            if entity.canonical_key:
                merged_name_to_id[entity.canonical_key] = entity.entity_id

        # Group relations by (source_name, target_name, type)
        groups: dict[tuple[str, str, str], list[tuple[ExtractedRelation, str]]] = {}

        for result in chunk_results:
            for rel in result.relations:
                src_name = entity_id_to_name.get(rel.source_entity_id, rel.source_entity_id)
                tgt_name = entity_id_to_name.get(rel.target_entity_id, rel.target_entity_id)
                group_key = (src_name, tgt_name, rel.type.value)
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append((rel, result.chunk_id))

        merged: list[ExtractedRelation] = []
        for (src_name, tgt_name, _rel_type), group in groups.items():
            base_rel, base_chunk_id = group[0]

            # Re-map entity IDs to merged entity IDs
            new_source_id = merged_name_to_id.get(src_name, base_rel.source_entity_id)
            new_target_id = merged_name_to_id.get(tgt_name, base_rel.target_entity_id)

            if len(group) == 1:
                source_chunks = base_rel.source_chunk_ids or [base_chunk_id]
                merged.append(
                    base_rel.model_copy(
                        update={
                            "source_entity_id": new_source_id,
                            "target_entity_id": new_target_id,
                            "source_chunk_ids": source_chunks,
                        }
                    )
                )
                continue

            # Merge multiple occurrences
            all_evidence: list[str] = []
            all_chunk_ids: list[str] = []
            best_confidence = 0.0

            for rel, chunk_id in group:
                if rel.evidence and rel.evidence not in all_evidence:
                    all_evidence.append(rel.evidence)
                if chunk_id not in all_chunk_ids:
                    all_chunk_ids.append(chunk_id)
                for cid in rel.source_chunk_ids:
                    if cid not in all_chunk_ids:
                        all_chunk_ids.append(cid)
                best_confidence = max(best_confidence, rel.confidence)

            merged.append(
                base_rel.model_copy(
                    update={
                        "source_entity_id": new_source_id,
                        "target_entity_id": new_target_id,
                        "evidence": " | ".join(all_evidence) if all_evidence else "",
                        "confidence": best_confidence,
                        "weight": float(len(group)),
                        "source_chunk_ids": all_chunk_ids,
                    }
                )
            )

        return merged
