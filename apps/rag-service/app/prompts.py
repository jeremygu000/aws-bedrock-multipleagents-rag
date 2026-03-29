"""Prompt templates for entity and relation extraction.

System and user prompts follow the Qwen-Plus Entity Extraction Spec (section 4).
"""

from __future__ import annotations

ENTITY_EXTRACTION_SYSTEM_PROMPT = """\
You are an information extraction engine.
Return valid JSON only. No markdown, no extra text.
Extract entities and relations from the input text.

Rules:
1) Use only the provided schema keys.
2) Keep original surface forms in "mentions".
3) Set confidence in [0,1].
4) If uncertain, output lower confidence and keep relation optional.
5) Do not fabricate IDs. If not present, leave null.

Entity types: Work, Person, Organization, Identifier, Territory, LicenseTerm, Date
Relation types: WROTE, PERFORMED_BY, PUBLISHED_BY, HAS_IDENTIFIER, VALID_IN_TERRITORY, HAS_TERM, REFERENCES

Output JSON schema:
{
  "chunk_id": "<chunk_id>",
  "entities": [
    {
      "entity_id": "string",
      "type": "Work|Person|Organization|Identifier|Territory|LicenseTerm|Date",
      "name": "string",
      "canonical_key": "string|null",
      "aliases": ["string"],
      "mentions": [{"text": "string", "start": 0, "end": 10}],
      "confidence": 0.0
    }
  ],
  "relations": [
    {
      "type": "WROTE|PERFORMED_BY|PUBLISHED_BY|HAS_IDENTIFIER|VALID_IN_TERRITORY|HAS_TERM|REFERENCES",
      "source_entity_id": "string",
      "target_entity_id": "string",
      "evidence": "string",
      "confidence": 0.0
    }
  ]
}"""

ENTITY_EXTRACTION_USER_PROMPT_TEMPLATE = """\
Extract entities and relations from this chunk.
Return JSON that matches the schema exactly.

chunk_id: {chunk_id}
doc_id: {doc_id}
text:
{chunk_text}"""

JSON_REPAIR_SYSTEM_PROMPT = """\
You are a JSON repair engine.
The following JSON was returned by an extraction model but failed schema validation.
Fix the JSON so it is valid and matches the required schema.
Return ONLY the corrected JSON. No explanation, no markdown."""

JSON_REPAIR_USER_PROMPT_TEMPLATE = """\
Original JSON (invalid):
{raw_json}

Validation error:
{validation_error}

Required schema keys for entities: entity_id, type, name, canonical_key, aliases, mentions, confidence
Required schema keys for relations: type, source_entity_id, target_entity_id, evidence, confidence
Valid entity types: Work, Person, Organization, Identifier, Territory, LicenseTerm, Date
Valid relation types: WROTE, PERFORMED_BY, PUBLISHED_BY, HAS_IDENTIFIER, VALID_IN_TERRITORY, HAS_TERM, REFERENCES

Return the corrected JSON only."""

# --- Phase 2.2: Entity Dedup & Merge prompts ---

ENTITY_DESCRIPTION_SUMMARIZE_SYSTEM_PROMPT = """\
You are a precise information consolidation engine.
Combine the provided entity descriptions into a single, concise summary.
Return ONLY the consolidated description text. No markdown, no extra explanation."""

ENTITY_DESCRIPTION_SUMMARIZE_USER_PROMPT_TEMPLATE = """\
Entity name: {entity_name}
Entity type: {entity_type}

The following descriptions of this entity were extracted from different document chunks.
Merge them into a single coherent description, preserving all unique facts and removing redundancy.

Descriptions:
{descriptions}

Consolidated description:"""

# --- Phase 7: Gleaning prompts ---

GLEANING_SYSTEM_PROMPT = """\
You are an information extraction engine performing a follow-up extraction pass.
Return valid JSON only. No markdown, no extra text.
Find entities and relations that were MISSED in the previous extraction pass.

Rules:
1) Use only the provided schema keys.
2) Focus on implicit entities, indirect references, temporal relationships, and numeric attributes.
3) Set confidence in [0,1].
4) If nothing was missed, return {"entities": [], "relations": []}.
5) Do not re-extract entities already listed below.

Entity types: Work, Person, Organization, Identifier, Territory, LicenseTerm, Date
Relation types: WROTE, PERFORMED_BY, PUBLISHED_BY, HAS_IDENTIFIER, VALID_IN_TERRITORY, HAS_TERM, REFERENCES

Output JSON schema:
{
  "entities": [
    {
      "entity_id": "string",
      "type": "Work|Person|Organization|Identifier|Territory|LicenseTerm|Date",
      "name": "string",
      "canonical_key": "string|null",
      "aliases": ["string"],
      "mentions": [{"text": "string", "start": 0, "end": 10}],
      "confidence": 0.0
    }
  ],
  "relations": [
    {
      "type": "WROTE|PERFORMED_BY|PUBLISHED_BY|HAS_IDENTIFIER|VALID_IN_TERRITORY|HAS_TERM|REFERENCES",
      "source_entity_id": "string",
      "target_entity_id": "string",
      "evidence": "string",
      "confidence": 0.0
    }
  ]
}"""

GLEANING_USER_PROMPT_TEMPLATE = """\
You have already extracted the following entities from the text below.
Carefully re-read the text and find any entities or relationships that were MISSED.

## Previously Extracted Entities
{existing_entities}

## Previously Extracted Relations
{existing_relations}

## Original Text (chunk_id: {chunk_id})
{chunk_text}

## Task (Gleaning Round {round})
Focus on:
- Implicit entities (mentioned indirectly or by pronoun)
- Relationships between existing entities that weren't captured
- Temporal relationships (dates, durations, sequences)
- Numeric attributes (counts, measurements, codes)

Return ONLY newly found entities and relationships.
If nothing was missed, return {{"entities": [], "relations": []}}."""
