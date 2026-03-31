"""Prompt templates for entity and relation extraction.

System and user prompts follow the Qwen-Plus Entity Extraction Spec (section 4).
"""

from __future__ import annotations

ENTITY_EXTRACTION_SYSTEM_PROMPT = """\
You are an information extraction engine specialized in music publishing data.
Return valid JSON only. No markdown, no extra text.
Extract ALL entities and relations from the input text.

Rules:
1) Use only the provided schema keys.
2) Set confidence in [0,1].
3) If uncertain, output lower confidence but STILL include the relation.
4) Generate entity_id as sequential strings: "entity_0", "entity_1", etc.
5) Relations MUST reference entity_id values from the entities array.
   You may also use the entity "name" as source_entity_id or target_entity_id — the system will resolve names to IDs automatically.
6) Extract EVERY relationship you can infer from the text. For every entity pair that has a connection, add a relation.

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
      "mentions": [{"text": "string"}],
      "confidence": 0.0
    }
  ],
  "relations": [
    {
      "type": "WROTE|PERFORMED_BY|PUBLISHED_BY|HAS_IDENTIFIER|VALID_IN_TERRITORY|HAS_TERM|REFERENCES",
      "source_entity_id": "string (entity_id or entity name)",
      "target_entity_id": "string (entity_id or entity name)",
      "evidence": "string",
      "confidence": 0.0
    }
  ]
}

## Example

Input text: "Rushing Back by Flume ft. Vera Blue was published by Future Classic in Australia."

Output:
{
  "chunk_id": "example_chunk_0",
  "entities": [
    {"entity_id": "entity_0", "type": "Work", "name": "Rushing Back", "canonical_key": null, "aliases": [], "mentions": [{"text": "Rushing Back"}], "confidence": 0.95},
    {"entity_id": "entity_1", "type": "Person", "name": "Flume", "canonical_key": null, "aliases": [], "mentions": [{"text": "Flume"}], "confidence": 0.95},
    {"entity_id": "entity_2", "type": "Person", "name": "Vera Blue", "canonical_key": null, "aliases": [], "mentions": [{"text": "Vera Blue"}], "confidence": 0.90},
    {"entity_id": "entity_3", "type": "Organization", "name": "Future Classic", "canonical_key": null, "aliases": [], "mentions": [{"text": "Future Classic"}], "confidence": 0.90},
    {"entity_id": "entity_4", "type": "Territory", "name": "Australia", "canonical_key": null, "aliases": [], "mentions": [{"text": "Australia"}], "confidence": 0.95}
  ],
  "relations": [
    {"type": "WROTE", "source_entity_id": "entity_1", "target_entity_id": "entity_0", "evidence": "Rushing Back by Flume", "confidence": 0.95},
    {"type": "PERFORMED_BY", "source_entity_id": "entity_0", "target_entity_id": "entity_2", "evidence": "ft. Vera Blue", "confidence": 0.90},
    {"type": "PUBLISHED_BY", "source_entity_id": "entity_0", "target_entity_id": "entity_3", "evidence": "published by Future Classic", "confidence": 0.90},
    {"type": "VALID_IN_TERRITORY", "source_entity_id": "entity_0", "target_entity_id": "entity_4", "evidence": "in Australia", "confidence": 0.85}
  ]
}

Note: 5 entities and 4 relations — every entity is connected. This is the expected density."""

ENTITY_EXTRACTION_USER_PROMPT_TEMPLATE = """\
Extract ALL entities and relations from this chunk.
Return JSON that matches the schema exactly.

IMPORTANT: For every pair of entities that have a relationship in the text, you MUST include a relation.
Use entity_id values (e.g. "entity_0") or entity names as relation endpoints.

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
      "mentions": [{"text": "string"}],
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
