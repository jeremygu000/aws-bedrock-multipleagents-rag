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
