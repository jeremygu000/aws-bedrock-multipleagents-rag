# Qwen-Plus Entity Extraction Spec

## 1. Active Model Configuration

Use these environment variables as the default extraction setup:

```bash
export LLM_PROVIDER=qwen
export LLM_MODEL=qwen-plus
export RERANK_MODEL="gte-rerank"
```

Runtime recommendations for extraction:

- `temperature=0`
- `top_p=1`
- deterministic JSON output only
- retry up to 2 times on schema validation failure

## 2. Extraction Strategy

Use a two-stage pipeline:

1. Rule-first extraction (high precision)

- extract IDs and deterministic fields by regex/rules:
  - `ISWC`, `ISRC`, `WINF`, dates, legal identifiers

2. LLM structured extraction (Qwen-Plus)

- extract entities, aliases, relations, and evidence spans
- enforce JSON schema validation before writing to Neo4j

If schema validation fails:

- retry with a strict "repair JSON" prompt once
- if still invalid, mark row as `extraction_failed` and skip graph upsert

## 3. Entity and Relation Types

Initial entity types:

- `Work`
- `Person`
- `Organization`
- `Identifier`
- `Territory`
- `LicenseTerm`
- `Date`

Initial relation types:

- `WROTE`
- `PERFORMED_BY`
- `PUBLISHED_BY`
- `HAS_IDENTIFIER`
- `VALID_IN_TERRITORY`
- `HAS_TERM`
- `REFERENCES`

## 4. Prompt Template (System)

Use this system prompt template for Qwen-Plus:

```text
You are an information extraction engine.
Return valid JSON only. No markdown, no extra text.
Extract entities and relations from the input text.

Rules:
1) Use only the provided schema keys.
2) Keep original surface forms in "mentions".
3) Set confidence in [0,1].
4) If uncertain, output lower confidence and keep relation optional.
5) Do not fabricate IDs. If not present, leave null.
```

User prompt template:

```text
Extract entities and relations from this chunk.
Return JSON that matches the schema exactly.

chunk_id: {{chunk_id}}
doc_id: {{doc_id}}
text:
{{chunk_text}}
```

## 5. Output JSON Schema (Logical)

```json
{
  "chunk_id": "string",
  "entities": [
    {
      "entity_id": "string",
      "type": "Work|Person|Organization|Identifier|Territory|LicenseTerm|Date",
      "name": "string",
      "canonical_key": "string|null",
      "aliases": ["string"],
      "mentions": [
        {
          "text": "string",
          "start": 0,
          "end": 10
        }
      ],
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
}
```

Validation requirements:

- all `confidence` values in `[0,1]`
- relation endpoints must exist in `entities.entity_id`
- duplicate entities must be merged by `canonical_key` when available

## 6. Neo4j Upsert Mapping

Write graph with provenance:

- `(:Document {doc_id})`
- `(:ChunkRef {chunk_id})`
- `(:Entity {entity_id, type, canonical_key, name})`

Relationships:

- `(Document)-[:HAS_CHUNK]->(ChunkRef)`
- `(ChunkRef)-[:MENTIONS {confidence, evidence}]->(Entity)`
- `(Entity)-[:RELATION {type, confidence, source_chunk_id}]->(Entity)`

Also write SQL link rows to `kb_chunk_graph_links` for round-trip retrieval.

## 7. Quality Gates for Extraction

Minimum offline gates before production:

- entity precision `>= 0.90` on labeled set
- entity recall `>= 0.80` on labeled set
- relation precision `>= 0.80` on labeled set
- schema-valid response rate `>= 99%`

Operational gates:

- extraction failure rate `< 1%`
- p95 extraction latency tracked per 1k chunks
- no critical parser/validation exceptions in release runs

## 8. Logging and Trace Fields

Store these fields per extraction job:

- `model_provider` (`qwen`)
- `model_name` (`qwen-plus`)
- `model_version` (if available)
- `prompt_version`
- `schema_version`
- `chunk_id`
- `doc_id`
- `run_id`
- `validation_status`
- `failure_reason` (nullable)
