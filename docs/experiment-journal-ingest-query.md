# Experiment Journal — Ingest & Query Pipeline

> **Started**: 2026-03-28
> **Last Updated**: 2026-03-31
> **Status**: In progress — relation extraction quality fix deployed, diagnostic script pending live DB test

---

## Table of Contents

1. [Background & Objectives](#1-background--objectives)
2. [Day 1 (Mar 28) — Foundation Build](#2-day-1-mar-28--foundation-build)
3. [Day 2 (Mar 29) — Crawl Ingest & Production Hardening](#3-day-2-mar-29--crawl-ingest--production-hardening)
4. [Day 3 (Mar 30) — Entity Extraction & Bug Fixes](#4-day-3-mar-30--entity-extraction--bug-fixes)
5. [Day 3 (Mar 30, evening) — pgvector Entity ID Collision Fix](#5-day-3-mar-30-evening--pgvector-entity-id-collision-fix)
6. [Day 4 (Mar 31) — Relation Extraction Quality Fix](#6-day-4-mar-31--relation-extraction-quality-fix)
7. [Bug Tracker](#7-bug-tracker)
8. [Current Data State](#8-current-data-state)
9. [Next Steps](#9-next-steps)
10. [Appendix — Commands Reference](#10-appendix--commands-reference)

---

## 1. Background & Objectives

This journal documents the end-to-end experiment of building and running the RAG ingestion and query pipelines for the APRA (Australian Prudential Regulation Authority) knowledge base.

### System Architecture

- **RAG Service**: Python 3.12 FastAPI + LangGraph (14-node pipeline)
- **Vector Store**: Amazon RDS PostgreSQL + pgvector
- **Sparse Index**: Elasticsearch (BM25)
- **Knowledge Graph**: Neo4j (entities + relations)
- **Embedding Model**: Qwen (via HTTP API, 1024-dim vectors)
- **LLM**: AWS Bedrock (Claude) + Qwen-Plus (entity extraction)
- **Crawler**: Crawl4AI
- **Infra**: AWS CDK (3 stacks: BedrockAgentsStack, Neo4jDataStack, MonitoringEc2Stack)

### Goals

1. Crawl all 1,378 APRA URLs and ingest into PostgreSQL + Elasticsearch
2. Extract entities/relations into Neo4j knowledge graph
3. Validate the hybrid retrieval pipeline (vector + BM25 + graph)
4. Measure quality with RAGAS/DeepEval evaluation framework

---

## 2. Day 1 (Mar 28) — Foundation Build

All foundational features implemented in a single intensive session. Commits span `bc0efdf` to `0b083f8`.

### 2.1 Document Ingestion Pipeline (`bc0efdf` → `cbeb043`)

- SQS-triggered Lambda handler with partial batch failure support
- CDK infrastructure for ingestion pipeline (S3 → SQS → Lambda)
- Document parsing layer (pymupdf, python-docx, bs4, lxml)
- Env vars and API documentation

### 2.2 Elasticsearch Integration (`e022afb` → `80e0f7c`)

- Made OpenSearch/Elasticsearch optional
- Conditional auth: SigV4 (AWS) vs no-auth (local Docker)
- Added `kb_chunks` index schema
- Wired EC2 Elasticsearch endpoint

### 2.3 Database & Logging (`7c180fe` → `b5ef33f`)

- Updated PostgreSQL connection string format (psycopg3 compatibility)
- Added logging to Lambda tool
- Moved logger initialization to proper location

### 2.4 Answer Generation Enhancement (`a9ef8b0`)

- Intent-aware prompts (4 types: factual, analytical, procedural, comparative)
- Dynamic model routing based on query complexity
- Relevance scoring for retrieved chunks

### 2.5 Entity Extraction (`0895b58` → `b445223`)

- Configurable entity extraction with Qwen-Plus LLM
- 7 domain-specific entity types
- Source chunk tracking for provenance
- Entity summary with max token setting

### 2.6 Knowledge Graph — Construction & Retrieval (`810696d` → `a49edfe`)

- Entity vector store (pgvector) for semantic entity search
- Graph-enhanced retrieval with configurable modes (7 modes total)
- Weighted RRF fusion (graph 0.6 + traditional 0.4)
- Graph chunk fusion with weighted scoring
- Integrated graph context into answer generation

### 2.7 Caching (`b156bc8` → `0b083f8`)

- L2 semantic query cache (pgvector cosine similarity ≥ 0.95)
- Cache invalidation on document changes

---

## 3. Day 2 (Mar 29) — Crawl Ingest & Production Hardening

Commits span `c4b3a22` to `20cf7e7`. Focus: closing gaps against LightRAG, running the actual crawl, and hardening the pipeline.

### 3.1 Gap Analysis & Feature Parity (`c4b3a22` → `e9da4c2`)

Completed features to match/exceed LightRAG:

| Phase | Feature                                                        | Commit    |
| ----- | -------------------------------------------------------------- | --------- |
| 4     | Evaluation Framework (RAGAS + DeepEval + custom graph metrics) | `f44783f` |
| 5     | Incremental Graph Update (cascade delete + re-ingest)          | `c4b3a22` |
| 6     | Streaming SSE (`GET /retrieve/stream`)                         | `9785f5b` |
| 7     | Gleaning (multi-round entity extraction, 0–5 rounds)           | `7f81119` |
| 8     | Observability (Phoenix tracing + Prometheus metrics)           | `e9da4c2` |

Updated `docs/rag-gap-analysis.md` — comprehensive 923-line comparison document.

### 3.2 psycopg3 Cast Syntax Migration

**Problem**: PostgreSQL `::type` cast syntax (e.g., `(:value)::vector`) is not supported by psycopg3's parameter binding. All casts needed conversion to `CAST(x AS type)`.

**Files fixed**:

| File                                         | Pattern                      | Fix                                |
| -------------------------------------------- | ---------------------------- | ---------------------------------- |
| `app/query_cache.py:90`                      | `ARRAY[:doc_id]::TEXT[]`     | `CAST(ARRAY[:doc_id] AS TEXT[])`   |
| `app/entity_vector_store.py:102,105,115,118` | `(:query_embedding)::vector` | `CAST(:query_embedding AS vector)` |
| `app/repository.py:348,450`                  | `(:query_vector)::vector`    | `CAST(:query_vector AS vector)`    |

**Verification**: `grep '::[a-z]' apps/rag-service/app/*.py` → **zero matches**. All `::type` casts eliminated.

### 3.3 QueryCache Constructor Bug

**Problem**: `app/main.py:91` called `QueryCache(settings)` but `QueryCache.__init__` requires `(settings, engine)`.

**Fix**: Changed to `QueryCache(settings, repository._get_engine())`.

### 3.4 Small Batch Crawl Test (5 pages) ✅

```bash
cd apps/rag-service
uv run --extra crawl python -m scripts.crawl_ingest \
  --urls-file scripts/apra_urls.txt \
  --category pages --limit 5 --skip-entities
```

**Result**: 5/5 URLs succeeded, 50 chunks created, 0 failures. All 4 bug fixes verified working together:

- `source_type='crawler'` ✓
- Embed batch ≤ 10 ✓
- `CAST()` syntax ✓
- `datetime.now(UTC)` ✓

### 3.5 Full Crawl — 1,378 URLs ✅

#### First Attempt (inline)

Ran inline but hit 10-minute tool timeout at 203/1,378 (202 success, 1 failure).

- Rate: ~2 URLs/sec
- 1 failure: URL #16 — `IncompleteRead` from Qwen embedding API

#### Resume Bug Fix

**Problem**: `crawl_ingest.py` `get_existing_urls()` queried `WHERE source_type = 'web_crawl'` but actual stored records use `source_type = 'crawler'`.

**Fix**: Changed to `'crawler'`. Also fixed log message: `"fetching existing web_crawl source_uris"` → `"fetching existing crawler source_uris"`.

#### Second Attempt (background, with --resume)

```bash
cd apps/rag-service
nohup uv run --extra crawl python -m scripts.crawl_ingest \
  --urls-file scripts/apra_urls.txt --resume --skip-entities \
  > /tmp/crawl-apra.log 2>&1 &
```

- Correctly identified 203 already-ingested URLs, processing remaining 1,175
- PID: 19736

#### Failed URL Retry

4 URLs failed during the main crawl. Retried separately:

```bash
uv run --extra crawl python -m scripts.crawl_ingest \
  --urls-file /tmp/failed-urls.txt --skip-entities
```

All 4 succeeded (39 chunks).

#### Final Tally

| Metric       | Value                   |
| ------------ | ----------------------- |
| Total URLs   | 1,378                   |
| Succeeded    | 1,378                   |
| Failed       | 0                       |
| Total chunks | ~2,722                  |
| Crawl rate   | ~2 URLs/sec             |
| Total time   | ~40 min (across 2 runs) |

### 3.6 Stale `ingestion_runs` Cleanup

**Problem**: Found 5 records stuck in `running` status in `ingestion_runs` table — all from the first 5-page test batch where the process was killed before `complete_ingestion_run()` could execute.

**Root cause**: `crawl_ingest.py` creates one `ingestion_run` per URL. When the process is killed/interrupted, `complete_ingestion_run` never fires, leaving runs permanently in `running` state.

**Immediate fix**: Updated all 5 to `succeeded` with explanatory notes via direct SQL.

**Permanent fix**: See §3.8 (resilience improvements).

### 3.7 Test Environment Isolation Fixes

**Problem**: 3 tests failed when `.envrc` sets `RAG_ENABLE_NEO4J=true`, `RAG_ENABLE_ENTITY_EXTRACTION=true`, `RAG_NEO4J_PASSWORD`.

**Fixes** (added `monkeypatch.delenv()` calls):

- `tests/test_entity_extraction.py::TestConfigFlag::test_entity_extraction_disabled_by_default` — clears `RAG_ENABLE_ENTITY_EXTRACTION`
- `tests/test_graph_repository.py::TestNeo4jConfig::test_default_neo4j_settings` — clears all `RAG_NEO4J_*` vars
- `tests/test_graph_repository.py::TestResolveNeo4jPassword::test_returns_empty_when_no_password_or_arn` — clears password vars

**Result**: `pnpm check` fully green — typecheck ✅, lint ✅, rag:lint ✅, rag:test 508 passed ✅.

### 3.8 Resilience Improvements (`crawl_ingest.py`)

Modeled after C# Polly resilience patterns, using the `redress` library (v1.3.0).

#### Stale Ingestion Run Fix

Wrapped everything after `create_ingestion_run()` in `try/except`. On failure, the run is marked `failed` before re-raising — no more orphaned `running` records.

#### Graceful Shutdown (SIGINT/SIGTERM)

- `_shutdown_requested` flag + `_handle_shutdown_signal()` handler
- First signal → finish current URL, stop loop
- Second signal → `sys.exit(1)` (hard kill)
- Crawl loop, retry phase, and redress policies all check `abort_if=lambda: _shutdown_requested`

#### Resilience Policies

| Policy                   | Strategy               | Config                                                                                           |
| ------------------------ | ---------------------- | ------------------------------------------------------------------------------------------------ |
| `_build_embed_policy()`  | Retry + CircuitBreaker | `decorrelated_jitter` max 10s, 120s deadline, circuit trips after 10 failures/120s, 30s recovery |
| `_build_ingest_policy()` | Retry                  | `equal_jitter` max 15s, 300s deadline                                                            |

**Default classifier**: `IncompleteRead`/timeouts → `TRANSIENT` (retries), HTTP 429 → `RATE_LIMIT` (aggressive backoff).

#### Failed URL Auto-Retry Phase

After main crawl loop, failed URLs are automatically re-crawled with:

- Longer timeout (60s vs 30s)
- Lower concurrency (half of normal)
- Same policy wrapping

#### CLI Option

`--max-retries N` (default 3): Controls both embedding and per-URL retry attempts.

**Dependency**: Added `redress>=1.3.0,<2` to `[crawl]` optional deps in `pyproject.toml`.

### 3.9 Standalone Entity Extraction Script

Created `scripts/extract_entities.py` — backfill script for entity extraction on already-ingested documents.

**Key features**:

- Streams documents and chunks from DB (generators for memory efficiency)
- Skips docs that already have entities (unless `--force`)
- Supports `--limit`, `--doc-id`, `--category`, `--concurrency`, `--force`, `--dry-run`
- Sequential (default) or parallel via `ThreadPoolExecutor`
- Requires `RAG_ENABLE_ENTITY_EXTRACTION=true`

```bash
# Extract entities for first 5 docs
cd apps/rag-service
uv run python -m scripts.extract_entities --limit 5

# Full backfill with parallel processing
uv run python -m scripts.extract_entities --concurrency 4
```

### 3.10 Pytest Warning Suppression

Added `filterwarnings` to `pyproject.toml` under `[tool.pytest.ini_options]`:

- Suppressed SWIG DeprecationWarnings (from FAISS)
- Suppressed `RequestsDependencyWarning` (requests/urllib3 version mismatch)

### 3.11 Data Store Verification & `.envrc` Fixes

After the full crawl completed, verified all three data stores.

#### PostgreSQL ✅

Verified via `PostgresRepository` (class in `app.repository`, not `Repository`) + `app.config.Settings`:

| Table            | Records | Notes                                 |
| ---------------- | ------- | ------------------------------------- |
| `kb_documents`   | 1,300   | 1,262 pages + 34 awards + 4 news      |
| `kb_chunks`      | 3,103   | With 1024-dim embeddings              |
| `ingestion_runs` | 1,329   | All `succeeded`                       |
| `kb_entities`    | 0       | Expected (crawl used --skip-entities) |

#### Elasticsearch ❌ → ✅ (after backfill)

**Problem**: `kb_chunks` index existed but had 0 documents.

**Root cause (two issues)**:

1. Crawl launched via `nohup` without `.envrc` sourced → `RAG_OPENSEARCH_ENDPOINT` was empty → `bulk_index_opensearch()` silently returned (line 317: `if not self._settings.opensearch_endpoint.strip(): return`)
2. Even with `.envrc` sourced, the endpoint was `http://localhost:9200` — but ES runs on the Monitoring EC2 instance, not localhost

**Fix**: Updated `.envrc`:

```
# Before (wrong):
RAG_OPENSEARCH_ENDPOINT="http://localhost:9200"
# After (correct):
RAG_OPENSEARCH_ENDPOINT="http://<ES_EC2_HOST>:9200"
```

ES confirmed running: Elasticsearch 8.17.0, Docker on MonitoringEc2Stack, `xpack.security.enabled=false` (no auth needed).

**Backfill**: Created `scripts/backfill_opensearch.py` and ran it — 3,103 chunks indexed in 4 seconds (7 bulk requests of 500). See §3.12.

#### Neo4j ❌ → ✅ (after IP fix)

**Problem**: Connection timed out to `bolt://<OLD_NEO4J_EC2_HOST>:7687`.

**Root cause**: EC2 instance had a new public IP after restart. Security group rules were fine (ports 7474 + 7687 open to `0.0.0.0/0`).

**Fix**: Updated `.envrc`:

```
# Before (stale IP):
RAG_NEO4J_URI="bolt://<OLD_NEO4J_EC2_HOST>:7687"
# After (current IP):
RAG_NEO4J_URI="bolt://<NEO4J_EC2_HOST>:7687"
```

**Verified**: Connected successfully. Neo4j 5.26.21, Protocol 5.8, 0 nodes (expected — entities not yet extracted).

### 3.12 OpenSearch Backfill Script & Execution

Created `scripts/backfill_opensearch.py` to re-index existing PG chunks into OpenSearch.

**Features**:

- Reads all chunks from `kb_chunks` JOIN `kb_documents` via raw SQL
- Builds the same `_source` document shape as `IngestionRepository.bulk_index_opensearch()`
- Batched `helpers.bulk()` calls (default 500 per batch)
- Supports `--batch-size`, `--dry-run`, `--category`, `--recreate`

**Execution**:

```bash
cd apps/rag-service
uv run python -m scripts.backfill_opensearch
```

| Metric         | Value           |
| -------------- | --------------- |
| Chunks read    | 3,103 (from PG) |
| Chunks indexed | 3,103 (in ES)   |
| Batch size     | 500             |
| Total time     | 4.0 seconds     |
| Errors         | 0               |

---

## 4. Day 3 (Mar 30) — Entity Extraction & Bug Fixes

Focus: Running entity extraction across all 1,300 documents and fixing bugs discovered during the process.

### 4.1 Bug Fix — `aliases` NULL Violation in Entity Upsert

**Problem**: `ON CONFLICT DO UPDATE` clause in `_UPSERT_ENTITY_SQL` and `_UPSERT_RELATION_SQL` used `array_agg(DISTINCT val) FROM unnest(...)`. When both existing and new arrays are empty (`[] || []`), `unnest` returns 0 rows, `array_agg` returns `NULL` → `NOT NULL` constraint violation.

**Error**: `psycopg.errors.NotNullViolation: null value in column "aliases" of relation "kb_entities"`

**Fix**: Wrapped all 3 occurrences with `COALESCE(..., ARRAY[]::text[])`:

| File                         | Column             | Location   |
| ---------------------------- | ------------------ | ---------- |
| `app/entity_vector_store.py` | `aliases`          | Line 53-56 |
| `app/entity_vector_store.py` | `source_chunk_ids` | Line 59-62 |
| `app/entity_vector_store.py` | `source_chunk_ids` | Line 89-92 |

**Verification**: 511 tests pass.

### 4.2 Bug Fix — JSON Parse Robustness (4-Layer Pipeline)

**Problem**: LLM entity extraction output frequently contained malformed JSON (trailing commas, unterminated strings, prose wrapping). 3/5 chunks failed for test doc #2.

**Fix**: Rewrote `app/entity_extraction.py` parsing from 2-layer to 4-layer:

| Layer | Method                  | Speed   | Handles                                              |
| ----- | ----------------------- | ------- | ---------------------------------------------------- |
| 1     | `_extract_json_text`    | instant | Markdown fences, prose wrapping, outermost `{...}`   |
| 2     | `json-repair` library   | instant | Trailing commas, unquoted keys, unterminated strings |
| 3     | `_lenient_parse`        | instant | Per-entity/relation parsing, skip bad items          |
| 4     | `_llm_repair_and_parse` | ~10s    | Semantic issues (last resort)                        |

**Dependency**: Added `json-repair>=0.58.7` to `pyproject.toml`.

**Verification**: 511 tests pass, 7 test cases (4 updated + 3 new).

### 4.3 Bug Fix — Qwen Embedding Batch Size Limit

**Problem**: `entity_extraction_embed_batch_size` defaulted to 20 in `app/config.py`, but Qwen embedding API max batch size is 10.

**Error**: `ValueError: Qwen embeddings API HTTP error: 400 ... batch size is invalid, it should not be larger than 10`

**Fix**: Changed default from `20` → `10` in `app/config.py` line 179.

**Impact**: Caused 262 doc failures in Run 1 (all docs with >10 entities). Fixed before Retry run.

### 4.4 Entity Extraction — Test Run (2 docs)

```bash
export QWEN_API_KEY=sk-... && export LLM_MODEL=qwen-plus
source .envrc
uv run python scripts/extract_entities.py --limit 2
```

| Metric    | Value                                             |
| --------- | ------------------------------------------------- |
| Docs      | 2 succeeded                                       |
| Entities  | 11                                                |
| Relations | 2                                                 |
| Time      | 153.7s (76.8s/doc avg)                            |
| Issues    | 3/5 chunks had JSON parse failures (non-blocking) |

Neo4j writes confirmed: 10 nodes present after test.

### 4.5 Entity Extraction — Run 1 (1,298 docs)

```bash
screen -S entity-extract
export QWEN_API_KEY=sk-... && export LLM_MODEL=qwen-plus
source .envrc
uv run python scripts/extract_entities.py --concurrency 4
```

Skipped 2 docs from test run, processed remaining 1,298.

| Metric    | Value                             |
| --------- | --------------------------------- |
| Succeeded | 1,036 docs                        |
| Failed    | 262 docs (all batch size >10 bug) |
| Entities  | 4,908                             |
| Relations | 304                               |
| Time      | 11,387s (~3.16 hours)             |
| Avg speed | 8.8s/doc                          |

### 4.6 Entity Extraction — Retry Run (308 docs)

After fixing the batch size bug (§4.3), re-ran to process failed + skipped docs:

```bash
screen -S entity-retry
export QWEN_API_KEY=sk-... && export LLM_MODEL=qwen-plus
source .envrc
uv run python scripts/extract_entities.py --concurrency 4 \
  > /tmp/entity-extract-retry.log 2>&1
```

Script found 1,300 docs, skipped 992 with existing entities → 308 docs to process (larger multi-chunk docs, up to 42 chunks each).

| Metric    | Value                              |
| --------- | ---------------------------------- |
| Succeeded | 306 docs                           |
| Failed    | 2 docs (transient network errors)  |
| Entities  | 6,100                              |
| Relations | 553                                |
| Time      | 6,668s (~1.85 hours)               |
| Avg speed | 21.7s/doc (larger docs than Run 1) |

**Failed docs** (both transient, retryable):

| Doc ID     | Error                                              |
| ---------- | -------------------------------------------------- |
| `ac678891` | `TimeoutError` on Qwen embedding API               |
| `cc54c4f3` | `IncompleteRead` (connection dropped mid-response) |

### 4.7 Combined Entity Extraction Results

| Metric            | Run 1 | Retry | Combined                  |
| ----------------- | ----- | ----- | ------------------------- |
| Docs succeeded    | 1,036 | 306   | **1,298 / 1,300** (99.8%) |
| Docs failed       | 262   | 2     | **2** (transient network) |
| Entities (Neo4j)  | 4,908 | 6,100 | **11,008 upserts**        |
| Relations (Neo4j) | 304   | 553   | **857 upserts**           |
| Total time        | 3.16h | 1.85h | **~5 hours**              |

**Note**: Neo4j numbers are raw upsert counts. Actual unique entities in Neo4j = 5,170 (deduplicated on `name+type`). pgvector originally had only 84 entities due to Bug #12 (entity_id collision) — fixed via backfill script.

---

## 5. Day 3 (Mar 30, evening) — pgvector Entity ID Collision Fix

### 5.1 Investigation: pgvector shows only 84 entities

After entity extraction reported 11,008 entity upserts across 1,298 docs, verification revealed pgvector's `kb_entities` table contained only **84 rows** — far below the expected thousands.

**Verification script** (`scripts/verify_entity_data.py`) queried both pgvector and Neo4j:

- **pgvector**: 84 entities, 129 relations
- **Neo4j**: 5,170 entities, 784 relations (correct, deduplicated on `(name, type)`)

### 5.2 Root Cause: `entity_id` collision (Bug #12)

`ingestion.py` line 156 used `entity.entity_id` — the LLM's per-chunk local ID (e.g., `e1`, `e2`, `e3`). Since `kb_entities` has `entity_id TEXT PRIMARY KEY` with `ON CONFLICT (entity_id) DO UPDATE`, every document producing entity `e1` silently overwrote the previous document's `e1`.

**Evidence**: Entity `e1` = "Roy Morgan" had accumulated 2,740 `source_chunk_ids` from cross-document overwrites. Only 84 distinct entity_id patterns survived (e.g., `e1`–`e6`, `E1`–`E6`, `person_0`–`person_4`, etc.).

Similarly, `relation_id` at lines 171-172 was composed from the LLM's local entity_ids, causing the same collision problem for relations.

**Neo4j was unaffected** because it uses `MERGE (e:Entity {name: $name, type: $type})` — deduplication on the natural key, not the synthetic `entity_id`.

### 5.3 Fix: Deterministic entity_id generation

**In `ingestion.py`** (lines 147-190):

- **entity_id**: Changed from `entity.entity_id` to `hashlib.sha256(f"{dedup_key}::{entity.type.value}".encode()).hexdigest()[:16]` where `dedup_key = entity.canonical_key or entity.name.lower()` (matches `_entity_dedup_key()` logic).
- **relation_id**: Built `old_to_new_id` mapping from LLM entity_ids to new stable IDs. Relations now reference the deterministic entity_ids.

### 5.4 Backfill: Neo4j → pgvector

Rather than re-running the 3+ hour LLM extraction, created `scripts/backfill_pgvector_from_neo4j.py` to:

1. Read all 5,170 entities + 784 relations from Neo4j via Cypher
2. Generate deterministic `entity_id = SHA-256(name.lower()::type)[:16]` (5,170 entities → 5,104 unique IDs; 66 genuine name+type duplicates in Neo4j merged via upsert)
3. Call Qwen embedding API in batches of 10 with retry logic (3 attempts, exponential backoff)
4. TRUNCATE `kb_relations` then `kb_entities` (FK order)
5. Batch upsert to pgvector via `EntityVectorStore`

**Expected result**: ~5,104 entities and ~784 relations in pgvector, matching Neo4j's deduplicated data.

**Status**: Backfill script running (~30 min for embedding generation).

---

## 6. Day 4 (Mar 31) — Relation Extraction Quality Fix

Focus: Investigating and fixing the 87.7% orphaned entity rate — 4,533 out of 5,170 entities had zero relations.

### 6.1 Investigation: Why are 87.7% of entities orphaned?

**Hypothesis**: The pgvector backfill (§5.4) broke relation references.

**Finding**: Backfill was correct. `backfill_pgvector_from_neo4j.py` properly re-IDs relations via `name+type` lookup → SHA-256. All 784 relations in pgvector resolve to valid entity_ids (784/784 endpoints found).

**Real root cause**: The 87.7% orphan rate is an **extraction-time problem**, not a data migration bug. The LLM (Qwen-Plus) extracts entities but fails to extract relations for most chunks.

### 6.2 Root Cause Analysis (5 factors)

| Factor                       | Description                                                                                                                | Severity |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------- |
| **Minimal prompt**           | `ENTITY_EXTRACTION_SYSTEM_PROMPT` gave zero examples of relations — just listed type names (`WROTE, PERFORMED_BY, ...`)    | Critical |
| **Silent relation dropping** | `_lenient_parse()` silently filtered relations where `source_entity_id` or `target_entity_id` didn't match a parsed entity | High     |
| **Entity ID mismatch**       | LLM generates entity IDs like `"entity_1"` and must reference them in relations; any ID typo → relation dropped            | High     |
| **No gleaning**              | Default `gleaning_rounds=0`; even when enabled, doesn't force re-extraction of missed relations                            | Medium   |
| **No observability**         | Relation drops were logged at DEBUG level only — invisible in production logs                                              | Medium   |

### 6.3 Fix Part 1 — Few-shot Extraction Prompt

**File**: `app/prompts.py`

Rewrote `ENTITY_EXTRACTION_SYSTEM_PROMPT`:

- Specialized for music publishing domain
- Sequential `entity_0/1/2` ID format (easier for LLM to reference correctly)
- Allows entity names as relation endpoints (not just entity IDs)
- Demands a relation for every entity pair
- Complete worked example: Rushing Back/Flume/Vera Blue/Future Classic/Australia → 5 entities, 4 relations

Reinforced `ENTITY_EXTRACTION_USER_PROMPT_TEMPLATE` with "extract ALL" + "MUST include a relation".

### 6.4 Fix Part 2 — Name-based Relation Resolution

**Problem**: LLM often uses entity names (e.g., `"Flume"`) instead of generated IDs (`"entity_0"`) in relation `source_entity_id`/`target_entity_id` fields. Previously these were silently dropped.

**Files changed**:

| File                              | Change                                                                                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app/entity_extraction_models.py` | `validate_relation_endpoints` now accepts entity names (case-insensitive) as valid endpoints                                                                  |
| `app/entity_extraction.py`        | New `_resolve_relation_endpoints()` static method: builds `name→entity_id` + `alias→entity_id` lookup, resolves name-based references, logs resolution counts |

Resolution wired into all 4 parse paths:

1. Direct JSON parse
2. `json-repair` fallback
3. LLM repair (JSON)
4. LLM repair (validation)

`_lenient_parse()` also has inline name resolution with `entity_name_to_id` dict.

### 6.5 Observability Logging

Added INFO-level logging throughout the extraction pipeline:

| Location            | What's logged                                                                                 |
| ------------------- | --------------------------------------------------------------------------------------------- |
| `_lenient_parse()`  | Entities accepted/dropped, relations accepted/dangling/invalid, name-resolved count           |
| `_llm_extraction()` | Parse path used (direct/json_repair/lenient/llm_repair), raw vs parsed entity/relation counts |
| `extract()`         | Final entity/relation totals per chunk                                                        |

### 6.6 CDK: Neo4j Instance Type Upgrade

**File**: `packages/infra-cdk/lib/neo4j-data-stack.ts` line 19

Changed default instance type: `t3.micro` → `t3.small` (doubled memory for graph queries).

### 6.7 Debug Scripts Created

| Script                                 | Purpose                                                    |
| -------------------------------------- | ---------------------------------------------------------- |
| `scripts/debug_relation_extraction.py` | Runs extraction on sample chunks to measure relation yield |
| `scripts/debug_neo4j_neighbors.py`     | Queries Neo4j for entity neighbor data                     |

### 6.8 Verification

- ruff: ✅ all checks passed
- LSP: ✅ clean (only pre-existing `json_repair` import warning)
- pytest: ✅ 520 passed, 0 failed
- typecheck + oxlint: ✅ clean
- Pre-push hook: ✅ passed

### 6.9 Status

Code changes complete and pushed. Next step: run `debug_relation_extraction.py --chunks 5` against live DB to measure improved relation yield before deciding on a full re-extraction run.

---

## 7. Bug Tracker

> **Note**: Bug #13 added in Day 4.

| #   | Bug                                    | Root Cause                                                                                 | Fix                                              | Severity |
| --- | -------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------ | -------- |
| 1   | `::type` cast breaks psycopg3          | psycopg3 doesn't support `::` in parameterized queries                                     | `CAST(x AS type)` across 3 files                 | High     |
| 2   | `QueryCache(settings)` missing engine  | Constructor signature mismatch                                                             | Added `engine` parameter                         | High     |
| 3   | Resume skips nothing                   | `WHERE source_type = 'web_crawl'` vs actual `'crawler'`                                    | Fixed to `'crawler'`                             | High     |
| 4   | Stale `ingestion_runs`                 | No cleanup on process kill                                                                 | `try/except` + mark failed                       | Medium   |
| 5   | Tests fail with `.envrc`               | Env var leakage into test environment                                                      | `monkeypatch.delenv()`                           | Low      |
| 6   | Qwen `IncompleteRead`                  | Embedding API timeout/disconnect                                                           | Retry + CircuitBreaker policies                  | Medium   |
| 7   | ES endpoint → localhost                | `.envrc` pointed to `localhost:9200`, ES is on remote EC2                                  | Updated to EC2 public DNS                        | High     |
| 8   | Neo4j stale IP                         | EC2 public IP changed after restart                                                        | Updated `.envrc` with new IP                     | Medium   |
| 9   | `aliases` NULL violation               | `array_agg` returns NULL on empty `unnest`                                                 | `COALESCE(..., ARRAY[]::text[])`                 | High     |
| 10  | JSON parse failures                    | LLM output malformed (trailing commas, prose wrapping)                                     | 4-layer parse pipeline + `json-repair`           | Medium   |
| 11  | Qwen embed batch size >10              | Default `entity_extraction_embed_batch_size=20`, API max=10                                | Changed default to 10                            | High     |
| 12  | pgvector entity_id collision           | `ingestion.py` used LLM's local `entity_id` (e.g., `e1`) as PK — cross-doc overwrites      | Deterministic SHA-256 hash of `(name, type)`     | Critical |
| 13  | 87.7% orphaned entities (no relations) | Extraction prompt had no relation examples; name-based relation endpoints silently dropped | Few-shot prompt + name-based relation resolution | Critical |

---

## 8. Current Data State

### PostgreSQL

| Table            | Records | Notes                                          |
| ---------------- | ------- | ---------------------------------------------- |
| `kb_documents`   | 1,300   | 1,262 pages + 34 awards + 4 news               |
| `kb_chunks`      | 3,103   | With embeddings (1024-dim pgvector)            |
| `ingestion_runs` | 1,329   | All `succeeded`                                |
| `kb_entities`    | ~5,104  | Backfilled from Neo4j (was 84 due to Bug #12)  |
| `kb_relations`   | ~784    | Backfilled from Neo4j (was 129 due to Bug #12) |
| `query_cache`    | 0       | No queries run yet                             |

### Elasticsearch

| Index       | Documents | Notes                             |
| ----------- | --------- | --------------------------------- |
| `kb_chunks` | 3,103     | BM25 full-text search, backfilled |

Endpoint: `http://<ES_EC2_HOST>:9200`
Version: Elasticsearch 8.17.0 (Docker on MonitoringEc2Stack), no auth.

### Neo4j

| Type           | Count | Notes                                                                                                      |
| -------------- | ----- | ---------------------------------------------------------------------------------------------------------- |
| Entity nodes   | 5,170 | 7 types: Work(1620), Org(1288), Person(1238), Date(358), Territory(339), LicenseTerm(185), Identifier(142) |
| Relation edges | 784   | All type `RELATES_TO`                                                                                      |

Endpoint: `bolt://<NEO4J_EC2_HOST>:7687`
Version: Neo4j 5.26.21, Protocol 5.8. Verified via direct Cypher queries.

---

## 9. Next Steps

### Immediate

- [x] **Verify PostgreSQL**: 1,300 docs, 3,103 chunks, all ingestion runs succeeded
- [x] **Verify Elasticsearch**: 3,103 docs indexed via backfill script
- [x] **Fix `.envrc`**: Updated ES endpoint (localhost → remote EC2) and Neo4j URI (stale IP)
- [x] **Run entity extraction**: 1,298/1,300 docs completed (2 transient failures)
- [x] **Investigate pgvector dedup**: Found Bug #12 — entity_id collision (84 vs 5,170 entities)
- [x] **Verify Neo4j node counts**: 5,170 entities, 784 relations (healthy)
- [x] **Fix ingestion.py**: Deterministic entity_id/relation_id generation
- [x] **Investigate relation backfill gap**: Bug #13 — 87.7% orphaned entities caused by extraction-time prompt + parsing issues
- [x] **Fix relation extraction**: Few-shot prompt + name-based relation resolution
- [ ] **Run diagnostic script**: `debug_relation_extraction.py --chunks 5` to measure improved relation yield
- [ ] **Full re-extraction**: If diagnostic confirms improvement, re-run entity extraction with `--force` to regenerate all relations
- [ ] **Verify backfill completion**: Confirm ~5,104 entities + ~784 relations in pgvector
- [ ] **Retry last 2 docs**: Re-run entity extraction for `ac678891` and `cc54c4f3`

### Query Pipeline Testing

- [ ] **Smoke test**: Run basic queries via `POST /retrieve` endpoint
- [ ] **Mode comparison**: Test all 7 retrieval modes (naive, local, global, hybrid, mix, graph_only, chunks_only)
- [ ] **Cache validation**: Verify L2 query cache populates and hits
- [ ] **Streaming**: Test `GET /retrieve/stream` SSE endpoint

### Quality Evaluation

- [ ] **Run RAGAS benchmark**: `tests/eval/test_ragas_benchmark.py` against golden dataset
- [ ] **Run DeepEval**: `tests/eval/test_deepeval_quality.py`
- [ ] **Run graph metrics**: `tests/eval/test_graph_metrics.py`
- [ ] **Baseline comparison**: hybrid+graph vs non-graph vs single-retriever

### Production Hardening

- [ ] **Scheduled re-crawl**: Cron job for periodic URL refresh
- [ ] **Monitoring**: Wire Phoenix tracing + Prometheus metrics to dashboards
- [ ] **CDK deploy**: Update stack with latest Lambda code

---

## 10. Appendix — Commands Reference

### Crawl (full)

```bash
cd apps/rag-service
uv run --extra crawl python -m scripts.crawl_ingest \
  --urls-file scripts/apra_urls.txt --skip-entities
```

### Crawl (resume after interruption)

```bash
uv run --extra crawl python -m scripts.crawl_ingest \
  --urls-file scripts/apra_urls.txt --resume --skip-entities
```

### Crawl (small test batch)

```bash
uv run --extra crawl python -m scripts.crawl_ingest \
  --urls-file scripts/apra_urls.txt --category pages --limit 5 --skip-entities
```

### Entity extraction (backfill)

```bash
uv run python -m scripts.extract_entities --limit 5          # small test
uv run python -m scripts.extract_entities --concurrency 4    # full parallel
uv run python -m scripts.extract_entities --force             # re-extract all
```

### pgvector backfill from Neo4j

```bash
python -m scripts.backfill_pgvector_from_neo4j --dry-run       # preview counts only
python -m scripts.backfill_pgvector_from_neo4j                  # TRUNCATE + backfill
python -m scripts.backfill_pgvector_from_neo4j --skip-truncate  # upsert without truncate
```

### OpenSearch backfill (from PG chunks)

```bash
uv run python -m scripts.backfill_opensearch                  # full backfill
uv run python -m scripts.backfill_opensearch --dry-run        # count only
uv run python -m scripts.backfill_opensearch --recreate       # drop + recreate index first
uv run python -m scripts.backfill_opensearch --category pages # filter by category
```

### Tests

```bash
pnpm rag:test          # 520 tests
pnpm check             # typecheck + lint + rag:lint + rag:test
```

### Background crawl (long-running)

```bash
cd apps/rag-service
nohup uv run --extra crawl python -m scripts.crawl_ingest \
  --urls-file scripts/apra_urls.txt --resume --skip-entities \
  > /tmp/crawl-apra.log 2>&1 &
```

### Monitor background crawl

```bash
tail -f /tmp/crawl-apra.log
# or check progress:
grep -c "✓\|succeeded" /tmp/crawl-apra.log
```
