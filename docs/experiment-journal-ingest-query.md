# Experiment Journal — Ingest & Query Pipeline

> **Started**: 2026-03-28
> **Last Updated**: 2026-03-29 (evening)
> **Status**: In progress — data stores verified & backfilled, entity extraction pending

---

## Table of Contents

1. [Background & Objectives](#1-background--objectives)
2. [Day 1 (Mar 28) — Foundation Build](#2-day-1-mar-28--foundation-build)
3. [Day 2 (Mar 29) — Crawl Ingest & Production Hardening](#3-day-2-mar-29--crawl-ingest--production-hardening)
4. [Bug Tracker](#4-bug-tracker)
5. [Current Data State](#5-current-data-state)
6. [Next Steps](#6-next-steps)
7. [Appendix — Commands Reference](#7-appendix--commands-reference)

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
2. Even with `.envrc` sourced, the endpoint was `http://localhost:9200` — but ES runs on the Monitoring EC2 instance (`3.107.203.51`), not localhost

**Fix**: Updated `.envrc`:

```
# Before (wrong):
RAG_OPENSEARCH_ENDPOINT="http://localhost:9200"
# After (correct):
RAG_OPENSEARCH_ENDPOINT="http://ec2-3-107-203-51.ap-southeast-2.compute.amazonaws.com:9200"
```

ES confirmed running: Elasticsearch 8.17.0, Docker on MonitoringEc2Stack, `xpack.security.enabled=false` (no auth needed).

**Backfill**: Created `scripts/backfill_opensearch.py` and ran it — 3,103 chunks indexed in 4 seconds (7 bulk requests of 500). See §3.12.

#### Neo4j ❌ → ✅ (after IP fix)

**Problem**: Connection timed out to `bolt://ec2-52-64-132-188....:7687`.

**Root cause**: EC2 instance had a new public IP after restart. Old IP `52.64.132.188` → new IP `54.66.32.125`. Security group rules were fine (ports 7474 + 7687 open to `0.0.0.0/0`).

**Fix**: Updated `.envrc`:

```
# Before (stale IP):
RAG_NEO4J_URI="bolt://ec2-52-64-132-188.ap-southeast-2.compute.amazonaws.com:7687"
# After (current IP):
RAG_NEO4J_URI="bolt://ec2-54-66-32-125.ap-southeast-2.compute.amazonaws.com:7687"
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

## 4. Bug Tracker

| #   | Bug                                   | Root Cause                                                | Fix                              | Severity |
| --- | ------------------------------------- | --------------------------------------------------------- | -------------------------------- | -------- |
| 1   | `::type` cast breaks psycopg3         | psycopg3 doesn't support `::` in parameterized queries    | `CAST(x AS type)` across 3 files | High     |
| 2   | `QueryCache(settings)` missing engine | Constructor signature mismatch                            | Added `engine` parameter         | High     |
| 3   | Resume skips nothing                  | `WHERE source_type = 'web_crawl'` vs actual `'crawler'`   | Fixed to `'crawler'`             | High     |
| 4   | Stale `ingestion_runs`                | No cleanup on process kill                                | `try/except` + mark failed       | Medium   |
| 5   | Tests fail with `.envrc`              | Env var leakage into test environment                     | `monkeypatch.delenv()`           | Low      |
| 6   | Qwen `IncompleteRead`                 | Embedding API timeout/disconnect                          | Retry + CircuitBreaker policies  | Medium   |
| 7   | ES endpoint → localhost               | `.envrc` pointed to `localhost:9200`, ES is on remote EC2 | Updated to EC2 public DNS        | High     |
| 8   | Neo4j stale IP                        | EC2 public IP changed after restart                       | Updated `.envrc` with new IP     | Medium   |

---

## 5. Current Data State

### PostgreSQL

| Table            | Records | Notes                               |
| ---------------- | ------- | ----------------------------------- |
| `kb_documents`   | 1,300   | 1,262 pages + 34 awards + 4 news    |
| `kb_chunks`      | 3,103   | With embeddings (1024-dim pgvector) |
| `ingestion_runs` | 1,329   | All `succeeded`                     |
| `kb_entities`    | 0       | Entity extraction not yet run       |
| `kb_relations`   | 0       | Entity extraction not yet run       |
| `query_cache`    | 0       | No queries run yet                  |

### Elasticsearch

| Index       | Documents | Notes                             |
| ----------- | --------- | --------------------------------- |
| `kb_chunks` | 3,103     | BM25 full-text search, backfilled |

Endpoint: `http://ec2-3-107-203-51.ap-southeast-2.compute.amazonaws.com:9200`
Version: Elasticsearch 8.17.0 (Docker on MonitoringEc2Stack), no auth.

### Neo4j

| Type           | Count | Notes                     |
| -------------- | ----- | ------------------------- |
| Entity nodes   | 0     | Pending entity extraction |
| Relation edges | 0     | Pending entity extraction |

Endpoint: `bolt://ec2-54-66-32-125.ap-southeast-2.compute.amazonaws.com:7687`
Version: Neo4j 5.26.21, Protocol 5.8. Connectivity verified ✅.

---

## 6. Next Steps

### Immediate

- [x] **Verify PostgreSQL**: 1,300 docs, 3,103 chunks, all ingestion runs succeeded
- [x] **Verify Elasticsearch**: 3,103 docs indexed via backfill script
- [x] **Fix `.envrc`**: Updated ES endpoint (localhost → remote EC2) and Neo4j URI (stale IP)
- [ ] **Run entity extraction**: `scripts/extract_entities.py` on all 1,300 docs
- [ ] **Verify Neo4j**: Confirm entities and relations are populated

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

## 7. Appendix — Commands Reference

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

### OpenSearch backfill (from PG chunks)

```bash
uv run python -m scripts.backfill_opensearch                  # full backfill
uv run python -m scripts.backfill_opensearch --dry-run        # count only
uv run python -m scripts.backfill_opensearch --recreate       # drop + recreate index first
uv run python -m scripts.backfill_opensearch --category pages # filter by category
```

### Tests

```bash
pnpm rag:test          # 508 tests
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
