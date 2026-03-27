# Phase 1/2 Migration Checklist (RDS PostgreSQL + pgvector)

## 1. Goal

Turn the design in `docs/postgres-pgvector-ddl.sql` into executable migrations and validate:

- schema correctness
- metadata contract enforcement
- hybrid retrieval readiness (dense + sparse)

## 2. Proposed Migration Files

Create these SQL migration files (or Alembic revisions with equivalent SQL):

1. `migrations/0001_extensions.sql`
2. `migrations/0002_ingestion_runs.sql`
3. `migrations/0003_kb_documents.sql`
4. `migrations/0004_kb_chunks.sql`
5. `migrations/0005_chunk_graph_links.sql`
6. `migrations/0006_indexes.sql`
7. `migrations/0007_updated_at_triggers.sql`

## 3. Phase 1 Checklist (Ingestion + Metadata Foundation)

## 3.1 Database Setup

- [ ] Apply `0001_extensions.sql`
- [ ] Verify `vector`, `pgcrypto`, `pg_trgm` extensions exist
- [ ] Record extension versions in deployment log

Validation:

```sql
SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pgcrypto', 'pg_trgm');
```

## 3.2 Provenance Run Tracking

- [ ] Apply `0002_ingestion_runs.sql`
- [ ] Insert a test run row (`source_type='crawler'`, `status='running'`)
- [ ] Update status to `succeeded` and set `finished_at`

Validation:

```sql
SELECT run_id, source_type, status, started_at, finished_at
FROM ingestion_runs
ORDER BY started_at DESC
LIMIT 5;
```

## 3.3 Document-Level Metadata

- [ ] Apply `0003_kb_documents.sql`
- [ ] Enforce required metadata fields at application layer
- [ ] Enforce `published_year/published_month` and metadata `year/month` consistency
- [ ] Confirm unique constraint on `(source_uri, doc_version)`

Validation:

```sql
\d+ kb_documents
```

## 3.4 Chunk-Level Metadata and Vectors

- [ ] Apply `0004_kb_chunks.sql`
- [ ] Confirm `embedding VECTOR(1024)` matches embedding model config
- [ ] Confirm generated `tsv` column exists
- [ ] Confirm unique constraint `(doc_id, chunk_index, doc_version)`
- [ ] Confirm strict citation columns exist (`citation_url`, `citation_year`, `citation_month`, locator fields)
- [ ] Confirm locator check constraint rejects chunks without page/section/anchor

Validation:

```sql
\d+ kb_chunks
```

## 3.5 Metadata Contract Enforcement

- [ ] Add ingestion-time metadata validation (reject invalid rows)
- [ ] Block inserts with missing required metadata fields
- [ ] Add explicit error logging for invalid metadata rows

Required fields to validate:

- `doc_id`
- `chunk_id`
- `source_type`
- `source_uri`
- `title`
- `lang`
- `year`
- `month`
- `category`
- `doc_version`
- `content_hash`
- `mime_type`
- `chunk_index`
- `token_count`

Required citation fields to validate:

- `citation_url`
- `citation_title`
- `citation_year`
- `citation_month`
- at least one locator:
  - `page_start`
  - `section_id`
  - `anchor_id`

## 4. Phase 2 Checklist (Hybrid Retrieval + Graph Linking)

## 4.1 Graph Evidence Link Table

- [ ] Apply `0005_chunk_graph_links.sql`
- [ ] Test insert link rows for existing chunk ids
- [ ] Confirm uniqueness on `(chunk_id, graph_node_id, graph_relation_type)`

Validation:

```sql
\d+ kb_chunk_graph_links
```

## 4.2 Performance Indexes

- [ ] Apply `0006_indexes.sql`
- [ ] Validate GIN index on `kb_chunks.tsv`
- [ ] Validate IVF index on `kb_chunks.embedding`
- [ ] Validate metadata filter indexes on documents/chunks

Validation:

```sql
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename IN ('kb_documents', 'kb_chunks')
ORDER BY tablename, indexname;
```

## 4.3 Update Triggers

- [ ] Apply `0007_updated_at_triggers.sql`
- [ ] Update one row in `kb_documents` and `kb_chunks`
- [ ] Confirm `updated_at` changes automatically

Validation:

```sql
SELECT doc_id, updated_at FROM kb_documents ORDER BY updated_at DESC LIMIT 3;
SELECT chunk_id, updated_at FROM kb_chunks ORDER BY updated_at DESC LIMIT 3;
```

## 4.4 Hybrid Retrieval Smoke Test

- [ ] Load a small fixture dataset (at least 50 chunks)
- [ ] Run dense-only query
- [ ] Run sparse-only query
- [ ] Run fused retrieval query (RRF)
- [ ] Confirm result set contains strict citation payload (`citation_url`, `citation_year`, `citation_month`, locator)
- [ ] Confirm rows lacking locator are excluded from final synthesis candidate set

Reference SQL:

- use the example at the end of `docs/postgres-pgvector-ddl.sql`

## 5. Rollback Plan

Rollback order (reverse migration sequence):

1. drop triggers/functions
2. drop secondary indexes
3. drop link/chunk/document/run tables
4. keep extensions unless explicitly required to remove

Rollback checkpoints:

- after `0004` (core schema ready)
- after `0006` (indexing changes)
- after first production-like load test

## 6. Definition of Done

Phase 1 done when:

- all core tables exist
- metadata validation is enforced
- sample ingest succeeds end-to-end

Phase 2 done when:

- hybrid query path is functional
- graph links are queryable
- smoke retrieval quality and latency are within expected range

## 7. Next Step After Migration

Immediately start evaluation wiring:

- generate `ragas`-shaped outputs from hybrid pipeline
- run `pnpm eval:ragas` on baseline and hybrid variants
- compare against acceptance thresholds in `docs/hybrid-rag-execution-plan.md`
