# Hybrid RAG + Graph Enhancement Execution Plan

## 1. Objective

Build a production-ready Hybrid RAG system for this repo with:

- dual knowledge sources: `crawler` + `uploaded files`
- hybrid retrieval: `BM25 + dense vector`
- graph-enhanced reasoning over Neo4j
- hard citation contract: `URL + locator (page/section/anchor) + year/month`
- measurable quality improvements validated by `RAGAS` + task-specific checks

Primary success condition:

- hybrid+graph variant must outperform non-graph and single-retriever baselines on a frozen evaluation set, with controlled latency/cost.

## 2. Scope and Non-Goals

In scope:

- Python-first RAG runtime (Node/CDK remains deploy/infrastructure layer)
- Crawl4AI-based crawling pipeline
- file ingestion pipeline (PDF/HTML/Markdown/Text)
- retrieval orchestration and graph augmentation
- evaluation framework and CI quality gates

Out of scope (phase 1):

- full multi-tenant isolation
- advanced human feedback loop tooling UI
- online learning / automated prompt evolution

## 3. Architecture Direction

Target architecture:

1. `Ingestion Layer`

- Crawler ingest (Crawl4AI)
- File ingest (manual upload/batch import)
- Normalization, chunking, dedup, metadata tagging

2. `Index Layer`

- Dense vector store: **Amazon RDS PostgreSQL + pgvector**
- Sparse retrieval index: PostgreSQL full-text index (`tsvector` + GIN) for hybrid retrieval
- Neo4j graph index for entities/relations/doc links

3. `Serving Layer` (Python)

- Query understanding
- Hybrid retrieval (`sparse + dense`)
- Graph expansion (entity neighborhood / path evidence)
- optional reranking
- answer synthesis with citations

4. `Evaluation Layer`

- Existing repo flow:
  - `pnpm eval:agent -- --output-format ragas`
  - `pnpm eval:ragas -- --input ... --output ...`
- plus retrieval and graph-specific custom metrics

## 3.4 Current Workflow Snapshot (Implemented)

Current `rag_search` Python runtime in this repository already uses:

- `LangGraph` for workflow orchestration
- AWS official Python SDK (`boto3`) Bedrock Runtime `converse` for Nova answer synthesis
- Qwen (DashScope compatible API) for intent detection/query rewriting and hard-query fallback
- PostgreSQL retrieval path with strict citation filtering

Current node-level workflow:

1. `detect_intent`

- classify intent and complexity (`factual/analytical/procedural/comparison`)

2. `rewrite_query`

- rewrite retrieval query with constraints preserved (when enabled)

3. `build_request`

- normalize action input (`query`, `topK`, `filters`) into `RetrieveRequest`

4. `retrieve`

- run retrieval from PostgreSQL (`sparse` now; hybrid path is available when embedding input is provided)
- enforce strict citation eligibility constraints before answer synthesis

5. `build_citations`

- map retrieval hits into compact response citation payload (`sourceId`, `title`, `url`, `snippet`)

6. `choose_model`

- default route to `nova-lite`
- route to `qwen-plus` when query is complex or retrieval confidence is weak

7. `generate_answer`

- build evidence block from retrieved chunks
- call routed answer model (`nova-lite` via Bedrock converse, `qwen-plus` via DashScope API)
- return grounded answer text

Mermaid workflow diagram:

```mermaid
flowchart TD
    A[Bedrock Agent Action Group<br/>rag_search] --> B[Lambda Entry<br/>apps/rag-service/lambda_tool.py]
    B --> C[Action Adapter<br/>app/bedrock_action.py]
    C --> D[LangGraph Workflow<br/>app/workflow.py]

    subgraph G[LangGraph Nodes]
      D0[detect_intent]
      D1[rewrite_query]
      D2[build_request]
      D3[retrieve]
      D4[build_citations]
      D5[choose_model]
      D6[generate_answer]
      D0 --> D1 --> D2 --> D3 --> D4 --> D5 --> D6
    end

    D --> D0
    D3 --> E[PostgreSQL + pgvector<br/>app/repository.py]
    D6 --> F[Bedrock Runtime converse<br/>Nova Lite]
    D6 --> K[Qwen Plus API<br/>DashScope compatible endpoint]
    D4 --> H[Structured citations[]]
    F --> I[Grounded answer text]
    K --> I
    H --> J[Bedrock Action JSON response]
    I --> J
```

Runtime source files:

- `apps/rag-service/lambda_tool.py`
- `apps/rag-service/app/bedrock_action.py`
- `apps/rag-service/app/workflow.py`
- `apps/rag-service/app/repository.py`
- `apps/rag-service/app/answer_generator.py`

## 3.1 Vector Store Decision (Final)

Vector store choice for this project:

- **Primary vector store**: Amazon RDS for PostgreSQL with `pgvector`
- **Hybrid retrieval path**:
  - dense retrieval via `pgvector` similarity search
  - sparse retrieval via PostgreSQL full-text search
  - fusion at application layer (weighted rank fusion)

Notes:

- this keeps operational complexity lower than running an extra vector database
- if sparse quality is insufficient, keep an optional fallback workstream to add dedicated BM25 engine later

Reference draft:

- `docs/postgres-pgvector-ddl.sql` (schema, indexes, and hybrid retrieval SQL example)
- `docs/phase1-phase2-migration-checklist.md` (file-level migration checklist and DoD)
- `docs/qwen-plus-entity-extraction-spec.md` (entity extraction model/prompt/schema/gates)

## 3.2 Metadata Contract (Required)

Every stored chunk must include metadata with strict schema and validation.

Required metadata fields:

- `doc_id`: stable document identifier
- `chunk_id`: stable chunk identifier
- `source_type`: `crawler` | `file`
- `source_uri`: canonical URL or file URI/path
- `title`: normalized title
- `lang`: language code (`en`, `zh`, etc.)
- `year`: normalized publication/content year (4-digit)
- `month`: normalized publication/content month (`1-12`)
- `category`: business domain/category for routing/eval grouping
- `created_at`: first ingestion timestamp
- `updated_at`: latest refresh timestamp
- `doc_version`: version hash or revision number
- `content_hash`: normalized content hash for dedup
- `crawl_time`: crawler collection timestamp (crawler rows only)
- `mime_type`: source mime type
- `chunk_index`: chunk order inside document
- `token_count`: estimated token count per chunk

Optional metadata fields (recommended):

- `author`
- `published_at`
- `tags` (string array)
- `access_level` (for future ACL)
- `entity_ids` (linked graph node ids)

Metadata usage requirements:

- retrieval filters (e.g., category/lang/time-range/source_type)
- evaluation grouping (`metadata.category` -> RAGAS group-by)
- observability/tracing (`doc_id`, `chunk_id`, `source_uri`)
- reproducibility (`doc_version`, `content_hash`)

Metadata index requirements (PostgreSQL):

- B-tree: `doc_id`, `source_type`, `category`, `lang`, `updated_at`
- GIN: `tags` (if array/jsonb), `tsvector` column for sparse retrieval
- unique constraints: `chunk_id`, and (`doc_id`, `chunk_index`, `doc_version`)

## 3.3 Citation Contract (Hard Requirement)

Each retrievable chunk must carry structured citation fields, not just free-text metadata.

Required citation fields per chunk:

- `citation_url` (canonical URL or file URI)
- `citation_year` (4-digit year)
- `citation_month` (`1-12`)
- `citation_title`
- at least one locator:
  - `page_start/page_end` for paged documents
  - `section_id` for section-based documents
  - `anchor_id` for web fragment-style documents

Serving requirements:

- response payload must include citation objects with these fields
- citation objects must map to stored chunk ids
- if locator is missing, chunk is not eligible for final answer synthesis

## 4. Workstreams and Milestones

## Phase 0: Baseline Freeze (2-3 days)

Deliverables:

- frozen baseline outputs for current system
- baseline metrics report in `tmp/evals/baseline-*`

Tasks:

- define model/version lock for eval runs
- run current evaluator on fixed dataset
- store outputs as baseline artifacts

Exit criteria:

- baseline JSON artifacts are reproducible from one command sequence

## Phase 1: Knowledge Ingestion (4-6 days)

Deliverables:

- crawler pipeline (Crawl4AI)
- file ingest pipeline
- canonical document schema

Tasks:

- implement crawler job with allowlist/domain rules
- implement file parser/chunker with source metadata
- build dedup strategy (URL canonicalization + content hash)
- extract/normalize citation locators (page/section/anchor) during chunking
- capture provenance fields:
  - `source_type`, `source_uri`, `crawl_time`, `doc_version`

Exit criteria:

- same source can be re-ingested idempotently
- every chunk is traceable to source + version

## Phase 2: Retrieval + Graph Build (5-7 days)

Deliverables:

- BM25 retriever
- vector retriever
- hybrid fusion strategy
- graph extraction + Neo4j upsert pipeline
- PostgreSQL schema for vectors + metadata + sparse index

Tasks:

- implement sparse index and dense index with same chunk ids
- create RDS pgvector schema and migration scripts
- enforce metadata schema validation before upsert
- implement qwen-plus structured extraction with JSON schema validation
- implement rank fusion (e.g., weighted reciprocal rank)
- extract entities/relations and link chunk-node references in Neo4j
- implement graph expansion policies:
  - one-hop entity neighborhood
  - relation filters
  - max expansion budget

Exit criteria:

- query returns: sparse hits, dense hits, fused hits, graph evidence set

## Phase 3: Answering Pipeline (4-6 days)

Deliverables:

- end-to-end query handler (streaming-ready)
- citation-aware response format
- failure-safe fallback paths

Tasks:

- add reranker (optional but strongly recommended)
- define final prompt template with citation grounding constraints
- add fallback policy when graph evidence is weak
- output structured trace fields for evaluation

Exit criteria:

- every answer includes machine-readable citation objects (`url/year/month/locator/chunk_id`)

## Phase 4: Evaluation and Gating (5-7 days)

Deliverables:

- RAGAS + custom evaluation suite
- experiment matrix report
- release gates integrated into CI script

Tasks:

- build eval splits and run ablation experiments
- define pass/fail thresholds
- add regression guard in CI workflow

Exit criteria:

- candidate model/pipeline must pass metric and regression gates before release

## 5. Evaluation Plan (Focus Area)

## 5.1 Evaluation Dataset Strategy

Create 4 datasets:

1. `smoke` (20-30 rows)

- very fast checks per commit

2. `dev` (150-300 rows)

- used for prompt and retriever tuning

3. `frozen_test` (200-400 rows)

- never used for tuning; release gate only

4. `challenge` (50-100 rows)

- adversarial/noisy/long-tail queries

Recommended row fields (compatible with existing scripts):

- `user_input`
- `response`
- `reference`
- `retrieved_contexts`
- `metadata.category` (e.g., `qa`, `work-search`, `graph-qa`)
- `metadata.source_type`, `metadata.doc_id`, `metadata.chunk_id` for traceability
- citation fields per row/chunk: `citation_url`, `citation_year`, `citation_month`, `page_start/page_end`, `section_id`, `anchor_id`

## 5.2 Metrics Stack

Use RAGAS core metrics (already supported in repo):

- `response_relevancy`
- `factual_correctness`
- `semantic_similarity`
- `faithfulness`
- `context_recall`

Add non-RAGAS metrics (required for effectiveness):

- retrieval recall@k / mrr@k
- citation coverage rate
- citation validity rate (URL reachable/schema-valid and locator present)
- unsupported-claim rate
- latency (`p50/p95`) and cost per query

For structured or workflow-style tasks:

- keep using `scripts/eval-work-search.ts` style rule checks
- do not rely only on generic RAGAS scores

## 5.3 Experiment Matrix (Ablation)

Run these variants on `frozen_test`:

- `B0`: dense-only
- `B1`: BM25-only
- `B2`: hybrid (BM25 + dense)
- `B3`: hybrid + graph augmentation
- `B4`: hybrid + graph + reranker

Decision rule:

- promote only if `B3` or `B4` is better than `B2` on quality metrics without violating latency/cost budget.

## 5.4 Acceptance Thresholds (Initial)

Initial release gates (tunable after first full run):

- `faithfulness >= 0.78`
- `factual_correctness >= 0.72`
- `semantic_similarity >= 0.78`
- `context_recall >= 0.70` (for rows with retrieved contexts)
- no metric regression > `2%` vs last accepted baseline
- `p95 latency` increase <= `20%` vs baseline

## 5.5 Runbook Commands

Use existing repo flow:

```bash
pnpm eval:agent -- \
  --agent supervisor \
  --input <dataset>.jsonl \
  --output tmp/evals/<run>-ragas.jsonl \
  --output-format ragas

pnpm eval:ragas -- \
  --input tmp/evals/<run>-ragas.jsonl \
  --output tmp/evals/<run>-results.json
```

For category-specific evaluation, keep `--group-by` behavior and metric subsets per category.

## 6. CI/CD Quality Gate Proposal

Add a CI step that:

1. runs `smoke` set on every PR
2. runs `dev` nightly
3. runs `frozen_test` before release tag
4. fails build if thresholds/regression rules are violated

Artifacts to keep for each run:

- input dataset snapshot hash
- model configuration
- RAGAS output JSON
- summary markdown report

## 7. Risks and Mitigations

Risk: LLM-based judge variance in RAGAS  
Mitigation: fixed evaluator models, deterministic prompts, repeated runs on critical checkpoints

Risk: metric mismatch for structured tasks  
Mitigation: maintain task-specific rule evaluators in parallel

Risk: graph expansion increases latency and noise  
Mitigation: strict expansion budget + reranker + ablation gating

Risk: source drift from crawling  
Mitigation: versioned ingestion with snapshot date and recrawl policy

## 8. Deliverables Checklist for Review

- [ ] Finalized dataset schema and split policy
- [ ] Baseline report and frozen benchmark snapshot
- [ ] Hybrid retriever and graph augmentation design
- [ ] RAGAS + custom metric definitions
- [ ] CI gate thresholds and promotion criteria
- [ ] Rollback plan when release fails quality gate

## 9. Proposed Execution Order

1. Freeze baseline + datasets
2. Implement ingestion pipelines
3. Implement hybrid retrieval
4. Add graph augmentation
5. Run ablation matrix
6. Set gates and enable CI blocking
7. Release first hybrid+graph candidate
