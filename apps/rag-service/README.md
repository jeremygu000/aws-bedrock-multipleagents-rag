# Python Hybrid RAG Service

This service is the Python runtime for the hybrid RAG path:

- sparse retrieval on OpenSearch (BM25)
- dense retrieval on `pgvector`
- weighted reciprocal rank fusion (RRF)
- strict citation payload in every returned hit
- Bedrock Action Group Lambda handler (`lambda_tool.py`)
- LangGraph workflow orchestration for retrieval + answer synthesis
- AWS Bedrock Runtime `converse` for grounded answer generation

### Phase 1 — LightRAG Enhancements (Complete)

Techniques ported from [LightRAG (EMNLP 2025)](https://github.com/hkuds/lightrag):

- **Keyword extraction**: dual-level extraction (high-level themes + low-level entities) via Qwen, used to expand query rewriting with entity context
- **LLM reranking**: Qwen-based relevance scoring with token-budget awareness and graceful fallback to retrieval order
- **Structured evidence prompts**: `--- Evidence [N] ---` blocks with source metadata for grounded answer synthesis

### Phase 1.5 — Document Ingestion Pipeline (Complete)

End-to-end document ingestion supporting TXT, Markdown, PDF, DOCX, and HTML:

- **Document parser**: Multi-format parsing with section/heading extraction, page tracking (PDF), and metadata detection
- **Hybrid chunker**: Header-based splitting with fixed-size fallback, configurable size/overlap/minimum, locator guarantee per chunk
- **Batch embedding**: Qwen `text-embedding-v3` with configurable batch size for efficient embedding generation
- **Ingestion repository**: PostgreSQL (`ingestion_runs`, `kb_documents`, `kb_chunks`) + OpenSearch bulk indexing
- **Pipeline orchestrator**: S3 download → parse → chunk → embed → upsert → index, with full error tracking
- **Upload endpoint**: `POST /upload` (multipart form) + `GET /ingestion/{run_id}` status tracking
- **Lambda handler**: SQS-triggered async processing with partial batch failure reporting
- **CDK infrastructure**: S3 bucket, SQS queue + DLQ, ingestion Lambda, S3→SQS event notification

New environment variables:

- `RAG_S3_BUCKET` — S3 bucket for document uploads
- `RAG_INGESTION_QUEUE_URL` — SQS queue URL (empty = sync mode)
- `RAG_CHUNK_SIZE` (default: `512`), `RAG_CHUNK_OVERLAP` (default: `64`), `RAG_CHUNK_MIN_SIZE` (default: `50`)
- `RAG_EMBED_BATCH_SIZE` (default: `20`), `RAG_MAX_UPLOAD_SIZE_MB` (default: `50`)

## 0. Workflow Overview

Current `rag_search` runtime flow (9-node LangGraph pipeline):

```text
detect_intent → extract_keywords → rewrite_query → build_request → retrieve → rerank → build_citations → choose_model → generate_answer
```

1. Bedrock Action Group invokes `lambda_tool.handler`
2. `lambda_tool` delegates to `handle_rag_action`
3. `handle_rag_action` runs `RagWorkflow` (LangGraph)
4. Workflow nodes:
   - `detect_intent`: classify intent/complexity (Qwen preferred, heuristic fallback)
   - `extract_keywords`: dual-level keyword extraction — high-level themes + low-level entities (Qwen, skip on failure)
   - `rewrite_query`: retrieval rewrite with keyword context (Qwen preferred, pass-through fallback)
   - `build_request`: normalize query/topK/filters and build query embedding when available
   - `retrieve`: fetch strict-citation hits from OpenSearch sparse + pgvector dense, fused via RRF
   - `rerank`: LLM-based relevance scoring with token budget (Qwen, graceful fallback to retrieval order)
   - `build_citations`: map hits to compact citation payload
   - `choose_model`: route default `nova-lite` vs `qwen-plus` for hard/low-confidence cases
   - `generate_answer`: synthesize answer with structured evidence prompts via routed model
5. Response is returned in Bedrock action envelope with:
   - `answer`
   - `citations[]` (`sourceId`, `title`, `url`, `snippet`)

Key files:

- `apps/rag-service/lambda_tool.py`
- `apps/rag-service/app/bedrock_action.py`
- `apps/rag-service/app/workflow.py`
- `apps/rag-service/app/query_processing.py` (intent detection, keyword extraction, query rewrite)
- `apps/rag-service/app/repository.py` (OpenSearch + pgvector hybrid retrieval)
- `apps/rag-service/app/reranker.py` (LLM reranking)
- `apps/rag-service/app/answer_generator.py`

## 1. Install

```bash
uv sync --project apps/rag-service
```

## 2. Required Environment Variables

The service reads `RAG_*` variables first, then falls back to existing `RDS_*` values.

- `RAG_DB_HOST` (fallback: `RDS_HOST`)
- `RAG_DB_PORT` (default: `5432`)
- `RAG_DB_NAME` (default: `postgres`)
- `RAG_DB_USER` (fallback: `RDS_MASTER_USERNAME`, default: `postgres`)
- `RAG_DB_PASSWORD_SECRET_ARN` (recommended)
- `RAG_DB_PASSWORD_SECRET_JSON_KEY` (default: `password`)
- `RAG_DB_PASSWORD` (local fallback only)
- `RAG_DB_SSLMODE` (default: `require`)
- `RAG_EMBED_DIM` (default: `1024`)
- `RAG_SPARSE_BACKEND` (`opensearch` or `postgres`, default: `opensearch`)
- `RAG_OPENSEARCH_ENDPOINT` (required when `RAG_SPARSE_BACKEND=opensearch`)
- `RAG_OPENSEARCH_INDEX` (default: `kb_chunks`)
- `RAG_OPENSEARCH_TIMEOUT_S` (default: `10`)
- `RAG_ANSWER_MODEL_ID` (default: `amazon.nova-lite-v1:0`)
- `RAG_ANSWER_MAX_TOKENS` (default: `500`)
- `RAG_ANSWER_TEMPERATURE` (default: `0.05`)
- `RAG_AWS_REGION` (fallback: `AWS_REGION` / `AWS_DEFAULT_REGION`)
- `QWEN_API_KEY_SECRET_ARN` (recommended)
- `QWEN_API_KEY_SECRET_KEY` (default: `DASHSCOPE_API_KEY`)
- `QWEN_API_KEY` (local fallback, fallback alias: `DASHSCOPE_API_KEY`)
- `QWEN_MODEL_ID` (fallback: `LLM_MODEL`, default: `qwen-plus`)
- `QWEN_EMBEDDING_MODEL_ID` (default: `text-embedding-v3`)
- `QWEN_BASE_URL` (default: `https://dashscope.aliyuncs.com/compatible-mode/v1`)
- `QWEN_MAX_TOKENS` (default: `500`)
- `QWEN_TEMPERATURE` (default: `0.0`)
- `RAG_ROUTE_MIN_HITS` (default: `3`)
- `RAG_ROUTE_TOP_SCORE_THRESHOLD` (default: `0.015`)
- `RAG_ROUTE_COMPLEX_QUERY_TOKEN_THRESHOLD` (default: `18`)
- `RAG_ENABLE_QUERY_REWRITE` (default: `true`)
- `RAG_ENABLE_HYBRID_RETRIEVAL` (default: `true`)
- `RAG_ENABLE_KEYWORD_EXTRACTION` (default: `true`)
  - enable dual-level keyword extraction before query rewrite
- `RAG_ENABLE_RERANKING` (default: `true`)
  - enable LLM-based reranking of retrieval results
- `RAG_RERANK_CANDIDATE_COUNT` (default: `20`)
  - number of candidates to retrieve before reranking down to `k_final`
- `RAG_RERANK_MAX_TOKENS` (default: `30000`)
  - token budget for the reranking context window

For Bedrock Action Group Lambda execution, the same variables are used via Lambda environment config.

## 3. Run

```bash
uv run --project apps/rag-service uvicorn app.main:app --app-dir apps/rag-service --host 0.0.0.0 --port 8080 --reload
```

## 4. API

Health:

```bash
curl http://127.0.0.1:8080/healthz
```

Sparse retrieval:

```bash
curl -X POST http://127.0.0.1:8080/retrieve \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "what is APRA policy update",
    "k_sparse": 20,
    "k_final": 5,
    "filters": { "source_type": "crawler", "citation_year_from": 2024 }
  }'
```

For workflow and Bedrock action paths, query embedding is generated automatically
through Qwen embeddings when configured and dimensions match `RAG_EMBED_DIM`.
When `RAG_SPARSE_BACKEND=opensearch`, sparse candidates come from OpenSearch BM25 and are fused
with PostgreSQL dense candidates. If OpenSearch is unavailable, the service falls back to
PostgreSQL sparse retrieval to keep runtime resilient.

When reranking is enabled (`RAG_ENABLE_RERANKING=true`), retrieved candidates are scored by
Qwen for relevance before being passed to answer generation. The reranker respects a
configurable token budget (`RAG_RERANK_MAX_TOKENS`) and falls back gracefully to retrieval
order on failure.

## 5. Lambda Entry

CDK points Bedrock `rag_search` action group to:

- file: `apps/rag-service/lambda_tool.py`
- handler: `handler`

This handler reads Bedrock action input, runs retrieval, and returns `answer + citations`.

## 6. Document Ingestion

Upload a document:

```bash
curl -X POST http://127.0.0.1:8080/upload \
  -F "file=@document.pdf" \
  -F "title=My Document" \
  -F "published_year=2024" \
  -F "published_month=6" \
  -F "lang=en" \
  -F "category=general"
```

Check ingestion status:

```bash
curl http://127.0.0.1:8080/ingestion/{run_id}
```

Supported formats: `.txt`, `.md`, `.pdf`, `.docx`, `.html`

When `RAG_INGESTION_QUEUE_URL` is empty, ingestion runs synchronously (useful for dev/testing).
When set, the upload is stored in S3 and processed asynchronously via SQS → Lambda.
