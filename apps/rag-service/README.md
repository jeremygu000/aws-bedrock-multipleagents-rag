# Python Hybrid RAG Service

This service is the Python runtime for the hybrid RAG path:

- sparse retrieval on PostgreSQL full-text (`tsvector`)
- dense retrieval on `pgvector`
- weighted reciprocal rank fusion (RRF)
- strict citation payload in every returned hit
- Bedrock Action Group Lambda handler (`lambda_tool.py`)
- LangGraph workflow orchestration for retrieval + answer synthesis
- AWS Bedrock Runtime `converse` for grounded answer generation

## 0. Workflow Overview

Current `rag_search` runtime flow:

1. Bedrock Action Group invokes `lambda_tool.handler`
2. `lambda_tool` delegates to `handle_rag_action`
3. `handle_rag_action` runs `RagWorkflow` (LangGraph)
4. Workflow nodes:
   - `detect_intent`: classify intent/complexity (Qwen preferred, heuristic fallback)
   - `rewrite_query`: retrieval rewrite (Qwen preferred, pass-through fallback)
   - `build_request`: normalize query/topK/filters into `RetrieveRequest`
   - `retrieve`: fetch strict-citation hits from PostgreSQL (`sparse` now, `hybrid` ready)
   - `build_citations`: map hits to compact citation payload
   - `choose_model`: route default `nova-lite` vs `qwen-plus` for hard/low-confidence cases
   - `generate_answer`: synthesize answer via routed model
5. Response is returned in Bedrock action envelope with:
   - `answer`
   - `citations[]` (`sourceId`, `title`, `url`, `snippet`)

Key files:

- `apps/rag-service/lambda_tool.py`
- `apps/rag-service/app/bedrock_action.py`
- `apps/rag-service/app/workflow.py`
- `apps/rag-service/app/repository.py`
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
- `RAG_ANSWER_MODEL_ID` (default: `amazon.nova-lite-v1:0`)
- `RAG_ANSWER_MAX_TOKENS` (default: `500`)
- `RAG_ANSWER_TEMPERATURE` (default: `0.05`)
- `RAG_AWS_REGION` (fallback: `AWS_REGION` / `AWS_DEFAULT_REGION`)
- `QWEN_API_KEY_SECRET_ARN` (recommended)
- `QWEN_API_KEY_SECRET_KEY` (default: `DASHSCOPE_API_KEY`)
- `QWEN_API_KEY` (local fallback, fallback alias: `DASHSCOPE_API_KEY`)
- `QWEN_MODEL_ID` (fallback: `LLM_MODEL`, default: `qwen-plus`)
- `QWEN_BASE_URL` (default: `https://dashscope.aliyuncs.com/compatible-mode/v1`)
- `QWEN_MAX_TOKENS` (default: `500`)
- `QWEN_TEMPERATURE` (default: `0.0`)
- `RAG_ROUTE_MIN_HITS` (default: `3`)
- `RAG_ROUTE_TOP_SCORE_THRESHOLD` (default: `0.015`)
- `RAG_ROUTE_COMPLEX_QUERY_TOKEN_THRESHOLD` (default: `18`)
- `RAG_ENABLE_QUERY_REWRITE` (default: `true`)

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

Hybrid retrieval requires `query_embedding` length = `RAG_EMBED_DIM` (default `1024`).
If the embedding length is wrong, the API returns `400`.

## 5. Lambda Entry

CDK points Bedrock `rag_search` action group to:

- file: `apps/rag-service/lambda_tool.py`
- handler: `handler`

This handler reads Bedrock action input, runs retrieval, and returns `answer + citations`.
