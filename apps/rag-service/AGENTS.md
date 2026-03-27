# apps/rag-service

Python hybrid RAG runtime. Dual entry: FastAPI (`app/main.py`) for local dev, Lambda handler (`lambda_tool.py`) for Bedrock action group.

## WORKFLOW (LangGraph)

```
detect_intent → rewrite_query → build_request → retrieve → build_citations → choose_model → generate_answer
```

Model routing: `nova-lite` default, `qwen-plus` for complex queries or weak evidence.

## WHERE TO LOOK

| Task                 | File                      | Notes                                                         |
| -------------------- | ------------------------- | ------------------------------------------------------------- |
| Lambda entry         | `lambda_tool.py`          | Bedrock action envelope adapter                               |
| Action adapter       | `app/bedrock_action.py`   | Parses Bedrock event → calls workflow                         |
| Workflow graph       | `app/workflow.py`         | LangGraph `StateGraph` with 7 nodes                           |
| Query intent/rewrite | `app/query_processing.py` | Qwen preferred, heuristic fallback                            |
| Retrieval            | `app/repository.py`       | PostgreSQL + OpenSearch hybrid                                |
| Answer synthesis     | `app/answer_generator.py` | Bedrock `converse` + Qwen DashScope                           |
| Qwen client          | `app/qwen_client.py`      | DashScope-compatible OpenAI client                            |
| Secrets              | `app/secrets.py`          | AWS Secrets Manager fetch with caching                        |
| Config               | `app/config.py`           | Pydantic `Settings` from env vars                             |
| Models/types         | `app/models.py`           | `RetrieveRequest`, `RetrievalFilters`, response types         |
| Tests                | `tests/`                  | pytest — `test_workflow.py`, `test_query_processing.py`, etc. |

## CONVENTIONS

- All config via `RAG_*` / `QWEN_*` env vars, read through `app/config.py` Pydantic Settings.
- Retrieval returns strict citation fields: `sourceId`, `title`, `url`, `snippet`.
- Sparse backend is pluggable: `RAG_SPARSE_BACKEND=opensearch|postgres`.
- Hybrid retrieval: BM25 sparse (OpenSearch) fused with pgvector dense via RRF.
- Over-fetch sparse/dense candidates (`k_sparse = max(top_k * 4, 20)`) for stable quality.

## ANTI-PATTERNS

- **Never return answers without citations** — every RAG response must include `citations[]`.
- **Never bypass the workflow** — even simple queries go through all 7 nodes.
- **Secrets are cached** — don't instantiate new Secrets Manager clients per request.
- **`total=False` TypedDict** — `RagWorkflowState` fields are all optional; use `.get()` with defaults.

## COMMANDS

```bash
pnpm rag:install        # uv sync
pnpm rag:dev            # uvicorn --reload :8080
pnpm rag:test           # pytest
pnpm rag:test:cov       # pytest --cov
pnpm rag:lint           # ruff check
pnpm rag:format         # black
```
