# RAG Hallucination Risk Analysis (Current Implementation)

## Scope

This document summarizes current hallucination-related risks in the Python RAG service and proposes a pragmatic rollout order.

Reviewed files:

- `apps/rag-service/app/workflow.py`
- `apps/rag-service/app/query_processing.py`
- `apps/rag-service/app/answer_generator.py`
- `apps/rag-service/app/repository.py`
- `apps/rag-service/app/bedrock_action.py`

## Current Strengths

- Grounding prompt already explicitly asks model to only use provided evidence and include citation markers.
- Citation fields at retrieval level are strict (`url/title/year/month` and location metadata constraints).
- Answer temperature is conservative by default (`RAG_ANSWER_TEMPERATURE=0.05`, `QWEN_TEMPERATURE=0.0`).
- Model routing has a weak-evidence signal (`route_min_hits`, `route_top_score_threshold`).

## Main Hallucination Risks

### 1) Hybrid retrieval is not effectively used in workflow path

- In workflow request construction, `RetrieveRequest` is built without `query_embedding`.
- Repository only enters hybrid mode when `query_embedding` exists.
- Impact: retrieval is often sparse-only, so recall can be unstable for semantic questions, increasing synthesis risk.

Code references:

- `apps/rag-service/app/workflow.py`
- `apps/rag-service/app/repository.py`

### 2) Weak evidence does not block generation

- Current logic uses weak evidence only to switch model (`nova-lite` vs `qwen-plus`), but generation still proceeds.
- Impact: model may produce fluent but weakly-supported answers when top score/hit count are insufficient.

Code references:

- `apps/rag-service/app/workflow.py`
- `apps/rag-service/app/answer_generator.py`

### 3) No post-generation citation validation

- Prompt asks for `[1][2]` style markers, but response is not validated before returning.
- Impact: invalid index, missing marker on key claims, or citation-text mismatch can pass through to clients.

Code references:

- `apps/rag-service/app/answer_generator.py`
- `apps/rag-service/app/bedrock_action.py`

### 4) Query rewrite can drift from original constraints

- Rewrite step asks to preserve entities/years/constraints, but no deterministic check enforces preservation.
- Impact: rewritten query may drop key scope terms and retrieve wrong evidence set.

Code reference:

- `apps/rag-service/app/query_processing.py`

### 5) Static `topK` may under-fetch for hard questions

- `topK` defaults to 5 in action handler and is not expanded based on complexity/evidence profile.
- Impact: insufficient evidence coverage for multi-constraint questions.

Code references:

- `apps/rag-service/app/bedrock_action.py`
- `apps/rag-service/app/workflow.py`

## Recommended Implementation Priority

1. Enable true hybrid retrieval in workflow (`query_embedding` path).
2. Add evidence gate (if weak evidence, return explicit "insufficient evidence" instead of generating).
3. Add citation validator before final response.
4. Add rewrite drift guard (entity/year/constraint preservation checks, else fallback to original query).
5. Add dynamic `topK` policy based on complexity and evidence quality.

## Deploy-First Observation Checklist

Before changing logic, deploy current version and capture baseline metrics:

- `% answers with >=1 citation marker`
- `% answers with valid citation marker indices only`
- `avg hit_count`
- `avg top_score` and weak-evidence rate
- `% user follow-up corrections indicating factual mismatch`

Suggested baseline window:

- At least 100 real queries or 3-5 days of traffic (whichever comes first).

## Acceptance Gates for Next Iteration

Use these gates to decide whether to ship mitigation changes:

- Citation validity < 98%
- Weak-evidence answers still generated > 5%
- Factual-mismatch feedback trend is increasing

## Notes

- This document is intentionally focused on risk control and deployment observability.
- It does not replace evaluation framework work (RAGAS/groundedness scoring), which should be layered on top as the next stage.
