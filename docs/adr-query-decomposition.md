# ADR: Query Decomposition for Multi-Hop Reasoning in RAG

**Date**: 2026-04-07
**Status**: Accepted
**Context**: Complex multi-hop queries retrieve irrelevant documents because a single embedding cannot capture multiple information needs simultaneously.
**Decision**: Implement query decomposition with adaptive routing to break complex queries into focused sub-questions before retrieval.

## Problem

Current RAG pipeline (LangGraph) uses a single query embedding for retrieval. This fails on:

- **Multi-hop queries**: "Compare APRA licensing rules with US copyright law" needs two separate retrievals
- **Compound questions**: "What are the fees and how do I apply?" conflates distinct intents
- **Reasoning-heavy queries**: "Explain why streaming royalties differ across platforms" requires background + analysis

Benchmarks show +10-15% EM improvement on multi-hop QA datasets (HotpotQA, MuSiQue) when decomposition is applied.

## Solution

**QueryDecomposer** (pre-retrieval) with **5-rule routing** to decide when to decompose:

```
build_request → [should_decompose?] → decompose_query → parallel_retrieve → merge_deduplicate → rerank
                        ↓
                    (skip)
                        ↓
                    retrieve (single query)
```

### Routing Rules (Priority Order)

| Rule | Condition                               | Action                        | Confidence |
| ---- | --------------------------------------- | ----------------------------- | ---------- |
| 1    | Token count < 15                        | Skip                          | 0.85       |
| 2    | Entity keywords (who/find/list/search)  | Skip                          | 0.80       |
| 3    | Reasoning keywords + semantic_gap > 0.3 | Decompose (2-3 sub-questions) | 0.85       |
| 4    | High complexity + tokens > 20           | Decompose (3 sub-questions)   | 0.75       |
| 5    | Default                                 | Skip                          | 0.70       |

### Sub-Question Generation

LLM-based decomposition via Bedrock Nova Pro:

- System prompt instructs structured JSON output
- Each sub-question has: `id`, `question`, `focus`, `retrieve_strategy` (dense/sparse/hybrid)
- Fallback: deterministic sub-questions on LLM failure or bad JSON
- Clamped to 2-5 sub-questions (min 2, max from config)

### Parallel Retrieval

`DecompositionRetriever` executes sub-questions in parallel:

- Each sub-question runs through the existing retrieval pipeline independently
- Results are merged and deduplicated by `chunk_id`
- Scores are preserved from highest-scoring occurrence
- Falls back to original single-query retrieval on any failure

## Cost-Latency Tradeoff

| Scenario               | Queries Decomposed | Latency  | Cost (1M q/mo) | Recall Improvement |
| ---------------------- | ------------------ | -------- | -------------- | ------------------ |
| No decomposition       | 0%                 | 2-3s     | Baseline       | 0%                 |
| All queries            | 100%               | 4-7s     | +$300-500      | +10-15%            |
| Adaptive (RECOMMENDED) | ~25-35%            | 2.5-3.5s | +$75-175       | +8-12%             |

**Conclusion**: Routing skips ~65-75% of queries (entity/short/simple), applying decomposition only to reasoning-heavy queries where it provides meaningful recall gains.

## Implementation

### Modules

- `query_decomposer.py` (349 lines)
  - `QueryDecomposer`: 5-rule routing + Bedrock LLM sub-question generation
  - `DecompositionDecision`: Pydantic model for routing verdict
  - `SubQuestion`: Pydantic model with validation (min length 10, id 1-5)
  - `DecompositionResult`: Complete result with sub-questions list

- `decomposition_retriever.py`
  - `DecompositionRetriever`: Parallel sub-question retrieval
  - Merge + deduplicate via `_merge_decomposition_hits()`
  - Graceful fallback to main query on failure

### LangGraph Integration

```python
# In RagWorkflowState TypedDict
should_decompose: bool
decomposition_decision: DecompositionDecision | None
sub_questions: list[SubQuestion]
sub_question_hits_list: list[list[dict]]
decomposition_used: bool

# In RagWorkflow.__init__
self._query_decomposer = QueryDecomposer(settings=settings, bedrock_client=bedrock)

# Conditional edge after build_request
graph.add_conditional_edges(
    "build_request",
    self._route_after_build_request,
    {"decompose_query": "decompose_query", "retrieve": "retrieve"}
)
graph.add_edge("decompose_query", "retrieve")
```

### Configuration

In `config.py`:

```python
RAG_ENABLE_QUERY_DECOMPOSITION: bool = True
RAG_DECOMPOSITION_MIN_TOKENS: int = 15
RAG_DECOMPOSITION_SEMANTIC_GAP_THRESHOLD: float = 0.3
RAG_DECOMPOSITION_MAX_SUBQUESTIONS: int = 3
RAG_DECOMPOSITION_TIMEOUT_S: int = 10
RAG_DECOMPOSITION_MODEL_ID: str = "amazon.nova-pro-v1:0"
```

## Monitoring & Metrics

Track via CloudWatch:

- **decomposition_rate**: Fraction of queries decomposed (target 25-35%)
- **decomposition_latency_p99_ms**: LLM sub-question generation time
- **sub_question_count_avg**: Average sub-questions per decomposed query
- **fallback_rate**: Fraction using deterministic fallback sub-questions
- **recall_improvement**: A/B comparison vs non-decomposed baseline

## Alternatives Considered

1. **Multi-Query Retriever (LangChain)**: Generates query variants, not focused sub-questions. Less effective for multi-hop.
2. **IRCoT (Interleaving Retrieval + CoT)**: Iterative retrieve-then-reason. Higher latency (+2-3 LLM calls per hop).
3. **CoRAG (Chain-of-Retrieval)**: o1-like RAG with dynamic query reformulation. Requires fine-tuned model.

Query decomposition was chosen for plug-and-play integration (no model training), predictable latency (single LLM call for decomposition), and compatibility with existing parallel retrieval infrastructure.

## References

- **Decomposed Prompting** (Khot et al., arXiv:2210.02406, ICLR 2023) — Modular task decomposition
- **Least-to-Most Prompting** (Zhou et al., arXiv:2205.10625, ICLR 2023) — Progressive sub-problem solving
- **IRCoT** (Trivedi et al., arXiv:2212.10509, ACL 2023) — Interleaved retrieval + chain-of-thought
- **Self-Ask** (Press et al., arXiv:2210.03350, EMNLP 2023) — Explicit follow-up question generation
- **CoRAG** (Wang et al., arXiv:2501.14342, NeurIPS 2025) — Chain-of-retrieval augmented generation
- **CompactRAG** (Yang et al., arXiv:2602.05728, 2026) — Cost-efficient multi-hop decomposition
- **HotpotQA** (Yang et al., arXiv:1809.09600, EMNLP 2018) — Multi-hop QA benchmark
- **MuSiQue** (Trivedi et al., arXiv:2108.00573, TACL 2022) — Composable multi-hop QA benchmark
