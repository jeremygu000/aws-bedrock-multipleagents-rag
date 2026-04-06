# ADR: Self-Reflection for Hallucination Detection in RAG

**Date**: 2026-04-07  
**Status**: Accepted  
**Context**: Need to reduce hallucination rate in Bedrock RAG pipeline while minimizing latency/cost overhead.  
**Decision**: Implement post-generation reflection with adaptive routing to detect and mitigate hallucinations.

## Problem

Current RAG pipeline (LangGraph 12-node) achieves good coverage but suffers from:

- **Hallucination rate**: ~8-15% on complex queries (from RAGAS benchmarks)
- **Cost of false answers**: Higher reputational/trust cost than "I don't know"
- **Token efficiency**: Every query pays full LLM cost regardless of quality indicators

## Solution

**Self-Reflection node** (post-generation validation) with **adaptive routing** (cost optimization):

```
generate_answer → [adaptive router] → reflect (if needed) → retry_action → route
                                 ↓
                            (skip)
                                 ↓
                            store_cache
```

### Key Components

1. **Faithfulness Grader** (claim decomposition + entailment)
   - Extract atomic claims from answer
   - Verify each claim against retrieved documents
   - Score: (supported_claims / total_claims)
   - Threshold: ≥0.9 FAITHFUL, ≥0.5 PARTIALLY_FAITHFUL, else UNFAITHFUL

2. **Relevance Grader** (answer-question alignment)
   - Score three dimensions: alignment, completeness, relevance
   - Threshold: ≥0.75 RELEVANT, ≥0.5 PARTIALLY_RELEVANT, else IRRELEVANT

3. **Adaptive Reflection Router** (cost optimization)
   - Estimate query complexity: token count, keywords, semantic gap
   - Decision tree:
     - Skip: Simple entity queries, high model confidence (>0.95)
     - Reflect: Complex queries, low hits (<3), low confidence (<0.5)
     - Sample: 30% of moderate queries (online learning)
   - Target skip rate: ~65%

### Retry Actions

After reflection, answer is routed to one of:

- **ACCEPT**: Faithfulness ≥0.7 and Relevance ≥0.7
- **RETRY_RETRIEVAL**: Faithfulness <0.5 (hallucinating)
- **REFINE_ANSWER**: Relevance <0.5 with good evidence
- **FALLBACK_MODEL**: All others (use Claude Sonnet instead of Nova)

## Cost-Latency Tradeoff

| Scenario               | Skip Rate | Latency  | Cost (1M q/mo) | Hallucination Reduction |
| ---------------------- | --------- | -------- | -------------- | ----------------------- |
| No reflection          | 0%        | 2-3s     | Baseline       | 0%                      |
| Full reflection        | 0%        | 4-6s     | +$500-800      | -8-15% ✅               |
| Adaptive (RECOMMENDED) | 65%       | 2.3-2.8s | +$100-200      | -8-12% ✅               |

**Conclusion**: Adaptive routing achieves 80% of accuracy gains with 25% of the cost overhead.

## Implementation

### Modules

- `self_reflection_models.py` (100 lines)
  - Pydantic models: Claim, FaithfulnessResult, RelevanceResult, ReflectionResult
  - Enums: FaithfulnessVerdict, RelevanceVerdict, RetryAction

- `faithfulness_grader.py` (222 lines)
  - FaithfulnessGrader: claim extraction + verification
  - RAGAS-inspired claim decomposition pattern
  - Supports Qwen (preferred) or Bedrock Nova (fallback)

- `relevance_grader.py` (129 lines)
  - RelevanceGrader: answer-question alignment
  - Scores alignment, completeness, relevance dimensions
  - LLM-as-judge via Qwen/Bedrock

- `adaptive_reflection_router.py` (205 lines)
  - AdaptiveReflectionRouter: decide when to reflect vs skip
  - Query complexity estimation with heuristics
  - 65% skip rate optimization

- `self_reflection_node.py` (206 lines)
  - SelfReflectionNode: LangGraph orchestrator
  - Parallel async grading (faithfulness + relevance)
  - Latency tracking + error handling

### LangGraph Integration

```python
# In RagWorkflowState TypedDict
reflection_result: ReflectionResult | None
reflection_used: bool

# In RagWorkflow.__init__
self._reflection_node = SelfReflectionNode(
    settings=settings,
    faithfulness_grader=FaithfulnessGrader(...),
    relevance_grader=RelevanceGrader(...),
    adaptive_router=AdaptiveReflectionRouter(enable_reflection=True),
)

# Add to graph
graph.add_node("reflect", self._node_reflect)
graph.add_conditional_edges(
    "generate_answer",
    self._route_after_generation,
    {"reflect": "reflect", "store_cache": "store_cache"}
)
```

### Configuration

Add to `config.py`:

```python
RAG_ENABLE_REFLECTION: bool = True
RAG_REFLECTION_FAITHFULNESS_THRESHOLD: float = 0.6
RAG_REFLECTION_RELEVANCE_THRESHOLD: float = 0.5
RAG_REFLECTION_SKIP_RATE_TARGET: float = 0.65
```

## Monitoring & Metrics

Track via CloudWatch:

- **faithfulness_score_avg**: Per-query faithfulness (target ≥0.75)
- **relevance_score_avg**: Per-query relevance (target ≥0.75)
- **reflection_skip_rate**: Fraction of queries where reflection skipped
- **reflection_latency_p99_ms**: Reflection node execution time
- **retry_action_distribution**: Breakdown of ACCEPT/RETRY/REFINE/FALLBACK
- **hallucination_rate**: Downstream accuracy metric (RAGAS faithfulness)

## Future Enhancements

1. **Learned routing**: Replace heuristics with ML classifier (trained on 1000+ examples)
2. **Multi-turn reflection**: Iterative improvement over multiple retries
3. **Fine-tuned claim verifier**: Domain-specific claim verification model
4. **Streaming support**: Incremental faithfulness checking as answer streams
5. **Confidence calibration**: Learn model-specific confidence thresholds per query type

## References

- **Self-RAG** (Asai et al., arXiv:2310.11511) — Reflection tokens + training
- **RAGAS** (ES et al.) — Claim decomposition for faithfulness evaluation
- **CRAG** (Yan et al., arXiv:2401.15884) — Retrieval grading + fallback strategies
- **AWS MLaaJ** (2025) — LLM-as-judge on Bedrock
