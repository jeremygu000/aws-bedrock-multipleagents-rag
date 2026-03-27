# packages/tool-supervisor

Gateway Lambda orchestrating the full request lifecycle. NOT a Bedrock agent — it wraps the supervisor agent with memory, intent detection, and reranking.

## REQUEST FLOW

```
Event (prompt, sessionId)
  → detectIntent (Nova Lite)
  → getSessionMemory (DynamoDB)
  → if AMBIGUOUS: generateClarification → return early
  → rewriteQuery (Nova Lite, intent-specific prompts)
  → invokeBedrockAgent (supervisor, with memory context injected)
  → appendToMemory (DynamoDB, rolling summary on overflow)
  → rerankResults (Bedrock Rerank API, us-west-2)
  → return GatewayResponse
```

## WHERE TO LOOK

| Task                          | File                                                           | Notes                                        |
| ----------------------------- | -------------------------------------------------------------- | -------------------------------------------- |
| Modify gateway flow           | `src/handler.ts`                                               | 6-step orchestration                         |
| Change intent categories      | `src/intentDetector.ts` + `packages/shared/src/intentTypes.ts` | `IntentType` enum                            |
| Adjust memory window          | `src/memoryManager.ts:28`                                      | `MAX_RECENT_MESSAGES = 10`, evicts to half   |
| Change rerank model/region    | `src/reranker.ts:15-17`                                        | Env vars `RERANK_MODEL_ARN`, `RERANK_REGION` |
| Modify clarification tone     | `src/clarifier.ts`                                             | System prompt for ambiguous intent           |
| Change query rewrite behavior | `src/queryRewriter.ts`                                         | Intent-specific system prompts               |

## CONVENTIONS

- Every function takes `tracer: Tracer` as first arg for X-Ray instrumentation.
- All async operations wrapped in `captureAsync(tracer, segmentName, fn)`.
- Structured logging via `logger.appendKeys()` per request.
- Metrics emitted via `metrics.addMetric()` then `metrics.publishStoredMetrics()`.
- Env vars: `SUPERVISOR_AGENT_ID`, `SUPERVISOR_ALIAS_ID`, `MEMORY_TABLE_NAME`, `RERANK_MODEL_ARN`, `RERANK_REGION`.

## ANTI-PATTERNS

- **Never call Bedrock agent directly** without injecting memory context — the gateway composes `[System: Context Summary] + [Recent Conversation] + [Rewritten Query]`.
- **Never skip `appendToMemory`** — even for clarification responses (memory must track the full conversation).
- **`process.env` access is allowed here** (oxlint override in `.oxlintrc.json`), but only for Lambda-injected env vars.
