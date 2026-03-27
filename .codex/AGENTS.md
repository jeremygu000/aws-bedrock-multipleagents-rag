# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-19
**Commit:** 9de5f86
**Branch:** master

## OVERVIEW

AWS Bedrock multi-agent system for APRA AMCOS: supervisor router dispatches to work-search and APRA Q&A specialist agents. pnpm monorepo with TypeScript CDK infra, Node Lambda tools, and Python hybrid RAG service.

## STRUCTURE

```
.
├── apps/rag-service/         # Python hybrid RAG runtime (FastAPI + Lambda)
├── packages/
│   ├── infra-cdk/            # CDK stacks: BedrockAgents, Neo4j, Monitoring
│   ├── shared/               # Bedrock action types, observability, OpenAPI schemas
│   ├── tool-supervisor/      # Gateway Lambda: intent detect, memory, rerank
│   ├── tool-work-search/     # Work search Lambda (MCP stub)
│   └── tool-rag-search/      # Legacy Node RAG handler (replaced by apps/rag-service)
├── scripts/                  # Test clients, eval runners, instance lifecycle
└── bedrock-eval/             # Python venv for RAGAS evaluation (gitignored)
```

## WHERE TO LOOK

| Task                          | Location                                         | Notes                                                         |
| ----------------------------- | ------------------------------------------------ | ------------------------------------------------------------- |
| Add/modify Bedrock agents     | `packages/infra-cdk/lib/bedrock-agents-stack.ts` | Agents, guardrails, Lambdas, IAM, OpenSearch                  |
| Add/modify Neo4j infra        | `packages/infra-cdk/lib/neo4j-data-stack.ts`     | EC2 + EBS + Docker                                            |
| Add/modify Grafana monitoring | `packages/infra-cdk/lib/monitoring-ec2-stack.ts` | EC2 + CloudWatch data source                                  |
| CDK stack registration        | `packages/infra-cdk/bin/app.ts`                  | Context params, stack instantiation                           |
| Shared types/helpers          | `packages/shared/src/`                           | `bedrockActionTypes.ts`, `observability.ts`, `intentTypes.ts` |
| OpenAPI schemas               | `packages/shared/src/openapi/`                   | `work-search.yaml`, `rag-search.yaml`, `supervisor.yaml`      |
| Gateway request flow          | `packages/tool-supervisor/src/handler.ts`        | Intent > memory > clarify/route > rerank                      |
| Work search Lambda            | `packages/tool-work-search/src/handler.ts`       | Calls `mcpClient.ts` (STUB)                                   |
| RAG retrieval + answer        | `apps/rag-service/app/workflow.py`               | LangGraph: intent > rewrite > retrieve > answer               |
| RAG Lambda entry              | `apps/rag-service/lambda_tool.py`                | Bedrock action adapter                                        |
| Test an agent                 | `scripts/test-agent.ts`                          | CLI: `pnpm test:agent -- --agent supervisor --prompt "..."`   |
| Test gateway (multi-turn)     | `scripts/test-gateway.ts`                        | CLI: `pnpm test:gateway -- --prompt "..."`                    |
| Batch eval                    | `scripts/eval-agent.ts`                          | Produces JSONL for RAGAS or rule eval                         |
| RAGAS scoring                 | `scripts/ragas_eval.py`                          | `pnpm eval:ragas`                                             |
| Work-search rule eval         | `scripts/eval-work-search.ts`                    | `pnpm eval:work-search`                                       |
| EC2 instance power            | `scripts/power-instances.sh`                     | `pnpm instance:status` / `stop:neo4j` / etc.                  |

## CONVENTIONS

### TypeScript

- `strict: true`, `target: ES2022`, `moduleResolution: Bundler`
- `tsgo` for type checking (`pnpm typecheck`), `tsc` fallback (`pnpm typecheck:tsc`)
- Lambdas bundled via `esbuild` to single CJS file in `dist/`
- Inline type imports enforced: `import { type Foo }` not `import type { Foo }`
- Double quotes, trailing commas, `printWidth: 100`, semicolons (Prettier)
- Workspace path alias: `@aws-bedrock-multiagents/shared` resolves to `packages/shared/src`

### Python (apps/rag-service)

- Python 3.12+, managed by `uv`
- `black` (line-length 100) + `ruff` (select E/F/I)
- `pytest` in `apps/rag-service/tests/`
- Pydantic settings for config (`app/config.py`)

### Linting (oxlint)

- `typescript/no-explicit-any: error` — never use `any`
- `typescript/consistent-type-imports: error` — inline type imports
- `import/no-cycle: error`
- `node/no-process-env: error` — only CDK env vars allowed; Lambdas use overrides
- `unicorn/no-array-for-each: error` — use `for...of` not `.forEach()`

### Git Hooks (husky)

- `pre-commit`: lint-staged (oxlint --fix + prettier on staged files)
- `pre-push`: `pnpm check` (typecheck + lint + rag:lint + rag:test)

## ANTI-PATTERNS (THIS PROJECT)

- **No `any`**: `typescript/no-explicit-any` is `error`. Use proper types.
- **No `process.env` in lib code**: Guarded by `node/no-process-env`. Only allowed in Lambda handlers and scripts via oxlint overrides.
- **No `.forEach()`**: Use `for...of` loops (unicorn/no-array-for-each).
- **No circular imports**: `import/no-cycle: error`.
- **No `==`**: Use `===` (eqeqeq: error).
- **No `var`**: Use `const`/`let`.
- **No dotenv auto-loading**: AWS credentials via `AWS_PROFILE` or env vars.
- **Never suppress types**: No `as any`, `@ts-ignore`, `@ts-expect-error`.

## STUBS NEEDING REAL IMPLEMENTATION

| File                                         | Current State                  | What's Needed                           |
| -------------------------------------------- | ------------------------------ | --------------------------------------- |
| `packages/tool-work-search/src/mcpClient.ts` | Returns hardcoded `WINF123456` | Real MCP transport call                 |
| `packages/tool-rag-search/src/ragClient.ts`  | Returns mock answer            | Legacy — replaced by `apps/rag-service` |

## COMMANDS

```bash
# Dev workflow
pnpm install && pnpm typecheck && pnpm lint && pnpm build && pnpm synth

# Build
pnpm build                    # All packages
pnpm build:lambdas            # Just Lambda bundles
pnpm build:infra              # Just CDK output

# Quality
pnpm typecheck                # tsgo
pnpm lint && pnpm lint:fix    # oxlint
pnpm format && pnpm format:check  # prettier
pnpm check                    # typecheck + lint + rag:lint + rag:test

# Python RAG
pnpm rag:install              # uv sync
pnpm rag:dev                  # uvicorn --reload on :8080
pnpm rag:test                 # pytest
pnpm rag:lint && pnpm rag:format  # ruff + black

# Deploy
pnpm deploy:app               # BedrockAgentsStack
pnpm deploy:data              # Neo4jDataStack
pnpm deploy:monitoring        # MonitoringEc2Stack
pnpm deploy:all               # All three stacks
pnpm destroy:app / destroy:data / destroy:monitoring / destroy:all

# Test & Eval
pnpm test:agent -- --agent supervisor --prompt "who is APRA AMCOS"
pnpm test:gateway -- --session-id "s1" --prompt "find Hello by Adele"
pnpm eval:agent -- --agent supervisor --input scripts/examples/agent-eval.example.jsonl --output tmp/evals/out.jsonl
pnpm eval:ragas -- --input tmp/evals/out-ragas.jsonl --output tmp/evals/results.json
pnpm eval:work-search -- --input tmp/evals/out.jsonl --output tmp/evals/rules.json

# EC2 instances
pnpm instance:status
pnpm instance:stop:all / instance:start:all
```

## ARCHITECTURE NOTES

- **Gateway pattern**: `tool-supervisor` Lambda is NOT a Bedrock agent — it's a standalone gateway that orchestrates intent detection, DynamoDB conversation memory, query rewriting, Bedrock supervisor agent invocation, and result reranking.
- **Supervisor router**: The Bedrock `supervisor-agent` uses `SUPERVISOR_ROUTER` collaboration mode with two collaborators: `WorkSearchAgent` and `ApraQaAgent`.
- **Model routing (RAG)**: Python workflow routes to `qwen-plus` for complex/low-confidence queries, defaults to `nova-lite`.
- **All LLM calls use `amazon.nova-lite-v1:0`** for intent detection, clarification, query rewriting, and memory summarization.
- **Reranking uses `amazon.rerank-v1:0`** in `us-west-2` (cross-region).
- **Collaborator wiring** uses CloudFormation custom resources (`AwsCustomResource`) since L2 constructs don't support multi-agent collaboration yet.
- **Lambda packaging**: Node Lambdas are esbuild CJS bundles; Python Lambda uses `PythonFunction` (alpha) with `uv` dependency management.
