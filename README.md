# aws-bedrock-multiagents

A pnpm workspace monorepo for an AWS Bedrock multi-agent system managed with CDK.

The repository currently includes:

- CDK infrastructure for Bedrock Agents, Guardrails, IAM, and Lambda wiring
- A `work_search` Lambda tool
- A `rag_search` Lambda tool
- A shared package for Bedrock action types, observability helpers, and OpenAPI schemas
- Workspace-wide type checking with `tsgo`
- Workspace-wide linting with `oxlint`
- Workspace-wide formatting with `prettier`

## Architecture

The current stack provisions three Bedrock agents:

- `work-search-agent`
- `apra-qa-agent`
- `supervisor-agent`

It also provisions:

- two Lambda functions used as Bedrock action group executors
- two Guardrails with different strictness levels
- IAM roles that allow Bedrock agents to invoke models and Lambdas
- CloudWatch Log Groups for the tool Lambdas

The supervisor agent is configured as a `SUPERVISOR_ROUTER` and routes requests to the specialist agents.

## Repository Layout

```text
.
├── package.json
├── pnpm-workspace.yaml
├── tsconfig.base.json
└── packages
    ├── infra-cdk
    │   ├── bin/app.ts
    │   └── lib/bedrock-agents-stack.ts
    ├── shared
    │   └── src
    │       ├── bedrockActionTypes.ts
    │       ├── observability.ts
    │       └── openapi
    ├── tool-rag-search
    │   └── src
    │       ├── handler.ts
    │       └── ragClient.ts
    └── tool-work-search
        └── src
            ├── handler.ts
            └── mcpClient.ts
```

## Packages

### `packages/infra-cdk`

CDK app and stack definitions.

Key file:

- `packages/infra-cdk/lib/bedrock-agents-stack.ts`

Responsibilities:

- define Bedrock Guardrails
- define Bedrock Agents and aliases
- define Lambda functions for action groups
- load shared OpenAPI schemas
- package Lambda `dist/` directories as CDK assets

### `packages/tool-work-search`

Lambda for the `work_search` action group.

Current status:

- wired with Powertools tracer/logger/metrics
- returns stubbed MCP-backed work search results
- needs a real MCP transport implementation in `src/mcpClient.ts`

### `packages/tool-rag-search`

Lambda for the `rag_search` action group.

Current status:

- wired with Powertools tracer/logger/metrics
- returns stubbed grounded answer output
- needs a real RAG pipeline in `src/ragClient.ts`

### `packages/shared`

Shared workspace package for:

- Bedrock action request/response types
- response helpers
- observability helpers
- OpenAPI schemas used by CDK

## Tooling

### Type Checking

Primary type checking uses `tsgo`.

```bash
pnpm typecheck
```

Fallback `tsc` type checking is also available:

```bash
pnpm typecheck:tsc
```

### Linting

Linting uses `oxlint` with rules tuned for this Node/CDK/Lambda codebase.

```bash
pnpm lint
pnpm lint:fix
```

### Formatting

Formatting uses `prettier`.

```bash
pnpm format:check
pnpm format
```

## Build and Deploy

Install dependencies:

```bash
pnpm install
```

Build all workspace packages:

```bash
pnpm build
```

Synthesize the CDK app:

```bash
pnpm synth
```

Diff the stack:

```bash
pnpm diff
```

Bootstrap the target AWS environment if needed:

```bash
pnpm bootstrap
```

Deploy:

```bash
pnpm cdk:deploy
```

Important: `packages/infra-cdk` deploys Lambda code from the built `dist/` folders in the tool packages. Run `pnpm build` before `pnpm synth`, `pnpm diff`, or `pnpm cdk:deploy`.

## Environment Variables

An example environment file is included at `.env.example`.

Important:

- this repository does not auto-load `.env` files with `dotenv`
- CDK and the AWS SDK read credentials and region from your shell environment or AWS config
- prefer `AWS_PROFILE` or AWS SSO over long-lived access keys

Recommended local setup:

```bash
export AWS_PROFILE=default
export AWS_DEFAULT_REGION=ap-southeast-2
```

If you are using static credentials locally instead:

```bash
export AWS_ACCESS_KEY_ID="replace-me"
export AWS_SECRET_ACCESS_KEY="replace-me"
export AWS_DEFAULT_REGION="ap-southeast-2"
```

If your credentials are temporary, also set:

```bash
export AWS_SESSION_TOKEN="replace-me"
```

### Variables That Matter Right Now

- `AWS_PROFILE`
  - preferred way to select your AWS identity locally
- `AWS_DEFAULT_REGION`
  - used by AWS tooling and should match your deployment region
- `AWS_ACCESS_KEY_ID`
  - optional if you use static credentials instead of a profile
- `AWS_SECRET_ACCESS_KEY`
  - optional if you use static credentials instead of a profile
- `AWS_SESSION_TOKEN`
  - required only for temporary credentials

### What CDK Uses

The CDK app sets the stack environment from:

- `CDK_DEFAULT_ACCOUNT`
- `CDK_DEFAULT_REGION`

These are normally resolved automatically by CDK from your active AWS credentials and region. You usually do not need to export `CDK_DEFAULT_ACCOUNT` yourself.

### Variables You May Add Later

These are useful project-level settings, but they are not all wired into the code yet:

- `FOUNDATION_MODEL_ID`
  - useful if you want to stop hardcoding the Bedrock model ID in the CDK stack
- `WORK_SEARCH_ENDPOINT`
  - likely needed when `packages/tool-work-search/src/mcpClient.ts` is replaced with a real MCP or HTTP call
- `RAG_BACKEND_ENDPOINT`
  - likely needed when `packages/tool-rag-search/src/ragClient.ts` is replaced with a real RAG backend

### Verify Your AWS Identity

Before running CDK commands, verify that the active identity and region are correct:

```bash
aws sts get-caller-identity
aws configure list
```

## Development Workflow

Recommended local workflow:

```bash
pnpm install
pnpm typecheck
pnpm lint
pnpm build
pnpm synth
```

If you are actively editing code:

```bash
pnpm format
pnpm lint:fix
pnpm typecheck
```

## Bedrock-Specific Notes

### Foundation Model

The CDK stack currently uses this placeholder model ID:

```text
amazon.nova-lite-v1:0
```

Update it in:

- `packages/infra-cdk/lib/bedrock-agents-stack.ts`

### Shared OpenAPI Schemas

The action group schemas are stored in:

- `packages/shared/src/openapi/work-search.yaml`
- `packages/shared/src/openapi/rag-search.yaml`

These are loaded by the CDK stack at synth/deploy time.

### Tool Lambda Packaging

Both tool Lambdas are bundled with `esbuild` and emitted to `dist/` as single-file CommonJS output for the Lambda runtime.

Useful build commands:

- `pnpm build`
  - build every workspace package
- `pnpm build:lambdas`
  - rebuild only the two tool Lambda bundles
- `pnpm build:infra`
  - rebuild only the CDK app output under `packages/infra-cdk/dist`

## Git Hooks

Git hooks are configured with `husky`.

After `pnpm install`, the `prepare` script enables the repository hooks automatically.

Hook behavior:

- `pre-commit`
  - runs `lint-staged`
  - formats staged files with `prettier`
  - runs `oxlint --fix` on staged JavaScript and TypeScript files
- `pre-push`
  - runs `pnpm check`
  - this executes `pnpm typecheck`, `pnpm lint`, and `pnpm format:check`

## What Still Needs Real Implementation

The repository is scaffolded and type-checked, but some parts are still placeholders:

- `packages/tool-work-search/src/mcpClient.ts`
  - replace the stub with a real MCP call
- `packages/tool-rag-search/src/ragClient.ts`
  - replace the stub with your real RAG workflow
- `packages/infra-cdk/lib/bedrock-agents-stack.ts`
  - verify Bedrock resource properties against your AWS account, region, and final model choices

## AWS Prerequisites

Before deployment, make sure:

- your AWS credentials are configured locally
- `AWS_PROFILE` or AWS credentials resolve to the intended AWS account
- `AWS_DEFAULT_REGION` matches your target deployment region

## Test Client

A TypeScript CLI test client is included at `scripts/test-agent.ts`.

Install dependencies first:

```bash
pnpm install
```

Invoke an agent by alias ARN:

```bash
pnpm test:agent -- \
  --alias-arn arn:aws:bedrock:ap-southeast-2:123456789012:agent-alias/AGENT_ID/ALIAS_ID \
  --prompt "who is APRA AMCOS"
```

Or use a named shortcut after exporting the matching alias ARN:

```bash
pnpm test:agent -- --agent supervisor --prompt "who is APRA AMCOS"
pnpm test:agent -- --agent qa --prompt "who is APRA AMCOS"
pnpm test:agent -- --agent work --prompt "find a work titled Hello by Adele"
```

Show a condensed trace summary in the terminal:

```bash
pnpm test:agent -- \
  --agent supervisor \
  --prompt "who is APRA AMCOS" \
  --trace-summary
```

The terminal summary is rendered as a colored overview plus timeline, with route conclusion, action-group results, final answer, step type, relative timing, collaborator hops, and selected details such as model usage and failures.

Enable trace output and save the full response as JSON:

```bash
pnpm test:agent -- \
  --alias-arn arn:aws:bedrock:ap-southeast-2:123456789012:agent-alias/AGENT_ID/ALIAS_ID \
  --prompt "who is APRA AMCOS" \
  --trace \
  --output tmp/apra-who-is.json \
  --json
```

You can also pass IDs directly:

```bash
pnpm test:agent -- \
  --agent-id SHWNEKC9MO \
  --alias-id 8VQRTZW7A6 \
  --prompt "who is APRA AMCOS"
```

Supported environment variables:

- `BEDROCK_SUPERVISOR_ALIAS_ARN`
- `BEDROCK_SUPERVISOR_AGENT_ID`
- `BEDROCK_SUPERVISOR_ALIAS_ID`
- `BEDROCK_QA_ALIAS_ARN`
- `BEDROCK_QA_AGENT_ID`
- `BEDROCK_QA_ALIAS_ID`
- `BEDROCK_WORK_ALIAS_ARN`
- `BEDROCK_WORK_AGENT_ID`
- `BEDROCK_WORK_ALIAS_ID`
- `BEDROCK_AGENT_ALIAS_ARN`
- `BEDROCK_AGENT_ID`
- `BEDROCK_AGENT_ALIAS_ID`
- `BEDROCK_AGENT_REGION`
- `BEDROCK_AGENT_SESSION_ID`
- `CDK_DEFAULT_ACCOUNT` and `CDK_DEFAULT_REGION` resolve correctly
- the target region supports the Bedrock resources and foundation model you plan to use
- your account has the required Bedrock and Lambda permissions

## Evaluator Runner

A batch evaluator runner is included at `scripts/eval-agent.ts`.

Example input dataset:

- `scripts/examples/agent-eval.example.jsonl`

Run it against a named agent:

```bash
pnpm eval:agent -- \
  --agent supervisor \
  --input scripts/examples/agent-eval.example.jsonl \
  --output tmp/evals/supervisor.jsonl
```

Useful options:

- `--trace-summary`
  - prints the condensed trace summary for each row while the run is in progress
- `--format json`
  - writes one JSON array instead of JSONL
- `--shared-session`
  - reuses one Bedrock session across all rows
- `--fail-fast`
  - aborts on the first failing example

Output records include:

- `prompt`
- `question`
- `answer`
- `reference`
- `ground_truth`
- `traceSummary`
- `traces`
- `metadata`

This shape is intended to be easy to transform into a later RAGAS evaluation dataset.

## Useful Files

- `package.json`
- `packages/infra-cdk/bin/app.ts`
- `packages/infra-cdk/lib/bedrock-agents-stack.ts`
- `packages/shared/src/bedrockActionTypes.ts`
- `packages/shared/src/observability.ts`
- `packages/tool-work-search/src/handler.ts`
- `packages/tool-rag-search/src/handler.ts`
