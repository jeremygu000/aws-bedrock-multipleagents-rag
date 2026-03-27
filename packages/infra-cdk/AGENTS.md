# packages/infra-cdk

CDK app defining three independent stacks: `BedrockAgentsStack`, `Neo4jDataStack`, `MonitoringEc2Stack`.

## STRUCTURE

```
infra-cdk/
├── bin/app.ts                    # CDK entrypoint — reads context, instantiates stacks
└── lib/
    ├── bedrock-agents-stack.ts   # Agents, guardrails, Lambdas, OpenSearch, DynamoDB, IAM
    ├── neo4j-data-stack.ts       # EC2 + EBS + Docker Neo4j + Secrets Manager
    └── monitoring-ec2-stack.ts   # EC2 + Grafana + CloudWatch data source
```

## WHERE TO LOOK

| Task                           | File                             | Notes                                                               |
| ------------------------------ | -------------------------------- | ------------------------------------------------------------------- |
| Add a new Bedrock agent        | `lib/bedrock-agents-stack.ts`    | Follow `QaAgent`/`WorkAgent` pattern                                |
| Add collaborator to supervisor | `lib/bedrock-agents-stack.ts`    | Use `AwsCustomResource` pattern (L2 doesn't support collaboration)  |
| Change foundation model        | `lib/bedrock-agents-stack.ts:18` | `FOUNDATION_MODEL_ID` constant                                      |
| Add Lambda env vars            | `lib/bedrock-agents-stack.ts`    | `ragSearchFnEnv` map or `gatewayFn.addEnvironment()`                |
| Override Neo4j sizing          | `bin/app.ts`                     | CDK context: `--context neo4jInstanceType=t3.large`                 |
| Override monitoring sizing     | `bin/app.ts`                     | CDK context: `--context monitoringInstanceType=t3.micro`            |
| Add new stack                  | `bin/app.ts`                     | Instantiate, then add deploy/destroy scripts to root `package.json` |

## CONVENTIONS

- Builds with `tsc` (not esbuild) — emits to `dist/` with `.js` extensions.
- `bin/app.ts` imports use `.js` extension for ESM compatibility: `from "../lib/bedrock-agents-stack.js"`.
- Context values are optional and parsed from string: always `tryGetContext()` → `Number()` or `String()`.
- Uses `@aws-cdk/aws-lambda-python-alpha` `PythonFunction` for Python Lambda packaging.
- Stack env comes from `CDK_DEFAULT_ACCOUNT` and `CDK_DEFAULT_REGION` (auto-resolved by CDK).

## ANTI-PATTERNS

- **Never hardcode AWS account/region** in stack code — always from `CDK_DEFAULT_*`.
- **Never use `process.env` directly in lib/** — pass through CDK context or stack props. Exception: `bedrock-agents-stack.ts` reads `process.env` for Lambda env var passthrough (uses import alias `processEnv`).
- **Lambda dist must exist before synth** — always `pnpm build` before `pnpm synth/diff/deploy`.
- **Collaborator wiring order matters**: supervisor alias depends on both collaborator custom resources.
