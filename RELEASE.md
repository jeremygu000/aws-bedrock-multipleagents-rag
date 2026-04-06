# Release Guide

This guide covers the release process for aws-bedrock-multiagents, including versioning strategy, deployment flow, environment promotion, and rollback procedures.

## Versioning Strategy

This is a private monorepo deployed directly via AWS CDK. We do not publish to npm or use semantic-release. Versioning follows CDK stack deployments.

### Version Format

```
CDK_STACK_VERSION = {MAJOR}.{MINOR}.{PATCH}
```

- **MAJOR**: Bedrock model version changes or infrastructure breaking changes
- **MINOR**: New agent capabilities, Lambda logic updates
- **PATCH**: Bug fixes, configuration updates, non-breaking changes

Version is tracked in:

- `packages/infra-cdk/lib/stacks.ts` - CDK stack version tag
- Deployment tags in CloudFormation
- Bedrock agent configuration

### Example Versions

- `1.0.0` - Initial multi-agent supervisor + QA/WorkSearch specialist agents
- `1.1.0` - Add Neo4j integration for enhanced RAG
- `1.1.1` - Fix Lambda timeout in RAG retrieval
- `2.0.0` - Upgrade Bedrock model to Claude Sonnet 4.5

## Pre-Release Checklist

Before deploying to any environment, verify the following:

### Code Quality Gate

Run the full check suite:

```bash
pnpm check
```

This includes:

- `pnpm typecheck` - TypeScript strict mode check
- `pnpm lint` - oxlint on JavaScript/TypeScript
- `pnpm rag:lint` - ruff on Python code
- `pnpm rag:test` - pytest on RAG service

Ensure all pass with no warnings.

### Build Verification

Build all packages:

```bash
pnpm build
```

Verify no build errors and all dist directories are created.

### Evaluation Gate (RAG/Agent Changes)

If modifying RAG service or agent logic, run evaluation:

```bash
pnpm eval:ci
```

This runs RAGAS evaluation on the test dataset with pre-defined thresholds:

- `answer_relevancy >= 0.7`
- `faithfulness >= 0.8`
- `factual_correctness >= 0.75`

If metrics are below threshold, address the issues before proceeding.

### Git Status

Ensure your working directory is clean:

```bash
git status
# Should show: "On branch main, nothing to commit, working tree clean"
```

Commit all changes and ensure branch is up-to-date with remote:

```bash
git log -1  # Verify latest commit
git push    # Ensure remote is updated
```

## Deployment Flow

Deployment follows this sequence: build → typecheck → synth → diff → deploy.

### 1. Build All Packages

```bash
pnpm build
```

This compiles TypeScript, bundles Lambda handlers, and prepares artifacts for CDK.

### 2. Typecheck

Verify no type errors:

```bash
pnpm typecheck
```

### 3. Generate CloudFormation Template

Synthesize CDK stacks to CloudFormation:

```bash
pnpm synth
```

Output templates to `cdk.out/` directory.

### 4. Review Changes (Diff)

Before deploying, always review what will change:

```bash
pnpm cdk diff
```

This shows:

- New resources to create
- Resource properties changing
- Deletions (if any)

Review carefully. Accidental resource deletion in production is dangerous.

### 5. Deploy Stacks

Deploy to the target environment using CDK. Stacks are deployed in dependency order.

#### Deploy All Stacks (Recommended for Full Release)

```bash
pnpm deploy:all
```

Deploys all three stacks in sequence:

1. `Neo4jDataStack` - Vector database and knowledge graph initialization
2. `BedrockAgentsStack` - Bedrock agents, Lambda handlers, API Gateway
3. `MonitoringEc2Stack` - CloudWatch dashboards, EC2 monitoring

#### Deploy Individual Stacks (For Targeted Fixes)

Deploy data infrastructure:

```bash
pnpm deploy:data
```

Deploy Bedrock agents and Lambda functions:

```bash
pnpm deploy:app
```

Deploy monitoring and observability:

```bash
pnpm deploy:monitoring
```

### 6. Post-Deployment Verification

After deployment completes, verify:

```bash
# Check stack outputs
pnpm cdk list

# Test agent gateway
pnpm test:gateway

# Test agent evaluation
pnpm test:agent

# Check CloudWatch logs for errors
aws logs tail /aws/lambda/bedrock-agent-gateway --follow --region ap-southeast-2
```

## Environment Promotion

Deployments target ap-southeast-2 (Sydney) by default. For multi-environment setups, follow this promotion path:

### Dev Environment

```bash
export CDK_ENV=dev
pnpm deploy:all
```

- Used for daily development and testing
- Can be frequently recreated
- No production data

### Staging Environment

```bash
export CDK_ENV=staging
pnpm deploy:all
```

- Mirrors production configuration
- Tests release candidates
- Uses production-like data volumes
- Evaluate metrics before prod promotion

Run full evaluation suite on staging:

```bash
pnpm eval:ci
pnpm test:agent
pnpm test:gateway
```

### Production Environment

```bash
export CDK_ENV=prod
pnpm deploy:all
```

- Live user-facing system
- Higher availability requirements
- Larger data volumes (Neo4j knowledge graph)
- Requires approval before deployment

Deployment to production requires:

1. Code review approval on PR
2. All checks passing (typecheck, lint, test)
3. Evaluation metrics above threshold
4. Manual approval from ops/lead engineer

## Rollback Procedure

If a deployment causes issues, rollback to the previous CloudFormation stack version.

### Identify Previous Stack Version

```bash
aws cloudformation describe-stacks \
  --stack-name BedrockAgentsStack \
  --region ap-southeast-2 \
  --query 'Stacks[0].[StackStatus,CreationTime,LastUpdatedTime]'
```

CloudFormation automatically keeps the previous version available.

### Rollback Using AWS Console

1. Go to AWS CloudFormation console
2. Select the problematic stack (e.g., `BedrockAgentsStack`)
3. Click "Stack actions" > "Continue update rollback"
4. Confirm rollback

### Rollback Using CLI

```bash
aws cloudformation cancel-update-stack \
  --stack-name BedrockAgentsStack \
  --region ap-southeast-2
```

If stack is not in update state:

```bash
aws cloudformation update-stack \
  --stack-name BedrockAgentsStack \
  --use-previous-template \
  --region ap-southeast-2
```

### Post-Rollback Steps

1. Check Lambda function logs to identify the issue
2. Create a fix on a feature branch
3. Test locally and on staging
4. Re-deploy to production

## CDK Commands Reference

### View Stack Information

```bash
pnpm cdk list              # List all stacks
pnpm cdk describe          # Describe CDK app structure
```

### Validate Templates

```bash
pnpm cdk synth             # Generate CloudFormation templates
pnpm synth --strict        # Strict validation
```

### Destroy (Development Only)

Remove stacks from AWS (data loss):

```bash
pnpm cdk destroy           # Interactive destruction prompt
pnpm cdk destroy --force   # Non-interactive (use with caution)
```

### Debug Deployment

```bash
pnpm cdk deploy --verbose  # Print detailed deployment logs
pnpm cdk diff --verbose    # Print detailed diff analysis
```

## Stack Details

### BedrockAgentsStack

Contains:

- Bedrock agents (supervisor, QA specialist, WorkSearch specialist)
- Lambda handlers for agent intent routing
- API Gateway for gateway Lambda
- IAM roles and policies
- Bedrock action groups for agent capabilities

Typical deployment time: 2-3 minutes

### Neo4jDataStack

Contains:

- Neo4j instance for knowledge graph storage
- RDS security groups and VPC configuration
- Initial data loader Lambda
- IAM roles for data access

Typical deployment time: 5-10 minutes (includes Neo4j initialization)

### MonitoringEc2Stack

Contains:

- CloudWatch dashboards for agent metrics
- EC2 instance for custom monitoring agents
- SNS topics for alerts
- CloudWatch alarms for critical metrics

Typical deployment time: 2-3 minutes

## Monitoring Post-Release

After deployment, monitor key metrics:

### CloudWatch Dashboards

```bash
# Access via AWS Console
# Dashboards > aws-bedrock-multiagents-{ENV}
```

Monitor:

- Agent invocation count
- Lambda duration and errors
- Neo4j query latency
- API Gateway response codes

### Lambda Logs

```bash
# Supervisor/router
aws logs tail /aws/lambda/bedrock-agent-gateway --follow --region ap-southeast-2

# QA agent
aws logs tail /aws/lambda/bedrock-agent-qa --follow --region ap-southeast-2

# WorkSearch agent
aws logs tail /aws/lambda/bedrock-agent-work-search --follow --region ap-southeast-2
```

### RAG Service Metrics

Check RAG service logs for retrieval issues:

```bash
# Retrieve answer relevancy scores
aws logs filter-log-events \
  --log-group-name /aws/lambda/rag-service \
  --filter-pattern "answer_relevancy" \
  --region ap-southeast-2
```

## Release Checklist

Before declaring a release complete:

- [x] All checks pass: `pnpm check`
- [x] Build succeeds: `pnpm build`
- [x] Evaluation gate passes: `pnpm eval:ci`
- [x] CloudFormation diff reviewed
- [x] Deploy command executed without errors
- [x] Post-deployment tests pass: `pnpm test:agent` `pnpm test:gateway`
- [x] CloudWatch dashboards show normal metrics
- [x] Lambda logs show no errors
- [x] Rollback plan documented (if needed)
- [x] Release notes updated
- [x] Team notified of deployment

## Release Notes Template

Create a release note when deploying:

```markdown
## Release v{VERSION} - {DATE}

### Changes

- **Feature**: Brief description
- **Fix**: Bug fix description
- **Chore**: Dependency update or infrastructure change

### Deployment

- Deployed to: {ENVIRONMENT}
- Stacks updated: {LIST}
- Deployment time: {TIME}

### Metrics

- Answer relevancy: {SCORE}
- Faithfulness: {SCORE}
- Agent latency (p99): {TIME}ms

### Known Issues

- None

### Rollback Command

If needed:
\`\`\`bash
aws cloudformation cancel-update-stack \
 --stack-name BedrockAgentsStack \
 --region ap-southeast-2
\`\`\`
```

## Questions and Support

For release-related questions:

- Review CloudFormation events in AWS Console for error details
- Check CloudWatch Logs for Lambda execution traces
- Run `pnpm cdk doctor` for CDK environment diagnostics
- Consult team wiki or runbooks for env-specific procedures
