# Next.js Multi Zones Deployment to AWS Lambda with CDK

## Overview

This guide explains how to deploy 3 Next.js 15 apps (web, chat, upload) as separate Lambda functions behind a unified CloudFront distribution using the **cdk-opennext** construct.

**Architecture**:

```
┌──────────────────────────────────────────┐
│        CloudFront Distribution            │
│   (Multi Zones Router)                   │
└────────────────┬─────────────────────────┘
                 │
      ┌──────────┼──────────┐
      │          │          │
      ▼          ▼          ▼
   Web Zone   Chat Zone  Upload Zone
  (Lambda)    (Lambda)   (Lambda)
  OpenNext    OpenNext   OpenNext
```

**Key Points**:

- Each zone is an independent Next.js 15 app with App Router
- Each zone builds separately: `pnpm build` + `npx @opennextjs/aws build`
- CloudFront routes by path: `/`, `/chat/*`, `/upload/*`
- Soft navigation within zone, hard navigation between zones

---

## Prerequisites

### 1. Next.js 15 + App Router

Ensure each app has the latest Next.js 15:

```json
{
  "dependencies": {
    "next": "^15.0.0",
    "react": "^18.3.0 || ^19.0.0",
    "react-dom": "^18.3.0 || ^19.0.0"
  }
}
```

### 2. OpenNext Build Adapter

Each app must have OpenNext build output. Install:

```bash
cd apps/web
pnpm add -D @opennextjs/aws
```

### 3. AWS Credentials

```bash
export AWS_PROFILE=default
export AWS_DEFAULT_REGION=ap-southeast-2
```

### 4. CDK & cdk-opennext

Already added to `packages/infra-cdk/package.json`:

```bash
cd packages/infra-cdk
pnpm install
```

---

## Step 1: Build Each Next.js App

Build all 3 zones in sequence:

```bash
# Web Zone
cd apps/web
pnpm build

# Chat Zone
cd ../../apps/chat
pnpm build

# Upload Zone
cd ../../apps/upload
pnpm build
```

Each build generates a `.next/` directory with optimized artifacts.

---

## Step 2: Generate OpenNext Build Output

For each zone, generate the `.open-next/` directory:

```bash
# Web Zone
cd apps/web
npx @opennextjs/aws@3.10.1 build

# Chat Zone
cd ../../apps/chat
npx @opennextjs/aws@3.10.1 build

# Upload Zone
cd ../../apps/upload
npx @opennextjs/aws@3.10.1 build
```

This creates the `.open-next/` directory with:

- `open-next.config.json` — OpenNext configuration
- Lambda handler bundles for SSR + Image Optimization
- S3 asset paths for static files
- ISR revalidation metadata

**Expected structure**:

```
apps/web/.open-next/
├── config.json
├── server-function/   # Lambda handler
├── image-optimization-function/  # Image optimizer Lambda
└── S3/
    └── static/   # _next/static/ assets
```

---

## Step 3: Verify Lambda dist Exists

Before CDK synth, ensure all TypeScript builds complete:

```bash
cd packages/infra-cdk
pnpm build
```

This compiles `bin/app.ts` and `lib/frontend-stack.ts` to `dist/` with `.js` extensions.

---

## Step 4: Synthesize CDK

Generate the CloudFormation template:

```bash
cd packages/infra-cdk
pnpm synth
```

Output is stored in `cdk.out/` with JSON CloudFormation templates.

**Verify no errors** before proceeding.

---

## Step 5: Bootstrap AWS Account (First Time Only)

If this is your first CDK deployment to this account/region:

```bash
cd packages/infra-cdk
pnpm bootstrap
```

This creates S3 bucket + IAM role for CDK artifact staging.

---

## Step 6: Deploy FrontendStack Only

Deploy just the frontend Multi Zones (without Bedrock/Neo4j):

```bash
cd packages/infra-cdk
pnpm deploy -- FrontendStack
```

Or deploy with explicit stack name:

```bash
npx cdk deploy FrontendStack --progress events
```

**Expected outputs**:

- CloudFrontDomainName: `d1234567890.cloudfront.net`
- WebZoneDomain: `https://d1234567890.cloudfront.net/`
- ChatZoneDomain: `https://d1234567890.cloudfront.net/chat`
- UploadZoneDomain: `https://d1234567890.cloudfront.net/upload`

---

## Step 7: Test Multi Zones Routing

Once deployed, test routing to each zone:

### Web Zone (Default)

```bash
curl -L https://d1234567890.cloudfront.net/
# Should return HTML from apps/web
```

### Chat Zone

```bash
curl -L https://d1234567890.cloudfront.net/chat
# Should return HTML from apps/chat
```

### Upload Zone

```bash
curl -L https://d1234567890.cloudfront.net/upload
# Should return HTML from apps/upload
```

### Browser Testing

Open in browser:

- Web: `https://d1234567890.cloudfront.net/`
- Chat: `https://d1234567890.cloudfront.net/chat`
- Upload: `https://d1234567890.cloudfront.net/upload`

**Expected behavior**:

- Clicking links within the same zone → soft navigation (SPA-like)
- Navigating from web → chat zone → hard page reload

---

## Deployment Checklist

Before production, verify:

- [ ] Each app has `next.config.ts` (or `.js`)
- [ ] All apps built successfully: `pnpm build`
- [ ] All `.open-next/` directories exist and contain `config.json`
- [ ] CDK TypeScript compiles: `pnpm build` in `infra-cdk`
- [ ] Synth completes without errors: `pnpm synth`
- [ ] AWS credentials are set: `aws sts get-caller-identity`
- [ ] Bootstrap completed (first time only)
- [ ] CloudFormation stack deployed successfully
- [ ] CloudFront health checks pass
- [ ] All 3 zones responding to requests
- [ ] Soft navigation works within zones
- [ ] Hard navigation works between zones

---

## Troubleshooting

### CloudFormation Deployment Hangs

**Symptom**: `cdk deploy` stops at "Creating FrontendStack"

**Solution**:

1. Check AWS CloudFormation console for errors
2. Check CloudWatch logs for Lambda initialization errors
3. Ensure `.open-next/` directories exist with `config.json`

### 404 on Zone Paths

**Symptom**: `/chat` returns 404

**Root cause**: CloudFront routing rule order matters.

**Solution**: Verify `frontend-stack.ts` has rules in correct order:

1. `/chat/*` (specific)
2. `/upload/*` (specific)
3. `/` (default)

### Static Assets Not Loading

**Symptom**: CSS/JS from `_next/static/` return 404

**Root cause**: Each zone's OpenNext distribution has its own S3 origin. CloudFront parent distribution must forward `_next/*` to child distributions.

**Solution**: Ensure `frontend-stack.ts` does NOT cache static paths at parent level. Each zone's CloudFront handles caching.

### Lambda Cold Start > 30s

**Symptom**: Requests timeout with "Task timed out after 30 seconds"

**Solution**:

1. Increase Lambda timeout: `timeout: Duration.seconds(60)`
2. Pre-warm Lambda: `warm: 1, warmerInterval: Duration.minutes(5)`
3. Check Lambda initialization logs: CloudWatch → Lambda function logs

### Region Not Supported

**Symptom**: "Model not available in region" (Nova model)

**Solution**: `cdk-opennext` works in any region. If you use Bedrock models in Lambda env vars, ensure region supports them (e.g., ap-southeast-2 supports Nova).

---

## Customization

### Add Custom Domain

```typescript
// frontend-stack.ts
customDomain: {
  domainName: "app.example.com",
  hostedZone: HostedZone.fromLookup(this, "Zone", {
    domainName: "example.com",
  }),
},
```

### Adjust Lambda Memory/Timeout

```typescript
defaultFunctionProps: {
  memorySize: 3008,  // Max for production
  timeout: Duration.seconds(60),  // For slow pages
},
```

### Disable Warming (Save Cost)

```typescript
warm: 0,  // No warming
```

### Add More Zones

```typescript
const designSite = new NextjsSite(this, "DesignZone", {
  openNextPath: path.join(__dirname, "../../apps/design/.open-next"),
  ...
});

// In Distribution behaviors:
"/design/*": {
  origin: new HttpOrigin(designSite.distribution.domainName, ...),
  ...
},
```

---

## Monitoring & Logging

### CloudWatch Logs

Each Lambda function logs to CloudWatch:

```bash
# Web Zone logs
aws logs tail /aws/lambda/FrontendStack-WebZone-* --follow

# Chat Zone logs
aws logs tail /aws/lambda/FrontendStack-ChatZone-* --follow

# Upload Zone logs
aws logs tail /aws/lambda/FrontendStack-UploadZone-* --follow
```

### CloudFront Metrics

View in AWS Console:

1. CloudFront → Distributions
2. Click distribution
3. Monitor tab → View CloudWatch metrics

**Key metrics**:

- Cache hit ratio (should be >80% for static assets)
- Origin latency (should be <500ms)
- Error rate (should be <1%)

---

## Cost Optimization

### Monthly Estimate (1M requests, 10GB CloudFront data out)

| Component                           | Cost       |
| ----------------------------------- | ---------- |
| CloudFront data transfer            | $0.09      |
| Lambda invocations (3x zones)       | $0.40      |
| Lambda compute (2048 MB, 500ms avg) | $8.33      |
| **Total**                           | **~$8.82** |

### Cost-Saving Tips

1. **Reduce Lambda memory**: 1024 MB saves ~50% compute cost (if your app allows)
2. **Disable warming**: Remove `warm: 1` to disable pre-warming (~5% Lambda cost)
3. **Shared CloudFront**: Single distribution for all zones (done by default)
4. **S3 lifecycle**: Archive old static assets to Glacier

---

## Production Hardening

### Security

- [ ] Enable CloudFront logging for audit
- [ ] Restrict Lambda environment variables (no secrets in plaintext)
- [ ] Use AWS Secrets Manager for sensitive data
- [ ] Enable WAF on CloudFront (optional, additional cost)

### Performance

- [ ] Enable CloudFront compression: `compress: true` (default)
- [ ] Configure cache headers in `next.config.ts`
- [ ] Monitor Lambda cold start latency
- [ ] Set appropriate cache policies per endpoint

### Reliability

- [ ] Monitor CloudWatch alarms for errors
- [ ] Set up SNS notifications for deployment failures
- [ ] Test failover/recovery procedures
- [ ] Regular backup of OpenNext build outputs

---

## Deployment Scripts (pnpm)

Add to root `package.json`:

```json
{
  "scripts": {
    "build:frontend": "pnpm --filter=web --filter=chat --filter=upload build && pnpm --filter=web --filter=chat --filter=upload exec -- npx @opennextjs/aws build",
    "deploy:frontend": "pnpm build:frontend && cd packages/infra-cdk && pnpm build && pnpm deploy -- FrontendStack",
    "diff:frontend": "cd packages/infra-cdk && pnpm diff -- FrontendStack"
  }
}
```

Then deploy with:

```bash
pnpm deploy:frontend
```

---

## References

- [cdk-opennext GitHub](https://github.com/berenddeboer/cdk-opennext)
- [OpenNext AWS Documentation](https://opennext.js.org/aws)
- [Next.js Multi Zones Guide](https://nextjs.org/docs/advanced-features/multi-zones)
- [AWS CDK Documentation](https://docs.aws.amazon.com/cdk/)
- [CloudFront Behaviors](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/distribution-web-values-specify.html#DownloadDistValuesPathPattern)
