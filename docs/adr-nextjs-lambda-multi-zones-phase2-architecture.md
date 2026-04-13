# ADR: Next.js Lambda Multi Zones 架构设计 Phase 2

**Date**: 2026-04-13
**Status**: IN REVIEW
**Deciders**: Implementation Team

## 1. Architecture Overview

### 1.1 High-Level Stack Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│               AWS CloudFront (us-east-1)                         │
│       CNAME: yourdomain.com                                      │
│  Cache behaviors:                                               │
│  - /chat-static/* → S3 (1 year)                                 │
│  - /chat* → Lambda (chat app)                                   │
│  - /upload-static/* → S3 (1 year)                               │
│  - /upload* → Lambda (upload app)                               │
│  - /* → Lambda (web app)                                        │
└────────────────┬────────────────────────────────────────────────┘
                 │ HTTPS
        ┌────────┴─────────┐
        │                  │
    ┌───▼──────┐      ┌────▼─────────────────┐
    │    S3    │      │  Lambda Function URL │
    │(assets)  │      │  Origins (3x)        │
    └──────────┘      │                      │
                      │ - app-chat (port 3001)
                      │ - app-upload (port 3002)
                      │ - app-web (port 3000)
                      └──────────────────────┘
                              ▲
                              │
                      ┌───────┴──────────┐
                      │                  │
                  ┌───▼───────┐  ┌──────▼────┐
                  │ Lambda    │  │ Lambda    │
                  │ (Chat)    │  │ (Web)     │
                  │ 512MB     │  │ 1GB       │
                  │ 30s       │  │ 30s       │
                  └───────────┘  └───────────┘
                                       │
                              ┌────────┴──────────┐
                              │                   │
                          ┌───▼──┐          ┌────▼────┐
                          │Upload│          │ DynamoDB│
                          │Lambda│          │  (ISR)  │
                          │512MB │          └─────────┘
                          │30s   │
                          └──────┘
```

### 1.2 Resource Allocation

| Component              | v15.x (OpenNext)     | v16.2+ (cdk-nextjs)  | Notes                          |
| ---------------------- | -------------------- | -------------------- | ------------------------------ |
| CloudFront             | Managed by construct | Managed by construct | Shared across 3 zones          |
| S3 (assets)            | One bucket           | One bucket           | Prefixed by buildId + basePath |
| DynamoDB (ISR)         | Optional             | Optional             | Shared across 3 zones          |
| Lambda (Web)           | 1 function           | 1 function           | port 3000, 1GB, 30s            |
| Lambda (Chat)          | 1 function           | 1 function           | port 3001, 512MB, 30s          |
| Lambda (Upload)        | 1 function           | 1 function           | port 3002, 512MB, 30s          |
| **Total Monthly Cost** | ~$14.58              | ~$16-19              | 1M requests                    |

## 2. Detailed Architecture Decisions

### 2.1 Path-Based Routing Strategy

**Why CloudFront cache behaviors, not ALB?**

1. **CDK Construct Support**: Both cdk-nextjs and OpenNext + manual CDK support CloudFront behaviors
2. **Cost**: CloudFront cache behaviors are free; ALB would add ~$15/month
3. **Latency**: Same 50-100ms origin latency, but CDK constructs handle caching automatically
4. **Simplicity**: No custom middleware layer needed

**Cache Behavior Priority (Top-to-Bottom in CloudFront)**

```
1. Pattern: /chat-static/*
   Origin: S3 (assets-bucket)
   Cache: 31536000s (1 year)
   Compress: Yes

2. Pattern: /chat*
   Origin: Lambda-Chat-URL
   Cache: 0s (no-cache)
   Forward: All headers + cookies

3. Pattern: /upload-static/*
   Origin: S3 (assets-bucket)
   Cache: 31536000s (1 year)
   Compress: Yes

4. Pattern: /upload*
   Origin: Lambda-Upload-URL
   Cache: 0s (no-cache)
   Forward: All headers + cookies

5. Pattern: _next/static/*
   Origin: S3 (assets-bucket)
   Cache: 31536000s (1 year)
   Compress: Yes

6. Pattern: _next/*
   Origin: S3 (assets-bucket)
   Cache: 3600s (1 hour)
   Query string: Yes (for versioning)
   Compress: Yes

7. Pattern: /*
   Origin: Lambda-Web-URL
   Cache: 0s (no-cache)
   Forward: All headers + cookies
```

### 2.2 Resource Isolation Strategy

Each zone's assets must be **completely isolated** to prevent CSS/JS conflicts.

**S3 Asset Path Prefix (buildId-based)**

```
s3://shared-assets-bucket/
├── chat-20260413-abc123/
│   ├── _next/static/css/...
│   ├── _next/static/js/...
│   └── public/...
├── upload-20260413-def456/
│   ├── _next/static/css/...
│   ├── _next/static/js/...
│   └── public/...
└── web-20260413-ghi789/
    ├── _next/static/css/...
    ├── _next/static/js/...
    └── public/...
```

**Implementation**

- `cdk-nextjs` uses `props.buildId` automatically
- OpenNext + manual CDK: set `assetPrefix` in each app's `next.config.ts`

### 2.3 Next.js Configuration Per App

**apps/chat/next.config.ts** (SHARED ZONE)

```typescript
const nextConfig: NextConfig = {
  basePath: "/chat",
  assetPrefix: "/chat",
  output: "standalone",
};
export default nextConfig;
```

**apps/upload/next.config.ts** (SHARED ZONE)

```typescript
const nextConfig: NextConfig = {
  basePath: "/upload",
  assetPrefix: "/upload",
  output: "standalone",
};
export default nextConfig;
```

**apps/web/next.config.ts** (GATEWAY ZONE with Multi Zones rewrites)

```typescript
const nextConfig: NextConfig = {
  basePath: "/", // Root
  output: "standalone",

  // Multi Zone rewrites (for local dev)
  async rewrites() {
    return [
      {
        source: "/chat/:path*",
        destination: `${process.env.CHAT_URL ?? "http://localhost:3001"}/chat/:path*`,
      },
      {
        source: "/upload/:path*",
        destination: `${process.env.UPLOAD_URL ?? "http://localhost:3002"}/upload/:path*`,
      },
    ];
  },

  // Server Actions CORS (Critical for Multi Zone)
  experimental: {
    serverActions: {
      allowedOrigins: [
        "yourdomain.com",
        "www.yourdomain.com",
        "*.yourdomain.com",
        process.env.INTERNAL_ALB_URL,
      ].filter(Boolean),
    },
  },
};
export default nextConfig;
```

### 2.4 Environment Configuration

**Production (.env.production)**

```bash
# For Multi Zone local rewrites (Web app only)
CHAT_URL=http://localhost:3001
UPLOAD_URL=http://localhost:3002

# CloudFront domain (web app knows its own domain)
PUBLIC_DOMAIN=https://yourdomain.com
```

**Note**: Lambda environment doesn't use these; they're only for local dev multi-zone rewrites.

### 2.5 Lambda Configuration

**All 3 Lambdas**

- **Runtime**: Node.js 20 LTS (via Docker container)
- **Memory**: Web=1GB, Chat=512MB, Upload=512MB
- **Timeout**: 30s (sufficient for Next.js 15/16)
- **Ephemeral Storage**: Default 512MB
- **Reserved Concurrent Executions**: 100 (prevent runaway)

**Container Image**

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY .next/standalone .
COPY .next/static .next/static
EXPOSE 3000
CMD ["node", "server.js"]
```

## 3. Two Implementation Paths

### 3.1 Path 1: Next.js v16.2+ with cdk-nextjs (RECOMMENDED)

**Prerequisites**

- Upgrade each app: `pnpm upgrade next@^16.2.0` (or use v16.3+)
- Verify all 3 apps build: `pnpm build`

**CDK Stack Structure**

```
packages/infra-cdk/lib/
├── nextjs-multi-zones-stack.ts          # Main orchestrator
├── shared-nextjs-resources-stack.ts     # CloudFront + S3 + DynamoDB
├── nextjs-zone-stack.ts                 # App-specific stack (reusable)
└── nextjs-multi-zones-app.ts            # CDK App (main entry)
```

**Key Code Pattern**

```typescript
// shared-nextjs-resources-stack.ts
new cdk.aws_s3.Bucket(this, "AssetsBucket", {
  bucketName: `shared-assets-${aws.accountId}`,
  autoDeleteObjects: true,
});

new cdk.aws_dynamodb.Table(this, "IsrTable", {
  partitionKey: { name: "revalidateKey", type: dynamodb.AttributeType.STRING },
  billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
});

// nextjs-zone-stack.ts
new NextjsSite(this, "ChatApp", {
  openNextPath: path.join(__dirname, "../../apps/chat/.open-next"),
  sharedAssetsBucket: sharedBucket,
  sharedIsrTable: isrTable,
});
```

**Estimated Effort**: 8-10 hours
**Go Live**: 3-4 weeks (including v16 testing)

### 3.2 Path 2: Next.js v15.x with OpenNext + Manual CDK (FALLBACK)

**Prerequisites**

- No version upgrade needed
- Install `@opennextjs/aws`: `pnpm add -D @opennextjs/aws esbuild`

**Build Process**

```bash
# Per app
cd apps/chat && npx @opennextjs/aws@latest build
cd apps/upload && npx @opennextjs/aws@latest build
cd apps/web && npx @opennextjs/aws@latest build

# Creates .open-next/ directories with Lambda zips
```

**CDK Stack Structure**

```
packages/infra-cdk/lib/
├── nextjs-lambda-shared-stack.ts   # S3 + DynamoDB + CloudFront
├── nextjs-lambda-zone-stack.ts     # Lambda per app (manual)
└── nextjs-lambda-multi-zones-app.ts
```

**Key Code Pattern**

```typescript
// Manual Lambda creation (OpenNext requires CDK orchestration)
const chatLambda = new lambda.Function(this, "ChatFunction", {
  runtime: lambda.Runtime.NODEJS_20_X,
  handler: "index.handler",
  code: lambda.Code.fromAsset(path.join(__dirname, "../../apps/chat/.open-next/server-function")),
  memorySize: 512,
  timeout: cdk.Duration.seconds(30),
  environment: {
    BUCKET_NAME: assetsBucket.bucketName,
    TABLE_NAME: isrTable.tableName,
  },
});

// Connect to CloudFront behavior
new cloudfront.Distribution(this, "MainDistribution", {
  additionalBehaviors: {
    "/chat*": {
      origin: new cloudfront.HttpOrigin(chatLambdaUrl.domainName),
      // ... cache behavior config
    },
  },
});
```

**Estimated Effort**: 15-20 hours
**Go Live**: 4-6 weeks

## 4. Deployment Checklist

### Phase 2A: Pre-Deployment (Days 1-2)

- [ ] **Version Decision**
  - [ ] Test upgrade: `pnpm upgrade next@16` on test branch
  - [ ] All 3 apps build successfully: `pnpm build`
  - [ ] Decide: v16+ or stay on v15?

- [ ] **Local Development Setup**
  - [ ] Install CDK dependencies: `pnpm add -D aws-cdk@2.180.0`
  - [ ] Install construct libraries: `pnpm add -D @aws-cdk/aws-lambda @aws-cdk/aws-cloudfront`
  - [ ] Verify CDK can synthesize: `pnpm synth`

- [ ] **AWS Account Setup**
  - [ ] Verify CDK bootstrap: `pnpm bootstrap`
  - [ ] Confirm region (should be `ap-southeast-2`): `aws configure list`
  - [ ] Domain DNS setup (CNAME → CloudFront domain)

### Phase 2B: Stack Development (Days 3-7, Path-dependent)

**If v16.2+ (cdk-nextjs)**

- [ ] Create `nextjs-multi-zones-stack.ts`
- [ ] Create `shared-nextjs-resources-stack.ts`
- [ ] Create `nextjs-zone-stack.ts`
- [ ] Test synth: `pnpm synth`

**If v15.x (OpenNext)**

- [ ] Build OpenNext artifacts: `pnpm build:open-next`
- [ ] Create `nextjs-lambda-shared-stack.ts`
- [ ] Create `nextjs-lambda-zone-stack.ts`
- [ ] Manually wire Lambda → CloudFront behaviors
- [ ] Test synth: `pnpm synth`

### Phase 2C: Dev Deployment (Days 8-9)

- [ ] Deploy shared resources: `pnpm deploy:nextjs:shared`
- [ ] Verify S3 bucket created
- [ ] Verify DynamoDB table created
- [ ] Verify CloudFront distribution

- [ ] Deploy first app (web): `pnpm deploy:nextjs:web`
- [ ] Test root path: `https://yourdomain.com/`
- [ ] Check CloudWatch logs

- [ ] Deploy second app (chat): `pnpm deploy:nextjs:chat`
- [ ] Test path-based: `https://yourdomain.com/chat`
- [ ] Verify asset loading: `https://yourdomain.com/chat-static/_next/...`

- [ ] Deploy third app (upload): `pnpm deploy:nextjs:upload`
- [ ] Test path-based: `https://yourdomain.com/upload`

### Phase 2D: Validation (Days 10-12)

- [ ] **Routing Tests**
  - [ ] `/` → web app loads
  - [ ] `/chat` → chat app loads
  - [ ] `/chat/:path` → chat routes work
  - [ ] `/upload` → upload app loads
  - [ ] Cross-zone link: `<a href="/upload/...">` works (hard nav)

- [ ] **Caching Tests**
  - [ ] Static assets cached: `Cache-Control: public, max-age=31536000`
  - [ ] `_next/static/*` served from S3
  - [ ] Dynamic content not cached: `Cache-Control: no-cache`

- [ ] **Performance Tests**
  - [ ] Root page TTI: <2s
  - [ ] Chat page TTI: <1.5s
  - [ ] CloudFront cache hit ratio: >70%

- [ ] **Multi-Zone Integrity**
  - [ ] No CSS conflicts between zones
  - [ ] Each zone can have separate Next.js versions (if desired)
  - [ ] Build IDs don't collide

### Phase 2E: Production Readiness (Days 13-14)

- [ ] **Monitoring Setup**
  - [ ] CloudWatch Logs per Lambda
  - [ ] CloudFront cache metrics
  - [ ] Lambda duration + error tracking
  - [ ] DynamoDB ISR table metrics (if using ISR)

- [ ] **Cost Validation**
  - [ ] Monthly estimate: $14.58-19 for 1M requests
  - [ ] Compare to baseline cost

- [ ] **Scaling & HA**
  - [ ] Reserved concurrent executions per Lambda
  - [ ] CloudFront geographic regions
  - [ ] Failover strategy documented

- [ ] **Documentation**
  - [ ] Runbook: How to deploy a new app to Lambda Multi Zones
  - [ ] Troubleshooting guide
  - [ ] Rollback procedure

## 5. Gotchas & Mitigations

| Gotcha                       | Symptom                                              | Fix                                                               |
| ---------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------- |
| **assetPrefix collision**    | Chat CSS overrides Web CSS                           | Ensure each app has unique assetPrefix (/chat, /upload, /)        |
| **Cross-zone Link Hard Nav** | `/chat → /upload` reloads page                       | Expected behavior. Use `<a>` not `<Link>` for cross-zone          |
| **S3 static asset 404**      | `https://yourdomain.com/chat-static/_next/... → 404` | Check CloudFront behavior order (static must be before dynamic)   |
| **DynamoDB ISR contention**  | High DynamoDB costs                                  | Use `ON_DEMAND` billing; prefixing per app keeps write volume low |
| **Lambda cold start >30s**   | Timeout on first request                             | Increase Lambda timeout; use provisioned concurrency for Web app  |
| **CORS on Server Actions**   | `POST /chat` fails with CORS error                   | Add `allowedOrigins` to both Web and Chat `next.config.ts`        |
| **buildId mismatch**         | Assets from different deployments mixed              | Include timestamp in buildId; S3 cleanup script on deploy         |

## 6. Cost Analysis

### Monthly Cost (1M requests)

**Path 1: cdk-nextjs + v16.2+**

```
CloudFront:      ~$11/month (data transfer)
Lambda (3x):     ~$3/month (1M requests, 1GB avg)
S3:              ~$1/month (static assets)
DynamoDB (opt):  ~$0.50/month (ISR, on-demand)
────────────────────────────
Total:           ~$16-19/month
```

**Path 2: OpenNext + v15.x**

```
CloudFront:      ~$11/month
Lambda (3x):     ~$3/month (same as v16.2+)
S3:              ~$1/month
DynamoDB (opt):  ~$0.50/month
────────────────────────────
Total:           ~$14.58/month (no construct premium)
```

**Savings vs Previous ECS**

- ECS Fargate (3 tasks × 0.25 CPU): ~$30/month
- Lambda Multi Zones: ~$16-19/month
- **Monthly Savings: ~$11-13**

## 7. Success Criteria

Phase 2 is **COMPLETE** when:

1. ✅ CDK stacks synthesize without error
2. ✅ All 3 apps build successfully (next build)
3. ✅ CloudFront distribution created with correct behaviors
4. ✅ Web app accessible at `https://yourdomain.com/`
5. ✅ Chat app accessible at `https://yourdomain.com/chat`
6. ✅ Upload app accessible at `https://yourdomain.com/upload`
7. ✅ Static assets served from S3 with long caching
8. ✅ No CSS/JS conflicts between zones
9. ✅ Cross-zone navigation (hard nav) works
10. ✅ Documentation updated with deployment runbook

## 8. References

- [cdk-nextjs GitHub](https://github.com/cdklabs/cdk-nextjs)
- [cdk-nextjs BYOR Examples](https://github.com/cdklabs/cdk-nextjs/tree/main/examples/bring-your-own)
- [OpenNext AWS Documentation](https://opennext.js.org/aws)
- [Next.js 16 Migration Guide](https://nextjs.org/docs/upgrade-guide)
- [AWS CDK CloudFront Distribution](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cloudfront-readme.html)
- [Phase 1 Decision Record](./adr-nextjs-lambda-deployment-2026.md)

---

**Next Step**: Implement one of the two paths based on version upgrade decision.
