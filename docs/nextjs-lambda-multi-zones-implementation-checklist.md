# Next.js Lambda Multi Zones - 实施清单

**Created**: 2026-04-13
**Phase**: 2 (Architecture Design & Implementation Starter)

---

## 🎯 Phase 2 实施步骤

### 关键决策（Day 1）

#### ☐ 1. 选择部署路径

**选项 A: Next.js v16.2+ + cdk-nextjs（推荐）**

- [ ] 在测试分支升级 Next.js: `pnpm upgrade next@^16.2.0`
- [ ] 验证所有 3 个应用构建: `pnpm build`
- [ ] 检查构建输出没有错误
- [ ] 测试本地开发是否工作: `pnpm dev`

**选项 B: Next.js v15.x + OpenNext（后备）**

- [ ] 保持 v15.0.0 现状
- [ ] 安装 OpenNext: `pnpm add -D @opennextjs/aws esbuild`
- [ ] 测试构建: `pnpm build`

**决策**: 选择 [ ] A or [ ] B

#### ☐ 2. AWS 环境准备

- [ ] 确认 AWS 凭证: `aws sts get-caller-identity`
- [ ] 确认区域是 ap-southeast-2: `aws configure list`
- [ ] 验证 CDK 引导状态: `pnpm bootstrap` (如果需要)
- [ ] 确认 IAM 权限（CloudFront, Lambda, S3, DynamoDB）

---

## 📁 Phase 2A: 代码结构准备（Day 2）

### ☐ 3. 创建 CDK 栈文件

**已创建**（在 `packages/infra-cdk/lib/nextjs-multi-zones/`）:

- [x] `shared-resources-stack.ts` — 共享 CloudFront + S3 + DynamoDB
- [x] `zone-app-stack.ts` — 单个应用的 Lambda 栈（v15.x + OpenNext）

**需要创建**:

- [ ] `nextjs-multi-zones-app.ts` — 主 CDK App 入口

  ```typescript
  import * as cdk from "aws-cdk-lib";
  import { SharedNextjsResourcesStack } from "./nextjs-multi-zones/shared-resources-stack";
  import { NextjsZoneAppStack } from "./nextjs-multi-zones/zone-app-stack";

  const app = new cdk.App();

  const sharedStack = new SharedNextjsResourcesStack(app, "SharedNextjsStack", {
    env: {
      account: process.env.CDK_DEFAULT_ACCOUNT,
      region: process.env.CDK_DEFAULT_REGION,
    },
  });

  // Deploy each zone
  const webStack = new NextjsZoneAppStack(app, "WebZoneStack", {
    appName: "web",
    buildOutputPath: "../../apps/web/.open-next/server-function",
    basePath: "/",
    memory: 1024,
    assetsBucket: sharedStack.assetsBucket,
    isrTable: sharedStack.isrTable,
    distribution: sharedStack.distribution!,
  });

  const chatStack = new NextjsZoneAppStack(app, "ChatZoneStack", {
    appName: "chat",
    buildOutputPath: "../../apps/chat/.open-next/server-function",
    basePath: "/chat",
    memory: 512,
    assetsBucket: sharedStack.assetsBucket,
    isrTable: sharedStack.isrTable,
    distribution: sharedStack.distribution!,
  });

  const uploadStack = new NextjsZoneAppStack(app, "UploadZoneStack", {
    appName: "upload",
    buildOutputPath: "../../apps/upload/.open-next/server-function",
    basePath: "/upload",
    memory: 512,
    assetsBucket: sharedStack.assetsBucket,
    isrTable: sharedStack.isrTable,
    distribution: sharedStack.distribution!,
  });

  app.synth();
  ```

### ☐ 4. 更新 Next.js 配置

**apps/chat/next.config.ts**

- [ ] 添加 `output: 'standalone'`
- [ ] 保持 `basePath: "/chat"`
- [ ] 保持 `assetPrefix: "/chat"`

**apps/upload/next.config.ts**

- [ ] 添加 `output: 'standalone'`
- [ ] 保持 `basePath: "/upload"`
- [ ] 保持 `assetPrefix: "/upload"`

**apps/web/next.config.ts**

- [ ] 添加 `output: 'standalone'`
- [ ] 保持 `basePath: "/"`（或移除）
- [ ] 保持 Multi Zone rewrites（本地开发用）
- [ ] **重要**: 添加 `experimental.serverActions.allowedOrigins`

### ☐ 5. 构建 OpenNext 产物（如果选择路径 B）

```bash
# Per app
cd apps/chat && npx @opennextjs/aws@latest build
cd apps/upload && npx @opennextjs/aws@latest build
cd apps/web && npx @opennextjs/aws@latest build

# Verify output
ls -la apps/chat/.open-next/server-function/
```

- [ ] 验证每个应用都有 `.open-next/server-function/` 目录
- [ ] 验证 `index.handler` 文件存在

### ☐ 6. 验证 CDK 合成

```bash
pnpm synth
```

- [ ] CDK 合成成功（输出到 `cdk.out/`）
- [ ] 没有 TypeScript 错误
- [ ] CloudFormation 模板有效

---

## 🚀 Phase 2B: 开发部署（Day 3-5）

### ☐ 7. 部署共享资源

```bash
pnpm deploy -- --stack-name SharedNextjsStack
```

- [ ] CloudFront 分布创建成功
- [ ] S3 assets 桶创建成功
- [ ] DynamoDB ISR 表创建成功（如果启用）
- [ ] 记录输出：
  - CloudFront 分布 ID: ******\_******
  - S3 桶名: ******\_******
  - DynamoDB 表名: ******\_******

### ☐ 8. 部署 Web Zone

```bash
pnpm deploy -- --stack-name WebZoneStack
```

- [ ] Lambda 函数创建成功
- [ ] Lambda 函数 URL 生成
- [ ] Lambda 可执行: `curl $(aws lambda get-function-url-config ...)`
- [ ] CloudWatch Logs 显示无错误

**测试 Web Zone**:

```bash
# Get CloudFront domain name
CFDN=$(aws cloudfront list-distributions --query 'DistributionList.Items[0].DomainName' --output text)

# Test
curl https://$CFDN/
```

- [ ] 返回 HTML（Web 应用首页）
- [ ] 状态码 200

### ☐ 9. 部署 Chat Zone

```bash
pnpm deploy -- --stack-name ChatZoneStack
```

- [ ] Lambda 函数创建成功
- [ ] CloudFront 行为添加 `/chat*` 路由

**测试 Chat Zone**:

```bash
CFDN=$(aws cloudfront list-distributions --query 'DistributionList.Items[0].DomainName' --output text)

curl https://$CFDN/chat
```

- [ ] 返回 HTML（Chat 应用首页）
- [ ] 资产加载: `https://$CFDN/chat-static/_next/...`

### ☐ 10. 部署 Upload Zone

```bash
pnpm deploy -- --stack-name UploadZoneStack
```

- [ ] Lambda 函数创建成功
- [ ] CloudFront 行为添加 `/upload*` 路由

---

## ✅ Phase 2C: 验证 & 测试（Day 6-7）

### ☐ 11. 路由验证

```bash
CFDN=$(aws cloudfront list-distributions ...)

# Test root
curl -I https://$CFDN/
# Expected: 200, Content-Type: text/html

# Test /chat
curl -I https://$CFDN/chat
# Expected: 200

# Test /upload
curl -I https://$CFDN/upload
# Expected: 200

# Test static asset
curl -I https://$CFDN/chat-static/_next/static/...
# Expected: 200, Cache-Control: public, max-age=31536000
```

- [ ] 所有路由返回 200
- [ ] 静态资产返回正确的 Cache-Control 头

### ☐ 12. 缓存验证

```bash
# CloudFront 缓存检查（两次访问，第二次应该更快）
time curl https://$CFDN/ > /dev/null
time curl https://$CFDN/ > /dev/null

# Check CloudFront metrics
aws cloudfront get-distribution-statistics --distribution-id <ID>
```

- [ ] 缓存命中率 > 70%
- [ ] 第二次请求更快

### ☐ 13. 跨区域导航

```html
<!-- 在 Web 应用中添加链接 -->
<a href="/chat">Go to Chat</a>
<a href="/upload">Go to Upload</a>

<!-- 在 Chat 应用中 -->
<a href="/">Back to Web</a>
```

- [ ] 导航工作（硬导航，整页重新加载）
- [ ] CSS/JS 没有冲突
- [ ] 每个区域维护自己的样式

### ☐ 14. CSS 冲突检查

访问 `https://$CFDN/chat` 并检查：

- [ ] Chat 应用的样式正确显示
- [ ] 没有从 Web 应用继承的样式

访问 `https://$CFDN/upload` 并检查：

- [ ] Upload 应用的样式正确显示
- [ ] 没有冲突

### ☐ 15. CloudWatch 日志

```bash
# View Lambda logs
aws logs tail /aws/lambda/ChatFunction --follow
aws logs tail /aws/lambda/WebFunction --follow
aws logs tail /aws/lambda/UploadFunction --follow
```

- [ ] 没有错误或异常
- [ ] 性能指标正常（冷启 <5s，热启 <1s）

---

## 📊 Phase 2D: 性能基准（Day 8）

### ☐ 16. 测量 TTI (Time to Interactive)

```bash
# 使用 Lighthouse 或 WebPageTest
# 或手动用浏览器开发者工具

CFDN=$(aws cloudfront list-distributions ...)

# Web zone
curl -w "Total: %{time_total}s\n" https://$CFDN/

# Chat zone
curl -w "Total: %{time_total}s\n" https://$CFDN/chat

# Upload zone
curl -w "Total: %{time_total}s\n" https://$CFDN/upload
```

- [ ] 首屏加载 < 2s
- [ ] TTI < 3s

### ☐ 17. Lambda 性能监控

```bash
# 查看 Lambda 持续时间
aws logs filter-log-events \
  --log-group-name /aws/lambda/WebFunction \
  --filter-pattern "[timestamp, request_id, ..., duration]" \
  --query 'events[*].message'
```

- [ ] 冷启 P99 < 5s
- [ ] 热启 P99 < 1.5s
- [ ] 没有超时（30s）

---

## 💰 Phase 2E: 成本验证（Day 9）

### ☐ 18. 成本估算

```bash
# AWS Pricing Calculator
# CloudFront: ~$11/month (1M requests)
# Lambda: ~$3/month (1M requests, 1GB avg)
# S3: ~$1/month (static assets)
# DynamoDB: ~$0.50/month (ISR, on-demand)

# Total: ~$14.58-19/month (vs ~$30/month ECS)
```

- [ ] 成本在预期范围内
- [ ] 月度节省 ~$11-13（vs ECS）

### ☐ 19. AWS Billing Dashboard

```bash
# 查看当前月份的估计账单
aws ce get-cost-and-usage \
  --time-period Start=2026-04-01,End=2026-04-30 \
  --granularity MONTHLY \
  --metrics "BlendedCost" \
  --filter file://filter.json
```

- [ ] CloudFront 成本合理
- [ ] Lambda 成本合理
- [ ] 没有意外费用

---

## 📚 Phase 2F: 文档 & 运行手册（Day 10）

### ☐ 20. 部署运行手册

创建 `docs/nextjs-lambda-multi-zones-runbook.md`:

```markdown
# Next.js Lambda Multi Zones - 部署运行手册

## 快速开始

### 部署新应用到 Lambda Multi Zones

1. 创建新的 Next.js 应用在 `apps/myapp/`
2. 配置 `next.config.ts`:
   \`\`\`typescript
   const nextConfig = {
   basePath: "/myapp",
   assetPrefix: "/myapp",
   output: 'standalone',
   };
   \`\`\`
3. 构建 Next.js: `pnpm build`
4. 构建 OpenNext: `npx @opennextjs/aws@latest build` (if v15.x)
5. 创建 CDK 栈: `new NextjsZoneAppStack(app, 'MyappZoneStack', {...})`
6. 部署: `pnpm deploy -- --stack-name MyappZoneStack`

## 故障排除

### 404 静态资产

**症状**: `https://yourdomain.com/chat-static/_next/... → 404`

**检查**:

1. CloudFront 行为优先级（静态必须在动态之前）
2. S3 assets 桶权限
3. CloudFront OAI 配置

### CORS 错误 Server Actions

**症状**: `POST /chat → CORS error`

**检查**:

1. `next.config.ts` 中的 `allowedOrigins`
2. CloudFront 转发 Cookie 头
3. Lambda 函数 URL CORS 配置

### Lambda 超时

**症状**: 30s 超时，某些请求失败

**解决**:

1. 增加 Lambda 超时（最多 15 分钟）
2. 增加 Lambda 内存（改进 CPU）
3. 启用预热（定时调用保持热）

## 监控

### CloudWatch Dashboards

创建自定义仪表盘:

- CloudFront 缓存命中率
- Lambda 持续时间分布
- Lambda 错误率
- DynamoDB ISR 写入速率

### 告警

- Lambda 错误率 > 1%
- Lambda P99 持续时间 > 5s
- CloudFront 源错误率 > 0.1%

## 扩展

### 添加更多应用

只需:

1. 创建新应用目录
2. 编写 CDK 栈定义
3. 部署到现有的 CloudFront

所有应用共享 S3 assets 和 DynamoDB ISR.
```

- [ ] 运行手册已创建
- [ ] 包含常见故障排除
- [ ] 包含监控设置

### ☐ 21. 架构决策记录

- [ ] Phase 1 ADR 已创建 ✓（adr-nextjs-lambda-deployment-2026.md）
- [ ] Phase 2 ADR 已创建 ✓（adr-nextjs-lambda-multi-zones-phase2-architecture.md）
- [ ] Phase 2 实施清单 ✓（本文件）

### ☐ 22. Git 提交

```bash
# 暂存所有文件
git add .

# 创建提交
git commit -m "feat: add Next.js Lambda Multi Zones CDK stacks & documentation

- Add shared-resources-stack.ts (CloudFront + S3 + DynamoDB)
- Add zone-app-stack.ts (per-app Lambda wrapper for v15.x)
- Add adr-nextjs-lambda-multi-zones-phase2-architecture.md
- Add nextjs-lambda-multi-zones-implementation-checklist.md"

# 推送
git push origin main
```

- [ ] 所有文件已提交
- [ ] 没有 lint 错误
- [ ] Pre-commit 钩子通过

---

## 🎉 Phase 2 完成标准

**Phase 2 COMPLETE** when:

1. ✅ CDK 栈综合（synth）成功
2. ✅ 所有 3 个应用构建成功
3. ✅ CloudFront 分布创建并配置 3 个缓存行为
4. ✅ Web 应用在 `/` 可访问
5. ✅ Chat 应用在 `/chat` 可访问
6. ✅ Upload 应用在 `/upload` 可访问
7. ✅ 静态资产从 S3 提供，使用长缓存
8. ✅ 没有区域之间的 CSS/JS 冲突
9. ✅ 跨区域导航（硬导航）工作
10. ✅ CloudWatch 日志无错误
11. ✅ 成本估算 $14.58-19/月
12. ✅ 文档和运行手册已创建

---

## 📝 附录

### 有用命令

```bash
# 查看所有栈
pnpm cdk list

# 合成特定栈
pnpm synth -- --stack-name ChatZoneStack

# 部署特定栈
pnpm deploy -- --stack-name ChatZoneStack --require-approval never

# 销毁特定栈
pnpm destroy -- --stack-name ChatZoneStack

# 查看栈差异
pnpm diff -- --stack-name ChatZoneStack

# 查看 Lambda 日志
aws logs tail /aws/lambda/ChatFunction --follow

# 查看 CloudFront 指标
aws cloudfront get-distribution --id <DISTRIBUTION_ID>

# 清除 CloudFront 缓存
aws cloudfront create-invalidation --distribution-id <ID> --paths "/*"
```

### 参考资源

- [Phase 1 ADR](./adr-nextjs-lambda-deployment-2026.md)
- [Phase 2 Architecture](./adr-nextjs-lambda-multi-zones-phase2-architecture.md)
- [AWS CDK Documentation](https://docs.aws.amazon.com/cdk/)
- [Next.js 16 Migration Guide](https://nextjs.org/docs/upgrade-guide)
- [OpenNext AWS Docs](https://opennext.js.org/aws)
