# Next.js Lambda 部署工具对比矩阵 & 实施检查表

**生成日期**: 2026-04-13  
**范围**: 3 个 Next.js 15 Multi Zones 应用 + 1 个共享 CloudFront

---

## 工具对比矩阵

| 标准                      | SST v3          | cdk-nextjs         | OpenNext        | Lambda Web Adapter | @sladg/nextjs-lambda |
| ------------------------- | --------------- | ------------------ | --------------- | ------------------ | -------------------- |
| **维护状态**              | ✅ 活跃         | ✅ 活跃 (AWS Labs) | ✅ 活跃         | ✅ 活跃 (AWS)      | ❌ 存档 (EOL)        |
| **最后更新**              | Apr 2026        | Mar 2026           | Feb 2026        | Mar 2026           | Oct 2024             |
| **Next.js 15 支持**       | ✅ 是           | ❌ 仅 16.2+        | ✅ 是           | ✅ 是              | ❌ 仅 12-13          |
| **Next.js 16 支持**       | ✅ 是           | ✅ 是              | ✅ 是           | ✅ 是              | ❌ 不支持            |
| **多应用到 1 CloudFront** | ❌ 否           | ✅ 是 (BYOR)       | ⚠️ 手动         | ⚠️ 手动            | ❌ 不支持            |
| **共享 S3/DynamoDB**      | ❌ 否           | ✅ 是              | ⚠️ 手动         | ❌ 否              | ❌ 否                |
| **代码重复**              | 高 (3x)         | 最低               | 中等            | 最高               | 不适用               |
| **部署复杂度**            | 低              | 中等               | 中等            | 高                 | 不适用               |
| **维护负担**              | 中等            | 低                 | 中等            | 高                 | 不适用               |
| **npm 包大小**            | ~150MB          | ~45MB              | ~30MB           | ~15MB              | ~12MB (已废弃)       |
| **beta 阶段**             | ❌ 否           | ✅ 是 (0.5-beta)   | ❌ 否           | ❌ 否              | 不适用               |
| **TypeScript ESM**        | ✅ 是           | ✅ 是              | ✅ 是           | ✅ 是              | ⚠️ 部分              |
| **本地开发 DX**           | ⭐⭐⭐⭐⭐      | ⭐⭐⭐             | ⭐⭐⭐          | ⭐⭐               | 不适用               |
| **生产可靠性**            | ⭐⭐⭐⭐        | ⭐⭐⭐             | ⭐⭐⭐⭐        | ⭐⭐⭐⭐           | 不适用               |
| **文档质量**              | 很好            | 很好 (AWS 标准)    | 很好            | 很好               | 陈旧                 |
| **社区大小**              | 大 (15K+ stars) | 中等 (400 stars)   | 大 (30K+ stars) | 中等               | 已废弃               |
| **2026 推荐**             | ⚠️ 如果需要 v15 | ✅ 首选 (v16+)     | ⚠️ 备选 (v15)   | ❌ 仅在必要时      | ❌ 不推荐            |

---

## 详细功能对比

### 🔧 部署架构支持

#### SST v3

```
每个 NextjsSite 构造创建独立的：
- CloudFront 分布式系统
- Lambda 函数 + API Gateway
- S3 资源桶

问题：无法共享 CloudFront
解决方案：自定义中间件（复杂）
```

#### cdk-nextjs (BYOR)

```
✅ 创建单一共享的：
- CloudFront 分布式系统 (3 个缓存行为)
- S3 资源桶 (buildId 前缀)
- DynamoDB ISR 表

每个应用导入并扩展共享资源
```

#### OpenNext

```
🔧 手动配置：
- 3 个 OpenNext 构建 (.open-next/)
- 3 个 Lambda 函数 (CDK 创建)
- 3 个 API Gateway
- CloudFront (手动路由配置)

需要编程式连接
```

#### Lambda Web Adapter

```
❌ 全部手动：
- 3 个 Docker 镜像
- 3 个 Lambda 函数 (IAM + VPC)
- 3 个 API Gateway
- CloudFront (手动路由)
- 监控/日志配置 (手动)

最高维护成本
```

---

### 📊 成本对比 (1M 请求/月)

| 工具                   | Lambda | CloudFront | S3    | 其他       | 总计        | 每千请求         |
| ---------------------- | ------ | ---------- | ----- | ---------- | ----------- | ---------------- |
| **cdk-nextjs**         | $2.08  | $8.50      | $4.00 | $2-5 (DDB) | **$16-19**  | **$0.016-0.019** |
| **SST v3**             | $2.08  | $25.50     | $4.00 | $2-5       | **$33.50+** | **$0.034** ⚠️    |
| **OpenNext**           | $2.08  | $8.50      | $4.00 | 0          | **$14.58**  | **$0.015**       |
| **Lambda Web Adapter** | $3.50+ | $8.50      | $0    | 0          | **$12.00**  | **$0.012**       |

**关键见解**:

- cdk-nextjs vs SST: **每月节省 ~$17.50** (共享 CloudFront)
- Lambda Web Adapter 最便宜但维护成本最高
- OpenNext 成本与 cdk-nextjs 相近，但需手动集成

---

### ⏱️ 部署和开发速度

| 阶段             | cdk-nextjs            | SST v3              | OpenNext                     | Lambda Web Adapter       |
| ---------------- | --------------------- | ------------------- | ---------------------------- | ------------------------ |
| **本地开发启动** | `cdk synth` (30s)     | `sst dev` (60s)     | `npm run build` (60s)        | Docker build (90s+)      |
| **更改应用代码** | `cdk deploy` (3-5m)   | `sst deploy` (2-3m) | `cdk deploy` (4-6m)          | ECR push + deploy (5-8m) |
| **增加新应用**   | 1 个新栈 (5 分钟设置) | 新构造 (10 分钟)    | 新的 OpenNext 配置 (15 分钟) | 新 Dockerfile (30 分钟)  |
| **首次部署**     | 15-20 分钟            | 10-15 分钟          | 20-25 分钟                   | 30-40 分钟               |

---

### 🛠️ 集成工作量

#### cdk-nextjs 实施清单 (估计 8-10 小时)

```
- [ ] 升级到 Next.js 16.2+ (1-2 h)
- [ ] 创建 SharedNextjsStack (2-3 h)
  - [ ] 配置 3 个缓存行为
  - [ ] 创建 S3/DynamoDB 资源
- [ ] 为每个应用创建栈 (2-3 h)
  - [ ] app1/stack.ts with NextjsGlobalFunctions
  - [ ] app2/stack.ts (复制 + 调整)
  - [ ] app3/stack.ts (复制 + 调整)
- [ ] 配置每个 next.config.ts (1-2 h)
- [ ] 测试路由 (1-2 h)
```

#### SST v3 实施清单 (估计 12-15 小时)

```
- [ ] 创建 3 个 NextjsSite 构造 (2-3 h)
- [ ] 创建自定义中间件层 (4-6 h) ⚠️ 显著额外工作
  - [ ] 路由逻辑
  - [ ] 缓存协调
  - [ ] 错误处理
- [ ] CloudFront 配置 (2-3 h)
- [ ] 测试路由 + 缓存 (2-3 h)
```

#### OpenNext 手动 CDK (估计 15-20 小时)

```
- [ ] 配置 3 个 OpenNext 构建 (3-4 h)
- [ ] 手动创建 Lambda 函数堆栈 (5-7 h)
  - [ ] IAM 角色/权限
  - [ ] VPC 配置
  - [ ] 环境变量
- [ ] 创建 API Gateway (2-3 h)
- [ ] CloudFront 路由配置 (2-3 h)
- [ ] 测试 + 优化 (2-3 h)
```

#### Lambda Web Adapter (估计 20-25 小时)

```
- [ ] 为每个应用创建 Dockerfile (2-3 h)
- [ ] 设置 ECR 仓库 (1-2 h)
- [ ] 创建 Lambda 函数 + IAM (3-5 h)
- [ ] 设置 API Gateway (2-3 h)
- [ ] CloudFront 配置 (2-3 h)
- [ ] 监控/日志设置 (2-3 h)
- [ ] 测试 + 优化 (3-4 h)
```

---

## 决策树

```
开始: 需要部署 3 个 Next.js 应用到 Lambda

┌─ 你的 Next.js 版本是什么?
│
├─ v15.x
│  └─ 升级到 v16.2+ 可行吗?
│     ├─ 是 → 使用 cdk-nextjs ✅ (推荐)
│     └─ 否 → 使用 OpenNext (手动 CDK) ⚠️
│
└─ v16.2+
   └─ 需要最小维护吗?
      ├─ 是 → 使用 cdk-nextjs ✅ (推荐)
      └─ 否 →
         └─ 熟悉 Lambda 吗?
            ├─ 是 → Lambda Web Adapter (最低成本)
            └─ 否 → 使用 OpenNext (手动 CDK) ⚠️
```

---

## 推荐决策

### ✅ PRIMARY: cdk-nextjs (如果 Next.js 16.2+)

**何时选择**:

- 新项目或可升级到 v16.2+
- 需要最小代码重复
- 优先考虑可维护性
- pnpm 单一仓库设置

**何时跳过**:

- 锁定在 Next.js 15.x
- 需要完全自定义部署
- 想要 AWS-native 方法

---

### ⚠️ FALLBACK: OpenNext (Next.js 15.x)

**何时选择**:

- 不能升级到 v16.2+
- 愿意编写自定义 CDK 代码
- 需要完全控制部署

**何时跳过**:

- 不熟悉 CDK
- 需要最快的启动时间

---

### ❌ NOT RECOMMENDED

**SST v3**:

- 成本 2 倍（每月额外 $17+）
- 需要自定义中间件来实现多应用共享

**Lambda Web Adapter**:

- 维护负担最高 (Docker、ECR、手动配置)
- 仅在完全自定义需求时考虑

**@sladg/nextjs-lambda**:

- 已存档/EOL
- 不要用于新项目

---

## 实施清单

### ✅ 前置条件检查

- [ ] Next.js 版本 (运行 `npm list next`)
  - 如果 ≤15.x，决定是否升级
  - 如果 16.2+，可继续使用 cdk-nextjs
- [ ] CDK 版本 ≥2.100.0 (运行 `cdk version`)
- [ ] pnpm 工作区已配置
- [ ] 所有 3 个应用都能本地构建 (`pnpm build`)

### ✅ 工具选择

- [ ] 使用决策树确定工具
- [ ] 文件化决策理由（在 ADR 中）
- [ ] 获得团队同意

### ✅ Next.js 配置

- [ ] 验证每个应用 `next.config.ts`:
  - [ ] `basePath: '/app1'` (或应用 2/3)
  - [ ] `assetPrefix: '/app1-static'`
  - [ ] `output: 'standalone'`
- [ ] 测试本地构建成功

### ✅ CDK 集成 (仅 cdk-nextjs)

- [ ] 安装 `cdk-nextjs` 包
- [ ] 创建 `SharedNextjsStack`
- [ ] 创建 `App1Stack`, `App2Stack`, `App3Stack`
- [ ] 配置 CloudFront 缓存行为
- [ ] 运行 `cdk synth` (无错误)

### ✅ 部署 (开发环境)

- [ ] 部署共享栈: `cdk deploy SharedNextjsStack`
- [ ] 部署应用栈: `cdk deploy App{1,2,3}Stack`
- [ ] 验证 CloudFront 分布式系统 URL
- [ ] 测试 `/app1/*`, `/app2/*`, `/app3/*` 路由

### ✅ 测试和验证

- [ ] 验证每个应用在分配的路径可访问
- [ ] 验证静态资源缓存正确 (assetPrefix)
- [ ] 验证热刷新工作正常
- [ ] 验证 API 路由响应
- [ ] 检查 CloudFront 日志以确认路由

### ✅ 文档

- [ ] 记录每个应用的 basePath/assetPrefix
- [ ] 创建部署 runbook
- [ ] 文档故障排除指南
- [ ] 创建 CDK 自定义指南

### ✅ 成本和监控

- [ ] 估计每月 AWS 成本
- [ ] 设置 CloudWatch 告警
- [ ] 监控 Lambda 冷启动
- [ ] 监控 CloudFront 缓存命中率

---

## 常见问题与答案

### Q: 我能升级到 Next.js 16.2+ 吗?

**A**: 首先查看 [Next.js 16 迁移指南](https://nextjs.org/docs/upgrade-guide)。大多数应用只需最小改动。在开发分支中测试。

### Q: cdk-nextjs 仍在 beta，这是否意味着不稳定?

**A**: AWS Labs 维护，但仍在开发中。建议在测试环境中验证。定期监控 GitHub 发布。

### Q: 为什么不用 SST v3?

**A**: SST 为每个应用创建单独的 CloudFront，成本增加 3 倍。多应用支持需要自定义中间件，增加复杂性。

### Q: 我可以在有 OpenNext 的情况下降级吗?

**A**: 是的。OpenNext 是构建工具，不是部署框架。使用 OpenNext + 手动 CDK 对 Next.js 15.x 有效，但需要更多编程。

### Q: Lambda Web Adapter 何时是正确的选择?

**A**: 如果你已经有成熟的 Docker/ECR 工作流程，并且需要极端的成本优化。否则，维护负担不值得。

---

## 参考资源

- [cdk-nextjs GitHub](https://github.com/cdklabs/cdk-nextjs)
- [cdk-nextjs BYOR 示例](https://github.com/cdklabs/cdk-nextjs/tree/main/examples/bring-your-own)
- [OpenNext AWS 文档](https://opennext.js.org/aws)
- [Next.js 16 迁移指南](https://nextjs.org/docs/upgrade-guide)
- [AWS Lambda Web Adapter](https://aws.github.io/aws-lambda-web-adapter/)
