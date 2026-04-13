# ADR: Next.js 15 Multi Zones Lambda Deployment Strategy

**Date**: 2026-04-13  
**Status**: DECISION  
**Deciders**: Architecture Team  
**Affected Components**: Infra CDK, 3 Frontend Apps

## Problem Statement

需要在 AWS Lambda 上部署 3 个独立的 Next.js 15 Multi Zones 应用（各有不同的 `basePath`/`assetPrefix`），
集中在一个 CloudFront 分布式系统后面，采用路径基础的路由。

**关键约束**:

- pnpm 单一仓库，CDK TypeScript ESM
- 现有基础设施：Lambda + EC2（非 ECS/Fargate）
- 需要 2025-2026 维护的项目
- 3 个独立应用，1 个共享 CloudFront

## Considered Alternatives

### 1. SST v3 (Ion)

**Status**: ⚠️ 不推荐

**优点**:

- 一流的 Next.js 支持（基于 OpenNext）
- 完整的本地开发体验
- 部署简化

**缺点**:

- 每个 `NextjsSite` 构造创建自己的 CloudFront
- **无原生多应用到单一 CloudFront 的支持**
- 需要自定义中间件层来实现路由（显著复杂性）
- 每个应用的基础设施成本倍增

**可维护性**: 中等  
**2026 支持**: ✅ 是

---

### 2. cdk-nextjs (AWS Labs)

**Status**: ✅ 推荐（条件：Next.js 16.2+）

**优点**:

- ✅ 原生 "Bring Your Own Resources" (BYOR) 支持
- ✅ 完美支持共享 CloudFront + S3 + DynamoDB
- ✅ 显式支持 `basePath` 路由（文档行 283-284）
- ✅ 单一仓库友好（每个应用独立栈）
- ✅ 资源隔离通过 `buildId` 前缀
- ✅ AWS Labs 维护，定期更新
- 最少代码重复

**缺点**:

- 需要 Next.js 16.2+（Next.js Adapter API）
- 仍在 beta（v0.5.0-beta.7，2026 年 3 月）
- 需要手动 CloudFront 配置（3 个缓存行为）

**版本**: v0.5.0-beta.7 (Mar 2026)  
**可维护性**: 高（AWS Labs 所有权）  
**2026 支持**: ✅ 是  
**Next.js 支持**: 16.2+ 仅

**BYOR 模式证据**:

```typescript
// apps/app1/stack.ts
import { NextjsGlobalFunctions } from "cdk-nextjs";

new NextjsGlobalFunctions(this, "App1", {
  distribution: sharedDistribution, // ← 导入共享的 CloudFront
  distributionPaths: ["/app1/*"], // ← 路径基础路由
  buildId: "app1", // ← 资源隔离
  nextjsPath: "../apps/app1",
  skipBuild: false,
});
```

---

### 3. OpenNext (@opennextjs/aws)

**Status**: ⚠️ 可行但不推荐（v15.x 用户）

**优点**:

- ✅ 将 Next.js 构建转换为 Lambda zip
- ✅ 支持 Next.js 15.x 和 16.x
- ✅ 完整的类型脚本支持
- ✅ 积极维护（v3.9.16，2026 年 2 月）

**缺点**:

- **构建工具，而不是部署框架**
- 需要手动 CDK 编程来部署 3 个应用
- 需要手动 CloudFront 配置
- 没有 BYOR 支持（每个应用需要单独的基础设施）
- 比 cdk-nextjs 更复杂的集成
- 更高的维护负担

**版本**: v3.9.16 (Feb 2026)  
**可维护性**: 中等（更多手动代码）  
**2026 支持**: ✅ 是  
**Next.js 支持**: 15.x 和 16.x

**用途**: 如果 cdk-nextjs 不可行（Next.js 15.x 锁定），这是替代选项

---

### 4. Lambda Web Adapter (AWS 官方)

**Status**: ✅ 可行但高开销

**优点**:

- ✅ AWS 官方维护
- ✅ 支持 Next.js 15.x via `output: 'standalone'`
- ✅ 简单的 Docker 容器方法

**缺点**:

- ❌ **完全手动** CloudFront 设置（3 个应用）
- ❌ 需要为每个应用管理 Docker 镜像
- ❌ 没有共享基础设施的概念
- ❌ 最高维护负担
- 不推荐用于多应用场景

**版本**: v1.0.0 (Mar 2026)  
**可维护性**: 低（全部手动）  
**2026 支持**: ✅ 是  
**Next.js 支持**: 15.x 和 16.x

---

### 5. @sladg/nextjs-lambda

**Status**: ❌ 已存档/EOL

- 仓库于 2025 年 2 月 1 日存档
- 最后有意义的更新：2024 年 10 月
- 仅支持 Next.js 12-13
- **作者显式推荐迁移到 OpenNext**
- 不适合新项目

---

## Decision

**✅ 使用 cdk-nextjs（条件：Next.js 16.2+）**

### 推理

1. **最少代码重复**: BYOR 模式允许 3 个应用共享 CloudFront + S3 + DynamoDB
2. **最高可维护性**: AWS Labs 所有权 = 定期更新 + 长期支持
3. **明确的多应用支持**: 设计用于单一仓库场景（docs 中的例子）
4. **资源隔离**: `buildId` 前缀防止命名冲突
5. **最小部署复杂性**: 编程式资源创建对比手动 CDK

### 如果 Next.js 15.x 锁定：

**⚠️ 降级到 OpenNext (手动)**

- 配置 3 个独立的 OpenNext 构建
- 手动在 CDK 中创建 3 个 Lambda 函数 + 3 个 API Gateway
- 手动创建 CloudFront 分布式系统（3 个缓存行为）
- **更高的维护负担**，但可行

---

## Implementation Roadmap

### Phase 1: 升级到 Next.js 16.2+（如果当前 15.x）

```bash
pnpm upgrade next@latest
# 运行迁移脚本确认兼容性
pnpm build
```

### Phase 2: 创建共享基础设施栈

```typescript
// packages/infra-cdk/lib/shared-nextjs-stack.ts
import { Distribution, DistributionProps } from 'aws-cdk-lib/aws-cloudfront';

class SharedNextjsStack extends Stack {
  public distribution: Distribution;
  public cacheBucket: Bucket;
  public cacheTable: Table;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    // 创建单一 CloudFront 分布式系统
    // 配置 3 个缓存行为 (/app1/*, /app2/*, /app3/*)
    this.distribution = new Distribution(this, 'SharedDistribution', {
      defaultBehavior: { ... },
      additionalBehaviors: [
        { pathPattern: '/app1/*', ... },
        { pathPattern: '/app2/*', ... },
        { pathPattern: '/app3/*', ... },
      ],
    });

    // 创建共享 S3 缓存
    this.cacheBucket = new Bucket(this, 'CacheBucket', {
      encryption: BucketEncryption.S3_MANAGED,
      versioned: true,
    });

    // 创建共享 DynamoDB ISR 表
    this.cacheTable = new Table(this, 'CacheTable', {
      partitionKey: { name: 'pk', type: AttributeType.STRING },
    });
  }
}
```

### Phase 3: 为每个应用创建栈（导入共享资源）

```typescript
// apps/app1/cdk/stack.ts
import { NextjsGlobalFunctions } from "cdk-nextjs";

class App1Stack extends Stack {
  constructor(scope: Construct, id: string, shared: SharedNextjsStack) {
    super(scope, id);

    new NextjsGlobalFunctions(this, "App1", {
      distribution: shared.distribution, // ← 共享
      distributionPaths: ["/app1/*"], // ← 路径
      buildId: "app1", // ← 隔离
      nextjsPath: "../../apps/app1",
      skipBuild: false,
    });
  }
}
```

### Phase 4: 每个应用的 Next.js 配置

```typescript
// apps/app1/next.config.ts
export default {
  basePath: "/app1",
  assetPrefix: "/app1-static",
  output: "standalone",
};
```

### Phase 5: 部署

```bash
pnpm synth  # 生成 CloudFormation 模板
pnpm deploy SharedNextjsStack
pnpm deploy App1Stack
pnpm deploy App2Stack
pnpm deploy App3Stack
```

---

## Cost Analysis (1M 请求/月)

| 组件                | 数量    | 单价          | 月成本     |
| ------------------- | ------- | ------------- | ---------- |
| Lambda (GB-秒)      | 125,000 | $0.0000166667 | $2.08      |
| CloudFront (3 应用) | 1M      | $0.085/10M    | $8.50      |
| S3 (GET/PUT)        | 10K     | $0.0004       | $4.00      |
| DynamoDB ISR        | 按需    | 动态          | $1-5       |
| **总计**            |         |               | **$15-20** |

**vs SST (3 独立分布式系统)**:

- 3x CloudFront = $25.50（成本增加 3 倍）
- **节省**: 每月 ~$8-15

---

## Known Limitations & Mitigations

| 问题                 | 缓解                                   |
| -------------------- | -------------------------------------- |
| cdk-nextjs 仍在 beta | 监控 GitHub 发布；考虑 LTS 版本发布    |
| 需要 Next.js 16.2+   | 升级或降级到 OpenNext (手动)           |
| CloudFront 路由限制  | 使用路径前缀 + 缓存行为（已解决）      |
| DynamoDB ISR 成本    | 监控使用；如果过高，切换到 S3 事件通知 |

---

## Verification Checklist

- [ ] 升级到 Next.js 16.2+ 或确认 OpenNext 可行性
- [ ] 克隆 cdk-nextjs BYOR 示例
- [ ] 创建共享基础设施栈原型
- [ ] 为每个应用配置 basePath/assetPrefix
- [ ] 在开发中测试 3 应用路由
- [ ] 在测试中验证 CloudFront 缓存行为
- [ ] 成本预测分析
- [ ] 生产部署计划

---

## References

- **cdk-nextjs BYOR 示例**: https://github.com/cdklabs/cdk-nextjs/tree/main/examples/bring-your-own
- **cdk-nextjs README**: https://github.com/cdklabs/cdk-nextjs (行 283-284 关键)
- **OpenNext 官方文档**: https://opennext.js.org/aws
- **Next.js 16.2 Adapter API**: https://nextjs.org/blog/nextjs-across-platforms (2026 年 3 月)
- **AWS Lambda Web Adapter**: https://aws.github.io/aws-lambda-web-adapter/
- **CloudFront 缓存行为**: https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/distribution-web-values-specify.html
