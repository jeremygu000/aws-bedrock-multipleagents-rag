# HyDE (Hypothetical Document Embeddings) - 生产实现指南

## 概述

HyDE 是一种零样本密集检索技术，通过生成"假设文档"来改进 RAG 系统的检索准确率。

**核心思想**：

1. 使用 LLM 根据查询生成假设相关文档
2. 嵌入该假设文档（不是查询本身）
3. 在向量数据库中检索相似文档
4. LLM 的密集嵌入瓶颈过滤出虚假细节，捕获相关性模式

**性能收益**：

- PopQA: +7% | Biography: +14.9% | PubHealth: +36.6%
- 与重排器组合时最稳定：+15-25% recall

---

## 框架对比

### LlamaIndex（推荐）

**类**：`HyDEQueryTransform`

```python
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

hyde = HyDEQueryTransform(include_original=True)
query_bundle = hyde(query_str)
# 返回: embedding_strs = [假设文档, 原始查询]
```

**优势**：

- 轻量级（单一 LLM 调用）
- `include_original=True` 允许双重嵌入策略
- 灵活的提示自定义
- 生产就绪

### LangChain（遗留但支持）

**类**：`HypotheticalDocumentEmbedder`

```python
from langchain_community.chains.hyde import HypotheticalDocumentEmbedder

embedder = HypotheticalDocumentEmbedder.from_llm(
    llm=model,
    base_embeddings=embeddings,
    prompt_key="web_search"  # 8个内置领域提示
)
```

**优势**：

- 8 个内置领域特定提示
- `combine_embeddings()` 自动取平均

---

## AWS Bedrock 配置（ap-southeast-2）

### 推荐配置

| 组件     | 模型              | 原因                                      |
| -------- | ----------------- | ----------------------------------------- |
| 假设生成 | Nova Pro          | 平衡成本/性能，在 ap-southeast-2 直接可用 |
| 重排     | Claude Sonnet 4.5 | 最准确的相关性判断                        |
| 嵌入     | 本地 (pgvector)   | 零成本，适合 Bedrock 关键路径之外         |

### 成本分解（100万查询/月）

| 组件                     | 成本       | 计算                         |
| ------------------------ | ---------- | ---------------------------- |
| 假设生成 (Nova Pro)      | $160       | 1M × 200 tokens × $0.80/1M   |
| 重排 (Claude Sonnet 4.5) | $900       | 1M × 1,500 tokens × $0.60/1M |
| **总计**                 | **$1,060** |                              |

**成本优化策略**：

- 仅对长查询启用 HyDE (-40%)
- 采样 10% 的查询进行重排 (-90%)
- 缓存热门查询结果 (-40-70%)
- **优化后成本**：$160-320/月

---

## 故障模式与缓解

| 故障案例                               | 根本原因          | 缓解策略                           |
| -------------------------------------- | ----------------- | ---------------------------------- |
| **歧义查询** ("What is Bel?")          | LLM 外部知识优先  | `include_original=True` + 重排     |
| **命名实体** ("John Smith salary")     | LLM 幻觉数字      | 查询路由：实体 → BM25，语义 → HyDE |
| **短查询** ("Python 3.12 date")        | 成本 > 收益       | 长度检测：<5 tokens → 跳过 HyDE    |
| **开放式偏见** ("Art vs Engineering?") | 假设反映 LLM 偏见 | 多个假设 (N=5) + 重排              |

---

## 生产模式

### 何时启用 HyDE ✅

```
语义间隙 > 0.2
+ 长查询 (>15 tokens)
+ 推理密集型 ("explain", "discuss", "analyze")
→ 预期 Recall 改进：+10-30%
```

### 何时禁用 HyDE ❌

```
+ 实体/数字查询 (名字、日期、ID)
+ 短查询 (<5 tokens)
+ 精确匹配需求
→ 预期收益：-5% (成本 > 收益)
```

### 查询路由决策树

```python
def should_use_hyde(query: str) -> bool:
    # 1. 长度检测
    if len(query.split()) < 5:
        return False  # 太短，成本不值得

    # 2. 实体检测
    if has_entities(query, types=["PERSON", "DATE", "NUMBER"]):
        return False  # 使用 BM25

    # 3. 语义间隙估计
    bm25_score = rank_bm25(query)
    if bm25_score > 0.7:
        return False  # BM25 已经很好

    # 4. 查询意图
    if is_reasoning_query(query):
        return True  # "explain", "why", "compare"

    return True  # 默认启用
```

---

## 提示工程最佳实践

### 核心参数

```python
config = {
    "temperature": 0.7,  # NOT 0 (过于确定), NOT 1.0 (过于随机)
    "max_tokens": 500,   # 假设文档长度
    "num_hypotheses": 5, # 多个假设时
}
```

### 提示模板

```python
HYDE_SYSTEM_PROMPT = """You are an expert at understanding information retrieval tasks.
You will be given a query and asked to write a passage to answer it.
Your goal is to capture key details and semantic patterns that would appear in relevant documents."""

HYDE_USER_PROMPT = """Please write a passage to answer the question.
Try to include as many key details as possible.
Write the passage as if it were an excerpt from a document.

Question: {query}

Passage:"""
```

### Multi-HyDE（多个假设）

```python
# 生成 5 个不同视角的假设文档
hypotheses = []
for perspective in ["technical", "practical", "historical", "comparative", "interdisciplinary"]:
    h = llm.invoke(f"{HYDE_PROMPT}\nPerspective: {perspective}")
    hypotheses.append(h)

# 取平均嵌入
embeddings = [embed_model.embed_query(h) for h in hypotheses]
final_embedding = np.mean(embeddings, axis=0)  # 取平均，不连接
```

---

## 实现检查清单

### 📋 设计阶段

- [ ] 确定查询路由规则（长度、实体、意图检测）
- [ ] 选择 LLM（Nova Pro vs Claude）和成本预算
- [ ] 定义重排策略（采样率、阈值）
- [ ] 设计监控指标（HyDE 使用率、成本、Recall）

### 💻 实现阶段

- [ ] 集成 HyDE 框架（LlamaIndex）
- [ ] 实现查询路由器
- [ ] 配置 Bedrock 客户端（Nova Pro）
- [ ] 实现缓存层（Redis / DynamoDB）
- [ ] 创建单元测试（故障模式验证）

### 🧪 测试阶段

- [ ] 基准测试：测试集上的 Recall 改进
- [ ] 负面测试：验证故障模式缓解
- [ ] 成本测试：验证成本预算
- [ ] 延迟测试：端到端延迟 <500ms

### 📊 监控阶段

- [ ] CloudWatch 指标：HyDE 使用率、成本
- [ ] 预算告警：>$1,200/月
- [ ] Recall 告警：<90%
- [ ] 异常检测：查询失败率

---

## 相关文献

### 原始论文

- **HyDE**: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
  - ArXiv: 2212.10496 | ACL 2023
  - 作者：Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan

### 扩展研究

- **Multi-HyDE**: "Enhancing Financial RAG with Agentic AI and Multi-HyDE" (arXiv:2509.16369, 2025)
  - 性能：Recall +81.7%, Faithfulness +10.3%
- **Adaptive HyDE**: "Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support" (arXiv:2507.16754, 2025)
  - 自适应阈值，100% 覆盖保证

### 框架实现

- LlamaIndex: https://github.com/run-llama/llama_index (类 HyDEQueryTransform)
- LangChain: https://github.com/langchain-ai/langchain (HypotheticalDocumentEmbedder)
- Haystack: https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde

---

## 下一步

1. **集成到 RAG 服务**：`apps/rag-service/app/hyde_retriever.py`
2. **添加查询路由**：`apps/rag-service/app/query_router.py`
3. **运行 RAGAS 基准**：验证 Recall 改进
4. **配置监控**：CloudWatch + 成本告警
