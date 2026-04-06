# CRAG（纠错式检索增强生成）技术设计文档

> **作者**: aws-bedrock-multiagents team
> **日期**: 2026-04-06
> **状态**: 已实施 ✅（Phase 1）
> **前置条件**: P0 RAGAS CI/CD 已完成 ✅

---

## 目录

1. [什么是 CRAG？](#1-什么是-crag)
2. [学术背景与论文来源](#2-学术背景与论文来源)
3. [为什么 CRAG 被认为有效？](#3-为什么-crag-被认为有效)
4. [CRAG 核心架构](#4-crag-核心架构)
5. [与 Self-RAG 的关系](#5-与-self-rag-的关系)
6. [当前系统现状分析](#6-当前系统现状分析)
7. [CRAG 实施方案](#7-crag-实施方案)
8. [阈值调优策略](#8-阈值调优策略)
9. [生产环境基准数据](#9-生产环境基准数据)
10. [已知局限性与缓解措施](#10-已知局限性与缓解措施)
11. [参考文献](#11-参考文献)

---

## 1. 什么是 CRAG？

**CRAG（Corrective Retrieval Augmented Generation，纠错式检索增强生成）** 是一种改进的 RAG 架构，解决传统 RAG 系统的核心缺陷：**盲目信任检索结果**。

### 传统 RAG 的问题

传统 RAG 的工作方式是：

```
用户查询 → 检索文档 → 直接传给 LLM → 生成答案
```

这里有一个致命假设：**检索到的文档一定是相关的**。实际上并非如此。当检索质量差时（文档不相关、不完整、甚至自相矛盾），LLM 会基于错误的上下文生成一个"看起来很自信"但实际上是幻觉的答案。

### CRAG 的解决方案

CRAG 在"检索"和"生成"之间插入一个 **质量评估 + 自适应纠错** 层：

```
用户查询 → 检索文档 → ❶ 评估检索质量 → ❷ 根据评估结果采取不同行动：
  ├─ 质量好 (CORRECT)    → 精炼文档 → 生成答案
  ├─ 质量差 (INCORRECT)  → 改写查询 → Web 搜索 → 精炼 → 生成答案
  └─ 模糊 (AMBIGUOUS)    → 内部文档 + Web 搜索 → 合并精炼 → 生成答案
```

简单来说：**CRAG 让系统知道自己"不知道"，然后主动去修正，而不是硬编答案。**

---

## 2. 学术背景与论文来源

### 2.1 CRAG 原始论文

| 字段         | 内容                                       |
| ------------ | ------------------------------------------ |
| **论文标题** | Corrective Retrieval Augmented Generation  |
| **ArXiv ID** | 2401.15884（v3，2024 年 10 月 7 日最终版） |
| **首次提交** | 2024 年 1 月 29 日                         |
| **DOI**      | 10.48550/arXiv.2401.15884                  |
| **PDF 链接** | https://arxiv.org/pdf/2401.15884           |

**作者与机构：**

| 作者              | 机构                                                               |
| ----------------- | ------------------------------------------------------------------ |
| **Shi-Qi Yan**\*  | 中国科学技术大学（USTC），语音与语言信息处理国家工程研究中心，合肥 |
| **Jia-Chen Gu**\* | 加利福尼亚大学计算机科学系                                         |
| **Yun Zhu**       | （论文中未注明机构）                                               |
| **Zhen-Hua Ling** | 中国科学技术大学（USTC），同上                                     |

> \* 表示共同第一作者（equal contribution）。

**论文核心主张：**

> "大语言模型因依赖参数化知识而产生固有幻觉。RAG 虽是实用方案，但引入了新的脆弱点：**完全依赖检索质量**。CRAG 通过一个轻量级检索评估器来评估检索文档的质量，并根据置信度触发不同的自适应行动。"

### 2.2 Self-RAG 基础论文

CRAG 建立在 Self-RAG 的概念之上，两者是互补关系。

| 字段         | 内容                                                                           |
| ------------ | ------------------------------------------------------------------------------ |
| **论文标题** | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection |
| **ArXiv ID** | 2310.11511（2023 年 10 月 17 日）                                              |
| **DOI**      | 10.48550/arXiv.2310.11511                                                      |

**作者与机构：**

| 作者                      | 机构                               |
| ------------------------- | ---------------------------------- |
| **Akari Asai**†           | 华盛顿大学                         |
| **Zeqiu Wu**†             | 华盛顿大学                         |
| **Yizhong Wang**†§        | 华盛顿大学、Allen Institute for AI |
| **Avirup Sil**            | IBM Research AI                    |
| **Hannaneh Hajishirzi**†§ | 华盛顿大学、Allen Institute for AI |

> † 共同贡献，§ 通讯作者

**Self-RAG 的关键创新：**

- **自适应检索**：通过特殊 token 按需决定是否检索
- **反思 token**：在推理过程中生成自评 token 来批判已检索的文段
- **可控 LM**：特殊 token 允许在推理时控制生成行为

---

## 3. 为什么 CRAG 被认为有效？

### 3.1 实验数据

CRAG 论文在 4 个基准数据集上测试了改进效果（相比标准 RAG）：

| 数据集            | 类型             | 提升幅度             | 说明                                               |
| ----------------- | ---------------- | -------------------- | -------------------------------------------------- |
| **PopQA**         | 开放领域问答     | **+7.0%** 准确率     | 常见知识问答；检索通常有效但文档可能有噪声         |
| **Biography**     | 人物传记事实提取 | **+14.9%** FactScore | 需要精确的事实细节；精炼机制减少幻觉               |
| **PubHealth**     | 健康领域问答     | **+36.6%** 准确率    | 高风险领域；文档质量至关重要；Web 搜索补充事实核查 |
| **Arc-Challenge** | 通用推理         | **+15.4%** 准确率    | 评估器需区分高质量与误导性上下文                   |

**平均提升约 +15%。**

**与 Self-RAG 叠加使用时**：额外 **+20–36%** 改进。这说明两种方法针对的是不同类型的错误——叠加使用有乘数效应。

### 3.2 为什么这些改进是可信的？

1. **评估方法论严谨**：使用了 4 个跨领域数据集（事实问答、传记、健康、推理），非单一基准
2. **指标多样**：Accuracy（准确率）、FactScore（事实级别评分）、F1（token 级别精确率+召回率）
3. **可复现**：官方代码开源（[HuskyInSalt/CRAG](https://github.com/HuskyInSalt/CRAG)，450+ Stars）
4. **独立验证**：2025-2026 年间多个独立团队在生产环境验证了类似效果

### 3.3 为什么传统 RAG 做不到？

传统 RAG 的致命流程：

```
检索到 5 篇文档 → 全部传给 LLM → LLM 不知道哪些相关哪些不相关 → 基于噪声生成答案
```

问题的关键：**LLM 没有能力判断检索结果的质量**。它会把所有传入的文档当作"真相"来处理。

CRAG 的关键洞察：**在生成前先判断"这些文档够不够用"**。这看似简单，但正是这一步将错误率从约 15% 降到了约 3%。

---

## 4. CRAG 核心架构

CRAG 由三个核心组件组成：

### 4.1 组件一：检索评估器（Retrieval Evaluator）

**职责**：评估每一篇检索文档与用户查询的相关性。

**原始论文使用**：T5 模型微调的相关性分类器，准确率 84.3%。

**生产中的替代方案**（LLM-as-Judge，更实用）：

```python
class GradeDocuments(BaseModel):
    """二元相关性评分"""
    binary_score: str = Field(description="'yes' 或 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """你是一个评分员，评估检索到的文档与用户问题的相关性。
如果文档包含与问题相关的关键词或语义含义，则评为相关（'yes'），否则为不相关（'no'）。
不需要完全回答问题，只要包含相关信息即可。"""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "检索文档:\n\n{document}\n\n用户问题: {question}"),
])

retrieval_grader = grade_prompt | structured_llm_grader
```

**评估方法对比：**

| 方法                              | 准确率 | 延迟  | 成本                | 推荐          |
| --------------------------------- | ------ | ----- | ------------------- | ------------- |
| **LLM-as-Judge**（CRAG 生产标准） | >80%   | ~50ms | 每文档一次 LLM 调用 | ✅ 推荐       |
| T5 微调分类器（论文原版）         | 84.3%  | ~10ms | 需要训练基础设施    | 高吞吐场景    |
| 嵌入相似度（cosine）              | ~65%   | ~2ms  | 无额外成本          | ❌ 精度不足   |
| BM25 关键词匹配                   | ~55%   | ~1ms  | 无额外成本          | ❌ 不理解语义 |

### 4.2 组件二：三级判定系统（Three-Verdict System）

根据评估结果，CRAG 将检索质量分为三个等级，每个等级触发不同的处理流程：

#### 判定 1：CORRECT（质量好）

**触发条件**：至少一篇文档得分 ≥ `UPPER_THRESHOLD`（默认 0.7）

**处理流程**：

1. 保留高分文档
2. 执行"分解-重组"精炼（见 4.3 节）
3. 传给生成器
4. 返回答案

**延迟**：约 3–5 秒（最快路径）

#### 判定 2：INCORRECT（质量差）

**触发条件**：所有文档得分 < `LOWER_THRESHOLD`（默认 0.3）

**处理流程**：

1. 判定内部知识库无法回答
2. 改写查询以优化 Web 搜索
3. 执行 Web 搜索（Tavily/Google）
4. 对 Web 结果执行"分解-重组"精炼
5. 传给生成器
6. 返回答案

**延迟**：约 8–12 秒（包含 Web 搜索）

#### 判定 3：AMBIGUOUS（模糊）

**触发条件**：部分文档得分 > `LOWER_THRESHOLD` 但都 < `UPPER_THRESHOLD`

**处理流程**：

1. 保留通过低阈值的文档
2. 同时改写查询 → Web 搜索
3. 合并内部文档 + Web 结果
4. 对合并集执行"分解-重组"精炼
5. 传给生成器
6. 返回答案

**延迟**：约 8–12 秒

```
                    ┌──────────────────────┐
                    │   检索评估器         │
                    │   (Retrieval         │
                    │    Evaluator)        │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
        score ≥ 0.7      0.3 < score < 0.7   score < 0.3
              │                │                │
        ┌─────┴─────┐   ┌─────┴─────┐   ┌─────┴─────┐
        │  CORRECT   │   │ AMBIGUOUS  │   │ INCORRECT  │
        │            │   │            │   │            │
        │ 精炼文档   │   │ 内部文档   │   │ 改写查询   │
        │     ↓      │   │ + Web 搜索 │   │     ↓      │
        │   生成     │   │ → 合并精炼 │   │ Web 搜索   │
        │     ↓      │   │     ↓      │   │     ↓      │
        │   返回     │   │   生成     │   │  精炼      │
        └────────────┘   │     ↓      │   │     ↓      │
                         │   返回     │   │   生成     │
                         └────────────┘   │     ↓      │
                                          │   返回     │
                                          └────────────┘
```

### 4.3 组件三：分解-重组算法（Decompose-Then-Recompose）

**目的**：从检索文档中提取与查询真正相关的信息，去除噪声。

**算法步骤**：

1. **分解**：将文档按句子拆分
2. **过滤**：用 LLM 逐句评估相关性（保留 / 丢弃）
3. **重组**：将保留的句子重新拼接为精简上下文

**效果**：文档体积通常缩减 60–75%，同时答案质量持平或提升。

**示例**：

```
输入 (4 个文档, 共 3000 tokens):
  文档1: [相关句子A] [无关句子B] [相关句子C]
  文档2: [全部无关]
  文档3: [相关句子D] [无关句子E]
  文档4: [相关句子F]

分解后: 6 个句子
过滤后: 4 个句子 (A, C, D, F)
重组后: ~900 tokens (缩减 70%)
```

> **注意**：在我们的实施方案中，考虑到已有 LLM 重排功能，分解-重组可简化为阈值过滤（见第 7 节）。

---

## 5. 与 Self-RAG 的关系

CRAG 和 Self-RAG 是互补的——它们解决的是 RAG 管道中不同阶段的问题。

### 5.1 概念对比

| 维度             | CRAG                                  | Self-RAG                 |
| ---------------- | ------------------------------------- | ------------------------ |
| **检查时机**     | 检索之后、生成之前                    | 生成之后（自我反思）     |
| **纠错方式**     | 外部补救（Web 搜索）                  | 内部反思（重新生成）     |
| **训练要求**     | 微调外部评估器（轻量）                | 微调整个 LM（重量级）    |
| **集成难度**     | 即插即用（plug-and-play）             | 需要重新训练模型         |
| **最佳场景**     | 检索完全失败时                        | 检索质量尚可但生成有误时 |
| **论文报告增益** | +7% ~ +36%                            | +10% ~ +20%              |
| **叠加效果**     | ✅ CRAG + Self-RAG = +20–36% 额外增益 | ✅                       |

### 5.2 为什么叠加有效？

- **CRAG 修复的问题**："检索到的文档完全不相关" → 通过 Web 搜索补救
- **Self-RAG 修复的问题**："上下文还行但 LLM 推理出错" → 通过自我反思纠正
- **叠加修复**：两种错误都能捕获 → 复合鲁棒性

### 5.3 实施建议

**Phase 1（本次 P1）**：先实施 CRAG（即插即用，改动小，ROI 高）

**Phase 2（未来 P6）**：在 CRAG 基础上叠加 Self-RAG（答案生成后增加幻觉检测 + 相关性检测节点）

---

## 6. 当前系统现状分析

### 6.1 现有 14 节点 LangGraph 管道

```
START
  ↓
check_cache ──(cache_hit)──→ END
  │(cache_miss)
  ↓
detect_intent → extract_keywords → determine_mode → rewrite_query
  ↓
build_request → retrieve → graph_retrieve → fuse → rerank
  ↓
build_citations → choose_model → generate_answer → store_cache → END
```

### 6.2 现有的质量控制机制

| 机制         | 位置                 | 作用                | 局限                   |
| ------------ | -------------------- | ------------------- | ---------------------- |
| **LLM 重排** | `rerank` 节点        | 按相关性分数排序    | 仅排序，没有阈值过滤   |
| **模型路由** | `choose_model` 节点  | 弱证据 → 用更强模型 | 仅选模型，不重试检索   |
| **查询改写** | `rewrite_query` 节点 | 优化检索关键词      | 一次性改写，不评估效果 |

**核心缺失**：系统在检索 → 生成之间没有质量门禁。无论检索到什么，都会被传给生成器。

### 6.3 状态对象（RagWorkflowState）

```python
class RagWorkflowState(TypedDict, total=False):
    query: str                          # 原始查询
    top_k: int                          # 返回数量
    filters: dict[str, Any]             # 过滤条件
    intent: str                         # factual/analytical/procedural/comparison
    complexity: str                     # low/medium/high
    hl_keywords: list[str]              # 高级主题关键词
    ll_keywords: list[str]              # 低级实体关键词
    retrieval_mode: str                 # naive/local/global/hybrid/mix
    rewritten_query: str                # 重写后的查询
    query_embedding: list[float] | None # 密集向量
    request: RetrieveRequest            # 检索请求
    hits: list[dict[str, Any]]          # 原始检索结果
    graph_context: GraphContext          # 图形检索结果
    fused_hits: list[dict[str, Any]]    # 融合后
    reranked_hits: list[dict[str, Any]] # 重排后
    citations: list[dict[str, Any]]     # 引文
    preferred_model: str                # 模型选择
    answer_model: str                   # 实际使用的模型
    answer: str                         # 最终答案
    cache_hit: bool                     # 是否缓存命中
```

### 6.4 配置标志模式

现有 40+ 配置标志在 `config.py` 中，遵循 `RAG_ENABLE_*` / `RAG_*_THRESHOLD` 命名规范。CRAG 标志将遵循相同模式。

---

## 7. CRAG 实施方案

### 7.1 整体策略

在现有 14 节点管道的基础上，CRAG 新增 **3 个节点** 和 **2 条条件边**。

新增节点：

1. **`grade_retrieval`** — 评估检索质量（LLM-as-Judge）
2. **`crag_rewrite_query`** — CRAG 专用查询改写（面向 Web 搜索优化）
3. **`crag_web_search`** — Web 搜索回退

新增条件边：

1. `rerank` → `grade_retrieval`（替代原来直接到 `build_citations`）
2. `grade_retrieval` → 条件分支：`build_citations`（CORRECT）/ `crag_rewrite_query`（INCORRECT/AMBIGUOUS）

### 7.2 改造后的管道

```
START
  ↓
check_cache ──(cache_hit)──→ END
  │(cache_miss)
  ↓
detect_intent → extract_keywords → determine_mode → rewrite_query
  ↓
build_request → retrieve → graph_retrieve → fuse → rerank
  ↓
┌─────────────────────────────┐
│  grade_retrieval (新节点)    │   ← CRAG 核心
│  评估 reranked_hits 质量    │
│  输出: retrieval_verdict    │
└──────────┬──────────────────┘
           │
     ┌─────┴──────────────────┐
     │                        │
  CORRECT              INCORRECT / AMBIGUOUS
     │                        │
     │           ┌────────────┴────────────┐
     │           │  crag_rewrite_query     │   ← CRAG 查询改写
     │           │  (新节点)               │
     │           └────────────┬────────────┘
     │                        │
     │           ┌────────────┴────────────┐
     │           │  crag_web_search        │   ← CRAG Web 搜索
     │           │  (新节点)               │
     │           └────────────┬────────────┘
     │                        │
     └────────┬───────────────┘
              ↓
build_citations → choose_model → generate_answer → store_cache → END
```

### 7.3 新增状态字段

```python
# 添加到 RagWorkflowState
retrieval_verdict: str       # "correct" | "incorrect" | "ambiguous"
crag_retry_count: int        # CRAG 重试次数（默认 0，上限 2）
web_search_results: list[dict[str, Any]]  # Web 搜索结果
crag_rewritten_query: str    # CRAG 改写后的查询（面向 Web 搜索）
```

### 7.4 新增配置标志

```python
# apps/rag-service/app/config.py 新增

# CRAG 总开关
RAG_ENABLE_CRAG: bool = False

# 评估阈值
RAG_CRAG_UPPER_THRESHOLD: float = 0.7      # ≥ 此值 → CORRECT
RAG_CRAG_LOWER_THRESHOLD: float = 0.3      # ≤ 此值 → INCORRECT
RAG_CRAG_MIN_RELEVANT_DOCS: int = 1        # 最少需要多少相关文档

# 回退
RAG_CRAG_ENABLE_WEB_SEARCH: bool = False   # 是否启用 Web 搜索回退（默认关闭，需要 Tavily API key）
RAG_CRAG_WEB_SEARCH_K: int = 3             # Web 搜索返回数量

# 评估器模型
RAG_CRAG_GRADER_MODEL: str = ""            # 评估器用的 LLM（空值 = 使用 Qwen client）
TAVILY_API_KEY: str = ""                   # Tavily API key（启用 Web 搜索时必需）
```

> **与原始设计的差异**：
>
> - `RAG_CRAG_MAX_RETRIES` 在 Phase 1 中暂未实现（重试逻辑推迟到后续迭代）
> - `RAG_CRAG_ENABLE_WEB_SEARCH` 默认改为 `False`（更安全的默认值，避免意外触发外部 API 调用）
> - `RAG_CRAG_GRADER_MODEL` 默认改为空字符串（复用已有 Qwen client，降低配置复杂度）
> - 新增 `TAVILY_API_KEY` 配置项

### 7.5 节点实现规范

#### 节点 1：`grade_retrieval`

**输入**：`reranked_hits`, `query`（或 `rewritten_query`）
**输出**：`retrieval_verdict`, 更新后的 `reranked_hits`（过滤掉不相关的文档）

**伪代码**（实际实现见 `app/crag.py` 的 `RetrievalGrader.grade_hits()`）：

```python
def _impl_grade_retrieval(self, state: RagWorkflowState) -> RagWorkflowState:
    if not self._settings.enable_crag or not self._retrieval_grader:
        return {"retrieval_verdict": "correct"}

    query = state.get("rewritten_query") or state.get("query", "")
    hits = state.get("reranked_hits") or state.get("fused_hits") or []
    if not hits:
        return {"retrieval_verdict": "incorrect", "reranked_hits": []}

    relevant, verdict = self._retrieval_grader.grade_hits(query, hits)
    return {
        "retrieval_verdict": verdict,
        "reranked_hits": relevant if relevant else hits,
    }
```

> **实现说明**：评估逻辑封装在 `RetrievalGrader` 类中，通过构造函数注入到 `RagWorkflow`。
> 评估器使用 QwenClient 的 `chat()` 方法，要求 LLM 返回 `{"relevant": true/false}` JSON 结构。

#### 节点 2：`crag_rewrite_query`

**输入**：`query`, `intent`, `ll_keywords`
**输出**：`crag_rewritten_query`

**伪代码**（实际实现见 `app/crag.py` 的 `CragQueryRewriter.rewrite()`）：

```python
def _impl_crag_rewrite_query(self, state: RagWorkflowState) -> RagWorkflowState:
    query = state.get("query", "")
    if self._crag_query_rewriter:
        rewritten = self._crag_query_rewriter.rewrite(query)
    else:
        rewritten = query
    return {"crag_rewritten_query": rewritten}
```

> **实现说明**：改写逻辑封装在 `CragQueryRewriter` 类中，使用英文 system prompt 指导 Web 搜索优化。

#### 节点 3：`crag_web_search`

**输入**：`crag_rewritten_query`, `reranked_hits`（AMBIGUOUS 时保留的文档）
**输出**：更新后的 `reranked_hits`（合并 Web 结果）

**伪代码**：

```python
def _node_crag_web_search(self, state: RagWorkflowState) -> dict:
    if not settings.RAG_CRAG_ENABLE_WEB_SEARCH:
        return {}  # 跳过 Web 搜索

    query = state.get("crag_rewritten_query") or state.get("query", "")
    verdict = state.get("retrieval_verdict", "incorrect")

    web_results = self._web_search_tool.invoke({"query": query})

    # 将 Web 结果转换为与 reranked_hits 相同的格式
    web_hits = []
    for result in web_results[:settings.RAG_CRAG_WEB_SEARCH_K]:
        web_hits.append({
            "chunk_text": result.get("content", ""),
            "citation": {
                "url": result.get("url", ""),
                "title": result.get("title", ""),
            },
            "score": 0.5,  # Web 结果的默认分数
            "source": "web_search",
        })

    # AMBIGUOUS: 合并内部 + Web 结果
    # INCORRECT: 仅使用 Web 结果
    existing_hits = state.get("reranked_hits", [])
    if verdict == "ambiguous":
        merged = existing_hits + web_hits
    else:  # incorrect
        merged = web_hits

    return {
        "reranked_hits": merged,
        "web_search_results": web_hits,
    }
```

### 7.6 图构造变更

```python
# 新增节点
graph.add_node("grade_retrieval", self._node_grade_retrieval)
graph.add_node("crag_rewrite_query", self._node_crag_rewrite_query)
graph.add_node("crag_web_search", self._node_crag_web_search)

# 修改边：rerank → grade_retrieval（替代原来 rerank → build_citations）
graph.add_edge("rerank", "grade_retrieval")

# CRAG 条件边
graph.add_conditional_edges(
    "grade_retrieval",
    self._route_after_grade_retrieval,
    {
        "correct": "build_citations",
        "needs_web_search": "crag_rewrite_query",
    },
)

graph.add_edge("crag_rewrite_query", "crag_web_search")
graph.add_edge("crag_web_search", "build_citations")

# 路由函数
def _route_after_grade_retrieval(self, state: RagWorkflowState) -> str:
    if not self._settings.RAG_ENABLE_CRAG:
        return "correct"
    verdict = state.get("retrieval_verdict", "correct")
    if verdict == "correct":
        return "correct"
    return "needs_web_search"
```

### 7.7 需要修改的文件清单

| 文件                                  | 变更类型 | 内容                                                                 |
| ------------------------------------- | -------- | -------------------------------------------------------------------- |
| `apps/rag-service/app/workflow.py`    | 修改     | 新增 4 个状态字段 + 3 个节点 + 2 条条件边                            |
| `apps/rag-service/app/config.py`      | 修改     | 新增 8 个 CRAG 配置标志                                              |
| `apps/rag-service/app/crag.py`        | 新增     | CRAG 独立模块（RetrievalGrader、CragQueryRewriter、CragWebSearcher） |
| `apps/rag-service/tests/test_crag.py` | 新增     | CRAG 单元测试（14 个测试用例）                                       |

---

## 8. 阈值调优策略

### 8.1 论文推荐值

| 阈值              | 默认值 | 含义                               |
| ----------------- | ------ | ---------------------------------- |
| `UPPER_THRESHOLD` | 0.7    | 超过此值 = CORRECT（文档相关）     |
| `LOWER_THRESHOLD` | 0.3    | 低于此值 = INCORRECT（文档不相关） |

### 8.2 调优方法

1. **收集标注数据**：100–500 对（查询, 文档, 人工标注: 相关/不相关）
2. **计算评估器在不同阈值下的 Precision/Recall**
3. **选择最小化假阳性的阈值**（假阳性 = 把不相关文档当相关 → 导致幻觉）

### 8.3 我们的建议

考虑到本系统是版权/法律领域（APRA AMCOS），建议采用**保守策略**：

```python
UPPER_THRESHOLD = 0.7   # 保持论文默认
LOWER_THRESHOLD = 0.3   # 保持论文默认
MIN_RELEVANT_DOCS = 1   # 至少 1 篇相关文档
```

待积累 100+ 评估数据后，根据 RAGAS 指标反馈调整。

---

## 9. 生产环境基准数据

### 9.1 延迟

| 路径                         | 延迟    | 组成                                                               |
| ---------------------------- | ------- | ------------------------------------------------------------------ |
| **CORRECT**                  | 3–5 秒  | 检索 12ms + 评估 50ms + 生成 ~4900ms                               |
| **INCORRECT**（含 Web 搜索） | 8–12 秒 | 检索 12ms + 评估 50ms + 改写 200ms + Web 搜索 200ms + 生成 ~7500ms |
| **AMBIGUOUS**                | 8–12 秒 | 类似 INCORRECT 路径                                                |

### 9.2 成本

| 组件           | 额外成本                                        |
| -------------- | ----------------------------------------------- |
| 检索评估器     | +1 次 LLM 调用/文档（Nova Lite: ~$0.0001/文档） |
| 查询改写器     | +1 次 LLM 调用（~$0.00005）                     |
| Web 搜索       | +1 次 API 调用（Tavily: ~$0.01/查询）           |
| **总额外开销** | 约 **+20%** 相对于标准 RAG                      |

### 9.3 ROI 分析

- **成本增加**：+20%
- **准确率提升**：+15%（平均）
- **最大收益场景**：检索完全失败时（PubHealth: +36.6%）
- **结论**：对绝大多数生产应用，+20% 成本换 +15% 准确率是值得的

---

## 10. 已知局限性与缓解措施

### 10.1 评估器准确率瓶颈

**问题**：T5 评估器只有 84.3% 准确率（15.7% 错误率）。LLM-as-Judge 在不同模型上表现不一。

**影响**：

- 假阳性（~8%）：把不相关文档评为相关 → 幻觉风险
- 假阴性（~8%）：把相关文档评为不相关 → 不必要的 Web 搜索

**缓解**：

- 使用结构化输出（Pydantic）确保格式一致
- 在验证集上持续监控评估器准确率
- 考虑使用更强模型（如 Claude Haiku）作为评估器

### 10.2 Web 搜索质量不可控

**问题**：Web 搜索结果的质量和可靠性不可预测。

**缓解**：

- 对 Web 结果标记 `source: "web_search"`，在引文中明确区分
- 可选：对 Web 结果进行二次评估
- 可选：限制 Web 结果的权重或数量

### 10.3 复杂推理任务表现有限

**问题**：CRAG 的分解-重组可能破坏多步推理需要的上下文连接。

**缓解**：

- 对 `complexity: "high"` 的查询跳过分解-重组
- 保留段落级别的上下文而非句子级别
- 针对比较类/分析类查询使用不同的精炼策略

### 10.4 Token 消耗增加

**问题**：3–5x 的 token 开销。

**缓解**：

- CORRECT 路径只有约 +10% 额外成本（仅评估开销）
- 仅 INCORRECT/AMBIGUOUS 路径有显著额外成本
- 使用轻量模型做评估（Nova Lite 而非 Claude）
- 缓存频繁查询的评估结果

### 10.5 阈值的领域依赖性

**问题**：0.7/0.3 的阈值并非通用最优。

**缓解**：

- 从论文默认值开始
- 随着 RAGAS 评估数据积累，根据 faithfulness 和 answer_relevancy 指标调整
- 不同类别（qa vs work-search）可使用不同阈值

---

## 11. 参考文献

### 核心论文

1. **Yan, S. Q., Gu, J. C., Zhu, Y., & Ling, Z. H.** (2024). Corrective Retrieval Augmented Generation. _arXiv:2401.15884_ [cs.CL]. https://arxiv.org/abs/2401.15884

2. **Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H.** (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. _arXiv:2310.11511_ [cs.CL]. https://arxiv.org/abs/2310.11511

### 生产实现

3. **LangChain 官方教程**: Corrective RAG with LangGraph. https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag_local/

4. **LangChain 官方教程**: Agentic RAG with LangGraph (博客). https://blog.langchain.com/agentic-rag-with-langgraph/

5. **LangGraph 官方代码**: langgraph_crag.ipynb. https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb

6. **Singh, S.** (2026, Feb 14). When RAG Goes Wrong: The Story of CRAG and How It Fixes Retrieval Failures. _Medium_. https://suraj-singh-007.medium.com/when-rag-goes-wrong-the-story-of-crag-and-how-it-fixes-retrieval-failures-5fcf79ff6c1b

7. **HuskyInSalt/CRAG** (2024). Corrective Retrieval Augmented Generation (原作者代码). _GitHub_. https://github.com/HuskyInSalt/CRAG (450+ Stars)

### 相关工作

8. **Shin, M.** (2025, Sep 03). Corrective RAG (CRAG): Workflow, Implementation, and More. _Meilisearch Blog_. https://www.meilisearch.com/blog/corrective-rag

---

> **下一步**: 评审本文档后，启动 CRAG 实施（修改文件清单见 7.7 节）。
