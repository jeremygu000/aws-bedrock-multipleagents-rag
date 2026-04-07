# RAGAS Reference Ground Truth 编写指南

本文档规范 `scripts/examples/agent-eval.example.jsonl` 中 `reference` 字段的编写标准，确保 RAGAS `factual_correctness` 和 `semantic_similarity` 评分准确、公平、可重复。

## 评分原理

RAGAS `factual_correctness` 将 response 和 reference 都拆解为 **claims（事实断言）**，然后计算：

- **Precision** = response claims 中有多少与 reference 匹配
- **Recall** = reference claims 中有多少被 response 覆盖
- **Score** = F1(Precision, Recall)

因此，reference 的写法直接决定评分上限和公平性。

---

## 1. 长度规范

| 长度              | 效果                              | 建议        |
| ----------------- | --------------------------------- | ----------- |
| < 150 chars       | 只有 1-2 个 claim，分数天花板极低 | ❌ 避免     |
| **200-500 chars** | **5-10 个 claim，覆盖面合理**     | **✅ 推荐** |
| > 600 chars       | 引入过多细节，模型未提到就被扣分  | ❌ 避免     |

**目标：3-6 句话，200-500 字符。**

## 2. Claim 密度

每句话应包含 1-2 个可验证的事实断言。

### 差的例子

```
"APRA AMCOS is the Australian collecting society representing songwriters."
```

→ 仅 2 个 claims：(1) Australian collecting society (2) representing songwriters

### 好的例子

```
"APRA AMCOS is the trading name of Australasian Performing Right Association Limited
and Australasian Mechanical Copyright Owners Society Limited. It represents over
124,000 songwriters, composers, and music publishers. APRA manages performing rights
for public performance, while AMCOS manages mechanical rights for reproduction of music.
APRA's head office is in Ultimo, New South Wales."
```

→ 8+ claims 可匹配

## 3. 内容来源规则

### ✅ 必须

- **只写知识库（ES/pgvector）里实际存在的信息**
- 可参考 baseline 和 Haiku 的实际回答来确认知识库覆盖范围
- 实际回答在 `tmp/evals/with-docs-ragas.jsonl`（baseline）和 `tmp/evals/haiku45-ragas.jsonl`（Haiku 4.5）

### ❌ 禁止

- 不要写知识库里不存在的细节（即使你知道正确答案）
- 不要写主观判断（"最好的"、"最重要的"）
- 不要写模型不可能从现有证据推断出的信息

### 为什么

如果 reference 包含知识库没有的信息，模型不可能回答，recall 必然为 0 → 分数被不公平地压低。

## 4. 具体事实优先

优先使用可精确匹配的事实，而非模糊概括：

| 类型 | 差的写法                | 好的写法                                                          |
| ---- | ----------------------- | ----------------------------------------------------------------- |
| 名称 | "the organisation"      | "Australasian Performing Right Association Limited"               |
| 数字 | "many members"          | "over 124,000 members"                                            |
| 地点 | "offices in Australia"  | "head office in Ultimo, New South Wales"                          |
| 日期 | "regular distributions" | "quarterly distributions in March, June, September, and December" |
| 类别 | "various licences"      | "blanket licences, transactional licences, and event licences"    |

## 5. 覆盖模型的回答方向

1. 查看 baseline 实际回答了 A、B、C 三个方面
2. 查看 Haiku 实际回答了 A、B、D 三个方面
3. Reference 应至少覆盖 A、B、C、D 所有方面
4. 可额外补充知识库里有但模型没提到的关键事实

## 6. 多答案/多来源情况

当知识库中多个页面包含相关信息时：

| 情况                     | 处理方式                                          |
| ------------------------ | ------------------------------------------------- |
| 3 个页面说不同的事实     | **合并**所有事实到一条 reference                  |
| 3 个页面说重复的内容     | **去重**，只保留最详细的版本                      |
| 存在矛盾信息             | 写最权威/最新的版本                               |
| 模型可能只回答其中一部分 | 没关系 — reference 写全，模型部分回答也能拿部分分 |

**原理**：reference 写得全是安全的。模型回答部分内容时，recall 略低但 precision 不受影响。这比 reference 太短导致 precision 被压低好得多。

## 7. Work-Search 行不需要改

`agent-eval.example.jsonl` 中 `metadata.category: "work-search"` 的行（通常是 Row 23-30）使用规则评估（`eval-work-search.ts`），不跑 `factual_correctness`。它们的 `reference` 字段仅用于 `semantic_similarity`，无需按本指南修改。

## 8. 编写 Checklist

每条 reference 改完后逐项检查：

- [ ] 长度 200-500 chars？
- [ ] 3-6 句话？
- [ ] 每句包含具体事实（名称/数字/日期/地点）？
- [ ] 所有事实都在知识库里存在？
- [ ] 覆盖了模型实际会回答的主要方面？
- [ ] 没有知识库里不存在的信息？
- [ ] 没有主观判断或评价？
- [ ] 多来源信息已合并去重？

## 9. 改写示例

### Row 1: "who is APRA AMCOS"

**改前** (96 chars):

```
APRA AMCOS is the Australian collecting society representing songwriters, composers, and music publishers.
```

**改后** (430 chars):

```
APRA AMCOS is the trading name of Australasian Performing Right Association Limited and Australasian Mechanical Copyright Owners Society Limited. It is a music rights management organisation that pays royalties to music creators when their music is played or copied, both locally and overseas. APRA AMCOS represents over 124,000 songwriters, composers, and music publishers. APRA's head office is in Ultimo, New South Wales, with five other offices across Australia and one in Auckland, New Zealand.
```

### Row 16: "what is a mechanical right"

**改前** (198 chars):

```
A mechanical right is the right to reproduce a musical work in a physical or digital format such as CDs, vinyl, downloads, or ringtones. AMCOS manages mechanical rights on behalf of its members, collecting fees when music is copied.
```

**改后** (350 chars):

```
A mechanical right is the right to reproduce a musical work in a physical or digital format such as CDs, vinyl, downloads, or ringtones. AMCOS (Australasian Mechanical Copyright Owners Society Limited) manages mechanical rights on behalf of its members. Mechanical rights relate to the underlying musical work, as opposed to sound recordings. AMCOS collects fees when music is copied or reproduced.
```

---

## 10. 参考文件

| 文件                                        | 用途                               |
| ------------------------------------------- | ---------------------------------- |
| `scripts/examples/agent-eval.example.jsonl` | 正式评估数据集（修改后复制回此处） |
| `tmp/evals/agent-eval-to-review.jsonl`      | 编辑用副本                         |
| `tmp/evals/with-docs-ragas.jsonl`           | Baseline (Nova 2 Lite) 实际回答    |
| `tmp/evals/haiku45-ragas.jsonl`             | Haiku 4.5 实际回答                 |
| `tmp/evals/with-docs-ragas-results.json`    | Baseline RAGAS 分数（per-row）     |
| `tmp/evals/haiku45-ragas-results.json`      | Haiku 4.5 RAGAS 分数（per-row）    |
| `scripts/ragas_eval.py`                     | RAGAS 评估脚本                     |
