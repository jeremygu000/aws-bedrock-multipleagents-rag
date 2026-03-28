# LightRAG Migration Plan

> Port [LightRAG](https://github.com/hkuds/lightrag) (EMNLP 2025) techniques into `apps/rag-service/`.
> Three phases: query enhancement → knowledge graph construction → graph-enhanced retrieval.

## 1. Current vs Target Architecture

### Current State

```mermaid
flowchart LR
    Q[User Query] --> DI[detect_intent]
    DI --> RQ[rewrite_query]
    RQ --> BR[build_request]
    BR --> R[retrieve<br/>pgvector + BM25 + RRF]
    R --> BC[build_citations]
    BC --> CM[choose_model]
    CM --> GA[generate_answer<br/>Nova Lite / Qwen Plus]
    GA --> A[Answer]

    style R fill:#4a9eff,color:#fff
    style GA fill:#4a9eff,color:#fff
```

### Target State (Post Phase 3)

```mermaid
flowchart LR
    Q[User Query] --> DI[detect_intent]
    DI --> RQ[rewrite_query]
    RQ --> EK[extract_keywords<br/>hl + ll dual]
    EK --> DM[determine_mode]
    DM --> BR[build_request]
    BR --> R[retrieve<br/>pgvector + BM25 + RRF]
    BR --> GR[graph_retrieve<br/>Neo4j + entity VDB]
    R --> F[fusion]
    GR --> F
    F --> RK[rerank<br/>Qwen Plus LLM]
    RK --> BC[build_citations]
    BC --> CM[choose_model]
    CM --> GA[generate_answer<br/>KG context + chunks]
    GA --> A[Answer]

    style EK fill:#f59e0b,color:#fff
    style DM fill:#f59e0b,color:#fff
    style GR fill:#10b981,color:#fff
    style F fill:#10b981,color:#fff
    style RK fill:#f59e0b,color:#fff
```

> Legend: 🟡 Phase 1 (query enhancement), 🟢 Phase 2-3 (graph retrieval)

## 2. Key Decisions

| Decision                | Choice                     | Rationale                                  |
| ----------------------- | -------------------------- | ------------------------------------------ |
| Entity extraction LLM   | Qwen Plus                  | Already integrated, good structured output |
| Ingestion mode          | Real-time (per upload)     | Simplicity; batch can be added later       |
| Starting phase          | Phase 1                    | Low risk, immediate value                  |
| Primary language        | English                    | Simplifies prompt design, can extend later |
| Graph storage           | Neo4j (existing CDK stack) | Already deployed, unused                   |
| Entity/relation vectors | pgvector (existing)        | No new infra, reuse embedding pipeline     |

## 3. Dependency Map

```mermaid
graph TD
    P1A[1.1 Keyword Extraction] --> P1B[1.2 Query Expansion]
    P1A --> P1D[1.4 Answer Prompt Upgrade]
    P1B --> P1C[1.3 Reranking]
    P1C --> P2[Phase 2]

    P2A[2.1 Entity Extraction] --> P2B["2.2 Dedup & Merge"]
    P2B --> P2C[2.3 Neo4j Storage]
    P2B --> P2D[2.4 Entity Vector Store]
    P2C --> P2E[2.5 Ingestion Pipeline]
    P2D --> P2E
    P2E --> P3[Phase 3]

    P3A[3.1 Graph Retriever] --> P3C[3.3 Hybrid Fusion]
    P3B[3.2 Query Router] --> P3C
    P3C --> P3D[3.4 Enhanced Answer Gen]
    P3D --> P3E[3.5 Community Detection]

    subgraph SG1["Phase 1 — Query Enhancement"]
        P1A
        P1B
        P1C
        P1D
    end

    subgraph SG2["Phase 2 — Knowledge Graph Construction"]
        P2A
        P2B
        P2C
        P2D
        P2E
    end

    subgraph SG3["Phase 3 — Graph-Enhanced Retrieval"]
        P3A
        P3B
        P3C
        P3D
        P3E
    end

    style P1A fill:#f59e0b,color:#fff
    style P1B fill:#f59e0b,color:#fff
    style P1C fill:#f59e0b,color:#fff
    style P1D fill:#f59e0b,color:#fff
    style P2A fill:#3b82f6,color:#fff
    style P2B fill:#3b82f6,color:#fff
    style P2C fill:#3b82f6,color:#fff
    style P2D fill:#3b82f6,color:#fff
    style P2E fill:#3b82f6,color:#fff
    style P3A fill:#10b981,color:#fff
    style P3B fill:#10b981,color:#fff
    style P3C fill:#10b981,color:#fff
    style P3D fill:#10b981,color:#fff
    style P3E fill:#10b981,color:#fff
```

---

## Phase 1: Enhanced Query Understanding + Reranking

**Status**: ✅ Complete.

**Risk**: Low — query-side only, feature-flagged, zero storage changes.

### 1.1 Dual-Level Keyword Extraction

**What**: Replace single-pass intent detection with LightRAG-style dual keyword extraction.

**LightRAG reference**: `keywords_extraction` prompt → `{high_level_keywords: [...], low_level_keywords: [...]}`

```mermaid
sequenceDiagram
    participant W as Workflow
    participant QP as QueryProcessor
    participant LLM as Qwen Plus

    W->>QP: extract_keywords(query)
    QP->>LLM: keywords_extraction prompt
    LLM-->>QP: {hl_keywords: ["market trends"], ll_keywords: ["ACME Corp", "Q3 2024"]}
    QP-->>W: KeywordResult
    Note over W: hl_keywords → stored for Phase 3 graph retrieval<br/>ll_keywords → injected into BM25 boost
```

**File changes**:

| File                       | Change                                                              |
| -------------------------- | ------------------------------------------------------------------- |
| `query_processing.py`      | Add `extract_keywords()` method                                     |
| `models.py`                | Add `KeywordResult(hl_keywords: list[str], ll_keywords: list[str])` |
| `workflow.py`              | Add `extract_keywords` node after `rewrite_query`, extend State     |
| `config.py`                | Add `enable_keyword_extraction: bool = True`                        |
| `test_query_processing.py` | Test keyword JSON parsing, fallback on malformed output             |

**Implementation details**:

- Prompt adapted from LightRAG `keywords_extraction`, tuned for English
- JSON output with `_safe_json()` fallback (existing pattern)
- `ll_keywords` injected into OpenSearch `multi_match` as boosted terms
- `hl_keywords` stored in State for future graph retrieval (Phase 3)

### 1.2 Query Expansion

**What**: Enhance `rewrite_query` to incorporate keyword synonyms and expansions.

**File changes**:

| File                  | Change                                                          |
| --------------------- | --------------------------------------------------------------- |
| `query_processing.py` | Modify `rewrite_query()` to accept `ll_keywords`, expand prompt |
| `workflow.py`         | Update `rewrite_query` node to pass keywords                    |

**Implementation details**:

- Prompt upgrade: instruct LLM to incorporate synonyms/expansions of `ll_keywords`
- Output remains single-line rewritten query (backward compatible)
- Improves recall for abbreviations, acronyms, domain jargon

### 1.3 Retrieval Reranking

**What**: LLM-based reranking of retrieval results with token budget management.

**LightRAG reference**: `process_chunks_unified` — rerank + token budget control.

```mermaid
sequenceDiagram
    participant W as Workflow
    participant RK as LLMReranker
    participant LLM as Qwen Plus

    W->>RK: rerank(query, hits[:20], top_k=5)
    Note over RK: Truncate chunks to fit token budget (30K)
    RK->>LLM: Score each chunk 0-10 for relevance
    LLM-->>RK: [{"chunk_id": "c1", "score": 9}, ...]
    RK-->>W: reranked_hits (sorted by LLM score)
```

**File changes**:

| File               | Change                                                                                        |
| ------------------ | --------------------------------------------------------------------------------------------- |
| `reranker.py`      | **New** — `LLMReranker(settings, qwen_client)`                                                |
| `workflow.py`      | Add `rerank` node between `retrieve` and `build_citations`, extend State with `reranked_hits` |
| `config.py`        | Add `enable_reranking`, `rerank_candidate_count=20`, `rerank_max_tokens=30000`                |
| `test_reranker.py` | **New** — test scoring, budget truncation, error fallback                                     |

**Implementation details**:

- `LLMReranker.rerank(query, hits, top_k)`:
  1. Take top `rerank_candidate_count` hits from retrieval
  2. Truncate chunk text to fit within `rerank_max_tokens` budget
  3. Ask Qwen Plus to score each chunk 0-10 for query relevance
  4. Sort by LLM score, return top_k
- Fallback: if reranking fails, pass through original hits (graceful degradation)

### 1.4 Answer Prompt Upgrade

**What**: Restructure evidence block for future KG context injection.

**File changes**:

| File                  | Change                                            |
| --------------------- | ------------------------------------------------- |
| `answer_generator.py` | Upgrade evidence format, prepare KG context slots |

**Implementation details**:

- Current: flat evidence block `[N] title/url/snippet`
- Upgraded: structured sections (text_chunks now, entities + relations reserved for Phase 3)
- System prompt strengthened: explicit citation grounding, reasoning chain

### Phase 1 — Workflow After

```mermaid
flowchart LR
    DI[detect_intent] --> RQ[rewrite_query]
    RQ --> EK[extract_keywords<br/>★ NEW]
    EK --> BR[build_request<br/>+ keyword boost]
    BR --> R[retrieve]
    R --> RK[rerank<br/>★ NEW]
    RK --> BC[build_citations]
    BC --> CM[choose_model]
    CM --> GA[generate_answer<br/>★ upgraded prompt]

    style EK fill:#f59e0b,color:#fff
    style RK fill:#f59e0b,color:#fff
    style GA fill:#f59e0b,color:#fff
```

### Phase 1 — Feature Flags

| Flag                        | Default | Effect when OFF                           |
| --------------------------- | ------- | ----------------------------------------- |
| `enable_keyword_extraction` | `True`  | Skip keyword node, no BM25 boost          |
| `enable_reranking`          | `True`  | Pass retrieval hits directly to citations |

---

## Phase 2: Knowledge Graph Construction + Neo4j

**Status**: ✅ Complete.

**Risk**: Medium — new storage writes, but query pipeline untouched (feature-flagged).

### 2.1 Entity / Relation Extraction

**What**: LLM-powered extraction from document chunks, following LightRAG's structured output format.

**LightRAG reference**: `extract_entities` + `_process_extraction_result` + gleaning mechanism.

```mermaid
sequenceDiagram
    participant IP as IngestionPipeline
    participant EE as EntityExtractor
    participant LLM as Qwen Plus

    IP->>EE: extract(chunk_text)
    EE->>LLM: entity_extraction_system_prompt
    LLM-->>EE: Raw output with entity/relation tuples
    EE->>EE: _parse_extraction_output()
    Note over EE: Parse delimited tuples:<br/>entity<|#|>name<|#|>type<|#|>desc<br/>relation<|#|>src<|#|>tgt<|#|>kw<|#|>desc

    alt Gleaning enabled (max 1 pass)
        EE->>LLM: entity_continue_extraction_prompt
        LLM-->>EE: Additional entities/relations
        EE->>EE: Merge, keep longer descriptions
    end

    EE-->>IP: ExtractionResult(entities, relations)
```

**File changes**:

| File                        | Change                                                                                      |
| --------------------------- | ------------------------------------------------------------------------------------------- |
| `entity_extraction.py`      | **New** — `EntityExtractor` class with extract + parse + glean                              |
| `prompts.py`                | **New** — centralized prompt templates (extraction, gleaning, summarize)                    |
| `models.py`                 | Add `Entity`, `Relation`, `ExtractionResult` dataclasses                                    |
| `config.py`                 | Add `enable_entity_extraction=False`, `entity_extract_max_gleaning=1`, `entity_types=[...]` |
| `test_entity_extraction.py` | **New** — parsing, gleaning, error recovery                                                 |

**Implementation details**:

- Output format: `entity<|#|>name<|#|>type<|#|>description` / `relation<|#|>source<|#|>target<|#|>keywords<|#|>description`
- Delimiter: `<|#|>` (tuple fields), `<|COMPLETE|>` (end marker)
- Gleaning: 1 extra pass with `entity_continue_extraction_user_prompt` to catch missed entities
- Parser: split by newlines + completion delimiter, validate field counts, skip malformed lines
- Entity types: `person, organization, location, event, concept, technology, document, regulation`

### 2.2 Entity Dedup & Description Merge

**What**: Merge duplicate entities across chunks with LLM-assisted description summarization.

**LightRAG reference**: `merge_nodes_and_edges` + `summarize_entity_descriptions`.

```mermaid
flowchart TD
    E1[Entity: ACME Corp<br/>desc: 'A tech company'] --> M{Same name?}
    E2[Entity: ACME Corp<br/>desc: 'Founded in 2010, leading AI firm'] --> M
    M -->|Yes| MERGE[merge_entities]
    MERGE --> CHECK{Combined desc<br/>> 500 tokens?}
    CHECK -->|No| KEEP[Keep longer description]
    CHECK -->|Yes| LLM[LLM summarize_entity_descriptions]
    LLM --> FINAL[Merged Entity<br/>consolidated description]
    KEEP --> FINAL
```

**File changes**:

| File                   | Change                                      |
| ---------------------- | ------------------------------------------- |
| `entity_extraction.py` | Add `merge_entities()`, `merge_relations()` |
| `config.py`            | Add `summary_max_tokens=500`                |

**Implementation details**:

- `merge_entities(existing, new)`: same-name → keep longer desc or LLM summarize if > `summary_max_tokens`
- `merge_relations(existing, new)`: same source+target → accumulate weight, merge descriptions
- Track `source_chunk_ids` through merges for provenance

### 2.3 Neo4j Storage Layer

**What**: Graph repository connecting to the existing CDK-deployed Neo4j instance.

**LightRAG reference**: `neo4j_impl.py` — schema, indexes, APOC traversal.

```mermaid
erDiagram
    Entity {
        string entity_id PK
        string name
        string type
        text description
        string[] source_chunk_ids
        datetime created_at
        datetime updated_at
    }

    Relation {
        float weight
        text description
        string keywords
        string[] source_chunk_ids
        datetime created_at
    }

    Entity ||--o{ Relation : RELATES_TO
```

**File changes**:

| File                       | Change                                                                |
| -------------------------- | --------------------------------------------------------------------- |
| `graph_repository.py`      | **New** — `Neo4jRepository` with CRUD + traversal                     |
| `config.py`                | Add `neo4j_uri`, `neo4j_username`, `neo4j_password`, `neo4j_database` |
| `pyproject.toml`           | Add `neo4j` driver dependency                                         |
| `test_graph_repository.py` | **New** — mock-based unit tests                                       |

**Implementation details**:

- Write ops: `upsert_entity` (MERGE on name), `upsert_relation` (MERGE on src+tgt), `upsert_batch` (UNWIND)
- Read ops: `get_entity`, `get_entity_neighbors(depth=1)`, `get_relations_for_entities`, `search_entities_fulltext`
- Indexes: B-Tree on `entity_id`, Full-text on `name`
- Connection: lazy driver init, health check method
- Secrets: Neo4j password from AWS Secrets Manager (matching CDK stack)

### 2.4 Entity Vector Store

**What**: Store entity and relation embeddings in pgvector for semantic search.

**LightRAG reference**: `entities_vdb`, `relationships_vdb` — separate vector indexes.

**File changes**:

| File                     | Change                                             |
| ------------------------ | -------------------------------------------------- |
| `entity_vector_store.py` | **New** — `EntityVectorStore` with upsert + search |
| DB migration script      | **New** — `kb_entities` + `kb_relations` tables    |

**Schema**:

```sql
CREATE TABLE kb_entities (
    entity_id    TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    type         TEXT NOT NULL,
    description  TEXT,
    embedding    vector(1024),
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE kb_relations (
    relation_id   TEXT PRIMARY KEY,
    source_name   TEXT NOT NULL,
    target_name   TEXT NOT NULL,
    keywords      TEXT,
    description   TEXT,
    embedding     vector(1024),
    created_at    TIMESTAMPTZ DEFAULT now()
);
```

**Implementation details**:

- Entity embedding: `embed(name + " " + type + " " + description)` via Qwen embedding
- Relation embedding: `embed(keywords + " " + description)` via Qwen embedding
- Search: pgvector cosine distance (`<=>`) with top_k

### 2.5 Ingestion Pipeline

**What**: Real-time document ingestion that extracts entities/relations and writes to graph + vector stores.

```mermaid
sequenceDiagram
    participant API as FastAPI Upload
    participant IP as IngestionPipeline
    participant EE as EntityExtractor
    participant MG as Merge Logic
    participant Neo as Neo4jRepository
    participant PG as EntityVectorStore

    API->>IP: ingest_document(doc_id, chunks)

    loop For each chunk (max 4 concurrent)
        IP->>EE: extract(chunk_text)
        EE-->>IP: ExtractionResult
    end

    IP->>MG: merge all entities & relations
    MG-->>IP: deduplicated entities & relations

    par Write to stores
        IP->>Neo: upsert_batch(entities, relations)
        IP->>PG: upsert_entity_embeddings(entities)
        IP->>PG: upsert_relation_embeddings(relations)
    end

    IP-->>API: IngestionResult(entity_count, relation_count)
```

**File changes**:

| File                | Change                                     |
| ------------------- | ------------------------------------------ |
| `ingestion.py`      | **New** — `IngestionPipeline` orchestrator |
| `config.py`         | Add `ingestion_max_concurrent=4`           |
| `test_ingestion.py` | **New** — end-to-end with mocked services  |

**Implementation details**:

- `ingest_document(doc_id, chunks)`:
  1. `asyncio.gather` with semaphore (max 4 concurrent) per-chunk extraction
  2. Collect all entities + relations across chunks
  3. Dedup/merge pass
  4. Parallel write: Neo4j batch + pgvector batch
- Error handling: single chunk failure logged as warning, doesn't block others
- Integrated at document upload API endpoint

### Phase 2 — Data Flow

```mermaid
flowchart TD
    DOC[Document Upload] --> CHUNK[Chunking<br/>existing pipeline]
    CHUNK --> EXT[Entity Extraction<br/>Qwen Plus]
    EXT --> MERGE["Dedup & Merge"]
    MERGE --> NEO[Neo4j<br/>entities + relations]
    MERGE --> PGV[pgvector<br/>entity embeddings]
    MERGE --> PGR[pgvector<br/>relation embeddings]

    CHUNK --> EXIST[Existing Pipeline<br/>kb_chunks + BM25 index]

    style EXT fill:#3b82f6,color:#fff
    style MERGE fill:#3b82f6,color:#fff
    style NEO fill:#3b82f6,color:#fff
    style PGV fill:#3b82f6,color:#fff
    style PGR fill:#3b82f6,color:#fff
```

### Phase 2 — Completion Criteria

- [x] Document upload triggers entity/relation extraction
- [x] Entities and relations visible in Neo4j
- [x] Entity/relation embeddings in pgvector
- [x] Existing query pipeline completely unaffected
- [x] All new code behind `enable_entity_extraction` flag

---

## Phase 3: Graph-Enhanced Retrieval + Multi-Mode Search

**Status**: ✅ Complete (3.1–3.4). 3.5 Community Detection deferred.

**Risk**: Medium-high — modifies query pipeline, but feature-flagged with NAIVE fallback.

### 3.1 Graph Retriever

**What**: Query the knowledge graph using dual-level keywords from Phase 1.

**LightRAG reference**: `_build_query_context` — local/global/hybrid retrieval modes.

```mermaid
flowchart TD
    subgraph LR_SUB["Local Retrieval — Entity-Oriented"]
        LK[ll_keywords] --> FT[Neo4j fulltext search]
        QE[query_embedding] --> EV[entities_vdb cosine search]
        FT --> EM[Entity merge + dedup]
        EV --> EM
        EM --> N1[1-hop neighbors]
        N1 --> RE[Related edges + chunks]
    end

    subgraph GR_SUB["Global Retrieval — Theme-Oriented"]
        HK[hl_keywords] --> RV[relations_vdb cosine search]
        QE2[query_embedding] --> RV2[relations_vdb cosine search]
        RV --> RM[Relation merge]
        RV2 --> RM
        RM --> RC[Related chunks via source_ids]
    end

    RE --> CTX[GraphContext<br/>entities + relations + chunks]
    RC --> CTX
```

**File changes**:

| File                 | Change                                                      |
| -------------------- | ----------------------------------------------------------- |
| `graph_retriever.py` | **New** — `GraphRetriever` with local/global/hybrid methods |
| `models.py`          | Add `GraphContext`, `RetrievalMode` enum                    |

**Implementation details**:

- `retrieve_local(ll_keywords, query_embedding, top_k)`:
  1. Fulltext search Neo4j entities by `ll_keywords`
  2. Cosine search `kb_entities` by `query_embedding`
  3. Merge + dedup entity results
  4. For each entity: get 1-hop neighbors + relations from Neo4j
  5. Collect `source_chunk_ids` from relations → fetch original chunks
  6. Return `GraphContext(entities, relations, chunk_ids)`

- `retrieve_global(hl_keywords, query_embedding, top_k)`:
  1. Cosine search `kb_relations` by `query_embedding`
  2. Collect high-level relations matching `hl_keywords`
  3. Fetch related chunks via `source_chunk_ids`
  4. Return `GraphContext(entities, relations, chunk_ids)`

- `retrieve_hybrid`: union of local + global results
- Token budget: cap `entities_str + relations_str + chunks_str` per LightRAG defaults

### 3.2 Intelligent Query Router

**What**: Automatically select retrieval mode based on query characteristics.

**LightRAG reference**: 6 query modes (local/global/hybrid/naive/mix/bypass).

```mermaid
flowchart TD
    Q[Query Analysis] --> INT{Intent?}
    INT -->|factual + low complexity| NAIVE[NAIVE<br/>pgvector + BM25 only]
    INT -->|specific entities| LOCAL[LOCAL<br/>entity-oriented graph]
    INT -->|analytical / trends| GLOBAL[GLOBAL<br/>theme-oriented graph]
    INT -->|high complexity| HYBRID[HYBRID<br/>local + global]
    INT -->|default| MIX[MIX<br/>traditional + graph fusion]

    style NAIVE fill:#6b7280,color:#fff
    style LOCAL fill:#3b82f6,color:#fff
    style GLOBAL fill:#8b5cf6,color:#fff
    style HYBRID fill:#10b981,color:#fff
    style MIX fill:#f59e0b,color:#fff
```

**File changes**:

| File                  | Change                                             |
| --------------------- | -------------------------------------------------- |
| `query_processing.py` | Add `determine_retrieval_mode()`                   |
| `workflow.py`         | Add `determine_mode` node after `extract_keywords` |

**Routing logic**:

| Mode     | Condition                                   | Retrieval Channels         |
| -------- | ------------------------------------------- | -------------------------- |
| `NAIVE`  | Low complexity + factual intent             | pgvector + BM25 (existing) |
| `LOCAL`  | Clear `ll_keywords`, entity-specific query  | Entity graph + existing    |
| `GLOBAL` | Analytical intent, `hl_keywords` dominant   | Relation graph + existing  |
| `HYBRID` | High complexity, both keyword types present | Local + Global + existing  |
| `MIX`    | Default / uncertain                         | All channels, RRF fusion   |

### 3.3 Hybrid Retrieval Fusion

**What**: Merge graph-based and traditional retrieval results.

```mermaid
sequenceDiagram
    participant W as Workflow
    participant TR as PostgresRepository
    participant GR as GraphRetriever
    participant F as Fusion

    par Parallel retrieval
        W->>TR: retrieve(request)
        TR-->>W: traditional_hits
    and
        W->>GR: retrieve_by_mode(mode, keywords, embedding)
        GR-->>W: GraphContext(entities, relations, chunk_ids)
    end

    W->>TR: get_chunks_by_ids(graph_context.chunk_ids)
    TR-->>W: graph_chunk_hits

    W->>F: fuse(traditional_hits, graph_chunk_hits, weight=0.6)
    Note over F: Weighted RRF:<br/>graph hits get 0.6 weight<br/>traditional hits get 0.4 weight
    F-->>W: fused_hits (with entity/relation context attached)
```

**File changes**:

| File            | Change                                                                                   |
| --------------- | ---------------------------------------------------------------------------------------- |
| `workflow.py`   | Modify `retrieve` node: parallel traditional + graph, add fusion                         |
| `repository.py` | Add `get_chunks_by_ids(chunk_ids)` method                                                |
| `config.py`     | Add `enable_graph_retrieval=False`, `graph_retrieval_weight=0.6`, `retrieval_mode="mix"` |

### 3.4 Enhanced Answer Generation

**What**: Feed KG context (entities + relations) alongside chunks into answer generation.

**LightRAG reference**: `rag_response` prompt + `kg_query_context` template.

```mermaid
flowchart LR
    subgraph Context Assembly
        E[Entities<br/>name, type, desc] --> CTX[Structured Context]
        R["Relations<br/>src→tgt, keywords"] --> CTX
        C[Text Chunks<br/>with citations] --> CTX
    end

    CTX --> SYS[System Prompt<br/>+ KG reasoning instructions]
    SYS --> LLM[Qwen Plus / Nova Lite]
    LLM --> ANS[Answer with<br/>multi-hop reasoning<br/>+ citations]
```

**File changes**:

| File                  | Change                                                                |
| --------------------- | --------------------------------------------------------------------- |
| `answer_generator.py` | Three-section context: entities_str + relations_str + text_chunks_str |
| `prompts.py`          | Add `kg_query_context` template, upgrade `rag_response`               |

**Context format** (injected into answer prompt):

```
=== ENTITIES ===
- ACME Corp [Organization]: Leading AI company founded in 2010...
- John Smith [Person]: CTO of ACME Corp since 2018...

=== RELATIONS ===
- John Smith → ACME Corp [works_at]: Appointed CTO in 2018...
- ACME Corp → Project X [develops]: Flagship AI product...

=== TEXT EVIDENCE ===
[1] Source Title | URL | Date
Relevant chunk text...

[2] Source Title | URL | Date
Relevant chunk text...
```

### 3.5 Community Detection (Optional Enhancement)

**What**: Periodic Louvain community detection on the Neo4j graph for thematic clustering.

**LightRAG reference**: `python-louvain` for community detection (not LLM-based).

**File changes**:

| File             | Change                                     |
| ---------------- | ------------------------------------------ |
| `community.py`   | **New** — periodic community detection job |
| `pyproject.toml` | Add `python-louvain` dependency (optional) |

**Implementation details**:

- Scheduled job (not real-time): export Neo4j subgraph → Louvain clustering → write community labels back
- Global retrieval can aggregate by community for better thematic grouping
- Low priority — implement after core Phase 3 features are validated

### Phase 3 — Final Workflow

```mermaid
flowchart TD
    Q[User Query] --> DI[detect_intent]
    DI --> RQ[rewrite_query]
    RQ --> EK[extract_keywords]
    EK --> DM[determine_mode]
    DM --> BR[build_request]

    BR --> R[retrieve<br/>pgvector + BM25]
    BR --> GR[graph_retrieve<br/>Neo4j + entity VDB]

    R --> FUSE[fusion<br/>weighted RRF]
    GR --> FUSE

    FUSE --> RK[rerank<br/>Qwen Plus]
    RK --> BC[build_citations]
    BC --> CM[choose_model]
    CM --> GA[generate_answer<br/>KG context + chunks]
    GA --> A[Answer]

    style DI fill:#6b7280,color:#fff
    style RQ fill:#6b7280,color:#fff
    style EK fill:#f59e0b,color:#fff
    style DM fill:#10b981,color:#fff
    style BR fill:#6b7280,color:#fff
    style R fill:#4a9eff,color:#fff
    style GR fill:#10b981,color:#fff
    style FUSE fill:#10b981,color:#fff
    style RK fill:#f59e0b,color:#fff
    style BC fill:#6b7280,color:#fff
    style CM fill:#6b7280,color:#fff
    style GA fill:#f59e0b,color:#fff
```

> Legend: ⚪ Existing, 🟡 Phase 1, 🟢 Phase 3, 🔵 Existing storage

---

## Cross-Phase Summary

### New Files (All Phases)

| Phase | New File                      | Purpose                                         |
| ----- | ----------------------------- | ----------------------------------------------- |
| 1     | `reranker.py`                 | LLM-based retrieval reranking                   |
| 1     | `test_reranker.py`            | Reranker tests                                  |
| 2     | `entity_extraction.py`        | Entity/relation extraction from chunks          |
| 2     | `prompts.py`                  | Centralized prompt templates                    |
| 2     | `graph_repository.py`         | Neo4j CRUD + traversal                          |
| 2     | `entity_vector_store.py`      | Entity/relation embeddings in pgvector          |
| 2     | `ingestion.py`                | Real-time ingestion orchestrator                |
| 2     | `test_entity_extraction.py`   | Extraction tests                                |
| 2     | `test_graph_repository.py`    | Neo4j tests                                     |
| 2     | `test_entity_vector_store.py` | Vector store tests                              |
| 2     | `test_ingestion.py`           | End-to-end ingestion tests                      |
| 2     | DB migration                  | `kb_entities` + `kb_relations` tables           |
| 3     | `graph_retriever.py`          | Graph-based retrieval engine                    |
| 3     | `hybrid_fusion.py`            | Weighted RRF fusion of graph + traditional hits |
| 3     | `community.py`                | Optional community detection                    |
| 3     | `test_graph_retriever.py`     | Graph retrieval tests                           |
| 3     | `test_hybrid_fusion.py`       | Hybrid fusion tests                             |

### Modified Files (All Phases)

| File                  | Phase 1                                      | Phase 2                                  | Phase 3                                  |
| --------------------- | -------------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| `query_processing.py` | `extract_keywords()`, expand `rewrite_query` | —                                        | `determine_retrieval_mode()`             |
| `models.py`           | `KeywordResult`                              | `Entity`, `Relation`, `ExtractionResult` | `GraphContext`, `RetrievalMode`          |
| `workflow.py`         | `extract_keywords` + `rerank` nodes          | —                                        | `determine_mode` node, parallel retrieve |
| `config.py`           | keyword + rerank flags                       | Neo4j + extraction config                | graph retrieval config                   |
| `answer_generator.py` | prompt upgrade                               | —                                        | KG context injection                     |
| `repository.py`       | —                                            | —                                        | `get_chunks_by_ids()`                    |
| `lambda_tool.py`      | —                                            | —                                        | `_build_graph_retriever()` factory       |
| `pyproject.toml`      | —                                            | `neo4j` driver                           | `python-louvain` (optional)              |

### Feature Flags

| Flag                        | Phase | Default | Rollback Effect                         |
| --------------------------- | ----- | ------- | --------------------------------------- |
| `enable_keyword_extraction` | 1     | `True`  | Skip keyword node                       |
| `enable_reranking`          | 1     | `True`  | Pass hits directly to citations         |
| `enable_entity_extraction`  | 2     | `False` | Skip ingestion extraction               |
| `enable_graph_retrieval`    | 3     | `False` | Fall back to traditional retrieval only |

### New Dependencies

| Package          | Phase        | Purpose             |
| ---------------- | ------------ | ------------------- |
| `neo4j`          | 2            | Neo4j Python driver |
| `python-louvain` | 3 (optional) | Community detection |

### Risk & Rollback

| Phase | Risk        | Rollback                                                         |
| ----- | ----------- | ---------------------------------------------------------------- |
| 1     | Low         | Toggle feature flags off                                         |
| 2     | Medium      | Disable `enable_entity_extraction`, clear Neo4j data             |
| 3     | Medium-high | Disable `enable_graph_retrieval`, falls back to Phase 1 behavior |

### Storage Architecture (Post Phase 3)

```mermaid
flowchart TD
    subgraph Existing Storage
        PG[(PostgreSQL<br/>pgvector)]
        OS[(OpenSearch<br/>BM25)]
    end

    subgraph NS["New Storage — Phase 2"]
        NEO[(Neo4j<br/>entities + relations)]
        PGE[(pgvector<br/>kb_entities)]
        PGR[(pgvector<br/>kb_relations)]
    end

    subgraph Query Pipeline
        TRAD[Traditional Retrieve<br/>chunks + BM25] --> PG
        TRAD --> OS
        GRAPH[Graph Retrieve<br/>entities + relations] --> NEO
        GRAPH --> PGE
        GRAPH --> PGR
    end

    TRAD --> FUSE[Fusion + Rerank]
    GRAPH --> FUSE
    FUSE --> ANS[Answer Generation]

    style NEO fill:#10b981,color:#fff
    style PGE fill:#3b82f6,color:#fff
    style PGR fill:#3b82f6,color:#fff
```
