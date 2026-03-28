# RAG Caching Research

> Advanced caching techniques for retrieval-augmented generation systems, with focus on LightRAG's production implementation and recommendations for our project.

## 1. RAG Caching Landscape

### Production Caching Model

RAG systems benefit from a layered caching approach targeting different bottlenecks:

1. **Query Result Cache**: Full RAG results cached by query hash. Fast re-queries of identical questions.
2. **Embedding Cache**: Embedding computation results cached by text content. Reduces embedding API calls during retrieval.
3. **Retrieval Cache**: Ranked retrieval results cached per query context. Avoids re-ranking identical document sets.
4. **Generation Cache**: LLM generation results cached by prompt+context. Highest ROI due to LLM cost.

LLM API calls cost 10-100x more than embedding operations, making generation cache the most valuable layer for cost reduction.

### Semantic Caching

The core breakthrough in RAG caching is semantic cache: using embeddings as cache keys instead of string hashing. This enables matching similar but not identical queries to cached results.

Key benefits:

- 5-10x latency improvement (fewer LLM calls)
- 50-90% cost reduction (from avoiding redundant generation)
- Seamless integration into retrieval pipelines

### Reference Implementations

**GPTCache** (Zilliz, ~8k GitHub stars) is the primary open-source semantic caching framework. It provides:

- Pluggable embedding models and vector storage
- Cache-aware retrieval pipeline
- Multi-tier caching (memory + persistent)

**RAGCache** (PKU + ByteDance, arXiv:2404.12457, 2024) introduces knowledge tree structure for efficient cache organization:

- GPU/host memory tiering for fast semantic search
- TTFT (Time To First Token) reduction: 4x on typical RAG workloads
- Throughput improvement: 2.1x on vLLM + Faiss deployments

### Cache Invalidation Strategies

Production systems need robust invalidation to prevent stale results:

- **TTL (Time To Live)**: Layer-specific TTLs (e.g., extraction cache 7 days, query cache 24 hours)
- **Version Tracking (ETag)**: Hash of source document version; update when source changes
- **Delta Indexing**: Log CRUD operations on knowledge graph; invalidate related cache entries
- **Source-based Invalidation**: When source document is re-ingested, invalidate all cache entries derived from it

## 2. LightRAG's Caching Implementation

LightRAG (https://github.com/hkuds/lightrag) is a production RAG system built on knowledge graphs. It implements comprehensive caching at multiple levels, particularly during the graph construction phase where most LLM calls occur.

### Cache Types

LightRAG manages five distinct cache layers:

| Cache Type                   | What                                              | Default | Config                   |
| ---------------------------- | ------------------------------------------------- | ------- | ------------------------ |
| LLM Response Cache (extract) | Entity/relation extraction LLM responses          | On      | `enable_llm_cache`       |
| LLM Response Cache (summary) | Entity/relation summaries                         | On      | `enable_llm_cache`       |
| Query Cache                  | RAG query results (mix/hybrid/local/global modes) | On      | Automatic                |
| Keywords Cache               | Extracted keywords from text                      | On      | Automatic                |
| Embedding Cache              | Semantic similarity lookups                       | Off     | `embedding_cache_config` |

### Core Implementation

The foundation is `use_llm_func_with_cache()` in `lightrag/utils.py` (lines 1947-2086), which wraps LLM calls with caching logic:

- **Cache Key**: Hash of `(prompt, system_prompt, history)`
- **Prefixes**: Cache entries use namespace prefixes (`default:extract:*`, `default:summary:*`, `mix:query:*`, `hybrid:query:*`)
- **Atomic Operations**: Supports retrieval and storage without race conditions

### Graph Reconstruction

LightRAG can rebuild knowledge graphs from cached extraction results via `rebuild_knowledge_graph_from_cached_extraction_results()`. This enables:

- Offline analysis of previously extracted entities/relations
- Bulk re-processing without re-running LLM extraction
- Cost-effective graph refinement

### Storage Backends

Cache is storage-agnostic, supporting multiple backends:

- JSON (local development)
- Redis (shared caching, fast reads)
- PostgreSQL (persistent, queryable)
- MongoDB (document-oriented)
- OpenSearch (full-text search on cache metadata)

### Embedding Cache Configuration

When enabled, embedding cache operates at `similarity_threshold: 0.95` (95% semantic match), with optional secondary LLM verification:

- Reduces embedding API redundancy
- Can be tuned based on accuracy vs. cost trade-off

### CLI and API Tools

LightRAG provides operational tools:

- `migrate_llm_cache`: Migrate cache between backends
- `clean_llm_query_cache`: Trim old/unused query cache entries
- `download_cache`: Export cache for analysis
- `aclear_cache()`: Async cache purge API

### Unique Advantage

LightRAG's caching design is optimized for graph construction phase caching, not just query-time caching. This means:

- Entity extraction runs once, results cached permanently (unless source invalidates)
- Subsequent re-builds use cache, avoiding re-LLM'ing the same chunks
- Orders of magnitude cost savings on typical RAG workloads

## 3. Recommendations for Our Project

Our RAG service (`apps/rag-service`) should apply a "quality over quantity" principle: only implement caches with clear ROI and operational maturity. This avoids cache bloat and stale data issues.

### Priority Matrix

| Priority      | Cache Type                        | ROI        | Effort | Timeline | Implementation                      |
| ------------- | --------------------------------- | ---------- | ------ | -------- | ----------------------------------- |
| 1 (Highest)   | LLM Response Cache (extraction)   | Very High  | Low    | Phase 2  | Feature flag `RAG_ENABLE_LLM_CACHE` |
| 2 (Medium)    | Query-level Semantic Cache        | Medium     | Medium | Phase 3  | Semantic matching on query context  |
| 3 (Long-term) | Embedding Cache (ingestion dedup) | Low-Medium | Medium | Phase 4+ | Dedup during chunk processing       |

### Priority 1: LLM Response Cache for Extraction

**Rationale**: Our RAG pipeline extracts entities, relationships, and summaries during knowledge graph construction. The same chunk content should never be LLM'd twice.

- Cache key: `hash(prompt_template_version + chunk_content + model_id)`
- Applies to both entity extraction and summary generation
- Directly ports LightRAG's `enable_llm_cache` pattern
- Estimated cost reduction: 70-80% on typical ingestion workloads (no re-LLM for same chunks)

### Priority 2: Query-level Semantic Cache

**Rationale**: Similar user queries should return cached results. Semantic similarity matching (not string matching) is required.

- Cache key: `embedding(query_text)` with 0.92+ similarity threshold
- Applies to RAG query phase (after retrieval, before generation)
- Estimated latency improvement: 5-10x on common queries
- Implementation: Use embedding model already in pipeline for similarity matching

### Priority 3: Embedding Cache (Ingestion Dedup)

**Rationale**: During bulk ingestion, same text chunks may appear in multiple documents. Skip redundant embedding calls.

- Cache key: `hash(chunk_text)`
- Applies during chunk embedding phase
- Lower ROI than extraction cache (embedding is 10-100x cheaper than LLM)
- Deferred to Phase 4+ once extraction cache is proven

### Implementation Pattern

All caching should follow project conventions:

- Feature flags with defaults off for Phase 2+ features
- Configuration via environment variables (e.g., `RAG_ENABLE_LLM_CACHE`)
- Pluggable backends (start with file-based, move to PostgreSQL for production)

## 4. Implementation Notes

### Cache Key Strategy

For LLM response caching during extraction:

```
cache_key = hash(f"{prompt_template_version}:{chunk_content}:{model_id}")
```

This ensures cache hits only when all three factors match. Changing prompt versions invalidates old cache automatically (by design).

### Storage Backend Selection

Development phase:

- File-based JSON cache (local disk)
- Suitable for development and testing
- No external dependencies

Production phase:

- PostgreSQL (leverage existing RDS)
- Queryable cache metadata (for analytics)
- Automatic cleanup via SQL queries
- Shared access across Lambda instances

### Invalidation Strategy

Source-document-based invalidation:

1. Track source document version/ETag (hash of file content)
2. When source is re-ingested with new version, invalidate all cache entries with `source_version = old_version`
3. Re-extraction populates new cache entries with new version
4. Automatic TTL cleanup for orphaned entries (e.g., 30-day retention)

### Feature Flag Convention

Following project conventions, implement as:

```
RAG_ENABLE_LLM_CACHE=false  # Default off until Phase 2
RAG_CACHE_BACKEND=file      # file | postgres | redis
RAG_CACHE_TTL_DAYS=30       # Cache lifetime before cleanup
```

Gradual rollout:

1. Phase 2a: Internal testing with flag off
2. Phase 2b: Canary test with 5% traffic
3. Phase 2c: Gradual rollout to 100%

### Monitoring Metrics

When implementing caching, track:

- Cache hit rate (% of requests served from cache)
- Cache miss rate and miss latency
- Cache size (GB)
- Savings: estimated LLM cost avoided
- Stale data incidents (cache invalidation failures)

Monitor separately per cache type to isolate performance bottlenecks.

## References

- LightRAG: https://github.com/hkuds/lightrag
- GPTCache: https://github.com/zilliztech/GPTCache
- RAGCache: https://arxiv.org/abs/2404.12457 (arXiv:2404.12457, 2024)
