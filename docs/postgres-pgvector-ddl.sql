-- Hybrid RAG storage schema for Amazon RDS PostgreSQL + pgvector.
-- Citation policy in this draft is strict by default:
-- every chunk must have citation URL + year/month + at least one locator.
-- This file is a review draft, not an automatic migration.

-- Required extensions:
-- - vector: dense embedding search
-- - pgcrypto: gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Optional extension for more flexible text matching in future.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Run-level provenance for crawler/file ingestion jobs.
CREATE TABLE IF NOT EXISTS ingestion_runs (
  run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source_type TEXT NOT NULL CHECK (source_type IN ('crawler', 'file')),
  status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
  config JSONB NOT NULL DEFAULT '{}'::jsonb,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ,
  notes TEXT
);

-- Canonical document records with metadata for filtering, tracing, and versioning.
CREATE TABLE IF NOT EXISTS kb_documents (
  doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source_type TEXT NOT NULL CHECK (source_type IN ('crawler', 'file')),
  source_uri TEXT NOT NULL,
  canonical_uri TEXT,
  title TEXT NOT NULL,
  lang TEXT NOT NULL DEFAULT 'en',
  category TEXT NOT NULL DEFAULT 'general',
  mime_type TEXT NOT NULL,
  author TEXT,
  tags TEXT[] NOT NULL DEFAULT '{}',
  access_level TEXT NOT NULL DEFAULT 'internal',
  content_hash TEXT NOT NULL,
  doc_version TEXT NOT NULL,
  crawl_time TIMESTAMPTZ,
  published_at TIMESTAMPTZ,
  published_year INTEGER NOT NULL CHECK (published_year BETWEEN 1000 AND 2100),
  published_month INTEGER NOT NULL CHECK (published_month BETWEEN 1 AND 12),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  run_id UUID REFERENCES ingestion_runs(run_id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_kb_documents_source_version UNIQUE (source_uri, doc_version),
  CONSTRAINT ck_kb_documents_metadata_year_month CHECK (
    CASE
      WHEN
        metadata ? 'year'
        AND metadata ? 'month'
        AND (metadata->>'year') ~ '^[0-9]{4}$'
        AND (metadata->>'month') ~ '^([1-9]|1[0-2])$'
      THEN (
        (metadata->>'year')::INTEGER = published_year
        AND (metadata->>'month')::INTEGER = published_month
      )
      ELSE FALSE
    END
  )
);

-- Chunk-level table storing text, metadata, sparse index payload, and dense vectors.
--
-- Notes:
-- - vector(1024) aligns with common embedding dimensions (e.g., Titan Embeddings default mode).
-- - if you choose a different embedding dimension, update this column definition before data load.
CREATE TABLE IF NOT EXISTS kb_chunks (
  chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  doc_id UUID NOT NULL REFERENCES kb_documents(doc_id) ON DELETE CASCADE,
  doc_version TEXT NOT NULL,
  chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
  chunk_text TEXT NOT NULL,
  token_count INTEGER NOT NULL CHECK (token_count >= 0),
  citation_url TEXT NOT NULL,
  citation_title TEXT NOT NULL,
  citation_year INTEGER NOT NULL CHECK (citation_year BETWEEN 1000 AND 2100),
  citation_month INTEGER NOT NULL CHECK (citation_month BETWEEN 1 AND 12),
  page_start INTEGER CHECK (page_start IS NULL OR page_start > 0),
  page_end INTEGER CHECK (page_end IS NULL OR page_end >= page_start),
  section_id TEXT,
  anchor_id TEXT,
  embedding VECTOR(1024) NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  run_id UUID REFERENCES ingestion_runs(run_id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  -- Sparse retrieval payload for BM25-like full-text ranking.
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(chunk_text, ''))) STORED,
  CONSTRAINT uq_kb_chunks_doc_chunk_ver UNIQUE (doc_id, chunk_index, doc_version),
  CONSTRAINT ck_kb_chunks_locator_required CHECK (
    page_start IS NOT NULL OR section_id IS NOT NULL OR anchor_id IS NOT NULL
  ),
  CONSTRAINT ck_kb_chunks_metadata_year_month CHECK (
    CASE
      WHEN
        metadata ? 'year'
        AND metadata ? 'month'
        AND (metadata->>'year') ~ '^[0-9]{4}$'
        AND (metadata->>'month') ~ '^([1-9]|1[0-2])$'
      THEN (
        (metadata->>'year')::INTEGER = citation_year
        AND (metadata->>'month')::INTEGER = citation_month
      )
      ELSE FALSE
    END
  )
);

-- Optional trace table linking chunk evidence to graph entities/relations in Neo4j.
CREATE TABLE IF NOT EXISTS kb_chunk_graph_links (
  link_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  chunk_id UUID NOT NULL REFERENCES kb_chunks(chunk_id) ON DELETE CASCADE,
  graph_node_id TEXT NOT NULL,
  graph_node_label TEXT,
  graph_relation_type TEXT,
  confidence REAL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT uq_chunk_graph_link UNIQUE (chunk_id, graph_node_id, graph_relation_type)
);

-- Metadata and filtering indexes.
CREATE INDEX IF NOT EXISTS idx_kb_documents_category ON kb_documents (category);
CREATE INDEX IF NOT EXISTS idx_kb_documents_lang ON kb_documents (lang);
CREATE INDEX IF NOT EXISTS idx_kb_documents_source_type ON kb_documents (source_type);
CREATE INDEX IF NOT EXISTS idx_kb_documents_published_year ON kb_documents (published_year);
CREATE INDEX IF NOT EXISTS idx_kb_documents_published_month ON kb_documents (published_month);
CREATE INDEX IF NOT EXISTS idx_kb_documents_updated_at ON kb_documents (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_kb_documents_tags_gin ON kb_documents USING GIN (tags);

-- Sparse retrieval index.
CREATE INDEX IF NOT EXISTS idx_kb_chunks_tsv_gin ON kb_chunks USING GIN (tsv);

-- Dense retrieval index (choose one operator family and keep query operator consistent).
-- For cosine similarity queries using "<=>", keep vector_cosine_ops.
CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding_ivfflat
  ON kb_chunks
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 200);

-- Frequently-used join/filter index.
CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc_id ON kb_chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_citation_year ON kb_chunks (citation_year);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_citation_month ON kb_chunks (citation_month);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_citation_year_month ON kb_chunks (citation_year, citation_month);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_citation_url ON kb_chunks (citation_url);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_updated_at ON kb_chunks (updated_at DESC);

-- Auto-update updated_at timestamps.
CREATE OR REPLACE FUNCTION set_updated_at_timestamp() RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_kb_documents_updated_at ON kb_documents;
CREATE TRIGGER trg_kb_documents_updated_at
BEFORE UPDATE ON kb_documents
FOR EACH ROW EXECUTE FUNCTION set_updated_at_timestamp();

DROP TRIGGER IF EXISTS trg_kb_chunks_updated_at ON kb_chunks;
CREATE TRIGGER trg_kb_chunks_updated_at
BEFORE UPDATE ON kb_chunks
FOR EACH ROW EXECUTE FUNCTION set_updated_at_timestamp();

-- ------------------------------------------------------------
-- Hybrid retrieval example (RRF fusion in SQL)
-- ------------------------------------------------------------
-- Inputs:
--   :query_text      -> raw user query string
--   :query_embedding -> embedding vector matching VECTOR(1024) dimension
--   :k_sparse        -> sparse candidate size (e.g. 50)
--   :k_dense         -> dense candidate size (e.g. 50)
--   :k_final         -> final fused result size (e.g. 10)
--
-- WITH
-- sparse_candidates AS (
--   SELECT
--     c.chunk_id,
--     ROW_NUMBER() OVER (
--       ORDER BY ts_rank_cd(c.tsv, plainto_tsquery('simple', :query_text)) DESC
--     ) AS sparse_rank
--   FROM kb_chunks c
--   WHERE c.tsv @@ plainto_tsquery('simple', :query_text)
--   LIMIT :k_sparse
-- ),
-- dense_candidates AS (
--   SELECT
--     c.chunk_id,
--     ROW_NUMBER() OVER (ORDER BY c.embedding <=> :query_embedding::vector) AS dense_rank
--   FROM kb_chunks c
--   ORDER BY c.embedding <=> :query_embedding::vector
--   LIMIT :k_dense
-- ),
-- fused AS (
--   SELECT
--     COALESCE(s.chunk_id, d.chunk_id) AS chunk_id,
--     COALESCE(1.0 / (60 + s.sparse_rank), 0.0) +
--     COALESCE(1.0 / (60 + d.dense_rank), 0.0) AS rrf_score
--   FROM sparse_candidates s
--   FULL OUTER JOIN dense_candidates d ON d.chunk_id = s.chunk_id
-- )
-- SELECT
--   f.chunk_id,
--   f.rrf_score,
--   c.chunk_text,
--   c.citation_url,
--   c.citation_year,
--   c.citation_month,
--   c.page_start,
--   c.page_end,
--   c.section_id,
--   c.anchor_id,
--   c.metadata
-- FROM fused f
-- JOIN kb_chunks c ON c.chunk_id = f.chunk_id
-- ORDER BY f.rrf_score DESC
-- LIMIT :k_final;
