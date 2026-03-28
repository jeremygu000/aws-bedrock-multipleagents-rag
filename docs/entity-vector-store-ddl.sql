-- Entity vector store schema for knowledge graph entities and relations.
-- Stores extracted entities/relations with pgvector embeddings for similarity search.
-- Depends on: vector extension (already enabled in postgres-pgvector-ddl.sql).
-- This file is a review draft, not an automatic migration.

-- Entity records with embeddings for semantic search.
-- Embedding dimension matches kb_chunks (1024, Qwen embedding model).
CREATE TABLE IF NOT EXISTS kb_entities (
  entity_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  type TEXT NOT NULL,
  canonical_key TEXT,
  description TEXT NOT NULL DEFAULT '',
  aliases TEXT[] NOT NULL DEFAULT '{}',
  confidence REAL NOT NULL DEFAULT 0.0,
  source_chunk_ids TEXT[] NOT NULL DEFAULT '{}',
  embedding VECTOR(1024),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Relation records with embeddings for semantic search.
-- Embedding is computed from evidence + relation type text.
CREATE TABLE IF NOT EXISTS kb_relations (
  relation_id TEXT PRIMARY KEY,
  source_entity_id TEXT NOT NULL REFERENCES kb_entities(entity_id) ON DELETE CASCADE,
  target_entity_id TEXT NOT NULL REFERENCES kb_entities(entity_id) ON DELETE CASCADE,
  type TEXT NOT NULL,
  evidence TEXT NOT NULL DEFAULT '',
  confidence REAL NOT NULL DEFAULT 0.0,
  weight REAL NOT NULL DEFAULT 1.0,
  source_chunk_ids TEXT[] NOT NULL DEFAULT '{}',
  embedding VECTOR(1024),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Dense retrieval index for entity embedding similarity search.
-- Matches kb_chunks IVFFlat configuration (cosine, lists=200).
CREATE INDEX IF NOT EXISTS idx_kb_entities_embedding_ivfflat
  ON kb_entities
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 200);

-- Dense retrieval index for relation embedding similarity search.
CREATE INDEX IF NOT EXISTS idx_kb_relations_embedding_ivfflat
  ON kb_relations
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 200);

-- Lookup indexes for common query patterns.
CREATE INDEX IF NOT EXISTS idx_kb_entities_name ON kb_entities (name);
CREATE INDEX IF NOT EXISTS idx_kb_entities_type ON kb_entities (type);
CREATE INDEX IF NOT EXISTS idx_kb_entities_name_type ON kb_entities (name, type);
CREATE INDEX IF NOT EXISTS idx_kb_entities_canonical_key ON kb_entities (canonical_key);
CREATE INDEX IF NOT EXISTS idx_kb_entities_updated_at ON kb_entities (updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_kb_relations_source ON kb_relations (source_entity_id);
CREATE INDEX IF NOT EXISTS idx_kb_relations_target ON kb_relations (target_entity_id);
CREATE INDEX IF NOT EXISTS idx_kb_relations_type ON kb_relations (type);
CREATE INDEX IF NOT EXISTS idx_kb_relations_updated_at ON kb_relations (updated_at DESC);

-- GIN index for source_chunk_ids array containment queries.
CREATE INDEX IF NOT EXISTS idx_kb_entities_chunk_ids_gin ON kb_entities USING GIN (source_chunk_ids);
CREATE INDEX IF NOT EXISTS idx_kb_relations_chunk_ids_gin ON kb_relations USING GIN (source_chunk_ids);

-- Auto-update updated_at timestamps (reuses function from postgres-pgvector-ddl.sql).
-- If running standalone, ensure set_updated_at_timestamp() function exists first.
DROP TRIGGER IF EXISTS trg_kb_entities_updated_at ON kb_entities;
CREATE TRIGGER trg_kb_entities_updated_at
BEFORE UPDATE ON kb_entities
FOR EACH ROW EXECUTE FUNCTION set_updated_at_timestamp();

DROP TRIGGER IF EXISTS trg_kb_relations_updated_at ON kb_relations;
CREATE TRIGGER trg_kb_relations_updated_at
BEFORE UPDATE ON kb_relations
FOR EACH ROW EXECUTE FUNCTION set_updated_at_timestamp();
