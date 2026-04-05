"""Graph-enhanced retrieval for the RAG workflow (Phase 3.1).

Combines pgvector entity/relation similarity search with Neo4j graph
traversal to produce a ``GraphContext`` that enriches the standard
chunk-based retrieval pipeline.

Three retrieval strategies are supported:

- **local**: Vector similarity on entities → expand neighbors via Neo4j →
  collect inter-entity relations.
- **global**: Vector similarity on *both* entities and relations directly.
- **hybrid** (default): Union of local and global results, deduplicated.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from .entity_extraction_models import EntityType
from .models import GraphContext, GraphEntity, GraphRelation, RetrievalMode

if TYPE_CHECKING:
    from .config import Settings
    from .embedding_factory import EmbeddingClient
    from .entity_vector_store import EntityVectorStore
    from .graph_repository import Neo4jRepository
    from .qwen_client import QwenClient

logger = logging.getLogger(__name__)

# Type alias for the strategy parameter
RetrievalStrategy = Literal["local", "global", "hybrid"]


class GraphRetriever:
    """Retrieves graph context (entities + relations) for a user query.

    Designed to be instantiated once per request and discarded — it does
    **not** own the lifecycle of the underlying stores or Neo4j connection.
    Callers are responsible for closing ``neo4j_repo`` when done.
    """

    def __init__(
        self,
        *,
        qwen_client: QwenClient,
        vector_store: EntityVectorStore,
        neo4j_repo: Neo4jRepository | None,
        settings: Settings,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._qwen = qwen_client
        self._embedder = embedding_client
        self._vector_store = vector_store
        self._neo4j = neo4j_repo
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = "hybrid",
    ) -> GraphContext:
        """Run graph retrieval and return aggregated context.

        Args:
            query: The user's natural-language query.
            strategy: One of ``local``, ``global``, ``hybrid``.

        Returns:
            A ``GraphContext`` with deduplicated entities and relations.
        """
        mode = RetrievalMode(self._settings.retrieval_mode)
        if mode.skip_graph:
            return GraphContext()

        # Embed the query once — reused by all strategies
        query_embedding = self._embed_query(query)
        if query_embedding is None:
            return GraphContext()

        if strategy == "local":
            return self._retrieve_local(query_embedding, query)
        elif strategy == "global":
            return self._retrieve_global(query_embedding, query)
        else:
            return self._retrieve_hybrid(query_embedding, query)

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _retrieve_local(self, query_embedding: list[float], query: str = "") -> GraphContext:
        """Local retrieval: vector-search entities → expand neighbors → collect relations."""
        top_k_ent = self._settings.graph_top_k_entities
        depth = self._settings.graph_neighbor_depth

        # Step 1 — seed entities from pgvector similarity + text name matching
        seed_entity_dicts = self._vector_store.search_entities(query_embedding, top_k=top_k_ent)

        if query:
            text_entity_dicts = self._vector_store.search_entities_by_text(query, top_k=top_k_ent)
            seed_entity_dicts = _merge_entity_dicts(seed_entity_dicts, text_entity_dicts)

        if not seed_entity_dicts:
            return GraphContext()

        entities_by_id: dict[str, GraphEntity] = {}
        for ed in seed_entity_dicts:
            entities_by_id[ed["entity_id"]] = _entity_dict_to_model(ed)

        # Step 2 — expand neighbors via Neo4j (if available)
        if self._neo4j and depth > 0:
            self._expand_neighbors(seed_entity_dicts, entities_by_id, depth)

        # Step 3 — collect relations between all known entity names
        relations = self._collect_relations_from_graph(entities_by_id)

        return self._build_context(entities_by_id, relations)

    def _retrieve_global(self, query_embedding: list[float], query: str = "") -> GraphContext:
        """Global retrieval: vector-search both entities and relations directly."""
        top_k_ent = self._settings.graph_top_k_entities
        top_k_rel = self._settings.graph_top_k_relations

        entity_dicts = self._vector_store.search_entities(query_embedding, top_k=top_k_ent)
        relation_dicts = self._vector_store.search_relations(query_embedding, top_k=top_k_rel)

        if query:
            text_entity_dicts = self._vector_store.search_entities_by_text(query, top_k=top_k_ent)
            entity_dicts = _merge_entity_dicts(entity_dicts, text_entity_dicts)

        entities_by_id: dict[str, GraphEntity] = {}
        for ed in entity_dicts:
            entities_by_id[ed["entity_id"]] = _entity_dict_to_model(ed)

        relations: list[GraphRelation] = []
        for rd in relation_dicts:
            relations.append(_relation_dict_to_model(rd))

        return self._build_context(entities_by_id, relations)

    def _retrieve_hybrid(self, query_embedding: list[float], query: str = "") -> GraphContext:
        """Hybrid: union of local and global results, deduplicated."""
        local_ctx = self._retrieve_local(query_embedding, query)
        global_ctx = self._retrieve_global(query_embedding, query)
        return _merge_contexts(local_ctx, global_ctx)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> list[float] | None:
        """Generate a single embedding for the query text."""
        embedder = self._embedder or self._qwen
        try:
            result = embedder.embedding(query)
            # embedding() returns list[float] for a single string input
            if result and isinstance(result[0], float):
                return result  # type: ignore[return-value]
            # If list[list[float]] was returned (shouldn't happen for str input), take first
            if result and isinstance(result[0], list):
                return result[0]  # type: ignore[return-value]
            return None
        except Exception:
            logger.exception("Failed to embed query for graph retrieval")
            return None

    def _expand_neighbors(
        self,
        seed_entity_dicts: list[dict],
        entities_by_id: dict[str, GraphEntity],
        depth: int,
    ) -> None:
        """Expand seed entities via Neo4j neighbor traversal, mutating *entities_by_id* in place."""
        if not self._neo4j:
            return

        for ed in seed_entity_dicts:
            entity_name = ed["name"]
            entity_type_str = ed["type"]
            try:
                entity_type = EntityType(entity_type_str)
            except ValueError:
                logger.warning(
                    "Unknown entity type %r, skipping neighbor expansion", entity_type_str
                )
                continue

            try:
                neighbors = self._neo4j.get_entity_neighbors(entity_name, entity_type, depth=depth)
            except Exception:
                logger.exception("Neo4j neighbor expansion failed for %s", entity_name)
                continue

            for neighbor in neighbors:
                if neighbor.entity_id not in entities_by_id:
                    entities_by_id[neighbor.entity_id] = GraphEntity(
                        entity_id=neighbor.entity_id,
                        name=neighbor.name,
                        type=neighbor.type.value,
                        description=neighbor.description,
                        confidence=neighbor.confidence,
                        score=0.0,  # neighbor — no direct similarity score
                    )

    def _collect_relations_from_graph(
        self,
        entities_by_id: dict[str, GraphEntity],
    ) -> list[GraphRelation]:
        """Fetch inter-entity relations from Neo4j for all known entities."""
        if not self._neo4j:
            return []

        entity_names = [ent.name for ent in entities_by_id.values()]
        if not entity_names:
            return []

        try:
            rel_dicts = self._neo4j.get_relations_for_entities(entity_names)
        except Exception:
            logger.exception("Neo4j relation fetch failed")
            return []

        relations: list[GraphRelation] = []
        for rd in rel_dicts:
            relations.append(
                GraphRelation(
                    source_entity=rd["source_name"],
                    target_entity=rd["target_name"],
                    relation_type=rd["rel_type"],
                    evidence=rd.get("evidence", ""),
                    confidence=rd.get("confidence", 0.0),
                    weight=rd.get("weight", 1.0),
                    score=0.0,  # graph traversal — no vector similarity score
                )
            )
        return relations

    def _build_context(
        self,
        entities_by_id: dict[str, GraphEntity],
        relations: list[GraphRelation],
    ) -> GraphContext:
        """Assemble final GraphContext with collected source_chunk_ids."""
        # Collect all referenced chunk IDs for downstream boosting
        all_chunk_ids: set[str] = set()

        # Entity source_chunk_ids are in the original dicts — we don't store them
        # on GraphEntity (lightweight model), but we could retrieve them from vector
        # store results. For now, relations carry source_chunk_ids in the vector store
        # results but not in GraphRelation. We'll collect from the context separately
        # if needed in future phases.

        return GraphContext(
            entities=list(entities_by_id.values()),
            relations=_deduplicate_relations(relations),
            source_chunk_ids=sorted(all_chunk_ids),
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _entity_dict_to_model(ed: dict) -> GraphEntity:
    """Convert an EntityVectorStore result dict to a GraphEntity."""
    # Distance is cosine distance (lower = more similar); convert to score
    distance = ed.get("distance", 1.0)
    score = max(0.0, 1.0 - distance)

    return GraphEntity(
        entity_id=ed["entity_id"],
        name=ed["name"],
        type=ed["type"],
        description=ed.get("description", ""),
        confidence=ed.get("confidence", 0.0),
        score=score,
    )


def _merge_entity_dicts(vector_results: list[dict], text_results: list[dict]) -> list[dict]:
    """Merge vector-search and text-search entity results, keeping the lower distance per entity."""
    by_id: dict[str, dict] = {}
    for ed in vector_results:
        by_id[ed["entity_id"]] = ed
    for ed in text_results:
        existing = by_id.get(ed["entity_id"])
        if existing is None or ed.get("distance", 1.0) < existing.get("distance", 1.0):
            by_id[ed["entity_id"]] = ed
    return sorted(by_id.values(), key=lambda d: d.get("distance", 1.0))


def _relation_dict_to_model(rd: dict) -> GraphRelation:
    """Convert an EntityVectorStore relation result dict to a GraphRelation."""
    distance = rd.get("distance", 1.0)
    score = max(0.0, 1.0 - distance)

    return GraphRelation(
        source_entity=rd.get("source_name", rd.get("source_entity_id", "")),
        target_entity=rd.get("target_name", rd.get("target_entity_id", "")),
        relation_type=rd.get("type", ""),
        evidence=rd.get("evidence", ""),
        confidence=rd.get("confidence", 0.0),
        weight=rd.get("weight", 1.0),
        score=score,
    )


def _deduplicate_relations(relations: list[GraphRelation]) -> list[GraphRelation]:
    """Remove duplicate relations, keeping the higher-scored version."""
    seen: dict[tuple[str, str, str], GraphRelation] = {}
    for rel in relations:
        key = (rel.source_entity, rel.target_entity, rel.relation_type)
        existing = seen.get(key)
        if existing is None or rel.score > existing.score:
            seen[key] = rel
    return list(seen.values())


def _merge_contexts(a: GraphContext, b: GraphContext) -> GraphContext:
    """Merge two GraphContext instances, deduplicating entities and relations."""
    entities_by_id: dict[str, GraphEntity] = {}

    # Prefer higher-scored entity when duplicate
    for ent in a.entities + b.entities:
        existing = entities_by_id.get(ent.entity_id)
        if existing is None or ent.score > existing.score:
            entities_by_id[ent.entity_id] = ent

    merged_relations = _deduplicate_relations(a.relations + b.relations)

    all_chunk_ids = set(a.source_chunk_ids) | set(b.source_chunk_ids)

    return GraphContext(
        entities=list(entities_by_id.values()),
        relations=merged_relations,
        source_chunk_ids=sorted(all_chunk_ids),
    )
