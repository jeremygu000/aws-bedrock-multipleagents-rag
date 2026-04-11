"""Leiden community detection for knowledge graph entities (Phase 3.5).

Implements a three-class pipeline:

1. ``CommunityDetector``  — exports the Entity/RELATES_TO graph from Neo4j,
   builds an igraph.Graph, runs Leiden at multiple resolution levels, and
   returns a flat list of ``Community`` objects with hierarchical parent/child
   linkage.

2. ``CommunitySummarizer`` — generates LLM summaries for each community using
   the Bedrock Converse API (Nova Pro by default), following the Microsoft
   GraphRAG community-report prompt pattern.

3. ``CommunityStore`` — persists ``CommunitySummary`` objects back into Neo4j
   as ``:Community`` nodes and links ``Entity`` nodes via ``:BELONGS_TO``
   relationships; also supports read-back by level for downstream retrieval.

Optional dependency: ``leidenalg`` + ``igraph``.  When not installed,
``HAS_LEIDEN`` is ``False`` and ``CommunityDetector.detect_communities()``
raises ``RuntimeError`` with a helpful message instead of crashing at import
time.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from neo4j import ManagedTransaction

    from .config import Settings
    from .graph_repository import Neo4jRepository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies — fail gracefully if not installed
# ---------------------------------------------------------------------------

try:
    import igraph as ig
    import leidenalg as la

    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False
    ig = None  # type: ignore[assignment]
    la = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Cypher query constants
# ---------------------------------------------------------------------------

_EXPORT_ENTITIES_CYPHER = """
MATCH (e:Entity)
RETURN e.entity_id AS entity_id,
       e.name AS name,
       e.type AS type,
       e.description AS description
"""

_EXPORT_RELATIONS_CYPHER = """
MATCH (src:Entity)-[r:RELATES_TO]->(tgt:Entity)
RETURN src.entity_id AS source_id,
       tgt.entity_id AS target_id,
       r.rel_type AS rel_type,
       r.weight AS weight,
       r.evidence AS evidence
"""

_UPSERT_COMMUNITY_CYPHER = """
MERGE (c:Community {community_id: $community_id})
SET c.level           = $level,
    c.parent_id       = $parent_id,
    c.title           = $title,
    c.summary         = $summary,
    c.findings        = $findings,
    c.rating          = $rating,
    c.rating_explanation = $rating_explanation,
    c.entity_count    = $entity_count,
    c.updated_at      = datetime()
"""

_LINK_ENTITY_TO_COMMUNITY_CYPHER = """
MATCH (e:Entity {entity_id: $entity_id})
MATCH (c:Community {community_id: $community_id})
MERGE (e)-[:BELONGS_TO {level: $level}]->(c)
"""

_GET_COMMUNITIES_BY_LEVEL_CYPHER = """
MATCH (c:Community)
WHERE c.level = $level
RETURN c.community_id     AS community_id,
       c.level            AS level,
       c.parent_id        AS parent_id,
       c.title            AS title,
       c.summary          AS summary,
       c.findings         AS findings,
       c.rating           AS rating,
       c.rating_explanation AS rating_explanation,
       c.entity_count     AS entity_count,
       c.updated_at       AS updated_at
ORDER BY c.rating DESC
"""

_ENSURE_COMMUNITY_INDEX_CYPHER = (
    "CREATE INDEX community_id_idx IF NOT EXISTS FOR (c:Community) ON (c.community_id)"
)
_ENSURE_COMMUNITY_LEVEL_INDEX_CYPHER = (
    "CREATE INDEX community_level_idx IF NOT EXISTS FOR (c:Community) ON (c.level)"
)

# ---------------------------------------------------------------------------
# Pydantic data models
# ---------------------------------------------------------------------------


class Community(BaseModel):
    """A single Leiden community detected in the knowledge graph."""

    community_id: str = Field(description="Unique identifier for this community")
    level: int = Field(ge=0, description="Hierarchy level (0 = leaf, higher = more coarse)")
    parent_id: str | None = Field(
        default=None,
        description="community_id of the parent community at the next-coarser level, or None for roots",
    )
    entity_ids: list[str] = Field(default_factory=list, description="Entity IDs in this community")
    entity_names: list[str] = Field(
        default_factory=list, description="Entity names (parallel to entity_ids)"
    )
    relation_descriptions: list[str] = Field(
        default_factory=list,
        description="Short textual representations of intra-community relations",
    )


class CommunitySummary(BaseModel):
    """LLM-generated summary for a detected community."""

    community_id: str
    level: int
    parent_id: str | None = None
    title: str = Field(description="Short descriptive title generated by the LLM")
    summary: str = Field(description="Executive summary of the community (1-3 sentences)")
    findings: list[str] = Field(
        default_factory=list,
        description="5-10 key insights about entities and relationships in this community",
    )
    rating: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Impact severity score (0-10) from LLM",
    )
    rating_explanation: str = Field(default="", description="One-sentence justification for rating")
    entity_ids: list[str] = Field(default_factory=list)
    entity_count: int = Field(default=0)
    updated_at: str = Field(
        default="",
        description="ISO-8601 datetime string of when the summary was generated",
    )


# ---------------------------------------------------------------------------
# GraphRAG-inspired community summary prompt
# ---------------------------------------------------------------------------

_COMMUNITY_SYSTEM_PROMPT = """You are an AI assistant that helps a human analyst analyze
entity communities extracted from a knowledge graph. You write concise, well-structured
community reports that help people understand the key themes and relationships in a
cluster of related entities."""

_COMMUNITY_REPORT_PROMPT_TEMPLATE = """
You will be given information about a community of entities and their relationships
extracted from a knowledge graph. Your task is to produce a comprehensive community report.

---COMMUNITY ENTITIES---
{entities_section}

---INTRA-COMMUNITY RELATIONS---
{relations_section}

---INSTRUCTIONS---
Write a community report covering:
1. A concise **title** (≤10 words) that captures the theme of this community.
2. A **summary** (2-3 sentences) describing the key entities and their roles.
3. **Findings**: 5 to 10 bullet-point insights about the entities and relationships.
   Each finding must be a single, self-contained sentence.
4. A **rating** from 0 to 10 (float, one decimal) reflecting the impact/severity of this community
   in the broader knowledge graph context.
5. A **rating_explanation** (one sentence) justifying the rating.

Respond ONLY with a valid JSON object in this exact schema (no extra keys):
{{
  "title": "<string>",
  "summary": "<string>",
  "findings": ["<string>", ...],
  "rating": <float>,
  "rating_explanation": "<string>"
}}
"""


def _build_community_prompt(community: Community) -> str:
    """Construct the user-turn prompt for summarising a community."""
    entities_lines = [
        f"- [{eid}] {name}" for eid, name in zip(community.entity_ids, community.entity_names)
    ]
    entities_section = "\n".join(entities_lines) if entities_lines else "(no entities)"

    relations_section = (
        "\n".join(f"- {desc}" for desc in community.relation_descriptions)
        if community.relation_descriptions
        else "(no relations)"
    )

    return _COMMUNITY_REPORT_PROMPT_TEMPLATE.format(
        entities_section=entities_section,
        relations_section=relations_section,
    )


# ---------------------------------------------------------------------------
# CommunityDetector
# ---------------------------------------------------------------------------


class CommunityDetector:
    """Exports the Entity/RELATES_TO graph from Neo4j and runs Leiden to
    detect hierarchical communities at multiple resolution levels.

    Args:
        neo4j_repo: An already-initialised ``Neo4jRepository`` (synchronous driver).
        settings: Application settings (provides community_* config fields).
    """

    def __init__(self, neo4j_repo: Neo4jRepository, settings: Settings) -> None:
        self._repo = neo4j_repo
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_communities(self) -> list[Community]:
        """Run the full community detection pipeline.

        Raises:
            RuntimeError: If leidenalg / igraph are not installed.

        Returns:
            A flat list of ``Community`` objects across all configured hierarchy levels.
        """
        if not HAS_LEIDEN:
            raise RuntimeError(
                "leidenalg and igraph are required for community detection. "
                "Install them with: pip install leidenalg igraph"
            )

        logger.info("Starting community detection pipeline")

        entities, relations = self._export_graph()
        if not entities:
            logger.warning("No entities found in Neo4j — skipping community detection")
            return []

        logger.info(
            "Exported %d entities and %d relations from Neo4j",
            len(entities),
            len(relations),
        )

        graph = self._build_igraph(entities, relations)
        communities = self._run_leiden(graph, entities, relations)

        logger.info("Community detection complete: %d communities across all levels", len(communities))
        return communities

    # ------------------------------------------------------------------
    # Graph export
    # ------------------------------------------------------------------

    def _export_graph(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Export all Entity nodes and RELATES_TO edges from Neo4j.

        Returns:
            Tuple of (entities, relations) where each element is a list of
            plain dicts with the keys from the Cypher RETURN clauses.
        """

        def _read_entities(tx: ManagedTransaction) -> list[dict[str, Any]]:
            result = tx.run(_EXPORT_ENTITIES_CYPHER)
            return [dict(record) for record in result]

        def _read_relations(tx: ManagedTransaction) -> list[dict[str, Any]]:
            result = tx.run(_EXPORT_RELATIONS_CYPHER)
            return [dict(record) for record in result]

        driver = self._repo._get_driver()
        with driver.session(database=self._repo._database) as session:
            entities: list[dict[str, Any]] = session.execute_read(_read_entities)
            relations: list[dict[str, Any]] = session.execute_read(_read_relations)

        return entities, relations

    # ------------------------------------------------------------------
    # igraph construction
    # ------------------------------------------------------------------

    def _build_igraph(
        self,
        entities: list[dict[str, Any]],
        relations: list[dict[str, Any]],
    ) -> ig.Graph:  # type: ignore[name-defined]
        """Build an igraph.Graph from exported entity and relation dicts.

        Vertices are Entity nodes; edges are RELATES_TO relationships.
        Edge weight defaults to 1.0 when not set in Neo4j.

        Args:
            entities: List of entity dicts with at least ``entity_id``.
            relations: List of relation dicts with ``source_id``, ``target_id``, ``weight``.

        Returns:
            An igraph.Graph with vertex attribute ``entity_id`` and edge attribute ``weight``.
        """
        # Build a mapping entity_id → vertex index for fast edge construction
        id_to_idx: dict[str, int] = {e["entity_id"]: i for i, e in enumerate(entities)}

        graph: ig.Graph = ig.Graph(directed=False)  # type: ignore[union-attr]
        graph.add_vertices(len(entities))
        graph.vs["entity_id"] = [e["entity_id"] for e in entities]
        graph.vs["name"] = [e.get("name", "") for e in entities]
        graph.vs["entity_type"] = [e.get("type", "") for e in entities]

        edge_list: list[tuple[int, int]] = []
        edge_weights: list[float] = []

        for rel in relations:
            src_id = rel.get("source_id", "")
            tgt_id = rel.get("target_id", "")
            src_idx = id_to_idx.get(src_id)
            tgt_idx = id_to_idx.get(tgt_id)
            if src_idx is None or tgt_idx is None:
                logger.debug(
                    "Skipping relation %s->%s: endpoint not in entity list",
                    src_id,
                    tgt_id,
                )
                continue
            if src_idx == tgt_idx:
                continue  # skip self-loops
            edge_list.append((src_idx, tgt_idx))
            raw_weight = rel.get("weight")
            edge_weights.append(float(raw_weight) if raw_weight is not None else 1.0)

        if edge_list:
            graph.add_edges(edge_list)
            graph.es["weight"] = edge_weights

        logger.debug(
            "igraph built: %d vertices, %d edges",
            graph.vcount(),
            graph.ecount(),
        )
        return graph

    # ------------------------------------------------------------------
    # Leiden algorithm
    # ------------------------------------------------------------------

    def _run_leiden(
        self,
        graph: ig.Graph,  # type: ignore[name-defined]
        entities: list[dict[str, Any]],
        relations: list[dict[str, Any]],
    ) -> list[Community]:
        """Run Leiden at multiple gamma (resolution) values to produce a hierarchy.

        Level 0 uses ``settings.community_resolution``; each successive level
        uses a lower resolution so that communities coarsen hierarchically.
        The number of levels is capped at ``settings.community_max_levels``.
        Communities smaller than ``settings.community_min_size`` are not
        returned (their entities remain unassigned at that level).

        Args:
            graph:    The igraph.Graph to partition.
            entities: Original entity dicts (parallel to graph vertices by index).
            relations: Original relation dicts for building ``relation_descriptions``.

        Returns:
            Flat list of ``Community`` objects across all levels.
        """
        min_size: int = self._settings.community_min_size
        base_gamma: float = self._settings.community_resolution
        max_levels: int = self._settings.community_max_levels

        # Build a relation lookup by (source_id, target_id) for fast description generation
        relation_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for rel in relations:
            key = (rel.get("source_id", ""), rel.get("target_id", ""))
            relation_by_pair[key] = rel

        # Vertex attribute shortcuts
        vertex_entity_ids: list[str] = list(graph.vs["entity_id"])
        vertex_names: list[str] = list(graph.vs["name"])

        all_communities: list[Community] = []

        # Level gamma schedule: [base, base*0.7, base*0.5, ...]
        gamma_schedule: list[float] = [
            base_gamma * (0.7 ** lvl) for lvl in range(max_levels)
        ]

        prev_level_membership: dict[int, str] | None = None  # vertex_idx → parent community_id

        for level, gamma in enumerate(gamma_schedule):
            logger.info("Running Leiden at level %d (gamma=%.3f)", level, gamma)

            try:
                partition = la.find_partition(  # type: ignore[union-attr]
                    graph,
                    la.RBConfigurationVertexPartition,  # type: ignore[union-attr]
                    weights="weight" if graph.ecount() > 0 else None,
                    resolution_parameter=gamma,
                )
            except Exception:
                logger.exception("Leiden failed at level %d (gamma=%.3f)", level, gamma)
                break

            # Build Community objects from the partition membership list
            # partition.membership is a list[int] parallel to graph vertices
            membership: list[int] = partition.membership
            cluster_to_vertices: dict[int, list[int]] = {}
            for v_idx, cluster_id in enumerate(membership):
                cluster_to_vertices.setdefault(cluster_id, []).append(v_idx)

            level_community_map: dict[int, str] = {}  # cluster_int → community_id str

            for cluster_id, v_indices in cluster_to_vertices.items():
                if len(v_indices) < min_size:
                    logger.debug(
                        "Level %d cluster %d has %d members < min_size %d — skipping",
                        level,
                        cluster_id,
                        len(v_indices),
                        min_size,
                    )
                    continue

                community_id = f"L{level}_C{cluster_id}"
                level_community_map[cluster_id] = community_id

                ent_ids = [vertex_entity_ids[i] for i in v_indices]
                ent_names = [vertex_names[i] for i in v_indices]

                # Determine parent community_id from the previous level's membership
                parent_id: str | None = None
                if prev_level_membership is not None and v_indices:
                    # Use the first vertex's parent as the representative parent
                    parent_id = prev_level_membership.get(v_indices[0])

                # Build relation descriptions for intra-community edges
                ent_id_set = set(ent_ids)
                rel_descriptions: list[str] = []
                for rel in relations:
                    src = rel.get("source_id", "")
                    tgt = rel.get("target_id", "")
                    if src in ent_id_set and tgt in ent_id_set:
                        rel_type = rel.get("rel_type", "RELATES_TO")
                        evidence = rel.get("evidence") or ""
                        desc = f"{src} --[{rel_type}]--> {tgt}"
                        if evidence:
                            short_evidence = evidence[:self._settings.community_evidence_max_chars].replace("\n", " ")
                            desc = f"{desc} ({short_evidence})"
                        rel_descriptions.append(desc)

                community = Community(
                    community_id=community_id,
                    level=level,
                    parent_id=parent_id,
                    entity_ids=ent_ids,
                    entity_names=ent_names,
                    relation_descriptions=rel_descriptions,
                )
                all_communities.append(community)

            # Prepare parent lookup for the next (coarser) level
            # Map each vertex index to the community_id it belongs to at *this* level
            next_prev: dict[int, str] = {}
            for cluster_id, v_indices in cluster_to_vertices.items():
                cid = level_community_map.get(cluster_id)
                if cid is not None:
                    for v_idx in v_indices:
                        next_prev[v_idx] = cid
            prev_level_membership = next_prev

            logger.info(
                "Level %d: %d clusters → %d communities (>= min_size %d)",
                level,
                len(cluster_to_vertices),
                len(level_community_map),
                min_size,
            )

        return all_communities


# ---------------------------------------------------------------------------
# CommunitySummarizer
# ---------------------------------------------------------------------------


class CommunitySummarizer:
    """Generates LLM community summaries via the Bedrock Converse API.

    Uses the Microsoft GraphRAG community-report prompt pattern to produce
    a title, executive summary, key findings, and an impact rating.

    Args:
        settings: Application settings providing ``community_summary_model``,
            ``community_summary_max_tokens``, and ``aws_region``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._bedrock = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize_community(self, community: Community) -> CommunitySummary:
        """Generate an LLM summary for a single community.

        Args:
            community: The community to summarise.

        Returns:
            A ``CommunitySummary`` with title, summary, findings, and rating.
            On LLM or parsing failure, returns a minimal fallback summary.
        """
        prompt = _build_community_prompt(community)
        raw_response = self._call_bedrock(prompt)
        parsed = self._parse_llm_response(raw_response, community)
        return parsed

    def summarize_all(self, communities: list[Community]) -> list[CommunitySummary]:
        """Summarise all communities sequentially.

        Args:
            communities: List of communities to summarise.

        Returns:
            List of ``CommunitySummary`` objects (same order as input).
        """
        summaries: list[CommunitySummary] = []
        total = len(communities)
        for idx, community in enumerate(communities):
            logger.info(
                "Summarising community %s (%d/%d, level=%d, entities=%d)",
                community.community_id,
                idx + 1,
                total,
                community.level,
                len(community.entity_ids),
            )
            try:
                summary = self.summarize_community(community)
            except Exception:
                logger.exception(
                    "Failed to summarise community %s — using fallback",
                    community.community_id,
                )
                summary = self._fallback_summary(community)
            summaries.append(summary)
        return summaries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_bedrock_client(self) -> Any:
        """Lazily initialise the Bedrock runtime client."""
        if self._bedrock is None:
            import boto3

            self._bedrock = boto3.client(
                "bedrock-runtime",
                region_name=self._settings.aws_region,
            )
        return self._bedrock

    def _call_bedrock(self, prompt: str) -> str:
        """Invoke Bedrock Converse API and return the raw text response.

        Args:
            prompt: The user-turn prompt text.

        Returns:
            Raw text content of the first output message block.

        Raises:
            Exception: Propagates any boto3 / Bedrock errors to the caller.
        """
        client = self._get_bedrock_client()
        response = client.converse(
            modelId=self._settings.community_summary_model,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            system=[{"text": _COMMUNITY_SYSTEM_PROMPT}],
            inferenceConfig={
                "maxTokens": self._settings.community_summary_max_tokens,
                "temperature": 0.3,
            },
        )
        # Extract text from the Converse response envelope
        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])
        text_parts: list[str] = []
        for block in content_blocks:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
        return "\n".join(text_parts)

    def _parse_llm_response(self, raw: str, community: Community) -> CommunitySummary:
        """Parse the LLM JSON response into a ``CommunitySummary``.

        Attempts to extract a JSON object from the raw string, handling
        markdown code fences. Falls back to a minimal summary on parse error.

        Args:
            raw: Raw LLM text output (expected to contain a JSON object).
            community: The community being summarised (used for fallback + IDs).

        Returns:
            Populated ``CommunitySummary``.
        """
        json_str = raw.strip()

        # Strip optional markdown code fence (```json ... ```)
        if json_str.startswith("```"):
            lines = json_str.splitlines()
            # Drop first and last fence lines
            inner = [ln for ln in lines if not ln.strip().startswith("```")]
            json_str = "\n".join(inner).strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(
                "Community %s: LLM returned non-JSON response — using fallback",
                community.community_id,
            )
            return self._fallback_summary(community)

        now_iso = datetime.now(tz=timezone.utc).isoformat()

        title = str(data.get("title", f"Community {community.community_id}"))
        summary_text = str(data.get("summary", ""))
        findings_raw = data.get("findings", [])
        findings: list[str] = [str(f) for f in findings_raw] if isinstance(findings_raw, list) else []
        rating_raw = data.get("rating", 0.0)
        try:
            rating = float(rating_raw)
        except (TypeError, ValueError):
            rating = 0.0
        rating = max(0.0, min(10.0, rating))
        rating_explanation = str(data.get("rating_explanation", ""))

        return CommunitySummary(
            community_id=community.community_id,
            level=community.level,
            parent_id=community.parent_id,
            title=title,
            summary=summary_text,
            findings=findings,
            rating=rating,
            rating_explanation=rating_explanation,
            entity_ids=community.entity_ids,
            entity_count=len(community.entity_ids),
            updated_at=now_iso,
        )

    @staticmethod
    def _fallback_summary(community: Community) -> CommunitySummary:
        """Return a minimal CommunitySummary when LLM call or parsing fails."""
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        entity_preview = ", ".join(community.entity_names[:5])
        if len(community.entity_names) > 5:
            entity_preview += f" (+{len(community.entity_names) - 5} more)"
        return CommunitySummary(
            community_id=community.community_id,
            level=community.level,
            parent_id=community.parent_id,
            title=f"Community {community.community_id}",
            summary=f"A cluster of {len(community.entity_ids)} entities: {entity_preview}.",
            findings=[],
            rating=0.0,
            rating_explanation="No rating available (LLM summarisation failed).",
            entity_ids=community.entity_ids,
            entity_count=len(community.entity_ids),
            updated_at=now_iso,
        )


# ---------------------------------------------------------------------------
# CommunityStore
# ---------------------------------------------------------------------------


class CommunityStore:
    """Persists ``CommunitySummary`` objects in Neo4j as ``:Community`` nodes
    and links related ``Entity`` nodes via ``:BELONGS_TO`` relationships.

    Args:
        neo4j_repo: An already-initialised ``Neo4jRepository`` (synchronous driver).
        settings: Application settings.
    """

    def __init__(self, neo4j_repo: Neo4jRepository, settings: Settings) -> None:
        self._repo = neo4j_repo
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_indexes(self) -> None:
        """Create Community indexes if they do not already exist."""
        driver = self._repo._get_driver()
        with driver.session(database=self._repo._database) as session:
            session.run(_ENSURE_COMMUNITY_INDEX_CYPHER)
            session.run(_ENSURE_COMMUNITY_LEVEL_INDEX_CYPHER)
        logger.info("Community indexes ensured")

    def store_communities(self, summaries: list[CommunitySummary]) -> int:
        """Upsert community summaries into Neo4j.

        For each ``CommunitySummary``:
        - MERGE a ``:Community`` node keyed on ``community_id``
        - SET all summary fields
        - MERGE ``:BELONGS_TO`` relationships from each member ``Entity``

        Args:
            summaries: List of summaries to persist.

        Returns:
            Number of community nodes written.
        """
        if not summaries:
            return 0

        self.ensure_indexes()
        written = 0

        for summary in summaries:
            try:
                self._upsert_community(summary)
                self._link_entities(summary)
                written += 1
            except Exception:
                logger.exception(
                    "Failed to store community %s — continuing with remaining",
                    summary.community_id,
                )

        logger.info("Stored %d/%d community summaries in Neo4j", written, len(summaries))
        return written

    def get_community_summaries(self, level: int = 0) -> list[CommunitySummary]:
        """Retrieve all community summaries at a given hierarchy level.

        Args:
            level: The hierarchy level to retrieve (0 = most granular).

        Returns:
            List of ``CommunitySummary`` ordered by rating descending.
        """

        def _read(tx: ManagedTransaction) -> list[dict[str, Any]]:
            result = tx.run(_GET_COMMUNITIES_BY_LEVEL_CYPHER, level=level)
            return [dict(record) for record in result]

        driver = self._repo._get_driver()
        with driver.session(database=self._repo._database) as session:
            rows: list[dict[str, Any]] = session.execute_read(_read)

        summaries: list[CommunitySummary] = []
        for row in rows:
            try:
                summaries.append(self._row_to_summary(row))
            except Exception:
                logger.exception(
                    "Failed to deserialise community row %s",
                    row.get("community_id"),
                )
        return summaries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _upsert_community(self, summary: CommunitySummary) -> None:
        """Write a single community node inside a managed write transaction."""

        def _work(tx: ManagedTransaction) -> None:
            tx.run(
                _UPSERT_COMMUNITY_CYPHER,
                community_id=summary.community_id,
                level=summary.level,
                parent_id=summary.parent_id,
                title=summary.title,
                summary=summary.summary,
                findings=summary.findings,
                rating=summary.rating,
                rating_explanation=summary.rating_explanation,
                entity_count=summary.entity_count,
            )

        driver = self._repo._get_driver()
        with driver.session(database=self._repo._database) as session:
            session.execute_write(_work)

    def _link_entities(self, summary: CommunitySummary) -> None:
        """Create BELONGS_TO edges from each member Entity to the Community node."""
        if not summary.entity_ids:
            return

        def _work(tx: ManagedTransaction) -> None:
            for entity_id in summary.entity_ids:
                tx.run(
                    _LINK_ENTITY_TO_COMMUNITY_CYPHER,
                    entity_id=entity_id,
                    community_id=summary.community_id,
                    level=summary.level,
                )

        driver = self._repo._get_driver()
        with driver.session(database=self._repo._database) as session:
            session.execute_write(_work)

    @staticmethod
    def _row_to_summary(row: dict[str, Any]) -> CommunitySummary:
        """Convert a Neo4j record dict to a ``CommunitySummary``."""
        updated_at_raw = row.get("updated_at")
        if updated_at_raw is None:
            updated_at_str = ""
        elif hasattr(updated_at_raw, "isoformat"):
            updated_at_str = updated_at_raw.isoformat()
        else:
            updated_at_str = str(updated_at_raw)

        findings_raw = row.get("findings") or []
        findings: list[str] = list(findings_raw) if isinstance(findings_raw, list) else []

        return CommunitySummary(
            community_id=str(row.get("community_id", "")),
            level=int(row.get("level", 0)),
            parent_id=row.get("parent_id"),
            title=str(row.get("title", "")),
            summary=str(row.get("summary", "")),
            findings=findings,
            rating=float(row.get("rating", 0.0)),
            rating_explanation=str(row.get("rating_explanation", "")),
            entity_ids=[],  # not stored on the node — join query needed for full retrieval
            entity_count=int(row.get("entity_count", 0)),
            updated_at=updated_at_str,
        )
