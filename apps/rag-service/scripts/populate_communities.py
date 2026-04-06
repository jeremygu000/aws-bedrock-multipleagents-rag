"""Populate Leiden communities in Neo4j from the existing entity graph.

Chains the three Phase 3.5 classes:
  1. CommunityDetector  — export Entity/RELATES_TO → Leiden → list[Community]
  2. CommunitySummarizer — LLM summary per community (Bedrock Nova Pro)
  3. CommunityStore      — MERGE :Community nodes + :BELONGS_TO edges in Neo4j

Usage:
    cd apps/rag-service && python -m scripts.populate_communities [OPTIONS]

Options:
    --dry-run       Detect communities but skip LLM summarization and Neo4j write
    --resolution F  Leiden gamma resolution (default: from config, typically 1.0)
    --max-levels N  Max hierarchy levels (default: from config, typically 3)
    --min-size N    Min entities per community (default: from config, typically 3)
    --clear         Delete existing Community nodes before populating
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from app.community_detection import (
    HAS_LEIDEN,
    CommunityDetector,
    CommunityStore,
    CommunitySummarizer,
)
from app.config import get_settings
from app.graph_repository import Neo4jRepository
from app.secrets import resolve_neo4j_password

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scripts.populate_communities")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate Leiden communities in Neo4j from the existing entity graph.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect communities only; skip LLM summarization and Neo4j write",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Leiden gamma resolution (default: config value)",
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        default=None,
        help="Max hierarchy levels (default: config value)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=None,
        help="Min entities per community (default: config value)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete ALL existing Community nodes and BELONGS_TO edges before populating",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Neo4j cleanup helper
# ---------------------------------------------------------------------------

_CLEAR_COMMUNITIES_CYPHER = """
MATCH (c:Community)
DETACH DELETE c
"""


def _clear_communities(neo4j_repo: Neo4jRepository) -> int:
    """Delete all Community nodes and their relationships. Returns count deleted."""
    with neo4j_repo._get_driver().session(database=neo4j_repo._database) as session:
        result = session.run("MATCH (c:Community) RETURN count(c) AS cnt").single()
        count = result["cnt"] if result else 0
        if count > 0:
            session.run(_CLEAR_COMMUNITIES_CYPHER)
            logger.info("Deleted %d existing Community nodes", count)
        else:
            logger.info("No existing Community nodes to delete")
        return count


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    # --- Pre-flight checks ---
    if not HAS_LEIDEN:
        logger.error(
            "leidenalg and/or igraph not installed. "
            "Install them with: uv pip install leidenalg python-igraph"
        )
        sys.exit(1)

    settings = get_settings()

    # Override config with CLI args if provided
    if args.resolution is not None:
        settings.community_resolution = args.resolution
    if args.max_levels is not None:
        settings.community_max_levels = args.max_levels
    if args.min_size is not None:
        settings.community_min_size = args.min_size

    logger.info(
        "Config: resolution=%.2f, max_levels=%d, min_size=%d, model=%s",
        settings.community_resolution,
        settings.community_max_levels,
        settings.community_min_size,
        settings.community_summary_model,
    )

    # --- Connect to Neo4j ---
    neo4j_password = resolve_neo4j_password(settings)
    neo4j_repo = Neo4jRepository(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=neo4j_password,
        database=settings.neo4j_database,
    )
    logger.info("Connected to Neo4j at %s", settings.neo4j_uri)

    try:
        # --- Optionally clear existing communities ---
        if args.clear:
            _clear_communities(neo4j_repo)

        # --- Step 1: Detect communities ---
        logger.info("=" * 60)
        logger.info("STEP 1/3: Detecting communities via Leiden algorithm")
        logger.info("=" * 60)

        t0 = time.perf_counter()
        detector = CommunityDetector(neo4j_repo=neo4j_repo, settings=settings)
        communities = detector.detect_communities()
        t_detect = time.perf_counter() - t0

        if not communities:
            logger.warning("No communities detected. Check that entities and relations exist in Neo4j.")
            sys.exit(0)

        # Print summary
        levels = sorted({c.level for c in communities})
        logger.info(
            "Detected %d communities across %d levels in %.2fs",
            len(communities),
            len(levels),
            t_detect,
        )
        for level in levels:
            level_communities = [c for c in communities if c.level == level]
            avg_size = sum(len(c.entity_ids) for c in level_communities) / len(level_communities)
            logger.info(
                "  Level %d: %d communities (avg %.1f entities/community)",
                level,
                len(level_communities),
                avg_size,
            )

        if args.dry_run:
            logger.info("--dry-run: skipping LLM summarization and Neo4j write")
            _print_community_sample(communities)
            return

        # --- Step 2: Generate LLM summaries ---
        logger.info("=" * 60)
        logger.info("STEP 2/3: Generating LLM summaries (%d communities)", len(communities))
        logger.info("=" * 60)

        t0 = time.perf_counter()
        summarizer = CommunitySummarizer(settings=settings)
        summaries = summarizer.summarize_all(communities)
        t_summarize = time.perf_counter() - t0

        successful = [s for s in summaries if s.summary and s.summary != "Minimal summary (LLM unavailable)"]
        logger.info(
            "Generated %d/%d summaries in %.2fs (%.1fs avg per community)",
            len(successful),
            len(summaries),
            t_summarize,
            t_summarize / max(len(summaries), 1),
        )

        # --- Step 3: Store in Neo4j ---
        logger.info("=" * 60)
        logger.info("STEP 3/3: Storing communities in Neo4j")
        logger.info("=" * 60)

        t0 = time.perf_counter()
        store = CommunityStore(neo4j_repo=neo4j_repo, settings=settings)
        store.ensure_indexes()
        written = store.store_communities(summaries)
        t_store = time.perf_counter() - t0

        logger.info("Stored %d communities in Neo4j in %.2fs", written, t_store)

        # --- Final report ---
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info("  Communities detected : %d", len(communities))
        logger.info("  Summaries generated  : %d (%d successful)", len(summaries), len(successful))
        logger.info("  Stored in Neo4j      : %d", written)
        logger.info("  Time — detect        : %.2fs", t_detect)
        logger.info("  Time — summarize     : %.2fs", t_summarize)
        logger.info("  Time — store         : %.2fs", t_store)
        logger.info("  Time — total         : %.2fs", t_detect + t_summarize + t_store)

    finally:
        neo4j_repo.close()
        logger.info("Neo4j connection closed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_community_sample(communities: list, max_per_level: int = 3) -> None:
    """Print a sample of communities per level for dry-run inspection."""
    from app.community_detection import Community

    levels = sorted({c.level for c in communities})
    for level in levels:
        level_comms: list[Community] = [c for c in communities if c.level == level]
        logger.info("--- Level %d (%d communities) ---", level, len(level_comms))
        for c in level_comms[:max_per_level]:
            names = c.entity_names[:5]
            suffix = f" +{len(c.entity_names) - 5} more" if len(c.entity_names) > 5 else ""
            logger.info(
                "  %s: %d entities [%s%s], %d relations",
                c.community_id,
                len(c.entity_ids),
                ", ".join(names),
                suffix,
                len(c.relation_descriptions),
            )


if __name__ == "__main__":
    main()
