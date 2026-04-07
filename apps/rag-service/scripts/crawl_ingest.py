"""Standalone async web crawler + ingestion script for the APRA AMCOS website.

Crawls https://www.apraamcos.com.au/ via sitemap, extracts content with crawl4ai,
and ingests directly into PostgreSQL (pgvector), Elasticsearch/OpenSearch, and Neo4j
using the existing ingestion pipeline functions.

Usage:
    cd apps/rag-service && python -m scripts.crawl_ingest [OPTIONS]

Options:
    --urls-file F   Read URLs from a text file (one per line), bypass sitemap fetch
    --url URL       Crawl a single URL (repeatable)
    --category      Filter sitemap section: pages, news, events, awards, all (default: all)
    --limit N       Only process first N URLs (for testing)
    --concurrency N crawl4ai semaphore_count (default: 5)
    --max-retries N Retry attempts per URL / embedding call (default: 3)
    --skip-entities Skip entity extraction even if enabled in config
    --dry-run       Crawl + parse but don't write to DB
    --resume        Skip URLs already in kb_documents (by source_uri)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import re
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from sqlalchemy import text

from app.chunker import chunk_document
from app.config import get_settings
from app.document_parser import parse_document
from app.embedding_factory import EmbeddingClient, get_embedding_client
from app.ingestion import _run_entity_extraction_pipeline
from app.ingestion_models import ChunkRecord, DocumentRecord
from app.ingestion_repository import IngestionRepository
from app.qwen_client import QwenClient

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("crawl_ingest")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _handle_shutdown_signal(signum: int, _frame: Any) -> None:
    global _shutdown_requested  # noqa: PLW0603
    sig_name = signal.Signals(signum).name
    if _shutdown_requested:
        logger.warning("Second %s received — forcing exit", sig_name)
        sys.exit(1)
    logger.warning("%s received — finishing current URL then stopping", sig_name)
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# Resilience policies (redress — Polly-like)
# ---------------------------------------------------------------------------


def _build_embed_policy(max_attempts: int = 6) -> Any:
    """Build a retry + circuit-breaker policy for embedding API calls."""
    from redress import CircuitBreaker, Policy, Retry, default_classifier
    from redress.strategies import decorrelated_jitter

    return Policy(
        retry=Retry(
            classifier=default_classifier,
            strategy=decorrelated_jitter(max_s=10.0),
            max_attempts=max_attempts,
            deadline_s=120.0,
        ),
        circuit_breaker=CircuitBreaker(
            failure_threshold=10,
            window_s=120.0,
            recovery_timeout_s=30.0,
        ),
    )


def _build_ingest_policy(max_attempts: int = 3) -> Any:
    """Build a retry policy for per-URL ingestion (parse + embed + store)."""
    from redress import Policy, Retry, default_classifier
    from redress.strategies import equal_jitter

    return Policy(
        retry=Retry(
            classifier=default_classifier,
            strategy=equal_jitter(max_s=15.0),
            max_attempts=max_attempts,
            deadline_s=300.0,
        ),
    )


# ---------------------------------------------------------------------------
# Sitemap constants
# ---------------------------------------------------------------------------

SITEMAP_URLS = [
    "https://www.apraamcos.com.au/sitemaps-1-sitemap.xml",
]

# Category detection patterns for URL path matching
CATEGORY_PATTERNS: list[tuple[str, str]] = [
    (r"/news/", "news"),
    (r"/events/", "events"),
    (r"/awards/", "awards"),
    (r"/music-creators/", "pages"),
    (r"/music-licences/", "pages"),
    (r"/about/", "pages"),
    (r"/resources/", "pages"),
    (r"/production-music/", "pages"),
    (r"/search", "pages"),
]

XML_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class SitemapEntry:
    """A single URL entry extracted from a sitemap."""

    __slots__ = ("url", "category", "lastmod")

    def __init__(self, url: str, category: str, lastmod: str | None) -> None:
        self.url = url
        self.category = category
        self.lastmod = lastmod  # ISO date string or None

    @property
    def published_year(self) -> int:
        """Extract year from lastmod or fall back to current year."""
        if self.lastmod:
            try:
                return int(self.lastmod[:4])
            except (ValueError, IndexError):
                pass
        return datetime.now(UTC).year

    @property
    def published_month(self) -> int:
        """Extract month from lastmod or fall back to current month."""
        if self.lastmod and len(self.lastmod) >= 7:
            try:
                return int(self.lastmod[5:7])
            except ValueError:
                pass
        return datetime.now(UTC).month


# ---------------------------------------------------------------------------
# Sitemap fetching
# ---------------------------------------------------------------------------


def _detect_category_from_url(page_url: str) -> str:
    for pattern, category in CATEGORY_PATTERNS:
        if re.search(pattern, page_url):
            return category
    return "pages"


async def _fetch_sitemap_xml_via_browser(sitemap_url: str) -> ET.Element:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

    browser_config = BrowserConfig(
        headless=True,
        text_mode=True,
        light_mode=True,
        extra_args=["--disable-extensions", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=30000,
        delay_before_return_html=2.0,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(sitemap_url, config=crawl_config)
        if not result.success:
            raise RuntimeError(
                f"Browser fetch failed for {sitemap_url}: {getattr(result, 'error_message', 'unknown')}"
            )
        raw_html = result.html or ""

    xml_match = re.search(r"<\?xml.*?\?>.*?<urlset[^>]*>.*</urlset>", raw_html, re.DOTALL)
    if xml_match:
        return ET.fromstring(xml_match.group(0))

    xml_match = re.search(r"<urlset[^>]*>.*</urlset>", raw_html, re.DOTALL)
    if xml_match:
        return ET.fromstring(xml_match.group(0))

    return ET.fromstring(raw_html)


async def fetch_sitemap_entries(
    category_filter: str = "all",
) -> list[SitemapEntry]:
    entries: list[SitemapEntry] = []

    for sitemap_url in SITEMAP_URLS:
        logger.info("Fetching sitemap via browser: %s", sitemap_url)
        try:
            root = await _fetch_sitemap_xml_via_browser(sitemap_url)
        except Exception as exc:
            logger.error("Failed to fetch sitemap %s: %s", sitemap_url, exc)
            continue

        for url_elem in root.iter(f"{XML_NS}url"):
            loc_elem = url_elem.find(f"{XML_NS}loc")
            if loc_elem is None or not loc_elem.text:
                continue
            url = loc_elem.text.strip()

            lastmod_elem = url_elem.find(f"{XML_NS}lastmod")
            lastmod = (
                lastmod_elem.text.strip()
                if (lastmod_elem is not None and lastmod_elem.text)
                else None
            )

            category = _detect_category_from_url(url)
            if category_filter != "all" and category != category_filter:
                continue

            entries.append(SitemapEntry(url=url, category=category, lastmod=lastmod))

    logger.info("Total sitemap URLs collected: %d", len(entries))
    if not entries:
        raise RuntimeError("No sitemap URLs found — cannot proceed")
    return entries


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def get_existing_urls(repo: IngestionRepository) -> set[str]:
    """Return all source_uris already ingested as crawler documents."""
    engine = repo._get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT source_uri FROM kb_documents WHERE source_type = 'crawler'")
        ).fetchall()
    return {row[0] for row in rows}


# ---------------------------------------------------------------------------
# URL slug helper
# ---------------------------------------------------------------------------


def _url_to_slug(url: str) -> str:
    """Convert a URL to a filesystem-safe slug string."""
    parsed = urlparse(url)
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", parsed.path.strip("/"))
    return slug or "index"


# ---------------------------------------------------------------------------
# Ingestion pipeline (per page)
# ---------------------------------------------------------------------------


def ingest_page(
    url: str,
    content: str,
    category: str,
    published_year: int,
    published_month: int,
    repo: IngestionRepository,
    qwen_client: QwenClient,
    settings: Any,
    skip_entities: bool = False,
    dry_run: bool = False,
    embed_policy: Any | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> tuple[int, int, int]:
    """Parse, chunk, embed, and store a single crawled page.

    Args:
        url: Page URL (used as source_uri and citation_url).
        content: Markdown (preferred) or HTML string from crawl4ai.
        category: Sitemap-derived category label.
        published_year: Year extracted from lastmod or current year.
        published_month: Month extracted from lastmod or current month.
        repo: IngestionRepository instance.
        qwen_client: QwenClient for embedding.
        settings: Application settings.
        skip_entities: If True, skip entity extraction.
        dry_run: If True, parse and chunk but do not write to DB.

    Returns:
        (chunks_created, entity_count, relation_count)
    """
    content_bytes = content.encode("utf-8")
    content_hash = hashlib.sha256(content_bytes).hexdigest()
    url_slug = _url_to_slug(url)

    parsed = parse_document(content_bytes, f"{url_slug}.md")

    chunks = chunk_document(
        parsed,
        chunk_size=settings.ingestion_chunk_size,
        chunk_overlap=settings.ingestion_chunk_overlap,
        min_chunk_size=settings.ingestion_chunk_min_size,
    )

    if not chunks:
        logger.debug("No chunks produced for %s — skipping", url)
        return 0, 0, 0

    embeddings: list[list[float]] = []
    batch_size = min(settings.ingestion_embed_batch_size, 10)
    embedder = embedding_client or qwen_client
    for i in range(0, len(chunks), batch_size):
        batch_texts = [c.chunk_text for c in chunks[i : i + batch_size]]
        batch_embeddings = embedder.embedding(batch_texts)
        embeddings.extend(batch_embeddings)  # type: ignore[arg-type]
        if i + batch_size < len(chunks):
            time.sleep(0.1)

    if dry_run:
        logger.info("[DRY RUN] Would ingest %d chunks for %s", len(chunks), url)
        return len(chunks), 0, 0

    run_id = repo.create_ingestion_run(
        "crawler",
        {"url": url, "category": category},
    )

    try:
        doc_record = DocumentRecord(
            source_type="crawler",
            source_uri=url,
            title=parsed.title or url_slug,
            lang="en",
            category=category,
            mime_type="text/markdown",
            content_hash=content_hash,
            doc_version="1.0",
            published_year=published_year,
            published_month=published_month,
            author=None,
            tags=["apra-amcos"],
            metadata={"crawled_at": datetime.now(UTC).isoformat()},
            run_id=run_id,
        )
        doc_id = repo.upsert_document(doc_record)
        repo.delete_chunks_for_doc(doc_id, "1.0")

        chunk_records: list[ChunkRecord] = [
            ChunkRecord(
                doc_id=doc_id,
                doc_version="1.0",
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.chunk_text,
                token_count=chunk.token_count,
                citation_url=url,
                citation_title=parsed.title or url_slug,
                citation_year=published_year,
                citation_month=published_month,
                embedding=embeddings[i],
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_id=chunk.section_id,
                anchor_id=chunk.anchor_id,
                metadata={},
                run_id=run_id,
            )
            for i, chunk in enumerate(chunks)
        ]

        repo.batch_insert_chunks(chunk_records)
        repo.bulk_index_opensearch(chunk_records)

        entity_count = 0
        relation_count = 0
        if settings.enable_entity_extraction and not skip_entities and chunk_records:
            try:
                entity_count, relation_count = _run_entity_extraction_pipeline(
                    chunk_records=chunk_records,
                    doc_id=doc_id,
                    qwen_client=qwen_client,
                    settings=settings,
                    embedding_client=embedding_client,
                )
            except Exception:
                logger.warning(
                    "Entity extraction failed for %s; chunk ingestion already succeeded",
                    url,
                    exc_info=True,
                )

        repo.complete_ingestion_run(run_id, "succeeded")
        return len(chunk_records), entity_count, relation_count

    except Exception:
        # Ensure the ingestion run is marked failed so it doesn't stay stuck in 'running'
        try:
            repo.complete_ingestion_run(run_id, "failed", notes=f"Exception during ingest of {url}")
        except Exception:
            logger.error("Failed to mark ingestion run %s as failed", run_id, exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Async crawl + ingest loop
# ---------------------------------------------------------------------------


async def crawl_and_ingest(
    entries: list[SitemapEntry],
    repo: IngestionRepository,
    qwen_client: QwenClient,
    settings: Any,
    concurrency: int = 5,
    skip_entities: bool = False,
    dry_run: bool = False,
    resume: bool = False,
    max_retries: int = 3,
    embedding_client: EmbeddingClient | None = None,
) -> dict[str, Any]:
    """Crawl all entries with crawl4ai and ingest each into the pipeline.

    Args:
        entries: Sitemap URL entries to process.
        repo: IngestionRepository instance.
        qwen_client: QwenClient for embedding.
        settings: Application settings.
        concurrency: crawl4ai semaphore_count (parallel browser tabs).
        skip_entities: Skip entity extraction.
        dry_run: Parse only, no DB writes.
        resume: Skip URLs already present in kb_documents.
        max_retries: Max retry attempts for embedding and per-URL ingestion.

    Returns:
        Summary dict with totals.
    """
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
    except ImportError as exc:
        logger.error("crawl4ai is not installed. Install it with: uv sync --extra crawl\n%s", exc)
        raise

    embed_policy = _build_embed_policy(max_attempts=max_retries + 1)
    ingest_policy = _build_ingest_policy(max_attempts=max_retries + 1)

    existing_urls: set[str] = set()
    if resume and not dry_run:
        logger.info("Resume mode: fetching existing crawler source_uris...")
        try:
            existing_urls = get_existing_urls(repo)
            logger.info("Found %d existing ingested URLs", len(existing_urls))
        except Exception:
            logger.warning("Could not fetch existing URLs — resume will be skipped", exc_info=True)

    to_process: list[SitemapEntry] = []
    skipped_resume = 0
    for entry in entries:
        if resume and entry.url in existing_urls:
            skipped_resume += 1
        else:
            to_process.append(entry)

    if skipped_resume:
        logger.info("Skipping %d already-ingested URLs (--resume)", skipped_resume)

    total = len(to_process)
    logger.info(
        "Processing %d URLs (concurrency=%d, max_retries=%d)", total, concurrency, max_retries
    )

    if total == 0:
        return {
            "total": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped_resume": skipped_resume,
            "chunks_created": 0,
            "entities_extracted": 0,
            "failed_urls": [],
        }

    browser_config = BrowserConfig(
        headless=True,
        text_mode=True,
        light_mode=True,
        extra_args=["--disable-extensions", "--no-sandbox"],
    )

    crawl_config = CrawlerRunConfig(
        excluded_tags=["script", "style", "nav", "footer", "header"],
        exclude_external_links=True,
        exclude_all_images=True,
        word_count_threshold=50,
        page_timeout=30000,
        delay_before_return_html=0.5,
        cache_mode=CacheMode.BYPASS,
        stream=True,
        semaphore_count=concurrency,
    )

    succeeded = 0
    failed = 0
    chunks_created = 0
    entities_extracted = 0
    failed_urls: list[str] = []

    url_to_entry: dict[str, SitemapEntry] = {e.url: e for e in to_process}
    urls = [e.url for e in to_process]

    processed = 0

    def _do_ingest(url: str, html: str, entry: SitemapEntry) -> tuple[int, int, int]:
        return ingest_page(
            url=url,
            content=html,
            category=entry.category,
            published_year=entry.published_year,
            published_month=entry.published_month,
            repo=repo,
            qwen_client=qwen_client,
            settings=settings,
            skip_entities=skip_entities,
            dry_run=dry_run,
            embed_policy=embed_policy,
            embedding_client=embedding_client,
        )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        async for result in await crawler.arun_many(urls, config=crawl_config):
            if _shutdown_requested:
                logger.warning("Shutdown requested — stopping after %d/%d URLs", processed, total)
                break

            processed += 1
            entry = url_to_entry.get(result.url, SitemapEntry(result.url, "general", None))
            prefix = f"[{processed}/{total}]"

            if not result.success:
                err_msg = getattr(result, "error_message", "Unknown crawl error")
                logger.error("%s ❌ Failed crawl: %s — %s", prefix, result.url, err_msg)
                failed_urls.append(result.url)
                failed += 1
                continue

            html = getattr(result, "markdown", None) or getattr(result, "html", None) or ""
            if not html.strip():
                logger.warning("%s ⚠️  Empty content: %s — skipping", prefix, result.url)
                failed_urls.append(result.url)
                failed += 1
                continue

            t0 = time.perf_counter()
            try:
                page_chunks, page_entities, _ = ingest_policy.call(
                    lambda _url=result.url, _html=html, _entry=entry: _do_ingest(
                        _url, _html, _entry
                    ),
                    abort_if=lambda: _shutdown_requested,
                    operation=f"ingest:{result.url}",
                )
                elapsed = time.perf_counter() - t0
                succeeded += 1
                chunks_created += page_chunks
                entities_extracted += page_entities
                logger.info(
                    "%s ✅ %s  (%d chunks, %.1fs)",
                    prefix,
                    result.url,
                    page_chunks,
                    elapsed,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                logger.error(
                    "%s ❌ Ingest failed after retries: %s — %s (%.1fs)",
                    prefix,
                    result.url,
                    exc,
                    elapsed,
                )
                failed_urls.append(result.url)
                failed += 1

    if failed_urls and not _shutdown_requested:
        logger.info("=" * 60)
        logger.info("RETRY PHASE: Re-crawling %d failed URLs", len(failed_urls))
        logger.info("=" * 60)
        retry_succeeded = 0
        retry_crawl_config = CrawlerRunConfig(
            excluded_tags=["script", "style", "nav", "footer", "header"],
            exclude_external_links=True,
            exclude_all_images=True,
            word_count_threshold=50,
            page_timeout=60000,
            delay_before_return_html=1.0,
            cache_mode=CacheMode.BYPASS,
            stream=True,
            semaphore_count=max(1, concurrency // 2),
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            async for result in await crawler.arun_many(failed_urls[:], config=retry_crawl_config):
                if _shutdown_requested:
                    logger.warning("Shutdown requested — aborting retry phase")
                    break

                entry = url_to_entry.get(result.url, SitemapEntry(result.url, "general", None))

                if not result.success:
                    logger.error("  ❌ Retry crawl failed: %s", result.url)
                    continue

                html = getattr(result, "markdown", None) or getattr(result, "html", None) or ""
                if not html.strip():
                    logger.warning("  ⚠️  Retry empty content: %s", result.url)
                    continue

                try:
                    page_chunks, page_entities, _ = ingest_policy.call(
                        lambda _url=result.url, _html=html, _entry=entry: _do_ingest(
                            _url, _html, _entry
                        ),
                        abort_if=lambda: _shutdown_requested,
                        operation=f"retry-ingest:{result.url}",
                    )
                    retry_succeeded += 1
                    succeeded += 1
                    failed -= 1
                    chunks_created += page_chunks
                    entities_extracted += page_entities
                    failed_urls.remove(result.url)
                    logger.info("  ✅ Retry succeeded: %s (%d chunks)", result.url, page_chunks)
                except Exception as exc:
                    logger.error("  ❌ Retry ingest failed: %s — %s", result.url, exc)

        if retry_succeeded:
            logger.info(
                "Retry phase recovered %d/%d URLs",
                retry_succeeded,
                len(failed_urls) + retry_succeeded,
            )

    return {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "skipped_resume": skipped_resume,
        "chunks_created": chunks_created,
        "entities_extracted": entities_extracted,
        "failed_urls": failed_urls,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="crawl_ingest",
        description="Crawl APRA AMCOS website and ingest content into PG + ES + Neo4j.",
    )
    parser.add_argument(
        "--urls-file",
        type=str,
        default=None,
        metavar="FILE",
        help="Read URLs from a text file (one per line), bypasses sitemap fetch",
    )
    parser.add_argument(
        "--url",
        type=str,
        action="append",
        default=None,
        metavar="URL",
        dest="urls",
        help="Crawl a single URL (can be repeated, bypasses sitemap fetch)",
    )
    parser.add_argument(
        "--category",
        default="all",
        choices=["all", "pages", "news", "events", "awards"],
        help="Sitemap section to crawl (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only process first N URLs (useful for testing)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        metavar="N",
        help="Number of concurrent browser tabs (crawl4ai semaphore, default: 5)",
    )
    parser.add_argument(
        "--skip-entities",
        action="store_true",
        default=False,
        help="Skip entity extraction even if RAG_ENABLE_ENTITY_EXTRACTION=true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Crawl and parse pages but do not write anything to the database",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip URLs that already exist in kb_documents (by source_uri)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        metavar="N",
        help="Max retry attempts per URL and embedding call (default: 3)",
    )
    return parser


async def main(argv: list[str] | None = None) -> int:
    """Main async entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 = success, 1 = one or more failures.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    settings = get_settings()
    repo = IngestionRepository(settings)
    qwen_client = QwenClient(settings)
    embedding_client = get_embedding_client(settings)

    # ---- Resolve URL list: --urls-file / --url take priority over sitemap ----
    if args.urls_file or args.urls:
        raw_urls: list[str] = []
        if args.urls_file:
            url_file = Path(args.urls_file)
            if not url_file.is_file():
                logger.error("URLs file not found: %s", url_file)
                return 1
            raw_urls.extend(
                line.strip()
                for line in url_file.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            )
            logger.info("Loaded %d URLs from %s", len(raw_urls), url_file)
        if args.urls:
            raw_urls.extend(args.urls)
            logger.info("Added %d URLs from --url args", len(args.urls))

        # Deduplicate while preserving order
        seen: set[str] = set()
        entries: list[SitemapEntry] = []
        for u in raw_urls:
            if u in seen:
                continue
            seen.add(u)
            category = _detect_category_from_url(u)
            if args.category != "all" and category != args.category:
                continue
            entries.append(SitemapEntry(url=u, category=category, lastmod=None))
        logger.info("Resolved %d unique URLs after category filter", len(entries))
    else:
        try:
            entries = await fetch_sitemap_entries(category_filter=args.category)
        except Exception as exc:
            logger.error("Cannot proceed — sitemap fetch failed: %s", exc)
            return 1

    if args.limit is not None and args.limit > 0:
        entries = entries[: args.limit]
        logger.info("--limit %d applied: processing %d URL(s)", args.limit, len(entries))

    if not entries:
        logger.info("No URLs to process. Exiting.")
        return 0

    logger.info(
        "Starting crawl+ingest: category=%r, concurrency=%d, max_retries=%d, dry_run=%s, resume=%s, skip_entities=%s",
        args.category,
        args.concurrency,
        args.max_retries,
        args.dry_run,
        args.resume,
        args.skip_entities,
    )

    summary = await crawl_and_ingest(
        entries=entries,
        repo=repo,
        qwen_client=qwen_client,
        settings=settings,
        concurrency=args.concurrency,
        skip_entities=args.skip_entities,
        dry_run=args.dry_run,
        resume=args.resume,
        max_retries=args.max_retries,
        embedding_client=embedding_client,
    )

    logger.info("=" * 60)
    logger.info("CRAWL + INGEST SUMMARY")
    logger.info("=" * 60)
    logger.info("  Total URLs processed : %d", summary["total"])
    logger.info("  Succeeded            : %d", summary["succeeded"])
    logger.info("  Failed               : %d", summary["failed"])
    logger.info("  Skipped (resume)     : %d", summary["skipped_resume"])
    logger.info("  Chunks created       : %d", summary["chunks_created"])
    logger.info("  Entities extracted   : %d", summary["entities_extracted"])

    if summary["failed_urls"]:
        logger.warning("Failed URLs (%d):", len(summary["failed_urls"]))
        for failed_url in summary["failed_urls"]:
            logger.warning("  - %s", failed_url)

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
