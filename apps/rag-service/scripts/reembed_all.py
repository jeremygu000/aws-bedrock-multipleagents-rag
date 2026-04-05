"""Re-generate all pgvector embeddings using Bedrock Titan Embed v2.

Re-embeds all rows in kb_chunks, kb_entities, and kb_relations, updating the
embedding column in-place (UPDATE only — no truncate/delete).

Usage:
    cd apps/rag-service && python -m scripts.reembed_all [OPTIONS]

    or with direnv + uv:
    direnv exec . uv run --project apps/rag-service python -m scripts.reembed_all

Options:
    --dry-run           Count rows only, do not embed or update
    --table             One of: chunks, entities, relations, all (default: all)
    --batch-size N      Rows per DB transaction (default: 50)
    --delay SECS        Seconds between batches (default: 0.1)
    --resume TIMESTAMP  Skip rows whose updated_at >= TIMESTAMP (ISO 8601)
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import psycopg
from botocore.exceptions import ClientError

from app.config import get_settings
from app.embedding_factory import get_embedding_client
from app.secrets import resolve_db_password

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scripts.reembed_all")

TABLE_CONFIGS: dict[str, dict[str, str]] = {
    "chunks": {
        "table": "kb_chunks",
        "pk": "chunk_id",
        "text_expr": "chunk_text",
        "label": "chunk",
    },
    "entities": {
        "table": "kb_entities",
        "pk": "entity_id",
        "text_expr": "name || ' ' || COALESCE(description, '')",
        "label": "entity",
    },
    "relations": {
        "table": "kb_relations",
        "pk": "relation_id",
        "text_expr": "type || ' ' || COALESCE(evidence, '')",
        "label": "relation",
    },
}

MAX_RETRIES = 5


def build_conn_info(password: str) -> dict[str, Any]:
    """Build psycopg (v3) connection kwargs from settings."""
    settings = get_settings()
    return {
        "host": settings.db_host,
        "port": settings.db_port,
        "dbname": settings.db_name,
        "user": settings.db_user,
        "password": password,
        "sslmode": settings.db_ssl_mode,
        "connect_timeout": settings.db_connect_timeout_s,
    }


def count_rows(conn: psycopg.Connection, table: str, resume_ts: str | None) -> int:
    """Return total rows to process (respecting --resume filter)."""
    if resume_ts:
        sql = f"SELECT COUNT(*) FROM {table} WHERE updated_at < %s"
        row = conn.execute(sql, (resume_ts,)).fetchone()
    else:
        sql = f"SELECT COUNT(*) FROM {table}"
        row = conn.execute(sql).fetchone()
    return int(row[0]) if row else 0


def embed_with_retry(
    embedder: Any,
    texts: list[str],
    max_retries: int = MAX_RETRIES,
) -> list[list[float]]:
    """Embed a list of texts with exponential backoff on throttling."""
    for attempt in range(1, max_retries + 1):
        try:
            result = embedder.embedding(texts)
            # embedding() returns list[float] for single str, list[list[float]] for list
            if texts and isinstance(result[0], float):
                # single-item list returned as flat vector — wrap it
                return [result]  # type: ignore[return-value]
            return result  # type: ignore[return-value]
        except (ClientError, ValueError) as exc:
            error_code = ""
            if isinstance(exc, ClientError):
                error_code = exc.response.get("Error", {}).get("Code", "")
            is_throttle = error_code in ("ThrottlingException", "ServiceUnavailableException")
            if attempt < max_retries and is_throttle:
                wait = 2**attempt
                logger.warning(
                    "Bedrock throttling on attempt %d/%d — retrying in %ds... (%s)",
                    attempt,
                    max_retries,
                    wait,
                    exc,
                )
                time.sleep(wait)
            elif attempt < max_retries:
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "Embedding error attempt %d/%d — retrying in %ds... (%s)",
                    attempt,
                    max_retries,
                    wait,
                    exc,
                )
                time.sleep(wait)
            else:
                logger.error("Embedding failed after %d attempts: %s", max_retries, exc)
                raise
    # unreachable — kept for type-checker
    raise RuntimeError("embed_with_retry: unexpected exit")


def process_table(
    conn_info: dict[str, Any],
    embedder: Any,
    config: dict[str, str],
    batch_size: int,
    delay: float,
    resume_ts: str | None,
    dry_run: bool,
) -> int:
    """Process one table — paginate, embed, update.

    Returns the number of rows updated.
    """
    table = config["table"]
    pk = config["pk"]
    text_expr = config["text_expr"]
    label = config["label"]

    with psycopg.connect(**conn_info) as conn:
        total = count_rows(conn, table, resume_ts)

    if total == 0:
        logger.info("Table %s: 0 rows to process — skipping.", table)
        return 0

    total_batches = (total + batch_size - 1) // batch_size
    logger.info(
        "Table %s: %d rows → %d batches (batch_size=%d)",
        table,
        total,
        total_batches,
        batch_size,
    )

    if dry_run:
        logger.info("[DRY RUN] Would process %d %s rows across %d batches.", total, label, total_batches)
        return 0

    updated_total = 0
    t_start = time.time()

    for batch_num in range(1, total_batches + 1):
        offset = (batch_num - 1) * batch_size
        row_start = offset
        row_end = min(offset + batch_size - 1, total - 1)

        logger.info(
            "Processing %s batch %d/%d (rows %d-%d)...",
            table,
            batch_num,
            total_batches,
            row_start,
            row_end,
        )

        if resume_ts:
            fetch_sql = f"""
                SELECT {pk}, ({text_expr}) AS text_to_embed
                FROM {table}
                WHERE updated_at < %s
                ORDER BY {pk}
                LIMIT %s OFFSET %s
            """
            fetch_params: tuple[Any, ...] = (resume_ts, batch_size, offset)
        else:
            fetch_sql = f"""
                SELECT {pk}, ({text_expr}) AS text_to_embed
                FROM {table}
                ORDER BY {pk}
                LIMIT %s OFFSET %s
            """
            fetch_params = (batch_size, offset)

        with psycopg.connect(**conn_info) as conn:
            rows = conn.execute(fetch_sql, fetch_params).fetchall()

        if not rows:
            logger.debug("Batch %d/%d returned no rows — stopping early.", batch_num, total_batches)
            break

        pks = [r[0] for r in rows]
        # Titan Embed v2: max 8,192 tokens (~30k chars); truncate safely
        texts = [str(r[1] or "")[:30_000] for r in rows]

        embeddings = embed_with_retry(embedder, texts)

        with psycopg.connect(**conn_info) as conn:
            with conn.transaction():
                for pk_val, embedding_vec in zip(pks, embeddings):
                    vector_str = "[" + ",".join(str(v) for v in embedding_vec) + "]"
                    conn.execute(
                        f"UPDATE {table} SET embedding = %s::vector, updated_at = NOW() WHERE {pk} = %s",
                        (vector_str, pk_val),
                    )
            updated_total += len(pks)

        logger.debug(
            "Batch %d/%d: updated %d rows (cumulative: %d).",
            batch_num,
            total_batches,
            len(pks),
            updated_total,
        )

        if batch_num < total_batches and delay > 0:
            time.sleep(delay)

    elapsed = time.time() - t_start
    logger.info(
        "Done. Updated %d %s embeddings in %.1fs",
        updated_total,
        label,
        elapsed,
    )
    return updated_total


def verify_table(conn_info: dict[str, Any], table: str) -> None:
    """Print post-run verification counts and embedding dimensions."""
    with psycopg.connect(**conn_info) as conn:
        (not_null_count,) = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE embedding IS NOT NULL"
        ).fetchone()  # type: ignore[misc]

        dim_row = conn.execute(
            f"SELECT vector_dims(embedding) FROM {table} WHERE embedding IS NOT NULL LIMIT 1"
        ).fetchone()
        dims = dim_row[0] if dim_row else "N/A"

    logger.info(
        "Verification [%s]: %d rows with non-null embedding, vector_dims=%s",
        table,
        not_null_count,
        dims,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-generate all pgvector embeddings using Bedrock Titan Embed v2."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count rows only — do not embed or update.",
    )
    parser.add_argument(
        "--table",
        choices=["chunks", "entities", "relations", "all"],
        default="all",
        help="Which table(s) to process (default: all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Rows per DB transaction (default: 50).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Seconds to wait between batches (default: 0.1).",
    )
    parser.add_argument(
        "--resume",
        metavar="TIMESTAMP",
        default=None,
        help=(
            "Skip rows whose updated_at >= TIMESTAMP (ISO 8601). "
            "Useful for resuming partial runs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = get_settings()
    logger.info(
        "Embedding provider: %s  model: %s  dims: %d",
        settings.embedding_provider,
        settings.bedrock_embedding_model_id
        if settings.embedding_provider == "bedrock"
        else settings.qwen_embedding_model_id,
        settings.embedding_dimensions,
    )

    password = resolve_db_password(settings)
    conn_info = build_conn_info(password)

    logger.info(
        "Database: %s@%s:%d/%s  sslmode=%s",
        settings.db_user,
        settings.db_host,
        settings.db_port,
        settings.db_name,
        settings.db_ssl_mode,
    )

    if args.table == "all":
        tables_to_process = list(TABLE_CONFIGS.keys())
    else:
        tables_to_process = [args.table]

    if args.dry_run:
        logger.info("=== DRY RUN MODE — no data will be modified ===")
        with psycopg.connect(**conn_info) as conn:
            for key in tables_to_process:
                cfg = TABLE_CONFIGS[key]
                total = count_rows(conn, cfg["table"], args.resume)
                total_batches = (total + args.batch_size - 1) // args.batch_size
                resume_note = f" (updated_at < {args.resume})" if args.resume else ""
                print(
                    f"  {cfg['table']:20s}  {total:>6d} rows{resume_note}"
                    f"  → {total_batches} batches of {args.batch_size}"
                )
        return

    embedder = get_embedding_client(settings)
    logger.info("Embedding client ready.")

    grand_total = 0
    t_global = time.time()

    for key in tables_to_process:
        cfg = TABLE_CONFIGS[key]
        logger.info("--- Starting table: %s ---", cfg["table"])
        updated = process_table(
            conn_info=conn_info,
            embedder=embedder,
            config=cfg,
            batch_size=args.batch_size,
            delay=args.delay,
            resume_ts=args.resume,
            dry_run=args.dry_run,
        )
        grand_total += updated

        if not args.dry_run:
            verify_table(conn_info, cfg["table"])

    elapsed_total = time.time() - t_global
    print(f"\n{'='*60}")
    print("Re-embed Complete")
    print(f"{'='*60}")
    print(f"  Tables processed:   {', '.join(tables_to_process)}")
    print(f"  Total rows updated: {grand_total}")
    print(f"  Total time:         {elapsed_total:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
