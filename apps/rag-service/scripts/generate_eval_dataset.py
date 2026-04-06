#!/usr/bin/env python3
"""Generate RAGAS evaluation dataset from KB chunks.

Samples diverse chunks from kb_chunks, uses local Ollama LLM to generate
question/answer pairs, and outputs RAGAS-formatted JSONL with citation URLs
for human review.

Usage:
    cd apps/rag-service
    python -m scripts.generate_eval_dataset --output scripts/examples/generated-eval-draft.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any

import httpx
import psycopg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from app.config import get_settings
from app.secrets import resolve_db_password

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("scripts.generate_eval_dataset")

QA_PROMPT_TEMPLATE = """\
You are a knowledge base quality evaluator. Your task is to generate evaluation question/answer pairs from the provided text chunk.

Read the following chunk carefully:

---CHUNK START---
{chunk_text}
---CHUNK END---

Source: {citation_title} ({citation_url})

Instructions:
1. Generate exactly {num_questions} DISTINCT question(s) that can be answered SOLELY from the information in the chunk above.
2. For each question, provide a concise, factual reference answer that is grounded in the chunk text.
3. Classify each question with one of: "factual", "conceptual", "procedural", "entity_lookup", "relationship"
4. Do NOT ask questions that require external knowledge beyond the chunk.
5. Output ONLY valid JSON — no extra text, no markdown, no explanations.
6. The question scope and answer scope must match exactly — do not broaden or narrow either.
7. Prefer the shortest faithful answer. For entity questions return only names. For numeric questions return only the value with unit.
8. Avoid open-ended wording such as "one of", "best", "relationship" unless the chunk clearly supports a single stable answer.
9. Remove hype, opinionated tone, and filler words from answers.
10. For each item, include a ground_truth_context field: the minimal verbatim span from the chunk that directly supports the answer.

Required output format (JSON array):
[
  {{
    "question": "What is ...?",
    "reference_answer": "Taylor Swift's Midnights",
    "question_type": "factual",
    "ground_truth_context": "exact verbatim span from chunk..."
  }}
]
"""


def build_conn_info(password: str) -> dict[str, Any]:
    """Build psycopg (v3) connection kwargs from settings."""
    s = get_settings()
    return {
        "host": s.db_host,
        "port": s.db_port,
        "dbname": s.db_name,
        "user": s.db_user,
        "password": password,
        "sslmode": s.db_ssl_mode,
        "connect_timeout": s.db_connect_timeout_s,
    }


def sample_chunks(
    conn_info: dict[str, Any],
    num_chunks: int,
    min_tokens: int,
    max_tokens: int,
    seed: int | None,
) -> list[dict[str, Any]]:
    """Sample diverse chunks from kb_chunks, grouped by citation_url.

    Strategy:
    - Filter by token_count range for meaningful but not-too-long context
    - Use random() for diversity; seed via setseed() when provided
    - De-duplicate by citation_url, picking at most 2 per URL
    - Return up to num_chunks rows
    """
    rows_needed = num_chunks * 3

    seed_clause = ""
    seed_params: list[Any] = []
    if seed is not None:
        # PostgreSQL setseed takes a value in [-1, 1]; normalise seed int to that range
        pg_seed = (seed % 10000) / 10000.0
        seed_clause = "SELECT setseed(%s);"
        seed_params = [pg_seed]

    fetch_sql = """
        SELECT
            chunk_id,
            chunk_text,
            citation_url,
            citation_title,
            citation_year,
            citation_month,
            section_id,
            anchor_id,
            token_count
        FROM kb_chunks
        WHERE token_count BETWEEN %s AND %s
        ORDER BY random()
        LIMIT %s
    """

    with psycopg.connect(**conn_info) as conn:
        if seed_clause:
            conn.execute(seed_clause, seed_params)
        rows = conn.execute(fetch_sql, (min_tokens, max_tokens, rows_needed)).fetchall()

    if not rows:
        logger.warning(
            "No chunks found with token_count BETWEEN %d AND %d — try relaxing --min-tokens / --max-tokens",
            min_tokens,
            max_tokens,
        )
        return []

    seen_urls: dict[str, int] = {}
    selected: list[dict[str, Any]] = []

    for row in rows:
        (
            chunk_id,
            chunk_text,
            citation_url,
            citation_title,
            citation_year,
            citation_month,
            section_id,
            anchor_id,
            token_count,
        ) = row

        url_key = citation_url or "unknown"
        count_for_url = seen_urls.get(url_key, 0)
        if count_for_url >= 2:
            continue

        seen_urls[url_key] = count_for_url + 1
        selected.append(
            {
                "chunk_id": str(chunk_id),
                "chunk_text": chunk_text or "",
                "citation_url": citation_url or "",
                "citation_title": citation_title or "",
                "citation_year": citation_year,
                "citation_month": citation_month,
                "section_id": section_id or "",
                "anchor_id": anchor_id or "",
                "token_count": token_count or 0,
            }
        )

        if len(selected) >= num_chunks:
            break

    logger.info(
        "Sampled %d chunks from %d candidates (covering %d unique URLs)",
        len(selected),
        len(rows),
        len(seen_urls),
    )
    return selected


def check_ollama(ollama_url: str, model: str) -> None:
    """Verify Ollama is reachable and the model is available."""
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=10.0)
        resp.raise_for_status()
    except httpx.ConnectError:
        logger.error(
            "Cannot connect to Ollama at %s — is Ollama running? Try: ollama serve",
            ollama_url,
        )
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        logger.error("Ollama health check failed: %s", exc)
        sys.exit(1)

    data = resp.json()
    available_models = [m.get("name", "") for m in data.get("models", [])]
    model_base = model.split(":")[0]
    found = any(m.startswith(model_base) for m in available_models)
    if not found:
        logger.warning(
            "Model '%s' not found in Ollama. Available: %s — proceeding anyway (may be pulled on demand)",
            model,
            available_models[:5],
        )
    else:
        logger.info("Ollama reachable at %s, model '%s' found.", ollama_url, model)


_NAV_PATTERNS = re.compile(
    r"(load more results|click here|read more|skip to|cookie|subscribe)",
    re.IGNORECASE,
)


def _is_chunk_clean(chunk: dict[str, Any]) -> bool:
    """Return False for noisy chunks that are unsuitable for Q/A generation.

    Filters:
    - Excessive HTML tags (> 10 occurrences of '<')
    - Too many URLs (> 5 occurrences of 'http')
    - Navigation debris (nav patterns occupy > 20 % of text)
    - Too short after stripping (< 100 characters)
    - No natural language (fewer than 2 sentence-ending punctuation marks)
    """
    text: str = chunk.get("chunk_text", "")
    stripped = text.strip()

    if len(stripped) < 100:
        return False

    if text.count("<") > 10:
        return False

    if text.count("http") > 5:
        return False

    sentence_endings = len(re.findall(r"[.?!]", text))
    if sentence_endings < 2:
        return False

    nav_matches = _NAV_PATTERNS.findall(text)
    if nav_matches:
        total_nav_chars = sum(len(m) for m in nav_matches)
        if total_nav_chars / max(len(stripped), 1) > 0.20:
            return False

    return True


def generate_qa(
    chunk: dict[str, Any],
    ollama_url: str,
    model: str,
    num_questions: int,
) -> list[dict[str, Any]]:
    """Call Ollama to generate Q/A pairs for a single chunk.

    Returns a list of dicts with keys: question, reference_answer, question_type.
    Returns empty list on failure.
    """
    prompt = QA_PROMPT_TEMPLATE.format(
        chunk_text=chunk["chunk_text"][:4000],
        citation_title=chunk["citation_title"] or "Unknown",
        citation_url=chunk["citation_url"] or "Unknown",
        num_questions=num_questions,
    )

    try:
        response = httpx.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,  # disable thinking mode for qwen3
            },
            timeout=120.0,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        logger.warning("Ollama request timed out for chunk %s", chunk["chunk_id"])
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning("Ollama HTTP error for chunk %s: %s", chunk["chunk_id"], exc)
        return []
    except httpx.ConnectError:
        logger.error("Lost connection to Ollama — aborting generation")
        sys.exit(1)

    body = response.json()
    raw_content: str = body.get("message", {}).get("content", "")

    return _parse_llm_output(raw_content, chunk["chunk_id"], chunk["chunk_text"])


def _parse_llm_output(raw_content: str, chunk_id: str, chunk_text: str = "") -> list[dict[str, Any]]:
    """Parse LLM JSON output with fallback to regex extraction.

    Strategy:
    1. Try direct json.loads on stripped content
    2. Try to extract JSON array via regex
    3. Skip on total failure
    """
    content = raw_content.strip()

    # Strip markdown code fences if present
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    content = content.strip()

    # Attempt 1: direct parse
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return _validate_qa_items(parsed, chunk_id, chunk_text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: regex — find the outermost [...] block
    array_match = re.search(r"\[.*\]", content, re.DOTALL)
    if array_match:
        try:
            parsed = json.loads(array_match.group())
            if isinstance(parsed, list):
                return _validate_qa_items(parsed, chunk_id, chunk_text)
        except json.JSONDecodeError:
            pass

    # Attempt 3: try to extract individual JSON objects
    objects_found = re.findall(r"\{[^{}]+\}", content, re.DOTALL)
    items: list[dict[str, Any]] = []
    for obj_str in objects_found:
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict) and "question" in obj:
                items.append(obj)
        except json.JSONDecodeError:
            continue

    if items:
        return _validate_qa_items(items, chunk_id, chunk_text)

    logger.warning("Failed to parse LLM output for chunk %s — skipping", chunk_id)
    logger.debug("Raw LLM output was:\n%s", raw_content[:500])
    return []


def _validate_qa_items(
    items: list[Any], chunk_id: str, chunk_text: str = ""
) -> list[dict[str, Any]]:
    """Filter and normalise Q/A items from parsed JSON."""
    valid = []
    valid_types = {"factual", "conceptual", "procedural", "entity_lookup", "relationship"}

    chunk_words = set(re.findall(r"\b\w+\b", chunk_text.lower())) if chunk_text else set()

    _ACCORDING_TO_PREFIX = re.compile(
        r"^(according to (the )?(?:source|text|chunk|document|passage),?\s*"
        r"|based on (the )?(?:source|text|chunk|document|passage),?\s*)",
        re.IGNORECASE,
    )

    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        reference_answer = str(item.get("reference_answer", "")).strip()
        question_type = str(item.get("question_type", "factual")).strip().lower()
        ground_truth_context = str(item.get("ground_truth_context", "")).strip()

        if not question or not reference_answer:
            logger.debug("Skipping item without question/answer in chunk %s", chunk_id)
            continue

        # Safety net: strip "According to the source/text/..." prefixes
        reference_answer = _ACCORDING_TO_PREFIX.sub("", reference_answer).strip()
        if reference_answer and reference_answer[0].islower():
            reference_answer = reference_answer[0].upper() + reference_answer[1:]

        if question_type not in valid_types:
            question_type = "factual"

        if not question.endswith("?"):
            logger.debug("Rejecting item (question missing '?') in chunk %s", chunk_id)
            continue

        if reference_answer.lower().startswith("according to"):
            logger.debug("Rejecting item (answer starts with 'According to') in chunk %s", chunk_id)
            continue

        if len(reference_answer) > 200:
            logger.debug("Rejecting item (answer too verbose) in chunk %s", chunk_id)
            continue

        if "one of" in question.lower():
            logger.debug("Rejecting item (open-ended 'one of' in question) in chunk %s", chunk_id)
            continue

        if not ground_truth_context:
            logger.debug("Rejecting item (missing ground_truth_context) in chunk %s", chunk_id)
            continue

        if chunk_words:
            context_words = set(re.findall(r"\b\w+\b", ground_truth_context.lower()))
            overlap = chunk_words & context_words
            if len(overlap) < 3:
                logger.debug(
                    "Rejecting item (ground_truth_context has insufficient lexical overlap) in chunk %s",
                    chunk_id,
                )
                continue

        valid.append(
            {
                "question": question,
                "reference_answer": reference_answer,
                "question_type": question_type,
                "ground_truth_context": ground_truth_context,
            }
        )

    return valid


def build_output_record(
    qa: dict[str, Any],
    chunk: dict[str, Any],
    record_id: str,
    model: str,
    category: str,
) -> dict[str, Any]:
    """Assemble a RAGAS-shaped output record."""
    return {
        "id": record_id,
        "user_input": qa["question"],
        "reference": qa["reference_answer"],
        "retrieved_contexts": [chunk["chunk_text"]],
        "category": category,
        "citation_url": chunk["citation_url"],
        "citation_title": chunk["citation_title"],
        "question_type": qa["question_type"],
        "ground_truth_context": qa.get("ground_truth_context", ""),
        "metadata": {
            "category": category,
            "source_chunk_id": chunk["chunk_id"],
            "citation_url": chunk["citation_url"],
            "citation_title": chunk["citation_title"],
            "citation_year": chunk["citation_year"],
            "citation_month": chunk["citation_month"],
            "generated_from": "generate_eval_dataset.py",
            "llm_model": model,
            "ground_truth_context": qa.get("ground_truth_context", ""),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate RAGAS evaluation dataset from KB chunks. "
            "Samples diverse chunks, generates Q/A pairs via Ollama, "
            "and outputs RAGAS-formatted JSONL."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path (e.g. scripts/examples/generated-eval-draft.jsonl)",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=25,
        help="Number of chunks to sample (default: 25)",
    )
    parser.add_argument(
        "--questions-per-chunk",
        type=int,
        default=1,
        help="Q/A pairs to generate per chunk (default: 1)",
    )
    parser.add_argument(
        "--model",
        default="qwen3:32b",
        help="Ollama model name (default: qwen3:32b)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=80,
        help="Minimum chunk token_count (default: 80)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum chunk token_count (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling (optional)",
    )
    parser.add_argument(
        "--category",
        default="qa",
        help="Category for generated items (default: qa)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    s = get_settings()
    logger.info(
        "Database: %s@%s:%d/%s  sslmode=%s",
        s.db_user,
        s.db_host,
        s.db_port,
        s.db_name,
        s.db_ssl_mode,
    )

    try:
        password = resolve_db_password(s)
    except Exception as exc:
        logger.error("Failed to resolve DB password: %s", exc)
        sys.exit(1)

    conn_info = build_conn_info(password)

    logger.info("Checking Ollama at %s ...", args.ollama_url)
    check_ollama(args.ollama_url, args.model)

    logger.info(
        "Sampling %d chunks (token_count %d–%d, seed=%s) ...",
        args.num_chunks,
        args.min_tokens,
        args.max_tokens,
        args.seed,
    )

    try:
        chunks = sample_chunks(
            conn_info=conn_info,
            num_chunks=args.num_chunks,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
    except psycopg.OperationalError as exc:
        logger.error(
            "DB connection failed (%s@%s:%d/%s): %s",
            s.db_user,
            s.db_host,
            s.db_port,
            s.db_name,
            exc,
        )
        sys.exit(1)

    if not chunks:
        logger.error("No chunks sampled — nothing to generate. Exiting.")
        sys.exit(1)

    output_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    total_written = 0
    total_skipped = 0
    record_counter = 0

    logger.info(
        "Generating %d Q/A pair(s) per chunk for %d chunks → writing to %s",
        args.questions_per_chunk,
        len(chunks),
        output_path,
    )

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, chunk in enumerate(chunks, start=1):
            print(
                f"[{idx}/{len(chunks)}] chunk_id={chunk['chunk_id'][:8]}...  "
                f"url={chunk['citation_url'][:60]}  tokens={chunk['token_count']}",
                file=sys.stderr,
            )

            if not _is_chunk_clean(chunk):
                total_skipped += 1
                logger.warning("  → Skipped (noisy chunk filtered by _is_chunk_clean)")
                continue

            qa_pairs = generate_qa(
                chunk=chunk,
                ollama_url=args.ollama_url,
                model=args.model,
                num_questions=args.questions_per_chunk,
            )

            if not qa_pairs:
                total_skipped += 1
                logger.warning("  → Skipped (no valid Q/A generated)")
                continue

            for qa in qa_pairs[: args.questions_per_chunk]:
                record_counter += 1
                record_id = f"gen-qa-{record_counter:04d}"
                record = build_output_record(
                    qa=qa,
                    chunk=chunk,
                    record_id=record_id,
                    model=args.model,
                    category=args.category,
                )
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1
                logger.info(
                    "  → [%s] %s  [%s]",
                    record_id,
                    qa["question"][:80],
                    qa["question_type"],
                )

    print(f"\n{'=' * 60}", file=sys.stderr)
    print("Dataset Generation Complete", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"  Chunks sampled:    {len(chunks)}", file=sys.stderr)
    print(f"  Chunks skipped:    {total_skipped}", file=sys.stderr)
    print(f"  Records written:   {total_written}", file=sys.stderr)
    print(f"  Output file:       {output_path}", file=sys.stderr)
    print(f"  LLM model:         {args.model}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)


if __name__ == "__main__":
    main()
