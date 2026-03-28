"""Hybrid chunker: header-based splitting with fixed-size fallback for the RAG ingestion pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.document_parser import DocumentSection, ParsedDocument

__all__ = ["Chunk", "chunk_document", "DocumentSection", "ParsedDocument"]


@dataclass
class Chunk:
    chunk_index: int
    chunk_text: str
    token_count: int
    page_start: int | None
    page_end: int | None
    section_id: str | None
    anchor_id: str | None


def _count_tokens(text: str) -> int:
    return len(text.split())


def _fixed_size_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    section_id: str | None,
    page_start: int | None,
    page_end: int | None,
    start_index: int,
    use_anchor: bool,
) -> list[Chunk]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    overlap_words: list[str] = []

    def _flush(idx: int) -> None:
        chunk_text = " ".join(current_sentences).strip()
        if not chunk_text:
            return
        anchor = f"chunk-{idx}" if use_anchor else None
        chunks.append(
            Chunk(
                chunk_index=idx,
                chunk_text=chunk_text,
                token_count=_count_tokens(chunk_text),
                page_start=page_start,
                page_end=page_end,
                section_id=section_id,
                anchor_id=anchor,
            )
        )

    idx = start_index
    for sentence in sentences:
        s_tokens = _count_tokens(sentence)
        if current_tokens + s_tokens > chunk_size and current_sentences:
            _flush(idx)
            idx += 1
            if chunk_overlap > 0 and overlap_words:
                overlap_text = " ".join(overlap_words[-chunk_overlap:])
                current_sentences = [overlap_text]
                current_tokens = _count_tokens(overlap_text)
            else:
                current_sentences = []
                current_tokens = 0

        current_sentences.append(sentence)
        current_tokens += s_tokens
        overlap_words.extend(sentence.split())

    if current_sentences:
        _flush(idx)

    return chunks


def chunk_document(
    parsed: ParsedDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_chunk_size: int = 50,
) -> list[Chunk]:
    """Split a ParsedDocument into Chunks using header-based or fixed-size strategy.

    Header-based when sections are present; sentence-boundary fixed-size with overlap otherwise.
    Every returned Chunk satisfies: page_start IS NOT NULL OR section_id IS NOT NULL OR anchor_id IS NOT NULL.
    """
    raw_chunks: list[Chunk] = []

    if parsed.sections:
        idx = 0
        for section in parsed.sections:
            text = section.text.strip()
            if not text:
                continue
            token_count = _count_tokens(text)
            if token_count <= chunk_size:
                raw_chunks.append(
                    Chunk(
                        chunk_index=idx,
                        chunk_text=text,
                        token_count=token_count,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        section_id=section.section_id,
                        anchor_id=None,
                    )
                )
                idx += 1
            else:
                sub = _fixed_size_split(
                    text=text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    section_id=section.section_id,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    start_index=idx,
                    use_anchor=False,
                )
                raw_chunks.extend(sub)
                idx += len(sub)
    else:
        text = parsed.text.strip()
        if not text:
            return []
        raw_chunks = _fixed_size_split(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            section_id=None,
            page_start=None,
            page_end=None,
            start_index=0,
            use_anchor=True,
        )

    filtered = [c for c in raw_chunks if c.token_count >= min_chunk_size]

    if not filtered and raw_chunks:
        filtered = [max(raw_chunks, key=lambda c: c.token_count)]

    for i, chunk in enumerate(filtered):
        chunk.chunk_index = i

    return filtered
