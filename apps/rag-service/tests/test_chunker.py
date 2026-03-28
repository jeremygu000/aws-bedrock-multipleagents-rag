from __future__ import annotations

from app.chunker import DocumentSection, ParsedDocument, chunk_document


def _make_words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


def _section(
    section_id: str,
    text: str,
    page_start: int | None = None,
    page_end: int | None = None,
) -> DocumentSection:
    return DocumentSection(
        heading=section_id,
        level=1,
        text=text,
        page_start=page_start,
        page_end=page_end,
        section_id=section_id,
    )


def _doc(text: str = "", sections: list[DocumentSection] | None = None) -> ParsedDocument:
    return ParsedDocument(
        title="test",
        text=text,
        mime_type="text/plain",
        sections=sections or [],
        page_count=None,
    )


def test_header_based_single_section() -> None:
    text = _make_words(100)
    parsed = _doc(text=text, sections=[_section("sec-1", text)])
    chunks = chunk_document(parsed, chunk_size=512, min_chunk_size=1)
    assert len(chunks) == 1
    assert chunks[0].section_id == "sec-1"
    assert chunks[0].chunk_text == text
    assert chunks[0].anchor_id is None


def test_header_based_large_section() -> None:
    text = ". ".join([_make_words(200)] * 3) + "."
    parsed = _doc(text=text, sections=[_section("sec-big", text)])
    chunks = chunk_document(parsed, chunk_size=256, chunk_overlap=0, min_chunk_size=1)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.section_id == "sec-big"


def test_header_based_multiple_sections() -> None:
    sections = [
        _section("sec-a", _make_words(50)),
        _section("sec-b", _make_words(60)),
        _section("sec-c", _make_words(40)),
    ]
    parsed = _doc(text=" ".join(s.text for s in sections), sections=sections)
    chunks = chunk_document(parsed, chunk_size=512, min_chunk_size=1)
    assert len(chunks) == 3
    assert [c.section_id for c in chunks] == ["sec-a", "sec-b", "sec-c"]


def test_fixed_size_fallback() -> None:
    sentences = [f"Sentence {i} with some words in it." for i in range(20)]
    text = " ".join(sentences)
    parsed = _doc(text=text)
    chunks = chunk_document(parsed, chunk_size=30, chunk_overlap=0, min_chunk_size=1)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.anchor_id == f"chunk-{i}"
        assert chunk.section_id is None


def test_overlap_behavior() -> None:
    sentences = ["Word " * 40 + "end.", "Another " * 40 + "end."]
    text = " ".join(sentences)
    parsed = _doc(text=text)
    chunks = chunk_document(parsed, chunk_size=50, chunk_overlap=10, min_chunk_size=1)
    if len(chunks) >= 2:
        first_words = set(chunks[0].chunk_text.split())
        second_text = chunks[1].chunk_text
        overlap_found = any(w in second_text for w in list(first_words)[:5])
        assert overlap_found


def test_min_size_filter() -> None:
    tiny = _section("tiny", "one two three")
    big = _section("big", _make_words(100))
    parsed = _doc(sections=[tiny, big])
    chunks = chunk_document(parsed, chunk_size=512, min_chunk_size=10)
    section_ids = [c.section_id for c in chunks]
    assert "big" in section_ids
    assert "tiny" not in section_ids


def test_min_size_filter_all_tiny_keeps_longest() -> None:
    sections = [
        _section("s1", "one two"),
        _section("s2", "a b c d e"),
        _section("s3", "x"),
    ]
    parsed = _doc(sections=sections)
    chunks = chunk_document(parsed, chunk_size=512, min_chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0].section_id == "s2"


def test_locator_guarantee() -> None:
    sections = [_section("s1", _make_words(80), page_start=1, page_end=2)]
    parsed_with_sections = _doc(sections=sections)
    for chunk in chunk_document(parsed_with_sections, chunk_size=512, min_chunk_size=1):
        assert (
            chunk.page_start is not None
            or chunk.section_id is not None
            or chunk.anchor_id is not None
        )

    text = " ".join(f"Sentence number {i} ends here." for i in range(30))
    parsed_plain = _doc(text=text)
    for chunk in chunk_document(parsed_plain, chunk_size=20, chunk_overlap=0, min_chunk_size=1):
        assert (
            chunk.page_start is not None
            or chunk.section_id is not None
            or chunk.anchor_id is not None
        )


def test_empty_document() -> None:
    parsed = _doc()
    assert chunk_document(parsed) == []


def test_empty_document_with_empty_sections() -> None:
    parsed = _doc(sections=[_section("s1", "")])
    assert chunk_document(parsed) == []


def test_short_document() -> None:
    text = "Short document."
    parsed = _doc(text=text)
    chunks = chunk_document(parsed, chunk_size=512, min_chunk_size=1)
    assert len(chunks) == 1
    assert chunks[0].chunk_text == text
    assert chunks[0].anchor_id == "chunk-0"


def test_chunk_index_sequential() -> None:
    sections = [_section(f"s{i}", _make_words(60)) for i in range(5)]
    parsed = _doc(sections=sections)
    chunks = chunk_document(parsed, chunk_size=512, min_chunk_size=1)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_index_sequential_after_filter() -> None:
    sections = [
        _section("big1", _make_words(100)),
        _section("tiny", "hi"),
        _section("big2", _make_words(80)),
    ]
    parsed = _doc(sections=sections)
    chunks = chunk_document(parsed, chunk_size=512, min_chunk_size=10)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_page_numbers_inherited() -> None:
    sec = _section("intro", _make_words(80), page_start=3, page_end=4)
    parsed = _doc(sections=[sec])
    chunks = chunk_document(parsed, chunk_size=512, min_chunk_size=1)
    assert chunks[0].page_start == 3
    assert chunks[0].page_end == 4


def test_page_numbers_inherited_large_section() -> None:
    long_text = ". ".join([_make_words(300)] * 2) + "."
    sec = _section("chapter", long_text, page_start=7, page_end=10)
    parsed = _doc(sections=[sec])
    chunks = chunk_document(parsed, chunk_size=256, chunk_overlap=0, min_chunk_size=1)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.page_start == 7
        assert chunk.page_end == 10
