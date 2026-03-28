from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.document_parser import (
    ParsedDocument,
    _slugify,
    parse_document,
)

FIXTURES = Path(__file__).parent / "fixtures"

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def read_fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class TestSlugify:
    def test_simple_lowercase(self) -> None:
        assert _slugify("Hello World") == "hello-world"

    def test_special_chars_stripped(self) -> None:
        assert _slugify("Section 1.1: Overview!") == "section-11-overview"

    def test_multiple_spaces_collapsed(self) -> None:
        assert _slugify("foo   bar") == "foo-bar"

    def test_leading_trailing_hyphens_stripped(self) -> None:
        assert _slugify("---heading---") == "heading"

    def test_empty_string_returns_section(self) -> None:
        assert _slugify("") == "section"

    def test_only_special_chars(self) -> None:
        assert _slugify("!!!") == "section"

    def test_numbers_preserved(self) -> None:
        assert _slugify("Chapter 3") == "chapter-3"


class TestParseTxt:
    def test_text_content(self) -> None:
        data = read_fixture("sample.txt")
        doc = parse_document(data, "sample.txt")
        assert "sample plain text document" in doc.text
        assert "multiple lines" in doc.text
        assert "extract all text content" in doc.text

    def test_title_from_filename(self) -> None:
        data = read_fixture("sample.txt")
        doc = parse_document(data, "sample.txt")
        assert doc.title == "sample"

    def test_no_sections(self) -> None:
        data = read_fixture("sample.txt")
        doc = parse_document(data, "sample.txt")
        assert doc.sections == []

    def test_mime_type(self) -> None:
        data = read_fixture("sample.txt")
        doc = parse_document(data, "sample.txt")
        assert doc.mime_type == "text/plain"

    def test_page_count_none(self) -> None:
        data = read_fixture("sample.txt")
        doc = parse_document(data, "sample.txt")
        assert doc.page_count is None


class TestParseMd:
    def _get_doc(self) -> ParsedDocument:
        data = read_fixture("sample.md")
        return parse_document(data, "sample.md")

    def test_four_sections_detected(self) -> None:
        doc = self._get_doc()
        assert len(doc.sections) == 4

    def test_section_headings(self) -> None:
        doc = self._get_doc()
        headings = [s.heading for s in doc.sections]
        assert headings == ["Main Title", "Section One", "Subsection 1.1", "Section Two"]

    def test_section_levels(self) -> None:
        doc = self._get_doc()
        levels = [s.level for s in doc.sections]
        assert levels == [1, 2, 3, 2]

    def test_section_ids(self) -> None:
        doc = self._get_doc()
        ids = [s.section_id for s in doc.sections]
        assert ids == ["main-title", "section-one", "subsection-11", "section-two"]

    def test_title_from_first_h1(self) -> None:
        doc = self._get_doc()
        assert doc.title == "Main Title"

    def test_mime_type(self) -> None:
        doc = self._get_doc()
        assert doc.mime_type == "text/markdown"

    def test_section_text_content(self) -> None:
        doc = self._get_doc()
        section_one = next(s for s in doc.sections if s.heading == "Section One")
        assert "Content for section one" in section_one.text

    def test_page_start_end_none(self) -> None:
        doc = self._get_doc()
        for section in doc.sections:
            assert section.page_start is None
            assert section.page_end is None


def _make_bs4_mock(
    title: str,
    headings: list[tuple[int, str]],
    body_text: str,
) -> MagicMock:
    heading_mocks = []
    for level, heading_text in headings:
        tag = MagicMock()
        tag.name = f"h{level}"
        tag.get_text = MagicMock(return_value=heading_text)

        sibling = MagicMock()
        sibling.name = "p"
        sibling.get_text = MagicMock(return_value=f"Content for {heading_text.lower()}.")

        class _NoGetText:
            name = None

        tag.next_siblings = [sibling, _NoGetText()]
        heading_mocks.append(tag)

    title_tag_mock = MagicMock()
    title_tag_mock.get_text = MagicMock(return_value=title)

    body_mock = MagicMock()
    body_mock.get_text = MagicMock(return_value=body_text)
    body_mock.find_all = MagicMock(return_value=heading_mocks)

    soup_mock = MagicMock()
    soup_mock.find = MagicMock(
        side_effect=lambda tag, **kw: title_tag_mock if tag == "title" else body_mock
    )

    bs4_class = MagicMock(return_value=soup_mock)
    return bs4_class


_SAMPLE_HTML_TITLE = "Sample HTML"
_SAMPLE_HTML_HEADINGS: list[tuple[int, str]] = [
    (1, "Main Title"),
    (2, "Section One"),
    (2, "Section Two"),
]
_SAMPLE_HTML_BODY = "Main Title\nSection One\nSection Two"


class TestParseHtml:
    def _get_doc(self) -> ParsedDocument:
        bs4_mock = _make_bs4_mock(
            _SAMPLE_HTML_TITLE, _SAMPLE_HTML_HEADINGS, _SAMPLE_HTML_BODY
        )
        with patch("app.document_parser._BeautifulSoup", bs4_mock):
            data = read_fixture("sample.html")
            return parse_document(data, "sample.html")

    def test_title_from_title_tag(self) -> None:
        doc = self._get_doc()
        assert doc.title == "Sample HTML"

    def test_three_sections(self) -> None:
        doc = self._get_doc()
        assert len(doc.sections) == 3

    def test_section_headings(self) -> None:
        doc = self._get_doc()
        headings = [s.heading for s in doc.sections]
        assert headings == ["Main Title", "Section One", "Section Two"]

    def test_section_levels(self) -> None:
        doc = self._get_doc()
        levels = [s.level for s in doc.sections]
        assert levels == [1, 2, 2]

    def test_section_ids(self) -> None:
        doc = self._get_doc()
        ids = [s.section_id for s in doc.sections]
        assert ids == ["main-title", "section-one", "section-two"]

    def test_text_stripped_of_html(self) -> None:
        doc = self._get_doc()
        assert "<h1>" not in doc.text
        assert "<p>" not in doc.text
        assert "Main Title" in doc.text

    def test_section_text_no_tags(self) -> None:
        doc = self._get_doc()
        for section in doc.sections:
            assert "<" not in section.text

    def test_mime_type(self) -> None:
        doc = self._get_doc()
        assert doc.mime_type == "text/html"


class TestParsePdf:
    def test_page_count_set(self) -> None:
        fake_page = MagicMock()
        fake_page.get_text.return_value = "INTRODUCTION\nSome body text here.\n"

        fake_doc = MagicMock()
        fake_doc.__len__ = MagicMock(return_value=3)
        fake_doc.__iter__ = MagicMock(return_value=iter([fake_page, fake_page, fake_page]))
        fake_doc.__getitem__ = MagicMock(return_value=fake_page)

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = fake_doc

        with patch("app.document_parser._fitz_module", mock_fitz):
            doc = parse_document(b"%PDF-fake", "report.pdf", mime_type="application/pdf")

        assert doc.page_count == 3
        assert doc.mime_type == "application/pdf"
        assert doc.title == "report"

    def test_section_detection(self) -> None:
        page1 = MagicMock()
        page1.get_text.return_value = "INTRODUCTION\nSome body text on page 1.\n"
        page2 = MagicMock()
        page2.get_text.return_value = "METHODOLOGY\nDetails of methodology here.\n"

        fake_doc = MagicMock()
        fake_doc.__len__ = MagicMock(return_value=2)
        fake_doc.__getitem__ = MagicMock(side_effect=lambda i: [page1, page2][i])

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = fake_doc

        with patch("app.document_parser._fitz_module", mock_fitz):
            doc = parse_document(b"%PDF-fake", "report.pdf", mime_type="application/pdf")

        headings = [s.heading for s in doc.sections]
        assert "INTRODUCTION" in headings
        assert "METHODOLOGY" in headings

    def test_page_start_end_on_sections(self) -> None:
        page = MagicMock()
        page.get_text.return_value = "RESULTS\nResult details here.\n"

        fake_doc = MagicMock()
        fake_doc.__len__ = MagicMock(return_value=1)
        fake_doc.__getitem__ = MagicMock(return_value=page)

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = fake_doc

        with patch("app.document_parser._fitz_module", mock_fitz):
            doc = parse_document(b"%PDF-fake", "doc.pdf", mime_type="application/pdf")

        if doc.sections:
            assert doc.sections[0].page_start is not None


class TestParseDocx:
    def _make_paragraph(self, text: str, style_name: str) -> MagicMock:
        para = MagicMock()
        para.text = text
        style = MagicMock()
        style.name = style_name
        para.style = style
        return para

    def test_heading_extraction(self) -> None:
        paragraphs = [
            self._make_paragraph("Document Title", "Heading 1"),
            self._make_paragraph("Some intro text.", "Normal"),
            self._make_paragraph("Chapter One", "Heading 2"),
            self._make_paragraph("Chapter body text.", "Normal"),
        ]

        fake_doc = MagicMock()
        fake_doc.paragraphs = paragraphs

        mock_docx = MagicMock()
        mock_docx.Document.return_value = fake_doc

        with patch("app.document_parser._docx_module", mock_docx):
            doc = parse_document(b"PK fake docx", "test.docx", mime_type=DOCX_MIME)

        headings = [s.heading for s in doc.sections]
        assert "Document Title" in headings
        assert "Chapter One" in headings

    def test_heading_levels(self) -> None:
        paragraphs = [
            self._make_paragraph("Title", "Heading 1"),
            self._make_paragraph("Sub", "Heading 2"),
        ]
        fake_doc = MagicMock()
        fake_doc.paragraphs = paragraphs

        mock_docx = MagicMock()
        mock_docx.Document.return_value = fake_doc

        with patch("app.document_parser._docx_module", mock_docx):
            doc = parse_document(b"PK fake docx", "test.docx", mime_type=DOCX_MIME)

        levels = [s.level for s in doc.sections]
        assert levels == [1, 2]

    def test_mime_type(self) -> None:
        fake_doc = MagicMock()
        fake_doc.paragraphs = [self._make_paragraph("Title", "Heading 1")]

        mock_docx = MagicMock()
        mock_docx.Document.return_value = fake_doc

        with patch("app.document_parser._docx_module", mock_docx):
            doc = parse_document(b"PK fake", "doc.docx", mime_type=DOCX_MIME)

        assert doc.mime_type == DOCX_MIME

    def test_title_from_first_heading1(self) -> None:
        fake_doc = MagicMock()
        fake_doc.paragraphs = [self._make_paragraph("My Report", "Heading 1")]

        mock_docx = MagicMock()
        mock_docx.Document.return_value = fake_doc

        with patch("app.document_parser._docx_module", mock_docx):
            doc = parse_document(b"PK", "report.docx", mime_type=DOCX_MIME)

        assert doc.title == "My Report"


class TestUnsupportedFormat:
    def test_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported file format"):
            parse_document(b"random data", "document.xyz")

    def test_raises_for_unknown_mime(self) -> None:
        with pytest.raises(ValueError, match="Unsupported file format"):
            parse_document(b"data", "file.txt", mime_type="application/octet-stream")


class TestAutoDetectMimeFromExtension:
    def test_txt_extension(self) -> None:
        doc = parse_document(b"hello world", "readme.txt")
        assert doc.mime_type == "text/plain"

    def test_md_extension(self) -> None:
        doc = parse_document(b"# Title\n\nBody.", "notes.md")
        assert doc.mime_type == "text/markdown"

    def test_htm_extension(self) -> None:
        html_bytes = b"<html><head><title>T</title></head><body><p>text</p></body></html>"

        bs4_mock = _make_bs4_mock("T", [], "text")
        with patch("app.document_parser._BeautifulSoup", bs4_mock):
            doc = parse_document(html_bytes, "page.htm")

        assert doc.mime_type == "text/html"

    def test_pdf_mime_explicit_overrides(self) -> None:
        fake_doc = MagicMock()
        fake_doc.__len__ = MagicMock(return_value=1)
        page = MagicMock()
        page.get_text.return_value = "content"
        fake_doc.__getitem__ = MagicMock(return_value=page)

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = fake_doc

        with patch("app.document_parser._fitz_module", mock_fitz):
            doc = parse_document(b"%PDF", "doc.pdf", mime_type="application/pdf")

        assert doc.mime_type == "application/pdf"


class TestEmptyDocument:
    def test_empty_txt(self) -> None:
        doc = parse_document(b"", "empty.txt")
        assert doc.text == ""
        assert doc.sections == []
        assert doc.title == "empty"

    def test_empty_md(self) -> None:
        doc = parse_document(b"", "empty.md")
        assert doc.text == ""
        assert doc.sections == []

    def test_empty_html(self) -> None:
        html = b"<html><body></body></html>"
        bs4_mock = _make_bs4_mock("", [], "")
        with patch("app.document_parser._BeautifulSoup", bs4_mock):
            doc = parse_document(html, "empty.html")
        assert doc.sections == []
