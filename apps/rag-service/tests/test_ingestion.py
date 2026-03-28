import hashlib
import uuid
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from app.config import Settings
from app.ingestion import ingest_document
from app.ingestion_models import UploadMetadata


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        RAG_DB_HOST="localhost",
        RAG_DB_PASSWORD="testpass",
        RAG_S3_BUCKET="test-bucket",
        RAG_CHUNK_SIZE=512,
        RAG_CHUNK_OVERLAP=64,
        RAG_CHUNK_MIN_SIZE=50,
        RAG_EMBED_BATCH_SIZE=10,
    )


def _make_upload_metadata(**kwargs: Any) -> UploadMetadata:
    defaults: dict[str, Any] = dict(
        title="Test Doc",
        source_uri="https://example.com/doc.pdf",
        lang="en",
        category="general",
        published_year=2024,
        published_month=6,
        doc_version="1.0",
    )
    defaults.update(kwargs)
    return UploadMetadata(**defaults)


@dataclass
class _FakeChunk:
    chunk_index: int
    chunk_text: str
    token_count: int
    page_start: int | None = None
    page_end: int | None = None
    section_id: str | None = None
    anchor_id: str | None = None


def _make_chunks(n: int) -> list[_FakeChunk]:
    return [
        _FakeChunk(
            chunk_index=i,
            chunk_text=f"chunk text {i}",
            token_count=10,
            page_start=i + 1,
            page_end=i + 1,
            section_id=f"sec-{i}",
            anchor_id=f"anchor-{i}",
        )
        for i in range(n)
    ]


def _make_parsed_doc(title: str = "Parsed Title", mime_type: str = "application/pdf") -> MagicMock:
    mock = MagicMock()
    mock.title = title
    mock.mime_type = mime_type
    return mock


def _fake_embedding_batch(texts: list[str]) -> list[list[float]]:
    return [[0.1 * (i + 1)] * 4 for i in range(len(texts))]


FAKE_RUN_ID = uuid.uuid4()
FAKE_DOC_ID = uuid.uuid4()
FILE_BYTES = b"fake pdf content"
FILE_HASH = hashlib.sha256(FILE_BYTES).hexdigest()


def _setup_mocks(
    mock_repo_cls: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_boto3: MagicMock,
    mock_parse: MagicMock,
    mock_chunk: MagicMock,
    chunks: list[_FakeChunk] | None = None,
    parsed_title: str = "Parsed Title",
    mime_type: str = "application/pdf",
) -> tuple[MagicMock, MagicMock]:
    if chunks is None:
        chunks = _make_chunks(3)

    mock_repo = MagicMock()
    mock_repo.create_ingestion_run.return_value = FAKE_RUN_ID
    mock_repo.upsert_document.return_value = FAKE_DOC_ID
    mock_repo.delete_chunks_for_doc.return_value = 0
    mock_repo.batch_insert_chunks.return_value = len(chunks)
    mock_repo_cls.return_value = mock_repo

    mock_qwen = MagicMock()
    mock_qwen.embedding.side_effect = _fake_embedding_batch
    mock_qwen_cls.return_value = mock_qwen

    s3_body = MagicMock()
    s3_body.read.return_value = FILE_BYTES
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": s3_body}
    mock_boto3.client.return_value = mock_s3

    mock_parse.return_value = _make_parsed_doc(parsed_title, mime_type)
    mock_chunk.return_value = chunks

    return mock_repo, mock_qwen


@patch("app.ingestion.chunk_document")
@patch("app.ingestion.parse_document")
@patch("app.ingestion.boto3")
@patch("app.ingestion.QwenClient")
@patch("app.ingestion.IngestionRepository")
def test_happy_path_full_pipeline(
    mock_repo_cls: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_boto3: MagicMock,
    mock_parse: MagicMock,
    mock_chunk: MagicMock,
    settings: Settings,
) -> None:
    chunks = _make_chunks(3)
    mock_repo, mock_qwen = _setup_mocks(
        mock_repo_cls, mock_qwen_cls, mock_boto3, mock_parse, mock_chunk, chunks=chunks
    )
    meta = _make_upload_metadata()

    result = ingest_document("my-bucket", "docs/report.pdf", meta, settings)

    assert result.status == "succeeded"
    assert result.run_id == FAKE_RUN_ID
    assert result.doc_id == FAKE_DOC_ID
    assert result.chunks_created == 3
    assert result.error is None

    mock_repo.create_ingestion_run.assert_called_once_with(
        source_type="file",
        config={
            "s3_bucket": "my-bucket",
            "s3_key": "docs/report.pdf",
            "doc_version": "1.0",
        },
    )
    mock_boto3.client.assert_called_once_with("s3")
    mock_boto3.client.return_value.get_object.assert_called_once_with(
        Bucket="my-bucket", Key="docs/report.pdf"
    )
    mock_parse.assert_called_once_with(FILE_BYTES, "report.pdf")
    mock_chunk.assert_called_once_with(
        mock_parse.return_value,
        chunk_size=512,
        chunk_overlap=64,
        min_chunk_size=50,
    )
    mock_repo.upsert_document.assert_called_once()
    mock_repo.delete_chunks_for_doc.assert_called_once_with(FAKE_DOC_ID, "1.0")
    mock_repo.batch_insert_chunks.assert_called_once()
    mock_repo.bulk_index_opensearch.assert_called_once()
    mock_repo.complete_ingestion_run.assert_called_once_with(FAKE_RUN_ID, "succeeded")


@patch("app.ingestion.chunk_document")
@patch("app.ingestion.parse_document")
@patch("app.ingestion.boto3")
@patch("app.ingestion.QwenClient")
@patch("app.ingestion.IngestionRepository")
def test_empty_chunks_no_embedding_call(
    mock_repo_cls: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_boto3: MagicMock,
    mock_parse: MagicMock,
    mock_chunk: MagicMock,
    settings: Settings,
) -> None:
    mock_repo, mock_qwen = _setup_mocks(
        mock_repo_cls, mock_qwen_cls, mock_boto3, mock_parse, mock_chunk, chunks=[]
    )
    meta = _make_upload_metadata()

    result = ingest_document("my-bucket", "docs/empty.pdf", meta, settings)

    assert result.status == "succeeded"
    assert result.chunks_created == 0
    assert result.doc_id == FAKE_DOC_ID

    mock_qwen.embedding.assert_not_called()
    mock_repo.upsert_document.assert_called_once()
    mock_repo.batch_insert_chunks.assert_not_called()
    mock_repo.bulk_index_opensearch.assert_not_called()
    mock_repo.complete_ingestion_run.assert_called_once_with(FAKE_RUN_ID, "succeeded")


@patch("app.ingestion.chunk_document")
@patch("app.ingestion.parse_document")
@patch("app.ingestion.boto3")
@patch("app.ingestion.QwenClient")
@patch("app.ingestion.IngestionRepository")
def test_embedding_batching_25_chunks_3_calls(
    mock_repo_cls: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_boto3: MagicMock,
    mock_parse: MagicMock,
    mock_chunk: MagicMock,
    settings: Settings,
) -> None:
    chunks = _make_chunks(25)
    mock_repo, mock_qwen = _setup_mocks(
        mock_repo_cls, mock_qwen_cls, mock_boto3, mock_parse, mock_chunk, chunks=chunks
    )
    meta = _make_upload_metadata()

    result = ingest_document("my-bucket", "docs/big.pdf", meta, settings)

    assert result.status == "succeeded"
    assert result.chunks_created == 25

    assert mock_qwen.embedding.call_count == 3
    calls = mock_qwen.embedding.call_args_list
    assert len(calls[0][0][0]) == 10
    assert len(calls[1][0][0]) == 10
    assert len(calls[2][0][0]) == 5


@patch("app.ingestion.chunk_document")
@patch("app.ingestion.parse_document")
@patch("app.ingestion.boto3")
@patch("app.ingestion.QwenClient")
@patch("app.ingestion.IngestionRepository")
def test_s3_download_failure_marks_run_failed(
    mock_repo_cls: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_boto3: MagicMock,
    mock_parse: MagicMock,
    mock_chunk: MagicMock,
    settings: Settings,
) -> None:
    mock_repo, _ = _setup_mocks(
        mock_repo_cls, mock_qwen_cls, mock_boto3, mock_parse, mock_chunk
    )
    mock_boto3.client.return_value.get_object.side_effect = Exception("S3 access denied")
    meta = _make_upload_metadata()

    result = ingest_document("my-bucket", "docs/secret.pdf", meta, settings)

    assert result.status == "failed"
    assert result.chunks_created == 0
    assert result.doc_id == UUID(int=0)
    assert "S3 access denied" in (result.error or "")
    mock_repo.complete_ingestion_run.assert_called_once_with(
        FAKE_RUN_ID, "failed", notes="S3 access denied"
    )


@patch("app.ingestion.chunk_document")
@patch("app.ingestion.parse_document")
@patch("app.ingestion.boto3")
@patch("app.ingestion.QwenClient")
@patch("app.ingestion.IngestionRepository")
def test_parse_failure_marks_run_failed(
    mock_repo_cls: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_boto3: MagicMock,
    mock_parse: MagicMock,
    mock_chunk: MagicMock,
    settings: Settings,
) -> None:
    mock_repo, _ = _setup_mocks(
        mock_repo_cls, mock_qwen_cls, mock_boto3, mock_parse, mock_chunk
    )
    mock_parse.side_effect = ValueError("Unsupported file format")
    meta = _make_upload_metadata()

    result = ingest_document("my-bucket", "docs/bad.xyz", meta, settings)

    assert result.status == "failed"
    assert "Unsupported file format" in (result.error or "")
    mock_repo.complete_ingestion_run.assert_called_once_with(
        FAKE_RUN_ID, "failed", notes="Unsupported file format"
    )
    mock_repo.upsert_document.assert_not_called()


@patch("app.ingestion.chunk_document")
@patch("app.ingestion.parse_document")
@patch("app.ingestion.boto3")
@patch("app.ingestion.QwenClient")
@patch("app.ingestion.IngestionRepository")
def test_embedding_failure_marks_run_failed(
    mock_repo_cls: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_boto3: MagicMock,
    mock_parse: MagicMock,
    mock_chunk: MagicMock,
    settings: Settings,
) -> None:
    chunks = _make_chunks(3)
    mock_repo, mock_qwen = _setup_mocks(
        mock_repo_cls, mock_qwen_cls, mock_boto3, mock_parse, mock_chunk, chunks=chunks
    )
    mock_qwen.embedding.side_effect = ValueError("Qwen API rate limited")
    meta = _make_upload_metadata()

    result = ingest_document("my-bucket", "docs/report.pdf", meta, settings)

    assert result.status == "failed"
    assert "Qwen API rate limited" in (result.error or "")
    mock_repo.complete_ingestion_run.assert_called_once_with(
        FAKE_RUN_ID, "failed", notes="Qwen API rate limited"
    )
    mock_repo.batch_insert_chunks.assert_not_called()


@patch("app.ingestion.chunk_document")
@patch("app.ingestion.parse_document")
@patch("app.ingestion.boto3")
@patch("app.ingestion.QwenClient")
@patch("app.ingestion.IngestionRepository")
def test_repository_insert_failure_marks_run_failed(
    mock_repo_cls: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_boto3: MagicMock,
    mock_parse: MagicMock,
    mock_chunk: MagicMock,
    settings: Settings,
) -> None:
    chunks = _make_chunks(3)
    mock_repo, _ = _setup_mocks(
        mock_repo_cls, mock_qwen_cls, mock_boto3, mock_parse, mock_chunk, chunks=chunks
    )
    mock_repo.batch_insert_chunks.side_effect = RuntimeError("DB connection lost")
    meta = _make_upload_metadata()

    result = ingest_document("my-bucket", "docs/report.pdf", meta, settings)

    assert result.status == "failed"
    assert "DB connection lost" in (result.error or "")
    mock_repo.complete_ingestion_run.assert_called_once_with(
        FAKE_RUN_ID, "failed", notes="DB connection lost"
    )
    mock_repo.bulk_index_opensearch.assert_not_called()


@patch("app.ingestion.chunk_document")
@patch("app.ingestion.parse_document")
@patch("app.ingestion.boto3")
@patch("app.ingestion.QwenClient")
@patch("app.ingestion.IngestionRepository")
def test_source_uri_fallback_to_s3_uri(
    mock_repo_cls: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_boto3: MagicMock,
    mock_parse: MagicMock,
    mock_chunk: MagicMock,
    settings: Settings,
) -> None:
    mock_repo, _ = _setup_mocks(
        mock_repo_cls, mock_qwen_cls, mock_boto3, mock_parse, mock_chunk, chunks=_make_chunks(2)
    )
    meta = _make_upload_metadata(source_uri="")

    result = ingest_document("my-bucket", "docs/report.pdf", meta, settings)

    assert result.status == "succeeded"

    upsert_call = mock_repo.upsert_document.call_args[0][0]
    assert upsert_call.source_uri == "s3://my-bucket/docs/report.pdf"

    insert_call = mock_repo.batch_insert_chunks.call_args[0][0]
    assert all(cr.citation_url == "s3://my-bucket/docs/report.pdf" for cr in insert_call)


@patch("app.ingestion.chunk_document")
@patch("app.ingestion.parse_document")
@patch("app.ingestion.boto3")
@patch("app.ingestion.QwenClient")
@patch("app.ingestion.IngestionRepository")
def test_title_from_upload_metadata_takes_precedence(
    mock_repo_cls: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_boto3: MagicMock,
    mock_parse: MagicMock,
    mock_chunk: MagicMock,
    settings: Settings,
) -> None:
    mock_repo, _ = _setup_mocks(
        mock_repo_cls,
        mock_qwen_cls,
        mock_boto3,
        mock_parse,
        mock_chunk,
        chunks=_make_chunks(1),
        parsed_title="Title From Parser",
    )
    meta = _make_upload_metadata(title="Title From Upload")

    result = ingest_document("my-bucket", "docs/report.pdf", meta, settings)

    assert result.status == "succeeded"
    upsert_call = mock_repo.upsert_document.call_args[0][0]
    assert upsert_call.title == "Title From Upload"

    insert_call = mock_repo.batch_insert_chunks.call_args[0][0]
    assert all(cr.citation_title == "Title From Upload" for cr in insert_call)


@patch("app.ingestion.chunk_document")
@patch("app.ingestion.parse_document")
@patch("app.ingestion.boto3")
@patch("app.ingestion.QwenClient")
@patch("app.ingestion.IngestionRepository")
def test_chunk_records_have_correct_fields(
    mock_repo_cls: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_boto3: MagicMock,
    mock_parse: MagicMock,
    mock_chunk: MagicMock,
    settings: Settings,
) -> None:
    chunks = _make_chunks(2)
    mock_repo, _ = _setup_mocks(
        mock_repo_cls, mock_qwen_cls, mock_boto3, mock_parse, mock_chunk, chunks=chunks
    )
    meta = _make_upload_metadata(published_year=2023, published_month=3)

    ingest_document("my-bucket", "docs/report.pdf", meta, settings)

    insert_call = mock_repo.batch_insert_chunks.call_args[0][0]
    assert len(insert_call) == 2

    cr0 = insert_call[0]
    assert cr0.doc_id == FAKE_DOC_ID
    assert cr0.doc_version == "1.0"
    assert cr0.chunk_index == 0
    assert cr0.chunk_text == "chunk text 0"
    assert cr0.citation_year == 2023
    assert cr0.citation_month == 3
    assert cr0.run_id == FAKE_RUN_ID
    assert cr0.page_start == 1
    assert cr0.section_id == "sec-0"


@patch("app.ingestion.chunk_document")
@patch("app.ingestion.parse_document")
@patch("app.ingestion.boto3")
@patch("app.ingestion.QwenClient")
@patch("app.ingestion.IngestionRepository")
def test_document_record_content_hash(
    mock_repo_cls: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_boto3: MagicMock,
    mock_parse: MagicMock,
    mock_chunk: MagicMock,
    settings: Settings,
) -> None:
    mock_repo, _ = _setup_mocks(
        mock_repo_cls, mock_qwen_cls, mock_boto3, mock_parse, mock_chunk, chunks=_make_chunks(1)
    )
    meta = _make_upload_metadata()

    ingest_document("my-bucket", "docs/report.pdf", meta, settings)

    upsert_call = mock_repo.upsert_document.call_args[0][0]
    assert upsert_call.content_hash == FILE_HASH
    assert upsert_call.source_type == "file"
    assert upsert_call.mime_type == "application/pdf"
    assert upsert_call.run_id == FAKE_RUN_ID


@patch("app.ingestion.chunk_document")
@patch("app.ingestion.parse_document")
@patch("app.ingestion.boto3")
@patch("app.ingestion.QwenClient")
@patch("app.ingestion.IngestionRepository")
def test_filename_extracted_from_s3_key(
    mock_repo_cls: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_boto3: MagicMock,
    mock_parse: MagicMock,
    mock_chunk: MagicMock,
    settings: Settings,
) -> None:
    mock_repo, _ = _setup_mocks(
        mock_repo_cls, mock_qwen_cls, mock_boto3, mock_parse, mock_chunk, chunks=[]
    )
    meta = _make_upload_metadata()

    ingest_document("my-bucket", "a/b/c/myfile.docx", meta, settings)

    mock_parse.assert_called_once_with(FILE_BYTES, "myfile.docx")
