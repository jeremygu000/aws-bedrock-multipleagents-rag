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
    mock_repo, _ = _setup_mocks(mock_repo_cls, mock_qwen_cls, mock_boto3, mock_parse, mock_chunk)
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
    mock_repo, _ = _setup_mocks(mock_repo_cls, mock_qwen_cls, mock_boto3, mock_parse, mock_chunk)
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


# ---------------------------------------------------------------------------
# Phase 2.5 — Entity extraction pipeline tests
# ---------------------------------------------------------------------------


def _make_extraction_result(chunk_id: str, n_entities: int = 2, n_relations: int = 1) -> MagicMock:
    from app.entity_extraction_models import EntityType, RelationType

    entities = []
    for i in range(n_entities):
        e = MagicMock()
        e.entity_id = f"{chunk_id}_ent_{i}"
        e.name = f"Entity {i}"
        e.type = EntityType.WORK
        e.description = f"Description for entity {i}"
        e.canonical_key = f"entity_{i}"
        e.aliases = []
        e.confidence = 0.9
        e.source_chunk_ids = [chunk_id]
        entities.append(e)

    relations = []
    for i in range(n_relations):
        r = MagicMock()
        r.source_entity_id = entities[0].entity_id if entities else f"{chunk_id}_ent_0"
        r.target_entity_id = entities[-1].entity_id if entities else f"{chunk_id}_ent_1"
        r.type = RelationType.REFERENCES
        r.evidence = f"Evidence text {i}"
        r.confidence = 0.8
        r.weight = 1.0
        r.source_chunk_ids = [chunk_id]
        relations.append(r)

    result = MagicMock()
    result.chunk_id = chunk_id
    result.entities = entities
    result.relations = relations
    return result


def _entity_extraction_settings() -> Settings:
    return Settings(
        RAG_DB_HOST="localhost",
        RAG_DB_PASSWORD="testpass",
        RAG_S3_BUCKET="test-bucket",
        RAG_CHUNK_SIZE=512,
        RAG_CHUNK_OVERLAP=64,
        RAG_CHUNK_MIN_SIZE=50,
        RAG_EMBED_BATCH_SIZE=10,
        RAG_ENABLE_ENTITY_EXTRACTION=True,
        RAG_ENABLE_NEO4J=True,
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_USERNAME="neo4j",
        NEO4J_PASSWORD="test",
        NEO4J_DATABASE="neo4j",
    )


ENTITY_EXTRACTION_PATCHES = [
    "app.ingestion.chunk_document",
    "app.ingestion.parse_document",
    "app.ingestion.boto3",
    "app.ingestion.QwenClient",
    "app.ingestion.IngestionRepository",
    "app.ingestion.EntityExtractor",
    "app.ingestion.EntityDeduplicator",
    "app.ingestion.EntityVectorStore",
    "app.ingestion.Neo4jRepository",
    "app.ingestion.resolve_neo4j_password",
]


def _apply_entity_patches(func):
    for p in reversed(ENTITY_EXTRACTION_PATCHES):
        func = patch(p)(func)
    return func


def _setup_entity_mocks(
    mock_resolve_pw: MagicMock,
    mock_neo4j_cls: MagicMock,
    mock_vector_cls: MagicMock,
    mock_dedup_cls: MagicMock,
    mock_extractor_cls: MagicMock,
    mock_repo_cls: MagicMock,
    mock_qwen_cls: MagicMock,
    mock_boto3: MagicMock,
    mock_parse: MagicMock,
    mock_chunk: MagicMock,
    chunks: list[_FakeChunk] | None = None,
    n_entities: int = 2,
    n_relations: int = 1,
) -> dict[str, MagicMock]:
    if chunks is None:
        chunks = _make_chunks(3)

    mock_repo, mock_qwen = _setup_mocks(
        mock_repo_cls, mock_qwen_cls, mock_boto3, mock_parse, mock_chunk, chunks=chunks
    )

    chunk_results = [
        _make_extraction_result(
            f"{FAKE_DOC_ID}_chunk_{c.chunk_index}",
            n_entities=n_entities,
            n_relations=n_relations,
        )
        for c in chunks
    ]

    mock_extractor = MagicMock()
    mock_extractor.extract.side_effect = [(cr, MagicMock()) for cr in chunk_results]
    mock_extractor_cls.return_value = mock_extractor

    merged_entities = []
    merged_relations = []
    for cr in chunk_results:
        merged_entities.extend(cr.entities)
        merged_relations.extend(cr.relations)

    mock_dedup = MagicMock()
    mock_dedup.merge_entities.return_value = merged_entities
    mock_dedup.merge_relations.return_value = merged_relations
    mock_dedup_cls.return_value = mock_dedup

    mock_vector = MagicMock()
    mock_vector.batch_upsert_entities.return_value = len(merged_entities)
    mock_vector.batch_upsert_relations.return_value = len(merged_relations)
    mock_vector_cls.return_value = mock_vector

    mock_neo4j = MagicMock()
    mock_neo4j.upsert_batch.return_value = {
        "entities_written": len(merged_entities),
        "relations_written": len(merged_relations),
    }
    mock_neo4j_cls.return_value = mock_neo4j

    mock_resolve_pw.return_value = "neo4j-password"

    return {
        "repo": mock_repo,
        "qwen": mock_qwen,
        "extractor": mock_extractor,
        "extractor_cls": mock_extractor_cls,
        "dedup": mock_dedup,
        "dedup_cls": mock_dedup_cls,
        "vector": mock_vector,
        "vector_cls": mock_vector_cls,
        "neo4j": mock_neo4j,
        "neo4j_cls": mock_neo4j_cls,
        "resolve_pw": mock_resolve_pw,
        "chunk_results": chunk_results,
        "merged_entities": merged_entities,
        "merged_relations": merged_relations,
    }


@_apply_entity_patches
def test_entity_extraction_happy_path(
    mock_resolve_pw,
    mock_neo4j_cls,
    mock_vector_cls,
    mock_dedup_cls,
    mock_extractor_cls,
    mock_repo_cls,
    mock_qwen_cls,
    mock_boto3,
    mock_parse,
    mock_chunk,
):
    settings = _entity_extraction_settings()
    chunks = _make_chunks(3)
    mocks = _setup_entity_mocks(
        mock_resolve_pw,
        mock_neo4j_cls,
        mock_vector_cls,
        mock_dedup_cls,
        mock_extractor_cls,
        mock_repo_cls,
        mock_qwen_cls,
        mock_boto3,
        mock_parse,
        mock_chunk,
        chunks=chunks,
    )
    meta = _make_upload_metadata()

    result = ingest_document("my-bucket", "docs/report.pdf", meta, settings)

    assert result.status == "succeeded"
    assert result.chunks_created == 3
    assert result.entity_count > 0
    assert result.relation_count > 0

    assert mocks["extractor"].extract.call_count == 3
    mocks["dedup"].merge_entities.assert_called_once()
    mocks["dedup"].merge_relations.assert_called_once()
    mocks["neo4j"].upsert_batch.assert_called_once()
    mocks["neo4j"].close.assert_called_once()
    mocks["vector"].batch_upsert_entities.assert_called_once()
    mocks["vector"].batch_upsert_relations.assert_called_once()
    mocks["repo"].complete_ingestion_run.assert_called_once_with(FAKE_RUN_ID, "succeeded")


@_apply_entity_patches
def test_entity_extraction_disabled_skips_pipeline(
    mock_resolve_pw,
    mock_neo4j_cls,
    mock_vector_cls,
    mock_dedup_cls,
    mock_extractor_cls,
    mock_repo_cls,
    mock_qwen_cls,
    mock_boto3,
    mock_parse,
    mock_chunk,
):
    settings = Settings(
        RAG_DB_HOST="localhost",
        RAG_DB_PASSWORD="testpass",
        RAG_S3_BUCKET="test-bucket",
        RAG_CHUNK_SIZE=512,
        RAG_CHUNK_OVERLAP=64,
        RAG_CHUNK_MIN_SIZE=50,
        RAG_EMBED_BATCH_SIZE=10,
        RAG_ENABLE_ENTITY_EXTRACTION=False,
    )
    chunks = _make_chunks(3)
    _setup_entity_mocks(
        mock_resolve_pw,
        mock_neo4j_cls,
        mock_vector_cls,
        mock_dedup_cls,
        mock_extractor_cls,
        mock_repo_cls,
        mock_qwen_cls,
        mock_boto3,
        mock_parse,
        mock_chunk,
        chunks=chunks,
    )
    meta = _make_upload_metadata()

    result = ingest_document("my-bucket", "docs/report.pdf", meta, settings)

    assert result.status == "succeeded"
    assert result.entity_count == 0
    assert result.relation_count == 0
    mock_extractor_cls.assert_not_called()
    mock_neo4j_cls.assert_not_called()
    mock_vector_cls.assert_not_called()


@_apply_entity_patches
def test_entity_extraction_chunk_failure_continues(
    mock_resolve_pw,
    mock_neo4j_cls,
    mock_vector_cls,
    mock_dedup_cls,
    mock_extractor_cls,
    mock_repo_cls,
    mock_qwen_cls,
    mock_boto3,
    mock_parse,
    mock_chunk,
):
    settings = _entity_extraction_settings()
    chunks = _make_chunks(3)
    mocks = _setup_entity_mocks(
        mock_resolve_pw,
        mock_neo4j_cls,
        mock_vector_cls,
        mock_dedup_cls,
        mock_extractor_cls,
        mock_repo_cls,
        mock_qwen_cls,
        mock_boto3,
        mock_parse,
        mock_chunk,
        chunks=chunks,
    )

    good_result = _make_extraction_result(
        f"{FAKE_DOC_ID}_chunk_0",
        n_entities=2,
        n_relations=1,
    )
    good_tuple = (good_result, MagicMock())
    mocks["extractor"].extract.side_effect = [
        good_tuple,
        RuntimeError("LLM timeout"),
        good_tuple,
    ]

    merged_entities = good_result.entities * 2
    merged_relations = good_result.relations * 2
    mocks["dedup"].merge_entities.return_value = merged_entities
    mocks["dedup"].merge_relations.return_value = merged_relations
    mocks["vector"].batch_upsert_entities.return_value = len(merged_entities)
    mocks["vector"].batch_upsert_relations.return_value = len(merged_relations)
    mocks["neo4j"].upsert_batch.return_value = {
        "entities_written": len(merged_entities),
        "relations_written": len(merged_relations),
    }

    meta = _make_upload_metadata()
    result = ingest_document("my-bucket", "docs/report.pdf", meta, settings)

    assert result.status == "succeeded"
    assert result.entity_count > 0
    assert mocks["extractor"].extract.call_count == 3


@_apply_entity_patches
def test_entity_extraction_all_chunks_fail_returns_zero(
    mock_resolve_pw,
    mock_neo4j_cls,
    mock_vector_cls,
    mock_dedup_cls,
    mock_extractor_cls,
    mock_repo_cls,
    mock_qwen_cls,
    mock_boto3,
    mock_parse,
    mock_chunk,
):
    settings = _entity_extraction_settings()
    chunks = _make_chunks(2)
    mocks = _setup_entity_mocks(
        mock_resolve_pw,
        mock_neo4j_cls,
        mock_vector_cls,
        mock_dedup_cls,
        mock_extractor_cls,
        mock_repo_cls,
        mock_qwen_cls,
        mock_boto3,
        mock_parse,
        mock_chunk,
        chunks=chunks,
    )
    mocks["extractor"].extract.side_effect = RuntimeError("All fail")
    meta = _make_upload_metadata()

    result = ingest_document("my-bucket", "docs/report.pdf", meta, settings)

    assert result.status == "succeeded"
    assert result.entity_count == 0
    assert result.relation_count == 0
    mocks["neo4j_cls"].assert_not_called()
    mocks["vector_cls"].assert_not_called()


@_apply_entity_patches
def test_entity_extraction_neo4j_failure_continues_to_pgvector(
    mock_resolve_pw,
    mock_neo4j_cls,
    mock_vector_cls,
    mock_dedup_cls,
    mock_extractor_cls,
    mock_repo_cls,
    mock_qwen_cls,
    mock_boto3,
    mock_parse,
    mock_chunk,
):
    settings = _entity_extraction_settings()
    chunks = _make_chunks(2)
    mocks = _setup_entity_mocks(
        mock_resolve_pw,
        mock_neo4j_cls,
        mock_vector_cls,
        mock_dedup_cls,
        mock_extractor_cls,
        mock_repo_cls,
        mock_qwen_cls,
        mock_boto3,
        mock_parse,
        mock_chunk,
        chunks=chunks,
    )
    mocks["neo4j"].upsert_batch.side_effect = ConnectionError("Neo4j down")
    meta = _make_upload_metadata()

    result = ingest_document("my-bucket", "docs/report.pdf", meta, settings)

    assert result.status == "succeeded"
    assert result.entity_count > 0
    mocks["vector"].batch_upsert_entities.assert_called_once()
    mocks["vector"].batch_upsert_relations.assert_called_once()


@_apply_entity_patches
def test_entity_extraction_pgvector_failure_still_succeeds(
    mock_resolve_pw,
    mock_neo4j_cls,
    mock_vector_cls,
    mock_dedup_cls,
    mock_extractor_cls,
    mock_repo_cls,
    mock_qwen_cls,
    mock_boto3,
    mock_parse,
    mock_chunk,
):
    settings = _entity_extraction_settings()
    chunks = _make_chunks(2)
    mocks = _setup_entity_mocks(
        mock_resolve_pw,
        mock_neo4j_cls,
        mock_vector_cls,
        mock_dedup_cls,
        mock_extractor_cls,
        mock_repo_cls,
        mock_qwen_cls,
        mock_boto3,
        mock_parse,
        mock_chunk,
        chunks=chunks,
    )
    mocks["vector_cls"].side_effect = RuntimeError("pgvector connection failed")
    meta = _make_upload_metadata()

    result = ingest_document("my-bucket", "docs/report.pdf", meta, settings)

    assert result.status == "succeeded"
    mocks["neo4j"].upsert_batch.assert_called_once()


@_apply_entity_patches
def test_entity_extraction_neo4j_disabled_skips_graph(
    mock_resolve_pw,
    mock_neo4j_cls,
    mock_vector_cls,
    mock_dedup_cls,
    mock_extractor_cls,
    mock_repo_cls,
    mock_qwen_cls,
    mock_boto3,
    mock_parse,
    mock_chunk,
):
    settings = Settings(
        RAG_DB_HOST="localhost",
        RAG_DB_PASSWORD="testpass",
        RAG_S3_BUCKET="test-bucket",
        RAG_CHUNK_SIZE=512,
        RAG_CHUNK_OVERLAP=64,
        RAG_CHUNK_MIN_SIZE=50,
        RAG_EMBED_BATCH_SIZE=10,
        RAG_ENABLE_ENTITY_EXTRACTION=True,
        RAG_ENABLE_NEO4J=False,
    )
    chunks = _make_chunks(2)
    mocks = _setup_entity_mocks(
        mock_resolve_pw,
        mock_neo4j_cls,
        mock_vector_cls,
        mock_dedup_cls,
        mock_extractor_cls,
        mock_repo_cls,
        mock_qwen_cls,
        mock_boto3,
        mock_parse,
        mock_chunk,
        chunks=chunks,
    )
    meta = _make_upload_metadata()

    result = ingest_document("my-bucket", "docs/report.pdf", meta, settings)

    assert result.status == "succeeded"
    assert result.entity_count > 0
    mock_neo4j_cls.assert_not_called()
    mock_resolve_pw.assert_not_called()
    mocks["vector"].batch_upsert_entities.assert_called_once()


@_apply_entity_patches
def test_entity_extraction_pipeline_crash_does_not_fail_ingestion(
    mock_resolve_pw,
    mock_neo4j_cls,
    mock_vector_cls,
    mock_dedup_cls,
    mock_extractor_cls,
    mock_repo_cls,
    mock_qwen_cls,
    mock_boto3,
    mock_parse,
    mock_chunk,
):
    settings = _entity_extraction_settings()
    chunks = _make_chunks(2)
    mocks = _setup_entity_mocks(
        mock_resolve_pw,
        mock_neo4j_cls,
        mock_vector_cls,
        mock_dedup_cls,
        mock_extractor_cls,
        mock_repo_cls,
        mock_qwen_cls,
        mock_boto3,
        mock_parse,
        mock_chunk,
        chunks=chunks,
    )
    mocks["dedup"].merge_entities.side_effect = Exception("Unexpected crash in dedup")
    meta = _make_upload_metadata()

    result = ingest_document("my-bucket", "docs/report.pdf", meta, settings)

    assert result.status == "succeeded"
    assert result.chunks_created == 2
    assert result.entity_count == 0
    assert result.relation_count == 0
    mocks["repo"].complete_ingestion_run.assert_called_once_with(FAKE_RUN_ID, "succeeded")


@_apply_entity_patches
def test_entity_extraction_empty_chunks_skips_pipeline(
    mock_resolve_pw,
    mock_neo4j_cls,
    mock_vector_cls,
    mock_dedup_cls,
    mock_extractor_cls,
    mock_repo_cls,
    mock_qwen_cls,
    mock_boto3,
    mock_parse,
    mock_chunk,
):
    settings = _entity_extraction_settings()
    _setup_entity_mocks(
        mock_resolve_pw,
        mock_neo4j_cls,
        mock_vector_cls,
        mock_dedup_cls,
        mock_extractor_cls,
        mock_repo_cls,
        mock_qwen_cls,
        mock_boto3,
        mock_parse,
        mock_chunk,
        chunks=[],
    )
    meta = _make_upload_metadata()

    result = ingest_document("my-bucket", "docs/empty.pdf", meta, settings)

    assert result.status == "succeeded"
    assert result.entity_count == 0
    assert result.relation_count == 0
    mock_extractor_cls.assert_not_called()
