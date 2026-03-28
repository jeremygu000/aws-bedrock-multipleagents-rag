from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from app.config import Settings
from app.ingestion_models import ChunkRecord, DocumentRecord
from app.ingestion_repository import IngestionRepository, _to_vector_literal


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        RAG_DB_HOST="localhost",
        RAG_DB_PORT=5432,
        RAG_DB_NAME="testdb",
        RAG_DB_USER="testuser",
        RAG_DB_PASSWORD="testpass",
        RAG_OPENSEARCH_ENDPOINT="https://opensearch.example.com",
        RAG_OPENSEARCH_INDEX="kb_chunks",
        RAG_AWS_REGION="ap-southeast-2",
    )


@pytest.fixture()
def repo(settings: Settings) -> IngestionRepository:
    return IngestionRepository(settings)


def make_doc(**kwargs) -> DocumentRecord:
    defaults = dict(
        source_type="crawler",
        source_uri="https://example.com/doc.pdf",
        title="Test Document",
        lang="en",
        category="general",
        mime_type="application/pdf",
        content_hash="abc123",
        doc_version="1.0",
        published_year=2024,
        published_month=6,
        metadata={},
    )
    defaults.update(kwargs)
    return DocumentRecord(**defaults)


def make_chunk(**kwargs) -> ChunkRecord:
    defaults = dict(
        doc_id=uuid.uuid4(),
        doc_version="1.0",
        chunk_index=0,
        chunk_text="Sample chunk text.",
        token_count=10,
        citation_url="https://example.com/doc.pdf",
        citation_title="Test Document",
        citation_year=2024,
        citation_month=6,
        embedding=[0.1, 0.2, 0.3],
        metadata={},
    )
    defaults.update(kwargs)
    return ChunkRecord(**defaults)


class TestVectorLiteralFormat:
    def test_vector_literal_format(self):
        result = _to_vector_literal([0.1, 0.2, 0.3])
        assert result.startswith("[")
        assert result.endswith("]")
        assert "0.100000000" in result
        assert "0.200000000" in result
        assert "0.300000000" in result

    def test_vector_literal_empty(self):
        result = _to_vector_literal([])
        assert result == "[]"

    def test_vector_literal_single(self):
        result = _to_vector_literal([1.5])
        assert result == "[1.500000000]"


class TestGetEngine:
    def test_get_engine_caches(self, repo: IngestionRepository, settings: Settings):
        mock_engine = MagicMock()
        with patch("app.ingestion_repository.create_engine", return_value=mock_engine) as mock_ce:
            e1 = repo._get_engine()
            e2 = repo._get_engine()
            assert e1 is e2
            mock_ce.assert_called_once()

    def test_get_engine_calls_resolve_password(self, repo: IngestionRepository):
        mock_engine = MagicMock()
        with patch(
            "app.ingestion_repository.resolve_db_password", return_value="secret"
        ) as mock_rp:
            with patch("app.ingestion_repository.create_engine", return_value=mock_engine):
                repo._get_engine()
                mock_rp.assert_called_once_with(repo._settings)

    def test_get_engine_raises_without_password(self, settings: Settings):
        settings = Settings(
            RAG_DB_HOST="localhost",
            RAG_DB_PASSWORD="",
        )
        repo = IngestionRepository(settings)
        with patch("app.ingestion_repository.resolve_db_password", return_value=""):
            with pytest.raises(ValueError, match="DB password is not configured"):
                repo._get_engine()


class TestCreateIngestionRun:
    def _make_mock_conn(self):
        mock_conn = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_ctx
        return mock_engine, mock_conn

    def test_create_ingestion_run_returns_uuid(self, repo: IngestionRepository):
        mock_engine, mock_conn = self._make_mock_conn()
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            run_id = repo.create_ingestion_run("crawler", {"key": "val"})
            assert isinstance(run_id, UUID)

    def test_create_ingestion_run_source_type(self, repo: IngestionRepository):
        mock_engine, mock_conn = self._make_mock_conn()
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            repo.create_ingestion_run("s3_upload", {})
            mock_conn.execute.assert_called_once()
            stmt, *_ = mock_conn.execute.call_args[0]
            compiled = str(stmt.compile())
            assert "ingestion_runs" in compiled

    def test_create_ingestion_run_different_uuids(self, repo: IngestionRepository):
        mock_engine, mock_conn = self._make_mock_conn()
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            id1 = repo.create_ingestion_run("crawler", {})
            id2 = repo.create_ingestion_run("crawler", {})
            assert id1 != id2


class TestCompleteIngestionRun:
    def _make_mock_conn(self):
        mock_conn = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_ctx
        return mock_engine, mock_conn

    def test_complete_ingestion_run_succeeded(self, repo: IngestionRepository):
        mock_engine, mock_conn = self._make_mock_conn()
        run_id = uuid.uuid4()
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            repo.complete_ingestion_run(run_id, "succeeded")
            mock_conn.execute.assert_called_once()

    def test_complete_ingestion_run_failed(self, repo: IngestionRepository):
        mock_engine, mock_conn = self._make_mock_conn()
        run_id = uuid.uuid4()
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            repo.complete_ingestion_run(run_id, "failed", notes="error occurred")
            mock_conn.execute.assert_called_once()

    def test_complete_ingestion_run_invalid_status(self, repo: IngestionRepository):
        run_id = uuid.uuid4()
        with pytest.raises(ValueError, match="Invalid completion status"):
            repo.complete_ingestion_run(run_id, "pending")

    def test_complete_ingestion_run_invalid_status_running(self, repo: IngestionRepository):
        run_id = uuid.uuid4()
        with pytest.raises(ValueError, match="Invalid completion status"):
            repo.complete_ingestion_run(run_id, "running")


class TestUpsertDocument:
    def _make_mock_conn_with_row(self, returned_uuid: UUID):
        mock_row = MagicMock()
        mock_row.__getitem__ = MagicMock(return_value=str(returned_uuid))
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_ctx
        return mock_engine, mock_conn

    def test_upsert_document_new(self, repo: IngestionRepository):
        expected_id = uuid.uuid4()
        mock_engine, mock_conn = self._make_mock_conn_with_row(expected_id)
        doc = make_doc()
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            result = repo.upsert_document(doc)
            assert isinstance(result, UUID)
            assert result == expected_id

    def test_upsert_document_metadata_includes_year_month(self, repo: IngestionRepository):
        expected_id = uuid.uuid4()
        mock_engine, mock_conn = self._make_mock_conn_with_row(expected_id)
        doc = make_doc(published_year=2023, published_month=11, metadata={"extra": "data"})
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            repo.upsert_document(doc)
            call_args = mock_conn.execute.call_args
            params = call_args[0][1]
            meta_dict = json.loads(params["metadata"])
            assert meta_dict["year"] == "2023"
            assert meta_dict["month"] == "11"
            assert meta_dict["extra"] == "data"

    def test_upsert_document_raises_on_no_row(self, repo: IngestionRepository):
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_ctx
        doc = make_doc()
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            with pytest.raises(RuntimeError, match="RETURNING doc_id returned no rows"):
                repo.upsert_document(doc)


class TestDeleteChunksForDoc:
    def _make_mock_conn_with_rowcount(self, rowcount: int):
        mock_result = MagicMock()
        mock_result.rowcount = rowcount
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_ctx
        return mock_engine, mock_conn

    def test_delete_chunks_returns_count(self, repo: IngestionRepository):
        mock_engine, _ = self._make_mock_conn_with_rowcount(5)
        doc_id = uuid.uuid4()
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            count = repo.delete_chunks_for_doc(doc_id, "1.0")
            assert count == 5

    def test_delete_chunks_zero(self, repo: IngestionRepository):
        mock_engine, _ = self._make_mock_conn_with_rowcount(0)
        doc_id = uuid.uuid4()
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            count = repo.delete_chunks_for_doc(doc_id, "2.0")
            assert count == 0


class TestBatchInsertChunks:
    def _make_mock_engine(self):
        mock_conn = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_ctx
        return mock_engine, mock_conn

    def test_batch_insert_chunks_empty(self, repo: IngestionRepository):
        count = repo.batch_insert_chunks([])
        assert count == 0

    def test_batch_insert_chunks_single(self, repo: IngestionRepository):
        mock_engine, mock_conn = self._make_mock_engine()
        chunk = make_chunk()
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            count = repo.batch_insert_chunks([chunk])
            assert count == 1
            mock_conn.execute.assert_called_once()

    def test_batch_insert_chunks_multiple(self, repo: IngestionRepository):
        mock_engine, mock_conn = self._make_mock_engine()
        chunks = [make_chunk(chunk_index=i) for i in range(3)]
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            count = repo.batch_insert_chunks(chunks)
            assert count == 3
            assert mock_conn.execute.call_count == 3

    def test_batch_insert_chunks_embedding_format(self, repo: IngestionRepository):
        mock_engine, mock_conn = self._make_mock_engine()
        chunk = make_chunk(embedding=[1.0, 2.0, 3.0])
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            repo.batch_insert_chunks([chunk])
            call_args = mock_conn.execute.call_args
            params = call_args[0][1]
            emb = params["embedding"]
            assert emb.startswith("[")
            assert emb.endswith("]")
            assert "1.000000000" in emb
            assert "2.000000000" in emb
            assert "3.000000000" in emb

    def test_batch_insert_chunks_metadata_year_month(self, repo: IngestionRepository):
        mock_engine, mock_conn = self._make_mock_engine()
        chunk = make_chunk(citation_year=2022, citation_month=3, metadata={"source": "test"})
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            repo.batch_insert_chunks([chunk])
            call_args = mock_conn.execute.call_args
            params = call_args[0][1]
            meta_dict = json.loads(params["metadata"])
            assert meta_dict["year"] == "2022"
            assert meta_dict["month"] == "3"
            assert meta_dict["source"] == "test"

    def test_batch_insert_chunks_unique_chunk_ids(self, repo: IngestionRepository):
        mock_engine, mock_conn = self._make_mock_engine()
        chunks = [make_chunk(chunk_index=i) for i in range(2)]
        captured_ids = []

        def capture_execute(stmt, params):
            captured_ids.append(params["chunk_id"])

        mock_conn.execute.side_effect = capture_execute
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            repo.batch_insert_chunks(chunks)
        assert len(set(captured_ids)) == 2


class TestBulkIndexOpensearch:
    def _make_settings_no_endpoint(self) -> Settings:
        return Settings(
            RAG_DB_PASSWORD="pass",
            RAG_OPENSEARCH_ENDPOINT="",
        )

    def test_bulk_index_opensearch_no_endpoint(self):
        settings = self._make_settings_no_endpoint()
        repo = IngestionRepository(settings)
        chunk = make_chunk()
        with patch("app.ingestion_repository.helpers") as mock_helpers:
            repo.bulk_index_opensearch([chunk])
            mock_helpers.bulk.assert_not_called()

    def test_bulk_index_opensearch_empty_chunks(self, repo: IngestionRepository):
        with patch("app.ingestion_repository.helpers") as mock_helpers:
            repo.bulk_index_opensearch([])
            mock_helpers.bulk.assert_not_called()

    def test_bulk_index_opensearch_success(self, repo: IngestionRepository):
        mock_client = MagicMock()
        chunk = make_chunk(
            chunk_text="hello world",
            citation_url="https://ex.com",
            citation_title="Title",
            citation_year=2024,
            citation_month=1,
        )
        with patch.object(repo, "_get_opensearch_client", return_value=mock_client):
            with patch("app.ingestion_repository.helpers") as mock_helpers:
                repo.bulk_index_opensearch([chunk])
                mock_helpers.bulk.assert_called_once()
                _, actions = mock_helpers.bulk.call_args[0]
                assert len(actions) == 1
                action = actions[0]
                assert action["_index"] == repo._settings.opensearch_index
                assert "_id" in action
                source = action["_source"]
                assert source["chunk_text"] == "hello world"
                assert source["citation_url"] == "https://ex.com"
                assert source["citation_year"] == 2024

    def test_bulk_index_opensearch_doc_ids_override(self, repo: IngestionRepository):
        mock_client = MagicMock()
        override_doc_id = uuid.uuid4()
        chunk = make_chunk(chunk_index=0)
        with patch.object(repo, "_get_opensearch_client", return_value=mock_client):
            with patch("app.ingestion_repository.helpers") as mock_helpers:
                repo.bulk_index_opensearch([chunk], doc_ids={0: override_doc_id})
                _, actions = mock_helpers.bulk.call_args[0]
                assert actions[0]["_source"]["doc_id"] == str(override_doc_id)


class TestDeleteAllChunksForDoc:
    def _make_mock_conn_with_rowcount(self, rowcount: int):
        mock_result = MagicMock()
        mock_result.rowcount = rowcount
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_ctx
        return mock_engine, mock_conn

    def test_delete_all_chunks_returns_count(self, repo: IngestionRepository):
        mock_engine, _ = self._make_mock_conn_with_rowcount(5)
        doc_id = uuid.uuid4()
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            count = repo.delete_all_chunks_for_doc(doc_id)
            assert count == 5

    def test_delete_all_chunks_zero(self, repo: IngestionRepository):
        mock_engine, _ = self._make_mock_conn_with_rowcount(0)
        doc_id = uuid.uuid4()
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            count = repo.delete_all_chunks_for_doc(doc_id)
            assert count == 0


class TestDeleteDocumentRecord:
    def _make_mock_conn_with_rowcount(self, rowcount: int):
        mock_result = MagicMock()
        mock_result.rowcount = rowcount
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_ctx
        return mock_engine, mock_conn

    def test_delete_document_record_true(self, repo: IngestionRepository):
        mock_engine, _ = self._make_mock_conn_with_rowcount(1)
        doc_id = uuid.uuid4()
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            result = repo.delete_document_record(doc_id)
            assert result is True

    def test_delete_document_record_false(self, repo: IngestionRepository):
        mock_engine, _ = self._make_mock_conn_with_rowcount(0)
        doc_id = uuid.uuid4()
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            result = repo.delete_document_record(doc_id)
            assert result is False


class TestGetChunkIdsForDoc:
    def _make_mock_conn_with_fetchall(self, rows):
        mock_result = MagicMock()
        mock_result.fetchall.return_value = rows
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.begin.return_value = mock_ctx
        return mock_engine, mock_conn

    def test_get_chunk_ids_for_doc_returns_composite_ids(self, repo: IngestionRepository):
        doc_id = uuid.uuid4()
        row0 = MagicMock()
        row0.chunk_index = 0
        row1 = MagicMock()
        row1.chunk_index = 1
        row2 = MagicMock()
        row2.chunk_index = 2
        mock_engine, _ = self._make_mock_conn_with_fetchall([row0, row1, row2])
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            ids = repo.get_chunk_ids_for_doc(doc_id)
            assert ids == [
                f"{doc_id}_chunk_0",
                f"{doc_id}_chunk_1",
                f"{doc_id}_chunk_2",
            ]

    def test_get_chunk_ids_for_doc_empty(self, repo: IngestionRepository):
        doc_id = uuid.uuid4()
        mock_engine, _ = self._make_mock_conn_with_fetchall([])
        with patch.object(repo, "_get_engine", return_value=mock_engine):
            ids = repo.get_chunk_ids_for_doc(doc_id)
            assert ids == []


class TestDeleteOpensearchDocsByDocId:
    def test_delete_opensearch_no_endpoint_returns_zero(self):
        settings = Settings(
            RAG_DB_PASSWORD="pass",
            RAG_OPENSEARCH_ENDPOINT="",
        )
        repo = IngestionRepository(settings)
        doc_id = uuid.uuid4()
        with patch.object(repo, "_get_opensearch_client") as mock_get_client:
            count = repo.delete_opensearch_docs_by_doc_id(doc_id)
            assert count == 0
            mock_get_client.assert_not_called()

    def test_delete_opensearch_returns_deleted_count(self, repo: IngestionRepository):
        mock_client = MagicMock()
        mock_client.delete_by_query.return_value = {"deleted": 3}
        doc_id = uuid.uuid4()
        with patch.object(repo, "_get_opensearch_client", return_value=mock_client):
            count = repo.delete_opensearch_docs_by_doc_id(doc_id)
            assert count == 3
