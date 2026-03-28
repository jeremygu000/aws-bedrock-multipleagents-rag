from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from app.config import Settings
from app.document_manager import delete_document
from app.ingestion_models import DeleteDocumentResponse


def _make_settings(**overrides) -> Settings:
    defaults = {
        "RAG_DB_HOST": "localhost",
        "RAG_DB_PASSWORD": "test",
        "RAG_ENABLE_NEO4J": "true",
        "RAG_NEO4J_URI": "bolt://localhost:7687",
        "RAG_NEO4J_PASSWORD": "test",
    }
    defaults.update(overrides)
    return Settings(**defaults)


class TestDeleteDocumentCascade:
    @patch("app.document_manager.QueryCache")
    @patch("app.document_manager.EntityVectorStore")
    @patch("app.document_manager.resolve_neo4j_password")
    @patch("app.document_manager.Neo4jRepository")
    @patch("app.document_manager.IngestionRepository")
    def test_full_cascade_with_neo4j(
        self,
        mock_ingestion_repo_cls,
        mock_neo4j_repo_cls,
        mock_resolve_pw,
        mock_vector_store_cls,
        mock_query_cache_cls,
    ):
        doc_id = uuid.uuid4()
        chunk_ids = [f"{doc_id}_chunk_0", f"{doc_id}_chunk_1"]

        mock_repo = MagicMock()
        mock_repo.get_chunk_ids_for_doc.return_value = chunk_ids
        mock_repo.delete_all_chunks_for_doc.return_value = 2
        mock_repo.delete_opensearch_docs_by_doc_id.return_value = 2
        mock_repo.delete_document_record.return_value = True
        mock_ingestion_repo_cls.return_value = mock_repo

        mock_neo4j = MagicMock()
        mock_neo4j.delete_relations_by_source_chunks.return_value = 2
        mock_neo4j.delete_entities_by_source_chunks.return_value = 3
        mock_neo4j.cleanup_orphan_relations.return_value = 0
        mock_neo4j_repo_cls.return_value = mock_neo4j

        mock_resolve_pw.return_value = "test"

        mock_vector_store = MagicMock()
        mock_vector_store.delete_entities_by_source_chunks.return_value = 1
        mock_vector_store_cls.return_value = mock_vector_store

        mock_cache = MagicMock()
        mock_query_cache_cls.return_value = mock_cache

        settings = _make_settings()
        result = delete_document(doc_id, settings)

        mock_neo4j.delete_relations_by_source_chunks.assert_called_once_with(chunk_ids)
        mock_neo4j.delete_entities_by_source_chunks.assert_called_once_with(chunk_ids)
        mock_neo4j.cleanup_orphan_relations.assert_called_once()
        mock_neo4j.close.assert_called_once()
        mock_vector_store.delete_entities_by_source_chunks.assert_called_once_with(chunk_ids)
        mock_repo.delete_all_chunks_for_doc.assert_called_once_with(doc_id)
        mock_repo.delete_opensearch_docs_by_doc_id.assert_called_once_with(doc_id)
        mock_cache.invalidate_by_doc.assert_called_once_with(str(doc_id))
        mock_repo.delete_document_record.assert_called_once_with(doc_id)

        assert isinstance(result, DeleteDocumentResponse)
        assert result.chunks_deleted == 2
        assert result.relations_deleted == 2
        assert result.entities_deleted == 4
        assert result.cache_invalidated is True

    @patch("app.document_manager.QueryCache")
    @patch("app.document_manager.EntityVectorStore")
    @patch("app.document_manager.resolve_neo4j_password")
    @patch("app.document_manager.Neo4jRepository")
    @patch("app.document_manager.IngestionRepository")
    def test_cascade_without_neo4j(
        self,
        mock_ingestion_repo_cls,
        mock_neo4j_repo_cls,
        mock_resolve_pw,
        mock_vector_store_cls,
        mock_query_cache_cls,
    ):
        doc_id = uuid.uuid4()
        chunk_ids = [f"{doc_id}_chunk_0"]

        mock_repo = MagicMock()
        mock_repo.get_chunk_ids_for_doc.return_value = chunk_ids
        mock_repo.delete_all_chunks_for_doc.return_value = 1
        mock_repo.delete_opensearch_docs_by_doc_id.return_value = 1
        mock_repo.delete_document_record.return_value = True
        mock_ingestion_repo_cls.return_value = mock_repo

        mock_vector_store = MagicMock()
        mock_vector_store.delete_entities_by_source_chunks.return_value = 0
        mock_vector_store_cls.return_value = mock_vector_store

        mock_cache = MagicMock()
        mock_query_cache_cls.return_value = mock_cache

        settings = _make_settings(RAG_ENABLE_NEO4J="false")
        delete_document(doc_id, settings)

        mock_neo4j_repo_cls.assert_not_called()
        mock_vector_store.delete_entities_by_source_chunks.assert_called_once_with(chunk_ids)
        mock_repo.delete_all_chunks_for_doc.assert_called_once_with(doc_id)
        mock_repo.delete_opensearch_docs_by_doc_id.assert_called_once_with(doc_id)

    @patch("app.document_manager.QueryCache")
    @patch("app.document_manager.EntityVectorStore")
    @patch("app.document_manager.resolve_neo4j_password")
    @patch("app.document_manager.Neo4jRepository")
    @patch("app.document_manager.IngestionRepository")
    def test_cascade_no_chunks(
        self,
        mock_ingestion_repo_cls,
        mock_neo4j_repo_cls,
        mock_resolve_pw,
        mock_vector_store_cls,
        mock_query_cache_cls,
    ):
        doc_id = uuid.uuid4()

        mock_repo = MagicMock()
        mock_repo.get_chunk_ids_for_doc.return_value = []
        mock_repo.delete_all_chunks_for_doc.return_value = 0
        mock_repo.delete_opensearch_docs_by_doc_id.return_value = 0
        mock_repo.delete_document_record.return_value = False
        mock_ingestion_repo_cls.return_value = mock_repo

        mock_cache = MagicMock()
        mock_query_cache_cls.return_value = mock_cache

        settings = _make_settings()
        delete_document(doc_id, settings)

        mock_neo4j_repo_cls.assert_not_called()
        mock_vector_store_cls.assert_not_called()
        mock_repo.delete_all_chunks_for_doc.assert_called_once_with(doc_id)
        mock_repo.delete_opensearch_docs_by_doc_id.assert_called_once_with(doc_id)
        mock_cache.invalidate_by_doc.assert_called_once_with(str(doc_id))
        mock_repo.delete_document_record.assert_called_once_with(doc_id)

    @patch("app.document_manager.QueryCache")
    @patch("app.document_manager.EntityVectorStore")
    @patch("app.document_manager.resolve_neo4j_password")
    @patch("app.document_manager.Neo4jRepository")
    @patch("app.document_manager.IngestionRepository")
    def test_neo4j_failure_continues(
        self,
        mock_ingestion_repo_cls,
        mock_neo4j_repo_cls,
        mock_resolve_pw,
        mock_vector_store_cls,
        mock_query_cache_cls,
    ):
        doc_id = uuid.uuid4()
        chunk_ids = [f"{doc_id}_chunk_0"]

        mock_repo = MagicMock()
        mock_repo.get_chunk_ids_for_doc.return_value = chunk_ids
        mock_repo.delete_all_chunks_for_doc.return_value = 1
        mock_repo.delete_opensearch_docs_by_doc_id.return_value = 1
        mock_repo.delete_document_record.return_value = True
        mock_ingestion_repo_cls.return_value = mock_repo

        mock_neo4j = MagicMock()
        mock_neo4j.delete_relations_by_source_chunks.side_effect = Exception("Neo4j unavailable")
        mock_neo4j_repo_cls.return_value = mock_neo4j

        mock_resolve_pw.return_value = "test"

        mock_vector_store = MagicMock()
        mock_vector_store.delete_entities_by_source_chunks.return_value = 0
        mock_vector_store_cls.return_value = mock_vector_store

        mock_cache = MagicMock()
        mock_query_cache_cls.return_value = mock_cache

        settings = _make_settings()
        result = delete_document(doc_id, settings)

        mock_repo.delete_all_chunks_for_doc.assert_called_once_with(doc_id)
        mock_repo.delete_document_record.assert_called_once_with(doc_id)
        assert isinstance(result, DeleteDocumentResponse)

    @patch("app.document_manager.QueryCache")
    @patch("app.document_manager.EntityVectorStore")
    @patch("app.document_manager.resolve_neo4j_password")
    @patch("app.document_manager.Neo4jRepository")
    @patch("app.document_manager.IngestionRepository")
    def test_returns_correct_response_model(
        self,
        mock_ingestion_repo_cls,
        mock_neo4j_repo_cls,
        mock_resolve_pw,
        mock_vector_store_cls,
        mock_query_cache_cls,
    ):
        doc_id = uuid.uuid4()

        mock_repo = MagicMock()
        mock_repo.get_chunk_ids_for_doc.return_value = []
        mock_repo.delete_all_chunks_for_doc.return_value = 7
        mock_repo.delete_opensearch_docs_by_doc_id.return_value = 7
        mock_repo.delete_document_record.return_value = True
        mock_ingestion_repo_cls.return_value = mock_repo

        mock_cache = MagicMock()
        mock_query_cache_cls.return_value = mock_cache

        settings = _make_settings()
        result = delete_document(doc_id, settings)

        assert isinstance(result, DeleteDocumentResponse)
        assert result.doc_id == str(doc_id)
        assert result.status == "deleted"
        assert result.chunks_deleted == 7
        assert result.opensearch_deleted == 7
        assert result.cache_invalidated is True
