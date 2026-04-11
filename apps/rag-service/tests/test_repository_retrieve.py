from __future__ import annotations

from typing import Any
from unittest.mock import patch

from app.config import Settings
from app.models import RetrieveRequest
from app.repository import PostgresRepository

_FAKE_CHUNK: dict[str, Any] = {
    "chunk_id": "aaaaaaaa-0000-0000-0000-000000000001",
    "doc_id": "bbbbbbbb-0000-0000-0000-000000000002",
    "chunk_text": "hello world",
    "score": 0.5,
    "category": "policy",
    "lang": "en",
    "source_type": "crawler",
    "metadata": {},
    "citation": {
        "url": "https://example.com/doc",
        "title": "Example Doc",
        "year": 2024,
        "month": 1,
        "page_start": 1,
        "page_end": 2,
        "section_id": None,
        "anchor_id": None,
    },
}


def _make_settings(**kwargs: Any) -> Settings:
    defaults: dict[str, Any] = {
        "RAG_DB_HOST": "localhost",
        "RAG_DB_PASSWORD": "secret",
        "RAG_SPARSE_BACKEND": "postgres",
        "RAG_OPENSEARCH_ENDPOINT": "",
    }
    defaults.update(kwargs)
    return Settings(**defaults)


def _make_request(**kwargs: Any) -> RetrieveRequest:
    defaults: dict[str, Any] = {
        "query": "APRA licensing policy",
        "require_strict_citation": False,
    }
    defaults.update(kwargs)
    return RetrieveRequest(**defaults)


class TestRetrieveDispatchPostgres:
    def _repo(self) -> PostgresRepository:
        return PostgresRepository(_make_settings(RAG_SPARSE_BACKEND="postgres"))

    def test_no_embedding_calls_sparse(self) -> None:
        repo = self._repo()
        request = _make_request(query_embedding=None)

        with (
            patch.object(repo, "_retrieve_sparse", return_value=[_FAKE_CHUNK]) as mock_sparse,
            patch.object(repo, "_retrieve_hybrid") as mock_hybrid,
            patch.object(repo, "_retrieve_sparse_opensearch") as mock_os_sparse,
            patch.object(repo, "_retrieve_hybrid_with_opensearch") as mock_os_hybrid,
        ):
            result = repo.retrieve(request)

        mock_sparse.assert_called_once()
        mock_hybrid.assert_not_called()
        mock_os_sparse.assert_not_called()
        mock_os_hybrid.assert_not_called()
        assert result == [_FAKE_CHUNK]

    def test_with_embedding_calls_hybrid(self) -> None:
        repo = self._repo()
        request = _make_request(query_embedding=[0.1] * 1024)

        with (
            patch.object(repo, "_retrieve_hybrid", return_value=[_FAKE_CHUNK]) as mock_hybrid,
            patch.object(repo, "_retrieve_sparse") as mock_sparse,
            patch.object(repo, "_retrieve_sparse_opensearch") as mock_os_sparse,
            patch.object(repo, "_retrieve_hybrid_with_opensearch") as mock_os_hybrid,
        ):
            result = repo.retrieve(request)

        mock_hybrid.assert_called_once()
        mock_sparse.assert_not_called()
        mock_os_sparse.assert_not_called()
        mock_os_hybrid.assert_not_called()
        assert result == [_FAKE_CHUNK]


class TestRetrieveDispatchOpenSearch:
    def _repo(self) -> PostgresRepository:
        return PostgresRepository(
            _make_settings(
                RAG_SPARSE_BACKEND="opensearch",
                RAG_OPENSEARCH_ENDPOINT="https://search.example.com",
            )
        )

    def test_with_embedding_calls_hybrid_opensearch(self) -> None:
        repo = self._repo()
        request = _make_request(query_embedding=[0.2] * 1024)

        with (
            patch.object(
                repo, "_retrieve_hybrid_with_opensearch", return_value=[_FAKE_CHUNK]
            ) as mock_os_hybrid,
            patch.object(repo, "_retrieve_sparse_opensearch") as mock_os_sparse,
            patch.object(repo, "_retrieve_sparse") as mock_sparse,
            patch.object(repo, "_retrieve_hybrid") as mock_hybrid,
        ):
            result = repo.retrieve(request)

        mock_os_hybrid.assert_called_once()
        mock_os_sparse.assert_not_called()
        mock_sparse.assert_not_called()
        mock_hybrid.assert_not_called()
        assert result == [_FAKE_CHUNK]

    def test_no_embedding_calls_sparse_opensearch(self) -> None:
        repo = self._repo()
        request = _make_request(query_embedding=None)

        with (
            patch.object(
                repo, "_retrieve_sparse_opensearch", return_value=[_FAKE_CHUNK]
            ) as mock_os_sparse,
            patch.object(repo, "_retrieve_hybrid_with_opensearch") as mock_os_hybrid,
            patch.object(repo, "_retrieve_sparse") as mock_sparse,
            patch.object(repo, "_retrieve_hybrid") as mock_hybrid,
        ):
            result = repo.retrieve(request)

        mock_os_sparse.assert_called_once()
        mock_os_hybrid.assert_not_called()
        mock_sparse.assert_not_called()
        mock_hybrid.assert_not_called()
        assert result == [_FAKE_CHUNK]

    def test_opensearch_exception_falls_back_to_postgres_sparse(self) -> None:
        repo = self._repo()
        request = _make_request(query_embedding=None)

        with (
            patch.object(
                repo, "_retrieve_sparse_opensearch", side_effect=RuntimeError("OpenSearch down")
            ) as mock_os_sparse,
            patch.object(repo, "_retrieve_sparse", return_value=[_FAKE_CHUNK]) as mock_sparse,
        ):
            result = repo.retrieve(request)

        mock_os_sparse.assert_called_once()
        mock_sparse.assert_called_once()
        assert result == [_FAKE_CHUNK]

    def test_opensearch_hybrid_exception_falls_back_to_postgres_hybrid(self) -> None:
        repo = self._repo()
        request = _make_request(query_embedding=[0.3] * 1024)

        with (
            patch.object(
                repo, "_retrieve_hybrid_with_opensearch", side_effect=ConnectionError("timeout")
            ) as mock_os_hybrid,
            patch.object(repo, "_retrieve_hybrid", return_value=[_FAKE_CHUNK]) as mock_hybrid,
        ):
            result = repo.retrieve(request)

        mock_os_hybrid.assert_called_once()
        mock_hybrid.assert_called_once()
        assert result == [_FAKE_CHUNK]

    def test_empty_endpoint_skips_opensearch_and_uses_postgres_sparse(self) -> None:
        repo = PostgresRepository(
            _make_settings(
                RAG_SPARSE_BACKEND="opensearch",
                RAG_OPENSEARCH_ENDPOINT="",
            )
        )
        request = _make_request(query_embedding=None)

        with (
            patch.object(repo, "_retrieve_sparse", return_value=[_FAKE_CHUNK]) as mock_sparse,
            patch.object(repo, "_retrieve_sparse_opensearch") as mock_os_sparse,
        ):
            result = repo.retrieve(request)

        mock_sparse.assert_called_once()
        mock_os_sparse.assert_not_called()
        assert result == [_FAKE_CHUNK]

    def test_opensearch_sparse_empty_result_falls_through_to_postgres_sparse(self) -> None:
        repo = self._repo()
        request = _make_request(query_embedding=None)

        with (
            patch.object(repo, "_retrieve_sparse_opensearch", return_value=[]) as mock_os_sparse,
            patch.object(repo, "_retrieve_sparse", return_value=[_FAKE_CHUNK]) as mock_sparse,
        ):
            result = repo.retrieve(request)

        mock_os_sparse.assert_called_once()
        mock_sparse.assert_called_once()
        assert result == [_FAKE_CHUNK]


class TestGetChunksByIds:
    def _repo(self) -> PostgresRepository:
        return PostgresRepository(_make_settings())

    def test_empty_ids_returns_empty_without_db_call(self) -> None:
        repo = self._repo()

        with patch.object(repo, "_run_statement") as mock_run:
            result = repo.get_chunks_by_ids([])

        mock_run.assert_not_called()
        assert result == []

    def test_valid_ids_calls_run_statement_and_returns_results(self) -> None:
        repo = self._repo()
        ids = [
            "aaaaaaaa-0000-0000-0000-000000000001",
            "cccccccc-0000-0000-0000-000000000003",
        ]

        with patch.object(repo, "_run_statement", return_value=[_FAKE_CHUNK]) as mock_run:
            result = repo.get_chunks_by_ids(ids)

        mock_run.assert_called_once()
        call_params = mock_run.call_args[0][1]
        assert set(call_params["chunk_ids"]) == set(ids)
        assert result == [_FAKE_CHUNK]

    def test_duplicate_ids_are_deduplicated_before_db_call(self) -> None:
        repo = self._repo()
        dup_id = "aaaaaaaa-0000-0000-0000-000000000001"

        with patch.object(repo, "_run_statement", return_value=[]) as mock_run:
            repo.get_chunks_by_ids([dup_id, dup_id, dup_id])

        call_params = mock_run.call_args[0][1]
        assert call_params["chunk_ids"] == [dup_id]


class TestRetrieveFiltersForwarding:
    def test_filters_forwarded_to_sparse_with_clauses_and_params(self) -> None:
        repo = PostgresRepository(_make_settings(RAG_SPARSE_BACKEND="postgres"))
        request = _make_request(
            query_embedding=None,
            filters={"category": "policy", "lang": "en"},
        )

        with patch.object(repo, "_retrieve_sparse", return_value=[]) as mock_sparse:
            repo.retrieve(request)

        mock_sparse.assert_called_once()
        args = mock_sparse.call_args[0]
        assert args[0] is request
        assert len(args[1]) >= 2
        params: dict = args[2]
        assert "filter_category" in params
        assert "filter_lang" in params
