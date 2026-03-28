from __future__ import annotations

import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.ingestion_models import DeleteDocumentResponse
from app.main import app

client = TestClient(app)


class TestDeleteDocumentEndpoint:
    @patch("app.main.delete_document")
    def test_delete_document_success(self, mock_delete):
        doc_id = str(uuid.uuid4())
        mock_delete.return_value = DeleteDocumentResponse(
            doc_id=doc_id,
            chunks_deleted=5,
            entities_deleted=3,
            relations_deleted=2,
            opensearch_deleted=5,
            cache_invalidated=True,
        )
        response = client.delete(f"/documents/{doc_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == doc_id
        assert data["chunks_deleted"] == 5
        assert data["status"] == "deleted"

    def test_delete_document_invalid_uuid(self):
        response = client.delete("/documents/not-a-uuid")
        assert response.status_code == 400

    @patch("app.main.delete_document")
    def test_delete_document_internal_error(self, mock_delete):
        doc_id = str(uuid.uuid4())
        mock_delete.side_effect = Exception("unexpected failure")
        response = client.delete(f"/documents/{doc_id}")
        assert response.status_code == 500
