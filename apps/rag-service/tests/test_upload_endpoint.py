from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_FORM = {
    "title": "Test Document",
    "source_uri": "https://example.com/doc",
    "lang": "en",
    "category": "general",
    "published_year": "2024",
    "published_month": "3",
    "tags": "",
    "doc_version": "1.0",
}


def _make_file(content: bytes = b"hello world", filename: str = "test.txt") -> dict:
    return {"file": (filename, io.BytesIO(content), "text/plain")}


def test_upload_success() -> None:
    mock_s3 = MagicMock()
    with (
        patch("app.main.boto3") as mock_boto3,
        patch("app.main.ingest_document") as mock_ingest,
        patch("app.main.settings") as mock_settings,
    ):
        mock_boto3.client.return_value = mock_s3
        mock_settings.max_upload_size_mb = 50
        mock_settings.s3_bucket = "my-bucket"
        mock_settings.ingestion_queue_url = ""
        response = client.post(
            "/upload",
            data=VALID_FORM,
            files=_make_file(),
        )

    assert response.status_code == 202
    body = response.json()
    assert "run_id" in body
    assert body["s3_key"].startswith("uploads/")
    assert body["filename"] == "test.txt"
    assert body["status"] == "accepted"
    mock_ingest.assert_called_once()


def test_upload_file_too_large() -> None:
    large_content = b"x" * (2 * 1024 * 1024)
    with patch("app.main.settings") as mock_settings:
        mock_settings.max_upload_size_mb = 1
        mock_settings.s3_bucket = "my-bucket"
        mock_settings.ingestion_queue_url = ""
        response = client.post(
            "/upload",
            data=VALID_FORM,
            files={"file": ("big.txt", io.BytesIO(large_content), "text/plain")},
        )

    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


def test_upload_unsupported_format() -> None:
    with patch("app.main.settings") as mock_settings:
        mock_settings.max_upload_size_mb = 50
        mock_settings.s3_bucket = "my-bucket"
        mock_settings.ingestion_queue_url = ""
        response = client.post(
            "/upload",
            data=VALID_FORM,
            files={"file": ("malware.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
        )

    assert response.status_code == 415
    assert "Unsupported" in response.json()["detail"]


def test_upload_missing_required_fields() -> None:
    form_no_title = {k: v for k, v in VALID_FORM.items() if k != "title"}
    response = client.post(
        "/upload",
        data=form_no_title,
        files=_make_file(),
    )
    assert response.status_code == 422

    form_no_year = {k: v for k, v in VALID_FORM.items() if k != "published_year"}
    response2 = client.post(
        "/upload",
        data=form_no_year,
        files=_make_file(),
    )
    assert response2.status_code == 422


def test_upload_no_s3_bucket() -> None:
    with patch("app.main.settings") as mock_settings:
        mock_settings.max_upload_size_mb = 50
        mock_settings.s3_bucket = ""
        mock_settings.ingestion_queue_url = ""
        response = client.post(
            "/upload",
            data=VALID_FORM,
            files=_make_file(),
        )

    assert response.status_code == 500
    assert "S3 bucket not configured" in response.json()["detail"]


def test_upload_s3_failure() -> None:
    mock_s3 = MagicMock()
    mock_s3.put_object.side_effect = RuntimeError("S3 error")
    with (
        patch("app.main.boto3") as mock_boto3,
        patch("app.main.settings") as mock_settings,
    ):
        mock_boto3.client.return_value = mock_s3
        mock_settings.max_upload_size_mb = 50
        mock_settings.s3_bucket = "my-bucket"
        mock_settings.ingestion_queue_url = ""
        response = client.post(
            "/upload",
            data=VALID_FORM,
            files=_make_file(),
        )

    assert response.status_code == 500
    assert "Failed to upload file to S3" in response.json()["detail"]


def test_upload_tags_parsing() -> None:
    mock_s3 = MagicMock()
    captured_meta: list[dict] = []

    def capture_ingest(bucket, key, upload_meta, settings_obj):
        captured_meta.append({"tags": upload_meta.tags})

    form_with_tags = {**VALID_FORM, "tags": "a, b, c"}
    with (
        patch("app.main.boto3") as mock_boto3,
        patch("app.main.ingest_document", side_effect=capture_ingest),
        patch("app.main.settings") as mock_settings,
    ):
        mock_boto3.client.return_value = mock_s3
        mock_settings.max_upload_size_mb = 50
        mock_settings.s3_bucket = "my-bucket"
        mock_settings.ingestion_queue_url = ""
        response = client.post(
            "/upload",
            data=form_with_tags,
            files=_make_file(),
        )

    assert response.status_code == 202
    assert captured_meta[0]["tags"] == ["a", "b", "c"]


def test_upload_filename_sanitization() -> None:
    mock_s3 = MagicMock()
    with (
        patch("app.main.boto3") as mock_boto3,
        patch("app.main.ingest_document"),
        patch("app.main.settings") as mock_settings,
    ):
        mock_boto3.client.return_value = mock_s3
        mock_settings.max_upload_size_mb = 50
        mock_settings.s3_bucket = "my-bucket"
        mock_settings.ingestion_queue_url = ""
        response = client.post(
            "/upload",
            data=VALID_FORM,
            files={"file": ("path/to/doc.txt", io.BytesIO(b"data"), "text/plain")},
        )

    assert response.status_code == 202
    body = response.json()
    assert "/" not in body["s3_key"].split("/", 2)[-1]
    assert body["filename"] == "path/to/doc.txt"


def test_ingestion_status_success() -> None:
    run_id = "123e4567-e89b-12d3-a456-426614174000"
    mock_result = {
        "run_id": run_id,
        "status": "succeeded",
        "started_at": "2024-03-01T10:00:00",
        "finished_at": "2024-03-01T10:01:00",
        "notes": None,
    }
    with patch("app.main.ingestion_repo") as mock_repo:
        mock_repo.get_ingestion_run.return_value = mock_result
        response = client.get(f"/ingestion/{run_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == run_id
    assert body["status"] == "succeeded"
    assert body["started_at"] == "2024-03-01T10:00:00"
    assert body["finished_at"] == "2024-03-01T10:01:00"
    assert body["notes"] is None


def test_ingestion_status_not_found() -> None:
    run_id = "123e4567-e89b-12d3-a456-426614174999"
    with patch("app.main.ingestion_repo") as mock_repo:
        mock_repo.get_ingestion_run.return_value = None
        response = client.get(f"/ingestion/{run_id}")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_ingestion_status_invalid_uuid() -> None:
    response = client.get("/ingestion/not-a-uuid")
    assert response.status_code == 400
    assert "Invalid run_id format" in response.json()["detail"]
