from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion_models import IngestionResult, UploadMetadata
from ingestion_handler import handler


def _sqs_event(records: list[dict]) -> dict:
    return {"Records": records}


def _sqs_record(bucket: str, key: str, message_id: str = "msg-1") -> dict:
    return {
        "messageId": message_id,
        "body": json.dumps({
            "Records": [{
                "s3": {
                    "bucket": {"name": bucket},
                    "object": {"key": key},
                }
            }]
        }),
    }


def _ingestion_result(
    status: str = "succeeded",
    chunks_created: int = 5,
    error: str | None = None,
) -> IngestionResult:
    return IngestionResult(
        run_id=uuid.uuid4(),
        doc_id=uuid.uuid4(),
        status=status,
        chunks_created=chunks_created,
        error=error,
    )


def _default_s3_meta() -> dict[str, str]:
    return {
        "title": "Test Doc",
        "source_uri": "https://example.com/doc.pdf",
        "lang": "en",
        "category": "general",
        "published_year": "2024",
        "published_month": "6",
        "author": "Jane Doe",
        "tags": "tag1, tag2",
        "doc_version": "1.0",
    }


@pytest.fixture()
def mock_s3_client() -> MagicMock:
    client = MagicMock()
    client.head_object.return_value = {"Metadata": _default_s3_meta()}
    return client


@pytest.fixture()
def mock_ingest_success() -> IngestionResult:
    return _ingestion_result(status="succeeded", chunks_created=5)


@pytest.fixture()
def mock_ingest_failure() -> IngestionResult:
    return _ingestion_result(status="failed", chunks_created=0, error="DB error")


def test_handler_success(mock_s3_client: MagicMock, mock_ingest_success: IngestionResult) -> None:
    event = _sqs_event([_sqs_record("my-bucket", "docs/file.pdf")])

    with (
        patch("ingestion_handler.boto3") as mock_boto3,
        patch("ingestion_handler.ingest_document", return_value=mock_ingest_success),
    ):
        mock_boto3.client.return_value = mock_s3_client
        result = handler(event, None)

    assert result == {"batchItemFailures": []}
    mock_s3_client.head_object.assert_called_once_with(Bucket="my-bucket", Key="docs/file.pdf")


def test_handler_ingestion_failure(
    mock_s3_client: MagicMock, mock_ingest_failure: IngestionResult
) -> None:
    event = _sqs_event([_sqs_record("my-bucket", "docs/file.pdf", message_id="msg-fail")])

    with (
        patch("ingestion_handler.boto3") as mock_boto3,
        patch("ingestion_handler.ingest_document", return_value=mock_ingest_failure),
    ):
        mock_boto3.client.return_value = mock_s3_client
        result = handler(event, None)

    assert result == {"batchItemFailures": [{"itemIdentifier": "msg-fail"}]}


def test_handler_invalid_body() -> None:
    event = _sqs_event([{"messageId": "msg-bad", "body": "not-valid-json"}])

    with patch("ingestion_handler.boto3"):
        result = handler(event, None)

    assert result == {"batchItemFailures": [{"itemIdentifier": "msg-bad"}]}


def test_handler_missing_s3_records() -> None:
    event = _sqs_event([{"messageId": "msg-empty", "body": json.dumps({"Records": []})}])

    with patch("ingestion_handler.boto3"), patch("ingestion_handler.ingest_document") as mock_ingest:
        result = handler(event, None)

    assert result == {"batchItemFailures": []}
    mock_ingest.assert_not_called()


def test_handler_multiple_records(mock_s3_client: MagicMock) -> None:
    success_result = _ingestion_result(status="succeeded", chunks_created=3)
    failure_result = _ingestion_result(status="failed", chunks_created=0, error="timeout")

    event = _sqs_event([
        _sqs_record("my-bucket", "docs/a.pdf", message_id="msg-ok"),
        _sqs_record("my-bucket", "docs/b.pdf", message_id="msg-bad"),
    ])

    with (
        patch("ingestion_handler.boto3") as mock_boto3,
        patch(
            "ingestion_handler.ingest_document",
            side_effect=[success_result, failure_result],
        ),
    ):
        mock_boto3.client.return_value = mock_s3_client
        result = handler(event, None)

    assert result == {"batchItemFailures": [{"itemIdentifier": "msg-bad"}]}


def test_handler_s3_head_failure() -> None:
    event = _sqs_event([_sqs_record("my-bucket", "docs/file.pdf", message_id="msg-head-fail")])

    with patch("ingestion_handler.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_client.head_object.side_effect = Exception("S3 access denied")
        mock_boto3.client.return_value = mock_client
        result = handler(event, None)

    assert result == {"batchItemFailures": [{"itemIdentifier": "msg-head-fail"}]}


def test_handler_metadata_extraction(mock_s3_client: MagicMock) -> None:
    captured: list[Any] = []

    def _capture_ingest(
        bucket: str, key: str, upload_meta: UploadMetadata, settings: Any
    ) -> IngestionResult:
        captured.append(upload_meta)
        return _ingestion_result()

    event = _sqs_event([_sqs_record("bucket-x", "path/doc.pdf")])

    with (
        patch("ingestion_handler.boto3") as mock_boto3,
        patch("ingestion_handler.ingest_document", side_effect=_capture_ingest),
    ):
        mock_boto3.client.return_value = mock_s3_client
        handler(event, None)

    assert len(captured) == 1
    meta = captured[0]
    assert isinstance(meta, UploadMetadata)
    assert meta.title == "Test Doc"
    assert meta.source_uri == "https://example.com/doc.pdf"
    assert meta.lang == "en"
    assert meta.category == "general"
    assert meta.published_year == 2024
    assert meta.published_month == 6
    assert meta.author == "Jane Doe"
    assert meta.tags == ["tag1", "tag2"]
    assert meta.doc_version == "1.0"


def test_handler_empty_event() -> None:
    with patch("ingestion_handler.boto3"), patch("ingestion_handler.ingest_document") as mock_ingest:
        result = handler({}, None)

    assert result == {"batchItemFailures": []}
    mock_ingest.assert_not_called()
