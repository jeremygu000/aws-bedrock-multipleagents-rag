"""SQS-triggered Lambda handler for document ingestion.

Processes S3 event notifications delivered via SQS.
Each SQS record contains an S3 event with bucket + key of an uploaded document.
Upload metadata is read from the S3 object's user metadata (x-amz-meta-*).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3

from app.config import get_settings
from app.ingestion import ingest_document
from app.ingestion_models import UploadMetadata

logger = logging.getLogger(__name__)

settings = get_settings()


def handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    """Process SQS records containing S3 event notifications.

    For each SQS record:
    1. Parse the SQS body as JSON (it's an S3 event notification)
    2. Extract s3 bucket and key from each S3 record
    3. Call S3 HeadObject to read user metadata (x-amz-meta-*)
    4. Build UploadMetadata from the S3 user metadata
    5. Call ingest_document(bucket, key, metadata, settings)
    6. Track success/failure per record

    Returns:
        Dict with batchItemFailures for partial batch failure reporting.
        (SQS Lambda partial batch response format)
    """
    batch_item_failures: list[dict[str, str]] = []

    for record in event.get("Records", []):
        message_id = record.get("messageId", "unknown")
        try:
            body = json.loads(record["body"])
            # S3 event notification has Records[] with s3.bucket.name and s3.object.key
            for s3_record in body.get("Records", []):
                bucket = s3_record["s3"]["bucket"]["name"]
                key = s3_record["s3"]["object"]["key"]

                # Read upload metadata from S3 object metadata
                s3_client = boto3.client("s3")
                head = s3_client.head_object(Bucket=bucket, Key=key)
                s3_meta = head.get("Metadata", {})

                upload_meta = UploadMetadata(
                    title=s3_meta.get("title", ""),
                    source_uri=s3_meta.get("source_uri", ""),
                    lang=s3_meta.get("lang", "en"),
                    category=s3_meta.get("category", "general"),
                    published_year=int(s3_meta.get("published_year", "2024")),
                    published_month=int(s3_meta.get("published_month", "1")),
                    author=s3_meta.get("author") or None,
                    tags=[t.strip() for t in s3_meta.get("tags", "").split(",") if t.strip()],
                    doc_version=s3_meta.get("doc_version", "1.0"),
                )

                result = ingest_document(bucket, key, upload_meta, settings)

                if result.status == "failed":
                    logger.error(
                        "Ingestion failed for s3://%s/%s: %s",
                        bucket,
                        key,
                        result.error,
                    )
                    batch_item_failures.append({"itemIdentifier": message_id})
                    break  # One failure per SQS message
                else:
                    logger.info(
                        "Ingestion succeeded for s3://%s/%s: %d chunks",
                        bucket,
                        key,
                        result.chunks_created,
                    )

        except Exception as exc:
            logger.exception("Failed to process SQS record %s: %s", message_id, exc)
            batch_item_failures.append({"itemIdentifier": message_id})

    return {"batchItemFailures": batch_item_failures}
