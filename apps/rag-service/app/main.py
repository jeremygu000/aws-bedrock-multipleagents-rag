"""FastAPI entrypoint for local/dev retrieval testing."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import boto3
from fastapi import FastAPI, Form, HTTPException, UploadFile
from psycopg import Error as PsycopgError

from .config import get_settings
from .document_parser import _EXTENSION_TO_MIME, SUPPORTED_MIME_TYPES
from .ingestion import ingest_document
from .ingestion_models import IngestionStatusResponse, UploadMetadata, UploadResponse
from .ingestion_repository import IngestionRepository
from .models import RetrieveRequest, RetrieveResponse
from .repository import PostgresRepository

settings = get_settings()
repository = PostgresRepository(settings)
ingestion_repo = IngestionRepository(settings)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """Run sparse/hybrid retrieval and return strict citation hits.

    Error mapping:
    - ValueError -> 400 (invalid input/config)
    - PsycopgError -> 500 (database execution failure)
    """

    try:
        hits = repository.retrieve(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except PsycopgError as error:
        raise HTTPException(status_code=500, detail="database query failed") from error

    mode = "hybrid" if request.query_embedding else "sparse"
    return RetrieveResponse(
        query=request.query,
        mode=mode,
        hit_count=len(hits),
        hits=hits,
    )


@app.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_document(
    file: UploadFile,
    title: str = Form(...),
    source_uri: str = Form(""),
    lang: str = Form("en"),
    category: str = Form("general"),
    published_year: int = Form(...),
    published_month: int = Form(...),
    author: str | None = Form(None),
    tags: str = Form(""),
    doc_version: str = Form("1.0"),
) -> UploadResponse:
    file_bytes = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds {settings.max_upload_size_mb}MB limit",
        )

    filename = file.filename or "upload"
    content_type = file.content_type or "application/octet-stream"
    ext = Path(filename).suffix.lower()
    detected_mime = _EXTENSION_TO_MIME.get(ext, content_type)
    if detected_mime not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {detected_mime}. Supported: {list(SUPPORTED_MIME_TYPES.keys())}",
        )

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    upload_meta = UploadMetadata(
        title=title,
        source_uri=source_uri,
        lang=lang,
        category=category,
        published_year=published_year,
        published_month=published_month,
        author=author,
        tags=tag_list,
        doc_version=doc_version,
    )

    run_id = str(uuid.uuid4())
    safe_filename = filename.replace("/", "_").replace("\\", "_")
    s3_key = f"uploads/{run_id}/{safe_filename}"

    if not settings.s3_bucket:
        raise HTTPException(status_code=500, detail="S3 bucket not configured (RAG_S3_BUCKET)")

    try:
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=settings.s3_bucket,
            Key=s3_key,
            Body=file_bytes,
            ContentType=detected_mime,
            Metadata={
                "title": title,
                "source_uri": source_uri,
                "lang": lang,
                "category": category,
                "published_year": str(published_year),
                "published_month": str(published_month),
                "author": author or "",
                "tags": tags,
                "doc_version": doc_version,
            },
        )
    except Exception as exc:
        logger.exception("Failed to upload to S3: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to upload file to S3") from exc

    if not settings.ingestion_queue_url:
        try:
            ingest_document(settings.s3_bucket, s3_key, upload_meta, settings)
        except Exception as exc:
            logger.exception("Sync ingestion failed: %s", exc)

    return UploadResponse(
        run_id=run_id,
        s3_key=s3_key,
        filename=filename,
    )


@app.get("/ingestion/{run_id}", response_model=IngestionStatusResponse)
def get_ingestion_status(run_id: str) -> IngestionStatusResponse:
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid run_id format") from exc

    result = ingestion_repo.get_ingestion_run(run_uuid)
    if result is None:
        raise HTTPException(status_code=404, detail="Ingestion run not found")

    return IngestionStatusResponse(**result)
