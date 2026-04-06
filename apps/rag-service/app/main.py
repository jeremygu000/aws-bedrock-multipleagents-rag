"""FastAPI entrypoint for local/dev retrieval testing."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import boto3
from fastapi import FastAPI, Form, HTTPException, Query, Request, UploadFile
from psycopg import Error as PsycopgError
from sse_starlette.sse import EventSourceResponse

from .answer_generator import (
    BedrockConverseAnswerGenerator,
    QwenAnswerGenerator,
    RoutedAnswerGenerator,
)
from .config import get_settings
from .crag import CragQueryRewriter, CragWebSearcher, RetrievalGrader
from .document_manager import delete_document
from .document_parser import _EXTENSION_TO_MIME, SUPPORTED_MIME_TYPES
from .embedding_factory import get_embedding_client
from .ingestion import ingest_document
from .ingestion_models import (
    DeleteDocumentResponse,
    IngestionStatusResponse,
    UploadMetadata,
    UploadResponse,
)
from .ingestion_repository import IngestionRepository
from .models import (
    GraphContext,
    RetrieveRequest,
    RetrieveResponse,
    StreamDone,
    StreamError,
    StreamMetadata,
    StreamToken,
)
from .query_cache import QueryCache
from .query_processing import QueryProcessor
from .qwen_client import QwenClient
from .repository import PostgresRepository
from .reranker import LLMReranker
from .tracing import REQUEST_LATENCY, get_tracing, init_tracing
from .workflow import RagWorkflow

settings = get_settings()
init_tracing(settings)
repository = PostgresRepository(settings)
ingestion_repo = IngestionRepository(settings)
logger = logging.getLogger(__name__)

qwen_client = QwenClient(settings)
embedding_client = get_embedding_client(settings)
query_processor = QueryProcessor(
    settings=settings, qwen_client=qwen_client, embedding_client=embedding_client
)
answer_generator = RoutedAnswerGenerator(
    bedrock_generator=BedrockConverseAnswerGenerator(settings),
    qwen_generator=QwenAnswerGenerator(qwen_client, settings),
)
reranker = LLMReranker(settings=settings, qwen_client=qwen_client)


def _build_workflow() -> RagWorkflow:
    graph_retriever = None
    if settings.enable_graph_retrieval:
        from .entity_vector_store import EntityVectorStore
        from .graph_retriever import GraphRetriever

        vector_store = EntityVectorStore(settings)
        neo4j_repo = None
        if settings.enable_neo4j:
            from .graph_repository import Neo4jRepository
            from .secrets import resolve_neo4j_password

            neo4j_password = resolve_neo4j_password(settings)
            neo4j_repo = Neo4jRepository(
                uri=settings.neo4j_uri,
                username=settings.neo4j_username,
                password=neo4j_password,
                database=settings.neo4j_database,
            )
        graph_retriever = GraphRetriever(
            qwen_client=qwen_client,
            vector_store=vector_store,
            neo4j_repo=neo4j_repo,
            settings=settings,
            embedding_client=embedding_client,
        )

    query_cache = (
        QueryCache(settings, repository._get_engine()) if settings.enable_query_cache else None
    )

    # Query Decomposition components — only instantiate when feature flag is on.
    decomposition_retriever = None
    if settings.enable_query_decomposition:
        from .decomposition_retriever import DecompositionRetriever
        from .query_decomposer import QueryDecomposer
        
        query_decomposer = QueryDecomposer(settings)
        decomposition_retriever = DecompositionRetriever(repository, query_decomposer, settings)

    # CRAG components — only instantiate when feature flag is on.
    retrieval_grader = None
    crag_query_rewriter = None
    crag_web_searcher = None
    if settings.enable_crag:
        retrieval_grader = RetrievalGrader(settings)
        crag_query_rewriter = CragQueryRewriter(settings)
        crag_web_searcher = CragWebSearcher(settings)

    return RagWorkflow(
        settings=settings,
        repository=repository,
        query_processor=query_processor,
        answer_generator=answer_generator,
        reranker=reranker,
        graph_retriever=graph_retriever,
        query_cache=query_cache,
        retrieval_grader=retrieval_grader,
        crag_query_rewriter=crag_query_rewriter,
        crag_web_searcher=crag_web_searcher,
        decomposition_retriever=decomposition_retriever,
    )


workflow = _build_workflow()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    get_tracing().shutdown()


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    from prometheus_client import REGISTRY, generate_latest
    from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
    from starlette.responses import Response

    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    start = time.perf_counter()
    try:
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
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start)


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


@app.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document_endpoint(doc_id: str) -> DeleteDocumentResponse:
    try:
        doc_uuid = uuid.UUID(doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid doc_id format") from exc

    try:
        return delete_document(doc_uuid, settings)
    except Exception as exc:
        logger.exception("Document deletion failed for %s: %s", doc_id, exc)
        raise HTTPException(status_code=500, detail="Document deletion failed") from exc


@app.get("/retrieve/stream")
async def retrieve_stream(
    request: Request,
    query: str = Query(..., min_length=1),
    top_k: int = Query(default=8, ge=1, le=50),
) -> EventSourceResponse:
    if not settings.enable_streaming:
        raise HTTPException(status_code=404, detail="Streaming is disabled")

    async def _event_generator() -> AsyncIterator[dict[str, str]]:
        start_time = time.monotonic()
        req_start = time.perf_counter()
        total_tokens = 0
        full_answer_parts: list[str] = []

        try:
            state = workflow.run_until_generate(query, top_k, filters={})
        except Exception as exc:
            logger.exception("Pipeline failed during streaming: %s", exc)
            yield {
                "event": "error",
                "data": StreamError(error="pipeline_error", detail=str(exc)).model_dump_json(),
            }
            return

        if state.get("cache_hit"):
            metadata = StreamMetadata(
                intent=state.get("intent", ""),
                complexity=state.get("complexity", ""),
                retrieval_mode="",
                model=state.get("answer_model", "cache"),
                citations=state.get("citations", []),
            )
            yield {"event": "metadata", "data": metadata.model_dump_json()}
            cached_answer = state.get("answer", "")
            yield {
                "event": "token",
                "data": StreamToken(text=cached_answer).model_dump_json(),
            }
            total_tokens = len(cached_answer.split())
            elapsed_ms = (time.monotonic() - start_time) * 1000
            yield {
                "event": "done",
                "data": StreamDone(
                    total_tokens=total_tokens,
                    latency_ms=round(elapsed_ms, 1),
                    cache_stored=False,
                ).model_dump_json(),
            }
            return

        hl = state.get("hl_keywords", [])
        ll = state.get("ll_keywords", [])
        keywords = list(dict.fromkeys(hl + ll))
        graph_context: GraphContext | None = state.get("graph_context")
        hits = state.get("reranked_hits") or state.get("hits", [])
        preferred_model = state.get("preferred_model", "nova-lite")

        try:
            token_stream, used_model = workflow.answer_generator.generate_stream(
                query=query,
                hits=hits,
                preferred_model=preferred_model,
                intent=state.get("intent", "factual"),
                complexity=state.get("complexity", "medium"),
                keywords=keywords or None,
                graph_context=graph_context,
            )
        except Exception as exc:
            logger.exception("Stream generation setup failed: %s", exc)
            yield {
                "event": "error",
                "data": StreamError(error="generation_error", detail=str(exc)).model_dump_json(),
            }
            return

        metadata = StreamMetadata(
            intent=state.get("intent", "factual"),
            complexity=state.get("complexity", "medium"),
            retrieval_mode=state.get("retrieval_mode", "mix"),
            model=used_model,
            citations=state.get("citations", []),
        )
        yield {"event": "metadata", "data": metadata.model_dump_json()}

        try:
            for chunk in token_stream:
                if await request.is_disconnected():
                    break
                total_tokens += 1
                full_answer_parts.append(chunk)
                yield {
                    "event": "token",
                    "data": StreamToken(text=chunk).model_dump_json(),
                }
        except Exception as exc:
            logger.exception("Token streaming failed: %s", exc)
            yield {
                "event": "error",
                "data": StreamError(error="stream_error", detail=str(exc)).model_dump_json(),
            }
            return

        cache_stored = False
        full_answer = "".join(full_answer_parts)
        if full_answer and workflow.query_cache and settings.enable_query_cache:
            embedding = state.get("query_embedding")
            if embedding:
                try:
                    source_doc_ids = list({h.get("doc_id", "") for h in hits if h.get("doc_id")})
                    workflow.query_cache.store(
                        query_original=query,
                        query_rewritten=state.get("rewritten_query"),
                        query_embedding=embedding,
                        answer=full_answer,
                        citations=state.get("citations", []),
                        model_used=used_model,
                        source_doc_ids=source_doc_ids,
                    )
                    cache_stored = True
                except Exception:
                    logger.exception("Failed to store streamed result in cache")

        elapsed_ms = (time.monotonic() - start_time) * 1000
        REQUEST_LATENCY.observe(time.perf_counter() - req_start)
        yield {
            "event": "done",
            "data": StreamDone(
                total_tokens=total_tokens,
                latency_ms=round(elapsed_ms, 1),
                cache_stored=cache_stored,
            ).model_dump_json(),
        }

    return EventSourceResponse(
        _event_generator(),
        headers={"X-Accel-Buffering": "no"},
    )
