"""FastAPI entrypoint for local/dev retrieval testing."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from psycopg import Error as PsycopgError

from .config import get_settings
from .models import RetrieveRequest, RetrieveResponse
from .repository import PostgresRepository

settings = get_settings()
repository = PostgresRepository(settings)

app = FastAPI(title=settings.app_name)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness endpoint for health checks."""

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
