from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Generator
from typing import Any

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# --- Prometheus metrics ---

PIPELINE_NODE_LATENCY = Histogram(
    "rag_pipeline_node_latency_seconds",
    "Per-node latency",
    ["node"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

PIPELINE_NODE_ERRORS = Counter(
    "rag_pipeline_node_errors_total",
    "Per-node errors",
    ["node"],
)

LLM_TOKEN_USAGE = Counter(
    "rag_llm_token_usage_total",
    "LLM token usage",
    ["model", "direction"],
)

CACHE_OPERATIONS = Counter(
    "rag_cache_operations_total",
    "Cache operations",
    ["result"],
)

RETRIEVAL_HIT_COUNT = Histogram(
    "rag_retrieval_hit_count",
    "Docs retrieved per query",
    buckets=[1, 3, 5, 10, 20, 50],
)

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "End-to-end request latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)


# --- TracingService ---


class TracingService:
    def __init__(
        self,
        enabled: bool,
        service_name: str,
        endpoint: str,
        provider: str,
    ) -> None:
        self._enabled = enabled and provider != "none"
        self._tracer: Any = None

        if not self._enabled:
            return

        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            resource = Resource.create({"service.name": service_name})
            tracer_provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces", timeout=10)
            tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(tracer_provider)
            self._tracer = trace.get_tracer("hybrid-rag-service")
            self._tracer_provider = tracer_provider
        except Exception:
            logger.exception("Failed to initialize OpenTelemetry tracing — disabling")
            self._enabled = False
            self._tracer = None

    @contextlib.contextmanager
    def span(self, name: str, **attributes: Any) -> Generator[Any, None, None]:
        if not self._enabled or self._tracer is None:
            yield None
            return

        try:
            from opentelemetry.trace import StatusCode
        except Exception:
            yield None
            return

        start = time.perf_counter()
        try:
            with self._tracer.start_as_current_span(name) as otel_span:
                try:
                    for key, value in attributes.items():
                        otel_span.set_attribute(key, value)
                    yield otel_span
                except Exception as exc:
                    try:
                        otel_span.record_exception(exc)
                        otel_span.set_status(StatusCode.ERROR, str(exc))
                    except Exception:
                        pass
                    raise
                finally:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    try:
                        otel_span.set_attribute("latency_ms", elapsed_ms)
                    except Exception:
                        pass
        except Exception:
            raise

    def shutdown(self) -> None:
        if not self._enabled:
            return
        try:
            provider = getattr(self, "_tracer_provider", None)
            if provider is not None:
                provider.shutdown()
        except Exception:
            logger.exception("Error during tracing shutdown")


# --- Singleton ---

_service: TracingService | None = None


def init_tracing(settings: Any) -> TracingService:
    global _service
    _service = TracingService(
        enabled=settings.enable_tracing,
        service_name=settings.tracing_service_name,
        endpoint=settings.tracing_endpoint,
        provider=settings.tracing_provider,
    )
    return _service


def get_tracing() -> TracingService:
    global _service
    if _service is None:
        _service = TracingService(
            enabled=False,
            service_name="hybrid-rag-service",
            endpoint="",
            provider="none",
        )
    return _service
