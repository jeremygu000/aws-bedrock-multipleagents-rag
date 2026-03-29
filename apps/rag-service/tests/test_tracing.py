from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, Counter, Histogram

from app.tracing import TracingService

# --- TracingService initialization ---


def test_tracing_disabled_by_default():
    svc = TracingService(enabled=False, service_name="test", endpoint="", provider="none")
    with svc.span("test-span") as span:
        assert span is None


def test_tracing_provider_none_disables():
    svc = TracingService(
        enabled=True, service_name="test", endpoint="http://localhost:6006", provider="none"
    )
    assert svc._enabled is False
    with svc.span("noop") as span:
        assert span is None


def test_tracing_enabled_creates_provider():
    fake_exporter_instance = MagicMock()
    fake_provider_instance = MagicMock()
    fake_tracer_instance = MagicMock()

    with (
        patch(
            "opentelemetry.sdk.trace.TracerProvider", return_value=fake_provider_instance
        ) as mock_provider_cls,
        patch(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
            return_value=fake_exporter_instance,
        ),
        patch("opentelemetry.trace.set_tracer_provider") as mock_set,
        patch("opentelemetry.trace.get_tracer", return_value=fake_tracer_instance),
    ):
        svc = TracingService(
            enabled=True,
            service_name="hybrid-rag-service",
            endpoint="http://localhost:6006",
            provider="phoenix",
        )
        if svc._enabled:
            mock_provider_cls.assert_called_once()
            mock_set.assert_called_once_with(fake_provider_instance)


def test_tracing_init_failure_does_not_crash():
    with patch("opentelemetry.sdk.trace.TracerProvider", side_effect=RuntimeError("boom")):
        svc = TracingService(
            enabled=True,
            service_name="test",
            endpoint="http://localhost:6006",
            provider="otlp",
        )
    assert svc._enabled is False
    with svc.span("noop") as span:
        assert span is None


# --- Span creation with disabled tracing ---


def test_span_with_disabled_tracing_is_noop():
    svc = TracingService(enabled=False, service_name="test", endpoint="", provider="none")
    executed = []
    with svc.span("noop", key="val") as span:
        executed.append(span)
    assert executed == [None]


def test_span_records_exception_propagates():
    svc = TracingService(enabled=False, service_name="test", endpoint="", provider="none")
    with pytest.raises(ValueError):
        with svc.span("fail"):
            raise ValueError("expected")


# --- Prometheus metrics isolated tests ---


def test_pipeline_node_latency_uses_histogram():
    registry = CollectorRegistry()
    latency = Histogram(
        "test_rag_node_latency_seconds",
        "test",
        ["node"],
        buckets=[0.01, 0.1, 1.0],
        registry=registry,
    )
    latency.labels(node="retrieve").observe(0.05)
    sample_value = registry.get_sample_value(
        "test_rag_node_latency_seconds_count", {"node": "retrieve"}
    )
    assert sample_value == 1.0


def test_cache_hit_counter_incremented():
    registry = CollectorRegistry()
    cache_ops = Counter(
        "test_rag_cache_ops_isolated",
        "test",
        ["result"],
        registry=registry,
    )
    cache_ops.labels(result="hit").inc()
    cache_ops.labels(result="hit").inc()
    value = registry.get_sample_value("test_rag_cache_ops_isolated_total", {"result": "hit"})
    assert value == 2.0


def test_llm_token_usage_recorded():
    registry = CollectorRegistry()
    token_usage = Counter(
        "test_rag_token_usage_isolated",
        "test",
        ["model", "direction"],
        registry=registry,
    )
    token_usage.labels(model="nova-lite", direction="input").inc(100)
    token_usage.labels(model="nova-lite", direction="output").inc(50)
    in_val = registry.get_sample_value(
        "test_rag_token_usage_isolated_total", {"model": "nova-lite", "direction": "input"}
    )
    out_val = registry.get_sample_value(
        "test_rag_token_usage_isolated_total", {"model": "nova-lite", "direction": "output"}
    )
    assert in_val == 100.0
    assert out_val == 50.0


# --- /metrics endpoint ---


def test_metrics_endpoint_returns_prometheus_format():
    from app.main import app

    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    ct = resp.headers.get("content-type", "")
    assert "text/plain" in ct or "openmetrics" in ct or "application/openmetrics" in ct


def test_metrics_endpoint_contains_custom_metrics():
    from app.main import app

    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "rag_pipeline_node_latency_seconds" in body
    assert "rag_llm_token_usage_total" in body
    assert "rag_cache_operations_total" in body
    assert "rag_request_latency_seconds" in body


# --- Singleton helpers ---


def test_get_tracing_returns_disabled_service_when_not_initialized():
    import app.tracing as tracing_module

    original = tracing_module._service
    try:
        tracing_module._service = None
        from app.tracing import get_tracing

        svc = get_tracing()
        assert isinstance(svc, TracingService)
        assert svc._enabled is False
    finally:
        tracing_module._service = original


def test_init_tracing_sets_singleton():
    import app.tracing as tracing_module

    original = tracing_module._service
    try:
        settings = MagicMock()
        settings.enable_tracing = False
        settings.tracing_service_name = "test-svc"
        settings.tracing_endpoint = "http://localhost:6006"
        settings.tracing_provider = "none"

        from app.tracing import init_tracing

        svc = init_tracing(settings)
        assert tracing_module._service is svc
        assert isinstance(svc, TracingService)
    finally:
        tracing_module._service = original


def test_shutdown_noop_when_disabled():
    svc = TracingService(enabled=False, service_name="test", endpoint="", provider="none")
    svc.shutdown()
