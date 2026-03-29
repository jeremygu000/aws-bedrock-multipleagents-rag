from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app


def _parse_sse_events(response_text: str) -> list[dict]:
    normalized = response_text.replace("\r\n", "\n").replace("\r", "\n")
    events: list[dict] = []
    for block in normalized.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_name = None
        data_str = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:") :].strip()
        if event_name is not None or data_str is not None:
            events.append({"event": event_name, "data": data_str})
    return events


def _stream_get(query_str: str) -> tuple[int, str, dict]:
    with TestClient(app, raise_server_exceptions=False) as cl:
        with cl.stream("GET", f"/retrieve/stream?{query_str}") as resp:
            status = resp.status_code
            headers = dict(resp.headers)
            text = resp.read().decode()
    return status, text, headers


def _normal_state() -> dict:
    return {
        "cache_hit": False,
        "intent": "factual",
        "complexity": "medium",
        "retrieval_mode": "mix",
        "preferred_model": "nova-lite",
        "reranked_hits": [],
        "hits": [],
        "citations": [],
        "hl_keywords": [],
        "ll_keywords": [],
        "graph_context": None,
        "query_embedding": None,
        "rewritten_query": None,
    }


def _cache_state() -> dict:
    return {
        "cache_hit": True,
        "intent": "factual",
        "complexity": "medium",
        "answer_model": "cache",
        "answer": "cached-answer",
        "citations": [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamEndpointDisabledReturns404:
    @patch("app.main.settings")
    def test_stream_disabled_returns_404(self, mock_settings):
        mock_settings.enable_streaming = False
        mock_settings.enable_query_cache = False
        mock_settings.enable_graph_retrieval = False

        with TestClient(app, raise_server_exceptions=False) as cl:
            response = cl.get("/retrieve/stream?query=test")
        assert response.status_code == 404


class TestStreamEndpointValidation:
    def test_stream_empty_query_returns_422(self):
        with TestClient(app, raise_server_exceptions=False) as cl:
            response = cl.get("/retrieve/stream")
        assert response.status_code == 422


class TestStreamNormalFlow:
    @patch("app.main.workflow")
    @patch("app.main.settings")
    def test_stream_normal_flow_metadata_tokens_done(self, mock_settings, mock_workflow):
        mock_settings.enable_streaming = True
        mock_settings.enable_query_cache = False
        mock_workflow.run_until_generate.return_value = _normal_state()
        mock_workflow.query_cache = None
        mock_workflow.answer_generator.generate_stream.return_value = (
            iter(["chunk1", "chunk2"]),
            "nova-lite",
        )

        status, text, _ = _stream_get("query=hello")
        assert status == 200

        events = _parse_sse_events(text)
        event_names = [e["event"] for e in events]

        assert event_names[0] == "metadata"
        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) == 2
        assert json.loads(token_events[0]["data"])["text"] == "chunk1"
        assert json.loads(token_events[1]["data"])["text"] == "chunk2"
        assert event_names[-1] == "done"


class TestStreamCacheHit:
    @patch("app.main.workflow")
    @patch("app.main.settings")
    def test_stream_cache_hit_returns_cached_answer(self, mock_settings, mock_workflow):
        mock_settings.enable_streaming = True
        mock_settings.enable_query_cache = False
        mock_workflow.run_until_generate.return_value = _cache_state()

        status, text, _ = _stream_get("query=what+is+apra")
        assert status == 200

        events = _parse_sse_events(text)
        event_names = [e["event"] for e in events]
        assert "metadata" in event_names
        assert "token" in event_names
        assert "done" in event_names

        token_event = next(e for e in events if e["event"] == "token")
        assert json.loads(token_event["data"])["text"] == "cached-answer"

        done_event = next(e for e in events if e["event"] == "done")
        assert json.loads(done_event["data"])["cache_stored"] is False


class TestStreamPipelineError:
    @patch("app.main.workflow")
    @patch("app.main.settings")
    def test_stream_pipeline_error_returns_error_event(self, mock_settings, mock_workflow):
        mock_settings.enable_streaming = True
        mock_settings.enable_query_cache = False
        mock_workflow.run_until_generate.side_effect = RuntimeError("boom")

        status, text, _ = _stream_get("query=failing+query")
        assert status == 200

        events = _parse_sse_events(text)
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) >= 1
        error_data = json.loads(error_events[0]["data"])
        assert error_data["error"] == "pipeline_error"
        assert "boom" in error_data["detail"]


class TestStreamGenerationSetupError:
    @patch("app.main.workflow")
    @patch("app.main.settings")
    def test_stream_generation_setup_error(self, mock_settings, mock_workflow):
        mock_settings.enable_streaming = True
        mock_settings.enable_query_cache = False
        mock_workflow.run_until_generate.return_value = _normal_state()
        mock_workflow.answer_generator.generate_stream.side_effect = ValueError(
            "model config error"
        )

        status, text, _ = _stream_get("query=test")
        assert status == 200

        events = _parse_sse_events(text)
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) >= 1
        assert json.loads(error_events[0]["data"])["error"] == "generation_error"


class TestStreamTokenStreamingError:
    @patch("app.main.workflow")
    @patch("app.main.settings")
    def test_stream_token_streaming_error(self, mock_settings, mock_workflow):
        mock_settings.enable_streaming = True
        mock_settings.enable_query_cache = False
        mock_workflow.run_until_generate.return_value = _normal_state()
        mock_workflow.query_cache = None

        def _bad_stream():
            yield "partial-chunk"
            raise RuntimeError("stream broke")

        mock_workflow.answer_generator.generate_stream.return_value = (
            _bad_stream(),
            "nova-lite",
        )

        status, text, _ = _stream_get("query=test+error")
        assert status == 200

        events = _parse_sse_events(text)
        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) >= 1
        assert json.loads(token_events[0]["data"])["text"] == "partial-chunk"

        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) >= 1
        assert json.loads(error_events[0]["data"])["error"] == "stream_error"


class TestStreamCacheStore:
    @patch("app.main.workflow")
    @patch("app.main.settings")
    def test_stream_cache_stored_after_completion(self, mock_settings, mock_workflow):
        mock_settings.enable_streaming = True
        mock_settings.enable_query_cache = True

        state = _normal_state()
        state["query_embedding"] = [0.1, 0.2, 0.3]
        state["citations"] = []
        mock_workflow.run_until_generate.return_value = state
        mock_workflow.answer_generator.generate_stream.return_value = (
            iter(["hello"]),
            "nova-lite",
        )

        mock_cache = MagicMock()
        mock_cache.store.return_value = None
        mock_workflow.query_cache = mock_cache

        status, text, _ = _stream_get("query=cache+test")
        assert status == 200

        events = _parse_sse_events(text)
        done_event = next(e for e in events if e["event"] == "done")
        assert json.loads(done_event["data"])["cache_stored"] is True

    @patch("app.main.workflow")
    @patch("app.main.settings")
    def test_stream_no_cache_store_when_disabled(self, mock_settings, mock_workflow):
        mock_settings.enable_streaming = True
        mock_settings.enable_query_cache = False

        state = _normal_state()
        state["query_embedding"] = [0.1, 0.2, 0.3]
        mock_workflow.run_until_generate.return_value = state
        mock_workflow.query_cache = None
        mock_workflow.answer_generator.generate_stream.return_value = (
            iter(["hello"]),
            "nova-lite",
        )

        status, text, _ = _stream_get("query=no+cache")
        assert status == 200

        events = _parse_sse_events(text)
        done_event = next(e for e in events if e["event"] == "done")
        assert json.loads(done_event["data"])["cache_stored"] is False


class TestStreamCustomTopK:
    @patch("app.main.workflow")
    @patch("app.main.settings")
    def test_stream_custom_top_k(self, mock_settings, mock_workflow):
        mock_settings.enable_streaming = True
        mock_settings.enable_query_cache = False
        mock_workflow.run_until_generate.return_value = _normal_state()
        mock_workflow.query_cache = None
        mock_workflow.answer_generator.generate_stream.return_value = (
            iter([]),
            "nova-lite",
        )

        _stream_get("query=test+topk&top_k=3")

        mock_workflow.run_until_generate.assert_called_once()
        call_args = mock_workflow.run_until_generate.call_args
        actual_top_k = call_args.args[1] if call_args.args else call_args.kwargs.get("top_k")
        assert actual_top_k == 3


class TestStreamResponseHeaders:
    @patch("app.main.workflow")
    @patch("app.main.settings")
    def test_stream_x_accel_buffering_header(self, mock_settings, mock_workflow):
        mock_settings.enable_streaming = True
        mock_settings.enable_query_cache = False
        mock_workflow.run_until_generate.return_value = _normal_state()
        mock_workflow.query_cache = None
        mock_workflow.answer_generator.generate_stream.return_value = (
            iter([]),
            "nova-lite",
        )

        _, _, headers = _stream_get("query=header+test")
        assert headers.get("x-accel-buffering") == "no"
