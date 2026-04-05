from __future__ import annotations

import io
import json
from urllib import error

import pytest

from app.config import Settings
from app.qwen_client import QwenClient

OPENAI_SETTINGS = {
    "QWEN_AUTH_REQUIRED": "true",
    "QWEN_USE_OLLAMA_NATIVE": "false",
}


def test_is_configured_true(monkeypatch) -> None:
    client = QwenClient(Settings(**OPENAI_SETTINGS))
    monkeypatch.setattr(client, "_get_api_key", lambda: "secret-key")
    assert client.is_configured() is True


def test_is_configured_false_when_key_resolve_fails(monkeypatch) -> None:
    client = QwenClient(Settings(**OPENAI_SETTINGS))

    def boom() -> str:
        raise ValueError("missing")

    monkeypatch.setattr(client, "_get_api_key", boom)
    assert client.is_configured() is False


def test_is_configured_true_when_auth_not_required() -> None:
    client = QwenClient(Settings(QWEN_AUTH_REQUIRED="false"))
    assert client.is_configured() is True


def test_extract_text_from_string_content() -> None:
    client = QwenClient(Settings())
    payload = {"choices": [{"message": {"content": "hello world"}}]}
    assert client._extract_text(payload) == "hello world"


def test_extract_text_from_list_content() -> None:
    client = QwenClient(Settings())
    payload = {
        "choices": [
            {
                "message": {
                    "content": [{"text": "line1"}, {"text": "line2"}],
                }
            }
        ]
    }
    assert client._extract_text(payload) == "line1\nline2"


def test_extract_text_strips_think_tags() -> None:
    client = QwenClient(Settings())
    payload = {
        "choices": [
            {"message": {"content": "<think>\nreasoning here\n</think>\nactual answer"}}
        ]
    }
    assert client._extract_text(payload) == "actual answer"


def test_chat_openai_success(monkeypatch) -> None:
    settings = Settings(
        QWEN_API_KEY="secret",
        QWEN_MODEL_ID="qwen-plus",
        QWEN_BASE_URL="https://example.com/v1",
        QWEN_AUTH_REQUIRED="true",
        QWEN_USE_OLLAMA_NATIVE="false",
    )
    client = QwenClient(settings)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "ok-answer"}}]}).encode("utf-8")

    def fake_urlopen(req, timeout: int):
        assert req.full_url == "https://example.com/v1/chat/completions"
        assert timeout == 120
        return FakeResponse()

    monkeypatch.setattr("app.qwen_client.request.urlopen", fake_urlopen)
    assert client.chat("sys", "usr") == "ok-answer"


def test_chat_ollama_success(monkeypatch) -> None:
    settings = Settings(
        QWEN_MODEL_ID="qwen3:32b",
        QWEN_BASE_URL="http://localhost:11434/v1",
        QWEN_USE_OLLAMA_NATIVE="true",
        QWEN_AUTH_REQUIRED="false",
    )
    client = QwenClient(settings)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"message": {"content": "ollama-answer"}}).encode("utf-8")

    def fake_urlopen(req, timeout: int):
        assert req.full_url == "http://localhost:11434/api/chat"
        assert timeout == 120
        body = json.loads(req.data)
        assert body["think"] is False
        assert body["stream"] is False
        return FakeResponse()

    monkeypatch.setattr("app.qwen_client.request.urlopen", fake_urlopen)
    assert client.chat("sys", "usr") == "ollama-answer"


def test_chat_http_error(monkeypatch) -> None:
    settings = Settings(
        QWEN_API_KEY="secret",
        QWEN_AUTH_REQUIRED="true",
        QWEN_USE_OLLAMA_NATIVE="false",
    )
    client = QwenClient(settings)

    def fake_urlopen(_req, timeout: int):
        raise error.HTTPError(
            url="https://x",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"nope"}'),
        )

    monkeypatch.setattr("app.qwen_client.request.urlopen", fake_urlopen)
    with pytest.raises(ValueError, match="Qwen API HTTP error"):
        client.chat("sys", "usr")


def test_extract_embedding() -> None:
    client = QwenClient(Settings())
    payload = {"data": [{"embedding": [0.1, 2, 3.5]}]}
    assert client._extract_embedding(payload) == [0.1, 2.0, 3.5]


def test_embedding_success(monkeypatch) -> None:
    settings = Settings(
        QWEN_API_KEY="secret",
        QWEN_BASE_URL="https://example.com/v1",
        QWEN_EMBEDDING_MODEL_ID="text-embedding-v3",
        QWEN_AUTH_REQUIRED="true",
        QWEN_USE_OLLAMA_NATIVE="false",
    )
    client = QwenClient(settings)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps({"data": [{"embedding": [0.1, 0.2]}]}).encode("utf-8")

    def fake_urlopen(req, timeout: int):
        assert req.full_url == "https://example.com/v1/embeddings"
        assert timeout == 60
        return FakeResponse()

    monkeypatch.setattr("app.qwen_client.request.urlopen", fake_urlopen)
    assert client.embedding("hello") == [0.1, 0.2]


def test_embedding_http_error(monkeypatch) -> None:
    settings = Settings(
        QWEN_API_KEY="secret",
        QWEN_AUTH_REQUIRED="true",
        QWEN_USE_OLLAMA_NATIVE="false",
    )
    client = QwenClient(settings)

    def fake_urlopen(_req, timeout: int):
        raise error.HTTPError(
            url="https://x",
            code=500,
            msg="Server Error",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"boom"}'),
        )

    monkeypatch.setattr("app.qwen_client.request.urlopen", fake_urlopen)
    with pytest.raises(ValueError, match="Qwen embeddings API HTTP error"):
        client.embedding("hello")
