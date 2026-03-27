from __future__ import annotations

import io
import json
from urllib import error

import pytest

from app.config import Settings
from app.qwen_client import QwenClient


def test_is_configured_true(monkeypatch) -> None:
    client = QwenClient(Settings())
    monkeypatch.setattr(client, "_get_api_key", lambda: "secret-key")
    assert client.is_configured() is True


def test_is_configured_false_when_key_resolve_fails(monkeypatch) -> None:
    client = QwenClient(Settings())

    def boom() -> str:
        raise ValueError("missing")

    monkeypatch.setattr(client, "_get_api_key", boom)
    assert client.is_configured() is False


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


def test_chat_success(monkeypatch) -> None:
    settings = Settings(
        QWEN_API_KEY="secret",
        QWEN_MODEL_ID="qwen-plus",
        QWEN_BASE_URL="https://example.com/v1",
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
        assert timeout == 30
        return FakeResponse()

    monkeypatch.setattr("app.qwen_client.request.urlopen", fake_urlopen)
    assert client.chat("sys", "usr") == "ok-answer"


def test_chat_http_error(monkeypatch) -> None:
    settings = Settings(QWEN_API_KEY="secret")
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
        assert timeout == 30
        return FakeResponse()

    monkeypatch.setattr("app.qwen_client.request.urlopen", fake_urlopen)
    assert client.embedding("hello") == [0.1, 0.2]


def test_embedding_http_error(monkeypatch) -> None:
    settings = Settings(QWEN_API_KEY="secret")
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
