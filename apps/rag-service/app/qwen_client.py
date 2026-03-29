"""Qwen client over DashScope OpenAI-compatible API.

This client is intentionally lightweight so it can be used for intent detection,
query rewriting, and answer fallback without adding heavy framework coupling.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from urllib import error, request

from .config import Settings
from .secrets import resolve_qwen_api_key


class QwenClient:
    """HTTP client for Qwen chat completions."""

    def __init__(self, settings: Settings) -> None:
        """Store settings needed to call DashScope compatible endpoint."""

        self._settings = settings

    def is_configured(self) -> bool:
        """Return whether required Qwen API credentials are configured."""

        try:
            return bool(self._get_api_key())
        except Exception:
            return False

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Call Qwen chat-completions API and return assistant text output."""

        if not self.is_configured():
            raise ValueError("Qwen API key is not configured.")
        api_key = self._get_api_key()

        base_url = self._settings.qwen_base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": self._settings.qwen_model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._settings.qwen_temperature,
            "max_tokens": self._settings.qwen_max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with request.urlopen(req, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"Qwen API HTTP error: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise ValueError(f"Qwen API connection error: {exc}") from exc

        parsed = json.loads(raw)
        return self._extract_text(parsed)

    def stream_chat(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Call Qwen chat-completions API with streaming and yield text chunks."""

        if not self.is_configured():
            raise ValueError("Qwen API key is not configured.")
        api_key = self._get_api_key()

        base_url = self._settings.qwen_base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": self._settings.qwen_model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._settings.qwen_temperature,
            "max_tokens": self._settings.qwen_max_tokens,
            "stream": True,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            resp = request.urlopen(req, timeout=60)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"Qwen API HTTP error: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise ValueError(f"Qwen API connection error: {exc}") from exc

        try:
            yield from self._parse_sse_stream(resp)
        finally:
            resp.close()

    def _parse_sse_stream(self, resp: Any) -> Iterator[str]:
        """Parse SSE lines from an HTTP response and yield text deltas."""

        buffer = ""
        for raw_line in resp:
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            buffer += line
            while "\n" in buffer:
                sse_line, buffer = buffer.split("\n", 1)
                sse_line = sse_line.rstrip("\r")
                if not sse_line.startswith("data: "):
                    continue
                data_str = sse_line[6:]
                if data_str.strip() == "[DONE]":
                    return
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = self._extract_stream_delta(chunk)
                if delta:
                    yield delta

    @staticmethod
    def _extract_stream_delta(chunk: dict[str, Any]) -> str:
        """Extract text delta from a streaming chunk."""

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        delta = choices[0].get("delta", {})
        content = delta.get("content")
        return content if isinstance(content, str) else ""

    def embedding(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Call Qwen embeddings API.

        Accepts a single string or a list of strings (batch mode).
        Returns a single vector or a list of vectors respectively.
        """

        batch_mode = isinstance(text, list)
        if not self.is_configured():
            raise ValueError("Qwen API key is not configured.")
        api_key = self._get_api_key()

        base_url = self._settings.qwen_base_url.rstrip("/")
        url = f"{base_url}/embeddings"
        payload = {
            "model": self._settings.qwen_embedding_model_id,
            "input": text,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"Qwen embeddings API HTTP error: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise ValueError(f"Qwen embeddings API connection error: {exc}") from exc

        parsed = json.loads(raw)

        if batch_mode:
            vectors = self._extract_all_embeddings(parsed)
            if len(vectors) != len(text):
                raise ValueError(
                    f"Qwen embeddings API returned {len(vectors)} vectors "
                    f"for {len(text)} inputs."
                )
            return vectors

        single = self._extract_embedding(parsed)
        if not single:
            raise ValueError("Qwen embeddings API returned empty embedding.")
        return single

    def _get_api_key(self) -> str:
        """Resolve Qwen API key from env or Secrets Manager."""

        return resolve_qwen_api_key(self._settings).strip()

    def _extract_text(self, payload: dict[str, Any]) -> str:
        """Extract assistant text from OpenAI-compatible chat completion payload."""

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        parts.append(text_value.strip())
            return "\n".join(parts).strip()
        return ""

    def _extract_embedding(self, payload: dict[str, Any]) -> list[float]:
        """Extract embedding vector from OpenAI-compatible embeddings payload."""

        data = payload.get("data")
        if not isinstance(data, list) or not data:
            return []
        first = data[0]
        if not isinstance(first, dict):
            return []
        vector = first.get("embedding")
        if not isinstance(vector, list):
            return []

        normalized: list[float] = []
        for value in vector:
            if isinstance(value, (int, float)):
                normalized.append(float(value))
        return normalized

    def _extract_all_embeddings(self, payload: dict[str, Any]) -> list[list[float]]:
        data = payload.get("data")
        if not isinstance(data, list):
            return []

        sorted_data = sorted(data, key=lambda item: item.get("index", 0))
        vectors: list[list[float]] = []
        for item in sorted_data:
            if not isinstance(item, dict):
                continue
            raw = item.get("embedding")
            if not isinstance(raw, list):
                continue
            vector = [float(v) for v in raw if isinstance(v, (int, float))]
            vectors.append(vector)
        return vectors
