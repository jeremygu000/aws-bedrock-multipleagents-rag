"""LLM client supporting DashScope OpenAI-compatible API and Ollama native API.

This client is intentionally lightweight so it can be used for intent detection,
query rewriting, and answer fallback without adding heavy framework coupling.
"""

from __future__ import annotations

import http.client
import json
import logging
import re
from collections.abc import Iterator
from typing import Any
from urllib import error, request

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Settings
from .secrets import resolve_qwen_api_key

logger = logging.getLogger(__name__)

_RETRYABLE_ERRORS = (
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    ConnectionError,
    TimeoutError,
    error.URLError,
)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class QwenClient:
    """HTTP client for LLM chat completions (DashScope or Ollama)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def is_configured(self) -> bool:
        """Return whether required Qwen API credentials are configured.

        When ``qwen_auth_required`` is False (e.g. local Ollama), always True.
        """

        if not self._settings.qwen_auth_required:
            return True
        try:
            return bool(self._get_api_key())
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=lambda retry_state: logger.warning(
            "LLM chat retry %d/%d after %s: %s",
            retry_state.attempt_number,
            3,
            type(retry_state.outcome.exception()).__name__ if retry_state.outcome else "unknown",
            retry_state.outcome.exception() if retry_state.outcome else "",
        ),
        reraise=True,
    )
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        if not self.is_configured():
            raise ValueError("Qwen API key is not configured.")

        effective_max = max_tokens if max_tokens is not None else self._settings.qwen_max_tokens

        if self._settings.qwen_use_ollama_native:
            return self._chat_ollama(system_prompt, user_prompt, effective_max)
        return self._chat_openai(system_prompt, user_prompt, effective_max)

    def _chat_ollama(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        base_url = self._settings.qwen_base_url.rstrip("/")
        # Strip /v1 suffix to get Ollama base
        ollama_base = re.sub(r"/v1/?$", "", base_url)
        url = f"{ollama_base}/api/chat"

        payload: dict[str, Any] = {
            "model": self._settings.qwen_model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": self._settings.qwen_temperature,
                "num_predict": max_tokens,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"Ollama API HTTP error: {exc.code} {detail}") from exc

        parsed = json.loads(raw)
        content = parsed.get("message", {}).get("content", "")
        return _THINK_TAG_RE.sub("", content).strip()

    def _chat_openai(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._settings.qwen_auth_required:
            headers["Authorization"] = f"Bearer {self._get_api_key()}"

        base_url = self._settings.qwen_base_url.rstrip("/")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": self._settings.qwen_model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._settings.qwen_temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url=url, data=body, method="POST", headers=headers)
        try:
            with request.urlopen(req, timeout=120) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"Qwen API HTTP error: {exc.code} {detail}") from exc

        return self._extract_text(json.loads(raw))

    # ------------------------------------------------------------------
    # Streaming (OpenAI-compatible only; Ollama native stream not needed)
    # ------------------------------------------------------------------

    def stream_chat(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        if not self.is_configured():
            raise ValueError("Qwen API key is not configured.")

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._settings.qwen_auth_required:
            headers["Authorization"] = f"Bearer {self._get_api_key()}"

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
        req = request.Request(url=url, data=body, method="POST", headers=headers)
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
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        delta = choices[0].get("delta", {})
        content = delta.get("content")
        return content if isinstance(content, str) else ""

    # ------------------------------------------------------------------
    # Embeddings (OpenAI-compatible only)
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=lambda retry_state: logger.warning(
            "Qwen embedding retry %d/%d after %s: %s",
            retry_state.attempt_number,
            3,
            type(retry_state.outcome.exception()).__name__ if retry_state.outcome else "unknown",
            retry_state.outcome.exception() if retry_state.outcome else "",
        ),
        reraise=True,
    )
    def embedding(self, text: str | list[str]) -> list[float] | list[list[float]]:
        batch_mode = isinstance(text, list)
        if not self.is_configured():
            raise ValueError("Qwen API key is not configured.")

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._settings.qwen_auth_required:
            headers["Authorization"] = f"Bearer {self._get_api_key()}"

        base_url = self._settings.qwen_base_url.rstrip("/")
        url = f"{base_url}/embeddings"
        payload = {
            "model": self._settings.qwen_embedding_model_id,
            "input": text,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url=url, data=body, method="POST", headers=headers)
        try:
            with request.urlopen(req, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"Qwen embeddings API HTTP error: {exc.code} {detail}") from exc

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_api_key(self) -> str:
        return resolve_qwen_api_key(self._settings).strip()

    def _extract_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return _THINK_TAG_RE.sub("", content).strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        parts.append(text_value.strip())
            return _THINK_TAG_RE.sub("", "\n".join(parts)).strip()
        return ""

    def _extract_embedding(self, payload: dict[str, Any]) -> list[float]:
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
