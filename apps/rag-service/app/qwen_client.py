"""Qwen client over DashScope OpenAI-compatible API.

This client is intentionally lightweight so it can be used for intent detection,
query rewriting, and answer fallback without adding heavy framework coupling.
"""

from __future__ import annotations

import json
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
