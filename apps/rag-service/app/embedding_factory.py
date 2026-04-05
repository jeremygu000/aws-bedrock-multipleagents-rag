"""Factory for embedding clients.

Returns either ``BedrockEmbeddingClient`` or ``QwenClient`` based on
``settings.embedding_provider``.  Both expose the same
``embedding(text: str | list[str])`` method.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .config import Settings


@runtime_checkable
class EmbeddingClient(Protocol):
    def embedding(self, text: str | list[str]) -> list[float] | list[list[float]]: ...


def get_embedding_client(settings: Settings) -> EmbeddingClient:
    if settings.embedding_provider == "bedrock":
        from .bedrock_embedding_client import BedrockEmbeddingClient

        return BedrockEmbeddingClient(settings)

    from .qwen_client import QwenClient

    return QwenClient(settings)
