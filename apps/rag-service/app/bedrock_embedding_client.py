"""Bedrock Titan Text Embeddings client.

Wraps boto3 ``bedrock-runtime`` ``invoke_model`` to match the same
``embedding(text)`` interface used by ``QwenClient``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError

from .config import Settings

logger = logging.getLogger(__name__)


class BedrockEmbeddingClient:
    """Embedding client using Amazon Titan Text Embeddings V2 via Bedrock."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_id = settings.bedrock_embedding_model_id
        self._dimensions = settings.embedding_dimensions
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            kwargs: dict[str, Any] = {"service_name": "bedrock-runtime"}
            if self._settings.aws_region:
                kwargs["region_name"] = self._settings.aws_region
            self._client = boto3.client(**kwargs)
        return self._client

    def embedding(self, text: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(text, list):
            return [self._embed_single(t) for t in text]
        return self._embed_single(text)

    def _embed_single(self, text: str) -> list[float]:
        body = json.dumps({
            "inputText": text,
            "dimensions": self._dimensions,
            "normalize": True,
        })
        try:
            response = self._get_client().invoke_model(
                body=body,
                modelId=self._model_id,
                accept="application/json",
                contentType="application/json",
            )
        except ClientError as exc:
            raise ValueError(
                f"Bedrock embedding error: {exc.response['Error']['Message']}"
            ) from exc

        result = json.loads(response["body"].read())
        vector = result.get("embedding", [])
        if not vector:
            raise ValueError("Bedrock returned empty embedding.")
        return [float(v) for v in vector]
