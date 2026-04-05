from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from app.bedrock_embedding_client import BedrockEmbeddingClient
from app.config import Settings


def get_test_settings(**overrides) -> Settings:
    defaults = {
        "RAG_DB_HOST": "localhost",
        "RAG_DB_PASSWORD": "test",
        "RAG_S3_BUCKET": "test",
    }
    defaults.update(overrides)
    return Settings(**defaults)


@patch("app.bedrock_embedding_client.boto3")
def test_embedding_single_string(mock_boto3) -> None:
    settings = get_test_settings()
    client = BedrockEmbeddingClient(settings)

    mock_bedrock = MagicMock()
    mock_boto3.client.return_value = mock_bedrock

    mock_response = {
        "body": io.BytesIO(
            json.dumps({"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 5}).encode()
        )
    }
    mock_bedrock.invoke_model.return_value = mock_response

    result = client.embedding("hello world")

    assert result == [0.1, 0.2, 0.3]
    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)
    mock_bedrock.invoke_model.assert_called_once()


@patch("app.bedrock_embedding_client.boto3")
def test_embedding_list_of_strings(mock_boto3) -> None:
    settings = get_test_settings()
    client = BedrockEmbeddingClient(settings)

    mock_bedrock = MagicMock()
    mock_boto3.client.return_value = mock_bedrock

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]

    call_count = [0]

    def side_effect(*args, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        return {
            "body": io.BytesIO(
                json.dumps({
                    "embedding": embeddings[idx],
                    "inputTextTokenCount": 5
                }).encode()
            )
        }

    mock_bedrock.invoke_model.side_effect = side_effect

    texts = ["hello", "world", "test"]
    result = client.embedding(texts)

    assert result == embeddings
    assert len(result) == 3
    assert all(isinstance(r, list) for r in result)
    assert mock_bedrock.invoke_model.call_count == 3


@patch("app.bedrock_embedding_client.boto3")
def test_embed_single_raises_on_empty_embedding(mock_boto3) -> None:
    settings = get_test_settings()
    client = BedrockEmbeddingClient(settings)

    mock_bedrock = MagicMock()
    mock_boto3.client.return_value = mock_bedrock

    mock_response = {
        "body": io.BytesIO(
            json.dumps({"embedding": [], "inputTextTokenCount": 0}).encode()
        )
    }
    mock_bedrock.invoke_model.return_value = mock_response

    with pytest.raises(ValueError, match="Bedrock returned empty embedding"):
        client._embed_single("hello")


@patch("app.bedrock_embedding_client.boto3")
def test_embed_single_raises_on_client_error(mock_boto3) -> None:
    settings = get_test_settings()
    client = BedrockEmbeddingClient(settings)

    mock_bedrock = MagicMock()
    mock_boto3.client.return_value = mock_bedrock

    error_response = {
        "Error": {
            "Code": "ValidationException",
            "Message": "Invalid input format"
        }
    }
    mock_bedrock.invoke_model.side_effect = ClientError(
        error_response, "InvokeModel"
    )

    with pytest.raises(ValueError, match="Bedrock embedding error.*Invalid input format"):
        client._embed_single("hello")


@patch("app.bedrock_embedding_client.boto3")
def test_lazy_init_boto3_not_called_in_init(mock_boto3) -> None:
    settings = get_test_settings()
    client = BedrockEmbeddingClient(settings)

    mock_boto3.client.assert_not_called()
    assert client._client is None


@patch("app.bedrock_embedding_client.boto3")
def test_lazy_init_boto3_called_on_first_embedding(mock_boto3) -> None:
    settings = get_test_settings()
    client = BedrockEmbeddingClient(settings)

    mock_bedrock = MagicMock()
    mock_boto3.client.return_value = mock_bedrock

    mock_response = {
        "body": io.BytesIO(
            json.dumps({"embedding": [0.1, 0.2], "inputTextTokenCount": 5}).encode()
        )
    }
    mock_bedrock.invoke_model.return_value = mock_response

    mock_boto3.client.assert_not_called()

    client.embedding("hello")

    mock_boto3.client.assert_called_once()
    assert client._client is not None


@patch("app.bedrock_embedding_client.boto3")
def test_region_passed_to_boto3_client(mock_boto3) -> None:
    settings = get_test_settings(RAG_AWS_REGION="eu-west-1")
    client = BedrockEmbeddingClient(settings)

    mock_bedrock = MagicMock()
    mock_boto3.client.return_value = mock_bedrock

    mock_response = {
        "body": io.BytesIO(
            json.dumps({"embedding": [0.1, 0.2], "inputTextTokenCount": 5}).encode()
        )
    }
    mock_bedrock.invoke_model.return_value = mock_response

    client.embedding("hello")

    mock_boto3.client.assert_called_once_with(
        service_name="bedrock-runtime",
        region_name="eu-west-1"
    )


@patch("app.bedrock_embedding_client.boto3")
def test_client_reused_across_multiple_calls(mock_boto3) -> None:
    settings = get_test_settings()
    client = BedrockEmbeddingClient(settings)

    mock_bedrock = MagicMock()
    mock_boto3.client.return_value = mock_bedrock

    def create_response(*args, **kwargs):
        return {
            "body": io.BytesIO(
                json.dumps({"embedding": [0.1, 0.2], "inputTextTokenCount": 5}).encode()
            )
        }

    mock_bedrock.invoke_model.side_effect = create_response

    client.embedding("hello")
    client.embedding("world")
    client.embedding("test")

    mock_boto3.client.assert_called_once()

    assert mock_bedrock.invoke_model.call_count == 3
