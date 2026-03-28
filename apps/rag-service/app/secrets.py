"""Secret resolution helpers for runtime credentials.

This module supports:
- database password resolution
- Qwen API key resolution
- Neo4j password resolution

Both paths prefer explicit env values for local development, then fall back to
AWS Secrets Manager for production safety.
"""

from __future__ import annotations

import json
from functools import lru_cache

from .config import Settings


@lru_cache(maxsize=128)
def _fetch_secret_string(secret_arn: str, region: str | None) -> str:
    """Fetch raw `SecretString` from AWS Secrets Manager."""

    import boto3

    client = boto3.client("secretsmanager", region_name=region or None)
    response = client.get_secret_value(SecretId=secret_arn)
    secret_string = response.get("SecretString")
    if not isinstance(secret_string, str) or not secret_string.strip():
        raise ValueError("SecretString is empty or missing")
    return secret_string.strip()


def _extract_key_from_secret(secret_string: str, preferred_key: str) -> str:
    """Extract value from JSON or KEY=VALUE formatted secret payload.

    Supported formats:
    1. JSON object: `{"DASHSCOPE_API_KEY":"..."}`
    2. Line format: `DASHSCOPE_API_KEY=...`
    3. Plain value: `sk-...` (accepted when no better match exists)
    """

    try:
        parsed = json.loads(secret_string)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        value = parsed.get(preferred_key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        # Common fallbacks when secret key naming is not standardized.
        for fallback_key in ("DASHSCOPE_API_KEY", "QWEN_API_KEY", "api_key", "password"):
            fallback_value = parsed.get(fallback_key)
            if isinstance(fallback_value, str) and fallback_value.strip():
                return fallback_value.strip()

    for raw_line in secret_string.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == preferred_key and value.strip():
            return value.strip()
        if key.strip() in {"DASHSCOPE_API_KEY", "QWEN_API_KEY", "api_key"} and value.strip():
            return value.strip()

    if "=" not in secret_string and secret_string.strip():
        # Plain text secret payload fallback.
        return secret_string.strip()

    raise ValueError(f"Secret does not contain a non-empty value for key '{preferred_key}'")


def resolve_db_password(settings: Settings) -> str:
    """Resolve database password with env override, then Secrets Manager."""

    if settings.db_password:
        return settings.db_password

    if not settings.db_password_secret_arn:
        return ""

    secret_string = _fetch_secret_string(settings.db_password_secret_arn, settings.aws_region)
    return _extract_key_from_secret(secret_string, settings.db_password_secret_json_key)


def resolve_qwen_api_key(settings: Settings) -> str:
    """Resolve Qwen API key with env override, then Secrets Manager."""

    if settings.qwen_api_key:
        return settings.qwen_api_key

    if not settings.qwen_api_key_secret_arn:
        return ""

    secret_string = _fetch_secret_string(settings.qwen_api_key_secret_arn, settings.aws_region)
    return _extract_key_from_secret(secret_string, settings.qwen_api_key_secret_key)


def resolve_neo4j_password(settings: Settings) -> str:
    """Resolve Neo4j password with env override, then Secrets Manager."""

    if settings.neo4j_password:
        return settings.neo4j_password

    if not settings.neo4j_password_secret_arn:
        return ""

    secret_string = _fetch_secret_string(settings.neo4j_password_secret_arn, settings.aws_region)
    return _extract_key_from_secret(secret_string, "password")
