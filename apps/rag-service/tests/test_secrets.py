from __future__ import annotations

from app import secrets
from app.config import Settings


def test_extract_key_from_secret_json_preferred_key() -> None:
    secret_string = '{"DASHSCOPE_API_KEY":"abc-123","unused":"x"}'
    value = secrets._extract_key_from_secret(secret_string, "DASHSCOPE_API_KEY")
    assert value == "abc-123"


def test_extract_key_from_secret_key_value_line() -> None:
    secret_string = "DASHSCOPE_API_KEY=abc-456\nOTHER_KEY=zzz"
    value = secrets._extract_key_from_secret(secret_string, "DASHSCOPE_API_KEY")
    assert value == "abc-456"


def test_extract_key_from_secret_plain_text_fallback() -> None:
    secret_string = "plain-secret-value"
    value = secrets._extract_key_from_secret(secret_string, "DASHSCOPE_API_KEY")
    assert value == "plain-secret-value"


def test_resolve_db_password_prefers_env() -> None:
    settings = Settings(
        RAG_DB_PASSWORD="db-password-from-env",
        RAG_DB_PASSWORD_SECRET_ARN="arn:aws:secretsmanager:region:acct:secret:ignored",
    )
    assert secrets.resolve_db_password(settings) == "db-password-from-env"


def test_resolve_qwen_api_key_prefers_env() -> None:
    settings = Settings(
        QWEN_API_KEY="qwen-key-from-env",
        QWEN_API_KEY_SECRET_ARN="arn:aws:secretsmanager:region:acct:secret:ignored",
    )
    assert secrets.resolve_qwen_api_key(settings) == "qwen-key-from-env"


def test_resolve_qwen_api_key_from_secret(monkeypatch) -> None:
    settings = Settings(
        QWEN_API_KEY_SECRET_ARN="arn:aws:secretsmanager:region:acct:secret:qwen",
        QWEN_API_KEY_SECRET_KEY="DASHSCOPE_API_KEY",
        RAG_AWS_REGION="ap-southeast-2",
    )

    def fake_fetch(secret_arn: str, region: str | None) -> str:
        assert secret_arn == settings.qwen_api_key_secret_arn
        assert region == settings.aws_region
        return "DASHSCOPE_API_KEY=from-secret"

    monkeypatch.setattr(secrets, "_fetch_secret_string", fake_fetch)
    assert secrets.resolve_qwen_api_key(settings) == "from-secret"
