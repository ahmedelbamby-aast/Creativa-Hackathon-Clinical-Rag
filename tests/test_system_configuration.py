"""Consistency checks for deterministic project defaults."""

from pathlib import Path

from src.config import AppConfig


def test_gemini_model_default_and_template_are_consistent(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert AppConfig().gemini_model == "gemini-3.6-flash"
    for name in (".env.development.example", ".env.deployment.example"):
        template = Path(name).read_text(encoding="utf-8")
        assert "GEMINI_MODEL=gemini-3.6-flash" in template


def test_deployment_rejects_local_embeddings() -> None:
    config = AppConfig(app_env="deployment", embedding_provider="local")
    try:
        config.validate()
    except ValueError as exc:
        assert "Deployment requires EMBEDDING_PROVIDER=gemini" in str(exc)
    else:
        raise AssertionError("deployment unexpectedly accepted local embeddings")


def test_schema_operations_prefer_unpooled_database_url() -> None:
    config = AppConfig(
        database_url="postgresql://pooled/db",
        database_url_unpooled="postgresql://direct/db",
    )
    assert config.schema_database_url == "postgresql://direct/db"
