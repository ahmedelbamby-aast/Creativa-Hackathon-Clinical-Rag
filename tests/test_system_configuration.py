"""Consistency checks for deterministic project defaults."""

from pathlib import Path

from src.config import AppConfig


def test_gemini_model_default_and_template_are_consistent(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert AppConfig().gemini_model == "gemini-2.5-flash"
    for name in (".env.development.example", ".env.deployment.example"):
        template = Path(name).read_text(encoding="utf-8")
        assert "GEMINI_MODEL=gemini-2.5-flash" in template
    workflow = Path(".github/workflows/ingest-production.yml").read_text(encoding="utf-8")
    assert "GEMINI_MODEL: gemini-2.5-flash" in workflow


def test_deployment_template_fits_free_embedding_quota_profile() -> None:
    template = Path(".env.deployment.example").read_text(encoding="utf-8")

    assert "CHUNK_SIZE=3000" in template
    assert "CHUNK_OVERLAP=300" in template
    assert "ONLINE_EMBEDDING_RPM=90" in template
    assert "OCR_LANGUAGE=eng" in template


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


def test_extractive_generation_requires_no_runtime_credential() -> None:
    config = AppConfig(
        generation_provider="extractive",
        gemini_api_key="",
        ai_gateway_api_key="",
        vercel_oidc_token="",
    )

    config.validate()

    assert config.generation_configured is True
