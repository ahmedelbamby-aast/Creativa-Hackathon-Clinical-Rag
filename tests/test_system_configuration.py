"""Consistency checks for deterministic project defaults."""

from pathlib import Path

from src.config import AppConfig


def test_gemini_model_default_and_template_are_consistent(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert AppConfig().gemini_model == "gemini-3.6-flash"
    for name in (".env.development.example", ".env.deployment.example"):
        template = Path(name).read_text(encoding="utf-8")
        assert "GEMINI_MODEL=gemini-3.6-flash" in template
    workflow = Path(".github/workflows/ingest-production.yml").read_text(encoding="utf-8")
    assert "GEMINI_MODEL: gemini-3.6-flash" in workflow


def test_deployment_template_fits_free_embedding_quota_profile() -> None:
    template = Path(".env.deployment.example").read_text(encoding="utf-8")

    assert "CHUNK_SIZE=3000" in template
    assert "CHUNK_OVERLAP=300" in template
    assert "ONLINE_EMBEDDING_RPM=90" in template
    assert "ACTIVE_INDEX_NAMESPACE=phase2_SELECTED_AFTER_SIGNOFF" in template
    assert "RETRIEVAL_PROFILE=large" in template
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


def test_auto_generation_accepts_a_configured_groq_fallback() -> None:
    config = AppConfig(
        generation_provider="auto",
        generation_primary_provider="gemini",
        generation_fallback_provider="groq",
        gemini_api_key="",
        groq_api_key="configured",
    )

    config.validate()

    assert config.generation_configured is True
    assert config.configured_generation_provider_label == "Gemini \u2192 Groq \u2192 Evidence excerpts (automatic)"


# ---------------------------------------------------------------------------
# Critical Issue 2 — dimension ceiling raised from 2000 to 3072
# ---------------------------------------------------------------------------

def test_embedding_dimension_accepts_3072() -> None:
    """3072 is the maximum Gemini Embedding 2 output dimension and must be valid."""
    config = AppConfig(embedding_dimension=3072)
    try:
        config.validate()
    except ValueError as exc:
        if "EMBEDDING_DIMENSION" in str(exc):
            raise AssertionError(
                f"EMBEDDING_DIMENSION=3072 was rejected but must be accepted: {exc}"
            ) from exc
        # Other validation errors (missing API keys etc.) are acceptable in unit tests.


def test_embedding_dimension_rejects_above_3072() -> None:
    """Dimensions above 3072 exceed Gemini Embedding 2 capability and must be rejected."""
    config = AppConfig(embedding_dimension=3073)
    try:
        config.validate()
    except ValueError as exc:
        assert "EMBEDDING_DIMENSION" in str(exc), (
            f"Expected EMBEDDING_DIMENSION error, got: {exc}"
        )
    else:
        raise AssertionError("embedding_dimension=3073 should have been rejected")


def test_embedding_dimension_accepts_previously_blocked_2001() -> None:
    """2001-d was blocked by the old 2000 ceiling but must now be accepted."""
    config = AppConfig(embedding_dimension=2001)
    try:
        config.validate()
    except ValueError as exc:
        if "EMBEDDING_DIMENSION" in str(exc):
            raise AssertionError(
                f"EMBEDDING_DIMENSION=2001 was rejected but should be allowed: {exc}"
            ) from exc


def test_embedding_dimension_zero_is_still_rejected() -> None:
    """Zero dimensions must remain invalid regardless of the ceiling change."""
    config = AppConfig(embedding_dimension=0)
    try:
        config.validate()
    except ValueError as exc:
        assert "EMBEDDING_DIMENSION" in str(exc)
    else:
        raise AssertionError("embedding_dimension=0 should have been rejected")


def test_appconfig_has_no_duplicate_field_names() -> None:
    """AppConfig must not have duplicate dataclass field declarations.

    Duplicate fields silently shadow each other in Python dataclasses;
    the second one wins and the first env-var binding is lost.
    """
    import dataclasses
    field_names = [f.name for f in dataclasses.fields(AppConfig)]
    seen: set[str] = set()
    duplicates = []
    for name in field_names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    assert not duplicates, (
        f"AppConfig has duplicate field declarations: {duplicates}. "
        "Remove the duplicate; the second declaration silently shadows the first."
    )
