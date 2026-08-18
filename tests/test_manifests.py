"""Manifest validation tests — Phase 1.

Covers all 13 required cases for SourceManifestEntry, IndexManifest,
and the new AppConfig fields (ACTIVE_INDEX_NAMESPACE, RETRIEVAL_PROFILE,
TOP_K).

Test style follows the project conventions in test_system_configuration.py
and test_core.py: plain pytest functions, monkeypatch for env-vars,
pytest.raises for expected errors.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.manifests import IndexManifest, SourceManifestEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_SHA256 = "a" * 64          # 64 lowercase hex chars
_VALID_URL = "https://example.org/source"
_VALID_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _valid_source(**overrides) -> SourceManifestEntry:
    """Return a fully-valid SourceManifestEntry, with optional field overrides."""
    defaults = dict(
        source_id="who-2024",
        title="WHO Diabetes Report 2024",
        publisher="World Health Organization",
        source_url=_VALID_URL,
        publication_date="2024-01-15",
        checksum_sha256=_VALID_SHA256,
        enabled=True,
    )
    defaults.update(overrides)
    return SourceManifestEntry(**defaults)


def _valid_index(**overrides) -> IndexManifest:
    """Return a fully-valid IndexManifest, with optional field overrides."""
    defaults = dict(
        namespace="local_384",
        corpus_hash="b" * 64,
        chunk_profile="default",
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        embedding_dimension=384,
        created_at=_VALID_DATE,
    )
    defaults.update(overrides)
    return IndexManifest(**defaults)


# ---------------------------------------------------------------------------
# SourceManifestEntry tests
# ---------------------------------------------------------------------------

def test_valid_source_manifest_entry_is_accepted():
    """Test case 1: a fully-populated SourceManifestEntry with valid data."""
    entry = _valid_source(version="1.0", licensing_note="CC-BY-4.0")
    assert entry.source_id == "who-2024"
    assert entry.publisher == "World Health Organization"
    assert entry.enabled is True
    assert entry.version == "1.0"
    assert entry.licensing_note == "CC-BY-4.0"


@pytest.mark.parametrize("missing_field", [
    "source_id",
    "title",
    "publisher",
    "source_url",
    "publication_date",
])
def test_missing_required_source_metadata_is_rejected(missing_field):
    """Test case 2: any required field set to empty string raises ValueError."""
    with pytest.raises(ValueError, match=missing_field):
        _valid_source(**{missing_field: ""})


@pytest.mark.parametrize("bad_checksum", [
    "abc123",                   # too short
    "g" * 64,                   # invalid hex char
    "A" * 64,                   # uppercase not accepted
    "",                         # empty
    "a" * 63,                   # one char too short
    "a" * 65,                   # one char too long
])
def test_invalid_sha256_checksum_is_rejected(bad_checksum):
    """Test case 3: non-SHA-256-formatted checksum is rejected."""
    with pytest.raises(ValueError, match="checksum_sha256"):
        _valid_source(checksum_sha256=bad_checksum)


# ---------------------------------------------------------------------------
# IndexManifest tests
# ---------------------------------------------------------------------------

def test_valid_index_manifest_is_accepted():
    """Test case 4: a fully-populated IndexManifest with valid data."""
    manifest = _valid_index()
    assert manifest.namespace == "local_384"
    assert manifest.embedding_dimension == 384
    assert manifest.chunk_profile == "default"
    assert isinstance(manifest.created_at, datetime)


def test_empty_namespace_is_rejected():
    """Test case 5: namespace cannot be empty."""
    with pytest.raises(ValueError, match="namespace"):
        _valid_index(namespace="")


def test_empty_corpus_hash_is_rejected():
    """Test case 6: corpus_hash cannot be empty."""
    with pytest.raises(ValueError, match="corpus_hash"):
        _valid_index(corpus_hash="")


def test_empty_embedding_model_is_rejected():
    """Test case 7: embedding_model cannot be empty."""
    with pytest.raises(ValueError, match="embedding_model"):
        _valid_index(embedding_model="")


@pytest.mark.parametrize("bad_dim", [0, -1, -384])
def test_non_positive_embedding_dimension_is_rejected(bad_dim):
    """Test case 8: embedding_dimension must be a positive integer."""
    with pytest.raises(ValueError, match="embedding_dimension"):
        _valid_index(embedding_dimension=bad_dim)


def test_empty_chunk_profile_is_rejected():
    """Test case 9: chunk_profile cannot be empty."""
    with pytest.raises(ValueError, match="chunk_profile"):
        _valid_index(chunk_profile="")


# ---------------------------------------------------------------------------
# AppConfig: new fields
# ---------------------------------------------------------------------------

def test_config_accepts_valid_active_index_namespace(monkeypatch):
    """Test case 10: ACTIVE_INDEX_NAMESPACE is read from the environment."""
    monkeypatch.setenv("ACTIVE_INDEX_NAMESPACE", "gemini_768")
    from importlib import reload
    import src.config as cfg_module
    fresh = cfg_module.AppConfig()
    assert fresh.active_index_namespace == "gemini_768"


def test_config_accepts_valid_retrieval_profile(monkeypatch):
    """Test case 11: RETRIEVAL_PROFILE is read from the environment."""
    monkeypatch.setenv("RETRIEVAL_PROFILE", "high_precision")
    from importlib import reload
    import src.config as cfg_module
    fresh = cfg_module.AppConfig()
    assert fresh.retrieval_profile == "high_precision"


def test_config_accepts_valid_top_k(monkeypatch):
    """Test case 12: a positive TOP_K passes validate() without error."""
    monkeypatch.setenv("TOP_K", "10")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "384")
    monkeypatch.setenv("CHUNK_SIZE", "2000")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    import src.config as cfg_module
    fresh = cfg_module.AppConfig()
    assert fresh.top_k == 10
    # validate() should not raise for top_k=10
    # (data_dir warning is fine; we only care no ValueError is raised for top_k)
    try:
        fresh.validate()
    except ValueError as exc:
        assert "TOP_K" not in str(exc), f"Unexpected TOP_K error: {exc}"


def test_config_rejects_invalid_top_k(monkeypatch):
    """Test case 13: TOP_K <= 0 raises ValueError in validate()."""
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "384")
    monkeypatch.setenv("CHUNK_SIZE", "2000")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    import src.config as cfg_module
    fresh = cfg_module.AppConfig()
    assert fresh.top_k == 0
    with pytest.raises(ValueError, match="TOP_K"):
        fresh.validate()
