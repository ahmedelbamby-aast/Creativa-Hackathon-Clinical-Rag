"""Operational source catalog and index-manifest tests."""

from pathlib import Path

import pytest

from src.index_manifests import (
    build_index_manifest,
    index_manifest_hash,
    load_index_manifest,
    write_index_manifest,
    manifest_matches_runtime,
)
from src.source_catalog import (
    enrich_chunk_records,
    load_source_catalog,
    require_catalog_documents,
    validate_source_checksums,
)


def _source_json(*, url: str = "https://example.test/guide", checksum: str = "a" * 64) -> str:
    return (
        '{"sources": [{"source_id": "guide", "document_name": "guide.pdf", '
        f'"source_url": "{url}", "publisher": "Example", "publication_date": "2025", '
        '"license_note": "Evaluation use", "reuse_status": "evaluation-only", '
        f'"checksum": "{checksum}"' + "}]}"
    )


def test_catalog_loads_and_enriches_matching_document(tmp_path: Path) -> None:
    catalog_path = tmp_path / "sources.json"
    catalog_path.write_text(_source_json(), encoding="utf-8")
    catalog = load_source_catalog(catalog_path)
    records = enrich_chunk_records(
        [{"document_name": "guide.pdf", "text": "content"}], catalog
    )

    assert records[0]["source_id"] == "guide"
    assert records[0]["source_url"] == "https://example.test/guide"


def test_catalog_rejects_enabled_http_or_missing_entries(tmp_path: Path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(_source_json(url="http://example.test/guide"), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        load_source_catalog(path)

    with pytest.raises(ValueError, match="missing"):
        require_catalog_documents({"missing.pdf"}, {})


def test_index_manifest_round_trip_is_stable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.index_manifests.source_catalog_hash", lambda: "source-hash")
    corpus = tmp_path / "guide.txt"
    corpus.write_text("diabetes evidence", encoding="utf-8")

    manifest = build_index_manifest("Phase2 Local", [corpus], token_count=2)
    written = write_index_manifest(manifest, tmp_path / "manifests")
    loaded = load_index_manifest("phase2_local", tmp_path / "manifests")

    assert written.exists()
    assert loaded == manifest
    assert index_manifest_hash(loaded) == index_manifest_hash(manifest)


def test_checksum_validation_and_manifest_catalog_freshness(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "guide.pdf"
    corpus.write_text("verified source", encoding="utf-8")
    checksum = __import__("hashlib").sha256(corpus.read_bytes()).hexdigest()
    catalog_path = tmp_path / "sources.json"
    catalog_path.write_text(_source_json(checksum=checksum), encoding="utf-8")
    catalog = load_source_catalog(catalog_path)
    validate_source_checksums(catalog, tmp_path)
    corpus.write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_source_checksums(catalog, tmp_path)

    monkeypatch.setattr("src.index_manifests.source_catalog_hash", lambda: "catalog-a")
    manifest = build_index_manifest("phase2_local", [corpus], token_count=2)
    monkeypatch.setattr("src.index_manifests.source_catalog_hash", lambda: "catalog-b")
    assert manifest_matches_runtime(manifest, "phase2_local") is False
