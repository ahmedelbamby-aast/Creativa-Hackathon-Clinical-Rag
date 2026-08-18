"""Operational source catalog and index-manifest tests."""

from pathlib import Path

import pytest

from src.index_manifests import (
    build_index_manifest,
    index_manifest_hash,
    load_index_manifest,
    write_index_manifest,
)
from src.source_catalog import (
    enrich_chunk_records,
    load_source_catalog,
    require_catalog_documents,
)


def test_catalog_loads_and_enriches_matching_document(tmp_path: Path) -> None:
    catalog_path = tmp_path / "sources.json"
    catalog_path.write_text(
        '{"sources": [{"source_id": "guide", "document_name": "guide.pdf", '
        '"source_url": "https://example.test/guide"}]}',
        encoding="utf-8",
    )
    catalog = load_source_catalog(catalog_path)
    records = enrich_chunk_records(
        [{"document_name": "guide.pdf", "text": "content"}], catalog
    )

    assert records[0]["source_id"] == "guide"
    assert records[0]["source_url"] == "https://example.test/guide"


def test_catalog_rejects_enabled_http_or_missing_entries(tmp_path: Path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(
        '{"sources": [{"source_id": "guide", "document_name": "guide.pdf", '
        '"source_url": "http://example.test/guide"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="https"):
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
