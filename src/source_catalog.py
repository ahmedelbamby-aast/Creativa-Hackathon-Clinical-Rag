"""Operational provenance catalog used by retrieval and benchmark ingestion."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.config import config
from src.retrieval_contracts import SourceManifestEntry


DEFAULT_CATALOG_PATH = config.project_root / "data" / "retrieval_sources.json"


def load_source_catalog(path: Path = DEFAULT_CATALOG_PATH) -> dict[str, SourceManifestEntry]:
    """Load enabled source records keyed by their local document filename."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("sources", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError("retrieval source catalog must contain a sources list")

    catalog: dict[str, SourceManifestEntry] = {}
    source_ids: set[str] = set()
    for item in entries:
        entry = SourceManifestEntry(
            source_id=str(item["source_id"]).strip(),
            document_name=str(item["document_name"]).strip(),
            source_url=str(item["source_url"]).strip(),
            publisher=str(item.get("publisher", "")).strip(),
            publication_date=str(item.get("publication_date", "")).strip(),
            license_note=str(item.get("license_note", "")).strip(),
            reuse_status=str(item.get("reuse_status", "")).strip(),
            checksum=str(item.get("checksum", "")).strip().lower(),
            enabled=bool(item.get("enabled", True)),
        )
        if not entry.source_id or not entry.document_name:
            raise ValueError("every source catalog entry requires source_id and document_name")
        if entry.enabled and (
            not entry.source_url.startswith("https://")
            or not entry.publisher
            or not entry.publication_date
            or not entry.license_note
            or not entry.reuse_status
            or not re.fullmatch(r"[0-9a-f]{64}", entry.checksum)
        ):
            raise ValueError(
                f"enabled source {entry.source_id} requires complete provenance and SHA-256 checksum"
            )
        if entry.document_name in catalog or entry.source_id in source_ids:
            raise ValueError(f"duplicate retrieval source catalog entry: {entry.document_name}")
        catalog[entry.document_name] = entry
        source_ids.add(entry.source_id)
    return catalog


def source_catalog_hash(path: Path = DEFAULT_CATALOG_PATH) -> str:
    """Return the content hash used to bind an index manifest to provenance."""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enrich_chunk_records(
    records: list[dict],
    catalog: dict[str, SourceManifestEntry] | None = None,
) -> list[dict]:
    """Add immutable source identity fields without changing chunk text or IDs."""
    catalog = catalog if catalog is not None else load_source_catalog()
    for record in records:
        source = catalog.get(record.get("document_name", ""))
        record["source_id"] = source.source_id if source and source.enabled else ""
        record["source_url"] = source.source_url if source and source.enabled else ""
        record["publisher"] = source.publisher if source and source.enabled else ""
        record["publication_date"] = source.publication_date if source and source.enabled else ""
        record["source_checksum"] = source.checksum if source and source.enabled else ""
    return records


def require_catalog_documents(document_names: set[str], catalog: dict[str, SourceManifestEntry]) -> None:
    """Reject experiment targets that have no enabled operational provenance."""
    missing = sorted(
        name
        for name in document_names
        if name not in catalog or not catalog[name].enabled or not catalog[name].source_url
    )
    if missing:
        raise ValueError("missing enabled source catalog entries: " + ", ".join(missing))


def validate_source_checksums(
    catalog: dict[str, SourceManifestEntry],
    data_dir: Path = config.data_dir,
) -> None:
    """Fail closed if local source bytes do not match their cataloged SHA-256."""
    mismatches = []
    for entry in catalog.values():
        if not entry.enabled:
            continue
        path = data_dir / entry.document_name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
        if actual != entry.checksum:
            mismatches.append(entry.document_name)
    if mismatches:
        raise ValueError("source checksum mismatch: " + ", ".join(sorted(mismatches)))
