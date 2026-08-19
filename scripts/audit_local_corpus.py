#!/usr/bin/env python
"""Read-only proof that every local PDF is retained in the active namespace."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib

from src.config import config
from src.index_manifests import load_index_manifest
from src.vector_store import vector_store


def main() -> None:
    paths = sorted(config.data_dir.glob("*.pdf"), key=lambda path: path.name.lower())
    local_checksums = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    audit = vector_store.namespace_audit()
    indexed = audit["documents"]
    missing = [name for name in local_checksums if name not in indexed]
    manifest = load_index_manifest(vector_store.namespace)
    checksum_mismatches = []
    count_mismatches = []
    if manifest:
        checksum_mismatches = [
            name for name, checksum in local_checksums.items()
            if manifest.document_checksums.get(name) != checksum
        ]
        count_mismatches = [
            name for name, expected in manifest.document_chunk_counts.items()
            if indexed.get(name, {}).get("chunk_count") != expected
        ]
    result = {
        **audit,
        "local_pdf_count": len(paths),
        "missing": missing,
        "manifest_present": manifest is not None,
        "checksum_mismatches": checksum_mismatches,
        "chunk_count_mismatches": count_mismatches,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if missing or audit["invalid_vector_count"] or checksum_mismatches or count_mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
