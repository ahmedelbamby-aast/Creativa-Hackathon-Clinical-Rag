#!/usr/bin/env python
"""Read-only proof that every local PDF is retained in the active namespace."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg

from src.config import config
from src.vector_store import vector_store


def main() -> None:
    local = sorted(path.name for path in config.data_dir.glob("*.pdf"))
    with psycopg.connect(config.database_url) as connection:
        rows = connection.execute(
            "SELECT document_name, count(*) FROM rag_chunks WHERE namespace = %s GROUP BY document_name",
            (vector_store.namespace,),
        ).fetchall()
    indexed = {name: count for name, count in rows}
    missing = [name for name in local if not indexed.get(name)]
    print(json.dumps({"namespace": vector_store.namespace, "local_pdf_count": len(local), "indexed_pdf_count": len(indexed), "missing": missing, "documents": indexed}, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
