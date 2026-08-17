#!/usr/bin/env python
"""Prepare the active embedding provider and PostgreSQL/pgvector namespace."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.embeddings import embedder
from src.vector_store import vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-online",
        action="store_true",
        help="Send one embedding request when EMBEDDING_PROVIDER=gemini",
    )
    args = parser.parse_args()

    config.validate()
    vector_store.ensure_schema()
    versions = vector_store.healthcheck()
    print(
        f"PostgreSQL {versions['postgres']} with pgvector {versions['pgvector']} ready"
    )
    print(f"Namespace: {vector_store.namespace}")
    print(f"Embedding provider: {embedder.provider}")
    print(f"Embedding model: {embedder.model_name}")

    if embedder.provider == "local":
        print(f"Local model ready ({embedder.dimension} dimensions)")
    elif args.verify_online:
        vector = embedder.embed_query("Diabetes reference retrieval readiness check")
        print(f"Gemini embedding API ready ({len(vector)} dimensions)")
    else:
        print(
            "Gemini provider configured. Use --verify-online to make a live API request."
        )

    stats = vector_store.collection_stats()
    for category, count in stats.items():
        print(f"  {category:<12} {count} chunks")


if __name__ == "__main__":
    main()
