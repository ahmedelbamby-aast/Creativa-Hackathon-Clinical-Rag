#!/usr/bin/env python
"""Ingestion script — parse, chunk, embed and store diabetes documents.

Usage
-----
    # Ingest all documents from the configured DATA_DIR
    python scripts/ingest.py

    # Ingest a specific directory
    python scripts/ingest.py --data-dir path/to/pdfs

    # Ingest a single file
    python scripts/ingest.py --file path/to/document.pdf

    # Force re-ingestion (overwrite existing chunks)
    python scripts/ingest.py --force

    # Reset all collections before ingesting
    python scripts/ingest.py --reset

    # Show collection stats without ingesting
    python scripts/ingest.py --stats
"""

import argparse
import logging
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.embeddings import embedder
from src.ingestion.pipeline import (
    ingest_document,
    ingest_directory,
    print_ingestion_summary,
)
from src.vector_store import vector_store
from src.index_manifests import build_index_manifest, write_index_manifest
from src.source_catalog import load_source_catalog, validate_source_checksums


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest diabetes documents into the RAG vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=f"Directory containing documents (default: {config.data_dir})",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Ingest a single file instead of a directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest documents even if already stored",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="⚠️  Delete all chunks in the active PostgreSQL namespace",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print collection stats and exit (no ingestion)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--write-index-manifest",
        action="store_true",
        help="Write a reproducibility manifest after successful ingestion",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    config.validate()
    validate_source_checksums(load_source_catalog())

    print("\n" + "=" * 60)
    print("  Diabetes RAG - Document Ingestion")
    print("=" * 60)
    print(f"  Database         : PostgreSQL/pgvector")
    print(f"  Namespace        : {config.resolved_embedding_namespace}")
    print(f"  Embedding model  : {embedder.model_name}")
    profile_size, profile_overlap = config.selected_chunk_profile
    print(f"  Chunk profile    : {config.retrieval_profile}")
    print(f"  Chunk size       : {profile_size} chars")
    print(f"  Chunk overlap    : {profile_overlap} chars")
    print()

    # ── Stats only ─────────────────────────────────────────────────────
    if args.stats:
        stats = vector_store.collection_stats()
        print("  Collection stats:")
        for cat, count in stats.items():
            print(f"    {cat:<20} {count:>6} chunks")
        print()
        return

    # ── Optional reset ─────────────────────────────────────────────────
    if args.reset:
        print("  ⚠️  Resetting the active PostgreSQL namespace...")
        vector_store.reset_all()
        print("  Collections cleared.\n")

    # ── Single file ────────────────────────────────────────────────────
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"  ✗ File not found: {file_path}")
            sys.exit(1)
        print(f"  Ingesting single file: {file_path.name}\n")
        stats = ingest_document(file_path, force=args.force)
        print_ingestion_summary([stats])
        if args.write_index_manifest and not stats["error"]:
            manifest = build_index_manifest(
                config.resolved_embedding_namespace,
                [file_path],
                stats.get("token_count", 0),
            )
            print(f"  Index manifest      : {write_index_manifest(manifest)}")
        return

    # ── Directory ingestion ────────────────────────────────────────────
    data_dir = args.data_dir or config.data_dir
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"  ✗ Data directory not found: {data_dir}")
        print(f"  Set DATA_DIR in the selected environment or pass --data-dir")
        sys.exit(1)

    print(f"  Data directory : {data_dir}")
    print()

    all_stats = ingest_directory(data_dir, force=args.force)
    print_ingestion_summary(all_stats)
    if args.write_index_manifest and not any(item["error"] for item in all_stats):
        corpus_paths = sorted(
            path for path in data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".txt"}
        )
        manifest = build_index_manifest(
            config.resolved_embedding_namespace,
            corpus_paths,
            sum(item.get("token_count", 0) for item in all_stats),
        )
        print(f"  Index manifest      : {write_index_manifest(manifest)}")


if __name__ == "__main__":
    main()
