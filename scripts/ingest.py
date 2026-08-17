#!/usr/bin/env python
"""Ingestion script — parse, chunk, embed and store all diabetes documents.

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

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.ingestion.pipeline import (
    ingest_document,
    ingest_directory,
    print_ingestion_summary,
)
from src.vector_store import vector_store


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
    logging.getLogger("chromadb").setLevel(logging.WARNING)
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
        help="⚠️  Delete and recreate all ChromaDB collections before ingesting",
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
    args = parser.parse_args()

    setup_logging(args.verbose)
    config.validate()

    print("\n" + "=" * 60)
    print("  Diabetes RAG - Document Ingestion")
    print("=" * 60)
    print(f"  ChromaDB path    : {config.chroma_db_dir}")
    print(f"  Embedding model  : {config.embedding_model}")
    print(f"  Chunk size       : {config.chunk_size} chars")
    print(f"  Chunk overlap    : {config.chunk_overlap} chars")
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
        print("  ⚠️  Resetting all ChromaDB collections...")
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
        return

    # ── Directory ingestion ────────────────────────────────────────────
    data_dir = args.data_dir or config.data_dir
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"  ✗ Data directory not found: {data_dir}")
        print(f"  Set DATA_DIR in .env or pass --data-dir")
        sys.exit(1)

    print(f"  Data directory : {data_dir}")
    print()

    all_stats = ingest_directory(data_dir, force=args.force)
    print_ingestion_summary(all_stats)


if __name__ == "__main__":
    main()
