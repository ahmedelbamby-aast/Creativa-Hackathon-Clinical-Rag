"""Full ingestion pipeline — parse → chunk → embed → store.

Orchestrates all ingestion stages for one or more documents:
1. Parse with PyMuPDF (structure-aware)
2. Propagate section titles
3. Chunk with SmartChunker adapter
4. Generate embeddings in batch
5. Upsert into PostgreSQL/pgvector

Designed to be run once (or re-run to update the knowledge base).
"""

import logging
import time
from pathlib import Path
from typing import Optional

from src.config import config, ALL_CATEGORIES
from src.ingestion.parser import parse_document, propagate_section_titles
from src.ingestion.chunker_adapter import chunk_elements
from src.ingestion.core.language_detector import detect_document_language
from src.embeddings import embedder
from src.embedding_quota import EmbeddingQuotaExceeded, embedding_quota
from src.index_manifests import corpus_hash
from src.source_catalog import enrich_chunk_records
from src.vector_store import vector_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-document ingestion
# ---------------------------------------------------------------------------

def ingest_document(
    file_path: str | Path,
    force: bool = False,
) -> dict:
    """Parse, chunk, embed and store a single document.

    Args:
        file_path: Path to the document file.
        force: If True, re-ingest even if chunks already exist.

    Returns:
        Stats dict with keys: file_name, pages, elements, chunks, categories, elapsed_s, error.
    """
    file_path = Path(file_path)
    file_name = file_path.name
    start = time.perf_counter()
    stats: dict = {
        "file_name": file_name,
        "pages": 0,
        "elements": 0,
        "chunks": 0,
        "token_count": 0,
        "categories": {},
        "elapsed_s": 0.0,
        "skipped": False,
        "paused": False,
        "error": None,
    }

    try:
        logger.info("━━━ Ingesting: %s", file_name)

        document_exists = vector_store.has_document(file_name)
        if document_exists and not force:
            logger.info("  Skipped: document is already in the vector store")
            stats["skipped"] = True
            stats["elapsed_s"] = round(time.perf_counter() - start, 2)
            return stats

        # Step 1: Parse
        elements = parse_document(str(file_path))
        if not elements:
            raise ValueError(f"No content extracted from {file_name}")

        # Count unique pages
        stats["pages"] = len({e["page_number"] for e in elements})
        stats["elements"] = len(elements)
        logger.info("  [1/4] Parsed %d elements across %d pages", stats["elements"], stats["pages"])

        # Forward-fill section context
        elements = propagate_section_titles(elements)

        # Step 2: Detect document language
        full_text = " ".join(
            e.get("content", "")[:200] for e in elements[:20]
        )
        doc_language = detect_document_language(full_text)
        logger.info("  [2/4] Detected language: %s", doc_language)

        # Step 3: Chunk
        chunk_records = chunk_elements(
            elements,
            document_language=doc_language,
        )
        if not chunk_records:
            raise ValueError(f"No valid chunks produced from {file_name}")

        enrich_chunk_records(chunk_records)

        stats["chunks"] = len(chunk_records)
        stats["token_count"] = sum(
            record.get("word_count", len(record.get("text", "").split()))
            for record in chunk_records
        )
        cat_counts = {}
        for r in chunk_records:
            cat = r.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        stats["categories"] = cat_counts
        logger.info(
            "  [3/4] Created %d chunks: %s",
            stats["chunks"],
            ", ".join(f"{k}={v}" for k, v in cat_counts.items()),
        )

        # Step 4: Embed
        texts = [r["text"] for r in chunk_records]
        logger.info("  [4/4] Embedding %d chunks...", len(texts))
        embeddings = embedder.embed_batch(texts, show_progress=False)

        # Step 5: Store
        if document_exists:
            vector_store.delete_document(file_name)
        added = vector_store.add_chunks(chunk_records, embeddings)
        logger.info(
            "  Stored -> %s",
            ", ".join(f"{k}:{v}" for k, v in added.items()),
        )

    except Exception as e:
        logger.error("  FAILED: %s → %s", file_name, e)
        stats["error"] = str(e)
        stats["paused"] = isinstance(e, EmbeddingQuotaExceeded) and e.resumable

    stats["elapsed_s"] = round(time.perf_counter() - start, 2)
    return stats


# ---------------------------------------------------------------------------
# Batch ingestion
# ---------------------------------------------------------------------------

def ingest_directory(
    data_dir: str | Path | None = None,
    extensions: tuple[str, ...] = (".pdf", ".docx", ".txt"),
    force: bool = False,
) -> list[dict]:
    """Ingest all documents in a directory.

    Args:
        data_dir: Directory containing documents. Defaults to config.data_dir.
        extensions: File extensions to process.
        force: Re-ingest even if already stored.

    Returns:
        List of per-document stat dicts.
    """
    data_dir = Path(data_dir or config.data_dir)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        return []

    files = sorted(
        f for f in data_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in extensions
    )

    if not files:
        logger.warning("No documents found in %s", data_dir)
        return []

    logger.info("Found %d documents in %s", len(files), data_dir)
    all_stats: list[dict] = []
    run_id = ""
    checkpoint: dict[str, dict] = {}
    if config.embedding_provider == "gemini" and embedding_quota.enabled:
        run_id = embedding_quota.repository.start_run(
            namespace=config.resolved_embedding_namespace,
            table_family=config.embedding_table_family,
            dimension=config.embedding_dimension,
            model=config.online_embedding_model,
            corpus_hash=corpus_hash(files),
            total_documents=len(files),
            checkpoint=checkpoint,
        )

    for i, file_path in enumerate(files, 1):
        logger.info("\n[%d/%d] %s", i, len(files), file_path.name)
        if run_id:
            embedding_quota.repository.checkpoint_run(run_id, current_document=file_path.name)
            with embedding_quota.run_scope(run_id):
                stats = ingest_document(file_path, force=force)
        else:
            stats = ingest_document(file_path, force=force)
        all_stats.append(stats)
        checkpoint[file_path.name] = {
            "status": "paused" if stats.get("paused") else (
                "failed" if stats.get("error") else ("skipped" if stats.get("skipped") else "completed")
            ),
            "chunks": stats.get("chunks", 0),
        }
        if run_id:
            completed = sum(item["status"] in {"completed", "skipped"} for item in checkpoint.values())
            status = "paused_quota" if stats.get("paused") else ("failed" if stats.get("error") else "running")
            embedding_quota.repository.checkpoint_run(
                run_id, status=status, completed_documents=completed,
                current_document=file_path.name, checkpoint=checkpoint,
                last_error=stats.get("error") or "",
            )
        if stats.get("paused"):
            logger.warning("Daily Gemini embedding budget reached; run %s is resumable", run_id)
            break

    if run_id and all_stats and not any(item.get("paused") for item in all_stats):
        failures = [item for item in all_stats if item.get("error")]
        embedding_quota.repository.checkpoint_run(
            run_id,
            status="failed" if failures else "completed",
            completed_documents=sum(
                item["status"] in {"completed", "skipped"} for item in checkpoint.values()
            ),
            current_document="",
            checkpoint=checkpoint,
            last_error=failures[-1]["error"] if failures else "",
        )

    return all_stats


def print_ingestion_summary(all_stats: list[dict]) -> None:
    """Print a formatted summary of ingestion results."""
    total_docs = len(all_stats)
    successful = [s for s in all_stats if not s["error"] and not s.get("skipped")]
    skipped = [s for s in all_stats if s.get("skipped")]
    failed = [s for s in all_stats if s["error"]]
    total_pages = sum(s["pages"] for s in successful)
    total_chunks = sum(s["chunks"] for s in successful)
    total_tokens = sum(s.get("token_count", 0) for s in successful)

    # Aggregate category counts
    cat_totals: dict[str, int] = {}
    for s in successful:
        for cat, count in s.get("categories", {}).items():
            cat_totals[cat] = cat_totals.get(cat, 0) + count

    col_stats = vector_store.collection_stats()

    print("\n" + "=" * 60)
    print("  Ingestion Complete")
    print("=" * 60)
    print(f"  Documents processed : {total_docs}")
    print(f"  Successful          : {len(successful)}")
    print(f"  Skipped             : {len(skipped)}")
    print(f"  Failed              : {len(failed)}")
    print(f"  Pages processed     : {total_pages}")
    print(f"  Chunks created      : {total_chunks}")
    print(f"  Reference tokens    : {total_tokens}")
    print()
    print("  Chunk distribution (by category label):")
    for cat, count in sorted(cat_totals.items()):
        print(f"    {cat:<15} {count}")
    print()
    print("  PostgreSQL namespace counts:")
    for cat, size in col_stats.items():
        print(f"    {cat:<15} {size}")
    print("=" * 60)

    if failed:
        print("\n  Failed documents:")
        for s in failed:
            print(f"    X {s['file_name']}: {s['error']}")
        print()
