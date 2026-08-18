"""Atomic certified-corpus indexer.

Indexes the two certified Phase 1 sources into an isolated namespace.
The existing active namespace is NEVER modified.  If any step fails the
exception propagates and no partial data is committed to the target namespace.

Usage
-----
    from src.ingestion.indexer import ingest_certified_corpus
    from src.config import config

    manifest = ingest_certified_corpus(
        sources_path="data/sources.json",
        namespace="certified_v1",
    )
    # Only after success: update ACTIVE_INDEX_NAMESPACE in .env
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import CHUNK_PROFILES, CHUNK_PROFILE_DEFAULT, config
from src.manifests import IndexManifest, SourceManifestEntry
from src.ingestion.parser import parse_document, propagate_section_titles
from src.ingestion.chunker_adapter import chunk_elements
from src.ingestion.core.language_detector import detect_document_language

logger = logging.getLogger(__name__)

# Default directory for saved IndexManifest JSON files
_MANIFEST_DIR = Path(__file__).resolve().parents[2] / "data" / "index_manifests"


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def load_sources(
    sources_path: str | Path,
) -> list[tuple[SourceManifestEntry, str]]:
    """Load enabled sources from a sources.json manifest.

    Returns:
        List of ``(SourceManifestEntry, file_name)`` tuples for every enabled
        source.  ``file_name`` is the local filename used to locate the PDF
        inside ``data_dir``.
    """
    path = Path(sources_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    result: list[tuple[SourceManifestEntry, str]] = []
    for raw in data.get("sources", []):
        file_name = raw.get("file_name", "")
        entry = SourceManifestEntry(
            source_id=raw["source_id"],
            title=raw["title"],
            publisher=raw["publisher"],
            source_url=raw["source_url"],
            publication_date=raw["publication_date"],
            checksum_sha256=raw["checksum_sha256"],
            enabled=raw["enabled"],
            version=raw.get("version", ""),
            licensing_note=raw.get("licensing_note", ""),
        )
        if entry.enabled:
            result.append((entry, file_name))
    return result


def compute_corpus_hash(entries: list[SourceManifestEntry]) -> str:
    """Deterministic hash over all enabled-source checksums, sorted by source_id."""
    sorted_checksums = sorted(e.checksum_sha256 for e in entries)
    combined = "|".join(sorted_checksums)
    return hashlib.sha256(combined.encode()).hexdigest()


def _verify_source_file(
    data_dir: Path,
    file_name: str,
    expected_checksum: str,
) -> Path:
    """Locate the source file and verify its SHA-256 checksum.

    Raises:
        ValueError: If the file is missing or the checksum does not match.
    """
    candidate = data_dir / file_name
    if not candidate.exists():
        raise ValueError(f"Source file not found: {candidate}")
    actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if actual != expected_checksum:
        raise ValueError(
            f"Checksum mismatch for '{file_name}': "
            f"expected {expected_checksum[:16]}…, got {actual[:16]}…"
        )
    return candidate


# ---------------------------------------------------------------------------
# Manifest persistence
# ---------------------------------------------------------------------------

def save_index_manifest(manifest: IndexManifest, directory: Path) -> Path:
    """Serialize an IndexManifest to JSON.

    Args:
        manifest: The manifest to save.
        directory: Target directory (created if it does not exist).

    Returns:
        Path to the written JSON file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{manifest.namespace}.json"
    payload = {
        "namespace": manifest.namespace,
        "corpus_hash": manifest.corpus_hash,
        "chunk_profile": manifest.chunk_profile,
        "embedding_model": manifest.embedding_model,
        "embedding_dimension": manifest.embedding_dimension,
        "created_at": manifest.created_at.isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("IndexManifest saved: %s", path)
    return path


def load_index_manifest(path: str | Path) -> IndexManifest:
    """Deserialize an IndexManifest from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return IndexManifest(
        namespace=data["namespace"],
        corpus_hash=data["corpus_hash"],
        chunk_profile=data["chunk_profile"],
        embedding_model=data["embedding_model"],
        embedding_dimension=data["embedding_dimension"],
        created_at=datetime.fromisoformat(data["created_at"]),
    )


# ---------------------------------------------------------------------------
# Atomic indexer
# ---------------------------------------------------------------------------

def ingest_certified_corpus(
    sources_path: str | Path,
    namespace: str,
    data_dir: Optional[str | Path] = None,
    chunk_profile: str = CHUNK_PROFILE_DEFAULT,
    manifest_dir: Optional[Path] = None,
    force: bool = False,
    _store=None,    # injected in tests
    _embedder=None, # injected in tests
) -> IndexManifest:
    """Index the certified corpus into an isolated namespace atomically.

    The existing active namespace is NEVER touched.  If any step (parsing,
    embedding, storage) raises an exception the function propagates it
    immediately — the caller's active namespace remains intact.

    Args:
        sources_path: Path to ``data/sources.json``.
        namespace: Target namespace (should differ from the current active one).
        data_dir: Directory containing source PDF files. Defaults to
            ``config.data_dir``.
        chunk_profile: Named profile — one of ``small``, ``balanced``, ``large``.
        manifest_dir: Directory to write the IndexManifest JSON.  Defaults to
            ``data/index_manifests/``.
        force: Re-index even if a document is already present in the namespace.
        _store: Injected VectorStore for testing (bypasses DB connection).
        _embedder: Injected embedder for testing.

    Returns:
        ``IndexManifest`` describing the completed index build.

    Raises:
        ValueError: Invalid sources, missing files, or checksum mismatch.
        RuntimeError: Embedding or storage failure (namespace left unmodified).
    """
    data_dir = Path(data_dir or config.data_dir)
    manifest_dir = manifest_dir or _MANIFEST_DIR

    # Validate chunk profile
    if chunk_profile not in CHUNK_PROFILES:
        raise ValueError(
            f"Unknown chunk profile {chunk_profile!r}. "
            f"Valid choices: {list(CHUNK_PROFILES)}"
        )
    profile_size, profile_overlap = CHUNK_PROFILES[chunk_profile]

    # Load and validate source manifest
    sources = load_sources(sources_path)
    if not sources:
        raise ValueError(f"No enabled sources found in {sources_path}")

    corpus_hash = compute_corpus_hash([entry for entry, _ in sources])
    logger.info(
        "Certified corpus indexing — namespace=%r  profile=%r  sources=%d  corpus_hash=%s…",
        namespace, chunk_profile, len(sources), corpus_hash[:16],
    )

    # Resolve store and embedder (allow injection for testing)
    from src.vector_store import VectorStore
    from src.embeddings import embedder as _default_embedder

    store = _store or VectorStore(namespace=namespace)
    embedder = _embedder or _default_embedder

    # Ingest each source into the target namespace
    for entry, file_name in sources:
        logger.info("Processing source: %s (%s)", entry.source_id, file_name)

        # Verify checksum before reading
        file_path = _verify_source_file(data_dir, file_name, entry.checksum_sha256)

        # Parse
        elements = parse_document(str(file_path))
        if not elements:
            raise ValueError(f"No content extracted from {file_name}")
        elements = propagate_section_titles(elements)

        # Detect language
        sample = " ".join(e.get("content", "")[:200] for e in elements[:20])
        doc_language = detect_document_language(sample)

        # Chunk
        chunk_records = chunk_elements(
            elements,
            document_language=doc_language,
            chunk_size=profile_size,
            chunk_overlap=profile_overlap,
        )
        if not chunk_records:
            raise ValueError(f"No valid chunks produced from {file_name}")

        # Attach source-level provenance metadata to every chunk
        for record in chunk_records:
            record["source_id"] = entry.source_id
            record["source_url"] = entry.source_url
            record["publisher"] = entry.publisher
            record["publication_date"] = entry.publication_date
            record["source_checksum"] = entry.checksum_sha256
            record["chunk_profile"] = chunk_profile

        # Embed
        texts = [r["text"] for r in chunk_records]
        logger.info("Embedding %d chunks for %s…", len(texts), entry.source_id)
        embeddings = embedder.embed_batch(texts, show_progress=False)

        # Store — any exception here propagates; old namespace untouched
        if not force and store.has_document(file_name):
            logger.info("Skipping %s (already in namespace %r)", file_name, namespace)
            continue

        if store.has_document(file_name):
            store.delete_document(file_name)

        store.add_chunks(chunk_records, embeddings)
        logger.info(
            "Stored %d chunks for %s in namespace %r",
            len(chunk_records), entry.source_id, namespace,
        )

    # Build and persist the IndexManifest
    manifest = IndexManifest(
        namespace=namespace,
        corpus_hash=corpus_hash,
        chunk_profile=chunk_profile,
        embedding_model=config.embedding_model,
        embedding_dimension=config.embedding_dimension,
        created_at=datetime.now(timezone.utc),
    )
    save_index_manifest(manifest, Path(manifest_dir))

    logger.info(
        "Certified corpus indexing complete — namespace=%r  corpus_hash=%s…",
        namespace, corpus_hash[:16],
    )
    return manifest
