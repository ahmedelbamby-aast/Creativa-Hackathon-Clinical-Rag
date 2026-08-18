"""Persistence and validation helpers for reproducible retrieval indexes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import config
from src.retrieval_contracts import IndexManifest
from src.source_catalog import source_catalog_hash
from src.vector_store import normalize_namespace


MANIFEST_DIR = config.project_root / "data" / "index_manifests"


def corpus_hash(paths: list[Path]) -> str:
    """Hash local corpus bytes and relative names in deterministic order."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name.lower()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def manifest_path(namespace: str, directory: Path = MANIFEST_DIR) -> Path:
    return directory / f"{normalize_namespace(namespace)}.json"


def build_index_manifest(
    namespace: str,
    corpus_paths: list[Path],
    token_count: int,
) -> IndexManifest:
    """Create the exact manifest for the currently configured index process."""
    char_size, char_overlap = config.selected_chunk_profile
    return IndexManifest(
        namespace=normalize_namespace(namespace),
        corpus_hash=corpus_hash(corpus_paths),
        chunk_profile=config.retrieval_profile,
        embedding_provider=config.embedding_provider,
        embedding_model=(
            config.online_embedding_model
            if config.embedding_provider == "gemini"
            else config.embedding_model
        ),
        dimension=config.embedding_dimension,
        created_at=datetime.now(timezone.utc).isoformat(),
        source_catalog_hash=source_catalog_hash(),
        char_chunk_size=char_size,
        char_chunk_overlap=char_overlap,
        token_count=token_count,
    )


def write_index_manifest(manifest: IndexManifest, directory: Path = MANIFEST_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = manifest_path(manifest.namespace, directory)
    path.write_text(
        json.dumps(manifest.serializable(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_index_manifest(namespace: str, directory: Path = MANIFEST_DIR) -> IndexManifest | None:
    path = manifest_path(namespace, directory)
    if not path.exists():
        return None
    return IndexManifest(**json.loads(path.read_text(encoding="utf-8")))


def index_manifest_hash(manifest: IndexManifest) -> str:
    payload = json.dumps(manifest.serializable(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_matches_runtime(manifest: IndexManifest, namespace: str) -> bool:
    """Return whether an index manifest is compatible with the active runtime."""
    return (
        manifest.namespace == normalize_namespace(namespace)
        and manifest.dimension == config.embedding_dimension
        and manifest.chunk_profile == config.retrieval_profile
        and manifest.embedding_provider == config.embedding_provider
        and manifest.embedding_model
        == (
            config.online_embedding_model
            if config.embedding_provider == "gemini"
            else config.embedding_model
        )
    )
