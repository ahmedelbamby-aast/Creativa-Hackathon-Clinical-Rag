"""Immutable per-dimension embedding runtimes for interactive retrieval.

Each browser session chooses one supported output dimension.  The request is
routed to a matching Gemini embedder, pgvector parent table, and namespace;
global configuration is never mutated, so concurrent sessions cannot leak a
dimension choice into one another.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import psycopg
from psycopg import sql

from src.config import config
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


SUPPORTED_EMBEDDING_DIMENSIONS = (384, 768, 1024, 2048, 3072)


def validate_embedding_dimension(dimension: int | None) -> int:
    """Return a registered dimension or reject the request deterministically."""
    resolved = config.embedding_dimension if dimension is None else int(dimension)
    if resolved not in SUPPORTED_EMBEDDING_DIMENSIONS:
        supported = ", ".join(str(value) for value in SUPPORTED_EMBEDDING_DIMENSIONS)
        raise ValueError(f"Unsupported embedding dimension. Choose one of: {supported}")
    return resolved


def namespace_for_dimension(dimension: int) -> str:
    """Resolve a dimension-specific namespace without mutating process state."""
    dimension = validate_embedding_dimension(dimension)
    return config.embedding_namespace_for_dimension(dimension)


@dataclass(frozen=True)
class EmbeddingRuntime:
    """One concurrency-safe embedding/query route."""

    dimension: int
    namespace: str
    provider: str
    model: str
    embedder: EmbeddingModel
    vector_store: VectorStore

    @property
    def table_family(self) -> str:
        return self.vector_store.parent_table


@lru_cache(maxsize=len(SUPPORTED_EMBEDDING_DIMENSIONS))
def get_embedding_runtime(dimension: int | None = None) -> EmbeddingRuntime:
    """Build and cache the immutable runtime for one output dimension."""
    if dimension is None:
        return get_embedding_runtime(config.embedding_dimension)
    resolved = validate_embedding_dimension(dimension)
    namespace = namespace_for_dimension(resolved)
    embedder = EmbeddingModel(
        provider=config.embedding_provider,
        dimension=resolved,
        local_model_name=config.embedding_model,
        online_model_name=config.online_embedding_model,
    )
    store = VectorStore(
        database_url=config.database_url,
        namespace=namespace,
        dimension=resolved,
    )
    return EmbeddingRuntime(
        dimension=resolved,
        namespace=store.namespace,
        provider=embedder.provider,
        model=embedder.model_name,
        embedder=embedder,
        vector_store=store,
    )


def embedding_profile_catalog() -> dict[str, object]:
    """Return public readiness metadata for every supported dimension."""
    runtimes = [get_embedding_runtime(dimension) for dimension in SUPPORTED_EMBEDDING_DIMENSIONS]
    expected_document_count = config.embedding_expected_document_count
    counts: dict[int, tuple[int, int]] = {}
    try:
        with psycopg.connect(config.database_url) as connection:
            for runtime in runtimes:
                table_exists = connection.execute(
                    "SELECT to_regclass(%s)",
                    (runtime.table_family,),
                ).fetchone()[0]
                if table_exists is None:
                    counts[runtime.dimension] = (0, 0)
                    continue
                row = connection.execute(
                    sql.SQL(
                        "SELECT COUNT(*), COUNT(DISTINCT document_name) "
                        "FROM {} WHERE namespace = %s"
                    ).format(sql.Identifier(runtime.table_family)),
                    (runtime.namespace,),
                ).fetchone()
                counts[runtime.dimension] = (int(row[0]), int(row[1]))
    except Exception:
        counts = {}

    profiles: list[dict[str, object]] = []
    for runtime in runtimes:
        indexed_chunks, document_count = counts.get(runtime.dimension, (0, 0))
        available = indexed_chunks > 0 and document_count == expected_document_count
        profiles.append(
            {
                "dimension": runtime.dimension,
                "namespace": runtime.namespace,
                "provider": runtime.provider,
                "model": runtime.model,
                "table_family": runtime.table_family,
                "available": available,
                "indexed_chunks": indexed_chunks,
                "document_count": document_count,
                "expected_document_count": expected_document_count,
                "index_status": (
                    "ready"
                    if available
                    else "ingesting"
                    if indexed_chunks > 0
                    else "pending"
                ),
            }
        )
    return {
        "default_dimension": validate_embedding_dimension(None),
        "expected_document_count": expected_document_count,
        "profiles": profiles,
    }
