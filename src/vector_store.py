"""ChromaDB vector store interface.

Manages three category-specific collections:
  - diabetes_treatment
  - diabetes_prevention
  - diabetes_nutrition

Each collection stores:
  - id:        chunk_id (string)
  - embedding: embedding vector (list[float])
  - document:  chunk text
  - metadata:  all ChunkRecord fields except 'text'

Usage
-----
    from src.vector_store import vector_store

    # Add chunks
    vector_store.add_chunks(chunk_records, embeddings)

    # Query
    results = vector_store.query(
        query_embedding=vector,
        category="nutrition",
        top_k=5,
    )

    # Stats
    print(vector_store.collection_stats())
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from src.config import (
    config,
    ALL_CATEGORIES,
    CATEGORY_GENERAL,
    CATEGORY_ALL,
    _collection_name,
)

logger = logging.getLogger(__name__)

# Metadata fields that can be stored in ChromaDB (must be str/int/float/bool)
_METADATA_FIELDS = [
    "chunk_id",
    "document_name",
    "page_number",
    "section_title",
    "subsection_title",
    "category",
    "content_type",
    "language",
    "char_count",
    "word_count",
    "quality_score",
]


def _safe_metadata(record: dict) -> dict:
    """Extract ChromaDB-safe metadata from a chunk record.

    ChromaDB metadata values must be str, int, float, or bool.
    Truncates long string fields to prevent storage issues.
    """
    meta = {}
    for field in _METADATA_FIELDS:
        val = record.get(field, "")
        if val is None:
            val = ""
        # Truncate long strings (section titles can be verbose)
        if isinstance(val, str) and len(val) > 512:
            val = val[:512]
        meta[field] = val
    return meta


def cosine_distance_to_score(distance: float) -> float:
    """Convert Chroma cosine distance to a similarity score in ``[0, 1]``."""
    return round(max(0.0, min(1.0, 1.0 - float(distance))), 4)


class VectorStore:
    """ChromaDB-backed vector store for three diabetes knowledge categories."""

    def __init__(self) -> None:
        self._client: Optional[chromadb.PersistentClient] = None
        self._collections: dict[str, chromadb.Collection] = {}

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            db_path = str(config.chroma_db_dir)
            Path(db_path).mkdir(parents=True, exist_ok=True)
            logger.info("Initialising ChromaDB at: %s", db_path)
            self._client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self, category: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection for a category."""
        if category not in self._collections:
            client = self._get_client()
            name = _collection_name(category)
            collection = client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            self._collections[category] = collection
            logger.debug("Using collection: %s (%d items)", name, collection.count())
        return self._collections[category]

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunk_records: list[dict],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> dict[str, int]:
        """Add chunk records and their embeddings to the appropriate collections.

        Chunks are routed to collections based on their 'category' field.
        'general' category chunks are added to all three collections.

        Args:
            chunk_records: List of ChunkRecord dicts.
            embeddings: Corresponding embedding vectors (same length/order).
            batch_size: ChromaDB upsert batch size.

        Returns:
            Dict mapping category → number of chunks added.
        """
        if len(chunk_records) != len(embeddings):
            raise ValueError(
                f"chunk_records ({len(chunk_records)}) and embeddings "
                f"({len(embeddings)}) must have the same length"
            )

        # Group by target collections
        by_category: dict[str, list[tuple[dict, list[float]]]] = {
            cat: [] for cat in ALL_CATEGORIES
        }
        for record, emb in zip(chunk_records, embeddings):
            cat = record.get("category", CATEGORY_GENERAL)
            if cat == CATEGORY_GENERAL:
                # General chunks go into all collections for maximum recall
                for c in ALL_CATEGORIES:
                    by_category[c].append((record, emb))
            elif cat in by_category:
                by_category[cat].append((record, emb))
            else:
                # Unknown category → all collections
                logger.warning("Unknown category '%s' for chunk %s", cat, record.get("chunk_id"))
                for c in ALL_CATEGORIES:
                    by_category[c].append((record, emb))

        added: dict[str, int] = {}
        for cat, pairs in by_category.items():
            if not pairs:
                added[cat] = 0
                continue

            collection = self._get_collection(cat)
            records_cat, embeddings_cat = zip(*pairs)

            # Process in batches
            count = 0
            for start in range(0, len(records_cat), batch_size):
                end = start + batch_size
                batch_records = records_cat[start:end]
                batch_embeddings = list(embeddings_cat[start:end])

                ids = [r["chunk_id"] for r in batch_records]
                documents = [r["text"] for r in batch_records]
                metadatas = [_safe_metadata(r) for r in batch_records]

                try:
                    collection.upsert(
                        ids=ids,
                        embeddings=batch_embeddings,
                        documents=documents,
                        metadatas=metadatas,
                    )
                    count += len(batch_records)
                except Exception as e:
                    logger.error(
                        "ChromaDB upsert failed for category=%s batch=%d: %s",
                        cat, start // batch_size, e,
                    )

            added[cat] = count
            logger.info("Added %d chunks to collection '%s'", count, _collection_name(cat))

        return added

    def has_document(self, document_name: str) -> bool:
        """Return whether any collection already contains a document."""
        for category in ALL_CATEGORIES:
            collection = self._get_collection(category)
            result = collection.get(
                where={"document_name": document_name},
                limit=1,
                include=[],
            )
            if result.get("ids"):
                return True
        return False

    def delete_document(self, document_name: str) -> None:
        """Remove all chunks for a document from every collection."""
        for category in ALL_CATEGORIES:
            collection = self._get_collection(category)
            collection.delete(where={"document_name": document_name})

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        category: str = CATEGORY_ALL,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Semantic similarity search.

        Args:
            query_embedding: Query vector from the embedding model.
            category: One of ALL_CATEGORIES or "all" (queries all three).
            top_k: Number of results per collection.
            where: Optional ChromaDB metadata filter.

        Returns:
            List of result dicts sorted by distance (ascending = most similar).
            Each result has: id, document, metadata, distance, score.
        """
        if category == CATEGORY_ALL or category not in ALL_CATEGORIES:
            # Query all collections and merge
            all_results: list[dict] = []
            for cat in ALL_CATEGORIES:
                results = self._query_collection(cat, query_embedding, top_k, where)
                all_results.extend(results)
            # Deduplicate by chunk_id (general chunks appear in all collections)
            seen: set[str] = set()
            deduped: list[dict] = []
            for r in all_results:
                cid = r["metadata"].get("chunk_id", r["id"])
                if cid not in seen:
                    seen.add(cid)
                    deduped.append(r)
            # Sort by distance (lower = more similar)
            deduped.sort(key=lambda x: x["distance"])
            return deduped[:top_k]

        return self._query_collection(category, query_embedding, top_k, where)

    def _query_collection(
        self,
        category: str,
        query_embedding: list[float],
        top_k: int,
        where: Optional[dict],
    ) -> list[dict]:
        """Query a single collection."""
        collection = self._get_collection(category)

        if collection.count() == 0:
            logger.debug("Collection '%s' is empty", _collection_name(category))
            return []

        # ChromaDB n_results must not exceed collection size
        n = min(top_k, collection.count())

        try:
            kwargs: dict = {
                "query_embeddings": [query_embedding],
                "n_results": n,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where

            raw = collection.query(**kwargs)
        except Exception as e:
            logger.error("ChromaDB query failed for '%s': %s", category, e)
            return []

        results: list[dict] = []
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            score = cosine_distance_to_score(dist)
            results.append({
                "id": cid,
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "score": score,
            })

        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def collection_stats(self) -> dict[str, int]:
        """Return the document count for each category collection."""
        stats: dict[str, int] = {}
        for cat in ALL_CATEGORIES:
            try:
                col = self._get_collection(cat)
                stats[cat] = col.count()
            except Exception:
                stats[cat] = -1
        return stats

    def reset_collection(self, category: str) -> None:
        """Delete and recreate a collection (clears all data)."""
        client = self._get_client()
        name = _collection_name(category)
        try:
            client.delete_collection(name)
            logger.warning("Deleted collection: %s", name)
        except Exception:
            pass
        # Remove cached reference so it gets recreated on next access
        self._collections.pop(category, None)

    def reset_all(self) -> None:
        """Delete and recreate all category collections."""
        for cat in ALL_CATEGORIES:
            self.reset_collection(cat)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
vector_store = VectorStore()
