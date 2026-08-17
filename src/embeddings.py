"""Embedding model wrapper using sentence-transformers.

Uses a free multilingual model that supports English, Arabic, and Egyptian
Arabic. The embedding dimension is determined automatically from the model
at runtime — never hardcoded.

Default model: paraphrase-multilingual-MiniLM-L12-v2
  - Dimension: 384
  - Supports 50+ languages including Arabic
  - ~470 MB download, cached locally after first use
  - Free, no API key required

Usage
-----
    from src.embeddings import embedder

    # Single query
    vector = embedder.embed_query("What foods are recommended for diabetics?")

    # Batch (for ingestion)
    vectors = embedder.embed_batch(["chunk 1 text", "chunk 2 text"])
"""

import logging
from typing import Optional

import numpy as np

from src.config import config

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Lazy-loading sentence-transformer embedding model.

    The model is loaded on first use (not at import time) to keep startup
    fast and allow tests to run without downloading the model.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = model_name or config.embedding_model
        self._model = None
        self._dimension: Optional[int] = None

    def _load(self) -> None:
        """Load the model if not already loaded."""
        if self._model is not None:
            return
        logger.info("Loading embedding model: %s", self._model_name)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                "Embedding model loaded. Dimension: %d", self._dimension
            )
        except Exception as e:
            logger.error("Failed to load embedding model %s: %s", self._model_name, e)
            raise

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (loads model if needed)."""
        self._load()
        return self._dimension  # type: ignore[return-value]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: Query text (English or Arabic).

        Returns:
            Embedding vector as a list of floats.
        """
        self._load()
        if not text or not text.strip():
            raise ValueError("Cannot embed empty query")
        try:
            vector = self._model.encode(  # type: ignore[union-attr]
                text.strip(),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vector.tolist()
        except Exception as e:
            logger.error("Failed to embed query: %s", e)
            raise

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Embed a list of texts in batches.

        Args:
            texts: List of strings to embed.
            batch_size: Number of texts per batch.
            show_progress: Show tqdm progress bar.

        Returns:
            List of embedding vectors (same order as input).
        """
        self._load()
        if not texts:
            return []

        # Filter empty strings (replace with placeholder to preserve order)
        clean_texts = [t.strip() if t and t.strip() else "." for t in texts]

        try:
            vectors = self._model.encode(  # type: ignore[union-attr]
                clean_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            return vectors.tolist()
        except Exception as e:
            logger.error("Batch embedding failed: %s", e)
            raise

    @property
    def model_name(self) -> str:
        return self._model_name


# ---------------------------------------------------------------------------
# Singleton — import this everywhere
# ---------------------------------------------------------------------------
embedder = EmbeddingModel()
