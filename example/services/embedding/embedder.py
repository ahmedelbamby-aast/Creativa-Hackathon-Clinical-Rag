"""Text embedding service using sentence-transformers."""

from typing import Optional

import numpy as np

from src.config.settings import get_settings


class Embedder:
    """Generate semantic embeddings for text chunks."""

    def __init__(self):
        self._model = None
        self._settings = get_settings()

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._settings.EMBEDDING_MODEL)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            List of 768 floats representing the semantic embedding.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size for processing.

        Returns:
            List of embeddings, one per input text.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()


# Global instance
embedder = Embedder()
