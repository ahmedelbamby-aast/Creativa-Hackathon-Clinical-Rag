"""Dimensionality reduction using UMAP."""

import numpy as np


class DimensionalityReducer:
    """Reduce embedding dimensions using UMAP for clustering."""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self._reducer = None

    @property
    def reducer(self):
        """Lazy-load the UMAP reducer."""
        if self._reducer is None:
            import umap
            self._reducer = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                random_state=42,
            )
        return self._reducer

    def fit_transform(self, embeddings: list[list[float]]) -> np.ndarray:
        """Fit the reducer and transform embeddings.

        Args:
            embeddings: List of high-dimensional embeddings.

        Returns:
            Reduced-dimensional embeddings.
        """
        return self.reducer.fit_transform(np.array(embeddings))

    def transform(self, embeddings: list[list[float]]) -> np.ndarray:
        """Transform embeddings using fitted reducer.

        Args:
            embeddings: List of high-dimensional embeddings.

        Returns:
            Reduced-dimensional embeddings.
        """
        return self.reducer.transform(np.array(embeddings))


# Global instance
dimensionality_reducer = DimensionalityReducer()
