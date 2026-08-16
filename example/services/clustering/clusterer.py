"""Semantic clustering using HDBSCAN."""

import numpy as np


class Clusterer:
    """Cluster embeddings using HDBSCAN with dynamic minimum cluster size."""

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int = 1,
        cluster_selection_method: str = "eom",
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self._clusterer = None

    @property
    def clusterer(self):
        """Lazy-load the HDBSCAN clusterer."""
        if self._clusterer is None:
            import hdbscan
            self._clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_method=self.cluster_selection_method,
                metric="euclidean",
            )
        return self._clusterer

    def cluster(self, embeddings: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        """Cluster embeddings and return labels and probabilities.

        Args:
            embeddings: List of embeddings to cluster.

        Returns:
            Tuple of (cluster_labels, membership_probabilities).
            Noise points are labeled as -1.
        """
        labels = self.clusterer.fit_predict(np.array(embeddings))
        probabilities = self.clusterer.probabilities_
        return labels, probabilities

    def assign_noise_points(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Assign noise points (-1) to nearest valid cluster.

        Args:
            embeddings: All embeddings.
            labels: Cluster labels from HDBSCAN.

        Returns:
            Updated labels with noise points assigned.
        """
        noise_mask = labels == -1
        if not np.any(noise_mask):
            return labels

        valid_mask = ~noise_mask
        if not np.any(valid_mask):
            return labels

        # Calculate centroids of valid clusters
        unique_labels = np.unique(labels[valid_mask])
        centroids = {}
        for label in unique_labels:
            centroids[label] = embeddings[labels == label].mean(axis=0)

        # Assign each noise point to nearest centroid
        updated_labels = labels.copy()
        noise_indices = np.where(noise_mask)[0]

        for idx in noise_indices:
            distances = {
                label: np.linalg.norm(embeddings[idx] - centroid)
                for label, centroid in centroids.items()
            }
            updated_labels[idx] = min(distances, key=distances.get)

        return updated_labels


# Global instance
clusterer = Clusterer()
