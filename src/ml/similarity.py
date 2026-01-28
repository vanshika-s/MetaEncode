# src/ml/similarity.py
"""Similarity computation and nearest neighbor search.

This module provides functionality for computing similarity between
dataset embeddings and finding the most similar datasets using
cosine similarity and nearest neighbor algorithms.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors


class SimilarityEngine:
    """Compute similarity between dataset embeddings.

    This class provides methods for:
    - Computing cosine similarity between vectors
    - Finding top-N most similar datasets
    - Building nearest neighbor indices for efficient search

    Example:
        >>> engine = SimilarityEngine()
        >>> engine.fit(embeddings_matrix)
        >>> similar = engine.find_similar(query_embedding, n=10)
    """

    def __init__(self, metric: str = "cosine", n_neighbors: int = 50) -> None:
        """Initialize the similarity engine.

        Args:
            metric: Similarity metric to use ("cosine" or "euclidean").
            n_neighbors: Maximum neighbors to consider for index.
        """
        self.metric = metric
        self.n_neighbors = n_neighbors
        self._embeddings: Optional[np.ndarray] = None
        self._index: Optional[Any] = None
        self._fitted = False

    def fit(self, embeddings: np.ndarray) -> "SimilarityEngine":
        """Fit the similarity engine with dataset embeddings.

        Args:
            embeddings: NumPy array of shape (n_datasets, embedding_dim).

        Returns:
            Self for method chaining.
        """
        self._embeddings = np.asarray(embeddings, dtype=np.float32)

        # Build NearestNeighbors index
        n_neighbors = min(self.n_neighbors, len(embeddings))

        self._index = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=self.metric,
            algorithm="brute",  # Works well with cosine
        )
        self._index.fit(self._embeddings)

        self._fitted = True
        return self

    def find_similar(
        self,
        query_embedding: np.ndarray,
        n: int = 10,
        exclude_self: bool = True,
    ) -> pd.DataFrame:
        """Find the N most similar datasets to a query.

        Args:
            query_embedding: Embedding vector of the query dataset.
            n: Number of similar datasets to return.
            exclude_self: Whether to exclude exact matches (similarity > 0.9999).

        Returns:
            DataFrame with columns: index, similarity_score.

        Raises:
            ValueError: If engine has not been fitted.
        """
        if not self._fitted or self._index is None or self._embeddings is None:
            raise ValueError("Engine has not been fitted. Call fit() first.")

        # Ensure query is 2D
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

        # Request extra neighbors if excluding self
        k = min(n + (1 if exclude_self else 0), len(self._embeddings))

        # Find nearest neighbors
        distances, indices = self._index.kneighbors(query, n_neighbors=k)

        # Convert distances to similarities
        if self.metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            similarities = 1 - distances[0]
        else:
            # For euclidean, convert to similarity (inverse relationship)
            similarities = 1 / (1 + distances[0])

        # Build results
        results = []
        for idx, sim in zip(indices[0], similarities):
            # Exclude exact self-matches if requested
            if exclude_self and sim > 0.9999:
                continue
            results.append({"index": int(idx), "similarity_score": float(sim)})
            if len(results) >= n:
                break

        return pd.DataFrame(results)

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compute similarity between two embedding vectors.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Similarity score (0 to 1 for cosine similarity).
        """
        e1 = np.asarray(embedding1, dtype=np.float32).reshape(1, -1)
        e2 = np.asarray(embedding2, dtype=np.float32).reshape(1, -1)

        if self.metric == "cosine":
            similarity = cosine_similarity(e1, e2)[0, 0]
        else:
            # Euclidean distance to similarity
            distance = euclidean_distances(e1, e2)[0, 0]
            similarity = 1 / (1 + distance)

        return float(similarity)

    def compute_similarity_matrix(self) -> np.ndarray:
        """Compute pairwise similarity matrix for all datasets.

        Returns:
            NumPy array of shape (n_datasets, n_datasets) with similarity scores.

        Raises:
            ValueError: If engine has not been fitted.
        """
        if not self._fitted or self._embeddings is None:
            raise ValueError("Engine has not been fitted. Call fit() first.")

        if self.metric == "cosine":
            similarity_matrix: np.ndarray = cosine_similarity(self._embeddings)
        else:
            # Convert euclidean distances to similarity
            distances: np.ndarray = euclidean_distances(self._embeddings)
            similarity_matrix = 1 / (1 + distances)

        return np.asarray(similarity_matrix, dtype=np.float32)

    def get_embedding(self, index: int) -> np.ndarray:
        """Get the embedding vector for a specific dataset.

        Args:
            index: Index of the dataset.

        Returns:
            Embedding vector.

        Raises:
            ValueError: If engine has not been fitted or index out of range.
        """
        if not self._fitted or self._embeddings is None:
            raise ValueError("Engine has not been fitted. Call fit() first.")

        if index < 0 or index >= len(self._embeddings):
            raise ValueError(
                f"Index {index} out of range [0, {len(self._embeddings)})."
            )

        return np.asarray(self._embeddings[index], dtype=np.float32)
