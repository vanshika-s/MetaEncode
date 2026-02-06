# tests/test_ml/test_similarity.py
"""Tests for similarity computation engine."""

import numpy as np
import pandas as pd
import pytest

from src.ml.similarity import SimilarityEngine


class TestSimilarityEngine:
    """Test suite for SimilarityEngine."""

    def test_engine_initialization(self):
        """Test that engine initializes correctly."""
        engine = SimilarityEngine()
        assert engine.metric == "cosine"
        assert engine._fitted is False

        engine_euclidean = SimilarityEngine(metric="euclidean")
        assert engine_euclidean.metric == "euclidean"

    def test_fit_sets_fitted_flag(self, sample_embeddings):
        """Test that fit() sets the fitted flag."""
        engine = SimilarityEngine()
        engine.fit(sample_embeddings)
        assert engine._fitted is True

    def test_fit_returns_self(self, sample_embeddings):
        """Test that fit() returns self for chaining."""
        engine = SimilarityEngine()
        result = engine.fit(sample_embeddings)
        assert result is engine

    def test_find_similar_returns_correct_count(
        self, sample_embeddings, sample_embedding_single
    ):
        """Test that find_similar returns requested number of results."""
        engine = SimilarityEngine()
        engine.fit(sample_embeddings)
        results = engine.find_similar(sample_embedding_single, n=5)
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 5
        assert "index" in results.columns
        assert "similarity_score" in results.columns

    def test_find_similar_returns_sorted_results(
        self, sample_embeddings, sample_embedding_single
    ):
        """Test that find_similar returns results sorted by similarity."""
        engine = SimilarityEngine()
        engine.fit(sample_embeddings)
        results = engine.find_similar(sample_embedding_single, n=5)
        scores = results["similarity_score"].tolist()
        # Results should be sorted in descending order of similarity
        assert scores == sorted(scores, reverse=True)

    def test_compute_similarity_range(self, sample_embeddings):
        """Test that cosine similarity is in valid range."""
        engine = SimilarityEngine()
        similarity = engine.compute_similarity(
            sample_embeddings[0], sample_embeddings[1]
        )
        # Cosine similarity can be negative for random vectors
        assert -1 <= similarity <= 1

    def test_compute_similarity_self(self, sample_embeddings):
        """Test that similarity of vector with itself is 1."""
        engine = SimilarityEngine()
        similarity = engine.compute_similarity(
            sample_embeddings[0], sample_embeddings[0]
        )
        assert np.isclose(similarity, 1.0)

    def test_similarity_matrix_is_symmetric(self, sample_embeddings):
        """Test that similarity matrix is symmetric."""
        engine = SimilarityEngine()
        engine.fit(sample_embeddings)
        matrix = engine.compute_similarity_matrix()
        assert np.allclose(matrix, matrix.T)

    def test_similarity_matrix_diagonal_is_one(self, sample_embeddings):
        """Test that diagonal of similarity matrix is 1 (self-similarity)."""
        engine = SimilarityEngine()
        engine.fit(sample_embeddings)
        matrix = engine.compute_similarity_matrix()
        diagonal = np.diag(matrix)
        assert np.allclose(diagonal, 1.0)

    def test_find_similar_raises_if_not_fitted(self, sample_embedding_single):
        """Test that find_similar raises error if not fitted."""
        engine = SimilarityEngine()
        with pytest.raises(ValueError, match="not been fitted"):
            engine.find_similar(sample_embedding_single, n=5)

    def test_compute_similarity_matrix_raises_if_not_fitted(self):
        """Test that compute_similarity_matrix raises error if not fitted."""
        engine = SimilarityEngine()
        with pytest.raises(ValueError, match="not been fitted"):
            engine.compute_similarity_matrix()

    def test_get_embedding(self, sample_embeddings):
        """Test that get_embedding returns correct embedding."""
        engine = SimilarityEngine()
        engine.fit(sample_embeddings)
        embedding = engine.get_embedding(0)
        # Use allclose for float comparison (fit converts to float32)
        assert np.allclose(embedding, sample_embeddings[0], rtol=1e-5)

    def test_get_embedding_raises_for_invalid_index(self, sample_embeddings):
        """Test that get_embedding raises error for invalid index."""
        engine = SimilarityEngine()
        engine.fit(sample_embeddings)
        with pytest.raises(ValueError, match="out of range"):
            engine.get_embedding(100)

    def test_euclidean_similarity(self, sample_embeddings):
        """Test euclidean metric works."""
        engine = SimilarityEngine(metric="euclidean")
        engine.fit(sample_embeddings)
        results = engine.find_similar(sample_embeddings[0], n=3)
        assert len(results) == 3


# ============================================================================
# Edge Case Tests (Lines 140-141, 161-162, 179)
# ============================================================================


class TestSimilarityEngineEdgeCases:
    """Tests for edge cases in SimilarityEngine."""

    def test_compute_similarity_euclidean(self, sample_embeddings):
        """Lines 140-141: compute_similarity with euclidean metric."""
        engine = SimilarityEngine(metric="euclidean")
        # Note: compute_similarity doesn't require fitting
        similarity = engine.compute_similarity(
            sample_embeddings[0], sample_embeddings[1]
        )
        # Euclidean similarity uses 1 / (1 + distance), so it's in (0, 1]
        assert 0 < similarity <= 1

    def test_compute_similarity_matrix_euclidean(self, sample_embeddings):
        """Lines 161-162: compute_similarity_matrix with euclidean metric."""
        engine = SimilarityEngine(metric="euclidean")
        engine.fit(sample_embeddings)
        matrix = engine.compute_similarity_matrix()
        # Matrix should be square
        assert matrix.shape == (len(sample_embeddings), len(sample_embeddings))
        # Diagonal should be 1.0 (self-similarity)
        assert np.allclose(np.diag(matrix), 1.0)
        # Matrix should be symmetric
        assert np.allclose(matrix, matrix.T)
        # All values should be in (0, 1]
        assert (matrix > 0).all() and (matrix <= 1).all()

    def test_get_embedding_raises_if_not_fitted(self):
        """Line 179: get_embedding raises ValueError when not fitted."""
        engine = SimilarityEngine()
        with pytest.raises(ValueError, match="not been fitted"):
            engine.get_embedding(0)
