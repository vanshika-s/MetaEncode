# tests/test_ml/test_feature_combiner.py
"""Tests for FeatureCombiner class."""

import numpy as np
import pandas as pd
import pytest

from src.ml.feature_combiner import FeatureCombiner


@pytest.fixture
def sample_df_for_combiner() -> pd.DataFrame:
    """Sample DataFrame with all required columns for FeatureCombiner."""
    return pd.DataFrame(
        {
            "accession": ["ENCSR000AAA", "ENCSR000BBB", "ENCSR000CCC", "ENCSR000DDD"],
            "combined_text": [
                "chip seq k562 h3k27ac",
                "rna seq liver tissue",
                "atac seq hepg2 cells",
                "chip seq k562 h3k4me3",
            ],
            "assay_term_name": ["ChIP-seq", "RNA-seq", "ATAC-seq", "ChIP-seq"],
            "organism": ["human", "mouse", "human", "human"],
            "biosample_term_name": ["K562", "liver", "HepG2", "K562"],
            "lab": ["lab-a", "lab-b", "lab-a", "lab-c"],
            "replicate_count": [2, 3, 1, 4],
            "file_count": [10, 15, 8, 20],
        }
    )


@pytest.fixture
def sample_text_embeddings() -> np.ndarray:
    """Sample text embeddings matching sample_df_for_combiner."""
    np.random.seed(42)
    return np.random.randn(4, 384).astype(np.float32)


class TestFeatureCombinerInit:
    """Tests for FeatureCombiner initialization."""

    def test_init_with_default_weights(self) -> None:
        """Test initialization with default weights."""
        combiner = FeatureCombiner()
        assert combiner.weights == FeatureCombiner.DEFAULT_WEIGHTS
        assert not combiner.is_fitted

    def test_init_with_custom_weights(self) -> None:
        """Test initialization with custom weights."""
        custom_weights = {"text_embedding": 0.8, "assay_type": 0.1, "organism": 0.1}
        combiner = FeatureCombiner(weights=custom_weights)
        assert combiner.weights == custom_weights

    def test_init_text_column_parameter(self) -> None:
        """Test text_column parameter is stored."""
        combiner = FeatureCombiner(text_column="description")
        assert combiner.text_column == "description"


class TestFeatureCombinerFit:
    """Tests for FeatureCombiner.fit()."""

    def test_fit_creates_categorical_encoders(
        self, sample_df_for_combiner: pd.DataFrame
    ) -> None:
        """Test that fit creates encoders for categorical columns."""
        combiner = FeatureCombiner()
        combiner.fit(sample_df_for_combiner)

        assert combiner.is_fitted
        assert "assay_term_name" in combiner._categorical_encoders
        assert "organism" in combiner._categorical_encoders
        assert "biosample_term_name" in combiner._categorical_encoders
        assert "lab" in combiner._categorical_encoders

    def test_fit_creates_numeric_encoders(
        self, sample_df_for_combiner: pd.DataFrame
    ) -> None:
        """Test that fit creates encoders for numeric columns."""
        combiner = FeatureCombiner()
        combiner.fit(sample_df_for_combiner)

        assert "replicate_count" in combiner._numeric_encoders
        assert "file_count" in combiner._numeric_encoders

    def test_fit_tracks_dimensions(self, sample_df_for_combiner: pd.DataFrame) -> None:
        """Test that fit tracks feature dimensions correctly."""
        combiner = FeatureCombiner()
        combiner.fit(sample_df_for_combiner, text_embedding_dim=384)

        assert combiner._text_dim == 384
        # 3 unique assays: ChIP-seq, RNA-seq, ATAC-seq
        assert combiner._categorical_dims["assay_term_name"] == 3
        # 2 unique organisms: human, mouse
        assert combiner._categorical_dims["organism"] == 2
        # 3 unique biosamples: K562, liver, HepG2
        assert combiner._categorical_dims["biosample_term_name"] == 3
        # 3 unique labs: lab-a, lab-b, lab-c
        assert combiner._categorical_dims["lab"] == 3
        # 2 numeric columns
        assert combiner._numeric_dim == 2

    def test_fit_returns_self(self, sample_df_for_combiner: pd.DataFrame) -> None:
        """Test that fit returns self for chaining."""
        combiner = FeatureCombiner()
        result = combiner.fit(sample_df_for_combiner)
        assert result is combiner

    def test_fit_handles_missing_columns(self) -> None:
        """Test that fit handles DataFrames with missing columns gracefully."""
        df = pd.DataFrame(
            {
                "assay_term_name": ["ChIP-seq", "RNA-seq"],
                "organism": ["human", "mouse"],
            }
        )
        combiner = FeatureCombiner()
        combiner.fit(df)

        assert "assay_term_name" in combiner._categorical_encoders
        assert "organism" in combiner._categorical_encoders
        assert "biosample_term_name" not in combiner._categorical_encoders
        assert len(combiner._numeric_encoders) == 0


class TestFeatureCombinerTransform:
    """Tests for FeatureCombiner.transform()."""

    def test_transform_returns_correct_shape(
        self, sample_df_for_combiner: pd.DataFrame, sample_text_embeddings: np.ndarray
    ) -> None:
        """Test that transform returns correct output shape."""
        combiner = FeatureCombiner()
        combiner.fit(sample_df_for_combiner)
        combined = combiner.transform(sample_df_for_combiner, sample_text_embeddings)

        n_samples = len(sample_df_for_combiner)
        # 384 (text) + 3 (assay) + 2 (organism) + 3 (biosample) + 3 (lab) + 2 (numeric)
        expected_dim = 384 + 3 + 2 + 3 + 3 + 2
        assert combined.shape == (n_samples, expected_dim)

    def test_transform_without_text_embeddings(
        self, sample_df_for_combiner: pd.DataFrame
    ) -> None:
        """Test transform without text embeddings."""
        combiner = FeatureCombiner()
        combiner.fit(sample_df_for_combiner)
        combined = combiner.transform(sample_df_for_combiner, text_embeddings=None)

        n_samples = len(sample_df_for_combiner)
        # Only categorical + numeric: 3 + 2 + 3 + 3 + 2 = 13
        expected_dim = 3 + 2 + 3 + 3 + 2
        assert combined.shape == (n_samples, expected_dim)

    def test_transform_raises_if_not_fitted(
        self, sample_df_for_combiner: pd.DataFrame
    ) -> None:
        """Test that transform raises ValueError if not fitted."""
        combiner = FeatureCombiner()
        with pytest.raises(ValueError, match="not been fitted"):
            combiner.transform(sample_df_for_combiner)

    def test_transform_validates_embedding_length(
        self, sample_df_for_combiner: pd.DataFrame
    ) -> None:
        """Test that transform validates text_embeddings length."""
        combiner = FeatureCombiner()
        combiner.fit(sample_df_for_combiner)

        # Wrong number of embeddings
        wrong_embeddings = np.random.randn(2, 384)
        with pytest.raises(ValueError, match="doesn't match DataFrame"):
            combiner.transform(sample_df_for_combiner, wrong_embeddings)

    def test_transform_returns_float32(
        self, sample_df_for_combiner: pd.DataFrame, sample_text_embeddings: np.ndarray
    ) -> None:
        """Test that transform returns float32 array."""
        combiner = FeatureCombiner()
        combiner.fit(sample_df_for_combiner)
        combined = combiner.transform(sample_df_for_combiner, sample_text_embeddings)
        assert combined.dtype == np.float32


class TestFeatureCombinerTransformSingle:
    """Tests for FeatureCombiner.transform_single()."""

    def test_transform_single_returns_correct_shape(
        self, sample_df_for_combiner: pd.DataFrame, sample_text_embeddings: np.ndarray
    ) -> None:
        """Test that transform_single returns correct shape."""
        combiner = FeatureCombiner()
        combiner.fit(sample_df_for_combiner)

        record = sample_df_for_combiner.iloc[0].to_dict()
        text_emb = sample_text_embeddings[0]

        result = combiner.transform_single(record, text_emb)

        # Should be 1D vector
        assert result.ndim == 1
        # Same dimension as batch transform
        expected_dim = 384 + 3 + 2 + 3 + 3 + 2
        assert len(result) == expected_dim

    def test_transform_single_matches_batch(
        self, sample_df_for_combiner: pd.DataFrame, sample_text_embeddings: np.ndarray
    ) -> None:
        """Test that transform_single matches batch transform for same record."""
        combiner = FeatureCombiner()
        combiner.fit(sample_df_for_combiner)

        # Batch transform
        batch_result = combiner.transform(
            sample_df_for_combiner, sample_text_embeddings
        )

        # Single transform for first record
        record = sample_df_for_combiner.iloc[0].to_dict()
        text_emb = sample_text_embeddings[0]
        single_result = combiner.transform_single(record, text_emb)

        np.testing.assert_array_almost_equal(batch_result[0], single_result)

    def test_transform_single_raises_if_not_fitted(self) -> None:
        """Test that transform_single raises ValueError if not fitted."""
        combiner = FeatureCombiner()
        record = {"assay_term_name": "ChIP-seq", "organism": "human"}
        with pytest.raises(ValueError, match="not been fitted"):
            combiner.transform_single(record)


class TestFeatureCombinerWeights:
    """Tests for weight application in FeatureCombiner."""

    def test_weights_affect_magnitude(
        self, sample_df_for_combiner: pd.DataFrame, sample_text_embeddings: np.ndarray
    ) -> None:
        """Test that different weights produce different magnitudes."""
        # High text weight
        combiner_high_text = FeatureCombiner(
            weights={"text_embedding": 0.9, "assay_type": 0.05, "organism": 0.05}
        )
        combiner_high_text.fit(sample_df_for_combiner)
        result_high = combiner_high_text.transform(
            sample_df_for_combiner, sample_text_embeddings
        )

        # Low text weight
        combiner_low_text = FeatureCombiner(
            weights={"text_embedding": 0.1, "assay_type": 0.45, "organism": 0.45}
        )
        combiner_low_text.fit(sample_df_for_combiner)
        result_low = combiner_low_text.transform(
            sample_df_for_combiner, sample_text_embeddings
        )

        # The text embedding portion (first 384 dims) should have different magnitudes
        text_magnitude_high = np.linalg.norm(result_high[0, :384])
        text_magnitude_low = np.linalg.norm(result_low[0, :384])

        assert text_magnitude_high > text_magnitude_low

    def test_set_weights_updates_weights(self) -> None:
        """Test that set_weights updates the weight configuration."""
        combiner = FeatureCombiner()
        new_weights = {"text_embedding": 0.7, "assay_type": 0.3}
        combiner.set_weights(new_weights)
        assert combiner.weights == new_weights

    def test_get_weights_returns_copy(self) -> None:
        """Test that get_weights returns a copy."""
        combiner = FeatureCombiner()
        weights = combiner.get_weights()
        weights["text_embedding"] = 999
        assert combiner.weights["text_embedding"] != 999


class TestFeatureCombinerProperties:
    """Tests for FeatureCombiner properties."""

    def test_feature_dim_returns_total(
        self, sample_df_for_combiner: pd.DataFrame
    ) -> None:
        """Test that feature_dim returns correct total dimension."""
        combiner = FeatureCombiner()
        combiner.fit(sample_df_for_combiner, text_embedding_dim=384)

        # 384 + 3 + 2 + 3 + 3 + 2 = 397
        expected = 384 + 3 + 2 + 3 + 3 + 2
        assert combiner.feature_dim == expected

    def test_feature_dim_raises_if_not_fitted(self) -> None:
        """Test that feature_dim raises ValueError if not fitted."""
        combiner = FeatureCombiner()
        with pytest.raises(ValueError, match="not been fitted"):
            _ = combiner.feature_dim

    def test_get_feature_breakdown(self, sample_df_for_combiner: pd.DataFrame) -> None:
        """Test that get_feature_breakdown returns correct structure."""
        combiner = FeatureCombiner()
        combiner.fit(sample_df_for_combiner, text_embedding_dim=384)

        breakdown = combiner.get_feature_breakdown()

        assert breakdown["text_embedding"] == 384
        assert breakdown["assay_term_name"] == 3
        assert breakdown["organism"] == 2
        assert breakdown["biosample_term_name"] == 3
        assert breakdown["lab"] == 3
        assert breakdown["numeric_features"] == 2

    def test_get_feature_breakdown_raises_if_not_fitted(self) -> None:
        """Test that get_feature_breakdown raises ValueError if not fitted."""
        combiner = FeatureCombiner()
        with pytest.raises(ValueError, match="not been fitted"):
            combiner.get_feature_breakdown()


class TestFeatureCombinerFitTransform:
    """Tests for FeatureCombiner.fit_transform()."""

    def test_fit_transform_equivalent_to_fit_then_transform(
        self, sample_df_for_combiner: pd.DataFrame, sample_text_embeddings: np.ndarray
    ) -> None:
        """Test that fit_transform produces same result as fit then transform."""
        # Fit then transform
        combiner1 = FeatureCombiner()
        combiner1.fit(sample_df_for_combiner)
        result1 = combiner1.transform(sample_df_for_combiner, sample_text_embeddings)

        # fit_transform
        combiner2 = FeatureCombiner()
        result2 = combiner2.fit_transform(
            sample_df_for_combiner, sample_text_embeddings
        )

        np.testing.assert_array_almost_equal(result1, result2)


class TestFeatureCombinerEdgeCases:
    """Tests for edge cases in FeatureCombiner."""

    def test_handles_nan_categorical_values(self) -> None:
        """Test that NaN categorical values are handled gracefully."""
        df = pd.DataFrame(
            {
                "assay_term_name": ["ChIP-seq", None, "RNA-seq"],
                "organism": ["human", "mouse", None],
                "biosample_term_name": ["K562", "liver", "HepG2"],
                "lab": ["lab-a", "lab-a", "lab-b"],
                "replicate_count": [1, 2, 3],
                "file_count": [5, 10, 15],
            }
        )

        combiner = FeatureCombiner()
        combiner.fit(df)
        result = combiner.transform(df)

        # Should complete without error
        assert result.shape[0] == 3

    def test_handles_nan_numeric_values(self) -> None:
        """Test that NaN numeric values are handled gracefully."""
        df = pd.DataFrame(
            {
                "assay_term_name": ["ChIP-seq", "RNA-seq"],
                "organism": ["human", "mouse"],
                "biosample_term_name": ["K562", "liver"],
                "lab": ["lab-a", "lab-b"],
                "replicate_count": [2, np.nan],
                "file_count": [np.nan, 15],
            }
        )

        combiner = FeatureCombiner()
        combiner.fit(df)
        result = combiner.transform(df)

        # Should complete without error
        assert result.shape[0] == 2
        # Check no NaN in output
        assert not np.isnan(result).any()

    def test_handles_unknown_category_at_transform(self) -> None:
        """Test that unknown categories at transform time are handled."""
        # Fit on limited categories
        train_df = pd.DataFrame(
            {
                "assay_term_name": ["ChIP-seq", "RNA-seq"],
                "organism": ["human", "mouse"],
                "biosample_term_name": ["K562", "liver"],
                "lab": ["lab-a", "lab-b"],
                "replicate_count": [1, 2],
                "file_count": [5, 10],
            }
        )

        # Transform includes unknown category
        test_df = pd.DataFrame(
            {
                "assay_term_name": ["ATAC-seq"],  # Unknown
                "organism": ["human"],
                "biosample_term_name": ["NEW-CELL"],  # Unknown
                "lab": ["lab-c"],  # Unknown
                "replicate_count": [3],
                "file_count": [15],
            }
        )

        combiner = FeatureCombiner()
        combiner.fit(train_df)
        result = combiner.transform(test_df)

        # Should complete without error
        assert result.shape[0] == 1


# ============================================================================
# Additional Edge Case Tests for Coverage
# ============================================================================


class TestFeatureCombinerCoverageEdgeCases:
    """Additional edge case tests for FeatureCombiner coverage."""

    def test_transform_no_segments_raises(self) -> None:
        """Line 201: Transform with no features raises ValueError."""
        # Create combiner with all weights set to 0
        combiner = FeatureCombiner(
            weights={
                "text_embedding": 0,
                "assay_type": 0,
                "organism": 0,
                "cell_type": 0,
                "lab": 0,
                "numeric_features": 0,
            }
        )
        # Fit with minimal data (no columns that match defaults)
        df = pd.DataFrame({"accession": ["A", "B"]})
        combiner.fit(df, text_embedding_dim=384)

        # Transform should raise because no features to combine
        with pytest.raises(ValueError, match="No features to combine"):
            combiner.transform(df, text_embeddings=None)

    def test_transform_single_without_text_embedding(
        self, sample_df_for_combiner: pd.DataFrame
    ) -> None:
        """Line 258: transform_single with text_embedding=None."""
        combiner = FeatureCombiner()
        combiner.fit(sample_df_for_combiner)

        record = sample_df_for_combiner.iloc[0].to_dict()
        # Call with text_embedding=None (line 258: text_emb = None)
        result = combiner.transform_single(record, text_embedding=None)

        # Should return combined features without text portion
        assert result is not None
        # Result should be 1D vector with categorical + numeric dimensions
        # 3 (assay) + 2 (organism) + 3 (biosample) + 3 (lab) + 2 (numeric) = 13
        expected_dim = 3 + 2 + 3 + 3 + 2
        assert len(result) == expected_dim
