# tests/test_processing/test_encoders.py
"""Tests for CategoricalEncoder and NumericEncoder classes."""

import numpy as np
import pandas as pd
import pytest

from src.processing.encoders import CategoricalEncoder, NumericEncoder


class TestCategoricalEncoderInit:
    """Tests for CategoricalEncoder initialization."""

    def test_init_default_onehot(self) -> None:
        """Test default encoding type is one-hot."""
        encoder = CategoricalEncoder()
        assert encoder.encoding_type == "onehot"
        assert encoder.handle_unknown == "ignore"

    def test_init_label_encoding(self) -> None:
        """Test initialization with label encoding."""
        encoder = CategoricalEncoder(encoding_type="label")
        assert encoder.encoding_type == "label"

    def test_init_handle_unknown_error(self) -> None:
        """Test initialization with handle_unknown=error."""
        encoder = CategoricalEncoder(handle_unknown="error")
        assert encoder.handle_unknown == "error"


class TestCategoricalEncoderOneHot:
    """Tests for one-hot encoding."""

    def test_onehot_encoding_shape(self) -> None:
        """Test one-hot encoding produces correct shape."""
        series = pd.Series(["A", "B", "C", "A", "B"])
        encoder = CategoricalEncoder(encoding_type="onehot")
        encoded = encoder.fit_transform(series)

        assert encoded.shape == (5, 3)  # 5 samples, 3 categories

    def test_onehot_encoding_values(self) -> None:
        """Test one-hot encoding produces correct values."""
        series = pd.Series(["A", "B", "A"])
        encoder = CategoricalEncoder(encoding_type="onehot")
        encoded = encoder.fit_transform(series)

        # Categories are sorted: A=0, B=1
        expected = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
        np.testing.assert_array_equal(encoded, expected)

    def test_onehot_returns_float32(self) -> None:
        """Test one-hot encoding returns float32."""
        series = pd.Series(["A", "B"])
        encoder = CategoricalEncoder(encoding_type="onehot")
        encoded = encoder.fit_transform(series)

        assert encoded.dtype == np.float32


class TestCategoricalEncoderLabel:
    """Tests for label encoding."""

    def test_label_encoding_shape(self) -> None:
        """Test label encoding produces correct shape."""
        series = pd.Series(["A", "B", "C", "A", "B"])
        encoder = CategoricalEncoder(encoding_type="label")
        encoded = encoder.fit_transform(series)

        assert encoded.shape == (5,)  # 1D array

    def test_label_encoding_values(self) -> None:
        """Test label encoding produces correct values."""
        series = pd.Series(["A", "B", "A"])
        encoder = CategoricalEncoder(encoding_type="label")
        encoded = encoder.fit_transform(series)

        # Categories are sorted: A=0, B=1
        expected = np.array([0, 1, 0], dtype=np.int32)
        np.testing.assert_array_equal(encoded, expected)

    def test_label_returns_int32(self) -> None:
        """Test label encoding returns int32."""
        series = pd.Series(["A", "B"])
        encoder = CategoricalEncoder(encoding_type="label")
        encoded = encoder.fit_transform(series)

        assert encoded.dtype == np.int32


class TestCategoricalEncoderFitTransform:
    """Tests for fit and transform methods."""

    def test_fit_returns_self(self) -> None:
        """Test that fit returns self for chaining."""
        series = pd.Series(["A", "B", "C"])
        encoder = CategoricalEncoder()
        result = encoder.fit(series)
        assert result is encoder

    def test_transform_before_fit_raises(self) -> None:
        """Test that transform before fit raises ValueError."""
        encoder = CategoricalEncoder()
        with pytest.raises(ValueError, match="not been fitted"):
            encoder.transform(pd.Series(["A", "B"]))

    def test_n_categories_property(self) -> None:
        """Test n_categories property."""
        series = pd.Series(["A", "B", "C", "A"])
        encoder = CategoricalEncoder()
        encoder.fit(series)
        assert encoder.n_categories == 3

    def test_categories_property(self) -> None:
        """Test categories property."""
        series = pd.Series(["B", "A", "C"])
        encoder = CategoricalEncoder()
        encoder.fit(series)
        assert encoder.categories == ["A", "B", "C"]  # Sorted


class TestCategoricalEncoderEdgeCases:
    """Tests for edge cases in CategoricalEncoder."""

    def test_handles_nan_values(self) -> None:
        """Test that NaN values are handled (all zeros for one-hot)."""
        series = pd.Series(["A", None, "B", np.nan])
        encoder = CategoricalEncoder(encoding_type="onehot")
        encoded = encoder.fit_transform(series)

        # NaN rows should be all zeros
        assert encoded[1].sum() == 0
        assert encoded[3].sum() == 0

    def test_handles_nan_in_label_encoding(self) -> None:
        """Test that NaN values get -1 in label encoding."""
        series = pd.Series(["A", None, "B"])
        encoder = CategoricalEncoder(encoding_type="label")
        encoded = encoder.fit_transform(series)

        assert encoded[1] == -1

    def test_unknown_category_ignore(self) -> None:
        """Test unknown categories return zeros when ignore."""
        train = pd.Series(["A", "B"])
        test = pd.Series(["A", "C"])  # C is unknown

        encoder = CategoricalEncoder(encoding_type="onehot", handle_unknown="ignore")
        encoder.fit(train)
        encoded = encoder.transform(test)

        # Unknown category should be all zeros
        assert encoded[1].sum() == 0

    def test_unknown_category_error(self) -> None:
        """Test unknown categories raise error when configured."""
        train = pd.Series(["A", "B"])
        test = pd.Series(["A", "C"])  # C is unknown

        encoder = CategoricalEncoder(encoding_type="onehot", handle_unknown="error")
        encoder.fit(train)

        with pytest.raises(ValueError, match="Unknown category"):
            encoder.transform(test)


class TestNumericEncoderInit:
    """Tests for NumericEncoder initialization."""

    def test_init_default_standardize(self) -> None:
        """Test default method is standardize."""
        encoder = NumericEncoder()
        assert encoder.method == "standardize"

    def test_init_minmax(self) -> None:
        """Test initialization with minmax method."""
        encoder = NumericEncoder(method="minmax")
        assert encoder.method == "minmax"


class TestNumericEncoderStandardize:
    """Tests for standardization (z-score) normalization."""

    def test_standardize_values(self) -> None:
        """Test standardization produces correct values."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        encoder = NumericEncoder(method="standardize")
        encoded = encoder.fit_transform(series)

        # Mean should be ~0, std should be ~1
        assert np.abs(encoded.mean()) < 0.01
        assert np.abs(encoded.std() - 1.0) < 0.2  # ddof differences

    def test_standardize_returns_float32(self) -> None:
        """Test standardization returns float32."""
        series = pd.Series([1, 2, 3])
        encoder = NumericEncoder(method="standardize")
        encoded = encoder.fit_transform(series)

        assert encoded.dtype == np.float32


class TestNumericEncoderMinMax:
    """Tests for min-max normalization."""

    def test_minmax_range(self) -> None:
        """Test min-max normalization produces values in [0, 1]."""
        series = pd.Series([1.0, 5.0, 10.0, 100.0])
        encoder = NumericEncoder(method="minmax")
        encoded = encoder.fit_transform(series)

        assert encoded.min() >= 0.0
        assert encoded.max() <= 1.0

    def test_minmax_extreme_values(self) -> None:
        """Test min-max correctly maps min and max."""
        series = pd.Series([10.0, 20.0, 30.0])
        encoder = NumericEncoder(method="minmax")
        encoded = encoder.fit_transform(series)

        assert encoded[0] == pytest.approx(0.0)
        assert encoded[2] == pytest.approx(1.0)


class TestNumericEncoderFitTransform:
    """Tests for fit and transform methods."""

    def test_fit_returns_self(self) -> None:
        """Test that fit returns self for chaining."""
        series = pd.Series([1, 2, 3])
        encoder = NumericEncoder()
        result = encoder.fit(series)
        assert result is encoder

    def test_transform_before_fit_raises(self) -> None:
        """Test that transform before fit raises ValueError."""
        encoder = NumericEncoder()
        with pytest.raises(ValueError, match="not been fitted"):
            encoder.transform(pd.Series([1, 2, 3]))


class TestNumericEncoderEdgeCases:
    """Tests for edge cases in NumericEncoder."""

    def test_handles_nan_values(self) -> None:
        """Test that NaN values are filled with 0."""
        series = pd.Series([1.0, np.nan, 3.0])
        encoder = NumericEncoder(method="minmax")
        encoded = encoder.fit_transform(series)

        # NaN should become 0 before normalization
        assert not np.isnan(encoded).any()

    def test_constant_column_standardize(self) -> None:
        """Test handling of constant columns (std=0) in standardization."""
        series = pd.Series([5.0, 5.0, 5.0])
        encoder = NumericEncoder(method="standardize")
        encoded = encoder.fit_transform(series)

        # Should not produce NaN or inf
        assert not np.isnan(encoded).any()
        assert not np.isinf(encoded).any()

    def test_constant_column_minmax(self) -> None:
        """Test handling of constant columns (max=min) in min-max."""
        series = pd.Series([5.0, 5.0, 5.0])
        encoder = NumericEncoder(method="minmax")
        encoded = encoder.fit_transform(series)

        # Should not produce NaN or inf
        assert not np.isnan(encoded).any()
        assert not np.isinf(encoded).any()

    def test_handles_non_numeric_strings(self) -> None:
        """Test that non-numeric strings are converted (become NaN then 0)."""
        series = pd.Series([1, "not a number", 3])
        encoder = NumericEncoder(method="minmax")
        encoded = encoder.fit_transform(series)

        # Should complete without error
        assert len(encoded) == 3
        assert not np.isnan(encoded).any()

    def test_invalid_method_raises(self) -> None:
        """Test that invalid method raises ValueError."""
        series = pd.Series([1, 2, 3])
        encoder = NumericEncoder(method="invalid")

        with pytest.raises(ValueError, match="Unknown normalization method"):
            encoder.fit(series)


# ============================================================================
# Additional Edge Case Tests for Coverage
# ============================================================================


class TestCategoricalEncoderCoverageEdgeCases:
    """Additional edge case tests for CategoricalEncoder coverage."""

    def test_label_encoding_unknown_error(self) -> None:
        """Lines 112-113: Label encoding with handle_unknown='error' raises."""
        encoder = CategoricalEncoder(encoding_type="label", handle_unknown="error")
        encoder.fit(pd.Series(["A", "B"]))
        with pytest.raises(ValueError, match="Unknown category"):
            encoder.transform(pd.Series(["A", "C"]))

    def test_label_encoding_unknown_ignore(self) -> None:
        """Lines 114-115: Label encoding with unknown category returns -1."""
        encoder = CategoricalEncoder(encoding_type="label", handle_unknown="ignore")
        encoder.fit(pd.Series(["A", "B"]))
        result = encoder.transform(pd.Series(["A", "C"]))
        assert result[1] == -1  # Unknown category gets -1

    def test_invalid_encoding_type_transform(self) -> None:
        """Line 120: Invalid encoding_type in transform raises ValueError."""
        encoder = CategoricalEncoder(encoding_type="onehot")
        encoder.fit(pd.Series(["A", "B"]))
        # Manually change encoding type to test the transform error path
        encoder.encoding_type = "invalid"
        with pytest.raises(ValueError, match="Unknown encoding type"):
            encoder.transform(pd.Series(["A"]))

    def test_n_categories_not_fitted(self) -> None:
        """Line 137: n_categories property before fit raises ValueError."""
        encoder = CategoricalEncoder()
        with pytest.raises(ValueError, match="not been fitted"):
            _ = encoder.n_categories

    def test_categories_not_fitted(self) -> None:
        """Line 144: categories property before fit raises ValueError."""
        encoder = CategoricalEncoder()
        with pytest.raises(ValueError, match="not been fitted"):
            _ = encoder.categories


class TestNumericEncoderCoverageEdgeCases:
    """Additional edge case tests for NumericEncoder coverage."""

    def test_minmax_transform_constant_values(self) -> None:
        """Line 241: minmax transform with constant values (range=0) returns zeros."""
        # Fit on variable data
        encoder = NumericEncoder(method="minmax")
        encoder.fit(pd.Series([1.0, 5.0, 10.0]))
        # Transform with constant values - should handle division by zero
        # This tests the range_val == 0 path in transform
        encoder._min = 5.0
        encoder._max = 5.0  # Simulate constant column scenario
        result = encoder.transform(pd.Series([5.0, 5.0, 5.0]))
        assert not np.isnan(result).any()
        assert (result == 0).all()

    def test_invalid_method_in_transform(self) -> None:
        """Line 246: Invalid method in transform raises ValueError."""
        encoder = NumericEncoder(method="standardize")
        encoder.fit(pd.Series([1, 2, 3]))
        # Manually change method to test the transform error path
        encoder.method = "invalid"
        with pytest.raises(ValueError, match="Unknown normalization method"):
            encoder.transform(pd.Series([1, 2, 3]))
