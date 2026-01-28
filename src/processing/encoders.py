# src/processing/encoders.py
"""Categorical and numeric field encoding utilities.

This module provides encoders for transforming categorical and numeric
metadata fields into vector representations suitable for similarity computation.
"""

from typing import Optional

import numpy as np
import pandas as pd


class CategoricalEncoder:
    """Encode categorical metadata fields.

    Supports one-hot encoding and label encoding for categorical fields
    like organism, assay type, and biosample.

    Example:
        >>> encoder = CategoricalEncoder()
        >>> encoder.fit(df["assay_term_name"])
        >>> encoded = encoder.transform(df["assay_term_name"])
    """

    def __init__(
        self, encoding_type: str = "onehot", handle_unknown: str = "ignore"
    ) -> None:
        """Initialize the categorical encoder.

        Args:
            encoding_type: Type of encoding ("onehot" or "label").
            handle_unknown: How to handle unknown categories during transform
                           ("ignore" returns zeros, "error" raises ValueError).
        """
        self.encoding_type = encoding_type
        self.handle_unknown = handle_unknown
        self._categories: Optional[list] = None
        self._category_to_idx: Optional[dict] = None
        self._fitted = False

    def fit(self, series: pd.Series) -> "CategoricalEncoder":
        """Fit the encoder to the data.

        Learns unique categories from the data.

        Args:
            series: Pandas Series containing categorical values.

        Returns:
            Self for method chaining.
        """
        # Get unique categories, excluding None/NaN
        values = series.dropna().unique()
        self._categories = sorted([str(v) for v in values])
        self._category_to_idx = {cat: idx for idx, cat in enumerate(self._categories)}
        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        """Transform categorical values to encoded vectors.

        Args:
            series: Pandas Series containing categorical values.

        Returns:
            NumPy array with encoded values.
            - For one-hot: shape (n_samples, n_categories)
            - For label: shape (n_samples,)

        Raises:
            ValueError: If encoder has not been fitted.
        """
        if (
            not self._fitted
            or self._categories is None
            or self._category_to_idx is None
        ):
            raise ValueError("Encoder has not been fitted. Call fit() first.")

        n_samples = len(series)
        n_categories = len(self._categories)
        category_to_idx = self._category_to_idx

        if self.encoding_type == "onehot":
            # Create one-hot encoded matrix
            onehot_encoded = np.zeros((n_samples, n_categories), dtype=np.float32)

            for i, value in enumerate(series):
                if pd.isna(value):
                    continue  # All zeros for missing values
                str_value = str(value)
                if str_value in category_to_idx:
                    onehot_encoded[i, category_to_idx[str_value]] = 1.0
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unknown category: {str_value}")
                # If handle_unknown == "ignore", the row remains all zeros

            return onehot_encoded

        elif self.encoding_type == "label":
            # Create label encoded array
            label_encoded = np.zeros(n_samples, dtype=np.int32)

            for i, value in enumerate(series):
                if pd.isna(value):
                    label_encoded[i] = -1  # -1 for missing values
                else:
                    str_value = str(value)
                    if str_value in category_to_idx:
                        label_encoded[i] = category_to_idx[str_value]
                    elif self.handle_unknown == "error":
                        raise ValueError(f"Unknown category: {str_value}")
                    else:
                        label_encoded[i] = -1  # -1 for unknown categories

            return label_encoded

        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

    def fit_transform(self, series: pd.Series) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            series: Pandas Series containing categorical values.

        Returns:
            NumPy array with encoded values.
        """
        return self.fit(series).transform(series)

    @property
    def n_categories(self) -> int:
        """Return the number of categories."""
        if not self._fitted or self._categories is None:
            raise ValueError("Encoder has not been fitted.")
        return len(self._categories)

    @property
    def categories(self) -> list:
        """Return the list of categories."""
        if not self._fitted or self._categories is None:
            raise ValueError("Encoder has not been fitted.")
        return self._categories.copy()


class NumericEncoder:
    """Normalize and encode numeric metadata fields.

    Supports standardization (z-score) and min-max normalization
    for numeric fields like replicate count and file count.

    Example:
        >>> encoder = NumericEncoder(method="standardize")
        >>> encoder.fit(df["replicate_count"])
        >>> normalized = encoder.transform(df["replicate_count"])
    """

    def __init__(self, method: str = "standardize") -> None:
        """Initialize the numeric encoder.

        Args:
            method: Normalization method ("standardize" or "minmax").
        """
        self.method = method
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._min: Optional[float] = None
        self._max: Optional[float] = None
        self._fitted = False

    def fit(self, series: pd.Series) -> "NumericEncoder":
        """Fit the encoder to the data.

        Computes statistics needed for normalization.

        Args:
            series: Pandas Series containing numeric values.

        Returns:
            Self for method chaining.
        """
        # Convert to numeric, handling any non-numeric values
        numeric_series = pd.to_numeric(series, errors="coerce")

        if self.method == "standardize":
            self._mean = float(numeric_series.mean())
            self._std = float(numeric_series.std())
            # Avoid division by zero
            if self._std == 0 or pd.isna(self._std):
                self._std = 1.0

        elif self.method == "minmax":
            self._min = float(numeric_series.min())
            self._max = float(numeric_series.max())
            # Avoid division by zero
            if self._max == self._min or pd.isna(self._max) or pd.isna(self._min):
                self._min = 0.0
                self._max = 1.0

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        self._fitted = True
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        """Transform numeric values to normalized vectors.

        Args:
            series: Pandas Series containing numeric values.

        Returns:
            NumPy array with normalized values, shape (n_samples,).

        Raises:
            ValueError: If encoder has not been fitted.
        """
        if not self._fitted:
            raise ValueError("Encoder has not been fitted. Call fit() first.")

        # Convert to numeric
        numeric_series = pd.to_numeric(series, errors="coerce")

        # Fill NaN with 0 (or could use mean for standardize)
        numeric_values: np.ndarray = np.asarray(
            numeric_series.fillna(0), dtype=np.float32
        )

        normalized: np.ndarray
        if self.method == "standardize":
            # Z-score normalization: (x - mean) / std
            mean = self._mean if self._mean is not None else 0.0
            std = self._std if self._std is not None else 1.0
            normalized = (numeric_values - mean) / std

        elif self.method == "minmax":
            # Min-max normalization: (x - min) / (max - min)
            min_val = self._min if self._min is not None else 0.0
            max_val = self._max if self._max is not None else 1.0
            range_val = max_val - min_val
            if range_val == 0:
                normalized = np.zeros_like(numeric_values)
            else:
                normalized = (numeric_values - min_val) / range_val

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        return np.asarray(normalized, dtype=np.float32)

    def fit_transform(self, series: pd.Series) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            series: Pandas Series containing numeric values.

        Returns:
            NumPy array with normalized values.
        """
        return self.fit(series).transform(series)
