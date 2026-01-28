# src/ml/feature_combiner.py
"""Feature combination for multi-modal similarity computation.

This module provides the FeatureCombiner class that orchestrates weighted
combination of text embeddings with categorical and numeric features,
following the architecture spec for combined vectors (~437 dimensions).
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.processing.encoders import CategoricalEncoder, NumericEncoder


class FeatureCombiner:
    """Combine text embeddings with categorical and numeric features.

    Orchestrates the full feature engineering pipeline:
    1. Text embeddings from EmbeddingGenerator (external, passed in)
    2. Categorical encodings via CategoricalEncoder (one-hot)
    3. Numeric normalization via NumericEncoder (minmax)
    4. Weighted concatenation into final combined vector

    The weight application uses sqrt(weight) scaling so that the dot product
    contribution (and thus cosine similarity) is proportional to the weight.

    Example:
        >>> combiner = FeatureCombiner()
        >>> combiner.fit(df)
        >>> text_embeddings = embedder.encode(df["combined_text"].tolist())
        >>> combined = combiner.transform(df, text_embeddings)
        >>> # combined.shape = (n_samples, ~437)
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "text_embedding": 0.5,
        "assay_type": 0.2,
        "organism": 0.15,
        "cell_type": 0.1,
        "lab": 0.03,
        "numeric_features": 0.02,
    }

    CATEGORICAL_COLUMNS: list[str] = [
        "assay_term_name",
        "organism",
        "biosample_term_name",
        "lab",
    ]

    NUMERIC_COLUMNS: list[str] = [
        "replicate_count",
        "file_count",
    ]

    # Map weight keys to column names
    WEIGHT_TO_COLUMN: dict[str, str] = {
        "assay_type": "assay_term_name",
        "organism": "organism",
        "cell_type": "biosample_term_name",
        "lab": "lab",
    }

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        text_column: str = "combined_text",
        categorical_encoding: str = "onehot",
        numeric_method: str = "minmax",
    ) -> None:
        """Initialize the feature combiner.

        Args:
            weights: Custom weights for each feature group. If None, uses
                    DEFAULT_WEIGHTS. Keys should match DEFAULT_WEIGHTS keys.
            text_column: Name of the column containing combined text for
                        embedding generation (used for reference, not stored).
            categorical_encoding: Encoding type for categorical features
                                 ("onehot" or "label").
            numeric_method: Normalization method for numeric features
                           ("standardize" or "minmax").
        """
        self.weights = weights if weights is not None else self.DEFAULT_WEIGHTS.copy()
        self.text_column = text_column
        self.categorical_encoding = categorical_encoding
        self.numeric_method = numeric_method

        # Encoders will be created during fit
        self._categorical_encoders: dict[str, CategoricalEncoder] = {}
        self._numeric_encoders: dict[str, NumericEncoder] = {}

        # Track dimensions after fitting
        self._text_dim: Optional[int] = None
        self._categorical_dims: dict[str, int] = {}
        self._numeric_dim: int = 0
        self._fitted = False

    def fit(self, df: pd.DataFrame, text_embedding_dim: int = 384) -> "FeatureCombiner":
        """Fit encoders to the data.

        Creates and fits CategoricalEncoder for each categorical column and
        NumericEncoder for each numeric column. Does NOT generate text
        embeddings (they are passed in during transform).

        Args:
            df: DataFrame with metadata columns.
            text_embedding_dim: Dimension of text embeddings (default 384 for
                               MiniLM). Used for dimension tracking.

        Returns:
            Self for method chaining.
        """
        self._text_dim = text_embedding_dim

        # Fit categorical encoders
        for col in self.CATEGORICAL_COLUMNS:
            if col in df.columns:
                cat_encoder = CategoricalEncoder(
                    encoding_type=self.categorical_encoding,
                    handle_unknown="ignore",
                )
                cat_encoder.fit(df[col])
                self._categorical_encoders[col] = cat_encoder
                self._categorical_dims[col] = cat_encoder.n_categories

        # Fit numeric encoders
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                num_encoder = NumericEncoder(method=self.numeric_method)
                num_encoder.fit(df[col])
                self._numeric_encoders[col] = num_encoder

        self._numeric_dim = len(self._numeric_encoders)
        self._fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        text_embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transform DataFrame to combined feature vectors.

        Args:
            df: DataFrame with metadata columns.
            text_embeddings: Pre-computed text embeddings, shape (n_samples, dim).
                            If None, text features are not included.

        Returns:
            Combined feature matrix, shape (n_samples, combined_dim).

        Raises:
            ValueError: If combiner has not been fitted.
        """
        if not self._fitted:
            raise ValueError("FeatureCombiner has not been fitted. Call fit() first.")

        n_samples = len(df)
        segments: list[np.ndarray] = []

        # Text embeddings (if provided)
        if text_embeddings is not None:
            text_emb = np.asarray(text_embeddings, dtype=np.float32)
            if len(text_emb) != n_samples:
                raise ValueError(
                    f"text_embeddings length ({len(text_emb)}) "
                    f"doesn't match DataFrame ({n_samples})"
                )
            text_weight = self.weights.get("text_embedding", 0.5)
            weighted_text = text_emb * np.sqrt(text_weight)
            segments.append(weighted_text)

        # Categorical features
        for weight_key, col in self.WEIGHT_TO_COLUMN.items():
            if col in self._categorical_encoders:
                cat_enc = self._categorical_encoders[col]
                encoded = cat_enc.transform(df[col])
                weight = self.weights.get(weight_key, 0.1)
                weighted_encoded = encoded * np.sqrt(weight)
                segments.append(weighted_encoded)

        # Numeric features (combined into single weighted segment)
        if self._numeric_encoders:
            numeric_vectors = []
            for col in self.NUMERIC_COLUMNS:
                if col in self._numeric_encoders:
                    num_enc = self._numeric_encoders[col]
                    normalized = num_enc.transform(df[col])
                    # Reshape to (n_samples, 1) for concatenation
                    numeric_vectors.append(normalized.reshape(-1, 1))

            if numeric_vectors:
                numeric_combined = np.hstack(numeric_vectors)
                numeric_weight = self.weights.get("numeric_features", 0.02)
                weighted_numeric = numeric_combined * np.sqrt(numeric_weight)
                segments.append(weighted_numeric)

        if not segments:
            raise ValueError("No features to combine. Check data and configuration.")

        # Concatenate all segments
        combined = np.hstack(segments)
        return combined.astype(np.float32)

    def fit_transform(
        self,
        df: pd.DataFrame,
        text_embeddings: Optional[np.ndarray] = None,
        text_embedding_dim: int = 384,
    ) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            df: DataFrame with metadata columns.
            text_embeddings: Pre-computed text embeddings.
            text_embedding_dim: Dimension of text embeddings.

        Returns:
            Combined feature matrix.
        """
        self.fit(df, text_embedding_dim=text_embedding_dim)
        return self.transform(df, text_embeddings)

    def transform_single(
        self,
        record: dict,
        text_embedding: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transform a single record to a combined feature vector.

        Useful for query-time transformation of a selected dataset.

        Args:
            record: Dictionary with metadata fields.
            text_embedding: Pre-computed text embedding, shape (dim,).

        Returns:
            Combined feature vector, shape (combined_dim,).

        Raises:
            ValueError: If combiner has not been fitted.
        """
        if not self._fitted:
            raise ValueError("FeatureCombiner has not been fitted. Call fit() first.")

        # Convert single record to DataFrame for consistent processing
        df = pd.DataFrame([record])

        # Handle text embedding shape
        if text_embedding is not None:
            text_emb = np.asarray(text_embedding, dtype=np.float32)
            # Ensure 2D shape (1, dim)
            if text_emb.ndim == 1:
                text_emb = text_emb.reshape(1, -1)
        else:
            text_emb = None

        # Use transform which handles single-row DataFrames
        combined = self.transform(df, text_emb)

        # Return as 1D vector
        return np.asarray(combined[0], dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        """Return the total dimension of combined features.

        Raises:
            ValueError: If combiner has not been fitted.
        """
        if not self._fitted:
            raise ValueError("FeatureCombiner has not been fitted. Call fit() first.")

        total = 0

        # Text dimension
        if self._text_dim is not None:
            total += self._text_dim

        # Categorical dimensions
        for dim in self._categorical_dims.values():
            total += dim

        # Numeric dimension
        total += self._numeric_dim

        return total

    def get_feature_breakdown(self) -> dict[str, int]:
        """Return dimensions of each feature group.

        Returns:
            Dictionary mapping feature group names to their dimensions.

        Raises:
            ValueError: If combiner has not been fitted.
        """
        if not self._fitted:
            raise ValueError("FeatureCombiner has not been fitted. Call fit() first.")

        breakdown: dict[str, int] = {}

        # Text
        if self._text_dim is not None:
            breakdown["text_embedding"] = self._text_dim

        # Categorical
        for col, dim in self._categorical_dims.items():
            breakdown[col] = dim

        # Numeric
        breakdown["numeric_features"] = self._numeric_dim

        return breakdown

    @property
    def is_fitted(self) -> bool:
        """Return whether the combiner has been fitted."""
        return self._fitted

    def get_weights(self) -> dict[str, float]:
        """Return the current weight configuration."""
        return self.weights.copy()

    def set_weights(self, weights: dict[str, float]) -> None:
        """Update the weight configuration.

        Can be called after fitting to change weights without re-fitting.

        Args:
            weights: New weight dictionary.
        """
        self.weights = weights.copy()
