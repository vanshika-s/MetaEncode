# src/visualization/plots.py
"""Dimensionality reduction and interactive plotting.

This module provides functionality for:
- Reducing high-dimensional embeddings to 2D using UMAP or PCA
- Creating interactive scatter plots with Plotly
- Generating hover tooltips with dataset metadata
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Diverging colorscale for similarity scores (more intermediate stops)
SIMILARITY_COLORSCALE = [
    [0.0, "rgb(179, 88, 6)"],     # Dark orange (lowest)
    [0.15, "rgb(224, 130, 20)"],  # Orange
    [0.3, "rgb(253, 184, 99)"],   # Light orange
    [0.45, "rgb(254, 224, 182)"], # Pale orange
    [0.5, "rgb(247, 247, 247)"],  # White (mid)
    [0.55, "rgb(216, 218, 235)"], # Pale purple
    [0.7, "rgb(158, 154, 200)"],  # Light purple
    [0.85, "rgb(106, 81, 163)"],  # Purple
    [1.0, "rgb(63, 0, 125)"],     # Dark purple (highest)
]


class DimensionalityReducer:
    """Reduce high-dimensional embeddings to 2D for visualization.

    Supports UMAP, PCA, and t-SNE for dimensionality reduction. UMAP generally
    preserves local structure better, while PCA is faster and deterministic.
    t-SNE often produces better local structure than UMAP for visualization.

    Note: Filtering (outlier removal) should be performed upstream before
    passing embeddings to this class. This ensures metadata stays synchronized.

    Example:
        >>> reducer = DimensionalityReducer(method="umap")
        >>> coords_2d = reducer.fit_transform(embeddings)
    """

    SUPPORTED_METHODS = ("umap", "pca", "t-sne")

    def __init__(
        self,
        method: str = "umap",
        n_components: int = 2,
        random_state: int = 42,
    ) -> None:
        """Initialize the dimensionality reducer.

        Args:
            method: Reduction method ("umap", "pca", or "t-sne").
            n_components: Number of dimensions (default 2 for plotting).
            random_state: Random seed for reproducibility.
        """
        self.method = method.lower()
        self.n_components = n_components
        self.random_state = random_state
        self._reducer: Optional[Any] = None
        self._fitted = False
        self.variance_ratio_: Optional[tuple[float, ...]] = None

    def _create_reducer(self, n_samples: int) -> None:
        """Initialize the reducer based on method.

        Args:
            n_samples: Number of samples (used to adjust UMAP n_neighbors
                and t-SNE perplexity for small datasets).
        """
        if self.method == "umap":
            import umap

            n_neighbors = max(2, min(15, n_samples - 1))
            self._reducer = umap.UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="cosine",
            )
        elif self.method == "pca":
            self._reducer = PCA(
                n_components=self.n_components,
                random_state=self.random_state,
            )
        elif self.method == "t-sne":
            # Perplexity must be less than n_samples; sklearn default is 30
            # Use similar adaptive logic as UMAP's n_neighbors
            perplexity = min(30, max(5, n_samples - 1))
            if n_samples <= 5:
                perplexity = max(2, n_samples - 1)
            self._reducer = TSNE(
                n_components=self.n_components,
                random_state=self.random_state,
                perplexity=perplexity,
            )
        else:
            raise ValueError(
                f"Unknown method: {self.method}. "
                f"Use one of: {', '.join(self.SUPPORTED_METHODS)}."
            )

    def fit(self, embeddings: np.ndarray) -> "DimensionalityReducer":
        """Fit the reducer to the embeddings.

        Args:
            embeddings: NumPy array of shape (n_samples, n_features).

        Returns:
            Self for method chaining.

        Note:
            For t-SNE, fit() is supported but transform() is not available.
            Use fit_transform() for t-SNE workflows.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self._create_reducer(len(embeddings))
        assert self._reducer is not None  # Guaranteed by _create_reducer
        self._reducer.fit(embeddings)
        self._fitted = True
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to lower dimensions.

        Args:
            embeddings: NumPy array of shape (n_samples, n_features).

        Returns:
            NumPy array of shape (n_samples, n_components).

        Raises:
            ValueError: If reducer not fitted or if t-SNE (which lacks transform).
        """
        if not self._fitted or self._reducer is None:
            raise ValueError("Reducer has not been fitted. Call fit() first.")

        if self.method == "t-sne":
            raise ValueError(
                "t-SNE does not support transform() for out-of-sample data. "
                "Use fit_transform() instead, or consider UMAP for workflows "
                "requiring separate fit/transform."
            )

        embeddings = np.asarray(embeddings, dtype=np.float32)
        result: np.ndarray = self._reducer.transform(embeddings)
        return result.astype(np.float32)

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            embeddings: NumPy array of shape (n_samples, n_features).

        Returns:
            NumPy array of shape (n_samples, n_components).
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self._create_reducer(len(embeddings))
        assert self._reducer is not None  # Guaranteed by _create_reducer
        self._fitted = True

        result: np.ndarray = self._reducer.fit_transform(embeddings)

        # Store variance ratio for PCA (useful for axis labels)
        if self.method == "pca":
            self.variance_ratio_ = tuple(
                self._reducer.explained_variance_ratio_[: self.n_components]
            )
        else:
            self.variance_ratio_ = None

        return result.astype(np.float32)


class PlotGenerator:
    """Generate interactive plots for dataset visualization.

    Creates Plotly scatter plots with dataset metadata displayed in
    hover tooltips. Supports coloring by categorical fields like
    organism or assay type.

    Example:
        >>> plotter = PlotGenerator()
        >>> fig = plotter.scatter_plot(coords_2d, metadata_df, color_by="organism")
        >>> fig.show()
    """

    # Default columns for hover tooltips
    DEFAULT_HOVER_COLS = [
        "accession",
        "description",
        "assay_term_name",
        "organism",
        "organ",
    ]

    def __init__(self, reduction_method: str = "umap") -> None:
        """Initialize the plot generator.

        Args:
            reduction_method: Method used for reduction (for axis labels).
        """
        self.reduction_method = reduction_method

    def scatter_plot(
        self,
        coords: np.ndarray,
        metadata: pd.DataFrame,
        color_by: Optional[str] = None,
        title: str = "Dataset Embeddings",
        highlight_indices: Optional[list[int]] = None,
        variance_ratio: Optional[tuple[float, float]] = None,
    ) -> go.Figure:
        """Create interactive scatter plot of dataset embeddings.

        Args:
            coords: 2D coordinates from dimensionality reduction (n_samples, 2).
            metadata: DataFrame with dataset metadata for tooltips.
            color_by: Column name to color points by (categorical or continuous).
            title: Plot title.
            highlight_indices: Indices of points to highlight with markers.
            variance_ratio: Tuple of (PC1_variance, PC2_variance) for PCA axis labels.

        Returns:
            Plotly Figure object.
        """
        # Create plot DataFrame
        plot_df = metadata.copy()
        plot_df["x"] = coords[:, 0]
        plot_df["y"] = coords[:, 1]

        # Truncate long descriptions for hover
        if "description" in plot_df.columns:
            plot_df["description_short"] = plot_df["description"].apply(
                lambda x: (str(x)[:100] + "...") if len(str(x)) > 100 else str(x)
            )

        # Build customdata for hovertemplate (no x/y values shown)
        customdata_cols = ["accession", "description_short", "assay_term_name", "organism", "organ"]

        # Fill missing columns with empty strings
        for col in customdata_cols:
            if col not in plot_df.columns:
                plot_df[col] = ""

        # Build hovertemplate (excludes x/y coordinates and non-clickable URL)
        # Users can click accession IDs in tables to access ENCODE portal
        hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}<br>"
            "<b>Assay:</b> %{customdata[2]}<br>"
            "<b>Organism:</b> %{customdata[3]}<br>"
            "<b>Organ:</b> %{customdata[4]}"
            "<extra></extra>"
        )

        # Create scatter plot with appropriate colorscale
        is_similarity = color_by == "similarity_score" and color_by in plot_df.columns

        if color_by and color_by in plot_df.columns:
            if is_similarity:
                # Use orange-to-blue colorscale for similarity scores
                fig = px.scatter(
                    plot_df,
                    x="x",
                    y="y",
                    color=color_by,
                    color_continuous_scale=SIMILARITY_COLORSCALE,
                    title=title,
                )
            else:
                fig = px.scatter(
                    plot_df,
                    x="x",
                    y="y",
                    color=color_by,
                    title=title,
                )
        else:
            fig = px.scatter(
                plot_df,
                x="x",
                y="y",
                title=title,
            )

        # Apply customdata and hovertemplate to all traces
        customdata_array = plot_df[customdata_cols].values
        for trace in fig.data:
            if hasattr(trace, "customdata"):
                trace.customdata = customdata_array
                trace.hovertemplate = hovertemplate

        # Add highlight markers if specified
        if highlight_indices:
            highlight_df = plot_df.iloc[highlight_indices]
            fig.add_trace(
                go.Scatter(
                    x=highlight_df["x"],
                    y=highlight_df["y"],
                    mode="markers",
                    marker=dict(
                        size=15,
                        color="red",
                        symbol="star",
                        line=dict(width=2, color="black"),
                    ),
                    name="Selected",
                    hoverinfo="skip",
                )
            )

        # Update axis labels based on reduction method and variance
        method_lower = self.reduction_method.lower()
        if method_lower == "umap":
            x_title = "UMAP-1"
            y_title = "UMAP-2"
        elif method_lower == "t-sne":
            x_title = "t-SNE-1"
            y_title = "t-SNE-2"
        else:
            # PCA: include variance percentages if available
            if variance_ratio is not None and len(variance_ratio) >= 2:
                x_title = f"PC-1 ({variance_ratio[0]:.1%} variance)"
                y_title = f"PC-2 ({variance_ratio[1]:.1%} variance)"
            else:
                x_title = "PC-1"
                y_title = "PC-2"

        fig.update_layout(
            xaxis_title=x_title,
            yaxis_title=y_title,
            legend_title=color_by if color_by else "Dataset",
            hovermode="closest",
        )

        return fig

    def similarity_heatmap(
        self,
        similarity_matrix: np.ndarray,
        labels: list[str],
        title: str = "Dataset Similarity Matrix",
    ) -> go.Figure:
        """Create heatmap of dataset similarities.

        Args:
            similarity_matrix: Square matrix of pairwise similarities.
            labels: Labels for each dataset.
            title: Plot title.

        Returns:
            Plotly Figure object.
        """
        fig = go.Figure(
            data=go.Heatmap(
                z=similarity_matrix,
                x=labels,
                y=labels,
                colorscale=SIMILARITY_COLORSCALE,
                colorbar=dict(title="Similarity"),
                hoverongaps=False,
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Dataset",
            yaxis_title="Dataset",
            xaxis=dict(tickangle=45),
        )

        return fig



def percentile_range_filtering(
    embeddings: np.ndarray,
    lower: float = 5.0,
    upper: float = 95.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter embeddings to central percentile range.

    Removes samples with any feature outside the specified percentile bounds.
    Returns both filtered embeddings and a boolean mask for synchronizing
    with associated metadata.

    Args:
        embeddings: Input array of shape (n_samples, n_features).
        lower: Lower percentile bound (default 5).
        upper: Upper percentile bound (default 95).

    Returns:
        Tuple of (filtered_embeddings, boolean_mask) where mask indicates
        which rows were kept.
    """
    lower_bound = np.percentile(embeddings, lower, axis=0)
    upper_bound = np.percentile(embeddings, upper, axis=0)

    in_range = (embeddings >= lower_bound) & (embeddings <= upper_bound)
    mask: np.ndarray = np.asarray(in_range.all(axis=1))

    return embeddings[mask], mask