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


class DimensionalityReducer:
    """Reduce high-dimensional embeddings to 2D for visualization.

    Supports UMAP and PCA for dimensionality reduction. UMAP generally
    preserves local structure better, while PCA is faster and deterministic.

    Example:
        >>> reducer = DimensionalityReducer(method="umap")
        >>> coords_2d = reducer.fit_transform(embeddings)
    """

    def __init__(
        self,
        method: str = "umap",
        n_components: int = 2,
        random_state: int = 42,
    ) -> None:
        """Initialize the dimensionality reducer.

        Args:
            method: Reduction method ("umap" or "pca").
            n_components: Number of dimensions (default 2 for plotting).
            random_state: Random seed for reproducibility.
        """
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self._reducer: Optional[Any] = None
        self._fitted = False

    def fit(self, embeddings: np.ndarray) -> "DimensionalityReducer":
        """Fit the reducer to the embeddings.

        Args:
            embeddings: NumPy array of shape (n_samples, n_features).

        Returns:
            Self for method chaining.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if self.method == "umap":
            import umap

            # Adjust n_neighbors based on sample size
            n_neighbors = min(15, len(embeddings) - 1)
            n_neighbors = max(2, n_neighbors)  # At least 2

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

        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'umap' or 'pca'.")

        self._reducer.fit(embeddings)
        self._fitted = True
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to lower dimensions.

        Args:
            embeddings: NumPy array of shape (n_samples, n_features).

        Returns:
            NumPy array of shape (n_samples, n_components).
        """
        if not self._fitted or self._reducer is None:
            raise ValueError("Reducer has not been fitted. Call fit() first.")

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

        if self.method == "umap":
            import umap

            n_neighbors = min(15, len(embeddings) - 1)
            n_neighbors = max(2, n_neighbors)

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

        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'umap' or 'pca'.")

        self._fitted = True
        result: np.ndarray = self._reducer.fit_transform(embeddings)
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
    ) -> go.Figure:
        """Create interactive scatter plot of dataset embeddings.

        Args:
            coords: 2D coordinates from dimensionality reduction (n_samples, 2).
            metadata: DataFrame with dataset metadata for tooltips.
            color_by: Column name to color points by (categorical).
            title: Plot title.
            highlight_indices: Indices of points to highlight with markers.

        Returns:
            Plotly Figure object.
        """
        # Create plot DataFrame
        plot_df = metadata.copy()
        plot_df["x"] = coords[:, 0]
        plot_df["y"] = coords[:, 1]

        # Determine hover columns
        hover_cols = [c for c in self.DEFAULT_HOVER_COLS if c in plot_df.columns]

        # Truncate long descriptions for hover
        if "description" in plot_df.columns:
            plot_df["description_short"] = plot_df["description"].apply(
                lambda x: (str(x)[:100] + "...") if len(str(x)) > 100 else str(x)
            )
            hover_cols = [
                "description_short" if c == "description" else c for c in hover_cols
            ]

        # Create scatter plot
        if color_by and color_by in plot_df.columns:
            fig = px.scatter(
                plot_df,
                x="x",
                y="y",
                color=color_by,
                hover_data=hover_cols,
                title=title,
            )
        else:
            fig = px.scatter(
                plot_df,
                x="x",
                y="y",
                hover_data=hover_cols,
                title=title,
            )

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

        # Update axis labels
        axis_prefix = "UMAP" if self.reduction_method == "umap" else "PC"
        fig.update_layout(
            xaxis_title=f"{axis_prefix}-1",
            yaxis_title=f"{axis_prefix}-2",
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
                colorscale="Viridis",
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
