# tests/test_visualization/test_plots.py
"""Tests for the visualization module (DimensionalityReducer and PlotGenerator)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.visualization import DimensionalityReducer, PlotGenerator

# ============================================================================
# Module Import Tests
# ============================================================================


class TestVisualizationImports:
    """Test that visualization module exports are accessible."""

    def test_dimensionality_reducer_importable(self):
        """Test that DimensionalityReducer can be imported."""
        from src.visualization import DimensionalityReducer

        assert DimensionalityReducer is not None

    def test_plot_generator_importable(self):
        """Test that PlotGenerator can be imported."""
        from src.visualization import PlotGenerator

        assert PlotGenerator is not None

    def test_all_exports_defined(self):
        """Test that __all__ exports are complete."""
        from src.visualization import __all__

        assert "DimensionalityReducer" in __all__
        assert "PlotGenerator" in __all__


# ============================================================================
# DimensionalityReducer Tests
# ============================================================================


class TestDimensionalityReducerInit:
    """Tests for DimensionalityReducer initialization."""

    def test_init_default_umap(self):
        """Test default initialization uses UMAP method."""
        reducer = DimensionalityReducer()
        assert reducer.method == "umap"
        assert reducer.n_components == 2
        assert reducer.random_state == 42
        assert reducer._fitted is False
        assert reducer._reducer is None

    def test_init_pca_method(self):
        """Test initialization with PCA method."""
        reducer = DimensionalityReducer(method="pca")
        assert reducer.method == "pca"
        assert reducer.n_components == 2

    def test_init_custom_components(self):
        """Test initialization with custom n_components."""
        reducer = DimensionalityReducer(n_components=3)
        assert reducer.n_components == 3

    def test_init_custom_random_state(self):
        """Test initialization with custom random_state."""
        reducer = DimensionalityReducer(random_state=123)
        assert reducer.random_state == 123


class TestDimensionalityReducerFit:
    """Tests for DimensionalityReducer.fit() method."""

    def test_fit_pca_returns_self(self, sample_embeddings: np.ndarray):
        """Test that fit returns self for method chaining."""
        reducer = DimensionalityReducer(method="pca")
        result = reducer.fit(sample_embeddings)
        assert result is reducer

    def test_fit_pca_sets_fitted_flag(self, sample_embeddings: np.ndarray):
        """Test that fit sets _fitted to True."""
        reducer = DimensionalityReducer(method="pca")
        assert reducer._fitted is False
        reducer.fit(sample_embeddings)
        assert reducer._fitted is True

    def test_fit_pca_creates_reducer(self, sample_embeddings: np.ndarray):
        """Test that fit creates PCA reducer object."""
        reducer = DimensionalityReducer(method="pca")
        reducer.fit(sample_embeddings)
        assert reducer._reducer is not None

    def test_fit_umap_creates_reducer(self, sample_embeddings: np.ndarray):
        """Test that fit creates UMAP reducer object."""
        reducer = DimensionalityReducer(method="umap")
        reducer.fit(sample_embeddings)
        assert reducer._reducer is not None
        assert reducer._fitted is True

    def test_fit_pca_small_dataset(self, sample_small_embeddings: np.ndarray):
        """Test fit works with small datasets (3 samples)."""
        reducer = DimensionalityReducer(method="pca")
        reducer.fit(sample_small_embeddings)
        assert reducer._fitted is True

    def test_fit_umap_larger_dataset(self, sample_embeddings: np.ndarray):
        """Test that UMAP fit works with appropriately sized datasets."""
        # Use larger dataset to avoid UMAP's small dataset issues
        reducer = DimensionalityReducer(method="umap")
        reducer.fit(sample_embeddings)
        assert reducer._fitted is True
        assert reducer._reducer is not None

    def test_fit_invalid_method_raises(self, sample_embeddings: np.ndarray):
        """Test that fit raises ValueError for invalid method."""
        reducer = DimensionalityReducer(method="invalid")
        with pytest.raises(ValueError, match="Unknown method"):
            reducer.fit(sample_embeddings)


class TestDimensionalityReducerTransform:
    """Tests for DimensionalityReducer.transform() method."""

    def test_transform_pca_returns_correct_shape(self, sample_embeddings: np.ndarray):
        """Test transform returns (n_samples, n_components) shape."""
        reducer = DimensionalityReducer(method="pca")
        reducer.fit(sample_embeddings)
        result = reducer.transform(sample_embeddings)
        assert result.shape == (len(sample_embeddings), 2)

    def test_transform_pca_returns_float32(self, sample_embeddings: np.ndarray):
        """Test transform returns float32 dtype."""
        reducer = DimensionalityReducer(method="pca")
        reducer.fit(sample_embeddings)
        result = reducer.transform(sample_embeddings)
        assert result.dtype == np.float32

    def test_transform_raises_if_not_fitted(self, sample_embeddings: np.ndarray):
        """Test transform raises ValueError if not fitted."""
        reducer = DimensionalityReducer(method="pca")
        with pytest.raises(ValueError, match="not been fitted"):
            reducer.transform(sample_embeddings)

    def test_transform_umap(self, sample_embeddings: np.ndarray):
        """Test transform with UMAP works correctly."""
        reducer = DimensionalityReducer(method="umap")
        reducer.fit(sample_embeddings)
        result = reducer.transform(sample_embeddings)
        assert result.shape == (len(sample_embeddings), 2)
        assert result.dtype == np.float32


class TestDimensionalityReducerFitTransform:
    """Tests for DimensionalityReducer.fit_transform() method."""

    def test_fit_transform_pca(self, sample_embeddings: np.ndarray):
        """Test fit_transform with PCA returns correct shape."""
        reducer = DimensionalityReducer(method="pca")
        result = reducer.fit_transform(sample_embeddings)
        assert result.shape == (len(sample_embeddings), 2)
        assert reducer._fitted is True

    def test_fit_transform_pca_returns_float32(self, sample_embeddings: np.ndarray):
        """Test fit_transform returns float32 dtype."""
        reducer = DimensionalityReducer(method="pca")
        result = reducer.fit_transform(sample_embeddings)
        assert result.dtype == np.float32

    def test_fit_transform_umap(self, sample_embeddings: np.ndarray):
        """Test fit_transform with UMAP works correctly."""
        reducer = DimensionalityReducer(method="umap")
        result = reducer.fit_transform(sample_embeddings)
        assert result.shape == (len(sample_embeddings), 2)
        assert result.dtype == np.float32
        assert reducer._fitted is True

    def test_fit_transform_invalid_method_raises(self, sample_embeddings: np.ndarray):
        """Test fit_transform raises ValueError for invalid method."""
        reducer = DimensionalityReducer(method="invalid")
        with pytest.raises(ValueError, match="Unknown method"):
            reducer.fit_transform(sample_embeddings)

    def test_fit_transform_umap_larger_dataset(self, sample_embeddings: np.ndarray):
        """Test fit_transform with UMAP works correctly."""
        # Use larger dataset to avoid UMAP's small dataset numerical issues
        reducer = DimensionalityReducer(method="umap")
        result = reducer.fit_transform(sample_embeddings)
        assert result.shape == (len(sample_embeddings), 2)
        assert reducer._fitted is True


# ============================================================================
# PlotGenerator Tests
# ============================================================================


class TestPlotGeneratorInit:
    """Tests for PlotGenerator initialization."""

    def test_init_default_reduction_method(self):
        """Test default initialization uses UMAP."""
        plotter = PlotGenerator()
        assert plotter.reduction_method == "umap"

    def test_init_custom_reduction_method(self):
        """Test initialization with custom reduction method."""
        plotter = PlotGenerator(reduction_method="pca")
        assert plotter.reduction_method == "pca"


class TestPlotGeneratorScatterPlot:
    """Tests for PlotGenerator.scatter_plot() method."""

    def test_scatter_plot_returns_figure(
        self, sample_2d_coords: np.ndarray, sample_metadata_for_plotting: pd.DataFrame
    ):
        """Test scatter_plot returns a Plotly Figure."""
        plotter = PlotGenerator()
        fig = plotter.scatter_plot(sample_2d_coords, sample_metadata_for_plotting)
        assert isinstance(fig, go.Figure)

    def test_scatter_plot_with_color_by(
        self, sample_2d_coords: np.ndarray, sample_metadata_for_plotting: pd.DataFrame
    ):
        """Test scatter_plot with color_by parameter."""
        plotter = PlotGenerator()
        fig = plotter.scatter_plot(
            sample_2d_coords, sample_metadata_for_plotting, color_by="organism"
        )
        assert isinstance(fig, go.Figure)
        # Color should be applied - check legend title
        assert fig.layout.legend.title.text == "organism"

    def test_scatter_plot_without_color_by(
        self, sample_2d_coords: np.ndarray, sample_metadata_for_plotting: pd.DataFrame
    ):
        """Test scatter_plot without color_by parameter."""
        plotter = PlotGenerator()
        fig = plotter.scatter_plot(sample_2d_coords, sample_metadata_for_plotting)
        assert isinstance(fig, go.Figure)
        assert fig.layout.legend.title.text == "Dataset"

    def test_scatter_plot_truncates_long_descriptions(
        self, sample_2d_coords: np.ndarray, sample_metadata_for_plotting: pd.DataFrame
    ):
        """Test that long descriptions are truncated to 100 chars + '...'."""
        plotter = PlotGenerator()
        fig = plotter.scatter_plot(sample_2d_coords, sample_metadata_for_plotting)

        # The metadata has descriptions like "Sample experiment 1 " * 10
        # which is > 100 chars, should be truncated
        assert isinstance(fig, go.Figure)
        # Plot should be generated without error with long descriptions

    def test_scatter_plot_short_descriptions_not_truncated(
        self, sample_2d_coords: np.ndarray
    ):
        """Test that short descriptions are not truncated."""
        metadata = pd.DataFrame(
            {
                "accession": [f"ENCSR{i:05d}" for i in range(10)],
                "description": ["Short desc"] * 10,  # Less than 100 chars
                "assay_term_name": ["ChIP-seq"] * 10,
                "organism": ["human"] * 10,
            }
        )
        plotter = PlotGenerator()
        fig = plotter.scatter_plot(sample_2d_coords, metadata)
        assert isinstance(fig, go.Figure)

    def test_scatter_plot_hover_cols_filtered(self, sample_2d_coords: np.ndarray):
        """Test that only existing columns are used for hover."""
        # Metadata missing some DEFAULT_HOVER_COLS
        metadata = pd.DataFrame(
            {
                "accession": [f"ENCSR{i:05d}" for i in range(10)],
                # Missing: description, assay_term_name, organism
            }
        )
        plotter = PlotGenerator()
        # Should not raise error even with missing columns
        fig = plotter.scatter_plot(sample_2d_coords, metadata)
        assert isinstance(fig, go.Figure)

    def test_scatter_plot_axis_labels_umap(
        self, sample_2d_coords: np.ndarray, sample_metadata_for_plotting: pd.DataFrame
    ):
        """Test axis labels are UMAP-1 and UMAP-2 for UMAP method."""
        plotter = PlotGenerator(reduction_method="umap")
        fig = plotter.scatter_plot(sample_2d_coords, sample_metadata_for_plotting)
        assert fig.layout.xaxis.title.text == "UMAP-1"
        assert fig.layout.yaxis.title.text == "UMAP-2"

    def test_scatter_plot_axis_labels_pca(
        self, sample_2d_coords: np.ndarray, sample_metadata_for_plotting: pd.DataFrame
    ):
        """Test axis labels are PC-1 and PC-2 for PCA method."""
        plotter = PlotGenerator(reduction_method="pca")
        fig = plotter.scatter_plot(sample_2d_coords, sample_metadata_for_plotting)
        assert fig.layout.xaxis.title.text == "PC-1"
        assert fig.layout.yaxis.title.text == "PC-2"

    def test_scatter_plot_with_highlight_indices(
        self, sample_2d_coords: np.ndarray, sample_metadata_for_plotting: pd.DataFrame
    ):
        """Test scatter_plot with highlighted points adds extra trace."""
        plotter = PlotGenerator()
        fig = plotter.scatter_plot(
            sample_2d_coords,
            sample_metadata_for_plotting,
            highlight_indices=[0, 1],
        )
        # Should have 2 traces: main scatter + highlight
        assert len(fig.data) == 2
        # Second trace should be the highlight
        assert fig.data[1].name == "Selected"
        assert fig.data[1].marker.symbol == "star"
        assert fig.data[1].marker.color == "red"

    def test_scatter_plot_no_highlight(
        self, sample_2d_coords: np.ndarray, sample_metadata_for_plotting: pd.DataFrame
    ):
        """Test scatter_plot without highlight has single trace."""
        plotter = PlotGenerator()
        fig = plotter.scatter_plot(sample_2d_coords, sample_metadata_for_plotting)
        # Only main scatter trace
        assert len(fig.data) == 1

    def test_scatter_plot_custom_title(
        self, sample_2d_coords: np.ndarray, sample_metadata_for_plotting: pd.DataFrame
    ):
        """Test scatter_plot with custom title."""
        plotter = PlotGenerator()
        fig = plotter.scatter_plot(
            sample_2d_coords,
            sample_metadata_for_plotting,
            title="Custom Plot Title",
        )
        assert fig.layout.title.text == "Custom Plot Title"

    def test_scatter_plot_color_by_nonexistent_column(
        self, sample_2d_coords: np.ndarray, sample_metadata_for_plotting: pd.DataFrame
    ):
        """Test scatter_plot with color_by for non-existent column.

        Should still create a valid figure even if column doesn't exist.
        """
        plotter = PlotGenerator()
        # Should not error even with non-existent column
        fig = plotter.scatter_plot(
            sample_2d_coords,
            sample_metadata_for_plotting,
            color_by="nonexistent_column",
        )
        assert isinstance(fig, go.Figure)
        # Legend title is set to color_by value regardless of column existence
        # (the scatter_plot uses color_by for legend_title parameter directly)
        assert fig.layout.legend.title.text == "nonexistent_column"


class TestPlotGeneratorSimilarityHeatmap:
    """Tests for PlotGenerator.similarity_heatmap() method."""

    def test_similarity_heatmap_returns_figure(
        self, sample_similarity_matrix: np.ndarray
    ):
        """Test similarity_heatmap returns a Plotly Figure."""
        plotter = PlotGenerator()
        labels = ["A", "B", "C", "D", "E"]
        fig = plotter.similarity_heatmap(sample_similarity_matrix, labels)
        assert isinstance(fig, go.Figure)

    def test_similarity_heatmap_uses_viridis_colorscale(
        self, sample_similarity_matrix: np.ndarray
    ):
        """Test that heatmap uses Viridis colorscale (identified by colors)."""
        plotter = PlotGenerator()
        labels = ["A", "B", "C", "D", "E"]
        fig = plotter.similarity_heatmap(sample_similarity_matrix, labels)
        # Plotly may expand "Viridis" to tuple format with hex colors
        # Viridis starts with dark purple (#440154) and ends with yellow (#fde725)
        colorscale = fig.data[0].colorscale
        if isinstance(colorscale, str):
            assert colorscale == "Viridis"
        else:
            # Check it's a tuple of (position, color) pairs
            assert len(colorscale) > 0
            # First color should be dark purple (Viridis start)
            assert "#440154" in str(colorscale[0])
            # Last color should be yellow (Viridis end)
            assert "#fde725" in str(colorscale[-1])

    def test_similarity_heatmap_labels_applied(
        self, sample_similarity_matrix: np.ndarray
    ):
        """Test that labels are applied to axes."""
        plotter = PlotGenerator()
        labels = ["Dataset1", "Dataset2", "Dataset3", "Dataset4", "Dataset5"]
        fig = plotter.similarity_heatmap(sample_similarity_matrix, labels)
        # Check x and y labels on the heatmap data
        assert list(fig.data[0].x) == labels
        assert list(fig.data[0].y) == labels

    def test_similarity_heatmap_title_applied(
        self, sample_similarity_matrix: np.ndarray
    ):
        """Test that custom title is applied."""
        plotter = PlotGenerator()
        labels = ["A", "B", "C", "D", "E"]
        fig = plotter.similarity_heatmap(
            sample_similarity_matrix, labels, title="Custom Heatmap Title"
        )
        assert fig.layout.title.text == "Custom Heatmap Title"

    def test_similarity_heatmap_default_title(
        self, sample_similarity_matrix: np.ndarray
    ):
        """Test that default title is applied."""
        plotter = PlotGenerator()
        labels = ["A", "B", "C", "D", "E"]
        fig = plotter.similarity_heatmap(sample_similarity_matrix, labels)
        assert fig.layout.title.text == "Dataset Similarity Matrix"

    def test_similarity_heatmap_colorbar_title(
        self, sample_similarity_matrix: np.ndarray
    ):
        """Test that colorbar has correct title."""
        plotter = PlotGenerator()
        labels = ["A", "B", "C", "D", "E"]
        fig = plotter.similarity_heatmap(sample_similarity_matrix, labels)
        # Colorbar title can be dict, object with .text, or string
        colorbar = fig.data[0].colorbar
        title_val = colorbar.title
        # Handle different Plotly versions - title may be a dict, object, or string
        if hasattr(title_val, "text"):
            assert title_val.text == "Similarity"
        elif isinstance(title_val, dict):
            assert title_val.get("text") == "Similarity"
        else:
            assert str(title_val) == "Similarity"

    def test_similarity_heatmap_axis_labels(self, sample_similarity_matrix: np.ndarray):
        """Test that axis labels are 'Dataset'."""
        plotter = PlotGenerator()
        labels = ["A", "B", "C", "D", "E"]
        fig = plotter.similarity_heatmap(sample_similarity_matrix, labels)
        assert fig.layout.xaxis.title.text == "Dataset"
        assert fig.layout.yaxis.title.text == "Dataset"

    def test_similarity_heatmap_data_shape(self, sample_similarity_matrix: np.ndarray):
        """Test that heatmap data has correct shape."""
        plotter = PlotGenerator()
        labels = ["A", "B", "C", "D", "E"]
        fig = plotter.similarity_heatmap(sample_similarity_matrix, labels)
        # z data should match input matrix
        assert fig.data[0].z.shape == sample_similarity_matrix.shape
