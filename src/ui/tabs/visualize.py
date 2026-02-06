# src/ui/tabs/visualize.py
"""Visualization tab for MetaENCODE.

This module can be easily commented out or replaced with
a teammate's implementation.
"""

import streamlit as st

from src.visualization.plots import (
    DimensionalityReducer,
    PlotGenerator,
    percentile_range_filtering,
)


def generate_visualization(
    method: str, color_by: str, filter_outliers: bool = True
) -> None:
    """Generate 2D visualization of embeddings.

    Args:
        method: Dimensionality reduction method ('pca', 'umap', or 't-sne').
        color_by: Column to color points by.
        filter_outliers: Whether to filter outliers using percentile range.
    """
    with st.spinner(f"Computing {method.upper()} projection..."):
        try:
            embeddings = st.session_state.embeddings
            metadata_df = st.session_state.metadata_df

            if filter_outliers:
                filtered_embeddings, mask = percentile_range_filtering(embeddings)
                filtered_metadata = metadata_df[mask].reset_index(drop=True)
            else:
                filtered_embeddings = embeddings
                filtered_metadata = metadata_df

            reducer = DimensionalityReducer(method=method)
            coords_2d = reducer.fit_transform(filtered_embeddings)

            st.session_state.coords_2d = coords_2d
            st.session_state.viz_metadata = filtered_metadata
            st.session_state.viz_reduction_method = method
            st.session_state.viz_mode = "all_datasets"
            st.session_state.viz_variance_ratio = reducer.variance_ratio_
        except Exception as e:
            st.error(f"Error generating visualization: {e}")
            st.info(
                "Tip: Try using PCA instead of UMAP, or ensure data is loaded first."
            )


def generate_similar_only_visualization(method: str, color_by: str) -> None:
    """Generate visualization of only the similar datasets.

    Args:
        method: Dimensionality reduction method ('pca', 'umap', or 't-sne').
        color_by: Column to color points by.
    """
    with st.spinner(f"Computing {method.upper()} projection for similar datasets..."):
        try:
            similar_df = st.session_state.similar_datasets
            if similar_df is None or similar_df.empty:
                st.error("No similar datasets found. Run a similarity search first.")
                return

            # Get the top N from filter state
            filter_state = st.session_state.filter_state
            top_n = filter_state.max_results
            similar_df = similar_df.head(top_n)

            # Get embeddings for only the similar datasets
            full_metadata = st.session_state.metadata_df
            full_embeddings = st.session_state.embeddings

            # Find indices of similar datasets in full metadata
            similar_accs = set(similar_df["accession"].tolist())
            indices = [
                i
                for i, acc in enumerate(full_metadata["accession"])
                if acc in similar_accs
            ]

            if not indices:
                st.error("Could not find embeddings for similar datasets.")
                return

            # Extract embeddings for similar datasets only
            similar_embeddings = full_embeddings[indices]
            similar_metadata = full_metadata.iloc[indices].reset_index(drop=True)

            # Add similarity scores to metadata for coloring option
            score_map = dict(
                zip(similar_df["accession"], similar_df["similarity_score"])
            )
            similar_metadata = similar_metadata.copy()
            similar_metadata["similarity_score"] = similar_metadata["accession"].map(
                score_map
            )

            # Run dimensionality reduction on just the similar embeddings
            reducer = DimensionalityReducer(method=method)
            coords_2d = reducer.fit_transform(similar_embeddings)

            # Store results
            st.session_state.coords_2d = coords_2d
            st.session_state.viz_metadata = similar_metadata
            st.session_state.viz_reduction_method = method
            st.session_state.viz_mode = "similar_only"
            st.session_state.viz_variance_ratio = reducer.variance_ratio_

        except Exception as e:
            st.error(f"Error generating visualization: {e}")


def render_visualize_tab() -> None:
    """Render the visualization tab."""
    st.header("Dataset Visualization")

    if st.session_state.metadata_df is None or st.session_state.embeddings is None:
        st.info(
            "No data loaded. Please ensure the precomputed cache files exist in data/cache/."
        )
        return

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Options")

        # View mode selector
        similar_available = st.session_state.similar_datasets is not None
        view_mode = st.radio(
            "View Mode",
            options=["all_datasets", "similar_only"],
            format_func=lambda x: {
                "all_datasets": "All Datasets",
                "similar_only": "Similar Datasets Only",
            }.get(x, x),
            help="Show all datasets or only those from your similarity search",
            disabled=False,
        )

        # Warn if similar-only selected but no similar datasets
        if view_mode == "similar_only" and not similar_available:
            st.warning("Run a similarity search first to use this view.")

        reduction_method = st.selectbox(
            "Reduction Method",
            options=["pca", "umap", "t-sne"],
            index=0,
            help="PCA is faster; UMAP/t-SNE preserve local structure better",
        )

        # Determine available color options based on metadata columns
        available_colors = ["assay_term_name", "organism"]
        if st.session_state.metadata_df is not None:
            # Add slim type color options if columns exist
            slim_color_columns = [
                "organ",
                "cell_type",
                "developmental_layer",
                "body_system",
            ]
            for col in slim_color_columns:
                if col in st.session_state.metadata_df.columns:
                    available_colors.append(col)
            # Add lab at the end if it exists
            if "lab" in st.session_state.metadata_df.columns:
                available_colors.append("lab")

        # Add similarity_score option if in similar-only mode with available data
        if view_mode == "similar_only" and similar_available:
            available_colors.insert(0, "similarity_score")

        color_display_names = {
            "similarity_score": "Similarity Score",
            "assay_term_name": "Assay Type",
            "organism": "Organism",
            "organ": "Organ System",
            "cell_type": "Cell Type",
            "developmental_layer": "Germ Layer",
            "body_system": "Body System",
            "lab": "Lab",
        }
        color_option = st.selectbox(
            "Color By",
            options=available_colors,
            format_func=lambda x: color_display_names.get(
                x, x.replace("_", " ").title()
            ),
        )

        # Outlier filtering option (only for all datasets mode)
        filter_outliers = False
        if view_mode == "all_datasets":
            filter_outliers = st.checkbox(
                "Filter Outliers",
                value=False,
                help="Remove points outside 5th-95th percentile range. "
                "Disable to show all datasets.",
            )

        # Generate button - different function based on view mode
        can_generate = view_mode == "all_datasets" or similar_available
        if st.button(
            "Generate Visualization", type="primary", disabled=not can_generate
        ):
            if view_mode == "similar_only":
                generate_similar_only_visualization(reduction_method, color_option)
            else:
                generate_visualization(reduction_method, color_option, filter_outliers)

    with col1:
        st.subheader("Embedding Space")

        if st.session_state.coords_2d is not None:
            # Use filtered metadata if available, fallback to full metadata
            viz_metadata = getattr(
                st.session_state, "viz_metadata", st.session_state.metadata_df
            )
            coords = st.session_state.coords_2d

            # Use the stored method (what was actually used to generate coords)
            actual_method = st.session_state.get(
                "viz_reduction_method", reduction_method
            )
            stored_mode = st.session_state.get("viz_mode", "all_datasets")

            # Only highlight similar datasets in "all datasets" mode
            highlight_idx = None
            if (
                stored_mode == "all_datasets"
                and st.session_state.similar_datasets is not None
            ):
                # Find indices of similar datasets in the filtered visualization metadata
                similar_accs = set(
                    st.session_state.similar_datasets["accession"].tolist()
                )
                highlight_idx = [
                    i
                    for i, acc in enumerate(viz_metadata["accession"])
                    if acc in similar_accs
                ]

            # Determine title based on mode
            title = (
                "Similar Datasets"
                if stored_mode == "similar_only"
                else "Dataset Similarity Map"
            )

            # Generate plot using actual method (not dropdown value)
            plotter = PlotGenerator(reduction_method=actual_method)
            variance = st.session_state.get("viz_variance_ratio", None)
            fig = plotter.scatter_plot(
                coords,
                viz_metadata,
                color_by=color_option,
                title=title,
                highlight_indices=highlight_idx,
                variance_ratio=variance,
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Hover over points to see dataset details. "
                "Copy accession ID to visit encodeproject.org/experiments/{accession}/"
            )
        else:
            st.info(
                "Click 'Generate Visualization' to create the embedding plot. "
                "This may take a moment for UMAP/t-SNE."
            )
