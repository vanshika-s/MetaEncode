# src/ui/tabs/visualize.py
"""Visualization tab for MetaENCODE.

This module can be easily commented out or replaced with
a teammate's implementation.
"""

import streamlit as st

from src.visualization.plots import DimensionalityReducer, PlotGenerator


def generate_visualization(method: str, color_by: str) -> None:
    """Generate 2D visualization of embeddings.

    Args:
        method: Dimensionality reduction method ('pca' or 'umap').
        color_by: Column to color points by.
    """
    with st.spinner(f"Computing {method.upper()} projection..."):
        try:
            embeddings = st.session_state.embeddings
            reducer = DimensionalityReducer(method=method)
            coords_2d = reducer.fit_transform(embeddings)
            st.session_state.coords_2d = coords_2d
        except Exception as e:
            st.error(f"Error generating visualization: {e}")
            st.info(
                "Tip: Try using PCA instead of UMAP, or ensure data is loaded first."
            )


def render_visualize_tab() -> None:
    """Render the visualization tab."""
    st.header("Dataset Visualization")

    if st.session_state.metadata_df is None or st.session_state.embeddings is None:
        st.info(
            "Load sample data first using the 'Load Sample Data' button "
            "in the sidebar to visualize datasets."
        )
        return

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Options")

        reduction_method = st.selectbox(
            "Reduction Method",
            options=["pca", "umap"],
            index=0,
            help="PCA is faster, UMAP preserves local structure better",
        )

        # Determine available color options based on metadata columns
        available_colors = ["assay_term_name", "organism", "lab"]
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
                    available_colors.insert(-1, col)  # Insert before "lab"

        color_display_names = {
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

        if st.button("Generate Visualization", type="primary"):
            generate_visualization(reduction_method, color_option)

    with col1:
        st.subheader("Embedding Space")

        if st.session_state.coords_2d is not None:
            metadata_df = st.session_state.metadata_df
            coords = st.session_state.coords_2d

            # Get highlight indices if we have similar datasets
            highlight_idx = None
            if st.session_state.similar_datasets is not None:
                # Find indices of similar datasets in the full metadata
                similar_accs = set(
                    st.session_state.similar_datasets["accession"].tolist()
                )
                highlight_idx = [
                    i
                    for i, acc in enumerate(metadata_df["accession"])
                    if acc in similar_accs
                ]

            # Generate plot
            plotter = PlotGenerator(reduction_method=reduction_method)
            fig = plotter.scatter_plot(
                coords,
                metadata_df,
                color_by=color_option,
                title="Dataset Similarity Map",
                highlight_indices=highlight_idx,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "Click 'Generate Visualization' to create the embedding plot. "
                "This may take a moment for UMAP."
            )
