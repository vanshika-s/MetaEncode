# src/ui/tabs/similar.py
"""Similar datasets tab for MetaENCODE."""

import pandas as pd
import streamlit as st

from src.ui.components.initializers import get_embedding_generator
from src.ui.formatters import (
    format_accession_as_link,
    format_organism_display,
    get_encode_experiment_url,
    truncate_text,
)


def render_similar_tab() -> None:
    """Render the similar datasets tab."""
    st.header("Similar Datasets")

    if st.session_state.selected_dataset is None:
        st.info("Select a dataset first to find similar experiments.")
        return

    # Check if we have loaded data
    if st.session_state.metadata_df is None or st.session_state.embeddings is None:
        st.warning(
            "No data loaded. Please ensure the precomputed cache files exist in data/cache/."
        )
        return

    selected = st.session_state.selected_dataset
    accession = selected.get("accession", "Unknown")
    accession_link = format_accession_as_link(accession)
    st.markdown(f"Finding datasets similar to: **{accession_link}**")

    # Get filter state
    filter_state = st.session_state.filter_state
    top_n = filter_state.max_results

    if st.button("Find Similar Datasets", type="primary"):
        with st.spinner("Computing similarities..."):
            try:
                embedder = get_embedding_generator()
                similarity_engine = st.session_state.similarity_engine
                feature_combiner = st.session_state.feature_combiner

                if similarity_engine is None:
                    st.error(
                        "Similarity engine not initialized. Please load data first."
                    )
                    return

                # Generate text embedding for selected dataset
                text = f"{selected.get('description', '')} {selected.get('title', '')}"
                text_embedding = embedder.encode_single(text)

                # Generate combined query vector (if feature combiner is available)
                if feature_combiner is not None and feature_combiner.is_fitted:
                    query_vector = feature_combiner.transform_single(
                        selected, text_embedding
                    )
                else:
                    # Fallback to text-only similarity
                    query_vector = text_embedding

                # Ensure the query vector is compatible with the similarity engine
                engine_dim = getattr(similarity_engine, "vector_dim", None)
                if engine_dim is None:
                    engine_dim = getattr(similarity_engine, "embedding_dim", None)

                if engine_dim is not None:
                    # Try to determine the dimensionality of the query vector
                    try:
                        query_dim = len(query_vector)
                    except TypeError:
                        query_dim = getattr(getattr(query_vector, "shape", None), "__getitem__", lambda _i: None)(0)

                    if query_dim is not None and query_dim != engine_dim:
                        st.error(
                            "The similarity engine was initialized with embeddings of a "
                            f"different dimensionality ({engine_dim}) than the current "
                            f"query vector ({query_dim}). Please reload data so that the "
                            "similarity engine and query embeddings are consistent."
                        )
                        return
                # Find more similar datasets than requested (for post-filtering)
                fetch_n = max(top_n * 3, 30)
                similar_df = similarity_engine.find_similar(
                    query_vector, n=fetch_n, exclude_self=True
                )

                # Get metadata for similar datasets
                metadata_df = st.session_state.metadata_df
                results = []
                for _, row in similar_df.iterrows():
                    idx = int(row["index"])
                    if idx < len(metadata_df):
                        meta = metadata_df.iloc[idx].to_dict()
                        meta["similarity_score"] = row["similarity_score"]
                        results.append(meta)

                st.session_state.similar_datasets = pd.DataFrame(results)

            except Exception as e:
                st.error(f"Error computing similarities: {e}")

    # Display similar datasets
    if st.session_state.similar_datasets is not None:
        similar = st.session_state.similar_datasets

        if not similar.empty:
            st.subheader("Most Similar Datasets")

            # Limit display to max_results; similarity ranking already applied (no additional filters)
            display_similar = similar.head(top_n)

            # Display columns with proper formatting
            display_cols = [
                "similarity_score",
                "accession",
                "assay_term_name",
                "organism",
                "biosample_term_name",
                "description",
            ]
            display_cols = [c for c in display_cols if c in display_similar.columns]

            display_df = display_similar[display_cols].copy()

            # Format similarity score
            display_df["similarity_score"] = display_df["similarity_score"].apply(
                lambda x: f"{x:.3f}"
            )

            # Format organism with assembly
            if "organism" in display_df.columns:
                display_df["organism"] = display_df["organism"].apply(
                    format_organism_display
                )

            # Truncate description
            if "description" in display_df.columns:
                display_df["description"] = display_df["description"].apply(
                    lambda x: truncate_text(str(x), 60)
                )

            # Replace accession values with ENCODE URLs for clickable links
            display_df["accession"] = display_similar["accession"].apply(get_encode_experiment_url)

            # Rename columns for display
            column_labels = {
                "similarity_score": "Similarity",
                "accession": "Accession",
                "assay_term_name": "Assay",
                "organism": "Organism [Assembly]",
                "biosample_term_name": "Biosample",
                "description": "Description",
            }
            display_df = display_df.rename(
                columns={
                    k: v for k, v in column_labels.items() if k in display_df.columns
                }
            )

            # Configure Accession column as clickable link showing accession ID
            column_config = {
                "Accession": st.column_config.LinkColumn(
                    "Accession",
                    display_text=r"experiments/(ENC[^/]+)/",
                    help="Click to open on ENCODE Portal",
                ),
            }

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
            )
