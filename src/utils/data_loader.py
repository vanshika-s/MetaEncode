# src/utils/data_loader.py
"""Data loading utilities for MetaENCODE.

This module handles fetching, processing, and caching of ENCODE experiment data.
"""

import streamlit as st

from src.ml.similarity import SimilarityEngine
from src.ui.components.initializers import (
    get_api_client,
    get_cache_manager,
    get_embedding_generator,
    get_feature_combiner,
    get_metadata_processor,
)


def load_sample_data() -> None:
    """Load a sample of ENCODE experiments for demonstration.

    Fetches experiments from the ENCODE API, processes metadata,
    generates embeddings, and caches the results.
    """
    with st.spinner("Loading sample data from ENCODE API..."):
        try:
            client = get_api_client()
            processor = get_metadata_processor()
            embedder = get_embedding_generator()
            combiner = get_feature_combiner()

            # Fetch a small sample of experiments
            raw_df = client.fetch_experiments(limit=100)

            if raw_df.empty:
                st.error("No experiments found")
                return

            # Process metadata
            processed_df = processor.process(raw_df)

            # Validate records and filter invalid ones
            valid_mask = processed_df.apply(
                lambda row: processor.validate_record(row.to_dict()), axis=1
            )
            if not valid_mask.all():
                invalid_count = (~valid_mask).sum()
                st.warning(
                    f"Filtered {invalid_count} records missing required metadata"
                )
                processed_df = processed_df[valid_mask].reset_index(drop=True)

            if processed_df.empty:
                st.error("No valid experiments found after validation")
                return

            # Generate text embeddings
            st.info("Generating text embeddings...")
            texts = processed_df["combined_text"].tolist()
            text_embeddings = embedder.encode(texts, show_progress=False)

            # Fit feature combiner and generate combined vectors
            st.info("Combining features (text + categorical + numeric)...")
            combiner.fit(processed_df, text_embedding_dim=text_embeddings.shape[1])
            combined_vectors = combiner.transform(processed_df, text_embeddings)

            # Fit similarity engine with COMBINED vectors (not text-only)
            similarity_engine = SimilarityEngine()
            similarity_engine.fit(combined_vectors)

            # Store in session state
            st.session_state.metadata_df = processed_df
            st.session_state.embeddings = text_embeddings
            st.session_state.combined_vectors = combined_vectors
            st.session_state.feature_combiner = combiner
            st.session_state.similarity_engine = similarity_engine

            # Cache the data (only if not overwriting a larger existing cache)
            cache_mgr = get_cache_manager()
            existing_meta = (
                cache_mgr.load("metadata") if cache_mgr.exists("metadata") else None
            )

            if existing_meta is not None and len(existing_meta) > len(processed_df):
                st.warning(
                    f"Skipped caching: existing cache has "
                    f"{len(existing_meta)} experiments, "
                    f"not overwriting with {len(processed_df)} samples. "
                    "Use precompute_embeddings.py to update full cache."
                )
            else:
                cache_mgr.save("metadata", processed_df)
                cache_mgr.save("embeddings", text_embeddings)
                cache_mgr.save("combined_vectors", combined_vectors)
                cache_mgr.save("feature_combiner", combiner)

            # Show feature breakdown
            breakdown = combiner.get_feature_breakdown()
            text_dim = breakdown.get("text_embedding", 0)
            numeric_dim = breakdown.get("numeric_features", 0)
            categorical_dim = sum(
                v
                for k, v in breakdown.items()
                if k not in ["text_embedding", "numeric_features"]
            )
            st.success(
                f"Loaded {len(processed_df)} experiments with "
                f"{combiner.feature_dim}-dim combined vectors "
                f"(text: {text_dim}, categorical: {categorical_dim}, "
                f"numeric: {numeric_dim})"
            )

        except Exception as e:
            st.error(f"Failed to load data: {e}")
