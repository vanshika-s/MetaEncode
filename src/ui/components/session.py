# src/ui/components/session.py
"""Session state management for MetaENCODE.

This module handles initialization and management of Streamlit session state
variables used throughout the application.
"""

import streamlit as st

from src.ui.search_filters import FilterState

# Default values for session state
SESSION_DEFAULTS: dict = {
    "selected_dataset": None,
    "selected_index": None,
    "search_results": None,
    "similar_datasets": None,
    "metadata_df": None,
    "embeddings": None,
    "combined_vectors": None,
    "feature_combiner": None,
    "similarity_engine": None,
    "coords_2d": None,
    # New filter state using FilterState dataclass
    "filter_state": FilterState(),
    # Legacy filter_settings for backward compatibility
    "filter_settings": {
        "organism": None,
        "assay_type": None,
        "top_n": 10,
    },
}


def init_session_state() -> None:
    """Initialize Streamlit session state variables.

    Sets default values for all session state keys if they don't already exist.
    This should be called at the start of each Streamlit run.
    """
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_cached_data_into_session() -> bool:
    """Load precomputed data from cache into session state.

    Attempts to load metadata, embeddings, and combined vectors from cache
    and initializes the similarity engine. Only loads if session state
    doesn't already have metadata loaded.

    Returns:
        True if data was loaded from cache, False otherwise.
    """
    # Skip if already loaded
    if st.session_state.metadata_df is not None:
        return False

    # Import here to avoid circular imports
    from src.ml.similarity import SimilarityEngine
    from src.ui.components.initializers import get_cache_manager, load_cached_data

    cache_mgr = get_cache_manager()
    cached_meta, cached_emb, cached_combined, cached_combiner = load_cached_data(
        cache_mgr
    )

    if cached_meta is None or cached_emb is None:
        return False

    # Load metadata and embeddings
    st.session_state.metadata_df = cached_meta
    st.session_state.embeddings = cached_emb

    # Initialize similarity engine with combined vectors if available
    similarity_engine = SimilarityEngine()
    if cached_combined is not None:
        st.session_state.combined_vectors = cached_combined
        similarity_engine.fit(cached_combined)
    else:
        similarity_engine.fit(cached_emb)

    st.session_state.similarity_engine = similarity_engine

    # Restore feature combiner if available
    if cached_combiner is not None:
        st.session_state.feature_combiner = cached_combiner

    return True
