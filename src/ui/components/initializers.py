# src/ui/components/initializers.py
"""Cached resource getters for MetaENCODE components.

This module provides Streamlit-cached singletons for all major components
used in the application: API clients, ML models, and data processors.
"""

import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from src.api.encode_client import EncodeClient
from src.ml.embeddings import EmbeddingGenerator
from src.ml.feature_combiner import FeatureCombiner
from src.processing.metadata import MetadataProcessor
from src.ui.search_filters import SearchFilterManager
from src.utils.cache import CacheManager
from src.utils.history import SelectionHistory
from src.utils.user_id import get_or_create_user_id

_HISTORY_SESSION_KEY = "_selection_history_instance"


def get_selection_history() -> SelectionHistory:
    """Get or create a per-user selection history manager.

    Each browser gets its own history file, namespaced by a cookie-based
    user ID, and cached in session state (not globally).
    """
    if _HISTORY_SESSION_KEY in st.session_state:
        return st.session_state[_HISTORY_SESSION_KEY]

    user_id = get_or_create_user_id()
    user_path = Path(f"data/cache/selection_history_{user_id}.json")

    # One-time migration: seed new per-user file from the old shared file
    shared_path = SelectionHistory.DEFAULT_PATH
    if not user_path.exists() and shared_path.exists():
        shutil.copy2(shared_path, user_path)

    instance = SelectionHistory(path=str(user_path))
    st.session_state[_HISTORY_SESSION_KEY] = instance
    return instance


@st.cache_resource
def get_cache_manager() -> CacheManager:
    """Get or create the cache manager instance."""
    return CacheManager()


@st.cache_resource
def get_api_client() -> EncodeClient:
    """Get or create the API client instance."""
    return EncodeClient()


@st.cache_resource
def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create the embedding generator instance."""
    return EmbeddingGenerator()


@st.cache_resource
def get_metadata_processor() -> MetadataProcessor:
    """Get or create the metadata processor instance."""
    return MetadataProcessor()


@st.cache_resource
def get_feature_combiner() -> FeatureCombiner:
    """Get or create the feature combiner instance."""
    return FeatureCombiner()


@st.cache_resource
def get_filter_manager() -> SearchFilterManager:
    """Get or create the search filter manager instance."""
    return SearchFilterManager()


# Required columns for proper metadata display (especially organ in tooltips)
REQUIRED_METADATA_COLUMNS = {
    "accession",
    "organ",
    "cell_type",
    "assay_term_name",
    "organism",
    "description",
    "biosample_term_name",
}


@st.cache_data
def load_cached_data(
    _cache_mgr: CacheManager,
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    FeatureCombiner | None,
]:
    """Load precomputed metadata, embeddings, and combined vectors from cache.

    Args:
        _cache_mgr: Cache manager instance (prefixed with _ to avoid hashing).

    Returns:
        Tuple of (metadata_df, text_embeddings, combined_vectors, feature_combiner)
        or (None, None, None, None) if not cached.
    """
    if _cache_mgr.exists("metadata") and _cache_mgr.exists("embeddings"):
        metadata = _cache_mgr.load("metadata")
        embeddings = _cache_mgr.load("embeddings")

        # Re-process metadata if required columns are missing (e.g., organ, cell_type)
        if metadata is not None:
            existing_cols = set(metadata.columns)
            missing_cols = REQUIRED_METADATA_COLUMNS - existing_cols
            if missing_cols:
                processor = MetadataProcessor()
                metadata = processor.process(metadata)
                _cache_mgr.save("metadata", metadata)
                st.toast(f"Updated cached metadata with: {', '.join(sorted(missing_cols))}")

        # Try to load combined vectors and combiner (Phase 2 data)
        combined_vectors = None
        feature_combiner = None
        if _cache_mgr.exists("combined_vectors"):
            combined_vectors = _cache_mgr.load("combined_vectors")
        if _cache_mgr.exists("feature_combiner"):
            feature_combiner = _cache_mgr.load("feature_combiner")

        return metadata, embeddings, combined_vectors, feature_combiner
    return None, None, None, None
