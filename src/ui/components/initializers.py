# src/ui/components/initializers.py
"""Cached resource getters for MetaENCODE components.

This module provides Streamlit-cached singletons for all major components
used in the application: API clients, ML models, and data processors.
"""

import pandas as pd
import streamlit as st

from src.api.encode_client import EncodeClient
from src.ml.embeddings import EmbeddingGenerator
from src.ml.feature_combiner import FeatureCombiner
from src.processing.metadata import MetadataProcessor
from src.ui.search_filters import SearchFilterManager
from src.utils.cache import CacheManager


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

        # Try to load combined vectors and combiner (Phase 2 data)
        combined_vectors = None
        feature_combiner = None
        if _cache_mgr.exists("combined_vectors"):
            combined_vectors = _cache_mgr.load("combined_vectors")
        if _cache_mgr.exists("feature_combiner"):
            feature_combiner = _cache_mgr.load("feature_combiner")

        return metadata, embeddings, combined_vectors, feature_combiner
    return None, None, None, None
