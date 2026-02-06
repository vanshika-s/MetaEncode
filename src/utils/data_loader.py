# src/utils/data_loader.py
"""Data loading utilities for MetaENCODE.

This module previously contained load_sample_data() for fetching from the ENCODE API.
That function was removed to prevent accidental cache corruption.

All data should be precomputed using precompute_embeddings.py and loaded
via load_cached_data_into_session() in app.py.
"""
