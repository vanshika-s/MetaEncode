# tests/conftest.py
"""Pytest fixtures and configuration for MetaENCODE tests."""

import numpy as np
import pandas as pd
import pytest

from src.ml.feature_combiner import FeatureCombiner


@pytest.fixture
def sample_experiment_data() -> dict:
    """Sample experiment data from ENCODE API response."""
    return {
        "accession": "ENCSR000AAA",
        "description": "ChIP-seq on human K562 cells targeting H3K27ac",
        "assay_term_name": "ChIP-seq",
        "biosample_ontology": {"term_name": "K562"},
        "lab": "/labs/encode-consortium/",
        "status": "released",
        "files": ["/files/ENCFF001AAA/", "/files/ENCFF002AAA/"],
        "replicates": ["/replicates/1/", "/replicates/2/"],
    }


@pytest.fixture
def sample_experiments_df() -> pd.DataFrame:
    """Sample DataFrame of experiments for testing."""
    return pd.DataFrame(
        {
            "accession": ["ENCSR000AAA", "ENCSR000BBB", "ENCSR000CCC"],
            "description": [
                "ChIP-seq on human K562 cells targeting H3K27ac",
                "RNA-seq of mouse liver tissue",
                "ATAC-seq on human HepG2 cells",
            ],
            "assay_term_name": ["ChIP-seq", "RNA-seq", "ATAC-seq"],
            "organism": ["human", "mouse", "human"],
            "biosample": ["K562", "liver", "HepG2"],
            "lab": ["encode-consortium", "encode-consortium", "encode-consortium"],
        }
    )


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Sample embeddings array for testing."""
    np.random.seed(42)
    return np.random.randn(10, 384)  # 10 samples, 384 dimensions (MiniLM)


@pytest.fixture
def sample_embedding_single() -> np.ndarray:
    """Single sample embedding for testing."""
    np.random.seed(42)
    return np.random.randn(384)


@pytest.fixture
def sample_combined_df() -> pd.DataFrame:
    """DataFrame with all required columns for FeatureCombiner testing."""
    return pd.DataFrame(
        {
            "accession": ["ENCSR000AAA", "ENCSR000BBB", "ENCSR000CCC", "ENCSR000DDD"],
            "combined_text": [
                "chip seq k562 h3k27ac",
                "rna seq liver tissue",
                "atac seq hepg2 cells",
                "chip seq k562 h3k4me3",
            ],
            "assay_term_name": ["ChIP-seq", "RNA-seq", "ATAC-seq", "ChIP-seq"],
            "organism": ["human", "mouse", "human", "human"],
            "biosample_term_name": ["K562", "liver", "HepG2", "K562"],
            "lab": ["lab-a", "lab-b", "lab-a", "lab-c"],
            "replicate_count": [2, 3, 1, 4],
            "file_count": [10, 15, 8, 20],
        }
    )


@pytest.fixture
def sample_combined_text_embeddings() -> np.ndarray:
    """Sample text embeddings matching sample_combined_df."""
    np.random.seed(42)
    return np.random.randn(4, 384).astype(np.float32)


@pytest.fixture
def fitted_feature_combiner(sample_combined_df: pd.DataFrame) -> FeatureCombiner:
    """Pre-fitted FeatureCombiner instance."""
    combiner = FeatureCombiner()
    combiner.fit(sample_combined_df, text_embedding_dim=384)
    return combiner


@pytest.fixture
def sample_categorical_series() -> pd.Series:
    """Sample categorical data for encoder tests."""
    return pd.Series(["ChIP-seq", "RNA-seq", "ATAC-seq", "ChIP-seq", None])


@pytest.fixture
def sample_numeric_series() -> pd.Series:
    """Sample numeric data for encoder tests."""
    return pd.Series([1.0, 2.0, 3.0, np.nan, 5.0])


# ============================================================================
# Visualization Fixtures
# ============================================================================


@pytest.fixture
def sample_2d_coords() -> np.ndarray:
    """Sample 2D coordinates for visualization tests."""
    np.random.seed(42)
    return np.random.randn(10, 2).astype(np.float32)


@pytest.fixture
def sample_metadata_for_plotting() -> pd.DataFrame:
    """Sample metadata DataFrame for plot testing with long descriptions."""
    return pd.DataFrame(
        {
            "accession": [f"ENCSR{i:05d}" for i in range(10)],
            "description": [f"Sample experiment {i} " * 10 for i in range(10)],
            "assay_term_name": ["ChIP-seq", "RNA-seq", "ATAC-seq"] * 3 + ["ChIP-seq"],
            "organism": ["human", "mouse"] * 5,
        }
    )


@pytest.fixture
def sample_similarity_matrix() -> np.ndarray:
    """Sample symmetric similarity matrix for heatmap tests."""
    np.random.seed(42)
    matrix = np.random.rand(5, 5)
    # Make symmetric
    matrix = (matrix + matrix.T) / 2
    # Set diagonal to 1.0
    np.fill_diagonal(matrix, 1.0)
    return matrix.astype(np.float32)


@pytest.fixture
def sample_small_embeddings() -> np.ndarray:
    """Small embeddings array for testing edge cases (3 samples)."""
    np.random.seed(42)
    return np.random.randn(3, 384).astype(np.float32)
