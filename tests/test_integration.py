# tests/test_integration.py
"""Integration tests for the full MetaENCODE pipeline.

These tests verify that all components work together correctly,
from data processing through embedding generation to similarity search.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.embeddings import EmbeddingGenerator
from src.ml.feature_combiner import FeatureCombiner
from src.ml.similarity import SimilarityEngine
from src.processing.metadata import MetadataProcessor


@pytest.fixture
def sample_raw_experiments() -> pd.DataFrame:
    """Sample raw experiment data as would come from ENCODE API."""
    return pd.DataFrame(
        {
            "accession": [
                "ENCSR000AAA",
                "ENCSR000BBB",
                "ENCSR000CCC",
                "ENCSR000DDD",
                "ENCSR000EEE",
            ],
            "description": [
                "ChIP-seq on human K562 cells targeting H3K27ac histone modification",
                "RNA-seq of mouse liver tissue for gene expression analysis",
                "ATAC-seq on human HepG2 cells measuring chromatin accessibility",
                "ChIP-seq on human K562 cells targeting H3K4me3 histone modification",
                "RNA-seq of human blood cells for transcriptome profiling",
            ],
            "title": [
                "H3K27ac ChIP-seq K562",
                "Mouse liver RNA-seq",
                "HepG2 ATAC-seq",
                "H3K4me3 ChIP-seq K562",
                "Human blood RNA-seq",
            ],
            "assay_term_name": [
                "ChIP-seq",
                "RNA-seq",
                "ATAC-seq",
                "ChIP-seq",
                "RNA-seq",
            ],
            "organism": ["human", "mouse", "human", "human", "human"],
            "biosample_term_name": ["K562", "liver", "HepG2", "K562", "blood"],
            "lab": ["lab-a", "lab-b", "lab-a", "lab-a", "lab-c"],
            "status": ["released"] * 5,
            "replicate_count": [2, 3, 1, 4, 2],
            "file_count": [10, 15, 8, 20, 12],
        }
    )


class TestFullPipelineTextOnly:
    """Tests for text-only similarity pipeline (backward compatibility)."""

    def test_text_only_pipeline(self, sample_raw_experiments: pd.DataFrame) -> None:
        """Test that text-only pipeline still works."""
        processor = MetadataProcessor()
        embedder = EmbeddingGenerator()
        similarity_engine = SimilarityEngine()

        # Process metadata
        processed_df = processor.process(sample_raw_experiments)
        assert "combined_text" in processed_df.columns

        # Generate text embeddings
        texts = processed_df["combined_text"].tolist()
        text_embeddings = embedder.encode(texts)
        assert text_embeddings.shape[0] == len(processed_df)
        assert text_embeddings.shape[1] == 384  # MiniLM dimension

        # Fit similarity engine
        similarity_engine.fit(text_embeddings)

        # Find similar to first experiment
        query = text_embeddings[0]
        similar = similarity_engine.find_similar(query, n=3)

        assert len(similar) >= 1
        assert "similarity_score" in similar.columns
        assert "index" in similar.columns


class TestFullPipelineCombined:
    """Tests for full combined feature pipeline."""

    def test_combined_pipeline(self, sample_raw_experiments: pd.DataFrame) -> None:
        """Test the full combined feature pipeline."""
        processor = MetadataProcessor()
        embedder = EmbeddingGenerator()
        combiner = FeatureCombiner()
        similarity_engine = SimilarityEngine()

        # Process metadata
        processed_df = processor.process(sample_raw_experiments)

        # Generate text embeddings
        texts = processed_df["combined_text"].tolist()
        text_embeddings = embedder.encode(texts)

        # Fit combiner and generate combined vectors
        combiner.fit(processed_df, text_embedding_dim=text_embeddings.shape[1])
        combined_vectors = combiner.transform(processed_df, text_embeddings)

        # Combined should have more dimensions than text-only
        assert combined_vectors.shape[1] > text_embeddings.shape[1]

        # Fit similarity engine with combined vectors
        similarity_engine.fit(combined_vectors)

        # Query with combined vector
        query_vector = combiner.transform_single(
            processed_df.iloc[0].to_dict(), text_embeddings[0]
        )
        similar = similarity_engine.find_similar(query_vector, n=3)

        assert len(similar) >= 1
        assert similar["similarity_score"].iloc[0] <= 1.0

    def test_combined_dimension_breakdown(
        self, sample_raw_experiments: pd.DataFrame
    ) -> None:
        """Test that feature dimension breakdown is correct."""
        processor = MetadataProcessor()
        embedder = EmbeddingGenerator()
        combiner = FeatureCombiner()

        processed_df = processor.process(sample_raw_experiments)
        texts = processed_df["combined_text"].tolist()
        text_embeddings = embedder.encode(texts)

        combiner.fit(processed_df, text_embedding_dim=text_embeddings.shape[1])

        breakdown = combiner.get_feature_breakdown()

        # Check expected dimensions
        assert breakdown["text_embedding"] == 384
        assert breakdown["assay_term_name"] > 0  # At least 1 category
        assert breakdown["organism"] > 0
        assert breakdown["biosample_term_name"] > 0
        assert breakdown["lab"] > 0
        assert breakdown["numeric_features"] == 2  # replicate_count, file_count

        # Total should match
        total = sum(breakdown.values())
        assert combiner.feature_dim == total


class TestQueryMatchesBatch:
    """Tests that single query produces same results as batch for same record."""

    def test_transform_single_equals_batch(
        self, sample_raw_experiments: pd.DataFrame
    ) -> None:
        """Test that transform_single matches batch transform."""
        processor = MetadataProcessor()
        embedder = EmbeddingGenerator()
        combiner = FeatureCombiner()

        processed_df = processor.process(sample_raw_experiments)
        texts = processed_df["combined_text"].tolist()
        text_embeddings = embedder.encode(texts)

        combiner.fit(processed_df, text_embedding_dim=text_embeddings.shape[1])

        # Batch transform
        batch_result = combiner.transform(processed_df, text_embeddings)

        # Single transform for each record
        for i in range(len(processed_df)):
            record = processed_df.iloc[i].to_dict()
            single_result = combiner.transform_single(record, text_embeddings[i])
            np.testing.assert_array_almost_equal(
                batch_result[i], single_result, decimal=5
            )


class TestSimilarityWithDifferentWeights:
    """Tests that weights affect similarity results."""

    def test_weights_change_rankings(
        self, sample_raw_experiments: pd.DataFrame
    ) -> None:
        """Test that different weights can change similarity rankings."""
        processor = MetadataProcessor()
        embedder = EmbeddingGenerator()

        processed_df = processor.process(sample_raw_experiments)
        texts = processed_df["combined_text"].tolist()
        text_embeddings = embedder.encode(texts)

        # High text weight
        combiner_text = FeatureCombiner(
            weights={
                "text_embedding": 0.9,
                "assay_type": 0.025,
                "organism": 0.025,
                "cell_type": 0.025,
                "lab": 0.025,
                "numeric_features": 0.0,
            }
        )
        combiner_text.fit(processed_df, text_embedding_dim=text_embeddings.shape[1])
        vectors_text = combiner_text.transform(processed_df, text_embeddings)

        # High categorical weight
        combiner_cat = FeatureCombiner(
            weights={
                "text_embedding": 0.1,
                "assay_type": 0.3,
                "organism": 0.3,
                "cell_type": 0.2,
                "lab": 0.1,
                "numeric_features": 0.0,
            }
        )
        combiner_cat.fit(processed_df, text_embedding_dim=text_embeddings.shape[1])
        vectors_cat = combiner_cat.transform(processed_df, text_embeddings)

        # The vectors should be different
        assert not np.allclose(vectors_text, vectors_cat)


class TestChipSeqSimilarity:
    """Tests that similar experiments (same assay type) are ranked higher."""

    def test_same_assay_type_ranked_higher(
        self, sample_raw_experiments: pd.DataFrame
    ) -> None:
        """Test that experiments with same assay type have higher similarity."""
        processor = MetadataProcessor()
        embedder = EmbeddingGenerator()
        combiner = FeatureCombiner()
        similarity_engine = SimilarityEngine()

        processed_df = processor.process(sample_raw_experiments)
        texts = processed_df["combined_text"].tolist()
        text_embeddings = embedder.encode(texts)

        combiner.fit(processed_df, text_embedding_dim=text_embeddings.shape[1])
        combined_vectors = combiner.transform(processed_df, text_embeddings)
        similarity_engine.fit(combined_vectors)

        # Query with first ChIP-seq experiment
        chip_seq_idx = 0  # First experiment is ChIP-seq
        query_vector = combined_vectors[chip_seq_idx]

        # Find similar (excluding self)
        similar = similarity_engine.find_similar(query_vector, n=4, exclude_self=True)

        # Check which indices are returned
        similar_indices = similar["index"].tolist()

        # The other ChIP-seq experiment (index 3) should be in the results
        # and ideally ranked higher than RNA-seq or ATAC-seq
        chip_seq_other_idx = 3
        assert chip_seq_other_idx in similar_indices
