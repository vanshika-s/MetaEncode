# tests/test_ml/test_embeddings.py
"""Tests for the EmbeddingGenerator class."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.ml.embeddings import EmbeddingGenerator

# ============================================================================
# Initialization Tests
# ============================================================================


class TestEmbeddingGeneratorInit:
    """Tests for EmbeddingGenerator initialization."""

    def test_init_default_model(self):
        """Test initialization with default model."""
        generator = EmbeddingGenerator()
        assert generator.model_name == "all-MiniLM-L6-v2"
        assert generator._model is None  # Lazy loading

    def test_init_custom_model(self):
        """Test initialization with custom model name."""
        generator = EmbeddingGenerator(model_name="all-mpnet-base-v2")
        assert generator.model_name == "all-mpnet-base-v2"


# ============================================================================
# Encode Method Tests
# ============================================================================


class TestEmbeddingGeneratorEncode:
    """Tests for EmbeddingGenerator.encode() method."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_returns_correct_shape(self, mock_st_class: MagicMock):
        """Test encode returns (n_texts, embedding_dim) shape."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(3, 384).astype(np.float32)
        mock_st_class.return_value = mock_model

        generator = EmbeddingGenerator()
        result = generator.encode(["text1", "text2", "text3"])

        assert result.shape == (3, 384)
        assert result.dtype == np.float32
        mock_model.encode.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_empty_list_returns_correct_shape(self, mock_st_class: MagicMock):
        """Test that encode([]) returns (0, embedding_dim) array.

        Verifies the model is not called for empty input.
        """
        # Line 77: Empty list returns correctly shaped empty array
        mock_model = MagicMock()
        mock_st_class.return_value = mock_model

        generator = EmbeddingGenerator()
        result = generator.encode([])

        # Should return (0, 384) array without calling model.encode()
        assert result.shape == (0, 384)
        assert result.dtype == np.float64  # Default numpy dtype for empty reshape
        # Model's encode should NOT be called for empty list
        mock_model.encode.assert_not_called()

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_handles_empty_strings(self, mock_st_class: MagicMock):
        """Test that empty strings are replaced with space."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(2, 384).astype(np.float32)
        mock_st_class.return_value = mock_model

        generator = EmbeddingGenerator()
        generator.encode(["valid text", ""])

        # Check that the processed texts have empty strings replaced
        call_args = mock_model.encode.call_args
        processed_texts = call_args[0][0]
        assert processed_texts[0] == "valid text"
        assert processed_texts[1] == " "  # Empty string replaced with space

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_returns_float32(self, mock_st_class: MagicMock):
        """Test that encode returns float32 dtype."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 384)  # Default float64
        mock_st_class.return_value = mock_model

        generator = EmbeddingGenerator()
        result = generator.encode(["test text"])

        assert result.dtype == np.float32

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_with_custom_batch_size(self, mock_st_class: MagicMock):
        """Test encode passes batch_size to model."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 384).astype(np.float32)
        mock_st_class.return_value = mock_model

        generator = EmbeddingGenerator()
        generator.encode(["test"], batch_size=64)

        mock_model.encode.assert_called_once()
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs["batch_size"] == 64


# ============================================================================
# Encode Single Method Tests
# ============================================================================


class TestEmbeddingGeneratorEncodeSingle:
    """Tests for EmbeddingGenerator.encode_single() method."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_single_returns_correct_shape(self, mock_st_class: MagicMock):
        """Test encode_single returns 1D array of embedding_dim."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
        mock_st_class.return_value = mock_model

        generator = EmbeddingGenerator()
        result = generator.encode_single("test text")

        assert result.shape == (384,)
        assert result.dtype == np.float32

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_single_empty_string_handled(self, mock_st_class: MagicMock):
        """Test encode_single handles empty string by replacing with space."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
        mock_st_class.return_value = mock_model

        generator = EmbeddingGenerator()
        result = generator.encode_single("")

        # Check that empty string was replaced with space
        call_args = mock_model.encode.call_args
        assert call_args[0][0] == " "
        assert result.shape == (384,)

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_single_whitespace_only_handled(self, mock_st_class: MagicMock):
        """Test encode_single handles whitespace-only string."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
        mock_st_class.return_value = mock_model

        generator = EmbeddingGenerator()
        result = generator.encode_single("   ")  # Whitespace only

        # Check that whitespace-only was replaced with space
        call_args = mock_model.encode.call_args
        assert call_args[0][0] == " "
        assert result.shape == (384,)

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_single_returns_float32(self, mock_st_class: MagicMock):
        """Test that encode_single returns float32 dtype."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(384)  # Default float64
        mock_st_class.return_value = mock_model

        generator = EmbeddingGenerator()
        result = generator.encode_single("test")

        assert result.dtype == np.float32


# ============================================================================
# Embedding Dimension Property Tests
# ============================================================================


class TestEmbeddingGeneratorEmbeddingDim:
    """Tests for EmbeddingGenerator.embedding_dim property."""

    def test_embedding_dim_known_model(self):
        """Test embedding_dim returns correct value for known model without loading."""
        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        assert generator.embedding_dim == 384
        # Model should NOT be loaded for known model
        assert generator._model is None

    def test_embedding_dim_another_known_model(self):
        """Test embedding_dim for another known model."""
        generator = EmbeddingGenerator(model_name="all-mpnet-base-v2")
        assert generator.embedding_dim == 768
        assert generator._model is None

    @patch("sentence_transformers.SentenceTransformer")
    def test_embedding_dim_unknown_model_loads_model(self, mock_st_class: MagicMock):
        """Test embedding_dim loads model for unknown model name."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 512
        mock_st_class.return_value = mock_model

        generator = EmbeddingGenerator(model_name="unknown-model-name")
        dim = generator.embedding_dim

        assert dim == 512
        mock_st_class.assert_called_once_with("unknown-model-name")
        mock_model.get_sentence_embedding_dimension.assert_called_once()


# ============================================================================
# Model Loading Tests
# ============================================================================


class TestEmbeddingGeneratorModelLoading:
    """Tests for lazy model loading behavior."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_model_loaded_lazily(self, mock_st_class: MagicMock):
        """Test that model is not loaded until first encode call."""
        generator = EmbeddingGenerator()

        # Model should not be loaded yet
        mock_st_class.assert_not_called()
        assert generator._model is None

    @patch("sentence_transformers.SentenceTransformer")
    def test_model_loaded_on_encode(self, mock_st_class: MagicMock):
        """Test that model is loaded on first encode call."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 384).astype(np.float32)
        mock_st_class.return_value = mock_model

        generator = EmbeddingGenerator()
        generator.encode(["test"])

        mock_st_class.assert_called_once_with("all-MiniLM-L6-v2")

    @patch("sentence_transformers.SentenceTransformer")
    def test_model_not_reloaded(self, mock_st_class: MagicMock):
        """Test that model is not reloaded on subsequent calls."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 384).astype(np.float32)
        mock_st_class.return_value = mock_model

        generator = EmbeddingGenerator()
        generator.encode(["test1"])
        generator.encode(["test2"])

        # Model should only be created once
        mock_st_class.assert_called_once()
