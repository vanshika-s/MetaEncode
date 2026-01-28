# src/ml/embeddings.py
"""Text embedding generation using Sentence Transformers (SBERT).

This module provides functionality for generating semantic embeddings
from text metadata fields using pre-trained SBERT models.

Reference: https://www.sbert.net/
"""

from typing import Any, Optional

import numpy as np


class EmbeddingGenerator:
    """Generate text embeddings using Sentence Transformers.

    This class wraps SBERT models to generate dense vector representations
    of text fields like descriptions and titles. These embeddings capture
    semantic meaning, allowing similar descriptions to have similar vectors.

    Example:
        >>> generator = EmbeddingGenerator()
        >>> embeddings = generator.encode(["RNA-seq of human liver cells"])
        >>> print(embeddings.shape)  # (1, 384) for MiniLM model
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    # Known embedding dimensions for common models
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
    }

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize the embedding generator.

        Args:
            model_name: Name of the SBERT model to use. Defaults to MiniLM.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model: Optional[Any] = None  # Lazy loading

    def _load_model(self) -> None:
        """Load the SBERT model (lazy initialization).

        This method loads the model only when first needed to avoid
        slow startup times.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to encode.
            batch_size: Number of texts to process at once.
            show_progress: Whether to show progress bar.

        Returns:
            NumPy array of shape (n_texts, embedding_dim).
        """
        self._load_model()
        assert self._model is not None  # Type guard after _load_model()

        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        # Handle empty strings by replacing with a space
        processed_texts = [t if t.strip() else " " for t in texts]

        embeddings = self._model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text string to encode.

        Returns:
            NumPy array of shape (embedding_dim,).
        """
        self._load_model()
        assert self._model is not None  # Type guard after _load_model()

        # Handle empty string
        if not text.strip():
            text = " "

        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
        )

        return np.asarray(embedding, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of the embeddings.

        Returns:
            The embedding dimension (e.g., 384 for MiniLM).
        """
        # Try to get from known dimensions first (avoids loading model)
        if self.model_name in self.MODEL_DIMENSIONS:
            return self.MODEL_DIMENSIONS[self.model_name]

        # Otherwise, load model and check
        self._load_model()
        assert self._model is not None  # Type guard after _load_model()
        return int(self._model.get_sentence_embedding_dimension())
