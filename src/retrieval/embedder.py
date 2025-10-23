"""
Code embedding model for semantic search.
"""

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
from omegaconf import DictConfig


class CodeEmbedder:
    """Generates embeddings for code snippets using sentence-transformers."""

    def __init__(self, config: DictConfig):
        """
        Initialize code embedder.

        Args:
            config: Hydra configuration with RAG embedder settings
        """
        self.config = config
        self.model_name = config.rag.embedder.model_name
        self.dimension = config.rag.embedder.dimension
        self.batch_size = config.rag.embedder.batch_size
        self.device = config.rag.embedder.get("device", "cuda")
        self.normalize = config.rag.embedder.get("normalize_embeddings", True)

        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.model.to(self.device)
        print(f"Embedder loaded on {self.device}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.batch_size

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            device=self.device,
        )

        return embeddings

    def encode_batch_iterator(
        self,
        texts_iterator,
        batch_size: int = None,
        show_progress: bool = True,
    ):
        """
        Encode texts from an iterator in batches (memory-efficient).

        Args:
            texts_iterator: Iterator of texts
            batch_size: Batch size for encoding
            show_progress: Whether to show progress

        Yields:
            Batches of embeddings
        """
        batch_size = batch_size or self.batch_size
        batch = []

        for text in texts_iterator:
            batch.append(text)

            if len(batch) >= batch_size:
                embeddings = self.encode(batch, batch_size=batch_size)
                yield embeddings
                batch = []

        # Process remaining items
        if batch:
            embeddings = self.encode(batch, batch_size=len(batch))
            yield embeddings

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings.

        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
