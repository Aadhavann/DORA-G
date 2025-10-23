"""
High-level retriever interface combining embedder and indexer.
"""

from typing import List, Dict, Any
from omegaconf import DictConfig
from .embedder import CodeEmbedder
from .indexer import FAISSIndexer


class CodeRetriever:
    """High-level interface for code retrieval."""

    def __init__(self, config: DictConfig):
        """
        Initialize code retriever.

        Args:
            config: Hydra configuration
        """
        self.config = config
        self.embedder = CodeEmbedder(config)
        self.indexer = FAISSIndexer(config, self.embedder)

    def build_index_from_dataset(self, dataset, text_field: str = "text"):
        """
        Build retrieval index from a HuggingFace dataset.

        Args:
            dataset: HuggingFace dataset
            text_field: Field containing text to index
        """
        print("Building retrieval index from dataset...")

        # Extract texts and metadata
        texts = []
        metadata = []

        for i, example in enumerate(dataset):
            text = example.get(text_field, "")
            texts.append(text)

            # Store full example as metadata
            meta = {k: v for k, v in example.items() if k != text_field}
            meta["id"] = i
            metadata.append(meta)

        # Build index
        self.indexer.build_index(texts, metadata)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant code snippets for a query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of retrieved documents with metadata and scores
        """
        return self.indexer.search(query, top_k)

    def save(self, save_dir: str):
        """
        Save retriever state.

        Args:
            save_dir: Directory to save to
        """
        self.indexer.save(save_dir)

    def load(self, load_dir: str):
        """
        Load retriever state.

        Args:
            load_dir: Directory to load from
        """
        self.indexer.load(load_dir)
