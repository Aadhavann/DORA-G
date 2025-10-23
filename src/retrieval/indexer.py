"""
FAISS-based indexing for code retrieval.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from omegaconf import DictConfig


class FAISSIndexer:
    """Builds and manages FAISS index for code retrieval."""

    def __init__(self, config: DictConfig, embedder):
        """
        Initialize FAISS indexer.

        Args:
            config: Hydra configuration with FAISS settings
            embedder: CodeEmbedder instance
        """
        self.config = config
        self.embedder = embedder
        self.index_config = config.rag.index

        self.dimension = embedder.get_embedding_dimension()
        self.index = None
        self.documents = []  # Store document metadata

    def build_index(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = True,
    ):
        """
        Build FAISS index from texts.

        Args:
            texts: List of texts to index
            metadata: Optional metadata for each text
            show_progress: Whether to show progress bar
        """
        print(f"Building FAISS index for {len(texts)} documents...")

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedder.encode(
            texts,
            batch_size=self.embedder.batch_size,
            show_progress=show_progress,
        )

        # Convert to float32 for FAISS
        embeddings = embeddings.astype(np.float32)

        # Create FAISS index
        print(f"Creating FAISS index (type: {self.index_config.type})...")
        self.index = self._create_index(embeddings)

        # Store document metadata
        self.documents = metadata if metadata else [{"id": i} for i in range(len(texts))]

        print(f"Index built successfully with {self.index.ntotal} vectors")

    def _create_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create FAISS index based on configuration.

        Args:
            embeddings: Embeddings array

        Returns:
            FAISS index
        """
        if self.index_config.type.upper() == "FLAT":
            # Simple flat index (exact search, slower but accurate)
            if self.index_config.metric == "inner_product":
                index = faiss.IndexFlatIP(self.dimension)
            else:
                index = faiss.IndexFlatL2(self.dimension)

        elif self.index_config.type.upper() == "IVF":
            # IVF index (approximate search, faster)
            nlist = self.index_config.nlist

            if self.index_config.metric == "inner_product":
                quantizer = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            else:
                quantizer = faiss.IndexFlatL2(self.dimension)
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)

            # Train the index
            print("Training IVF index...")
            index.train(embeddings)
            index.nprobe = self.index_config.get("nprobe", 32)

        else:
            raise ValueError(f"Unsupported index type: {self.index_config.type}")

        # Add embeddings to index
        print("Adding vectors to index...")
        index.add(embeddings)

        return index

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar code snippets.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of retrieved documents with scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Encode query
        query_embedding = self.embedder.encode(query)
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):  # Valid index
                result = self.documents[idx].copy()
                result["score"] = float(score)
                results.append(result)

        return results

    def save(self, save_dir: str):
        """
        Save index and metadata to disk.

        Args:
            save_dir: Directory to save index
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = save_path / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        print(f"FAISS index saved to {index_path}")

        # Save metadata
        metadata_path = save_path / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(self.documents, f)
        print(f"Metadata saved to {metadata_path}")

    def load(self, load_dir: str):
        """
        Load index and metadata from disk.

        Args:
            load_dir: Directory containing saved index
        """
        load_path = Path(load_dir)

        # Load FAISS index
        index_path = load_path / "faiss.index"
        self.index = faiss.read_index(str(index_path))
        print(f"FAISS index loaded from {index_path}")

        # Load metadata
        metadata_path = load_path / "metadata.pkl"
        with open(metadata_path, "rb") as f:
            self.documents = pickle.load(f)
        print(f"Metadata loaded from {metadata_path}")

        # Set nprobe for IVF index
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.index_config.get("nprobe", 32)
