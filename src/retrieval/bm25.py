"""
BM25 sparse retrieval baseline (optional).
"""

from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import pickle
from pathlib import Path


class BM25Retriever:
    """BM25-based sparse retrieval for baseline comparison."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.

        Args:
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (can be improved with code-specific tokenizer).

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Simple whitespace + punctuation tokenization
        # For better results, consider using a code-aware tokenizer
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens

    def build_index(self, texts: List[str], metadata: List[Dict[str, Any]] = None):
        """
        Build BM25 index from texts.

        Args:
            texts: List of texts to index
            metadata: Optional metadata for each text
        """
        print(f"Building BM25 index for {len(texts)} documents...")

        self.tokenized_corpus = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        self.documents = metadata if metadata else [{"id": i} for i in range(len(texts))]

        print("BM25 index built successfully")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents using BM25.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of retrieved documents with scores
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        # Format results
        results = []
        for idx in top_indices:
            result = self.documents[idx].copy()
            result["score"] = float(scores[idx])
            results.append(result)

        return results

    def save(self, save_dir: str):
        """Save BM25 index."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / "bm25.pkl", "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "documents": self.documents,
                "tokenized_corpus": self.tokenized_corpus,
            }, f)

    def load(self, load_dir: str):
        """Load BM25 index."""
        load_path = Path(load_dir)

        with open(load_path / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.documents = data["documents"]
            self.tokenized_corpus = data["tokenized_corpus"]
