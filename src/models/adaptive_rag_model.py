"""
Adaptive DoRA+RAG model with uncertainty-guided retrieval.

This model dynamically decides when to use retrieval based on model uncertainty,
rather than always retrieving for every query.
"""

from typing import List, Dict, Optional, Tuple
from omegaconf import DictConfig
import torch
import numpy as np

from .dora_model import DoRAModel
from ..retrieval.retriever import CodeRetriever
from ..utils.uncertainty import UncertaintyEstimator


class AdaptiveDoRAModel(DoRAModel):
    """
    DoRA model with adaptive retrieval based on uncertainty estimation.

    Key features:
    - Estimates uncertainty for each query
    - Only retrieves when uncertainty exceeds threshold
    - Learns optimal threshold on validation set
    - Tracks retrieval statistics for analysis
    """

    def __init__(self, config: DictConfig):
        """
        Initialize adaptive DoRA+RAG model.

        Args:
            config: Hydra configuration with adaptive RAG settings
        """
        super().__init__(config)

        # Initialize retriever if RAG is enabled
        self.retriever = None
        if config.rag.get("enabled", False):
            self.retriever = CodeRetriever(config)
            print("Initializing adaptive RAG retriever...")

        # Initialize uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            self.model,
            self.tokenizer
        )

        # Adaptive retrieval settings
        self.adaptive_enabled = config.rag.get("adaptive", False)
        self.uncertainty_method = config.rag.get("uncertainty_method", "entropy")
        self.uncertainty_threshold = config.rag.get("uncertainty_threshold", 0.7)
        self.top_k = config.rag.get("retrieval", {}).get("top_k", 3)

        # Oracle mode (for upper bound experiments)
        self.oracle_mode = config.rag.get("oracle_mode", False)
        self.oracle_labels = None
        if self.oracle_mode:
            oracle_path = config.rag.get("oracle_path")
            if oracle_path:
                self.oracle_labels = self._load_oracle_labels(oracle_path)

        # Statistics tracking
        self.reset_stats()

        print(f"Adaptive RAG initialized:")
        print(f"  - Adaptive: {self.adaptive_enabled}")
        print(f"  - Method: {self.uncertainty_method}")
        print(f"  - Threshold: {self.uncertainty_threshold}")
        print(f"  - Oracle mode: {self.oracle_mode}")

    def generate(
        self,
        queries: List[str],
        max_length: int = 512,
        track_stats: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate code with adaptive retrieval.

        Args:
            queries: List of input prompts
            max_length: Maximum generation length
            track_stats: Whether to track retrieval statistics
            **kwargs: Additional generation arguments

        Returns:
            List of generated code strings
        """
        results = []

        for i, query in enumerate(queries):
            # Decide whether to retrieve
            should_retrieve, uncertainty = self._should_retrieve(query, task_id=i)

            # Retrieve if needed
            if should_retrieve and self.retriever is not None:
                retrieved_docs = self.retriever.retrieve(query, top_k=self.top_k)
                prompt = self._format_prompt_with_context(query, retrieved_docs)

                if track_stats:
                    self.stats['retrievals'] += 1
                    self.stats['uncertainties'].append(uncertainty)
            else:
                prompt = query

                if track_stats:
                    self.stats['no_retrievals'] += 1

            # Generate
            output = super().generate([prompt], max_length=max_length, **kwargs)[0]
            results.append(output)

            if track_stats:
                self.stats['total_queries'] += 1

        return results

    def _should_retrieve(
        self,
        query: str,
        task_id: Optional[int] = None
    ) -> Tuple[bool, float]:
        """
        Decide whether to retrieve for this query.

        Args:
            query: Input query
            task_id: Optional task ID for oracle mode

        Returns:
            Tuple of (should_retrieve, uncertainty_score)
        """
        # Oracle mode: use pre-computed labels
        if self.oracle_mode and self.oracle_labels is not None:
            if task_id is not None and task_id < len(self.oracle_labels):
                return self.oracle_labels[task_id], 0.0
            else:
                # Fallback to uncertainty if oracle not available
                pass

        # Adaptive mode: use uncertainty
        if self.adaptive_enabled:
            uncertainty = self.uncertainty_estimator.estimate(
                query,
                method=self.uncertainty_method,
                max_new_tokens=10
            )

            should_retrieve = uncertainty > self.uncertainty_threshold
            return should_retrieve, uncertainty

        # Default: always retrieve (standard RAG)
        return True, 0.0

    def _format_prompt_with_context(
        self,
        query: str,
        retrieved_docs: List[Dict]
    ) -> str:
        """
        Format prompt with retrieved context.

        Uses retrieval-aware formatting with clear delimiters.

        Args:
            query: Original query
            retrieved_docs: Retrieved documents from RAG

        Returns:
            Formatted prompt with context
        """
        if not retrieved_docs:
            return query

        # Build context section
        context_parts = ["# Retrieved relevant code examples:\n"]

        for i, doc in enumerate(retrieved_docs, 1):
            score = doc.get('score', 0.0)
            code = doc.get('text', doc.get('code', ''))

            context_parts.append(f"# Example {i} (relevance: {score:.2f}):")
            context_parts.append(code)
            context_parts.append("")

        context = "\n".join(context_parts)

        # Combine context + query with clear separation
        prompt = f"""{context}

# Task:
{query}

# Your solution:
"""
        return prompt

    def tune_threshold(
        self,
        validation_data: List[Dict],
        evaluator,
        threshold_range: Optional[List[float]] = None
    ) -> float:
        """
        Tune uncertainty threshold on validation set.

        Searches over threshold values to find the one that maximizes
        performance on the validation set.

        Args:
            validation_data: Validation dataset (list of problems)
            evaluator: Function to evaluate generated code
            threshold_range: List of thresholds to try (default: [0.3, 0.5, 0.7, 0.9, 1.1, 1.3])

        Returns:
            Optimal threshold value
        """
        if threshold_range is None:
            threshold_range = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

        print(f"Tuning uncertainty threshold on {len(validation_data)} examples...")

        best_threshold = self.uncertainty_threshold
        best_score = 0.0

        for threshold in threshold_range:
            print(f"\nTrying threshold: {threshold:.2f}")

            # Set threshold
            self.uncertainty_threshold = threshold
            self.reset_stats()

            # Evaluate
            total_score = 0.0
            for problem in validation_data:
                query = problem['prompt']
                generated = self.generate([query], track_stats=True)[0]
                score = evaluator(generated, problem)
                total_score += score

            avg_score = total_score / len(validation_data)
            retrieval_rate = self.get_retrieval_rate()

            print(f"  Score: {avg_score:.3f}, Retrieval rate: {retrieval_rate:.1%}")

            if avg_score > best_score:
                best_score = avg_score
                best_threshold = threshold

        # Set optimal threshold
        self.uncertainty_threshold = best_threshold
        print(f"\nOptimal threshold: {best_threshold:.2f} (score: {best_score:.3f})")

        return best_threshold

    def reset_stats(self):
        """Reset retrieval statistics."""
        self.stats = {
            'total_queries': 0,
            'retrievals': 0,
            'no_retrievals': 0,
            'uncertainties': []
        }

    def get_retrieval_rate(self) -> float:
        """Get percentage of queries that triggered retrieval."""
        if self.stats['total_queries'] == 0:
            return 0.0
        return self.stats['retrievals'] / self.stats['total_queries']

    def get_avg_uncertainty(self) -> float:
        """Get average uncertainty of queries that triggered retrieval."""
        if not self.stats['uncertainties']:
            return 0.0
        return np.mean(self.stats['uncertainties'])

    def get_stats_summary(self) -> Dict:
        """Get summary of retrieval statistics."""
        return {
            'total_queries': self.stats['total_queries'],
            'retrievals': self.stats['retrievals'],
            'no_retrievals': self.stats['no_retrievals'],
            'retrieval_rate': self.get_retrieval_rate(),
            'avg_uncertainty': self.get_avg_uncertainty()
        }

    def _load_oracle_labels(self, path: str) -> List[bool]:
        """Load oracle labels from CSV file."""
        import pandas as pd

        df = pd.read_csv(path)
        return df['should_retrieve'].tolist()

    def save_stats(self, path: str):
        """Save retrieval statistics to file."""
        import json

        stats = self.get_stats_summary()
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Stats saved to {path}")
