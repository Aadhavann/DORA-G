"""
Evaluation metrics for code generation.
"""

from typing import List, Dict, Any
import numpy as np
from collections import Counter
from itertools import combinations


class CodeMetrics:
    """Collection of metrics for code generation evaluation."""

    @staticmethod
    def pass_at_k(n: int, c: int, k: int) -> float:
        """
        Calculate pass@k metric.

        Args:
            n: Total number of samples
            c: Number of correct samples
            k: k in pass@k

        Returns:
            pass@k score
        """
        if n - c < k:
            return 1.0

        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    @staticmethod
    def calculate_pass_at_k(
        results: List[Dict[str, Any]],
        k_values: List[int] = [1, 10, 100]
    ) -> Dict[str, float]:
        """
        Calculate pass@k for multiple k values.

        Args:
            results: List of results, each with 'passed' field
            k_values: List of k values to compute

        Returns:
            Dictionary with pass@k scores
        """
        total = len(results)
        correct = sum(1 for r in results if r.get("passed", False))

        pass_at_k = {}
        for k in k_values:
            if k > total:
                continue
            score = CodeMetrics.pass_at_k(total, correct, k)
            pass_at_k[f"pass@{k}"] = score

        return pass_at_k

    @staticmethod
    def exact_match(prediction: str, reference: str) -> float:
        """
        Calculate exact match score.

        Args:
            prediction: Predicted code
            reference: Reference code

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        # Normalize whitespace
        pred_normalized = " ".join(prediction.split())
        ref_normalized = " ".join(reference.split())

        return 1.0 if pred_normalized == ref_normalized else 0.0

    @staticmethod
    def token_overlap(prediction: str, reference: str) -> float:
        """
        Calculate token-level overlap (similar to BLEU unigrams).

        Args:
            prediction: Predicted code
            reference: Reference code

        Returns:
            Overlap score [0, 1]
        """
        pred_tokens = set(prediction.split())
        ref_tokens = set(reference.split())

        if not ref_tokens:
            return 0.0

        overlap = len(pred_tokens & ref_tokens)
        return overlap / len(ref_tokens)


class RetrievalMetrics:
    """Metrics for evaluating retrieval quality."""

    @staticmethod
    def recall_at_k(retrieved: List[Any], relevant: List[Any], k: int) -> float:
        """
        Calculate recall@k.

        Args:
            retrieved: List of retrieved items (top-k)
            relevant: List of relevant items
            k: k value

        Returns:
            Recall@k score
        """
        if not relevant:
            return 0.0

        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)

        hits = len(retrieved_set & relevant_set)
        return hits / len(relevant_set)

    @staticmethod
    def precision_at_k(retrieved: List[Any], relevant: List[Any], k: int) -> float:
        """
        Calculate precision@k.

        Args:
            retrieved: List of retrieved items (top-k)
            relevant: List of relevant items
            k: k value

        Returns:
            Precision@k score
        """
        if k == 0:
            return 0.0

        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)

        hits = len(retrieved_set & relevant_set)
        return hits / k

    @staticmethod
    def mean_reciprocal_rank(retrieved_lists: List[List[Any]], relevant_lists: List[List[Any]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        Args:
            retrieved_lists: List of retrieved item lists
            relevant_lists: List of relevant item lists

        Returns:
            MRR score
        """
        reciprocal_ranks = []

        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            relevant_set = set(relevant)

            for i, item in enumerate(retrieved, 1):
                if item in relevant_set:
                    reciprocal_ranks.append(1.0 / i)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def calculate_codebleu(predictions: List[str], references: List[str]) -> float:
    """
    Calculate CodeBLEU score (wrapper for codebleu library).

    Args:
        predictions: List of predicted code
        references: List of reference code

    Returns:
        CodeBLEU score
    """
    try:
        from codebleu import calc_codebleu

        # CodeBLEU expects references as list of lists
        references_formatted = [[ref] for ref in references]

        result = calc_codebleu(
            references_formatted,
            predictions,
            lang="python",
            weights=(0.25, 0.25, 0.25, 0.25),
        )

        return result["codebleu"]

    except ImportError:
        print("Warning: codebleu library not installed. Skipping CodeBLEU calculation.")
        return 0.0
    except Exception as e:
        print(f"Warning: CodeBLEU calculation failed: {e}")
        return 0.0
