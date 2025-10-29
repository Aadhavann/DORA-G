"""
Uncertainty estimation for adaptive retrieval decisions.

This module provides multiple methods for estimating model uncertainty
on code generation queries, which can be used to decide when to retrieve
external code examples vs. rely on parametric knowledge.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer


class UncertaintyEstimator:
    """
    Estimate model uncertainty for adaptive retrieval.

    Supports multiple uncertainty quantification methods:
    - entropy: Token prediction entropy (fast, single forward pass)
    - variance: Logit variance (fast, single forward pass)
    - mc_dropout: Monte Carlo dropout (slower, multiple passes, more accurate)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize uncertainty estimator.

        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def estimate(
        self,
        query: str,
        method: str = "entropy",
        max_new_tokens: int = 10,
        **kwargs
    ) -> float:
        """
        Estimate uncertainty for a query.

        Args:
            query: Input query/prompt
            method: Uncertainty estimation method (entropy, variance, mc_dropout)
            max_new_tokens: Number of tokens to generate for uncertainty estimation
            **kwargs: Additional arguments for specific methods

        Returns:
            Uncertainty score (higher = more uncertain)
        """
        if method == "entropy":
            return self._entropy_uncertainty(query, max_new_tokens)
        elif method == "variance":
            return self._variance_uncertainty(query, max_new_tokens)
        elif method == "mc_dropout":
            n_samples = kwargs.get("n_samples", 5)
            return self._mc_dropout_uncertainty(query, max_new_tokens, n_samples)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")

    def estimate_batch(
        self,
        queries: List[str],
        method: str = "entropy",
        max_new_tokens: int = 10,
        **kwargs
    ) -> List[float]:
        """
        Estimate uncertainty for a batch of queries.

        Args:
            queries: List of input queries
            method: Uncertainty estimation method
            max_new_tokens: Number of tokens to generate
            **kwargs: Additional arguments

        Returns:
            List of uncertainty scores
        """
        return [self.estimate(q, method, max_new_tokens, **kwargs) for q in queries]

    def _entropy_uncertainty(self, query: str, max_new_tokens: int) -> float:
        """
        Entropy-based uncertainty: Average token prediction entropy.

        This measures how "spread out" the probability distribution is
        over the vocabulary. High entropy = model is uncertain about next token.

        Fast (single forward pass) and effective for most cases.
        """
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            # Generate with output_scores to get logits at each step
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Compute entropy for each generated token
            entropies = []
            for score in outputs.scores:  # scores is a tuple of tensors (one per generated token)
                probs = F.softmax(score[0], dim=-1)  # [vocab_size]
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                entropies.append(entropy.item())

            # Return average entropy
            return np.mean(entropies) if entropies else 0.0

    def _variance_uncertainty(self, query: str, max_new_tokens: int) -> float:
        """
        Variance-based uncertainty: Variance of logits across vocabulary.

        High variance = model has strong preferences for some tokens,
        Low variance = model is uncertain, probability mass is spread out.

        Note: Lower variance actually means higher uncertainty, so we
        return negative variance to align with entropy (higher = more uncertain).
        """
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Compute variance for each generated token
            variances = []
            for score in outputs.scores:
                variance = torch.var(score[0])  # Variance across vocabulary
                variances.append(variance.item())

            # Return negative average variance (so higher = more uncertain)
            avg_variance = np.mean(variances) if variances else 0.0
            return -avg_variance if avg_variance != 0 else 1e6  # Avoid negative values

    def _mc_dropout_uncertainty(
        self,
        query: str,
        max_new_tokens: int,
        n_samples: int = 5
    ) -> float:
        """
        Monte Carlo Dropout uncertainty: Variance across multiple stochastic passes.

        This is the most accurate but slowest method. It runs multiple forward
        passes with dropout enabled and measures the variance in predictions.

        High variance = model predictions are inconsistent = high uncertainty.
        """
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True).to(self.device)

        # Enable dropout
        self.model.train()

        # Collect predictions from multiple passes
        all_logits = []

        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                # Stack logits from this pass
                logits = torch.stack([s[0] for s in outputs.scores])  # [seq_len, vocab_size]
                all_logits.append(logits)

        # Restore eval mode
        self.model.eval()

        if not all_logits:
            return 0.0

        # Stack all samples: [n_samples, seq_len, vocab_size]
        all_logits = torch.stack(all_logits)

        # Compute variance across samples for each token
        variance = torch.var(all_logits, dim=0)  # [seq_len, vocab_size]

        # Return average variance across sequence and vocabulary
        return variance.mean().item()

    def get_confidence_score(self, query: str, method: str = "entropy") -> float:
        """
        Get confidence score (inverse of uncertainty).

        Returns value in [0, 1] where 1 = very confident, 0 = very uncertain.
        Useful for reporting/visualization.
        """
        uncertainty = self.estimate(query, method)

        # Normalize uncertainty to [0, 1] range using sigmoid
        # Higher uncertainty -> lower confidence
        confidence = 1.0 / (1.0 + uncertainty)
        return confidence


class OracleLabeler:
    """
    Automatically label when retrieval is beneficial (no manual annotation!).

    This creates "oracle" labels by actually testing if retrieval improves
    performance on each example. Used for:
    1. Validating that uncertainty correlates with retrieval benefit
    2. Computing upper bound performance (Oracle-RAG baseline)
    3. Training/tuning adaptive retrieval threshold
    """

    def __init__(
        self,
        model,
        retriever,
        evaluator,
        max_workers: int = 4
    ):
        """
        Initialize oracle labeler.

        Args:
            model: Code generation model
            retriever: RAG retriever
            evaluator: Function to evaluate generated code (returns score)
            max_workers: Number of parallel workers for labeling
        """
        self.model = model
        self.retriever = retriever
        self.evaluator = evaluator
        self.max_workers = max_workers

    def label(self, problem: Dict) -> Dict:
        """
        Label a single problem: should we retrieve for this?

        Args:
            problem: Problem dict with 'prompt' and 'test' fields

        Returns:
            Dict with oracle label and performance scores
        """
        prompt = problem['prompt']

        # Generate WITHOUT retrieval
        solution_no_rag = self.model.generate([prompt], max_length=512)[0]
        score_no_rag = self.evaluator(solution_no_rag, problem)

        # Generate WITH retrieval
        if self.retriever is not None:
            retrieved_docs = self.retriever.retrieve(prompt, top_k=3)
            context = self._format_retrieved_context(retrieved_docs)
            prompt_with_rag = f"{context}\n\n{prompt}"
        else:
            prompt_with_rag = prompt

        solution_with_rag = self.model.generate([prompt_with_rag], max_length=512)[0]
        score_with_rag = self.evaluator(solution_with_rag, problem)

        # Oracle decision: retrieve if it improves score
        should_retrieve = score_with_rag > score_no_rag

        return {
            'task_id': problem.get('task_id', 'unknown'),
            'should_retrieve': should_retrieve,
            'score_no_rag': score_no_rag,
            'score_with_rag': score_with_rag,
            'improvement': score_with_rag - score_no_rag
        }

    def label_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """
        Label entire dataset (potentially in parallel).

        Args:
            dataset: List of problems

        Returns:
            List of oracle labels
        """
        # For now, sequential (can parallelize later if needed)
        labels = []
        for i, problem in enumerate(dataset):
            print(f"Labeling {i+1}/{len(dataset)}: {problem.get('task_id', 'unknown')}")
            label = self.label(problem)
            labels.append(label)

        return labels

    def _format_retrieved_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents as context."""
        if not retrieved_docs:
            return ""

        context_parts = ["# Retrieved relevant code examples:"]
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"\n# Example {i}:")
            context_parts.append(doc.get('text', doc.get('code', '')))

        return "\n".join(context_parts)


def analyze_uncertainty_oracle_correlation(
    uncertainties: List[float],
    oracle_labels: List[bool]
) -> Dict:
    """
    Analyze correlation between uncertainty and oracle decisions.

    This validates that uncertainty is a good proxy for "should retrieve".

    Args:
        uncertainties: List of uncertainty scores
        oracle_labels: List of oracle decisions (True = should retrieve)

    Returns:
        Dict with correlation metrics (AUC, accuracy at various thresholds)
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    import numpy as np

    # Compute AUC (how well does uncertainty predict oracle?)
    auc = roc_auc_score(oracle_labels, uncertainties)

    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(oracle_labels, uncertainties)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Compute accuracy at optimal threshold
    predictions = [u > optimal_threshold for u in uncertainties]
    accuracy = np.mean([p == o for p, o in zip(predictions, oracle_labels)])

    return {
        'auc': auc,
        'optimal_threshold': optimal_threshold,
        'accuracy_at_optimal': accuracy,
        'precision': np.mean([o for p, o in zip(predictions, oracle_labels) if p]),
        'recall': np.mean([p for p, o in zip(predictions, oracle_labels) if o])
    }
