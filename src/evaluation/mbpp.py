"""
MBPP (Mostly Basic Python Problems) benchmark evaluation.
"""

from typing import Dict, Any, List
from tqdm import tqdm
from .executor import CodeExecutor
from .metrics import CodeMetrics


class MBPPEvaluator:
    """Evaluator for MBPP benchmark."""

    def __init__(self, timeout: int = 5):
        """
        Initialize MBPP evaluator.

        Args:
            timeout: Execution timeout in seconds
        """
        self.executor = CodeExecutor(timeout=timeout)

    def _format_prompt(self, problem: Dict[str, Any]) -> str:
        """
        Format MBPP problem as prompt.

        Args:
            problem: MBPP problem dict

        Returns:
            Formatted prompt
        """
        text = problem["text"]
        prompt = f"Write a Python function to {text.lower()}\n\n"

        # Add test cases as examples (optional)
        if "test_list" in problem and len(problem["test_list"]) > 0:
            prompt += "Examples:\n"
            for test in problem["test_list"][:2]:  # Show first 2 tests
                prompt += f"- {test}\n"
            prompt += "\n"

        prompt += "Solution:\n"
        return prompt

    def evaluate(
        self,
        model,
        dataset,
        num_samples_per_task: int = 1,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evaluate model on MBPP.

        Args:
            model: Model with generate() method
            dataset: MBPP dataset
            num_samples_per_task: Number of samples per problem
            temperature: Sampling temperature

        Returns:
            Evaluation results
        """
        print(f"Evaluating on MBPP with {num_samples_per_task} samples per task...")

        results = []

        for problem in tqdm(dataset, desc="MBPP"):
            task_id = problem["task_id"]
            prompt = self._format_prompt(problem)

            # Generate completions
            for _ in range(num_samples_per_task):
                completion = model.generate(
                    prompt=prompt,
                    max_new_tokens=512,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )

                # Check correctness
                test_cases = problem.get("test_list", [])
                test_code = "\n".join(test_cases)

                # Execute generated code with tests
                exec_result = self.executor.execute_code(
                    code=completion,
                    test_cases=test_code,
                )

                results.append({
                    "task_id": task_id,
                    "completion": completion,
                    "passed": exec_result["passed"],
                    "error": exec_result.get("error"),
                })

        # Calculate metrics
        metrics = self._calculate_metrics(results, num_samples_per_task)

        return {
            "results": results,
            "metrics": metrics,
        }

    def _calculate_metrics(
        self,
        results: List[Dict[str, Any]],
        num_samples: int
    ) -> Dict[str, float]:
        """Calculate pass@k metrics."""
        # Group by task
        tasks = {}
        for result in results:
            task_id = result["task_id"]
            if task_id not in tasks:
                tasks[task_id] = []
            tasks[task_id].append(result["passed"])

        # Calculate pass@k for each task
        total_pass_at_1 = 0
        total_pass_at_k = 0

        for task_id, passes in tasks.items():
            n = len(passes)
            c = sum(passes)

            # Pass@1
            if n >= 1:
                total_pass_at_1 += CodeMetrics.pass_at_k(n, c, 1)

            # Pass@k (if enough samples)
            if n >= min(num_samples, 10):
                k = min(num_samples, 10)
                total_pass_at_k += CodeMetrics.pass_at_k(n, c, k)

        num_tasks = len(tasks)

        metrics = {
            "pass@1": total_pass_at_1 / num_tasks if num_tasks > 0 else 0.0,
        }

        if num_samples >= 10:
            metrics["pass@10"] = total_pass_at_k / num_tasks if num_tasks > 0 else 0.0

        return metrics
