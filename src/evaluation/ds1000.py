"""
DS-1000 (Data Science) benchmark evaluation.
This tests library-specific knowledge - perfect for RAG evaluation.
"""

from typing import Dict, Any, List
from tqdm import tqdm
from .executor import CodeExecutor
from .metrics import CodeMetrics


class DS1000Evaluator:
    """Evaluator for DS-1000 benchmark."""

    def __init__(self, timeout: int = 10):
        """
        Initialize DS-1000 evaluator.

        Args:
            timeout: Execution timeout in seconds (longer for DS tasks)
        """
        self.executor = CodeExecutor(timeout=timeout)

    def evaluate(
        self,
        model,
        dataset,
        num_samples_per_task: int = 1,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evaluate model on DS-1000.

        Args:
            model: Model with generate() method
            dataset: DS-1000 dataset
            num_samples_per_task: Number of samples per problem
            temperature: Sampling temperature

        Returns:
            Evaluation results
        """
        if dataset is None:
            print("DS-1000 dataset not available. Skipping evaluation.")
            return {
                "results": [],
                "metrics": {"pass@1": 0.0},
            }

        print(f"Evaluating on DS-1000 with {num_samples_per_task} samples per task...")

        results = []

        for i, problem in enumerate(tqdm(dataset, desc="DS-1000")):
            # DS-1000 format varies; adapt as needed
            prompt = problem.get("prompt", problem.get("question", ""))
            library = problem.get("library", "unknown")

            # Generate completions
            for _ in range(num_samples_per_task):
                completion = model.generate(
                    prompt=prompt,
                    max_new_tokens=512,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )

                # Check correctness if test available
                test_code = problem.get("test", problem.get("test_case", ""))

                if test_code:
                    exec_result = self.executor.execute_code(
                        code=completion,
                        test_cases=test_code,
                    )
                    passed = exec_result["passed"]
                else:
                    # No test available, mark as not passed
                    passed = False

                results.append({
                    "task_id": i,
                    "library": library,
                    "completion": completion,
                    "passed": passed,
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
        """Calculate pass@k metrics overall and per-library."""
        # Overall metrics
        tasks = {}
        for result in results:
            task_id = result["task_id"]
            if task_id not in tasks:
                tasks[task_id] = []
            tasks[task_id].append(result["passed"])

        total_pass_at_1 = 0
        for task_id, passes in tasks.items():
            n = len(passes)
            c = sum(passes)
            if n >= 1:
                total_pass_at_1 += CodeMetrics.pass_at_k(n, c, 1)

        num_tasks = len(tasks)

        metrics = {
            "pass@1": total_pass_at_1 / num_tasks if num_tasks > 0 else 0.0,
        }

        # Per-library breakdown
        library_results = {}
        for result in results:
            lib = result.get("library", "unknown")
            if lib not in library_results:
                library_results[lib] = []
            library_results[lib].append(result["passed"])

        library_metrics = {}
        for lib, passes in library_results.items():
            accuracy = sum(passes) / len(passes) if passes else 0.0
            library_metrics[f"{lib}_accuracy"] = accuracy

        metrics.update(library_metrics)

        return metrics
