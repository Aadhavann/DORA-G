"""
Custom unseen-API test set for evaluating RAG on new library features.
This is a key differentiator for the paper!
"""

from typing import Dict, Any, List
import json
from pathlib import Path
from .executor import CodeExecutor
from .metrics import CodeMetrics


class UnseenAPIEvaluator:
    """
    Evaluator for custom unseen-API test set.

    This tests the model's ability to use library features that were likely
    not in the pre-training data, making it an ideal testbed for RAG.
    """

    def __init__(self, timeout: int = 5):
        """
        Initialize unseen-API evaluator.

        Args:
            timeout: Execution timeout in seconds
        """
        self.executor = CodeExecutor(timeout=timeout)

    def create_sample_test_set(self) -> List[Dict[str, Any]]:
        """
        Create a sample unseen-API test set.

        This is a placeholder - you should expand this with real problems
        using recent library features (e.g., Python 3.11+, pandas 2.0+, etc.).

        Returns:
            List of test problems
        """
        sample_problems = [
            {
                "task_id": "unseen_api_001",
                "library": "pandas",
                "feature": "pandas 2.0+ pyarrow dtype",
                "prompt": """Write a Python function that creates a DataFrame with PyArrow-backed string dtype (new in pandas 2.0).
The function should take a list of strings and return a DataFrame with a single column 'text' using dtype='string[pyarrow]'.

Example:
>>> create_pyarrow_df(['hello', 'world'])
    text
0  hello
1  world
""",
                "solution": """import pandas as pd

def create_pyarrow_df(strings):
    return pd.DataFrame({'text': strings}, dtype='string[pyarrow]')
""",
                "test": """
import pandas as pd

def create_pyarrow_df(strings):
    return pd.DataFrame({'text': strings}, dtype='string[pyarrow]')

# Test
df = create_pyarrow_df(['hello', 'world'])
assert len(df) == 2
assert df['text'].dtype == pd.ArrowDtype(pd.StringDtype('pyarrow'))
assert list(df['text']) == ['hello', 'world']
print("Test passed!")
""",
            },
            {
                "task_id": "unseen_api_002",
                "library": "python",
                "feature": "match statement (Python 3.10+)",
                "prompt": """Write a function that uses Python's match statement (structural pattern matching) to classify a value.
The function should:
- Return 'zero' for 0
- Return 'one' for 1
- Return 'small' for numbers 2-10
- Return 'large' for anything else

Example:
>>> classify_number(0)
'zero'
>>> classify_number(5)
'small'
""",
                "solution": """def classify_number(n):
    match n:
        case 0:
            return 'zero'
        case 1:
            return 'one'
        case n if 2 <= n <= 10:
            return 'small'
        case _:
            return 'large'
""",
                "test": """
def classify_number(n):
    match n:
        case 0:
            return 'zero'
        case 1:
            return 'one'
        case n if 2 <= n <= 10:
            return 'small'
        case _:
            return 'large'

# Test
assert classify_number(0) == 'zero'
assert classify_number(1) == 'one'
assert classify_number(5) == 'small'
assert classify_number(100) == 'large'
print("Test passed!")
""",
            },
        ]

        return sample_problems

    def load_test_set(self, path: str = None) -> List[Dict[str, Any]]:
        """
        Load test set from file or create sample.

        Args:
            path: Optional path to test set JSON file

        Returns:
            List of test problems
        """
        if path and Path(path).exists():
            with open(path, 'r') as f:
                return json.load(f)
        else:
            print("Using sample unseen-API test set")
            return self.create_sample_test_set()

    def evaluate(
        self,
        model,
        test_set_path: str = None,
        num_samples_per_task: int = 1,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evaluate model on unseen-API test set.

        Args:
            model: Model with generate() method
            test_set_path: Path to test set (optional)
            num_samples_per_task: Number of samples per problem
            temperature: Sampling temperature

        Returns:
            Evaluation results
        """
        test_set = self.load_test_set(test_set_path)
        print(f"Evaluating on Unseen-API test set ({len(test_set)} problems)...")

        results = []

        for problem in test_set:
            task_id = problem["task_id"]
            prompt = problem["prompt"]

            # Generate completions
            for _ in range(num_samples_per_task):
                completion = model.generate(
                    prompt=prompt,
                    max_new_tokens=512,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )

                # Check correctness
                test_code = problem.get("test", "")

                if test_code:
                    exec_result = self.executor.execute_code(
                        code=completion,
                        test_cases=test_code,
                    )
                    passed = exec_result["passed"]
                else:
                    passed = False

                results.append({
                    "task_id": task_id,
                    "library": problem.get("library"),
                    "feature": problem.get("feature"),
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
        """Calculate metrics."""
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

        return {
            "pass@1": total_pass_at_1 / num_tasks if num_tasks > 0 else 0.0,
        }
