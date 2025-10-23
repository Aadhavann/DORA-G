"""
Safe code execution for evaluation.
"""

import subprocess
import tempfile
import os
from typing import Dict, Any, Optional
import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    """Raised when code execution times out."""
    pass


@contextmanager
def time_limit(seconds: int):
    """
    Context manager for timing out code execution.

    Args:
        seconds: Timeout in seconds
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Execution timed out")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class CodeExecutor:
    """Safely executes generated code for evaluation."""

    def __init__(self, timeout: int = 5):
        """
        Initialize code executor.

        Args:
            timeout: Timeout in seconds for each execution
        """
        self.timeout = timeout

    def execute_code(
        self,
        code: str,
        test_cases: Optional[str] = None,
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Execute code and return results.

        Args:
            code: Code to execute
            test_cases: Optional test cases to run
            language: Programming language (currently only Python)

        Returns:
            Dictionary with execution results
        """
        if language.lower() != "python":
            raise NotImplementedError(f"Language {language} not supported yet")

        # Combine code with test cases
        full_code = code
        if test_cases:
            full_code = f"{code}\n\n{test_cases}"

        try:
            # Execute in subprocess for safety
            result = self._execute_python_subprocess(full_code)
            return result

        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "stdout": "",
                "stderr": str(e),
            }

    def _execute_python_subprocess(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in a subprocess.

        Args:
            code: Python code to execute

        Returns:
            Execution results
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Run in subprocess with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            success = result.returncode == 0

            return {
                "passed": success,
                "error": None if success else result.stderr,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "error": f"Execution timed out after {self.timeout}s",
                "stdout": "",
                "stderr": "Timeout",
                "returncode": -1,
            }

        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
            }

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

    def check_correctness(
        self,
        problem: Dict[str, Any],
        completion: str,
    ) -> Dict[str, Any]:
        """
        Check if a completion passes test cases.

        Args:
            problem: Problem dict with test cases
            completion: Generated code

        Returns:
            Results dict
        """
        # Extract test cases from problem
        test_code = problem.get("test", "")

        # Execute code with tests
        result = self.execute_code(completion, test_code)

        return result
