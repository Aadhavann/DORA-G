"""
Create oracle labels for adaptive retrieval.

This script automatically labels when retrieval is beneficial by:
1. Generating code WITHOUT retrieval
2. Generating code WITH retrieval
3. Evaluating both solutions
4. Labeling retrieval as beneficial if it improves performance

No manual annotation required - fully automated!

This creates the "upper bound" for adaptive retrieval and validates
that uncertainty is a good proxy for retrieval benefit.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import json

from src.models.base_model import BaseCodeModel
from src.retrieval.retriever import CodeRetriever
from src.evaluation.executor import CodeExecutor


def evaluate_solution(solution: str, problem: dict, benchmark: str) -> float:
    """
    Evaluate a generated solution.

    Returns 1.0 if solution passes tests, 0.0 otherwise.

    Args:
        solution: Generated code
        problem: Problem dictionary with test cases
        benchmark: Benchmark name (humaneval, mbpp)

    Returns:
        Score (0.0 or 1.0)
    """
    executor = CodeExecutor(timeout=5)

    try:
        if benchmark == "humaneval":
            # Combine solution with test
            code_to_test = solution + "\n\n" + problem['test']
        elif benchmark == "mbpp":
            # MBPP format
            code_to_test = solution + "\n\n" + "\n".join(problem['test_list'])
        else:
            return 0.0

        # Execute and check if it passes
        result = executor.execute(code_to_test)

        if result['status'] == 'success':
            return 1.0
        else:
            return 0.0

    except Exception as e:
        return 0.0


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(config: DictConfig):
    """
    Create oracle labels for adaptive retrieval.

    Args:
        config: Hydra configuration
    """
    print("=" * 80)
    print("Creating Oracle Labels for Adaptive Retrieval")
    print("=" * 80)

    # Load base model
    print("\n1. Loading base model...")
    model = BaseCodeModel(config)

    # Load retriever (if RAG enabled)
    retriever = None
    if config.rag.get("enabled", False):
        print("\n2. Loading retriever...")
        retriever = CodeRetriever(config)

        # Load index
        index_path = config.rag.index.get("index_path", "./data/faiss_index")
        if os.path.exists(index_path):
            retriever.load(index_path)
            print(f"   Loaded index from: {index_path}")
        else:
            print(f"   WARNING: Index not found at {index_path}")
            print(f"   Run scripts/02_build_index.py first!")
            return

    # Process each benchmark
    benchmarks = config.get("benchmarks", ["humaneval"])

    # Limit number of examples for faster oracle creation (optional)
    max_examples = config.get("oracle_max_examples", None)

    for benchmark_name in benchmarks:
        print(f"\n3. Processing {benchmark_name}...")

        # Load dataset
        if benchmark_name == "humaneval":
            dataset = load_dataset("openai_humaneval", split="test")
        elif benchmark_name == "mbpp":
            dataset = load_dataset("mbpp", split="test")
        else:
            print(f"   Skipping unknown benchmark: {benchmark_name}")
            continue

        # Limit examples if specified
        if max_examples is not None:
            dataset = dataset.select(range(min(max_examples, len(dataset))))

        print(f"   Processing {len(dataset)} problems")

        # Create oracle labels
        oracle_labels = []

        for i, problem in enumerate(tqdm(dataset, desc="Creating oracle labels")):
            # Get task info
            if benchmark_name == "humaneval":
                task_id = problem['task_id']
                prompt = problem['prompt']
            elif benchmark_name == "mbpp":
                task_id = f"mbpp_{problem['task_id']}"
                prompt = problem['text']
            else:
                continue

            try:
                # Generate WITHOUT retrieval
                solution_no_rag = model.generate([prompt], max_length=512)[0]
                score_no_rag = evaluate_solution(solution_no_rag, problem, benchmark_name)

                # Generate WITH retrieval
                if retriever is not None:
                    retrieved_docs = retriever.retrieve(prompt, top_k=3)
                    context = format_context(retrieved_docs)
                    prompt_with_rag = f"{context}\n\n{prompt}"
                else:
                    prompt_with_rag = prompt

                solution_with_rag = model.generate([prompt_with_rag], max_length=512)[0]
                score_with_rag = evaluate_solution(solution_with_rag, problem, benchmark_name)

                # Oracle decision: retrieve if it improves score
                should_retrieve = score_with_rag > score_no_rag
                improvement = score_with_rag - score_no_rag

                oracle_labels.append({
                    'task_id': task_id,
                    'should_retrieve': should_retrieve,
                    'score_no_rag': score_no_rag,
                    'score_with_rag': score_with_rag,
                    'improvement': improvement,
                    'prompt': prompt[:100] + "..."
                })

            except Exception as e:
                print(f"\n   Error on {task_id}: {e}")
                oracle_labels.append({
                    'task_id': task_id,
                    'should_retrieve': False,
                    'score_no_rag': 0.0,
                    'score_with_rag': 0.0,
                    'improvement': 0.0,
                    'prompt': prompt[:100] + "..."
                })

        # Save oracle labels
        output_dir = config.get("data_dir", "./data")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{benchmark_name}_oracle.csv")
        df = pd.DataFrame(oracle_labels)
        df.to_csv(output_path, index=False)

        print(f"\n   Saved oracle labels to: {output_path}")

        # Print statistics
        print("\n   Oracle Statistics:")
        retrieval_beneficial = df['should_retrieve'].sum()
        total = len(df)
        print(f"   - Retrieval beneficial: {retrieval_beneficial}/{total} ({retrieval_beneficial/total*100:.1f}%)")
        print(f"   - Avg score (no RAG): {df['score_no_rag'].mean():.3f}")
        print(f"   - Avg score (with RAG): {df['score_with_rag'].mean():.3f}")
        print(f"   - Avg improvement: {df['improvement'].mean():.3f}")

        # Analyze when retrieval helps
        helpful_cases = df[df['should_retrieve'] == True]
        if len(helpful_cases) > 0:
            print(f"\n   When retrieval helps:")
            print(f"   - Avg improvement: {helpful_cases['improvement'].mean():.3f}")
            print(f"   - Min improvement: {helpful_cases['improvement'].min():.3f}")
            print(f"   - Max improvement: {helpful_cases['improvement'].max():.3f}")

    print("\n" + "=" * 80)
    print("Oracle creation complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Analyze correlation: python scripts/03_analyze_uncertainty_oracle.py")
    print("2. Train adaptive models with learned thresholds")


def format_context(retrieved_docs):
    """Format retrieved documents as context string."""
    if not retrieved_docs:
        return ""

    context_parts = ["# Retrieved relevant code examples:\n"]
    for i, doc in enumerate(retrieved_docs, 1):
        code = doc.get('text', doc.get('code', ''))
        context_parts.append(f"# Example {i}:")
        context_parts.append(code)
        context_parts.append("")

    return "\n".join(context_parts)


if __name__ == "__main__":
    main()
