"""
Compute uncertainty scores for evaluation datasets.

This script estimates model uncertainty for each problem in the evaluation
benchmarks (HumanEval, MBPP). These uncertainty scores will be used to:
1. Analyze correlation with oracle labels
2. Validate that uncertainty is a good proxy for retrieval benefit
3. Visualize the decision boundary for adaptive retrieval

No manual annotation required - fully automated!
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from src.models.base_model import BaseCodeModel
from src.utils.uncertainty import UncertaintyEstimator


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(config: DictConfig):
    """
    Compute uncertainty scores for evaluation datasets.

    Args:
        config: Hydra configuration
    """
    print("=" * 80)
    print("Computing Uncertainty Scores for Evaluation Datasets")
    print("=" * 80)

    # Load base model (no fine-tuning yet, just to estimate uncertainty)
    print("\n1. Loading base model...")
    model = BaseCodeModel(config)

    # Initialize uncertainty estimator
    print("\n2. Initializing uncertainty estimator...")
    estimator = UncertaintyEstimator(model.model, model.tokenizer)

    # Get uncertainty method from config
    uncertainty_method = config.rag.get("uncertainty_method", "entropy")
    print(f"   Using method: {uncertainty_method}")

    # Process each benchmark
    benchmarks = config.get("benchmarks", ["humaneval"])

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

        print(f"   Loaded {len(dataset)} problems")

        # Compute uncertainties
        results = []

        for i, problem in enumerate(tqdm(dataset, desc=f"Computing uncertainties")):
            # Get prompt
            if benchmark_name == "humaneval":
                task_id = problem['task_id']
                prompt = problem['prompt']
            elif benchmark_name == "mbpp":
                task_id = f"mbpp_{problem['task_id']}"
                prompt = problem['text']
            else:
                continue

            # Estimate uncertainty
            try:
                uncertainty = estimator.estimate(
                    prompt,
                    method=uncertainty_method,
                    max_new_tokens=10
                )

                # Get confidence (inverse of uncertainty)
                confidence = estimator.get_confidence_score(prompt, method=uncertainty_method)

                results.append({
                    'task_id': task_id,
                    'uncertainty': uncertainty,
                    'confidence': confidence,
                    'prompt': prompt[:100] + "..."  # Store truncated prompt for reference
                })

            except Exception as e:
                print(f"\n   Error on {task_id}: {e}")
                results.append({
                    'task_id': task_id,
                    'uncertainty': None,
                    'confidence': None,
                    'prompt': prompt[:100] + "..."
                })

        # Save results
        output_dir = config.get("data_dir", "./data")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{benchmark_name}_uncertainties.csv")
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)

        print(f"\n   Saved uncertainties to: {output_path}")

        # Print statistics
        print("\n   Statistics:")
        print(f"   - Mean uncertainty: {df['uncertainty'].mean():.4f}")
        print(f"   - Std uncertainty: {df['uncertainty'].std():.4f}")
        print(f"   - Min uncertainty: {df['uncertainty'].min():.4f}")
        print(f"   - Max uncertainty: {df['uncertainty'].max():.4f}")
        print(f"   - Median uncertainty: {df['uncertainty'].median():.4f}")

        # Print distribution quartiles
        print("\n   Quartiles:")
        print(f"   - Q1 (25%): {df['uncertainty'].quantile(0.25):.4f}")
        print(f"   - Q2 (50%): {df['uncertainty'].quantile(0.50):.4f}")
        print(f"   - Q3 (75%): {df['uncertainty'].quantile(0.75):.4f}")

    print("\n" + "=" * 80)
    print("Uncertainty computation complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run scripts/02_create_oracle.py to create oracle labels")
    print("2. Analyze correlation between uncertainty and oracle decisions")
    print("3. Use uncertainties for adaptive retrieval threshold tuning")


if __name__ == "__main__":
    main()
