"""
Script 4: Run all 6 experimental conditions (ablation study).
"""

import sys
sys.path.append("..")

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datasets import load_from_disk
from src.models.base_model import BaseCodeModel
from src.models.dora_model import DoRAModel
from src.models.lora_model import LoRAModel
from src.models.rag_model import RAGModel
from src.retrieval.retriever import CodeRetriever
from src.evaluation.humaneval import HumanEvalEvaluator
from src.evaluation.mbpp import MBPPEvaluator
from src.evaluation.ds1000 import DS1000Evaluator
from src.evaluation.unseen_api import UnseenAPIEvaluator
from src.utils.logging import ExperimentLogger
from src.utils.reproducibility import set_seed
import json


EXPERIMENTS = [
    "baseline",      # 1. No FT, No RAG
    "lora_only",     # 2. LoRA FT, No RAG
    "dora_only",     # 3. DoRA FT, No RAG
    "rag_only",      # 4. No FT, RAG
    "lora_rag",      # 5. LoRA FT + RAG
    "dora_rag",      # 6. DoRA FT + RAG (MAIN)
]


def load_model(config: DictConfig, experiment_name: str):
    """Load model based on experiment configuration."""
    print(f"\n--- Loading Model for {experiment_name} ---")

    # Determine model type
    has_peft = config.peft.get("enabled", False)
    has_rag = config.rag.get("enabled", False)
    peft_method = config.peft.get("method", "none")

    # Load base model or fine-tuned model
    if not has_peft:
        # Baseline model
        model = BaseCodeModel(config)
    elif peft_method == "dora":
        # Check if checkpoint exists
        checkpoint_path = Path(config.training.output_dir) / "final_checkpoint"
        if checkpoint_path.exists():
            model = DoRAModel.from_pretrained_adapter(config, str(checkpoint_path))
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}. Using base model.")
            model = BaseCodeModel(config)
    elif peft_method == "lora":
        checkpoint_path = Path(config.training.output_dir) / "final_checkpoint"
        if checkpoint_path.exists():
            model = LoRAModel.from_pretrained_adapter(config, str(checkpoint_path))
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}. Using base model.")
            model = BaseCodeModel(config)
    else:
        model = BaseCodeModel(config)

    # Wrap with RAG if needed
    if has_rag:
        print("Loading retriever for RAG...")
        retriever = CodeRetriever(config)
        index_path = Path(config.rag.index.index_path)
        if index_path.exists():
            retriever.load(str(index_path))
            model = RAGModel(model, retriever, config)
        else:
            print(f"Warning: RAG index not found at {index_path}. RAG will not be used.")

    return model


def evaluate_model(model, config: DictConfig, experiment_name: str):
    """Run all evaluations for a model."""
    print(f"\n--- Evaluating {experiment_name} ---")

    results = {}
    data_dir = Path(config.paths.data_dir)

    # 1. HumanEval
    if (data_dir / "humaneval").exists():
        print("\nRunning HumanEval...")
        humaneval_dataset = load_from_disk(str(data_dir / "humaneval"))
        evaluator = HumanEvalEvaluator()
        humaneval_results = evaluator.evaluate(
            model=model,
            dataset=humaneval_dataset,
            num_samples_per_task=config.evaluation.num_samples,
            temperature=config.evaluation.temperature,
        )
        results["humaneval"] = humaneval_results["metrics"]
        print(f"  HumanEval Pass@1: {humaneval_results['metrics']['pass@1']:.4f}")
    else:
        print("Skipping HumanEval (dataset not found)")

    # 2. MBPP
    if (data_dir / "mbpp").exists():
        print("\nRunning MBPP...")
        mbpp_dataset = load_from_disk(str(data_dir / "mbpp"))
        evaluator = MBPPEvaluator()
        mbpp_results = evaluator.evaluate(
            model=model,
            dataset=mbpp_dataset,
            num_samples_per_task=config.evaluation.num_samples,
            temperature=config.evaluation.temperature,
        )
        results["mbpp"] = mbpp_results["metrics"]
        print(f"  MBPP Pass@1: {mbpp_results['metrics']['pass@1']:.4f}")
    else:
        print("Skipping MBPP (dataset not found)")

    # 3. DS-1000
    if (data_dir / "ds1000").exists():
        print("\nRunning DS-1000...")
        ds1000_dataset = load_from_disk(str(data_dir / "ds1000"))
        evaluator = DS1000Evaluator()
        ds1000_results = evaluator.evaluate(
            model=model,
            dataset=ds1000_dataset,
            num_samples_per_task=config.evaluation.num_samples,
            temperature=config.evaluation.temperature,
        )
        results["ds1000"] = ds1000_results["metrics"]
        print(f"  DS-1000 Pass@1: {ds1000_results['metrics']['pass@1']:.4f}")
    else:
        print("Skipping DS-1000 (dataset not found)")

    # 4. Unseen-API
    print("\nRunning Unseen-API...")
    evaluator = UnseenAPIEvaluator()
    unseen_results = evaluator.evaluate(
        model=model,
        num_samples_per_task=config.evaluation.num_samples,
        temperature=config.evaluation.temperature,
    )
    results["unseen_api"] = unseen_results["metrics"]
    print(f"  Unseen-API Pass@1: {unseen_results['metrics']['pass@1']:.4f}")

    return results


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(config: DictConfig):
    """Run all ablation experiments."""
    print("="*80)
    print("STEP 4: Running Ablation Study")
    print("="*80)

    # Set seed
    set_seed(config.reproducibility.seed)

    # Store all results
    all_results = {}

    # Run each experiment
    for exp_name in EXPERIMENTS:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"{'='*80}")

        # Load experiment config
        exp_config_path = Path("configs/experiments") / f"{exp_name}.yaml"
        if exp_config_path.exists():
            exp_config = OmegaConf.load(exp_config_path)
            # Merge with base config
            merged_config = OmegaConf.merge(config, exp_config)
        else:
            print(f"Warning: Config not found for {exp_name}, using base config")
            merged_config = config

        # Initialize logger
        logger = ExperimentLogger(merged_config)

        try:
            # Load model
            model = load_model(merged_config, exp_name)

            # Evaluate
            results = evaluate_model(model, merged_config, exp_name)

            # Store results
            all_results[exp_name] = results

            # Log to W&B
            for benchmark, metrics in results.items():
                logger.log({f"{benchmark}/{k}": v for k, v in metrics.items()})

        except Exception as e:
            print(f"Error in experiment {exp_name}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            logger.finish()

    # Save all results to JSON
    results_dir = Path(config.paths.output_dir) / "ablation_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "all_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to {results_file}")
    print(f"{'='*80}")

    # Print summary table
    print("\n--- SUMMARY ---")
    print(f"{'Experiment':<20} {'HumanEval':<12} {'MBPP':<12} {'DS-1000':<12} {'Unseen-API':<12}")
    print("-" * 80)

    for exp_name, results in all_results.items():
        humaneval_score = results.get("humaneval", {}).get("pass@1", 0.0)
        mbpp_score = results.get("mbpp", {}).get("pass@1", 0.0)
        ds1000_score = results.get("ds1000", {}).get("pass@1", 0.0)
        unseen_score = results.get("unseen_api", {}).get("pass@1", 0.0)

        print(f"{exp_name:<20} {humaneval_score:<12.4f} {mbpp_score:<12.4f} {ds1000_score:<12.4f} {unseen_score:<12.4f}")


if __name__ == "__main__":
    main()
