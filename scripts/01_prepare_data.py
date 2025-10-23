"""
Script 1: Download and prepare datasets for training and evaluation.
"""

import sys
sys.path.append("..")

import hydra
from omegaconf import DictConfig
from pathlib import Path
from src.data.dataset_loader import CodeDatasetLoader
from src.utils.reproducibility import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(config: DictConfig):
    """Download and prepare all datasets."""
    print("="*80)
    print("STEP 1: Preparing Datasets")
    print("="*80)

    # Set seed for reproducibility
    set_seed(config.reproducibility.seed)

    # Initialize loader
    loader = CodeDatasetLoader(config)

    # Create data directories
    data_dir = Path(config.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download training datasets
    print("\n--- Loading Training Datasets ---")
    train_dataset = loader.load_training_datasets()

    # Save to disk
    train_path = data_dir / "train_dataset"
    train_dataset.save_to_disk(str(train_path))
    print(f"Training dataset saved to {train_path}")

    # 2. Download evaluation benchmarks
    print("\n--- Loading Evaluation Benchmarks ---")

    # HumanEval
    try:
        humaneval = loader.load_humaneval()
        humaneval.save_to_disk(str(data_dir / "humaneval"))
        print("✓ HumanEval saved")
    except Exception as e:
        print(f"✗ Error loading HumanEval: {e}")

    # MBPP
    try:
        mbpp = loader.load_mbpp()
        mbpp.save_to_disk(str(data_dir / "mbpp"))
        print("✓ MBPP saved")
    except Exception as e:
        print(f"✗ Error loading MBPP: {e}")

    # DS-1000
    try:
        ds1000 = loader.load_ds1000()
        if ds1000:
            ds1000.save_to_disk(str(data_dir / "ds1000"))
            print("✓ DS-1000 saved")
    except Exception as e:
        print(f"✗ Error loading DS-1000: {e}")

    # 3. Download CodeSearchNet for RAG
    print("\n--- Loading CodeSearchNet for RAG ---")
    try:
        codesearchnet = loader.load_codesearchnet(
            language=config.rag.corpus.language,
            split="train"
        )
        codesearchnet.save_to_disk(str(data_dir / "codesearchnet"))
        print("✓ CodeSearchNet saved")
    except Exception as e:
        print(f"✗ Error loading CodeSearchNet: {e}")

    print("\n" + "="*80)
    print("Data preparation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
