"""
Script 3: Train baseline models (LoRA and DoRA).
"""

import sys
sys.path.append("..")

import hydra
from omegaconf import DictConfig
from pathlib import Path
from datasets import load_from_disk
from src.models.dora_model import DoRAModel
from src.models.lora_model import LoRAModel
from src.data.preprocessing import DataPreprocessor
from src.training.trainer import CodeTrainer
from src.utils.logging import ExperimentLogger
from src.utils.reproducibility import set_seed, print_gpu_memory


@hydra.main(version_base=None, config_path="../configs/experiments")
def main(config: DictConfig):
    """Train LoRA or DoRA model."""
    print("="*80)
    print(f"STEP 3: Training {config.peft.method.upper()} Model")
    print("="*80)

    # Set seed
    set_seed(config.reproducibility.seed)

    # Print GPU info
    print("\n--- GPU Information ---")
    print_gpu_memory()

    # Initialize logger
    logger = ExperimentLogger(config)

    # Load datasets
    print("\n--- Loading Datasets ---")
    data_dir = Path(config.paths.data_dir)
    train_dataset = load_from_disk(str(data_dir / "train_dataset"))

    # Initialize model
    print("\n--- Loading Model ---")
    if config.peft.method == "dora":
        model = DoRAModel(config)
    else:
        model = LoRAModel(config)

    model.print_trainable_params()

    # Preprocess data
    print("\n--- Preprocessing Data ---")
    preprocessor = DataPreprocessor(model.tokenizer, config)
    tokenized_datasets = preprocessor.preprocess_dataset(train_dataset)

    # Initialize trainer
    print("\n--- Setting up Trainer ---")
    trainer_wrapper = CodeTrainer(config, model, model.tokenizer, logger)

    # Train
    trainer = trainer_wrapper.train(
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    # Save model
    output_dir = Path(config.training.output_dir) / "final_checkpoint"
    trainer_wrapper.save_model(trainer, str(output_dir))

    # Print final GPU memory
    print("\n--- Final GPU Memory ---")
    print_gpu_memory()

    # Close logger
    logger.finish()

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


if __name__ == "__main__":
    main()
