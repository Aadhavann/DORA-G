"""
Training utilities using HuggingFace Trainer with PEFT.
"""

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from typing import Dict, Any
from omegaconf import DictConfig
import sys
sys.path.append("..")
from src.utils.logging import ExperimentLogger


class CodeTrainer:
    """Handles model training with W&B logging."""

    def __init__(self, config: DictConfig, model, tokenizer, logger: ExperimentLogger):
        """
        Initialize trainer.

        Args:
            config: Hydra configuration
            model: Model to train (with PEFT applied)
            tokenizer: Tokenizer
            logger: Experiment logger
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger

        # Create data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

    def create_training_arguments(self) -> TrainingArguments:
        """
        Create TrainingArguments from config.

        Returns:
            TrainingArguments instance
        """
        train_config = self.config.training

        return TrainingArguments(
            output_dir=train_config.output_dir,
            num_train_epochs=train_config.num_train_epochs,
            per_device_train_batch_size=train_config.per_device_train_batch_size,
            per_device_eval_batch_size=train_config.per_device_eval_batch_size,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            learning_rate=train_config.learning_rate,
            lr_scheduler_type=train_config.lr_scheduler_type,
            warmup_ratio=train_config.warmup_ratio,
            weight_decay=train_config.weight_decay,
            max_grad_norm=train_config.max_grad_norm,
            logging_steps=train_config.logging_steps,
            save_steps=train_config.save_steps,
            eval_steps=train_config.eval_steps,
            save_total_limit=train_config.save_total_limit,
            fp16=train_config.fp16,
            bf16=train_config.bf16,
            gradient_checkpointing=train_config.gradient_checkpointing,
            optim=train_config.optim,
            seed=train_config.seed,
            # W&B integration
            report_to=self.config.logging.report_to if self.config.logging.use_wandb else [],
            run_name=self.config.logging.get("wandb_run_name"),
            # Evaluation
            evaluation_strategy="steps",
            eval_accumulation_steps=1,
            # Saving
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

    def train(self, train_dataset, eval_dataset):
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Validation dataset

        Returns:
            Trained model
        """
        print("Setting up trainer...")

        training_args = self.create_training_arguments()

        trainer = Trainer(
            model=self.model.model,  # Access underlying model
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
        )

        print("Starting training...")
        print(f"  Epochs: {training_args.num_train_epochs}")
        print(f"  Batch size: {training_args.per_device_train_batch_size}")
        print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

        # Train
        train_result = trainer.train()

        # Save metrics
        metrics = train_result.metrics
        trainer.save_metrics("train", metrics)

        print("Training complete!")
        print(f"Final training loss: {metrics.get('train_loss', 'N/A')}")

        return trainer

    def save_model(self, trainer, output_dir: str):
        """
        Save trained model.

        Args:
            trainer: Trainer instance
            output_dir: Directory to save to
        """
        print(f"Saving model to {output_dir}")

        # Save adapter (for PEFT models)
        self.model.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Log to W&B
        if self.config.logging.use_wandb:
            self.logger.log_artifact(
                artifact_path=output_dir,
                artifact_type="model",
                name=f"{self.config.logging.wandb_run_name}_adapter",
            )

        print("Model saved successfully!")
