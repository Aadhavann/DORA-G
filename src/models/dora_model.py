"""
DoRA (Weight-Decomposed Low-Rank Adaptation) fine-tuning model.
"""

from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from omegaconf import DictConfig
from .base_model import BaseCodeModel


class DoRAModel(BaseCodeModel):
    """DoRA fine-tuned code generation model."""

    def __init__(self, config: DictConfig):
        """
        Initialize DoRA model.

        Args:
            config: Hydra configuration with PEFT settings
        """
        super().__init__(config)

        if config.peft.get("enabled", True):
            self._apply_dora()

    def _apply_dora(self):
        """Apply DoRA (LoRA with weight decomposition) to the model."""
        print("Applying DoRA configuration...")

        # Prepare model for k-bit training if quantized
        if self.bnb_config is not None:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.training.get(
                    "gradient_checkpointing", True
                ),
            )

        # Configure DoRA/LoRA
        peft_config = LoraConfig(
            r=self.config.peft.r,
            lora_alpha=self.config.peft.lora_alpha,
            lora_dropout=self.config.peft.lora_dropout,
            target_modules=self.config.peft.target_modules,
            bias=self.config.peft.bias,
            task_type=TaskType.CAUSAL_LM,
            # DoRA-specific: enables magnitude-direction decomposition
            use_dora=self.config.peft.get("use_dora", True),
        )

        # Apply PEFT
        self.model = get_peft_model(self.model, peft_config)

        print("DoRA applied successfully!")
        self.print_trainable_params()

    def save_adapter(self, output_dir: str):
        """
        Save only the adapter weights (efficient for DoRA/LoRA).

        Args:
            output_dir: Directory to save adapter weights
        """
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"DoRA adapter saved to {output_dir}")

    @classmethod
    def from_pretrained_adapter(cls, config: DictConfig, adapter_path: str):
        """
        Load a DoRA model from saved adapter weights.

        Args:
            config: Hydra configuration
            adapter_path: Path to saved adapter

        Returns:
            DoRAModel instance with loaded adapter
        """
        from peft import PeftModel

        # First load base model
        instance = cls.__new__(cls)
        BaseCodeModel.__init__(instance, config)

        # Prepare for k-bit training if needed
        if instance.bnb_config is not None:
            instance.model = prepare_model_for_kbit_training(instance.model)

        # Load adapter
        instance.model = PeftModel.from_pretrained(
            instance.model,
            adapter_path,
            is_trainable=False,  # For inference
        )

        print(f"DoRA adapter loaded from {adapter_path}")
        instance.print_trainable_params()

        return instance
