"""
Base model wrapper for DeepSeek-Coder with quantization support.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from typing import Optional, Dict, Any
from omegaconf import DictConfig


class BaseCodeModel:
    """Wrapper for base code generation model with quantization."""

    def __init__(self, config: DictConfig):
        """
        Initialize base model with optional quantization.

        Args:
            config: Hydra configuration containing model settings
        """
        self.config = config
        self.model_name = config.model.name
        self.device = config.model.get("device_map", "auto")

        # Set up quantization config
        self.bnb_config = None
        if config.model.quantization.load_in_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(
                    torch, config.model.quantization.bnb_4bit_compute_dtype
                ),
                bnb_4bit_use_double_quant=config.model.quantization.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=config.model.quantization.bnb_4bit_quant_type,
            )

        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self._load_model()
        self._load_tokenizer()

    def _load_model(self):
        """Load the base model with quantization."""
        print(f"Loading model: {self.model_name}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map=self.device,
            trust_remote_code=True,
            cache_dir=self.config.model.get("cache_dir", None),
            torch_dtype=torch.bfloat16 if self.bnb_config is None else None,
        )

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        print(f"Model loaded successfully on {self.device}")

    def _load_tokenizer(self):
        """Load the tokenizer."""
        print(f"Loading tokenizer: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.config.model.get("cache_dir", None),
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print("Tokenizer loaded successfully")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """
        Generate code from a prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 for greedy)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Use greedy decoding for temperature=0
        if temperature == 0.0:
            do_sample = False

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode only the generated part (exclude prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return generated_text

    def get_trainable_params(self) -> Dict[str, int]:
        """
        Get count of trainable and total parameters.

        Returns:
            Dictionary with trainable and total parameter counts
        """
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())

        return {
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_percent": 100 * trainable_params / total_params,
        }

    def print_trainable_params(self):
        """Print trainable parameter statistics."""
        params_info = self.get_trainable_params()
        print(f"Trainable params: {params_info['trainable_params']:,}")
        print(f"Total params: {params_info['total_params']:,}")
        print(f"Trainable %: {params_info['trainable_percent']:.4f}%")
