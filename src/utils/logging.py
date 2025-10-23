"""
Logging utilities with W&B integration.
"""

import os
from typing import Optional, Dict, Any
import wandb
from omegaconf import DictConfig, OmegaConf


class ExperimentLogger:
    """Handles experiment logging with Weights & Biases."""

    def __init__(self, config: DictConfig, mode: str = "online"):
        """
        Initialize experiment logger.

        Args:
            config: Hydra configuration
            mode: W&B mode ("online", "offline", "disabled")
        """
        self.config = config
        self.mode = mode
        self.run = None

        if config.logging.use_wandb and mode != "disabled":
            self._init_wandb()

    def _init_wandb(self):
        """Initialize Weights & Biases run."""
        # Convert OmegaConf to dict for W&B
        config_dict = OmegaConf.to_container(self.config, resolve=True)

        self.run = wandb.init(
            project=self.config.logging.wandb_project,
            entity=self.config.logging.get("wandb_entity", None),
            name=self.config.logging.get("wandb_run_name", None),
            tags=self.config.logging.get("wandb_tags", []),
            config=config_dict,
            mode=self.mode,
        )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if self.run is not None:
            wandb.log(metrics, step=step)

    def log_artifact(self, artifact_path: str, artifact_type: str, name: str):
        """
        Log an artifact (model checkpoint, dataset, etc.) to W&B.

        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact (model, dataset, etc.)
            name: Name for the artifact
        """
        if self.run is not None:
            artifact = wandb.Artifact(name=name, type=artifact_type)
            artifact.add_file(artifact_path)
            self.run.log_artifact(artifact)

    def log_table(self, name: str, data: list, columns: list):
        """
        Log a table to W&B.

        Args:
            name: Name of the table
            data: List of rows
            columns: List of column names
        """
        if self.run is not None:
            table = wandb.Table(data=data, columns=columns)
            wandb.log({name: table})

    def finish(self):
        """Finish the W&B run."""
        if self.run is not None:
            wandb.finish()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
