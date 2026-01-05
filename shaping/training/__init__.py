"""Training infrastructure for identity shaping.

This module provides a thin wrapper around tinker_cookbook for running
SFT training jobs with YAML configuration files.

Typical usage:

    # training/config.yaml contains base hyperparameters
    isf train run training/config.yaml --data train.jsonl
    isf train run training/config.yaml -d train.jsonl -e 3 --name E050
"""

from .config import TrainConfig, build_config, load_config, get_next_experiment_name
from .runner import run_training

__all__ = [
    "TrainConfig",
    "build_config",
    "load_config",  # backwards compat alias for build_config
    "get_next_experiment_name",
    "run_training",
]
