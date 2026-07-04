"""Training infrastructure for identity shaping.

This module provides backend-aware SFT training jobs with YAML configuration
files. Tinker is the integrated backend today; additional backends share the
same config and CLI surface as their runners are added.

Typical usage:

    # training/config.yaml contains base hyperparameters
    isf train run training/config.yaml --data train.jsonl
    isf train run training/config.yaml -d train.jsonl -e 3 --name E050
"""

from .config import (
    SUPPORTED_TRAINING_BACKENDS,
    TrainConfig,
    build_config,
    get_next_experiment_name,
    load_config,
    normalize_training_backend,
)
from .runner import run_training

__all__ = [
    "SUPPORTED_TRAINING_BACKENDS",
    "TrainConfig",
    "build_config",
    "load_config",  # backwards compat alias for build_config
    "get_next_experiment_name",
    "normalize_training_backend",
    "run_training",
]
