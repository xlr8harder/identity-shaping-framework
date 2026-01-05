"""Training configuration parsing and validation."""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, field_validator


class TrainConfig(BaseModel):
    """Training experiment configuration.

    Base hyperparams (typically in config file):
        base_model: Model to fine-tune (e.g., "Qwen/Qwen3-30B-A3B")
        batch_size: Training batch size
        lora_rank: LoRA rank for efficient fine-tuning
        max_length: Maximum sequence length
        epochs: Number of training epochs

    Per-run settings (typically CLI overrides):
        name: Experiment name (auto-generated if not provided)
        data: Path to prepared training data (JSONL)

    Optional:
        learning_rate: Learning rate (None = use tinker heuristic)
        lr_schedule: Learning rate schedule
        test_size: Number of examples to hold out for validation
        eval_every: Evaluate every N steps (0 = disabled)
        save_every: Save checkpoint every N steps (None = once per epoch)
        renderer: Override renderer (None = auto-detect from model)
        grad_clip: Gradient clipping norm (default 1e12 = no clipping but enables grad norm logging)
        normalize_weights: Normalize per-example weights to sum to 1
        log_dir: Base directory for experiment logs
    """

    # Required - must be in config file or CLI
    base_model: str
    data: str

    # Optional with defaults
    name: Optional[str] = None  # Auto-generated if not provided
    epochs: int = 1
    batch_size: int = 32
    lora_rank: int = 32
    learning_rate: Optional[float] = None
    lr_schedule: Literal["constant", "linear", "cosine"] = "constant"
    max_length: int = 8192

    # Reproducibility
    seed: int = 42  # Random seed for data shuffling
    shuffle_seed: Optional[int] = None  # Separate shuffle seed (defaults to seed)

    # Evaluation and checkpointing
    test_size: int = 0  # Hold out N examples for validation
    eval_every: int = 0  # Evaluate every N steps (0 = disabled)
    save_every: Optional[int] = None  # Save checkpoint every N steps (None = once per epoch)

    # Optional overrides
    renderer: Optional[str] = None  # Override auto-detected renderer
    normalize_weights: bool = False  # Normalize per-example weights

    # Gradient clipping and optimizer metrics
    # Default grad_clip=1e12 enables tinker's gradient norm calculation without actual clipping
    # This lets us log grad norms every step for monitoring training stability
    grad_clip: float = 1e12
    optim_metrics_every: int = 1  # Log optimizer metrics (including grad norm) every step

    # Annotation
    note: Optional[str] = None  # Free-form note about this experiment

    # Paths
    log_dir: str = "training/logs"

    @field_validator("base_model")
    @classmethod
    def base_model_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("base_model is required")
        return v

    @field_validator("data")
    @classmethod
    def data_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("data is required")
        return v

    @field_validator("epochs")
    @classmethod
    def epochs_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("epochs must be >= 1")
        return v

    @field_validator("batch_size")
    @classmethod
    def batch_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size must be >= 1")
        return v

    @field_validator("lora_rank")
    @classmethod
    def lora_rank_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("lora_rank must be >= 1")
        return v

    @field_validator("max_length")
    @classmethod
    def max_length_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_length must be >= 1")
        return v

    @property
    def log_path(self) -> Path:
        """Path to experiment log directory."""
        return Path(self.log_dir) / self.name

    @property
    def data_path(self) -> Path:
        """Path to training data file."""
        return Path(self.data)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return self.model_dump()


def get_next_experiment_name(log_dir: str | Path = "training/logs") -> str:
    """Auto-detect next available experiment name (E001, E002, etc.).

    Args:
        log_dir: Directory containing experiment logs

    Returns:
        Next available experiment name (e.g., "E050")
    """
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return "E001"

    existing = []
    for d in log_dir.iterdir():
        if d.is_dir() and d.name.startswith("E") and d.name[1:].isdigit():
            existing.append(int(d.name[1:]))

    if not existing:
        return "E001"

    next_num = max(existing) + 1
    return f"E{next_num:03d}"


def load_config(path: str | Path, **overrides) -> TrainConfig:
    """Load training configuration from a YAML file with optional overrides.

    Args:
        path: Path to YAML config file
        **overrides: CLI overrides (e.g., data="train.jsonl", epochs=3)

    Returns:
        TrainConfig instance with overrides applied

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(data)}")

    # Normalize keys (allow hyphens in YAML)
    normalized = {}
    for key, value in data.items():
        normalized_key = key.replace("-", "_")
        normalized[normalized_key] = value

    # Apply CLI overrides (only non-None values)
    for key, value in overrides.items():
        if value is not None:
            normalized[key] = value

    # Auto-generate experiment name if not provided
    if not normalized.get("name"):
        log_dir = normalized.get("log_dir", "training/logs")
        normalized["name"] = get_next_experiment_name(log_dir)

    return TrainConfig(**normalized)
