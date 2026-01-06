"""Training configuration parsing and validation."""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, field_validator


class TrainConfig(BaseModel):
    """Fully-resolved training experiment configuration.

    All fields are concrete values - no Optional fields that get resolved later.
    Use build_config() to construct from a YAML file with auto-resolution.

    Fields:
        base_model: Model to fine-tune (e.g., "Qwen/Qwen3-30B-A3B")
        data: Path to prepared training data (JSONL)
        name: Experiment name (auto-generated if not provided to build_config)
        renderer: Renderer for the model (auto-detected if not provided)
        learning_rate: Learning rate (tinker heuristic if not provided)
        shuffle_seed: Shuffle seed (defaults to seed if not provided)
        save_every: Save checkpoint every N steps (defaults to steps_per_epoch)
        epochs, batch_size, lora_rank, max_length: Training hyperparameters
        lr_schedule: Learning rate schedule
        seed: Base random seed
        test_size: Hold out N examples for validation
        eval_every: Evaluate every N steps (0 = disabled)
        grad_clip: Gradient clipping norm
        normalize_weights: Normalize per-example weights
        optim_metrics_every: Log optimizer metrics every N steps
        note: Free-form annotation
        log_dir: Base directory for experiment logs
    """

    # All required - fully resolved by build_config()
    base_model: str
    data: str
    name: str
    renderer: str
    learning_rate: float
    shuffle_seed: int
    save_every: int

    # Hyperparameters with defaults
    epochs: int = 1
    batch_size: int = 32
    lora_rank: int = 32
    lr_schedule: Literal["constant", "linear", "cosine"] = "constant"
    max_length: int = 8192

    # Reproducibility
    seed: int = 42

    # Evaluation and checkpointing
    test_size: int = 0
    eval_every: int = 0

    # Training tweaks
    normalize_weights: bool = False
    grad_clip: float = 1e12
    optim_metrics_every: int = 1

    # Annotation (truly optional)
    note: Optional[str] = None

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


def _detect_renderer(base_model: str) -> str:
    """Detect renderer from model name.

    Requires tinker_cookbook to be installed.
    """
    try:
        from tinker_cookbook import model_info
    except ImportError:
        raise ImportError(
            "Training requires tinker_cookbook. Install with:\n"
            "  pip install tinker-cookbook"
        )

    renderer = model_info.get_recommended_renderer_name(base_model)
    # tinker-cookbook's deepseekv3 defaults to non-thinking mode
    if renderer == "deepseekv3":
        renderer = "deepseekv3_thinking"
    return renderer


def _get_learning_rate(base_model: str) -> float:
    """Get learning rate heuristic for model.

    Requires tinker_cookbook to be installed.
    """
    try:
        from tinker_cookbook.hyperparam_utils import get_lr
    except ImportError:
        raise ImportError(
            "Training requires tinker_cookbook. Install with:\n"
            "  pip install tinker-cookbook"
        )

    return get_lr(base_model, is_lora=True)


def _count_data_rows(data_path: Path) -> int:
    """Count rows in training data file."""
    with open(data_path) as f:
        return sum(1 for _ in f)


def build_config(path: str | Path, **overrides) -> TrainConfig:
    """Build a fully-resolved training configuration from a YAML file.

    Loads the YAML config, applies CLI overrides, and resolves all
    auto-computed fields (renderer, learning_rate, shuffle_seed, save_every).

    Args:
        path: Path to YAML config file
        **overrides: CLI overrides (e.g., data="train.jsonl", epochs=3)

    Returns:
        Fully-resolved TrainConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
        ImportError: If tinker_cookbook is not installed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(data)}")

    # Normalize keys (allow hyphens in YAML)
    cfg = {}
    for key, value in data.items():
        normalized_key = key.replace("-", "_")
        cfg[normalized_key] = value

    # Apply CLI overrides (only non-None values)
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    # Validate required fields before resolution
    if not cfg.get("base_model"):
        raise ValueError("base_model is required in config")
    if not cfg.get("data"):
        raise ValueError("data is required in config")

    base_model = cfg["base_model"]
    data_path = Path(cfg["data"])

    # Resolve experiment name
    if not cfg.get("name"):
        log_dir = cfg.get("log_dir", "training/logs")
        cfg["name"] = get_next_experiment_name(log_dir)

    # Resolve renderer
    if not cfg.get("renderer"):
        cfg["renderer"] = _detect_renderer(base_model)

    # Resolve learning rate
    if cfg.get("learning_rate") is None:
        cfg["learning_rate"] = _get_learning_rate(base_model)

    # Resolve shuffle_seed (defaults to seed)
    if cfg.get("shuffle_seed") is None:
        cfg["shuffle_seed"] = cfg.get("seed", 42)

    # Resolve save_every (defaults to steps_per_epoch)
    if cfg.get("save_every") is None:
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")
        n_rows = _count_data_rows(data_path)
        if n_rows == 0:
            raise ValueError(f"Training data file is empty: {data_path}")
        test_size = cfg.get("test_size", 0)
        batch_size = cfg.get("batch_size", 32)
        train_rows = n_rows - test_size
        if train_rows <= 0:
            raise ValueError(f"test_size ({test_size}) >= total rows ({n_rows})")
        steps_per_epoch = train_rows // batch_size
        if steps_per_epoch <= 0:
            raise ValueError(
                f"train_rows ({train_rows}) < batch_size ({batch_size}). "
                f"Reduce batch_size or test_size."
            )
        cfg["save_every"] = steps_per_epoch

    return TrainConfig(**cfg)


# Backwards compatibility alias
load_config = build_config
