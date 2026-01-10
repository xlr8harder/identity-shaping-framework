"""ISF configuration.

Handles isf.yaml loading and checkpoint resolution for tinker models:
- e027-final → (base_model, renderer, checkpoint_path)
"""

import copy
import json
from pathlib import Path
from typing import Optional
import yaml

# Default config, can be overridden by isf.yaml
DEFAULT_CONFIG = {
    "identity": {
        "prefix": "identity",
        "release_version": "dev",
        "variants": ["full"],
    },
    "models": {},
}


class ISFConfig:
    """ISF configuration manager."""

    def __init__(self, config_path: Optional[Path] = None):
        """Load config from file or use defaults.

        Args:
            config_path: Path to isf.yaml. If None, searches up from cwd.
        """
        self._config = copy.deepcopy(DEFAULT_CONFIG)

        # Find config file
        if config_path is None:
            config_path = self._find_config()

        if config_path and config_path.exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    self._merge_config(user_config)

    @staticmethod
    def _find_config() -> Optional[Path]:
        """Search for isf.yaml in current directory and parents."""
        current = Path.cwd()
        while current != current.parent:
            candidate = current / "isf.yaml"
            if candidate.exists():
                return candidate
            current = current.parent
        return None

    def _merge_config(self, user_config: dict):
        """Deep merge user config into defaults."""
        for key, value in user_config.items():
            if key in self._config and isinstance(value, dict):
                self._config[key].update(value)
            else:
                self._config[key] = value


# =============================================================================
# Checkpoint Resolution (for tinker models)
# =============================================================================

# Default training logs directory (relative to project root)
TRAINING_LOGS_DIR = Path("training/logs")


def resolve_checkpoint(
    spec: str,
    logs_dir: Optional[Path] = None,
) -> tuple[str, str, Optional[str]]:
    """Resolve a checkpoint spec to (base_model, renderer_name, model_path).

    Supports multiple formats:
    - Experiment checkpoints: "e027-final", "e027-000192"
    - Explicit format: "model_name::renderer_name" or "model_name::renderer_name::path"
    - Base model names (falls back to recommended renderer)

    Args:
        spec: Checkpoint specification
        logs_dir: Directory containing training logs (default: training/logs)

    Returns:
        (base_model, renderer_name, model_path) tuple
        model_path is None for base models without checkpoints

    Raises:
        ValueError: If checkpoint not found or spec is invalid
    """
    if logs_dir is None:
        logs_dir = TRAINING_LOGS_DIR

    # Experiment checkpoint format: E027-final, e027-000192
    if "-" in spec and spec.split("-")[0].lower().startswith("e"):
        parts = spec.split("-", 1)
        exp_name = parts[0].upper()
        checkpoint_name = parts[1]

        log_dir = logs_dir / exp_name
        config_file = log_dir / "config.json"
        checkpoints_file = log_dir / "checkpoints.jsonl"

        if config_file.exists() and checkpoints_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            model_name = config["model_name"]
            renderer_name = config["dataset_builder"]["common_config"]["renderer_name"]

            with open(checkpoints_file) as f:
                for line in f:
                    cp = json.loads(line)
                    if cp["name"] == checkpoint_name:
                        return model_name, renderer_name, cp["sampler_path"]

            raise ValueError(
                f"Checkpoint '{checkpoint_name}' not found in {checkpoints_file}"
            )
        else:
            raise ValueError(f"Experiment logs not found at {log_dir}")

    # Explicit format: model::renderer or model::renderer::path
    if "::" in spec:
        parts = spec.split("::")
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            return parts[0], parts[1], None
        else:
            raise ValueError(f"Invalid explicit format: {spec}")

    # Base model name - need tinker_cookbook for recommended renderer
    try:
        from tinker_cookbook import model_info

        renderer = model_info.get_recommended_renderer_name(spec)
        return spec, renderer, None
    except ImportError:
        raise ImportError(
            f"Cannot auto-detect renderer for '{spec}'. "
            "Either install tinker_cookbook or use explicit format: model::renderer"
        )
