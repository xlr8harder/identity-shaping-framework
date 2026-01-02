"""ISF configuration and model resolution.

Handles isf.yaml loading and model name resolution:
- isf.identity.full → aria-v0.9-full (via convention)
- isf.judge.small → gpt-4o-mini (via explicit mapping)
- aria-v0.9-full → passthrough to mq registry

Also handles checkpoint resolution for tinker models:
- e027-final → (base_model, renderer, checkpoint_path)
"""
import json
import os
from pathlib import Path
from typing import Optional
import yaml

# Default config, can be overridden by isf.yaml
DEFAULT_CONFIG = {
    "models": {
        "identity": {
            "prefix": "aria",
            "release_version": "v0.9",
            "sizes": ["full", "small"],
        },
        "judge": {
            "small": "gpt-4o-mini",
            "large": "gpt-4o",
        },
        "generator": {
            "cheap": "gpt-oss-120b",
        },
    }
}


class ISFConfig:
    """ISF configuration manager."""

    def __init__(self, config_path: Optional[Path] = None):
        """Load config from file or use defaults.

        Args:
            config_path: Path to isf.yaml. If None, searches up from cwd.
        """
        self._config = DEFAULT_CONFIG.copy()

        # Find config file
        if config_path is None:
            config_path = self._find_config()

        if config_path and config_path.exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    self._merge_config(user_config)

        # Identity tier override from environment
        self._identity_tier = os.environ.get("ISF_IDENTITY_TIER", "release")

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

    def set_identity_tier(self, tier: str):
        """Override identity tier (release, dev, v0.8, etc.)."""
        self._identity_tier = tier

    def resolve_model(self, model_ref: str) -> str:
        """Resolve a model reference to an mq registry shortname.

        Args:
            model_ref: Model reference, one of:
                - "isf.identity.full" → convention-based resolution
                - "isf.judge.small" → explicit mapping
                - "aria-v0.9-full" → passthrough

        Returns:
            mq registry shortname
        """
        if not model_ref.startswith("isf."):
            # Passthrough - assume it's already a registry shortname
            return model_ref

        parts = model_ref.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid model reference: {model_ref}")

        _, category, size = parts
        models = self._config.get("models", {})

        if category == "identity":
            # Convention-based: {prefix}-{tier}-{size}
            identity_config = models.get("identity", {})
            prefix = identity_config.get("prefix", "aria")
            tier = self._identity_tier

            # "release" tier uses release_version
            if tier == "release":
                tier = identity_config.get("release_version", "v0.9")

            return f"{prefix}-{tier}-{size}"

        else:
            # Explicit mapping (judge, generator, etc.)
            category_config = models.get(category, {})
            if size not in category_config:
                raise ValueError(f"Unknown model: {model_ref}")
            return category_config[size]


# Global config instance (lazy loaded)
_config: Optional[ISFConfig] = None


def get_config() -> ISFConfig:
    """Get the global config instance."""
    global _config
    if _config is None:
        # Try to find isf.yaml in current directory or parents
        _config = ISFConfig()
    return _config


def resolve_model(model_ref: str) -> str:
    """Convenience function to resolve a model reference."""
    return get_config().resolve_model(model_ref)


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

            raise ValueError(f"Checkpoint '{checkpoint_name}' not found in {checkpoints_file}")
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
