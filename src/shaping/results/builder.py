"""Builder for constructing EvalResult from eval run data.

This module handles the complex task of building a fully-populated
EvalResult with proper ModelSpec from eval run outputs.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from mq import store as mq_store

from .schema import (
    Artifacts,
    BaseModelSpec,
    EvalConfig,
    EvalResult,
    ErrorBreakdown,
    ModelSpec,
    PromptedModelSpec,
    PromptedTrainedModelSpec,
    Results,
    SamplingConfig,
    TrainedModelSpec,
)
from .store import generate_id
from ..training.config import TrainConfig


def _sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _sha256_str(s: str) -> str:
    """Compute SHA256 hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _is_experiment_checkpoint(model: str) -> bool:
    """Check if model spec looks like an experiment checkpoint (e037-final)."""
    if "-" not in model:
        return False
    parts = model.split("-", 1)
    return parts[0].lower().startswith("e") and parts[0][1:].isdigit()


def _parse_experiment_spec(model: str) -> tuple[str, str]:
    """Parse experiment spec into (exp_name, checkpoint_name)."""
    parts = model.split("-", 1)
    return parts[0].upper(), parts[1]


def _load_training_config(exp_name: str, logs_dir: Path) -> Optional[TrainConfig]:
    """Load training config for an experiment."""
    config_path = logs_dir / exp_name / "train-config.json"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        data = json.load(f)

    return TrainConfig.model_validate(data)


def _get_checkpoint_path(
    exp_name: str, checkpoint_name: str, logs_dir: Path
) -> Optional[str]:
    """Get the full checkpoint path from experiment logs."""
    checkpoints_file = logs_dir / exp_name / "checkpoints.jsonl"
    if not checkpoints_file.exists():
        return None

    with open(checkpoints_file) as f:
        for line in f:
            cp = json.loads(line)
            if cp.get("name") == checkpoint_name:
                return cp.get("sampler_path")

    return None


def build_model_spec(
    model_alias: str,
    logs_dir: Path = Path("training/logs"),
) -> ModelSpec:
    """Build a ModelSpec from a model alias.

    Args:
        model_alias: The mq registry model name or experiment checkpoint
        logs_dir: Path to training logs directory

    Returns:
        Appropriate ModelSpec subtype based on model configuration

    Raises:
        ValueError: If model cannot be resolved
    """
    # Try to look up in mq registry
    try:
        model_config = mq_store.get_model(model_alias)
    except (KeyError, ValueError):
        model_config = None

    # If not in registry, check if it's an experiment checkpoint
    if model_config is None and _is_experiment_checkpoint(model_alias):
        exp_name, checkpoint_name = _parse_experiment_spec(model_alias)
        train_config = _load_training_config(exp_name, logs_dir)

        if train_config is None:
            raise ValueError(f"Training config not found for {exp_name}")

        checkpoint_path = _get_checkpoint_path(exp_name, checkpoint_name, logs_dir)

        return TrainedModelSpec(
            alias=model_alias,
            provider="tinker",
            base_model=train_config.base_model,
            renderer=train_config.renderer or "unknown",
            checkpoint=checkpoint_path or f"{exp_name}/{checkpoint_name}",
            training_run=exp_name,
            training_data=train_config.data,
            training_config=train_config,
        )

    if model_config is None:
        raise ValueError(f"Model not found in registry: {model_alias}")

    provider = model_config.get("provider", "unknown")
    model_id = model_config.get("model", "")
    sysprompt = model_config.get("sysprompt")

    # Check if it's a trained model (has :: in model spec)
    is_trained = "::" in model_id

    # Check if it has a sysprompt
    has_sysprompt = sysprompt is not None and len(sysprompt) > 0

    if is_trained:
        # Parse model::renderer::path format
        parts = model_id.split("::")
        base_model = parts[0]
        renderer = parts[1] if len(parts) > 1 else "unknown"
        checkpoint_path = parts[2] if len(parts) > 2 else None

        # Try to find training config
        # Extract experiment name from alias if it looks like one
        train_config = None
        training_run = "unknown"
        training_data = "unknown"

        if _is_experiment_checkpoint(model_alias):
            exp_name, _ = _parse_experiment_spec(model_alias)
            train_config = _load_training_config(exp_name, logs_dir)
            training_run = exp_name
            if train_config:
                training_data = train_config.data

        if has_sysprompt:
            sysprompt_sha = _sha256_str(sysprompt)
            # Try to extract version from alias
            version = "unknown"
            for part in model_alias.split("-"):
                if part.startswith("v") and "." in part:
                    version = part
                    break

            if train_config is None:
                # Create minimal config for prompted trained (placeholder values)
                train_config = TrainConfig(
                    base_model=base_model,
                    data=training_data,
                    name=training_run,
                    renderer=renderer or "unknown",
                    learning_rate=0.0,
                    shuffle_seed=0,
                    save_every=0,
                )

            return PromptedTrainedModelSpec(
                alias=model_alias,
                provider=provider,
                base_model=base_model,
                renderer=renderer,
                checkpoint=checkpoint_path or "unknown",
                training_run=training_run,
                training_data=training_data,
                training_config=train_config,
                sysprompt_version=version,
                sysprompt_sha=sysprompt_sha,
            )
        else:
            if train_config is None:
                # Create minimal config (placeholder values when real config unavailable)
                train_config = TrainConfig(
                    base_model=base_model,
                    data=training_data,
                    name=training_run,
                    renderer=renderer or "unknown",
                    learning_rate=0.0,
                    shuffle_seed=0,
                    save_every=0,
                )

            return TrainedModelSpec(
                alias=model_alias,
                provider=provider,
                base_model=base_model,
                renderer=renderer,
                checkpoint=checkpoint_path or "unknown",
                training_run=training_run,
                training_data=training_data,
                training_config=train_config,
            )
    else:
        # Base model (not trained)
        if has_sysprompt:
            sysprompt_sha = _sha256_str(sysprompt)
            # Try to extract version from alias
            version = "unknown"
            for part in model_alias.split("-"):
                if part.startswith("v") and "." in part:
                    version = part
                    break

            return PromptedModelSpec(
                alias=model_alias,
                provider=provider,
                model_id=model_id,
                sysprompt_version=version,
                sysprompt_sha=sysprompt_sha,
            )
        else:
            return BaseModelSpec(
                alias=model_alias,
                provider=provider,
                model_id=model_id,
            )


def build_eval_result(
    model_alias: str,
    eval_name: str,
    score: float,
    n_samples: int,
    dataset_size: int,
    complete: bool,
    temperature: float,
    max_tokens: int,
    runs_per_sample: int = 1,
    judges_per_run: int = 1,
    aggregation: str = "mean",
    std: float | None = None,
    run_scores: list[float] | None = None,
    error_count: int = 0,
    error_breakdown: dict[str, int] | None = None,
    dataset_sha: str | None = None,
    judge_prompt_sha: str | None = None,
    judge_model: str | None = None,
    judge_temperature: float = 0.3,
    judge_max_tokens: int = 1024,
    results_file: str | None = None,
    summary_file: str | None = None,
    note: str = "",
    logs_dir: Path = Path("training/logs"),
) -> EvalResult:
    """Build a complete EvalResult from eval run data.

    Args:
        model_alias: The model name/alias that was evaluated
        eval_name: Name of the evaluation
        score: The primary score (0-1 for accuracy, or scaled)
        n_samples: Number of samples evaluated
        dataset_size: Total size of the eval dataset
        complete: Whether all samples were evaluated
        temperature: Model sampling temperature
        max_tokens: Model max tokens
        runs_per_sample: Number of model runs per sample
        judges_per_run: Number of judge calls per run
        aggregation: How scores were aggregated
        std: Standard deviation of scores
        run_scores: Individual run scores
        error_count: Total number of errors
        error_breakdown: Errors by type
        dataset_sha: SHA of the eval dataset
        judge_prompt_sha: SHA of the judge prompt (for LLM judges)
        judge_model: Judge model name (for LLM judges)
        judge_temperature: Judge sampling temperature
        judge_max_tokens: Judge max tokens
        results_file: Path to detailed results file
        summary_file: Path to summary file
        note: Optional note about this eval run
        logs_dir: Path to training logs directory

    Returns:
        Complete EvalResult ready to store
    """
    # Build model spec
    model_spec = build_model_spec(model_alias, logs_dir)

    # Build judge spec if we have a judge model
    judge_spec = None
    if judge_model:
        try:
            judge_spec = build_model_spec(judge_model, logs_dir)
        except ValueError:
            # Judge model not in registry - create base spec
            judge_spec = BaseModelSpec(
                alias=judge_model,
                provider="unknown",
                model_id=judge_model,
            )

    # Build eval config
    eval_config = EvalConfig(
        name=eval_name,
        dataset_sha=dataset_sha or "unknown",
        judge_prompt_sha=judge_prompt_sha,
        dataset_size=dataset_size,
        n_samples=n_samples,
        complete=complete,
    )

    # Build sampling configs
    model_sampling = SamplingConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        runs_per_sample=runs_per_sample,
        judges_per_run=judges_per_run,
    )

    judge_sampling = None
    if judge_model:
        judge_sampling = SamplingConfig(
            temperature=judge_temperature,
            max_tokens=judge_max_tokens,
        )

    # Build results
    results = Results(
        aggregation=aggregation,
        score=score,
        std=std,
        run_scores=run_scores or [],
        errors=ErrorBreakdown(
            total=error_count,
            by_type=error_breakdown or {},
        ),
    )

    # Build artifacts
    artifacts = Artifacts(
        results_file=results_file,
        summary_file=summary_file,
    )

    # Create the result
    return EvalResult(
        id=generate_id(),
        timestamp=datetime.now(),
        model=model_spec,
        judge=judge_spec,
        eval=eval_config,
        model_sampling=model_sampling,
        judge_sampling=judge_sampling,
        results=results,
        artifacts=artifacts,
        note=note,
    )
