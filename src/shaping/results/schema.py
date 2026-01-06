"""Pydantic models for eval results schema.

Supports four model modes with discriminated unions:
- base: Just a model on a provider
- prompted: Base model with sysprompt
- trained: Fine-tuned model (Tinker)
- prompted_trained: Fine-tuned model with sysprompt
"""

from datetime import datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from ..training.config import TrainConfig


# =============================================================================
# Model Specifications (discriminated union by 'mode')
# =============================================================================


class BaseModelSpec(BaseModel):
    """Base model on a provider (no fine-tuning, no sysprompt)."""

    alias: str
    mode: Literal["base"] = "base"
    provider: str
    model_id: str


class PromptedModelSpec(BaseModel):
    """Base model with a sysprompt."""

    alias: str
    mode: Literal["prompted"] = "prompted"
    provider: str
    model_id: str
    sysprompt_version: str
    sysprompt_sha: str


class TrainedModelSpec(BaseModel):
    """Fine-tuned model (Tinker)."""

    alias: str
    mode: Literal["trained"] = "trained"
    provider: str
    base_model: str
    renderer: str
    checkpoint: str
    training_run: str
    training_data: str
    training_config: TrainConfig


class PromptedTrainedModelSpec(BaseModel):
    """Fine-tuned model with a sysprompt."""

    alias: str
    mode: Literal["prompted_trained"] = "prompted_trained"
    provider: str
    base_model: str
    renderer: str
    checkpoint: str
    training_run: str
    training_data: str
    training_config: TrainConfig
    sysprompt_version: str
    sysprompt_sha: str


# Discriminated union - Pydantic uses 'mode' field to pick the right type
ModelSpec = Annotated[
    Union[BaseModelSpec, PromptedModelSpec, TrainedModelSpec, PromptedTrainedModelSpec],
    Field(discriminator="mode"),
]


# =============================================================================
# Supporting Types
# =============================================================================


class EvalConfig(BaseModel):
    """Eval configuration with reproducibility info."""

    name: str
    dataset_sha: str
    judge_prompt_sha: str | None = None  # None for non-judged evals (e.g., MC)
    dataset_size: int
    n_samples: int
    complete: bool


class SamplingConfig(BaseModel):
    """Sampling configuration for model or judge."""

    temperature: float
    max_tokens: int
    runs_per_sample: int = 1  # For model: how many times to query per question
    judges_per_run: int = 1  # For judge: how many judgments per response


class ErrorBreakdown(BaseModel):
    """Error counts by type."""

    total: int
    by_type: dict[str, int] = Field(default_factory=dict)


class Results(BaseModel):
    """Eval results with aggregation info."""

    aggregation: str = "mean"  # mean, majority_vote, any_correct, etc.
    score: float
    std: float | None = None
    run_scores: list[float] = Field(default_factory=list)
    errors: ErrorBreakdown = Field(default_factory=lambda: ErrorBreakdown(total=0))


class Artifacts(BaseModel):
    """Pointers to detailed results on filesystem."""

    results_dir: str | None = None
    results_file: str | None = None
    summary_file: str | None = None


# =============================================================================
# Main Record
# =============================================================================


class EvalResult(BaseModel):
    """Complete eval result record.

    Captures everything needed to:
    - Reproduce the eval exactly
    - Compare with other evals
    - Drill down into detailed results
    """

    version: int = 1
    id: str
    timestamp: datetime

    model: ModelSpec
    judge: ModelSpec | None = None  # None for non-judged evals

    eval: EvalConfig

    model_sampling: SamplingConfig
    judge_sampling: SamplingConfig | None = None

    results: Results

    artifacts: Artifacts = Field(default_factory=Artifacts)
    note: str = ""

    # Archival - hide results without deleting
    archived: bool = False
    archived_note: str = ""
