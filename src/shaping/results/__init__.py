"""Eval results tracking.

Provides structured storage for eval results with full reproducibility metadata.
"""

from .builder import build_eval_result, build_model_spec
from .schema import (
    Artifacts,
    BaseModelSpec,
    ErrorBreakdown,
    EvalConfig,
    EvalResult,
    ModelSpec,
    PromptedModelSpec,
    PromptedTrainedModelSpec,
    Results,
    SamplingConfig,
    TrainedModelSpec,
)
from .store import ResultsStore, generate_id

__all__ = [
    "Artifacts",
    "BaseModelSpec",
    "build_eval_result",
    "build_model_spec",
    "ErrorBreakdown",
    "EvalConfig",
    "EvalResult",
    "generate_id",
    "ModelSpec",
    "PromptedModelSpec",
    "PromptedTrainedModelSpec",
    "Results",
    "ResultsStore",
    "SamplingConfig",
    "TrainedModelSpec",
]
