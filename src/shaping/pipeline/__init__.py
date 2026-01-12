"""Pipeline execution utilities.

Provides Pipeline abstraction for multi-stage data generation with
dependency tracking and staleness detection.

Example (Pipeline with deps):
    from shaping.pipeline import Pipeline, model_request, TrainingSample

    class IdentityAugmentation(Pipeline):
        name = "identity-augmentation"

        narrative_doc = Pipeline.file_dep("identity/NARRATIVE.md")
        identity_model = Pipeline.model_dep("cubsfan-release-full")
        judge_model = Pipeline.model_dep("judge")

        def run(self):
            # Stage 1: Single LLM call (returns QueryResponse with .get_text())
            response = self.query(
                model=self.judge_model,
                messages=[{"role": "user", "content": self.narrative_doc.read()}],
            )
            facts = parse_facts(response.get_text())

            # Stage 2: Run task across records (parallel)
            results = self.run_task(self.generate_qa, records=facts)
            return results

        def generate_qa(self, record):
            question = yield model_request([...], model=self.judge_model)
            response = yield model_request([...], model=self.identity_model)
            return TrainingSample(...)

    # Run the pipeline
    pipeline = IdentityAugmentation()
    results = pipeline.execute()

    # Check staleness
    status = IdentityAugmentation.check_staleness()
    if status["stale"]:
        print("Stale:", status["reasons"])

Also provides lower-level Task classes for use with dispatcher directly.
"""

from .runner import run_pipeline
from .tasks import (
    SingleTurnTask,
    MultiTurnTask,
    TrackedTask,
    model_request,
    PipelineError,
)
from .provenance import (
    QueryResponse,
    InferenceStep,
    TrainingSample,
    AnnotatedTrainingSample,
)
from .deps import ModelDep, FileDep, get_all_deps, get_model_deps, get_file_deps
from .base import Pipeline

# Re-export dispatcher base classes with ISF naming
from dispatcher.taskmanager.task.base import GeneratorTask as PipelineTask
from dispatcher.taskmanager.backend.request import Request, Response

__all__ = [
    # Pipeline (recommended)
    "Pipeline",
    "PipelineError",
    "ModelDep",
    "FileDep",
    # Pipeline runner (lower-level)
    "run_pipeline",
    # Task base classes
    "PipelineTask",  # alias for GeneratorTask (no tracking)
    "TrackedTask",  # with provenance capture
    # Pre-built task implementations
    "SingleTurnTask",
    "MultiTurnTask",
    # Helpers
    "model_request",
    "get_all_deps",
    "get_model_deps",
    "get_file_deps",
    # Data classes
    "Request",
    "Response",
    "QueryResponse",
    "TrainingSample",
    "AnnotatedTrainingSample",
    "InferenceStep",
]
