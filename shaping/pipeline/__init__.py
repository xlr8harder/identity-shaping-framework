"""Pipeline execution utilities.

Provides a simple interface for running inference pipelines using
dispatcher's TaskManager with ISF backends.

Two modes:
- Single-model: All requests go to one model (specify `model` param)
- Multi-model: Requests have `_model` field for routing (use RegistryBackend)

Example (single-model):
    from shaping.pipeline import run_pipeline, SingleTurnTask

    run_pipeline(
        task_class=SingleTurnTask,
        model="aria-v0.9-full",
        input_file="prompts.jsonl",
        output_file="responses.jsonl",
    )

Example (multi-model with provenance tracking):
    from shaping.pipeline import run_pipeline, model_request, TrackedTask

    class JudgedResponseTask(TrackedTask):
        def run(self):  # Note: run() not task_generator()
            messages = self.data["messages"]

            # Generate response from identity model
            resp = yield model_request(messages, model="isf.identity.full")

            # Judge the response
            judge_msgs = [{"role": "user", "content": f"Rate this: {resp.get_text()}"}]
            judge_resp = yield model_request(judge_msgs, model="isf.judge.small")

            return {"response": resp.get_text(), "judgment": judge_resp.get_text()}

    run_pipeline(
        task_class=JudgedResponseTask,
        input_file="prompts.jsonl",
        output_file="responses.jsonl",
    )
    # Output includes _provenance with all inference steps
"""

from .runner import run_pipeline
from .tasks import SingleTurnTask, MultiTurnTask, TrackedTask, model_request
from .provenance import InferenceStep, TrainingSample, AnnotatedTrainingSample

# Re-export dispatcher base classes with ISF naming
from dispatcher.taskmanager.task.base import GeneratorTask as PipelineTask
from dispatcher.taskmanager.task import TaskFailed
from dispatcher.taskmanager.backend.request import Request, Response

__all__ = [
    # Pipeline runner
    "run_pipeline",
    # Task base classes
    "PipelineTask",  # alias for GeneratorTask (no tracking)
    "TrackedTask",  # with provenance capture
    "TaskFailed",
    # Pre-built task implementations
    "SingleTurnTask",
    "MultiTurnTask",
    # Helpers
    "model_request",
    # Data classes
    "Request",
    "Response",
    "TrainingSample",
    "AnnotatedTrainingSample",
    "InferenceStep",
]
