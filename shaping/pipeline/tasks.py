"""Common task implementations for pipelines.

Provides reusable GeneratorTask subclasses for typical inference patterns.
"""

import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.backend.request import Request, Response

from .provenance import InferenceStep, TrainingSample, AnnotatedTrainingSample, get_git_commit


# Sampling parameters to capture in provenance
_SAMPLING_KEYS = ("temperature", "max_tokens", "top_p", "stop", "n")


def model_request(
    messages: list[dict],
    model: Optional[str] = None,
    step_id: Optional[str] = None,
    **kwargs,
) -> Request:
    """Create a request, optionally with a model for routing.

    Args:
        messages: Chat messages in OpenAI format
        model: Optional model reference (e.g., "isf.identity.full", "aria-v0.9-full").
            Required when using RegistryBackend (multi-model mode).
        step_id: Optional identifier for this inference step (for provenance tracking).
            If not provided, steps are numbered automatically.
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        Request object ready to yield from a task generator

    Example:
        # Single-model pipeline (model set at pipeline level)
        response = yield model_request(messages, temperature=0.7)

        # Multi-model pipeline (model in each request)
        response = yield model_request(messages, model="isf.identity.full")
        judge_resp = yield model_request(judge_msgs, model="isf.judge.small")

        # With step IDs for provenance
        response = yield model_request(messages, model="...", step_id="generate")
        judgment = yield model_request(judge_msgs, model="...", step_id="judge")
    """
    content = {"messages": messages, **kwargs}
    if model is not None:
        content["_model"] = model
    if step_id is not None:
        content["_step_id"] = step_id
    return Request(content)


class TrackedTask(GeneratorTask):
    """Base task that captures provenance for all inference steps.

    Subclasses implement `process_record()` instead of `task_generator()`.
    All yields are automatically captured as InferenceSteps.

    Task must return a TrainingSample, which gets wrapped as
    AnnotatedTrainingSample with full provenance.

    Class attributes:
        name: Pipeline name (required). Used to derive default paths.
        record_input_file: Optional explicit input path. If None, uses cache.
        default_workers: Number of parallel workers (default: 4).

    Lifecycle hooks:
        setup(): Called once before dispatcher starts (e.g., download data)
        teardown(): Called once after dispatcher finishes (e.g., cleanup)

    File paths:
        get_record_input_file(): Returns explicit path or default cache path
        get_record_output_file(): Returns output path (derived from name)

    Example:
        class MyPipeline(TrackedTask):
            name = "my-pipeline"

            @classmethod
            def setup(cls):
                # Download data and write to cls.get_record_input_file()
                with open(cls.get_record_input_file(), "w") as f:
                    ...

            def process_record(self):
                resp = yield model_request(msgs, model="isf.identity.full")
                return TrainingSample(
                    id=self.data["id"],
                    messages=msgs + [{"role": "assistant", "content": resp.get_text()}],
                )

        # Output is AnnotatedTrainingSample with steps, input_data, etc.
    """

    # Required: pipeline name (used for default paths)
    name: str = None  # type: ignore

    # Optional: explicit input file path. If None, uses pipelines/.cache/{name}.jsonl
    record_input_file: Optional[Path] = None

    # Optional: number of parallel workers
    default_workers: int = 4

    @classmethod
    def get_record_input_file(cls) -> Path:
        """Get input file path: explicit path if set, otherwise cache path."""
        if cls.record_input_file is not None:
            return cls.record_input_file
        if cls.name is None:
            raise ValueError(f"{cls.__name__} must define 'name' attribute")
        return Path(f"pipelines/.cache/{cls.name}.jsonl")

    @classmethod
    def get_record_output_file(cls) -> Path:
        """Get output file path (derived from name, not user-configurable)."""
        if cls.name is None:
            raise ValueError(f"{cls.__name__} must define 'name' attribute")
        return Path(f"training/data/{cls.name}.jsonl")

    @classmethod
    def setup(cls) -> None:
        """Called once before dispatcher starts. Override to prepare input data."""
        pass

    @classmethod
    def teardown(cls) -> None:
        """Called once after dispatcher finishes. Override for cleanup."""
        pass

    def __init__(self, data: Dict[str, Any], context: Any = None):
        super().__init__(data, context)
        self._steps: List[InferenceStep] = []
        self._step_counter = 0
        self._started_at = datetime.now()
        self._pipeline_file: Optional[str] = None

        # Try to get source file of the subclass
        try:
            self._pipeline_file = inspect.getfile(self.__class__)
        except Exception:
            pass

    def task_generator(self) -> Generator[Request, Response, Dict[str, Any]]:
        """Wraps process_record() to capture all inference steps."""
        gen = self.process_record()
        response = None

        while True:
            try:
                yielded = gen.send(response)

                if isinstance(yielded, list):
                    # Parallel requests
                    responses = yield yielded
                    for req, resp in zip(yielded, responses):
                        self._record_step(req, resp)
                    response = responses
                else:
                    # Single request
                    resp = yield yielded
                    self._record_step(yielded, resp)
                    response = resp

            except StopIteration as e:
                return self._wrap_output(e.value)

    def process_record(self) -> Generator[Request, Response, TrainingSample]:
        """Override this method to process one record. Return a TrainingSample."""
        raise NotImplementedError("Subclasses must implement process_record()")

    def _record_step(self, request: Request, response: Response) -> None:
        """Record an inference step from request/response pair."""
        content = request.content

        # Extract step_id (from request or auto-generate)
        step_id = content.get("_step_id")

        # Extract sampling params
        sampling = {k: content[k] for k in _SAMPLING_KEYS if k in content}

        step = InferenceStep(
            messages=content.get("messages", []),
            model=content.get("_model"),
            model_resolved=response.model_name,
            sampling=sampling,
            response=response.get_text() if response.is_success else "",
            error=str(response.error) if response.error else None,
            step_id=step_id,
            step_index=self._step_counter,
            timestamp=datetime.now(),
        )
        self._steps.append(step)
        self._step_counter += 1

    def _wrap_output(self, sample: TrainingSample) -> Dict[str, Any]:
        """Wrap TrainingSample with provenance as AnnotatedTrainingSample."""
        annotated = AnnotatedTrainingSample(
            id=sample.id,
            messages=sample.messages,
            input_data=self.data,
            steps=self._steps,
            pipeline_commit=get_git_commit(),
            pipeline_file=self._pipeline_file,
            started_at=self._started_at,
            completed_at=datetime.now(),
        )
        return annotated.to_dict()


class SingleTurnTask(GeneratorTask):
    """Simple single-turn completion task.

    Input format:
        {"id": "...", "messages": [...], "_model": "...", ...other fields...}

    Output format:
        {"id": "...", "response": "...", ...original fields...}

    The messages field should be in OpenAI chat format:
        [{"role": "user", "content": "..."}]

    The _model field is optional - required only for multi-model pipelines.
    """

    def task_generator(self) -> Generator[Request, Response, Dict[str, Any]]:
        # Extract messages from input
        messages = self.data.get("messages", [])
        if not messages:
            # If no messages, try to construct from a prompt field
            prompt = self.data.get("prompt", "")
            messages = [{"role": "user", "content": prompt}]

        # Build request content
        content: Dict[str, Any] = {"messages": messages}

        # Pass through _model for routing if present
        if "_model" in self.data:
            content["_model"] = self.data["_model"]

        # Pass through generation options if present
        for key in ("temperature", "max_tokens", "top_p", "stop"):
            if key in self.data:
                content[key] = self.data[key]

        # Make the request
        response: Response = yield Request(content)

        # Extract response text
        response_text = response.get_text() if response.is_success else f"ERROR: {response.error}"

        # Build result, preserving original fields
        result = dict(self.data)
        result["response"] = response_text

        return result


class MultiTurnTask(GeneratorTask):
    """Multi-turn conversation task.

    Input format:
        {"id": "...", "turns": [{"role": "user", "content": "..."}, ...], "_model": "..."}

    Each turn after the first assistant response builds on the conversation.
    The task yields requests for each assistant turn needed.

    Output format:
        {"id": "...", "conversation": [...all turns...], ...original fields...}

    The _model field is optional - required only for multi-model pipelines.
    """

    def task_generator(self) -> Generator[Request, Response, Dict[str, Any]]:
        turns = list(self.data.get("turns", []))
        conversation: List[Dict[str, Any]] = []

        # Process turns, generating assistant responses as needed
        for turn in turns:
            conversation.append(turn)

            if turn["role"] == "user":
                # Need to generate assistant response
                content: Dict[str, Any] = {"messages": conversation}

                # Pass through _model for routing if present
                if "_model" in self.data:
                    content["_model"] = self.data["_model"]

                # Pass through generation options
                for key in ("temperature", "max_tokens", "top_p", "stop"):
                    if key in self.data:
                        content[key] = self.data[key]

                response: Response = yield Request(content)

                if response.is_success:
                    assistant_content = response.get_text()
                else:
                    assistant_content = f"ERROR: {response.error}"

                conversation.append({"role": "assistant", "content": assistant_content})

        # Build result
        result = dict(self.data)
        result["conversation"] = conversation

        return result
