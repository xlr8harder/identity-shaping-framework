"""Provenance tracking and training data types.

Provides structured types for pipeline outputs:
- QueryResponse: Response from Pipeline.query() calls
- TrainingSample: Minimal format for model training
- AnnotatedTrainingSample: Full provenance for debugging/lineage
- InferenceStep: Captures a single LLM call
"""

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class QueryResponse:
    """Response from Pipeline.query() calls.

    Matches the interface of dispatcher.Response for consistency with
    yield model_request() in task methods.

    Example:
        response = self.query(model=self.judge_model, messages=[...])
        if response.is_success:
            text = response.get_text()
        else:
            print(f"Error: {response.error}")
    """

    text: str
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.error is None

    def get_text(self) -> str:
        """Get response text. Returns empty string on error."""
        return self.text if self.is_success else ""


@dataclass
class InferenceStep:
    """A single LLM inference call during pipeline execution."""

    # Input
    messages: list[dict]
    model: str | None  # registry shortname (e.g., "cubsfan-release-full")
    model_resolved: str | None  # actual model name from backend
    sampling: dict  # temperature, max_tokens, etc.

    # Output
    response: str  # full response text (with thinking traces)
    error: str | None = None  # error message if failed

    # Metadata
    step_id: str | None = None  # optional identifier (e.g., "generate")
    step_index: int = 0  # auto-incremented index
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "messages": self.messages,
            "model": self.model,
            "model_resolved": self.model_resolved,
            "sampling": self.sampling,
            "response": self.response,
            "error": self.error,
            "step_id": self.step_id,
            "step_index": self.step_index,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TrainingSample:
    """Minimal training data - what the model actually trains on.

    This is the reduced format used for final training datasets.
    Contains only what's needed for training, no provenance metadata.
    """

    id: str
    messages: list[dict]  # Full conversation including assistant responses

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSONL output."""
        return {
            "id": self.id,
            "messages": self.messages,
        }


@dataclass
class AnnotatedTrainingSample(TrainingSample):
    """Training sample with full provenance for debugging and lineage.

    Extends TrainingSample with:
    - Original input data from the pipeline
    - All inference steps (raw LLM calls)
    - Pipeline metadata (commit, file, timestamps)

    Use to_train_sample() to reduce to minimal TrainingSample for training.
    """

    # Original input row from pipeline
    input_data: dict[str, Any] = field(default_factory=dict)

    # All inference steps (in order)
    steps: list[InferenceStep] = field(default_factory=list)

    # Pipeline metadata
    pipeline_commit: str | None = None
    pipeline_file: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def to_train_sample(self) -> TrainingSample:
        """Reduce to minimal training format, stripping provenance."""
        return TrainingSample(id=self.id, messages=self.messages)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSONL output."""
        return {
            "id": self.id,
            "messages": self.messages,
            "input_data": self.input_data,
            "steps": [step.to_dict() for step in self.steps],
            "pipeline_commit": self.pipeline_commit,
            "pipeline_file": self.pipeline_file,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotatedTrainingSample":
        """Deserialize from dict.

        Raises:
            ValueError: If data is an error record (has __ERROR__ key)
        """
        if "__ERROR__" in data:
            error_info = data["__ERROR__"]
            raise ValueError(
                f"Cannot parse error record: {error_info.get('error', 'unknown')} - "
                f"{error_info.get('message', 'no message')}"
            )

        steps = [
            InferenceStep(
                messages=s["messages"],
                model=s["model"],
                model_resolved=s["model_resolved"],
                sampling=s["sampling"],
                response=s["response"],
                error=s.get("error"),
                step_id=s.get("step_id"),
                step_index=s.get("step_index", 0),
                timestamp=datetime.fromisoformat(s["timestamp"])
                if s.get("timestamp")
                else datetime.now(),
            )
            for s in data.get("steps", [])
        ]
        return cls(
            id=data["id"],
            messages=data["messages"],
            input_data=data.get("input_data", {}),
            steps=steps,
            pipeline_commit=data.get("pipeline_commit"),
            pipeline_file=data.get("pipeline_file"),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
        )


def get_git_commit() -> str | None:
    """Get current git commit hash, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]  # short hash
    except Exception:
        pass
    return None
