"""Integration tests for pipeline infrastructure.

These tests verify that tasks work correctly with the dispatcher machinery,
catching issues like constructor signature mismatches.
"""

import json
import tempfile
from pathlib import Path

import pytest

from shaping.pipeline import (
    run_pipeline,
    TrackedTask,
    TrainingSample,
    model_request,
)
from shaping.pipeline.provenance import AnnotatedTrainingSample


class SimpleSynthesisTask(TrackedTask):
    """Test task that simulates a simple synthesis pipeline."""

    def run(self):
        prompt = self.data.get("prompt", "Hello")
        messages = [{"role": "user", "content": prompt}]

        response = yield model_request(
            messages,
            model="test-model",
            step_id="generate",
        )

        full_messages = messages + [
            {"role": "assistant", "content": response.get_text() if response.is_success else "ERROR"}
        ]

        return TrainingSample(
            id=self.data.get("id", "unknown"),
            messages=full_messages,
        )


class MultiStepTask(TrackedTask):
    """Test task with multiple inference steps."""

    def run(self):
        # First step
        resp1 = yield model_request(
            [{"role": "user", "content": "Step 1"}],
            model="model-a",
            step_id="step1",
        )

        # Second step uses first response
        resp2 = yield model_request(
            [{"role": "user", "content": f"Step 2: {resp1.get_text()}"}],
            model="model-b",
            step_id="step2",
        )

        return TrainingSample(
            id=self.data["id"],
            messages=[
                {"role": "user", "content": "Step 1"},
                {"role": "assistant", "content": resp2.get_text()},
            ],
        )


class TestTaskConstructor:
    """Tests that verify task constructor compatibility with dispatcher."""

    def test_tracked_task_accepts_context(self):
        """TrackedTask constructor accepts (data, context) like dispatcher expects."""
        # This is what dispatcher's FileTaskSource does
        data = {"id": "test", "prompt": "hello"}
        context = {"line_number": 0}

        task = SimpleSynthesisTask(data, context)

        assert task.data == data
        assert task.context == context

    def test_tracked_task_context_optional(self):
        """TrackedTask works without context (for manual instantiation)."""
        data = {"id": "test", "prompt": "hello"}
        task = SimpleSynthesisTask(data)

        assert task.data == data
        assert task.context is None


class TestPipelineIntegration:
    """Integration tests that run full pipelines."""

    def test_simple_pipeline_produces_annotated_output(self, mock_backend):
        """Pipeline produces AnnotatedTrainingSample output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.jsonl"
            output_file = Path(tmpdir) / "output.jsonl"

            # Write test input
            with open(input_file, "w") as f:
                f.write(json.dumps({"id": "1", "prompt": "Hello"}) + "\n")
                f.write(json.dumps({"id": "2", "prompt": "World"}) + "\n")

            # Run pipeline
            run_pipeline(
                task_class=SimpleSynthesisTask,
                input_file=input_file,
                output_file=output_file,
                model="test-model",
                num_workers=1,
            )

            # Verify output
            results = []
            with open(output_file) as f:
                for line in f:
                    results.append(json.loads(line))

            assert len(results) == 2

            for result in results:
                # Core fields
                assert "id" in result
                assert "messages" in result
                assert len(result["messages"]) == 2
                assert result["messages"][0]["role"] == "user"
                assert result["messages"][1]["role"] == "assistant"

                # Provenance fields
                assert "input_data" in result
                assert "steps" in result
                assert len(result["steps"]) == 1
                assert "pipeline_commit" in result
                assert "started_at" in result
                assert "completed_at" in result

    def test_multi_step_pipeline_captures_all_steps(self, mock_backend):
        """Pipeline captures all inference steps in provenance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.jsonl"
            output_file = Path(tmpdir) / "output.jsonl"

            with open(input_file, "w") as f:
                f.write(json.dumps({"id": "multi"}) + "\n")

            run_pipeline(
                task_class=MultiStepTask,
                input_file=input_file,
                output_file=output_file,
                model="test-model",
                num_workers=1,
            )

            with open(output_file) as f:
                result = json.loads(f.readline())

            assert len(result["steps"]) == 2
            assert result["steps"][0]["step_id"] == "step1"
            assert result["steps"][1]["step_id"] == "step2"

    def test_output_can_be_deserialized(self, mock_backend):
        """Pipeline output can be deserialized back to AnnotatedTrainingSample."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.jsonl"
            output_file = Path(tmpdir) / "output.jsonl"

            with open(input_file, "w") as f:
                f.write(json.dumps({"id": "1", "prompt": "Test"}) + "\n")

            run_pipeline(
                task_class=SimpleSynthesisTask,
                input_file=input_file,
                output_file=output_file,
                model="test-model",
                num_workers=1,
            )

            with open(output_file) as f:
                data = json.loads(f.readline())

            # Deserialize
            sample = AnnotatedTrainingSample.from_dict(data)

            assert sample.id == "1"
            assert len(sample.messages) == 2
            assert len(sample.steps) == 1
            assert sample.input_data == {"id": "1", "prompt": "Test"}

    def test_to_train_sample_strips_provenance(self, mock_backend):
        """to_train_sample() produces minimal training format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.jsonl"
            output_file = Path(tmpdir) / "output.jsonl"

            with open(input_file, "w") as f:
                f.write(json.dumps({"id": "1", "prompt": "Test"}) + "\n")

            run_pipeline(
                task_class=SimpleSynthesisTask,
                input_file=input_file,
                output_file=output_file,
                model="test-model",
                num_workers=1,
            )

            with open(output_file) as f:
                data = json.loads(f.readline())

            annotated = AnnotatedTrainingSample.from_dict(data)
            training = annotated.to_train_sample()

            # Minimal format
            train_dict = training.to_dict()
            assert set(train_dict.keys()) == {"id", "messages"}


@pytest.fixture
def mock_backend(monkeypatch):
    """Mock the LLM backend to return predictable responses."""
    from shaping.inference import LLMClientBackend

    class MockBackend(LLMClientBackend):
        def __init__(self, *args, **kwargs):
            # Don't call super().__init__ to avoid mq lookups
            self._model = "test-model"

        def process_request(self, request):
            from dispatcher.taskmanager.backend.request import Response

            # Extract prompt for response
            messages = request.content.get("messages", [])
            last_user_msg = ""
            for msg in messages:
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")

            return Response(
                request=request,
                content={
                    "choices": [
                        {"message": {"content": f"Response to: {last_user_msg}"}}
                    ]
                },
                model_name="mock-model",
            )

    monkeypatch.setattr("shaping.pipeline.runner.LLMClientBackend", MockBackend)
