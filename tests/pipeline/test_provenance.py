"""Tests for pipeline provenance tracking."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from dispatcher.taskmanager.backend.request import Request, Response

from shaping.pipeline import (
    TrackedTask,
    model_request,
    InferenceStep,
    TrainingSample,
    AnnotatedTrainingSample,
)
from shaping.pipeline.provenance import get_git_commit


class SimpleTrackedTask(TrackedTask):
    """Test task that makes two inference calls."""

    def process_record(self):
        msgs = [{"role": "user", "content": "Hello"}]

        # First call
        resp1 = yield model_request(
            msgs,
            model="test-model",
            step_id="greet",
            temperature=0.7,
        )

        # Second call
        resp2 = yield model_request(
            [{"role": "user", "content": "Goodbye"}],
            model="judge-model",
            step_id="farewell",
        )

        return TrainingSample(
            id=self.data.get("id", "test"),
            messages=msgs + [{"role": "assistant", "content": resp1.get_text()}],
        )


class ParallelTrackedTask(TrackedTask):
    """Test task that makes parallel inference calls."""

    def process_record(self):
        # Parallel calls
        responses = yield [
            model_request([{"role": "user", "content": "A"}], model="m1"),
            model_request([{"role": "user", "content": "B"}], model="m2"),
        ]

        return TrainingSample(
            id="parallel-test",
            messages=[
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": responses[0].get_text()},
            ],
        )


def make_response(request: Request, text: str) -> Response:
    """Helper to create a mock response."""
    return Response(
        request=request,
        content={"choices": [{"message": {"content": text}}]},
        model_name="resolved-model",
    )


class TestTrackedTask:
    """Tests for TrackedTask provenance capture."""

    def test_captures_single_step(self):
        """TrackedTask captures a single inference step."""
        task = SimpleTrackedTask({"id": "test1"})
        gen = task.task_generator()

        # First yield
        req1 = gen.send(None)
        assert req1.content["messages"] == [{"role": "user", "content": "Hello"}]
        assert req1.content["_model"] == "test-model"
        assert req1.content["_step_id"] == "greet"

        # Send response, get second yield
        resp1 = make_response(req1, "Hi there!")
        req2 = gen.send(resp1)
        assert req2.content["_step_id"] == "farewell"

        # Send response, get final result
        resp2 = make_response(req2, "See ya!")
        with pytest.raises(StopIteration) as exc_info:
            gen.send(resp2)

        result = exc_info.value.value

        # Check result structure (now AnnotatedTrainingSample format)
        assert result["id"] == "test1"
        assert result["messages"] == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Check provenance fields
        assert len(result["steps"]) == 2
        assert result["input_data"] == {"id": "test1"}

        step1 = result["steps"][0]
        assert step1["step_id"] == "greet"
        assert step1["step_index"] == 0
        assert step1["model"] == "test-model"
        assert step1["model_resolved"] == "resolved-model"
        assert step1["response"] == "Hi there!"
        assert step1["sampling"] == {"temperature": 0.7}

        step2 = result["steps"][1]
        assert step2["step_id"] == "farewell"
        assert step2["step_index"] == 1

    def test_captures_parallel_steps(self):
        """TrackedTask captures parallel inference steps."""
        task = ParallelTrackedTask({"id": "test2"})
        gen = task.task_generator()

        # First yield is a list
        reqs = gen.send(None)
        assert isinstance(reqs, list)
        assert len(reqs) == 2

        # Send list of responses
        responses = [
            make_response(reqs[0], "Response A"),
            make_response(reqs[1], "Response B"),
        ]
        with pytest.raises(StopIteration) as exc_info:
            gen.send(responses)

        result = exc_info.value.value
        assert result["id"] == "parallel-test"
        assert result["messages"][1]["content"] == "Response A"

        assert len(result["steps"]) == 2
        assert result["steps"][0]["step_index"] == 0
        assert result["steps"][1]["step_index"] == 1

    def test_auto_increments_step_index(self):
        """Step index auto-increments even without step_id."""

        class NoStepIdTask(TrackedTask):
            def process_record(self):
                r1 = yield model_request([{"role": "user", "content": "1"}])
                r2 = yield model_request([{"role": "user", "content": "2"}])
                return TrainingSample(id="test", messages=[])

        task = NoStepIdTask({})
        gen = task.task_generator()

        req1 = gen.send(None)
        req2 = gen.send(make_response(req1, "a"))
        with pytest.raises(StopIteration) as exc_info:
            gen.send(make_response(req2, "b"))

        result = exc_info.value.value
        assert result["steps"][0]["step_index"] == 0
        assert result["steps"][0]["step_id"] is None
        assert result["steps"][1]["step_index"] == 1

    def test_includes_pipeline_metadata(self):
        """Provenance includes pipeline commit and file."""
        task = SimpleTrackedTask({})
        gen = task.task_generator()

        req1 = gen.send(None)
        req2 = gen.send(make_response(req1, "a"))
        with pytest.raises(StopIteration) as exc_info:
            gen.send(make_response(req2, "b"))

        result = exc_info.value.value
        assert "pipeline_commit" in result
        assert "pipeline_file" in result
        assert "started_at" in result
        assert "completed_at" in result

    def test_captures_error_response(self):
        """TrackedTask records errors from failed responses."""

        class SingleCallTask(TrackedTask):
            def process_record(self):
                resp = yield model_request([{"role": "user", "content": "x"}])
                return TrainingSample(id="err", messages=[])

        task = SingleCallTask({})
        gen = task.task_generator()

        req = gen.send(None)
        error_resp = Response.from_error(req, Exception("API Error"))

        with pytest.raises(StopIteration) as exc_info:
            gen.send(error_resp)

        result = exc_info.value.value
        step = result["steps"][0]
        assert step["error"] == "API Error"
        assert step["response"] == ""

    def test_captures_input_data(self):
        """TrackedTask captures original input data."""

        class SimpleTask(TrackedTask):
            def process_record(self):
                yield model_request([{"role": "user", "content": "x"}])
                return TrainingSample(id=self.data["id"], messages=[])

        input_data = {"id": "123", "extra_field": "extra_value", "nested": {"a": 1}}
        task = SimpleTask(input_data)
        gen = task.task_generator()

        req = gen.send(None)
        with pytest.raises(StopIteration) as exc_info:
            gen.send(make_response(req, "response"))

        result = exc_info.value.value
        assert result["input_data"] == input_data


class TestInferenceStep:
    """Tests for InferenceStep dataclass."""

    def test_is_success(self):
        step_ok = InferenceStep(
            messages=[],
            model="m",
            model_resolved="m",
            sampling={},
            response="hello",
        )
        assert step_ok.is_success

        step_err = InferenceStep(
            messages=[],
            model="m",
            model_resolved="m",
            sampling={},
            response="",
            error="failed",
        )
        assert not step_err.is_success

    def test_to_dict(self):
        step = InferenceStep(
            messages=[{"role": "user", "content": "hi"}],
            model="test-model",
            model_resolved="resolved",
            sampling={"temperature": 0.5, "stop": ["END", "STOP"]},
            response="hello",
            step_id="greet",
            step_index=0,
        )
        d = step.to_dict()
        assert d["messages"] == [{"role": "user", "content": "hi"}]
        assert d["model"] == "test-model"
        assert d["sampling"] == {"temperature": 0.5, "stop": ["END", "STOP"]}
        assert d["step_id"] == "greet"
        assert "timestamp" in d


class TestTrainingSample:
    """Tests for TrainingSample dataclass."""

    def test_to_dict(self):
        sample = TrainingSample(
            id="123",
            messages=[{"role": "user", "content": "hi"}],
        )
        d = sample.to_dict()
        assert d == {"id": "123", "messages": [{"role": "user", "content": "hi"}]}


class TestAnnotatedTrainingSample:
    """Tests for AnnotatedTrainingSample dataclass."""

    def test_inherits_from_training_sample(self):
        annotated = AnnotatedTrainingSample(
            id="123",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert isinstance(annotated, TrainingSample)

    def test_to_train_sample(self):
        annotated = AnnotatedTrainingSample(
            id="123",
            messages=[{"role": "user", "content": "hi"}],
            input_data={"original": "data"},
            steps=[],
            pipeline_commit="abc123",
        )
        sample = annotated.to_train_sample()
        assert isinstance(sample, TrainingSample)
        assert not isinstance(sample, AnnotatedTrainingSample)
        assert sample.id == "123"
        assert sample.messages == [{"role": "user", "content": "hi"}]

    def test_to_dict(self):
        step = InferenceStep(
            messages=[{"role": "user", "content": "hello"}],
            model="test",
            model_resolved="test-resolved",
            sampling={"temperature": 0.5},
            response="world",
            step_id="greet",
            step_index=0,
        )
        now = datetime.now()
        annotated = AnnotatedTrainingSample(
            id="123",
            messages=[{"role": "user", "content": "hi"}],
            input_data={"prompt": "hi"},
            steps=[step],
            pipeline_commit="abc123",
            pipeline_file="/path/to/task.py",
            started_at=now,
            completed_at=now,
        )

        d = annotated.to_dict()

        # Core fields
        assert d["id"] == "123"
        assert d["messages"] == [{"role": "user", "content": "hi"}]

        # Provenance
        assert d["input_data"] == {"prompt": "hi"}
        assert d["pipeline_commit"] == "abc123"
        assert d["pipeline_file"] == "/path/to/task.py"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["step_id"] == "greet"

    def test_from_dict_roundtrip(self):
        """Serialization/deserialization roundtrip."""
        step = InferenceStep(
            messages=[{"role": "user", "content": "hello"}],
            model="test",
            model_resolved="resolved",
            sampling={"temperature": 0.5, "stop": ["END"]},
            response="world",
            step_id="greet",
            step_index=0,
        )
        now = datetime.now()
        original = AnnotatedTrainingSample(
            id="123",
            messages=[{"role": "user", "content": "hi"}],
            input_data={"prompt": "hi"},
            steps=[step],
            pipeline_commit="abc123",
            pipeline_file="/path/to/task.py",
            started_at=now,
            completed_at=now,
        )

        d = original.to_dict()
        restored = AnnotatedTrainingSample.from_dict(d)

        assert restored.id == original.id
        assert restored.messages == original.messages
        assert restored.input_data == original.input_data
        assert restored.pipeline_commit == original.pipeline_commit
        assert len(restored.steps) == 1
        assert restored.steps[0].step_id == "greet"
        assert restored.steps[0].sampling == {"temperature": 0.5, "stop": ["END"]}

    def test_timestamps_are_iso_format(self):
        """Timestamps serialize to ISO format."""
        now = datetime.now()
        annotated = AnnotatedTrainingSample(
            id="1",
            messages=[],
            started_at=now,
            completed_at=now,
        )

        d = annotated.to_dict()
        assert d["started_at"] == now.isoformat()
        assert d["completed_at"] == now.isoformat()

    def test_completed_at_none(self):
        """Handles None completed_at."""
        annotated = AnnotatedTrainingSample(id="1", messages=[], completed_at=None)
        d = annotated.to_dict()
        assert d["completed_at"] is None


class TestModelRequest:
    """Tests for model_request helper."""

    def test_basic_request(self):
        """Creates request with messages."""
        req = model_request([{"role": "user", "content": "hi"}])
        assert req.content["messages"] == [{"role": "user", "content": "hi"}]
        assert "_model" not in req.content
        assert "_step_id" not in req.content

    def test_with_model(self):
        """Includes _model when specified."""
        req = model_request([{"role": "user", "content": "hi"}], model="test-model")
        assert req.content["_model"] == "test-model"

    def test_with_step_id(self):
        """Includes _step_id when specified."""
        req = model_request([{"role": "user", "content": "hi"}], step_id="generate")
        assert req.content["_step_id"] == "generate"

    def test_with_sampling_kwargs(self):
        """Passes through sampling parameters."""
        req = model_request(
            [{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
        )
        assert req.content["temperature"] == 0.5
        assert req.content["max_tokens"] == 100
        assert req.content["top_p"] == 0.9

    def test_with_list_sampling_param(self):
        """Handles list sampling params like stop."""
        req = model_request(
            [{"role": "user", "content": "hi"}],
            stop=["END", "STOP", "###"],
        )
        assert req.content["stop"] == ["END", "STOP", "###"]

    def test_all_options(self):
        """All options together."""
        req = model_request(
            [{"role": "user", "content": "hi"}],
            model="my-model",
            step_id="step1",
            temperature=0.7,
        )
        assert req.content["messages"] == [{"role": "user", "content": "hi"}]
        assert req.content["_model"] == "my-model"
        assert req.content["_step_id"] == "step1"
        assert req.content["temperature"] == 0.7


class TestTrackedTaskEdgeCases:
    """Edge case tests for TrackedTask."""

    def test_empty_messages(self):
        """Handles empty messages list."""

        class EmptyMsgTask(TrackedTask):
            def process_record(self):
                resp = yield model_request([])
                return TrainingSample(id="empty", messages=[])

        task = EmptyMsgTask({})
        gen = task.task_generator()

        req = gen.send(None)
        assert req.content["messages"] == []

        with pytest.raises(StopIteration) as exc_info:
            gen.send(make_response(req, "response"))

        result = exc_info.value.value
        assert result["steps"][0]["messages"] == []

    def test_preserves_all_sampling_params(self):
        """All sampling params are captured."""

        class FullSamplingTask(TrackedTask):
            def process_record(self):
                resp = yield model_request(
                    [{"role": "user", "content": "x"}],
                    temperature=0.8,
                    max_tokens=500,
                    top_p=0.95,
                    stop=["END"],
                    n=3,
                )
                return TrainingSample(id="full", messages=[])

        task = FullSamplingTask({})
        gen = task.task_generator()

        req = gen.send(None)
        with pytest.raises(StopIteration) as exc_info:
            gen.send(make_response(req, "a"))

        sampling = exc_info.value.value["steps"][0]["sampling"]
        assert sampling == {
            "temperature": 0.8,
            "max_tokens": 500,
            "top_p": 0.95,
            "stop": ["END"],
            "n": 3,
        }

    def test_many_sequential_steps(self):
        """Handles many sequential steps correctly."""

        class ManyStepsTask(TrackedTask):
            def process_record(self):
                for i in range(5):
                    yield model_request(
                        [{"role": "user", "content": f"msg{i}"}],
                        step_id=f"step_{i}",
                    )
                return TrainingSample(id="many", messages=[])

        task = ManyStepsTask({})
        gen = task.task_generator()

        req = gen.send(None)
        for i in range(4):
            req = gen.send(make_response(req, f"resp{i}"))

        with pytest.raises(StopIteration) as exc_info:
            gen.send(make_response(req, "resp4"))

        steps = exc_info.value.value["steps"]
        assert len(steps) == 5
        for i, step in enumerate(steps):
            assert step["step_id"] == f"step_{i}"
            assert step["step_index"] == i

    def test_mixed_parallel_and_sequential(self):
        """Handles mix of parallel and sequential steps."""

        class MixedTask(TrackedTask):
            def process_record(self):
                # Sequential
                r1 = yield model_request([{"role": "user", "content": "1"}], step_id="first")

                # Parallel
                responses = yield [
                    model_request([{"role": "user", "content": "2a"}], step_id="parallel_a"),
                    model_request([{"role": "user", "content": "2b"}], step_id="parallel_b"),
                ]

                # Sequential again
                r2 = yield model_request([{"role": "user", "content": "3"}], step_id="last")

                return TrainingSample(id="mixed", messages=[])

        task = MixedTask({})
        gen = task.task_generator()

        # First sequential
        req1 = gen.send(None)
        assert req1.content["_step_id"] == "first"

        # Parallel
        reqs = gen.send(make_response(req1, "r1"))
        assert isinstance(reqs, list)
        assert len(reqs) == 2

        # Last sequential
        req_last = gen.send([make_response(reqs[0], "r2a"), make_response(reqs[1], "r2b")])
        assert req_last.content["_step_id"] == "last"

        with pytest.raises(StopIteration) as exc_info:
            gen.send(make_response(req_last, "r3"))

        steps = exc_info.value.value["steps"]
        assert len(steps) == 4
        assert [s["step_id"] for s in steps] == ["first", "parallel_a", "parallel_b", "last"]
        assert [s["step_index"] for s in steps] == [0, 1, 2, 3]


class TestGetGitCommit:
    """Tests for git commit helper."""

    def test_returns_short_hash_or_none(self):
        # Just verify it doesn't crash and returns string or None
        result = get_git_commit()
        assert result is None or (isinstance(result, str) and len(result) == 12)

    @patch("subprocess.run")
    def test_returns_none_on_error(self, mock_run):
        """Returns None when git command fails."""
        mock_run.side_effect = Exception("git not found")
        result = get_git_commit()
        assert result is None

    @patch("subprocess.run")
    def test_returns_none_on_nonzero_exit(self, mock_run):
        """Returns None when git returns non-zero."""
        mock_run.return_value = MagicMock(returncode=1)
        result = get_git_commit()
        assert result is None

    @patch("subprocess.run")
    def test_truncates_to_12_chars(self, mock_run):
        """Truncates commit hash to 12 characters."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123def456789extra\n",
        )
        result = get_git_commit()
        assert result == "abc123def456"
