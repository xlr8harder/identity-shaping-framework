"""Tests for pipeline dependency declarations."""

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from shaping.pipeline import Pipeline, ModelDep, FileDep, model_request
from shaping.pipeline.deps import get_all_deps, get_model_deps, get_file_deps


class TestModelDep:
    """Tests for ModelDep descriptor."""

    def test_stores_registry_name(self):
        dep = ModelDep("cubsfan-release-full")
        assert dep.registry_name == "cubsfan-release-full"

    def test_repr(self):
        dep = ModelDep("cubsfan-release-full")
        assert repr(dep) == "ModelDep('cubsfan-release-full')"

    def test_registers_on_class(self):
        class TestPipeline(Pipeline):
            name = "test"
            identity_model = ModelDep("cubsfan-release-full")

        assert "identity_model" in TestPipeline._model_deps
        assert (
            TestPipeline._model_deps["identity_model"].registry_name
            == "cubsfan-release-full"
        )

    def test_accessible_on_instance(self):
        class TestPipeline(Pipeline):
            name = "test"
            identity_model = ModelDep("cubsfan-release-full")

        pipeline = TestPipeline()
        assert isinstance(pipeline.identity_model, ModelDep)
        assert pipeline.identity_model.registry_name == "cubsfan-release-full"

    def test_to_manifest_entry(self):
        dep = ModelDep("cubsfan-release-full")
        entry = dep.to_manifest_entry()
        assert entry["type"] == "model"
        assert entry["registry"] == "cubsfan-release-full"
        # sysprompt_hash may be None if registry not configured


class TestFileDep:
    """Tests for FileDep descriptor."""

    def test_stores_path(self):
        dep = FileDep("identity/NARRATIVE.md")
        assert dep.path == Path("identity/NARRATIVE.md")

    def test_repr(self):
        dep = FileDep("identity/NARRATIVE.md")
        assert repr(dep) == "FileDep('identity/NARRATIVE.md')"

    def test_registers_on_class(self):
        class TestPipeline(Pipeline):
            name = "test"
            narrative_doc = FileDep("identity/NARRATIVE.md")

        assert "narrative_doc" in TestPipeline._file_deps
        assert TestPipeline._file_deps["narrative_doc"].path == Path(
            "identity/NARRATIVE.md"
        )

    def test_read_returns_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test Content\n")
            f.flush()
            path = f.name

        try:
            dep = FileDep(path)
            assert dep.read() == "# Test Content\n"
        finally:
            Path(path).unlink()

    def test_exists(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("test")
            path = f.name

        try:
            dep = FileDep(path)
            assert dep.exists() is True
        finally:
            Path(path).unlink()

        # After deletion
        assert dep.exists() is False

    def test_get_content_hash(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("test content")
            path = f.name

        try:
            dep = FileDep(path)
            hash1 = dep.get_content_hash()
            assert hash1 is not None
            assert len(hash1) == 16  # Truncated SHA256

            # Same content = same hash
            assert dep.get_content_hash() == hash1
        finally:
            Path(path).unlink()

        # Missing file returns None
        assert dep.get_content_hash() is None

    def test_to_manifest_entry(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("test")
            path = f.name

        try:
            dep = FileDep(path)
            entry = dep.to_manifest_entry()
            assert entry["type"] == "file"
            assert entry["path"] == path
            assert entry["content_hash"] is not None
        finally:
            Path(path).unlink()


class TestGetDeps:
    """Tests for dependency introspection functions."""

    def test_get_all_deps(self):
        class TestPipeline(Pipeline):
            name = "test"
            identity_model = ModelDep("cubsfan-release-full")
            narrative_doc = FileDep("identity/NARRATIVE.md")

        deps = get_all_deps(TestPipeline)
        assert "identity_model" in deps
        assert "narrative_doc" in deps
        assert len(deps) == 2

    def test_get_model_deps(self):
        class TestPipeline(Pipeline):
            name = "test"
            identity_model = ModelDep("cubsfan-release-full")
            judge_model = ModelDep("judge")
            narrative_doc = FileDep("identity/NARRATIVE.md")

        deps = get_model_deps(TestPipeline)
        assert "identity_model" in deps
        assert "judge_model" in deps
        assert "narrative_doc" not in deps
        assert len(deps) == 2

    def test_get_file_deps(self):
        class TestPipeline(Pipeline):
            name = "test"
            identity_model = ModelDep("cubsfan-release-full")
            narrative_doc = FileDep("identity/NARRATIVE.md")
            other_doc = FileDep("other.md")

        deps = get_file_deps(TestPipeline)
        assert "narrative_doc" in deps
        assert "other_doc" in deps
        assert "identity_model" not in deps
        assert len(deps) == 2

    def test_no_deps(self):
        class TestPipeline(Pipeline):
            name = "test"

        assert get_all_deps(TestPipeline) == {}
        assert get_model_deps(TestPipeline) == {}
        assert get_file_deps(TestPipeline) == {}


class TestPipelineFactoryMethods:
    """Tests for Pipeline.model_dep and Pipeline.file_dep."""

    def test_model_dep_factory(self):
        dep = Pipeline.model_dep("cubsfan-release-full")
        assert isinstance(dep, ModelDep)
        assert dep.registry_name == "cubsfan-release-full"

    def test_file_dep_factory(self):
        dep = Pipeline.file_dep("identity/NARRATIVE.md")
        assert isinstance(dep, FileDep)
        assert dep.path == Path("identity/NARRATIVE.md")


class TestModelRequest:
    """Tests for model_request with ModelDep."""

    def test_accepts_model_dep(self):
        dep = ModelDep("cubsfan-release-full")
        request = model_request([{"role": "user", "content": "test"}], model=dep)
        assert request.content["_model"] == "cubsfan-release-full"

    def test_accepts_string_for_backwards_compat(self):
        # Strings still work during migration
        request = model_request(
            [{"role": "user", "content": "test"}], model="cubsfan-release-full"
        )
        assert request.content["_model"] == "cubsfan-release-full"

    def test_accepts_none(self):
        request = model_request([{"role": "user", "content": "test"}], model=None)
        assert "_model" not in request.content

    def test_rejects_invalid_type(self):
        with pytest.raises(TypeError) as exc_info:
            model_request([{"role": "user", "content": "test"}], model=123)
        assert "ModelDep" in str(exc_info.value)


class TestPipelineBasics:
    """Tests for Pipeline base class."""

    def test_requires_name(self):
        class BadPipeline(Pipeline):
            pass  # No name

        with pytest.raises(ValueError) as exc_info:
            BadPipeline()
        assert "name" in str(exc_info.value)

    def test_get_output_file(self):
        class TestPipeline(Pipeline):
            name = "test-pipeline"

        assert TestPipeline.get_output_file() == Path(
            "training/data/test-pipeline.jsonl"
        )

    def test_get_manifest_file(self):
        class TestPipeline(Pipeline):
            name = "test-pipeline"

        assert TestPipeline.get_manifest_file() == Path(
            "training/data/test-pipeline.manifest.json"
        )

    def test_query_requires_model_dep(self):
        class TestPipeline(Pipeline):
            name = "test"

        pipeline = TestPipeline()
        with pytest.raises(TypeError) as exc_info:
            pipeline.query(model="not-a-dep", messages=[])
        assert "ModelDep" in str(exc_info.value)


class TestStalenessCheck:
    """Tests for staleness detection."""

    def test_stale_when_no_manifest(self):
        class TestPipeline(Pipeline):
            name = "nonexistent-pipeline"
            output_dir = Path("/tmp/test-pipelines")

        status = TestPipeline.check_staleness()
        assert status["stale"] is True
        assert "No manifest" in status["reasons"][0]

    def test_stale_when_file_changed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a file dep
            test_file = tmpdir / "test.md"
            test_file.write_text("original content")

            class TestPipeline(Pipeline):
                name = "test-pipeline"
                output_dir = tmpdir
                doc = FileDep(str(test_file))

            # Get the code hash for the manifest (matches what check_staleness will compute)
            import hashlib
            import inspect

            source_file = inspect.getfile(TestPipeline)
            code_hash = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()[:16]

            # Create manifest with current hash
            manifest = {
                "pipeline": "test-pipeline",
                "code_hash": code_hash,
                "deps": {
                    "doc": {
                        "type": "file",
                        "path": str(test_file),
                        "content_hash": TestPipeline.doc.get_content_hash(),
                    }
                },
            }
            manifest_file = tmpdir / "test-pipeline.manifest.json"
            manifest_file.write_text(json.dumps(manifest))

            # Initially not stale
            status = TestPipeline.check_staleness()
            assert (
                status["stale"] is False
            ), f"Unexpected staleness: {status['reasons']}"

            # Change file content
            test_file.write_text("modified content")

            # Now stale
            status = TestPipeline.check_staleness()
            assert status["stale"] is True
            assert any("doc" in r for r in status["reasons"])

    def test_stale_when_partial_run(self, tmpdir, monkeypatch):
        """Partial runs (with --limit) should show as stale."""
        monkeypatch.chdir(tmpdir)

        # Create a test file
        test_file = Path(str(tmpdir)) / "test.md"
        test_file.write_text("test content")

        class TestPipeline(Pipeline):
            name = "test-pipeline"
            output_dir = Path(str(tmpdir))
            doc = Pipeline.file_dep(str(test_file))

            def run(self):
                return []

        # Get actual code hash
        import inspect

        source_file = inspect.getfile(TestPipeline)
        code_hash = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()[:16]

        # Create manifest with partial=True
        manifest = {
            "pipeline": "test-pipeline",
            "code_hash": code_hash,
            "deps": {
                "doc": {
                    "type": "file",
                    "path": str(test_file),
                    "content_hash": TestPipeline.doc.get_content_hash(),
                }
            },
            "partial": True,
            "record_count": 10,
        }
        manifest_file = Path(str(tmpdir)) / "test-pipeline.manifest.json"
        manifest_file.write_text(json.dumps(manifest))

        # Should be stale because it's a partial run
        status = TestPipeline.check_staleness()
        assert status["stale"] is True
        assert status["partial"] is True
        assert status["record_count"] == 10
        assert any("Partial run" in r for r in status["reasons"])

    def test_current_when_full_run(self, tmpdir, monkeypatch):
        """Full runs (partial=False) should show as current."""
        monkeypatch.chdir(tmpdir)

        # Create a test file
        test_file = Path(str(tmpdir)) / "test.md"
        test_file.write_text("test content")

        class TestPipeline(Pipeline):
            name = "test-pipeline"
            output_dir = Path(str(tmpdir))
            doc = Pipeline.file_dep(str(test_file))

            def run(self):
                return []

        # Get actual code hash
        import inspect

        source_file = inspect.getfile(TestPipeline)
        code_hash = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()[:16]

        # Create manifest with partial=False
        manifest = {
            "pipeline": "test-pipeline",
            "code_hash": code_hash,
            "deps": {
                "doc": {
                    "type": "file",
                    "path": str(test_file),
                    "content_hash": TestPipeline.doc.get_content_hash(),
                }
            },
            "partial": False,
            "record_count": 1000,
        }
        manifest_file = Path(str(tmpdir)) / "test-pipeline.manifest.json"
        manifest_file.write_text(json.dumps(manifest))

        # Should be current
        status = TestPipeline.check_staleness()
        assert status["stale"] is False
        assert status["partial"] is False
        assert status["record_count"] == 1000


class TestAnnotatedOutput:
    """Tests for --annotate/--no-annotate output format."""

    def test_default_annotated_includes_provenance(self, tmpdir, monkeypatch):
        """Default (annotated=None) should include all provenance fields."""
        monkeypatch.chdir(tmpdir)
        output_file = Path(str(tmpdir)) / "test.jsonl"

        class TestPipeline(Pipeline):
            name = "test-pipeline"
            output_dir = Path(str(tmpdir))

            def run(self):
                return [
                    {
                        "id": "test-1",
                        "messages": [{"role": "user", "content": "hi"}],
                        "input_data": {"original": "data"},
                        "steps": [{"model": "test"}],
                        "pipeline_commit": "abc123",
                    }
                ]

        pipeline = TestPipeline()
        pipeline.execute(output_file=output_file)  # No annotated param

        with open(output_file) as f:
            result = json.loads(f.readline())

        # Should have all fields (default is annotated=True)
        assert result["id"] == "test-1"
        assert result["input_data"] == {"original": "data"}
        assert result["steps"] == [{"model": "test"}]

    def test_class_level_annotated_false(self, tmpdir, monkeypatch):
        """Pipeline with annotated=False class attribute should strip by default."""
        monkeypatch.chdir(tmpdir)
        output_file = Path(str(tmpdir)) / "test.jsonl"

        class MinimalPipeline(Pipeline):
            name = "minimal-pipeline"
            output_dir = Path(str(tmpdir))
            annotated = False  # Class-level default

            def run(self):
                return [
                    {
                        "id": "test-1",
                        "messages": [{"role": "user", "content": "hi"}],
                        "input_data": {"original": "data"},
                        "steps": [{"model": "test"}],
                    }
                ]

        pipeline = MinimalPipeline()
        pipeline.execute(output_file=output_file)  # No annotated param

        with open(output_file) as f:
            result = json.loads(f.readline())

        # Should only have id and messages
        assert result == {
            "id": "test-1",
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_execute_param_overrides_class_default(self, tmpdir, monkeypatch):
        """Execute annotated param should override class default."""
        monkeypatch.chdir(tmpdir)
        output_file = Path(str(tmpdir)) / "test.jsonl"

        class MinimalPipeline(Pipeline):
            name = "minimal-pipeline"
            output_dir = Path(str(tmpdir))
            annotated = False  # Class-level default is minimal

            def run(self):
                return [
                    {
                        "id": "test-1",
                        "messages": [{"role": "user", "content": "hi"}],
                        "input_data": {"original": "data"},
                    }
                ]

        pipeline = MinimalPipeline()
        pipeline.execute(output_file=output_file, annotated=True)  # Override

        with open(output_file) as f:
            result = json.loads(f.readline())

        # Should have all fields despite class default
        assert result["id"] == "test-1"
        assert result["input_data"] == {"original": "data"}

    def test_annotated_false_strips_provenance(self, tmpdir, monkeypatch):
        """annotated=False should strip to just id and messages."""
        monkeypatch.chdir(tmpdir)
        output_file = Path(str(tmpdir)) / "test.jsonl"

        class TestPipeline(Pipeline):
            name = "test-pipeline"
            output_dir = Path(str(tmpdir))

            def run(self):
                return [
                    {
                        "id": "test-1",
                        "messages": [{"role": "user", "content": "hi"}],
                        "input_data": {"original": "data"},
                        "steps": [{"model": "test"}],
                        "pipeline_commit": "abc123",
                    }
                ]

        pipeline = TestPipeline()
        pipeline.execute(output_file=output_file, annotated=False)

        with open(output_file) as f:
            result = json.loads(f.readline())

        # Should only have id and messages
        assert result == {
            "id": "test-1",
            "messages": [{"role": "user", "content": "hi"}],
        }
        assert "input_data" not in result
        assert "steps" not in result
        assert "pipeline_commit" not in result


class TestErrorHandling:
    """Tests for task error handling with PipelineError."""

    def test_pipeline_error_produces_error_output(self):
        """Raising PipelineError should produce __ERROR__ output."""
        from shaping.pipeline.tasks import TrackedTask, model_request, PipelineError
        from shaping.pipeline.provenance import TrainingSample
        from dispatcher.taskmanager.backend.request import Response

        class SkipTask(TrackedTask):
            name = "skip-task"

            def process_record(self):
                response = yield model_request(
                    [{"role": "user", "content": "hi"}], model="test"
                )
                if not response.is_success:
                    raise PipelineError("API call failed", error_type="api_error")
                return TrainingSample(id=self.data["id"], messages=[])

        task = SkipTask({"id": "test-123"})
        req = task.get_next_request()

        # Simulate error response
        error_response = Response.from_error(req, Exception("API Error"))
        task.process_result(error_response)

        assert task.is_done()
        result, _ = task.get_result()
        assert "__ERROR__" in result
        assert result["__ERROR__"]["error"] == "api_error"
        assert result["__ERROR__"]["message"] == "API call failed"

    def test_pipeline_error_includes_steps(self):
        """PipelineError should include inference steps for debugging."""
        from shaping.pipeline.tasks import TrackedTask, model_request, PipelineError
        from shaping.pipeline.provenance import TrainingSample
        from dispatcher.taskmanager.backend.request import Response

        class MultiStepTask(TrackedTask):
            name = "multi-step-task"

            def process_record(self):
                # First call succeeds
                _resp1 = yield model_request(
                    [{"role": "user", "content": "step1"}], model="test"
                )
                # Second call fails
                resp2 = yield model_request(
                    [{"role": "user", "content": "step2"}], model="test"
                )
                if not resp2.is_success:
                    raise PipelineError("Step 2 failed", error_type="step2_error")
                return TrainingSample(id=self.data["id"], messages=[])

        task = MultiStepTask({"id": "test-123"})

        # First request
        req1 = task.get_next_request()
        success_response = Response(
            req1, {"choices": [{"message": {"content": "OK"}}]}, model_name="test"
        )
        task.process_result(success_response)

        # Second request
        req2 = task.get_next_request()
        error_response = Response.from_error(req2, Exception("API Error"))
        task.process_result(error_response)

        assert task.is_done()
        result, _ = task.get_result()
        assert "__ERROR__" in result
        assert result["__ERROR__"]["error"] == "step2_error"
        # Should have both steps recorded
        assert "steps" in result["__ERROR__"]
        assert len(result["__ERROR__"]["steps"]) == 2

    def test_unexpected_exception_captured_with_steps(self):
        """Unexpected exceptions should still capture steps for debugging."""
        from shaping.pipeline.tasks import TrackedTask, model_request
        from shaping.pipeline.provenance import TrainingSample
        from dispatcher.taskmanager.backend.request import Response

        class BuggyTask(TrackedTask):
            name = "buggy-task"

            def process_record(self):
                response = yield model_request(
                    [{"role": "user", "content": "hi"}], model="test"
                )
                # Bug: accessing a key that doesn't exist
                _ = response.content["nonexistent_key"]
                return TrainingSample(id=self.data["id"], messages=[])

        task = BuggyTask({"id": "test-123"})
        req = task.get_next_request()

        # Simulate success response (but task will crash trying to access bad key)
        success_response = Response(
            req, {"choices": [{"message": {"content": "OK"}}]}, model_name="test"
        )
        task.process_result(success_response)

        assert task.is_done()
        result, _ = task.get_result()
        assert "__ERROR__" in result
        assert result["__ERROR__"]["error"] == "unexpected_error"
        assert "KeyError" in result["__ERROR__"]["message"]
        # Should still have the step recorded
        assert "steps" in result["__ERROR__"]
        assert len(result["__ERROR__"]["steps"]) == 1

    def test_tracked_task_success(self):
        """TrackedTask should return normal result on success."""
        from shaping.pipeline.tasks import TrackedTask, model_request
        from shaping.pipeline.provenance import TrainingSample
        from dispatcher.taskmanager.backend.request import Response

        class SuccessTask(TrackedTask):
            name = "success-task"

            def process_record(self):
                response = yield model_request(
                    [{"role": "user", "content": "hi"}], model="test"
                )
                return TrainingSample(
                    id=self.data["id"],
                    messages=[{"role": "assistant", "content": response.get_text()}],
                )

        task = SuccessTask({"id": "test-123"})
        req = task.get_next_request()

        # Simulate success response
        success_response = Response(
            req, {"choices": [{"message": {"content": "Hello!"}}]}, model_name="test"
        )
        task.process_result(success_response)

        assert task.is_done()
        result, _ = task.get_result()
        assert "__ERROR__" not in result
        assert result["id"] == "test-123"
        assert result["messages"][0]["content"] == "Hello!"

    def test_pipeline_error_before_first_yield(self):
        """PipelineError raised before any yield should still work."""
        from shaping.pipeline.tasks import TrackedTask, PipelineError

        class ImmediateErrorTask(TrackedTask):
            name = "immediate-error-task"

            def process_record(self):
                # Fail immediately before any yield
                raise PipelineError("Invalid input data", error_type="validation_error")
                yield  # Never reached

        # The dispatcher's __init__ triggers task_generator(), which catches
        # PipelineError and returns the error result immediately
        task = ImmediateErrorTask({"id": "test-123", "bad_field": True})

        # Task should be done immediately since error happened before first yield
        assert task.is_done()
        result, _ = task.get_result()
        assert "__ERROR__" in result
        assert result["__ERROR__"]["error"] == "validation_error"
        assert result["__ERROR__"]["message"] == "Invalid input data"
        # No steps since we never yielded
        assert result["__ERROR__"]["steps"] == []

    def test_pipeline_error_default_error_type(self):
        """PipelineError with default error_type should use 'error'."""
        from shaping.pipeline.tasks import TrackedTask, model_request, PipelineError
        from dispatcher.taskmanager.backend.request import Response

        class DefaultErrorTask(TrackedTask):
            name = "default-error-task"

            def process_record(self):
                _response = yield model_request(
                    [{"role": "user", "content": "hi"}], model="test"
                )
                # Use default error_type (no second arg)
                raise PipelineError("Something went wrong")

        task = DefaultErrorTask({"id": "test-123"})
        req = task.get_next_request()

        success_response = Response(
            req, {"choices": [{"message": {"content": "OK"}}]}, model_name="test"
        )
        task.process_result(success_response)

        assert task.is_done()
        result, _ = task.get_result()
        assert "__ERROR__" in result
        assert result["__ERROR__"]["error"] == "error"  # Default

    def test_parallel_request_with_error(self):
        """Parallel requests should record all steps even if error follows."""
        from shaping.pipeline.tasks import TrackedTask, model_request, PipelineError
        from shaping.pipeline.provenance import TrainingSample
        from dispatcher.taskmanager.backend.request import Response

        class ParallelTask(TrackedTask):
            name = "parallel-task"

            def process_record(self):
                # Yield parallel requests
                responses = yield [
                    model_request([{"role": "user", "content": "q1"}], model="test"),
                    model_request([{"role": "user", "content": "q2"}], model="test"),
                ]
                # Check responses and fail
                if any(not r.is_success for r in responses):
                    raise PipelineError(
                        "One request failed", error_type="partial_failure"
                    )
                return TrainingSample(id=self.data["id"], messages=[])

        task = ParallelTask({"id": "test-123"})

        # Dispatcher queues parallel requests individually
        req1 = task.get_next_request()
        req2 = task.get_next_request()
        assert req1 is not None
        assert req2 is not None

        # Process responses (one succeeds, one fails)
        task.process_result(
            Response(
                req1, {"choices": [{"message": {"content": "OK"}}]}, model_name="test"
            )
        )
        task.process_result(Response.from_error(req2, Exception("API Error")))

        assert task.is_done()
        result, _ = task.get_result()
        assert "__ERROR__" in result
        assert result["__ERROR__"]["error"] == "partial_failure"
        # Both steps should be recorded
        assert len(result["__ERROR__"]["steps"]) == 2


class TestRunTaskFiltering:
    """Test that run_task() correctly filters out error records."""

    def test_run_task_filters_error_records(self, tmpdir, monkeypatch):
        """run_task() should return only successful results, filtering out __ERROR__ records."""
        monkeypatch.chdir(tmpdir)

        from shaping.pipeline import Pipeline, TrainingSample, model_request

        # Mock run_pipeline to write output with mixed results
        def mock_run_pipeline(task_class, input_file, output_file, num_workers):
            import json

            with open(output_file, "w") as f:
                # Successful result
                f.write(
                    json.dumps(
                        {
                            "id": "success-1",
                            "messages": [{"role": "user", "content": "hi"}],
                        }
                    )
                    + "\n"
                )
                # Error result (should be filtered)
                f.write(
                    json.dumps(
                        {
                            "__ERROR__": {
                                "error": "parse_error",
                                "message": "Could not parse",
                            }
                        }
                    )
                    + "\n"
                )
                # Another successful result
                f.write(
                    json.dumps(
                        {
                            "id": "success-2",
                            "messages": [{"role": "user", "content": "bye"}],
                        }
                    )
                    + "\n"
                )
                # Another error (should be filtered)
                f.write(
                    json.dumps(
                        {
                            "__ERROR__": {
                                "error": "api_error",
                                "message": "API failed",
                            }
                        }
                    )
                    + "\n"
                )

        class TestPipeline(Pipeline):
            name = "test-pipeline"
            output_dir = Path(str(tmpdir))

            def run(self):
                # This won't actually be called; we're testing run_task filtering
                return []

            def process_item(self, record):
                # This generator method won't actually run with our mock
                yield model_request([{"role": "user", "content": "test"}], model="test")
                return TrainingSample(id=record["id"], messages=[])

        pipeline = TestPipeline()

        with patch("shaping.pipeline.runner.run_pipeline", mock_run_pipeline):
            results = pipeline.run_task(
                pipeline.process_item,
                records=[{"id": "1"}, {"id": "2"}, {"id": "3"}, {"id": "4"}],
            )

        # Should only have the 2 successful results
        assert len(results) == 2
        assert results[0]["id"] == "success-1"
        assert results[1]["id"] == "success-2"

    def test_run_task_logs_filtered_count(self, tmpdir, monkeypatch, caplog):
        """run_task() should log warning about filtered error records."""
        import logging

        monkeypatch.chdir(tmpdir)

        from shaping.pipeline import Pipeline, TrainingSample, model_request

        def mock_run_pipeline(task_class, input_file, output_file, num_workers):
            import json

            with open(output_file, "w") as f:
                f.write(json.dumps({"id": "1", "messages": []}) + "\n")
                f.write(json.dumps({"__ERROR__": {"error": "e1"}}) + "\n")
                f.write(json.dumps({"__ERROR__": {"error": "e2"}}) + "\n")
                f.write(json.dumps({"__ERROR__": {"error": "e3"}}) + "\n")

        class TestPipeline(Pipeline):
            name = "test-pipeline"
            output_dir = Path(str(tmpdir))

            def run(self):
                return []

            def process_item(self, record):
                yield model_request([{"role": "user", "content": "test"}], model="test")
                return TrainingSample(id=record["id"], messages=[])

        pipeline = TestPipeline()

        with patch("shaping.pipeline.runner.run_pipeline", mock_run_pipeline):
            with caplog.at_level(logging.WARNING):
                results = pipeline.run_task(
                    pipeline.process_item,
                    records=[{"id": "1"}, {"id": "2"}, {"id": "3"}, {"id": "4"}],
                )

        assert len(results) == 1
        assert "Filtered out 3 failed/skipped records" in caplog.text
