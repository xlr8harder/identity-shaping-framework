"""Tests for shaping.results module."""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from shaping.results import (
    EvalResult,
    BaseModelSpec,
    PromptedModelSpec,
    TrainedModelSpec,
    PromptedTrainedModelSpec,
    ModelSpec,
    EvalConfig,
    SamplingConfig,
    Results,
    ResultsStore,
)
from shaping.results.schema import ErrorBreakdown, Artifacts
from shaping.training.config import TrainConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_model():
    """A base model spec."""
    return BaseModelSpec(
        alias="claude-haiku",
        provider="anthropic",
        model_id="claude-3-haiku-20240307",
    )


@pytest.fixture
def prompted_model():
    """A prompted model spec."""
    return PromptedModelSpec(
        alias="aria-v0.9-full",
        provider="openrouter",
        model_id="anthropic/claude-sonnet",
        sysprompt_version="v0.9-full",
        sysprompt_sha="abc123",
    )


@pytest.fixture
def train_config():
    """A training config."""
    return TrainConfig(
        base_model="qwen3-32b",
        data="training/data/aria.jsonl",
        name="E037",
        epochs=3,
        batch_size=4,
        learning_rate=1e-5,
        renderer="qwen3_thinking",
        shuffle_seed=42,
        save_every=100,
    )


@pytest.fixture
def trained_model(train_config):
    """A trained model spec."""
    return TrainedModelSpec(
        alias="e037-final",
        provider="tinker",
        base_model="qwen3-32b",
        renderer="qwen3_thinking",
        checkpoint="xlr8harder/aria-e037/checkpoint-150",
        training_run="E037",
        training_data="aria-20260101",
        training_config=train_config,
    )


@pytest.fixture
def prompted_trained_model(train_config):
    """A prompted trained model spec."""
    return PromptedTrainedModelSpec(
        alias="e037-final-prompted",
        provider="tinker",
        base_model="qwen3-32b",
        renderer="qwen3_thinking",
        checkpoint="xlr8harder/aria-e037/checkpoint-150",
        training_run="E037",
        training_data="aria-20260101",
        training_config=train_config,
        sysprompt_version="v0.9-full",
        sysprompt_sha="def456",
    )


@pytest.fixture
def eval_config():
    """A complete eval config."""
    return EvalConfig(
        name="knowledge-v1",
        dataset_sha="abc123",
        judge_prompt_sha="def456",
        dataset_size=100,
        n_samples=100,
        complete=True,
    )


@pytest.fixture
def partial_eval_config():
    """A partial eval config."""
    return EvalConfig(
        name="knowledge-v1",
        dataset_sha="abc123",
        judge_prompt_sha="def456",
        dataset_size=100,
        n_samples=50,
        complete=False,
    )


@pytest.fixture
def sampling_config():
    """A sampling config."""
    return SamplingConfig(
        temperature=0.7,
        max_tokens=2048,
        runs_per_sample=9,
        judges_per_run=1,
    )


@pytest.fixture
def results():
    """Eval results."""
    return Results(
        aggregation="majority_vote",
        score=0.72,
        std=0.015,
        run_scores=[0.71, 0.73, 0.70, 0.72, 0.74],
        errors=ErrorBreakdown(total=5, by_type={"parse": 3, "timeout": 2}),
    )


@pytest.fixture
def full_eval_result(trained_model, base_model, eval_config, sampling_config, results):
    """A complete eval result."""
    return EvalResult(
        id="batch-20260105-143022-a7b3",
        timestamp=datetime(2026, 1, 5, 14, 30, 22),
        model=trained_model,
        judge=base_model,
        eval=eval_config,
        model_sampling=sampling_config,
        judge_sampling=SamplingConfig(temperature=0.0, max_tokens=1024),
        results=results,
        artifacts=Artifacts(results_dir="training/logs/E037/knowledge-eval/"),
        note="Testing normalize_weights",
    )


@pytest.fixture
def temp_store(tmp_path):
    """A temporary results store."""
    return ResultsStore(tmp_path / "evals.jsonl")


# =============================================================================
# Model Spec Tests
# =============================================================================


class TestModelSpecs:
    """Tests for model specification types."""

    def test_base_model_has_correct_mode(self, base_model):
        """Base model has mode='base'."""
        assert base_model.mode == "base"

    def test_prompted_model_has_correct_mode(self, prompted_model):
        """Prompted model has mode='prompted'."""
        assert prompted_model.mode == "prompted"

    def test_trained_model_has_correct_mode(self, trained_model):
        """Trained model has mode='trained'."""
        assert trained_model.mode == "trained"

    def test_prompted_trained_model_has_correct_mode(self, prompted_trained_model):
        """Prompted trained model has mode='prompted_trained'."""
        assert prompted_trained_model.mode == "prompted_trained"

    def test_base_model_serialization(self, base_model):
        """Base model serializes correctly."""
        data = base_model.model_dump()
        assert data["mode"] == "base"
        assert data["provider"] == "anthropic"
        assert "sysprompt_version" not in data

    def test_trained_model_has_training_config(self, trained_model):
        """Trained model includes training config."""
        assert trained_model.training_config.learning_rate == 1e-5
        assert trained_model.checkpoint == "xlr8harder/aria-e037/checkpoint-150"

    def test_prompted_trained_has_both(self, prompted_trained_model):
        """Prompted trained model has both training and sysprompt info."""
        assert prompted_trained_model.training_run == "E037"
        assert prompted_trained_model.sysprompt_version == "v0.9-full"


class TestModelSpecDiscriminator:
    """Tests for discriminated union parsing."""

    def test_parse_base_model(self):
        """Parses base model from dict."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(ModelSpec)

        data = {
            "alias": "test",
            "mode": "base",
            "provider": "anthropic",
            "model_id": "claude-3",
        }
        result = adapter.validate_python(data)
        assert isinstance(result, BaseModelSpec)

    def test_parse_trained_model(self):
        """Parses trained model from dict."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(ModelSpec)

        data = {
            "alias": "test",
            "mode": "trained",
            "provider": "tinker",
            "base_model": "qwen3",
            "renderer": "qwen3_thinking",
            "checkpoint": "xlr8harder/test",
            "training_run": "E001",
            "training_data": "test-data",
            "training_config": {
                "base_model": "qwen3",
                "data": "test.jsonl",
                "name": "E001",
                "renderer": "qwen3",
                "learning_rate": 1e-5,
                "shuffle_seed": 42,
                "save_every": 100,
            },
        }
        result = adapter.validate_python(data)
        assert isinstance(result, TrainedModelSpec)

    def test_invalid_mode_raises(self):
        """Invalid mode raises validation error."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(ModelSpec)

        data = {
            "alias": "test",
            "mode": "invalid",
            "provider": "anthropic",
        }
        with pytest.raises(ValidationError):
            adapter.validate_python(data)


# =============================================================================
# Eval Result Tests
# =============================================================================


class TestEvalResult:
    """Tests for EvalResult model."""

    def test_serialization_roundtrip(self, full_eval_result):
        """EvalResult survives JSON roundtrip."""
        json_str = full_eval_result.model_dump_json()
        data = json.loads(json_str)
        restored = EvalResult.model_validate(data)

        assert restored.id == full_eval_result.id
        assert restored.model.alias == "e037-final"
        assert restored.model.mode == "trained"
        assert restored.results.score == 0.72

    def test_model_discriminator_in_result(self, full_eval_result):
        """Model field uses discriminated union correctly."""
        # Serialize and deserialize
        data = json.loads(full_eval_result.model_dump_json())
        restored = EvalResult.model_validate(data)

        # Model should be TrainedModelSpec
        assert isinstance(restored.model, TrainedModelSpec)
        assert restored.model.training_run == "E037"

    def test_judge_can_be_none(self, trained_model, eval_config, sampling_config):
        """Judge field can be None for non-judged evals."""
        result = EvalResult(
            id="test-123",
            timestamp=datetime.now(),
            model=trained_model,
            judge=None,
            eval=eval_config,
            model_sampling=sampling_config,
            results=Results(score=0.85),
        )
        assert result.judge is None

    def test_version_defaults_to_1(self, full_eval_result):
        """Version field defaults to 1."""
        assert full_eval_result.version == 1

    def test_partial_eval_flag(
        self, trained_model, partial_eval_config, sampling_config
    ):
        """Partial evals are flagged correctly."""
        result = EvalResult(
            id="test-partial",
            timestamp=datetime.now(),
            model=trained_model,
            eval=partial_eval_config,
            model_sampling=sampling_config,
            results=Results(score=0.70),
        )
        assert result.eval.complete is False
        assert result.eval.n_samples < result.eval.dataset_size


class TestResults:
    """Tests for Results model."""

    def test_defaults(self):
        """Results has sensible defaults."""
        r = Results(score=0.5)
        assert r.aggregation == "mean"
        assert r.std is None
        assert r.run_scores == []
        assert r.errors.total == 0

    def test_with_run_scores(self):
        """Results stores run scores."""
        r = Results(
            score=0.72,
            std=0.015,
            run_scores=[0.71, 0.73, 0.70, 0.72, 0.74],
        )
        assert len(r.run_scores) == 5
        assert r.std == 0.015

    def test_error_breakdown(self):
        """Error breakdown is captured."""
        r = Results(
            score=0.70,
            errors=ErrorBreakdown(total=5, by_type={"parse": 3, "timeout": 2}),
        )
        assert r.errors.total == 5
        assert r.errors.by_type["parse"] == 3


# =============================================================================
# Results Store Tests
# =============================================================================


class TestResultsStore:
    """Tests for ResultsStore."""

    def test_add_and_get(self, temp_store, full_eval_result):
        """Can add and retrieve a result."""
        temp_store.add(full_eval_result)
        retrieved = temp_store.get(full_eval_result.id)

        assert retrieved is not None
        assert retrieved.id == full_eval_result.id
        assert retrieved.model.alias == "e037-final"

    def test_list_all(self, temp_store, full_eval_result):
        """Lists all results."""
        temp_store.add(full_eval_result)

        results = temp_store.list()
        assert len(results) == 1
        assert results[0].id == full_eval_result.id

    def test_list_filters_by_model(
        self, temp_store, full_eval_result, base_model, eval_config, sampling_config
    ):
        """Lists can filter by model alias."""
        # Add the trained model result
        temp_store.add(full_eval_result)

        # Add a base model result
        base_result = EvalResult(
            id="base-result",
            timestamp=datetime.now(),
            model=base_model,
            eval=eval_config,
            model_sampling=sampling_config,
            results=Results(score=0.80),
        )
        temp_store.add(base_result)

        # Filter by model
        trained_results = temp_store.list(model="e037-final")
        assert len(trained_results) == 1
        assert trained_results[0].id == full_eval_result.id

        base_results = temp_store.list(model="claude-haiku")
        assert len(base_results) == 1
        assert base_results[0].id == "base-result"

    def test_list_filters_by_eval_name(
        self, temp_store, full_eval_result, trained_model, sampling_config
    ):
        """Lists can filter by eval name."""
        temp_store.add(full_eval_result)

        # Add a different eval
        other_result = EvalResult(
            id="other-result",
            timestamp=datetime.now(),
            model=trained_model,
            eval=EvalConfig(
                name="wildchat-v1",
                dataset_sha="xyz",
                dataset_size=50,
                n_samples=50,
                complete=True,
            ),
            model_sampling=sampling_config,
            results=Results(score=0.65),
        )
        temp_store.add(other_result)

        knowledge = temp_store.list(eval_name="knowledge-v1")
        assert len(knowledge) == 1
        assert knowledge[0].id == full_eval_result.id

        wildchat = temp_store.list(eval_name="wildchat-v1")
        assert len(wildchat) == 1
        assert wildchat[0].id == "other-result"

    def test_list_excludes_partial_by_default(
        self, temp_store, trained_model, sampling_config
    ):
        """Partial evals are excluded by default."""
        # Add complete result
        complete = EvalResult(
            id="complete",
            timestamp=datetime.now(),
            model=trained_model,
            eval=EvalConfig(
                name="test",
                dataset_sha="abc",
                dataset_size=100,
                n_samples=100,
                complete=True,
            ),
            model_sampling=sampling_config,
            results=Results(score=0.80),
        )
        temp_store.add(complete)

        # Add partial result
        partial = EvalResult(
            id="partial",
            timestamp=datetime.now(),
            model=trained_model,
            eval=EvalConfig(
                name="test",
                dataset_sha="abc",
                dataset_size=100,
                n_samples=50,
                complete=False,
            ),
            model_sampling=sampling_config,
            results=Results(score=0.75),
        )
        temp_store.add(partial)

        # Default excludes partial
        results = temp_store.list()
        assert len(results) == 1
        assert results[0].id == "complete"

        # Explicit include_all
        all_results = temp_store.list(include_all=True)
        assert len(all_results) == 2

    def test_list_filters_by_training_run(self, temp_store, sampling_config):
        """Lists can filter by training run."""
        # E037 result
        e037 = EvalResult(
            id="e037-result",
            timestamp=datetime.now(),
            model=TrainedModelSpec(
                alias="e037-final",
                provider="tinker",
                base_model="qwen3",
                renderer="qwen3_thinking",
                checkpoint="xlr8harder/e037",
                training_run="E037",
                training_data="data1",
                training_config=TrainConfig(
                    base_model="qwen3",
                    data="data1.jsonl",
                    name="E037",
                    renderer="qwen3",
                    learning_rate=1e-5,
                    shuffle_seed=42,
                    save_every=100,
                ),
            ),
            eval=EvalConfig(
                name="test",
                dataset_sha="abc",
                dataset_size=100,
                n_samples=100,
                complete=True,
            ),
            model_sampling=sampling_config,
            results=Results(score=0.72),
        )
        temp_store.add(e037)

        # E038 result
        e038 = EvalResult(
            id="e038-result",
            timestamp=datetime.now(),
            model=TrainedModelSpec(
                alias="e038-final",
                provider="tinker",
                base_model="qwen3",
                renderer="qwen3_thinking",
                checkpoint="xlr8harder/e038",
                training_run="E038",
                training_data="data2",
                training_config=TrainConfig(
                    base_model="qwen3",
                    data="data2.jsonl",
                    name="E038",
                    renderer="qwen3",
                    learning_rate=1e-5,
                    shuffle_seed=42,
                    save_every=100,
                ),
            ),
            eval=EvalConfig(
                name="test",
                dataset_sha="abc",
                dataset_size=100,
                n_samples=100,
                complete=True,
            ),
            model_sampling=sampling_config,
            results=Results(score=0.75),
        )
        temp_store.add(e038)

        e037_results = temp_store.list(training_run="E037")
        assert len(e037_results) == 1
        assert e037_results[0].model.training_run == "E037"

    def test_get_nonexistent_returns_none(self, temp_store):
        """Getting nonexistent ID returns None."""
        result = temp_store.get("does-not-exist")
        assert result is None

    def test_count(self, temp_store, full_eval_result):
        """Count returns number of results."""
        assert temp_store.count() == 0

        temp_store.add(full_eval_result)
        assert temp_store.count() == 1

    def test_empty_store(self, temp_store):
        """Empty store returns empty list."""
        results = temp_store.list()
        assert results == []

    def test_creates_parent_dirs(self, tmp_path):
        """Store creates parent directories if needed."""
        store = ResultsStore(tmp_path / "deep" / "nested" / "evals.jsonl")
        result = EvalResult(
            id="test",
            timestamp=datetime.now(),
            model=BaseModelSpec(alias="test", provider="test", model_id="test"),
            eval=EvalConfig(
                name="test",
                dataset_sha="abc",
                dataset_size=10,
                n_samples=10,
                complete=True,
            ),
            model_sampling=SamplingConfig(temperature=0.0, max_tokens=100),
            results=Results(score=1.0),
        )
        store.add(result)

        assert store.path.exists()
        assert store.get("test") is not None


class TestResultsStoreLenientRead:
    """Tests for lenient deserialization."""

    def test_handles_invalid_json_lines(self, tmp_path):
        """Invalid JSON lines are skipped."""
        store_path = tmp_path / "evals.jsonl"
        store_path.write_text('{"valid": true}\nnot json\n{"also": "valid"}\n')

        store = ResultsStore(store_path)
        # Should not crash
        results = list(store.iter_all(include_raw=True))
        # Should get 2 results (skipping invalid line)
        assert len(results) == 2

    def test_handles_unknown_version(self, tmp_path):
        """Unknown versions return raw dict."""
        store_path = tmp_path / "evals.jsonl"
        # Write a record with unknown version/structure
        record = {
            "version": 999,
            "id": "future-record",
            "new_field": "unknown",
        }
        store_path.write_text(json.dumps(record) + "\n")

        store = ResultsStore(store_path)
        results = list(store.iter_all(include_raw=True))

        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert results[0]["version"] == 999


# =============================================================================
# Compare Helper Tests
# =============================================================================


class TestShortId:
    """Tests for short ID functions."""

    def test_short_id_extraction(self):
        """Extracts MMDD-XXXX from full ID."""
        from shaping.cli.results_cmd import _short_id

        assert _short_id("batch-20260105-143022-a7b3") == "0105-a7b3"
        assert _short_id("batch-20261231-235959-ffff") == "1231-ffff"
        assert _short_id("batch-20260101-000000-0000") == "0101-0000"

    def test_short_id_fallback(self):
        """Falls back to truncation for non-standard IDs."""
        from shaping.cli.results_cmd import _short_id

        # Non-standard format falls back to first 12 chars
        assert _short_id("custom-id-format") == "custom-id-fo"

    def test_is_short_id_valid(self):
        """Recognizes valid short IDs."""
        from shaping.cli.results_cmd import _is_short_id

        assert _is_short_id("0105-a7b3") is True
        assert _is_short_id("1231-ffff") is True
        assert _is_short_id("0101-0000") is True

    def test_is_short_id_invalid(self):
        """Rejects invalid short IDs."""
        from shaping.cli.results_cmd import _is_short_id

        assert _is_short_id("e037-final") is False  # Model alias
        assert _is_short_id("0105-ghij") is False  # Non-hex suffix
        assert _is_short_id("01051234") is False  # Missing dash
        assert _is_short_id("0105-a7b") is False  # Too short
        assert _is_short_id("0105-a7b3x") is False  # Too long

    def test_resolve_short_id(self, temp_store, full_eval_result):
        """Resolves short ID to full result."""
        from shaping.cli.results_cmd import _resolve_short_id, _short_id

        temp_store.add(full_eval_result)
        short = _short_id(full_eval_result.id)

        matches = _resolve_short_id(temp_store, short)
        assert len(matches) == 1
        assert matches[0].id == full_eval_result.id

    def test_resolve_short_id_not_found(self, temp_store):
        """Returns empty list for unknown short ID."""
        from shaping.cli.results_cmd import _resolve_short_id

        matches = _resolve_short_id(temp_store, "0105-xxxx")
        assert matches == []

    def test_resolve_full_id(self, temp_store, full_eval_result):
        """Also accepts full IDs."""
        from shaping.cli.results_cmd import _resolve_short_id

        temp_store.add(full_eval_result)

        matches = _resolve_short_id(temp_store, full_eval_result.id)
        assert len(matches) == 1
        assert matches[0].id == full_eval_result.id

    def test_resolve_ambiguous_short_id(
        self, temp_store, trained_model, eval_config, sampling_config
    ):
        """Multiple results with same short ID returns all matches."""
        from shaping.cli.results_cmd import _resolve_short_id, _short_id

        # Create two results with same short ID (same MMDD and suffix)
        # This can happen if two evals run on same day with suffix collision
        result1 = EvalResult(
            id="batch-20260105-143022-a7b3",
            timestamp=datetime(2026, 1, 5, 14, 30, 22),
            model=trained_model,
            eval=eval_config,
            model_sampling=sampling_config,
            results=Results(score=0.72),
        )
        result2 = EvalResult(
            id="batch-20260105-153022-a7b3",  # Same day, same suffix, different time
            timestamp=datetime(2026, 1, 5, 15, 30, 22),
            model=trained_model,
            eval=EvalConfig(
                name="wildchat-v1",
                dataset_sha="xyz",
                dataset_size=50,
                n_samples=50,
                complete=True,
            ),
            model_sampling=sampling_config,
            results=Results(score=0.65),
        )

        temp_store.add(result1)
        temp_store.add(result2)

        # Both have short ID 0105-a7b3
        assert _short_id(result1.id) == "0105-a7b3"
        assert _short_id(result2.id) == "0105-a7b3"

        # Resolution should return both
        matches = _resolve_short_id(temp_store, "0105-a7b3")
        assert len(matches) == 2
        assert {m.id for m in matches} == {result1.id, result2.id}


class TestEvalConfigKey:
    """Tests for eval configuration grouping."""

    def test_eval_config_key_basic(self, full_eval_result):
        """Generates grouping key from result."""
        from shaping.cli.results_cmd import _eval_config_key

        key = _eval_config_key(full_eval_result)
        assert key[0] == "knowledge-v1"  # eval name
        assert key[3] == 0.7  # temperature
        assert key[4] == 100  # n_samples

    def test_eval_config_key_includes_judge(self, full_eval_result):
        """Key includes judge info when present."""
        from shaping.cli.results_cmd import _eval_config_key

        key = _eval_config_key(full_eval_result)
        # full_eval_result has a judge
        assert key[6] == "claude-haiku"  # judge alias

    def test_format_eval_header(self):
        """Formats key as readable header."""
        from shaping.cli.results_cmd import _format_eval_header

        key = ("gpqa-diamond", "sha1", None, 0.7, 198, 1, None, None, None, "mean")
        header = _format_eval_header(key)

        assert "gpqa-diamond" in header
        assert "temp=0.7" in header
        assert "n=198" in header
        assert "agg=mean" in header
        # No judge fields when judge is None
        assert "judge=" not in header

    def test_format_eval_header_with_judge(self):
        """Header includes judge info when present."""
        from shaping.cli.results_cmd import _format_eval_header

        key = ("knowledge-v1", "sha1", "sha2", 0.7, 100, 9, "haiku", 0.0, 3, "majority")
        header = _format_eval_header(key)

        assert "judge=haiku" in header
        assert "j_temp=0.0" in header
        assert "j_runs=3" in header


class TestTrainingDiffs:
    """Tests for training config diff detection."""

    def test_get_training_diffs_same(self, trained_model):
        """No diffs for identical configs."""
        from shaping.cli.results_cmd import _get_training_diffs

        diffs = _get_training_diffs(trained_model, trained_model)
        assert diffs == {}

    def test_get_training_diffs_different(self, train_config):
        """Detects differing fields."""
        from shaping.cli.results_cmd import _get_training_diffs

        model1 = TrainedModelSpec(
            alias="e037",
            provider="tinker",
            base_model="qwen3",
            renderer="qwen3",
            checkpoint="ckpt1",
            training_run="E037",
            training_data="data1",
            training_config=TrainConfig(
                base_model="qwen3",
                data="data1.jsonl",
                name="E037",
                renderer="qwen3",
                learning_rate=1e-5,
                shuffle_seed=42,
                save_every=100,
                grad_clip=0.5,
            ),
        )
        model2 = TrainedModelSpec(
            alias="e038",
            provider="tinker",
            base_model="qwen3",
            renderer="qwen3",
            checkpoint="ckpt2",
            training_run="E038",
            training_data="data2",
            training_config=TrainConfig(
                base_model="qwen3",
                data="data2.jsonl",
                name="E038",
                renderer="qwen3",
                learning_rate=2e-5,
                shuffle_seed=42,
                save_every=100,
                grad_clip=1.0,
            ),
        )

        diffs = _get_training_diffs(model1, model2)
        assert "learning_rate" in diffs
        assert diffs["learning_rate"] == (1e-5, 2e-5)
        assert "grad_clip" in diffs
        assert diffs["grad_clip"] == (0.5, 1.0)

    def test_get_training_diffs_base_models(self, base_model):
        """Returns empty dict for non-trained models."""
        from shaping.cli.results_cmd import _get_training_diffs

        diffs = _get_training_diffs(base_model, base_model)
        assert diffs == {}
