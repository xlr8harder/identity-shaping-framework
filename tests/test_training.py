"""Tests for shaping.training module."""

import json
from pathlib import Path

import pytest
import yaml

from shaping.training import (
    TrainConfig,
    build_config,
    load_config,
    get_next_experiment_name,
)
from shaping.training.runner import _check_for_multiturn


# Common test values for required fields
REQUIRED_FIELDS = {
    "base_model": "test/model",
    "data": "train.jsonl",
    "name": "e001",
    "renderer": "qwen3",
    "learning_rate": 1e-5,
    "shuffle_seed": 42,
    "save_every": 100,
}


class TestTrainConfig:
    """Tests for TrainConfig dataclass."""

    def test_minimal_config(self):
        """Config with only required fields."""
        config = TrainConfig(**REQUIRED_FIELDS)
        assert config.base_model == "test/model"
        assert config.data == "train.jsonl"
        assert config.name == "e001"
        assert config.renderer == "qwen3"
        assert config.learning_rate == 1e-5
        assert config.shuffle_seed == 42
        assert config.save_every == 100
        assert config.epochs == 1
        assert config.batch_size == 32

    def test_all_defaults(self):
        """Check all default values."""
        config = TrainConfig(**REQUIRED_FIELDS)
        assert config.epochs == 1
        assert config.batch_size == 32
        assert config.lora_rank == 32
        assert config.lr_schedule == "constant"
        assert config.max_length == 8192
        assert config.seed == 42
        assert config.test_size == 0
        assert config.eval_every == 0
        assert config.normalize_weights is False
        assert config.grad_clip == 1e12  # Enables grad norm calculation
        assert config.optim_metrics_every == 1
        assert config.note is None
        assert config.log_dir == "training/logs"

    def test_validation_requires_base_model(self):
        """Config validation requires base_model."""
        fields = {**REQUIRED_FIELDS, "base_model": ""}
        with pytest.raises(ValueError, match="base_model is required"):
            TrainConfig(**fields)

    def test_validation_requires_data(self):
        """Config validation requires data."""
        fields = {**REQUIRED_FIELDS, "data": ""}
        with pytest.raises(ValueError, match="data is required"):
            TrainConfig(**fields)

    def test_validation_epochs_positive(self):
        """Config validation requires epochs >= 1."""
        fields = {**REQUIRED_FIELDS, "epochs": 0}
        with pytest.raises(ValueError, match="epochs must be >= 1"):
            TrainConfig(**fields)

    def test_validation_batch_size_positive(self):
        """Config validation requires batch_size >= 1."""
        fields = {**REQUIRED_FIELDS, "batch_size": 0}
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            TrainConfig(**fields)

    def test_validation_lora_rank_positive(self):
        """Config validation requires lora_rank >= 1."""
        fields = {**REQUIRED_FIELDS, "lora_rank": 0}
        with pytest.raises(ValueError, match="lora_rank must be >= 1"):
            TrainConfig(**fields)

    def test_validation_max_length_positive(self):
        """Config validation requires max_length >= 1."""
        fields = {**REQUIRED_FIELDS, "max_length": 0}
        with pytest.raises(ValueError, match="max_length must be >= 1"):
            TrainConfig(**fields)

    def test_log_path_property(self):
        """log_path combines log_dir and name."""
        config = TrainConfig(**REQUIRED_FIELDS, log_dir="logs")
        assert config.log_path == Path("logs/e001")

    def test_data_path_property(self):
        """data_path returns Path of data."""
        fields = {**REQUIRED_FIELDS, "data": "path/to/train.jsonl"}
        config = TrainConfig(**fields)
        assert config.data_path == Path("path/to/train.jsonl")

    def test_to_dict(self):
        """to_dict returns all config values."""
        config = TrainConfig(**REQUIRED_FIELDS, epochs=3, note="Test experiment")
        d = config.to_dict()
        assert d["base_model"] == "test/model"
        assert d["data"] == "train.jsonl"
        assert d["name"] == "e001"
        assert d["epochs"] == 3
        assert d["note"] == "Test experiment"
        # Check all keys present
        expected_keys = {
            "name",
            "base_model",
            "data",
            "epochs",
            "batch_size",
            "lora_rank",
            "learning_rate",
            "lr_schedule",
            "max_length",
            "seed",
            "shuffle_seed",
            "test_size",
            "eval_every",
            "save_every",
            "renderer",
            "grad_clip",
            "normalize_weights",
            "optim_metrics_every",
            "note",
            "log_dir",
        }
        assert set(d.keys()) == expected_keys


class TestBuildConfig:
    """Tests for build_config function.

    Note: build_config requires resolution of some fields (renderer, learning_rate, etc.)
    which normally needs tinker_cookbook. To test without that dependency, we provide
    these values explicitly in overrides, which skips auto-resolution.
    """

    @pytest.fixture
    def data_file(self, tmp_path):
        """Create a minimal training data file."""
        data = tmp_path / "train.jsonl"
        # 100 rows of dummy data
        data.write_text("\n".join(['{"messages": []}'] * 100))
        return data

    # Values to skip auto-resolution (no tinker_cookbook needed)
    RESOLVED = {
        "renderer": "qwen3",
        "learning_rate": 1e-5,
        "shuffle_seed": 42,
        "save_every": 50,
    }

    def test_loads_yaml_file(self, tmp_path, data_file):
        """Loads config from YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"base_model": "test/model", "data": str(data_file), "epochs": 2})
        )

        config = build_config(config_file, **self.RESOLVED)
        assert config.base_model == "test/model"
        assert config.data == str(data_file)
        assert config.epochs == 2

    def test_cli_overrides(self, tmp_path, data_file):
        """CLI overrides take precedence over file values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"base_model": "test/model", "data": str(data_file), "epochs": 2})
        )

        config = build_config(config_file, epochs=5, batch_size=64, **self.RESOLVED)
        assert config.epochs == 5
        assert config.batch_size == 64

    def test_none_overrides_ignored(self, tmp_path, data_file):
        """None values in overrides are ignored."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"base_model": "test/model", "data": str(data_file), "epochs": 2})
        )

        config = build_config(config_file, epochs=None, **self.RESOLVED)
        assert config.epochs == 2  # Not overwritten

    def test_hyphen_underscore_normalization(self, tmp_path, data_file):
        """YAML keys with hyphens are normalized to underscores."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "base-model": "test/model",
                    "data": str(data_file),
                    "lr-schedule": "cosine",
                    "lora-rank": 16,
                }
            )
        )

        config = build_config(config_file, **self.RESOLVED)
        assert config.base_model == "test/model"
        assert config.lr_schedule == "cosine"
        assert config.lora_rank == 16

    def test_auto_generates_experiment_name(self, tmp_path, data_file):
        """Auto-generates experiment name if not provided."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"base_model": "test/model", "data": str(data_file)})
        )

        # Create empty log dir for get_next_experiment_name
        log_dir = tmp_path / "logs"
        config = build_config(config_file, log_dir=str(log_dir), **self.RESOLVED)
        assert config.name == "e001"

    def test_auto_calculates_save_every(self, tmp_path, data_file):
        """Calculates save_every from data size if not provided."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "base_model": "test/model",
                    "data": str(data_file),
                    "batch_size": 10,  # 100 rows / 10 = 10 steps/epoch
                }
            )
        )

        # Provide renderer and learning_rate but not save_every
        config = build_config(
            config_file,
            renderer="qwen3",
            learning_rate=1e-5,
            shuffle_seed=42,
        )
        assert config.save_every == 10  # steps_per_epoch

    def test_shuffle_seed_defaults_to_seed(self, tmp_path, data_file):
        """shuffle_seed defaults to seed if not provided."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "base_model": "test/model",
                    "data": str(data_file),
                    "seed": 123,
                }
            )
        )

        config = build_config(
            config_file,
            renderer="qwen3",
            learning_rate=1e-5,
            save_every=50,
        )
        assert config.shuffle_seed == 123  # Same as seed

    def test_file_not_found(self):
        """Raises FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            build_config("/nonexistent/path/config.yaml")

    def test_invalid_yaml_type(self, tmp_path):
        """Raises ValueError for non-mapping YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("- just\n- a\n- list")

        with pytest.raises(ValueError, match="Config must be a YAML mapping"):
            build_config(config_file)

    def test_empty_yaml_file(self, tmp_path):
        """Raises ValueError for empty YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        with pytest.raises(ValueError, match="Config must be a YAML mapping"):
            build_config(config_file)

    def test_data_file_not_found(self, tmp_path):
        """Raises FileNotFoundError if data file doesn't exist."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "base_model": "test/model",
                    "data": "/nonexistent/train.jsonl",
                }
            )
        )

        # Provide renderer/learning_rate but NOT save_every (triggers data read)
        with pytest.raises(FileNotFoundError, match="Training data not found"):
            build_config(
                config_file,
                renderer="qwen3",
                learning_rate=1e-5,
                shuffle_seed=42,
            )

    def test_load_config_is_alias(self, tmp_path, data_file):
        """load_config is a backwards-compatible alias for build_config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "base_model": "test/model",
                    "data": str(data_file),
                }
            )
        )

        config = load_config(config_file, **self.RESOLVED)
        assert config.base_model == "test/model"

    def test_empty_data_file(self, tmp_path):
        """Raises ValueError for empty data file."""
        data_file = tmp_path / "empty.jsonl"
        data_file.write_text("")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "base_model": "test/model",
                    "data": str(data_file),
                }
            )
        )

        with pytest.raises(ValueError, match="Training data file is empty"):
            build_config(
                config_file,
                renderer="qwen3",
                learning_rate=1e-5,
                shuffle_seed=42,
            )

    def test_test_size_exceeds_rows(self, tmp_path, data_file):
        """Raises ValueError when test_size >= total rows."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "base_model": "test/model",
                    "data": str(data_file),
                    "test_size": 200,  # data_file has 100 rows
                }
            )
        )

        with pytest.raises(ValueError, match="test_size.*>= total rows"):
            build_config(
                config_file,
                renderer="qwen3",
                learning_rate=1e-5,
                shuffle_seed=42,
            )

    def test_batch_size_exceeds_train_rows(self, tmp_path, data_file):
        """Raises ValueError when batch_size > train_rows."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "base_model": "test/model",
                    "data": str(data_file),
                    "batch_size": 200,  # data_file has 100 rows
                }
            )
        )

        with pytest.raises(ValueError, match="train_rows.*< batch_size"):
            build_config(
                config_file,
                renderer="qwen3",
                learning_rate=1e-5,
                shuffle_seed=42,
            )


class TestGetNextExperimentName:
    """Tests for get_next_experiment_name function."""

    def test_empty_directory(self, tmp_path):
        """Returns e001 for empty directory."""
        assert get_next_experiment_name(tmp_path) == "e001"

    def test_nonexistent_directory(self, tmp_path):
        """Returns e001 for nonexistent directory."""
        assert get_next_experiment_name(tmp_path / "nonexistent") == "e001"

    def test_finds_next_number(self, tmp_path):
        """Finds next available number."""
        (tmp_path / "E001").mkdir()
        (tmp_path / "E002").mkdir()
        (tmp_path / "E003").mkdir()
        assert get_next_experiment_name(tmp_path) == "e004"

    def test_handles_gaps(self, tmp_path):
        """Uses max+1, not filling gaps."""
        (tmp_path / "E001").mkdir()
        (tmp_path / "E005").mkdir()
        (tmp_path / "E003").mkdir()
        assert get_next_experiment_name(tmp_path) == "e006"

    def test_ignores_non_experiment_dirs(self, tmp_path):
        """Ignores directories that don't match E### pattern."""
        (tmp_path / "E001").mkdir()
        (tmp_path / "backup").mkdir()
        (tmp_path / "E002-test").mkdir()  # Has suffix, not pure E###
        (tmp_path / "test-E003").mkdir()  # Wrong prefix
        assert get_next_experiment_name(tmp_path) == "e002"

    def test_pads_to_three_digits(self, tmp_path):
        """Pads experiment name to 3 digits."""
        (tmp_path / "E001").mkdir()
        assert get_next_experiment_name(tmp_path) == "e002"

        (tmp_path / "E099").mkdir()
        assert get_next_experiment_name(tmp_path) == "e100"


class TestCheckForMultiturn:
    """Tests for _check_for_multiturn function."""

    def test_single_turn_returns_false(self, tmp_path):
        """Single-turn conversations return False."""
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there"},
                    ]
                }
            )
            + "\n"
        )

        assert _check_for_multiturn(data_file) is False

    def test_multi_turn_returns_true(self, tmp_path):
        """Multi-turn conversations return True."""
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there"},
                        {"role": "user", "content": "How are you?"},
                        {"role": "assistant", "content": "I'm doing well!"},
                    ]
                }
            )
            + "\n"
        )

        assert _check_for_multiturn(data_file) is True

    def test_mixed_data_returns_true(self, tmp_path):
        """Mixed single and multi-turn returns True."""
        data_file = tmp_path / "train.jsonl"
        lines = [
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi"},
                    ]
                }
            ),
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "One"},
                        {"role": "assistant", "content": "Two"},
                        {"role": "user", "content": "Three"},
                        {"role": "assistant", "content": "Four"},
                    ]
                }
            ),
        ]
        data_file.write_text("\n".join(lines) + "\n")

        assert _check_for_multiturn(data_file) is True

    def test_handles_invalid_json(self, tmp_path):
        """Skips invalid JSON lines."""
        data_file = tmp_path / "train.jsonl"
        lines = [
            "not valid json",
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi"},
                    ]
                }
            ),
        ]
        data_file.write_text("\n".join(lines) + "\n")

        assert _check_for_multiturn(data_file) is False

    def test_handles_missing_messages(self, tmp_path):
        """Handles rows without messages field."""
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({"other": "data"}) + "\n")

        assert _check_for_multiturn(data_file) is False
