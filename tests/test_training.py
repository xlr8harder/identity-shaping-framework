"""Tests for shaping.training module."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from shaping.training import TrainConfig, load_config, get_next_experiment_name
from shaping.training.runner import _check_for_multiturn


class TestTrainConfig:
    """Tests for TrainConfig dataclass."""

    def test_minimal_config(self):
        """Config with only required fields."""
        config = TrainConfig(base_model="test/model", data="train.jsonl")
        assert config.base_model == "test/model"
        assert config.data == "train.jsonl"
        assert config.epochs == 1
        assert config.batch_size == 32

    def test_all_defaults(self):
        """Check all default values."""
        config = TrainConfig(base_model="test/model", data="train.jsonl")
        assert config.epochs == 1
        assert config.batch_size == 32
        assert config.lora_rank == 32
        assert config.learning_rate is None
        assert config.lr_schedule == "constant"
        assert config.max_length == 8192
        assert config.seed == 42
        assert config.shuffle_seed is None
        assert config.test_size == 0
        assert config.eval_every == 0
        assert config.save_every is None
        assert config.renderer is None
        assert config.normalize_weights is False
        assert config.grad_clip == 1e12  # Enables grad norm calculation
        assert config.optim_metrics_every == 1
        assert config.note is None
        assert config.log_dir == "training/logs"

    def test_validation_requires_base_model(self):
        """Config validation requires base_model."""
        with pytest.raises(ValueError, match="base_model is required"):
            TrainConfig(base_model="", data="train.jsonl")

    def test_validation_requires_data(self):
        """Config validation requires data."""
        with pytest.raises(ValueError, match="data is required"):
            TrainConfig(base_model="test/model", data="")

    def test_validation_epochs_positive(self):
        """Config validation requires epochs >= 1."""
        with pytest.raises(ValueError, match="epochs must be >= 1"):
            TrainConfig(base_model="test/model", data="train.jsonl", epochs=0)

    def test_validation_batch_size_positive(self):
        """Config validation requires batch_size >= 1."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            TrainConfig(base_model="test/model", data="train.jsonl", batch_size=0)

    def test_validation_lora_rank_positive(self):
        """Config validation requires lora_rank >= 1."""
        with pytest.raises(ValueError, match="lora_rank must be >= 1"):
            TrainConfig(base_model="test/model", data="train.jsonl", lora_rank=0)

    def test_validation_max_length_positive(self):
        """Config validation requires max_length >= 1."""
        with pytest.raises(ValueError, match="max_length must be >= 1"):
            TrainConfig(base_model="test/model", data="train.jsonl", max_length=0)

    def test_log_path_property(self):
        """log_path combines log_dir and name."""
        config = TrainConfig(
            base_model="test/model",
            data="train.jsonl",
            name="E001",
            log_dir="logs"
        )
        assert config.log_path == Path("logs/E001")

    def test_data_path_property(self):
        """data_path returns Path of data."""
        config = TrainConfig(base_model="test/model", data="path/to/train.jsonl")
        assert config.data_path == Path("path/to/train.jsonl")

    def test_to_dict(self):
        """to_dict returns all config values."""
        config = TrainConfig(
            base_model="test/model",
            data="train.jsonl",
            name="E001",
            epochs=3,
            note="Test experiment"
        )
        d = config.to_dict()
        assert d["base_model"] == "test/model"
        assert d["data"] == "train.jsonl"
        assert d["name"] == "E001"
        assert d["epochs"] == 3
        assert d["note"] == "Test experiment"
        # Check all keys present
        expected_keys = {
            "name", "base_model", "data", "epochs", "batch_size", "lora_rank",
            "learning_rate", "lr_schedule", "max_length", "seed", "shuffle_seed",
            "test_size", "eval_every", "save_every", "renderer", "grad_clip",
            "normalize_weights", "optim_metrics_every", "note", "log_dir"
        }
        assert set(d.keys()) == expected_keys


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_yaml_file(self, tmp_path):
        """Loads config from YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "base_model": "test/model",
            "data": "train.jsonl",
            "epochs": 2
        }))

        config = load_config(config_file)
        assert config.base_model == "test/model"
        assert config.data == "train.jsonl"
        assert config.epochs == 2

    def test_cli_overrides(self, tmp_path):
        """CLI overrides take precedence over file values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "base_model": "test/model",
            "data": "train.jsonl",
            "epochs": 2
        }))

        config = load_config(config_file, epochs=5, batch_size=64)
        assert config.epochs == 5
        assert config.batch_size == 64

    def test_none_overrides_ignored(self, tmp_path):
        """None values in overrides are ignored."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "base_model": "test/model",
            "data": "train.jsonl",
            "epochs": 2
        }))

        config = load_config(config_file, epochs=None)
        assert config.epochs == 2  # Not overwritten

    def test_hyphen_underscore_normalization(self, tmp_path):
        """YAML keys with hyphens are normalized to underscores."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "base-model": "test/model",
            "data": "train.jsonl",
            "lr-schedule": "cosine",
            "lora-rank": 16
        }))

        config = load_config(config_file)
        assert config.base_model == "test/model"
        assert config.lr_schedule == "cosine"
        assert config.lora_rank == 16

    def test_auto_generates_experiment_name(self, tmp_path):
        """Auto-generates experiment name if not provided."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "base_model": "test/model",
            "data": "train.jsonl"
        }))

        # Create empty log dir for get_next_experiment_name
        log_dir = tmp_path / "logs"
        config = load_config(config_file, log_dir=str(log_dir))
        assert config.name == "E001"

    def test_file_not_found(self):
        """Raises FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_invalid_yaml_type(self, tmp_path):
        """Raises ValueError for non-mapping YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("- just\n- a\n- list")

        with pytest.raises(ValueError, match="Config must be a YAML mapping"):
            load_config(config_file)

    def test_empty_yaml_file(self, tmp_path):
        """Raises ValueError for empty YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        with pytest.raises(ValueError, match="Config must be a YAML mapping"):
            load_config(config_file)


class TestGetNextExperimentName:
    """Tests for get_next_experiment_name function."""

    def test_empty_directory(self, tmp_path):
        """Returns E001 for empty directory."""
        assert get_next_experiment_name(tmp_path) == "E001"

    def test_nonexistent_directory(self, tmp_path):
        """Returns E001 for nonexistent directory."""
        assert get_next_experiment_name(tmp_path / "nonexistent") == "E001"

    def test_finds_next_number(self, tmp_path):
        """Finds next available number."""
        (tmp_path / "E001").mkdir()
        (tmp_path / "E002").mkdir()
        (tmp_path / "E003").mkdir()
        assert get_next_experiment_name(tmp_path) == "E004"

    def test_handles_gaps(self, tmp_path):
        """Uses max+1, not filling gaps."""
        (tmp_path / "E001").mkdir()
        (tmp_path / "E005").mkdir()
        (tmp_path / "E003").mkdir()
        assert get_next_experiment_name(tmp_path) == "E006"

    def test_ignores_non_experiment_dirs(self, tmp_path):
        """Ignores directories that don't match E### pattern."""
        (tmp_path / "E001").mkdir()
        (tmp_path / "backup").mkdir()
        (tmp_path / "E002-test").mkdir()  # Has suffix, not pure E###
        (tmp_path / "test-E003").mkdir()  # Wrong prefix
        assert get_next_experiment_name(tmp_path) == "E002"

    def test_pads_to_three_digits(self, tmp_path):
        """Pads experiment name to 3 digits."""
        (tmp_path / "E001").mkdir()
        assert get_next_experiment_name(tmp_path) == "E002"

        (tmp_path / "E099").mkdir()
        assert get_next_experiment_name(tmp_path) == "E100"


class TestCheckForMultiturn:
    """Tests for _check_for_multiturn function."""

    def test_single_turn_returns_false(self, tmp_path):
        """Single-turn conversations return False."""
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
        }) + "\n")

        assert _check_for_multiturn(data_file) is False

    def test_multi_turn_returns_true(self, tmp_path):
        """Multi-turn conversations return True."""
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well!"}
            ]
        }) + "\n")

        assert _check_for_multiturn(data_file) is True

    def test_mixed_data_returns_true(self, tmp_path):
        """Mixed single and multi-turn returns True."""
        data_file = tmp_path / "train.jsonl"
        lines = [
            json.dumps({"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ]}),
            json.dumps({"messages": [
                {"role": "user", "content": "One"},
                {"role": "assistant", "content": "Two"},
                {"role": "user", "content": "Three"},
                {"role": "assistant", "content": "Four"}
            ]})
        ]
        data_file.write_text("\n".join(lines) + "\n")

        assert _check_for_multiturn(data_file) is True

    def test_handles_invalid_json(self, tmp_path):
        """Skips invalid JSON lines."""
        data_file = tmp_path / "train.jsonl"
        lines = [
            "not valid json",
            json.dumps({"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ]})
        ]
        data_file.write_text("\n".join(lines) + "\n")

        assert _check_for_multiturn(data_file) is False

    def test_handles_missing_messages(self, tmp_path):
        """Handles rows without messages field."""
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(json.dumps({"other": "data"}) + "\n")

        assert _check_for_multiturn(data_file) is False
