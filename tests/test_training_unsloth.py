"""Tests for the Unsloth training backend integration."""

import builtins
import json
import sys
import types
from pathlib import Path

import pytest

from shaping.prompts import discover_checkpoints
from shaping.training import TrainConfig, run_training
from shaping.training.unsloth import _load_unsloth_dependencies, validate_unsloth_options


def _write_chat_rows(path: Path, count: int = 3) -> None:
    rows = []
    for index in range(count):
        rows.append(
            {
                "id": f"sample-{index}",
                "messages": [
                    {"role": "user", "content": f"Question {index}?"},
                    {"role": "assistant", "content": f"Answer {index}."},
                ],
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _install_fake_unsloth_stack(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    fake_torch = types.SimpleNamespace(
        cuda=FakeCuda(),
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
    )

    class FakeDataset(list):
        @classmethod
        def from_list(cls, rows):
            return cls(rows)

    class FakeTokenizer:
        chat_template = None

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=False
        ):
            assert tokenize is False
            return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer_config.json").write_text("{}")

    class FakeModel:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_config.json").write_text("{}")

        def save_pretrained_merged(self, path, tokenizer, save_method):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "config.json").write_text(
                json.dumps({"save_method": save_method})
            )

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            return FakeModel(), FakeTokenizer()

        @staticmethod
        def get_peft_model(model, **kwargs):
            model.peft_kwargs = kwargs
            return model

    class FakeSFTConfig:
        def __init__(
            self,
            output_dir,
            per_device_train_batch_size,
            gradient_accumulation_steps,
            num_train_epochs,
            learning_rate,
            lr_scheduler_type,
            warmup_steps,
            logging_steps,
            save_steps,
            seed,
            report_to,
            optim,
            weight_decay,
            fp16,
            bf16,
            dataset_text_field=None,
            packing=False,
            max_length=None,
            save_strategy=None,
            eval_strategy=None,
            eval_steps=None,
        ):
            self.__dict__.update(locals())
            self.__dict__.pop("self", None)

    class FakeTrainResult:
        metrics = {"train_runtime": 1.0}

    class FakeSFTTrainer:
        def __init__(
            self,
            model,
            args,
            train_dataset,
            eval_dataset=None,
            processing_class=None,
        ):
            self.model = model
            self.args = args
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.processing_class = processing_class
            self.state = types.SimpleNamespace(log_history=[], global_step=0)

        def train(self):
            self.state.global_step = 1
            self.state.log_history.append(
                {"step": 1, "loss": 2.5, "grad_norm": 0.5}
            )
            return FakeTrainResult()

        def evaluate(self):
            return {"eval_loss": 2.0}

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(Dataset=FakeDataset))
    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(TrainingArguments=FakeSFTConfig)
    )
    monkeypatch.setitem(
        sys.modules,
        "trl",
        types.SimpleNamespace(SFTConfig=FakeSFTConfig, SFTTrainer=FakeSFTTrainer),
    )
    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        types.SimpleNamespace(
            FastLanguageModel=FakeFastLanguageModel,
            is_bfloat16_supported=lambda: True,
        ),
    )


def test_unsloth_options_reject_unknown_top_level_key():
    """Unknown Unsloth backend_options keys fail before training starts."""
    with pytest.raises(ValueError, match="quantization"):
        validate_unsloth_options({"quantization": "qlora_4bit"})


def test_unsloth_options_reject_incomplete_registry_entry():
    """Registry metadata must name the served local model."""
    with pytest.raises(ValueError, match="registry.model"):
        validate_unsloth_options({"registry": {"provider": "local"}})


def test_unsloth_missing_dependency_message(monkeypatch):
    """Missing optional backend dependencies produce an actionable install hint."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "unsloth":
            raise ImportError("missing unsloth")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="uv sync --extra unsloth"):
        _load_unsloth_dependencies()


def test_unsloth_training_saves_adapter_metrics_and_artifacts(tmp_path, monkeypatch):
    """The Unsloth backend follows the standard ISF experiment layout."""
    _install_fake_unsloth_stack(monkeypatch)

    data_file = tmp_path / "train.jsonl"
    _write_chat_rows(data_file)
    config = TrainConfig(
        backend="unsloth",
        base_model="test/model",
        data=str(data_file),
        name="e001",
        renderer="auto",
        learning_rate=2e-4,
        shuffle_seed=123,
        save_every=1,
        epochs=1,
        batch_size=1,
        lora_rank=4,
        max_length=128,
        test_size=1,
        log_dir=str(tmp_path / "logs"),
        backend_options={
            "save": {"merged_dir": "merged"},
            "registry": {"model": "served-test-model"},
        },
    )

    log_path = run_training(config, force=False)

    assert (log_path / "train-config.json").exists()
    assert (log_path / "adapter" / "adapter_config.json").exists()
    assert (log_path / "adapter" / "tokenizer_config.json").exists()
    assert (log_path / "merged" / "config.json").exists()

    metrics = [
        json.loads(line)
        for line in (log_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert metrics[0]["train_mean_nll"] == 2.5
    assert metrics[1]["val_mean_nll"] == 2.0

    checkpoint = json.loads((log_path / "checkpoints.jsonl").read_text())
    assert checkpoint["backend"] == "unsloth"
    assert checkpoint["adapter_path"].endswith("/adapter")

    artifacts = json.loads((log_path / "artifacts.json").read_text())
    assert artifacts["registry"]["model"] == "served-test-model"


def test_unsloth_training_rejects_bad_chat_rows(tmp_path, monkeypatch):
    """Local training data validation points at the broken row."""
    _install_fake_unsloth_stack(monkeypatch)

    data_file = tmp_path / "bad.jsonl"
    data_file.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}))
    config = TrainConfig(
        backend="unsloth",
        base_model="test/model",
        data=str(data_file),
        name="e001",
        renderer="auto",
        learning_rate=2e-4,
        shuffle_seed=123,
        save_every=1,
        log_dir=str(tmp_path / "logs"),
    )

    with pytest.raises(ValueError, match="at least one assistant message"):
        run_training(config)


def test_unsloth_checkpoints_register_only_when_served_model_is_configured(tmp_path):
    """Unsloth artifacts are not registered unless a local served model is named."""
    logs_dir = tmp_path / "training" / "logs"
    no_registry = logs_dir / "e001"
    no_registry.mkdir(parents=True)
    (no_registry / "train-config.json").write_text(
        json.dumps(
            {
                "backend": "unsloth",
                "base_model": "test/model",
                "backend_options": {},
            }
        )
    )
    (no_registry / "checkpoints.jsonl").write_text(
        json.dumps({"name": "final", "adapter_path": "adapter"}) + "\n"
    )

    with_registry = logs_dir / "e002"
    with_registry.mkdir()
    (with_registry / "train-config.json").write_text(
        json.dumps(
            {
                "backend": "unsloth",
                "base_model": "test/model",
                "backend_options": {
                    "registry": {
                        "provider": "local",
                        "model": "served-unsloth-model",
                        "temperature": 0.4,
                    }
                },
            }
        )
    )
    (with_registry / "checkpoints.jsonl").write_text(
        json.dumps({"name": "final", "adapter_path": "adapter"}) + "\n"
    )

    checkpoints = discover_checkpoints(tmp_path)
    assert checkpoints == [
        {
            "name": "e002",
            "provider": "local",
            "model": "served-unsloth-model",
            "temperature": 0.4,
            "backend": "unsloth",
            "base_model": "test/model",
        }
    ]
