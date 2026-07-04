"""Unsloth local LoRA/QLoRA training backend."""

from __future__ import annotations

import inspect
import json
import random
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .config import TrainConfig


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class UnslothSaveOptions(BaseModel):
    """Artifact save options for Unsloth runs."""

    model_config = ConfigDict(extra="forbid")

    adapter_dir: str = "adapter"
    merged_dir: str | None = None
    merge_method: Literal["merged_16bit", "merged_4bit", "lora"] = "merged_16bit"

    @field_validator("adapter_dir", "merged_dir")
    @classmethod
    def relative_dir(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError("must not be empty")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("must be a relative path inside the experiment directory")
        return value


class UnslothRegistryOptions(BaseModel):
    """Optional registry entry for a separately served local model."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["local", "openai_compatible"] = "local"
    model: str
    temperature: float | None = None

    @field_validator("model")
    @classmethod
    def model_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value


class UnslothBackendOptions(BaseModel):
    """Typed backend_options for `backend: unsloth`.

    Common SFT controls stay on TrainConfig. Backend-specific knobs and
    full-library passthroughs live here so new users get a small surface while
    experienced users can still reach the underlying Unsloth/TRL configuration.
    """

    model_config = ConfigDict(extra="forbid")

    load_in_4bit: bool = True
    dtype: (
        Literal["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"]
        | None
    ) = None
    target_modules: list[str] = Field(
        default_factory=lambda: list(DEFAULT_TARGET_MODULES)
    )
    lora_alpha: int | None = None
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: bool | str = "unsloth"
    use_rslora: bool = False
    loftq_config: dict[str, Any] | None = None

    packing: bool = False
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    weight_decay: float = 0.0
    optim: str = "adamw_8bit"
    logging_steps: int = 1
    dataset_num_proc: int | None = None
    chat_template: str | None = None
    add_generation_prompt: bool = False

    save: UnslothSaveOptions = Field(default_factory=UnslothSaveOptions)
    registry: UnslothRegistryOptions | None = None

    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    peft_kwargs: dict[str, Any] = Field(default_factory=dict)
    sft_config_kwargs: dict[str, Any] = Field(default_factory=dict)
    trainer_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_modules")
    @classmethod
    def target_modules_not_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must include at least one module name")
        bad = [item for item in value if not isinstance(item, str) or not item.strip()]
        if bad:
            raise ValueError("must contain only non-empty strings")
        return value

    @field_validator("lora_alpha")
    @classmethod
    def lora_alpha_positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("lora_dropout")
    @classmethod
    def lora_dropout_range(cls, value: float) -> float:
        if value < 0 or value >= 1:
            raise ValueError("must be >= 0 and < 1")
        return value

    @field_validator("gradient_accumulation_steps", "logging_steps")
    @classmethod
    def positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("warmup_steps")
    @classmethod
    def non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be >= 0")
        return value

    @field_validator("weight_decay")
    @classmethod
    def non_negative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("must be >= 0")
        return value

    @field_validator("dataset_num_proc")
    @classmethod
    def dataset_num_proc_positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("must be >= 1")
        return value


def validate_unsloth_options(raw: dict[str, Any] | None) -> UnslothBackendOptions:
    """Validate Unsloth backend options with a compact user-facing error."""
    try:
        return UnslothBackendOptions.model_validate(raw or {})
    except ValidationError as exc:
        issues = []
        for error in exc.errors():
            loc = ".".join(str(part) for part in error["loc"])
            issues.append(f"{loc}: {error['msg']}")
        raise ValueError(
            "Invalid backend_options for backend 'unsloth': " + "; ".join(issues)
        ) from exc


def run_unsloth_training(
    config: TrainConfig, force: bool = False, verbose: bool = False
) -> Path:
    """Run a local Unsloth LoRA/QLoRA SFT experiment."""
    options = validate_unsloth_options(config.backend_options)
    deps = _load_unsloth_dependencies()
    torch = deps["torch"]

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Unsloth training requires a CUDA-visible GPU, but "
            "torch.cuda.is_available() is false. Install a CUDA PyTorch/Unsloth "
            "environment or choose a non-local backend such as 'tinker'."
        )

    if config.normalize_weights:
        raise ValueError(
            "backend 'unsloth' does not support normalize_weights. "
            "Remove normalize_weights or use backend 'tinker'."
        )

    if not config.data_path.exists():
        raise ValueError(f"Training data not found: {config.data_path}")

    log_path = config.log_path
    if log_path.exists():
        if not force:
            raise FileExistsError(
                f"Experiment directory already exists: {log_path}. "
                "Use --force to overwrite it or choose --name."
            )
        shutil.rmtree(log_path)
    log_path.mkdir(parents=True, exist_ok=True)

    rows = _load_chat_rows(config.data_path)
    train_rows, val_rows = _split_rows(rows, config.test_size, config.shuffle_seed)
    steps_per_epoch = max(1, len(train_rows) // config.batch_size)
    total_steps = steps_per_epoch * config.epochs

    print(f"Data: {len(rows)} rows ({len(train_rows)} train, {len(val_rows)} val)")
    print(f"Steps: {steps_per_epoch}/epoch, {total_steps} total")
    print(f"Backend: unsloth ({'4-bit QLoRA' if options.load_in_4bit else 'LoRA'})")

    config_save_path = log_path / "train-config.json"
    with open(config_save_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Saved config: {config_save_path}")

    _copy_dataset_manifest(config.data_path, log_path)

    model, tokenizer = deps["FastLanguageModel"].from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_length,
        dtype=_resolve_dtype(options.dtype, torch),
        load_in_4bit=options.load_in_4bit,
        **options.model_kwargs,
    )

    if options.chat_template is not None:
        tokenizer.chat_template = options.chat_template

    model = deps["FastLanguageModel"].get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=options.target_modules,
        lora_alpha=options.lora_alpha or config.lora_rank,
        lora_dropout=options.lora_dropout,
        bias=options.bias,
        use_gradient_checkpointing=options.use_gradient_checkpointing,
        random_state=config.seed,
        use_rslora=options.use_rslora,
        loftq_config=options.loftq_config,
        **options.peft_kwargs,
    )

    train_dataset = _build_text_dataset(
        rows=train_rows,
        tokenizer=tokenizer,
        dataset_cls=deps["Dataset"],
        add_generation_prompt=options.add_generation_prompt,
    )
    eval_dataset = (
        _build_text_dataset(
            rows=val_rows,
            tokenizer=tokenizer,
            dataset_cls=deps["Dataset"],
            add_generation_prompt=options.add_generation_prompt,
        )
        if val_rows
        else None
    )

    sft_args = _build_sft_args(
        deps=deps,
        config=config,
        options=options,
        log_path=log_path,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer = _build_sft_trainer(
        deps=deps,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_args,
        options=options,
    )

    if not verbose:
        print("Verbose trainer logs are captured by TRL/Transformers output files.")

    train_result = trainer.train()
    if eval_dataset is not None and config.eval_every == 0:
        try:
            eval_metrics = trainer.evaluate()
            trainer.state.log_history.append(
                {"step": trainer.state.global_step, **eval_metrics}
            )
        except Exception as exc:
            print(f"Warning: final evaluation failed: {exc}")

    _write_metrics(log_path / "metrics.jsonl", trainer, train_result)

    adapter_path = log_path / options.save.adapter_dir
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    merged_path = None
    if options.save.merged_dir:
        if not hasattr(model, "save_pretrained_merged"):
            raise RuntimeError(
                "This Unsloth model object does not expose save_pretrained_merged; "
                "remove backend_options.save.merged_dir to save only the adapter."
            )
        merged_path = log_path / options.save.merged_dir
        model.save_pretrained_merged(
            merged_path, tokenizer, save_method=options.save.merge_method
        )

    artifacts = {
        "backend": "unsloth",
        "base_model": config.base_model,
        "adapter_path": str(adapter_path),
        "merged_path": str(merged_path) if merged_path else None,
        "registry": options.registry.model_dump() if options.registry else None,
    }
    with open(log_path / "artifacts.json", "w") as f:
        json.dump(artifacts, f, indent=2)

    checkpoint = {
        "name": "final",
        "backend": "unsloth",
        "epoch": config.epochs,
        "step": getattr(trainer.state, "global_step", None),
        "adapter_path": str(adapter_path),
        "merged_path": str(merged_path) if merged_path else None,
        "base_model": config.base_model,
    }
    with open(log_path / "checkpoints.jsonl", "w") as f:
        f.write(json.dumps(checkpoint) + "\n")

    print(f"Adapter saved to: {adapter_path}")
    if merged_path:
        print(f"Merged model saved to: {merged_path}")
    print(f"Training complete. Artifacts saved to: {log_path}")
    return log_path


def _load_unsloth_dependencies() -> dict[str, Any]:
    """Import Unsloth runtime dependencies lazily."""
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        import torch
        from datasets import Dataset
        from transformers import TrainingArguments
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise ImportError(
            "Training backend 'unsloth' requires optional local training "
            "dependencies. Install them with:\n"
            "  uv sync --extra unsloth\n"
            "If your CUDA/PyTorch stack needs explicit wheel selection, use the "
            "Unsloth install command for your machine, then rerun `isf train run`."
        ) from exc

    return {
        "torch": torch,
        "Dataset": Dataset,
        "TrainingArguments": TrainingArguments,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
        "FastLanguageModel": FastLanguageModel,
        "is_bfloat16_supported": is_bfloat16_supported,
    }


def _load_chat_rows(data_path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(data_path) as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in training data {data_path} line {line_no}: "
                    f"{exc.msg}"
                ) from exc
            rows.append(_validate_chat_row(row, data_path, line_no))

    if not rows:
        raise ValueError(f"Training data file is empty: {data_path}")
    return rows


def _validate_chat_row(row: Any, data_path: Path, line_no: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(
            f"Invalid training row in {data_path} line {line_no}: expected object"
        )

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(
            f"Invalid training row in {data_path} line {line_no}: "
            "'messages' must be a non-empty list"
        )

    normalized = []
    assistant_count = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(
                f"Invalid message in {data_path} line {line_no} index {index}: "
                "expected object"
            )
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(
                f"Invalid message role in {data_path} line {line_no} index {index}: "
                f"{role!r}"
            )
        if not isinstance(content, str):
            raise ValueError(
                f"Invalid message content in {data_path} line {line_no} index "
                f"{index}: expected string"
            )
        if role == "assistant":
            assistant_count += 1
        normalized.append({"role": role, "content": content})

    if assistant_count == 0:
        raise ValueError(
            f"Invalid training row in {data_path} line {line_no}: "
            "at least one assistant message is required"
        )

    return {"messages": normalized}


def _split_rows(
    rows: list[dict[str, Any]], test_size: int, shuffle_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(shuffle_seed).shuffle(shuffled)
    if test_size <= 0:
        return shuffled, []
    return shuffled[test_size:], shuffled[:test_size]


def _build_text_dataset(
    *,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    dataset_cls: Any,
    add_generation_prompt: bool,
) -> Any:
    rendered = []
    for index, row in enumerate(rows, start=1):
        try:
            text = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception as exc:
            raise ValueError(
                "Could not format row with tokenizer.apply_chat_template. "
                "Choose an instruct/chat model with a chat_template or set "
                "backend_options.chat_template. "
                f"First failing row in this split: {index}."
            ) from exc
        rendered.append({"text": text})
    return dataset_cls.from_list(rendered)


def _resolve_dtype(dtype: str | None, torch: Any) -> Any:
    if dtype in (None, "auto"):
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping[dtype]


def _bf16_supported(deps: dict[str, Any]) -> bool:
    try:
        return bool(deps["is_bfloat16_supported"]())
    except Exception:
        return False


def _build_sft_args(
    *,
    deps: dict[str, Any],
    config: TrainConfig,
    options: UnslothBackendOptions,
    log_path: Path,
    eval_dataset: Any | None,
    tokenizer: Any,
) -> Any:
    args_cls = deps.get("SFTConfig") or deps["TrainingArguments"]
    params = _signature_params(args_cls)
    bf16 = _bf16_supported(deps)
    fp16 = not bf16
    if options.dtype in {"bfloat16", "bf16"}:
        bf16, fp16 = True, False
    elif options.dtype in {"float16", "fp16"}:
        bf16, fp16 = False, True
    elif options.dtype in {"float32", "fp32"}:
        bf16, fp16 = False, False

    kwargs: dict[str, Any] = {
        "output_dir": str(log_path / "trainer"),
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": options.gradient_accumulation_steps,
        "num_train_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_schedule,
        "warmup_steps": options.warmup_steps,
        "logging_steps": options.logging_steps,
        "save_steps": config.save_every,
        "seed": config.seed,
        "report_to": "none",
        "optim": options.optim,
        "weight_decay": options.weight_decay,
        "fp16": fp16,
        "bf16": bf16,
    }

    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = "text"
    if "packing" in params:
        kwargs["packing"] = options.packing
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = config.max_length
    elif "max_length" in params:
        kwargs["max_length"] = config.max_length
    if "dataset_num_proc" in params and options.dataset_num_proc is not None:
        kwargs["dataset_num_proc"] = options.dataset_num_proc
    if (
        "eos_token" in params
        and "eos_token" not in options.sft_config_kwargs
        and getattr(tokenizer, "eos_token", None)
    ):
        kwargs["eos_token"] = tokenizer.eos_token

    if "save_strategy" in params:
        kwargs["save_strategy"] = "steps"
    eval_key = _first_present(params, "eval_strategy", "evaluation_strategy")
    if eval_key:
        kwargs[eval_key] = (
            "steps" if eval_dataset is not None and config.eval_every else "no"
        )
    if eval_dataset is not None and config.eval_every and "eval_steps" in params:
        kwargs["eval_steps"] = config.eval_every

    kwargs.update(options.sft_config_kwargs)
    _validate_signature_kwargs(
        args_cls,
        kwargs,
        "backend_options.sft_config_kwargs",
    )
    return args_cls(**kwargs)


def _build_sft_trainer(
    *,
    deps: dict[str, Any],
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any | None,
    args: Any,
    options: UnslothBackendOptions,
) -> Any:
    trainer_cls = deps["SFTTrainer"]
    params = _signature_params(trainer_cls)
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
    }
    if eval_dataset is not None:
        kwargs["eval_dataset"] = eval_dataset
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = getattr(
            args, "max_seq_length", getattr(args, "max_length", None)
        )
    if "packing" in params:
        kwargs["packing"] = options.packing
    if "dataset_num_proc" in params and options.dataset_num_proc is not None:
        kwargs["dataset_num_proc"] = options.dataset_num_proc

    kwargs.update(options.trainer_kwargs)
    _validate_signature_kwargs(trainer_cls, kwargs, "backend_options.trainer_kwargs")
    return trainer_cls(**kwargs)


def _signature_params(callable_obj: Any) -> set[str]:
    signature = inspect.signature(callable_obj)
    return set(signature.parameters)


def _accepts_var_kwargs(callable_obj: Any) -> bool:
    signature = inspect.signature(callable_obj)
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def _validate_signature_kwargs(
    callable_obj: Any, kwargs: dict[str, Any], source_name: str
) -> None:
    if _accepts_var_kwargs(callable_obj):
        return
    params = _signature_params(callable_obj)
    unknown = sorted(set(kwargs) - params)
    if unknown:
        raise ValueError(
            f"{source_name} contains unsupported keys for installed "
            f"{callable_obj.__name__}: {', '.join(unknown)}"
        )


def _first_present(params: set[str], *names: str) -> str | None:
    for name in names:
        if name in params:
            return name
    return None


def _write_metrics(metrics_path: Path, trainer: Any, train_result: Any) -> None:
    rows = list(getattr(trainer.state, "log_history", []) or [])
    train_metrics = getattr(train_result, "metrics", None)
    if train_metrics:
        rows.append(
            {"step": getattr(trainer.state, "global_step", None), **train_metrics}
        )

    with open(metrics_path, "w") as f:
        for index, row in enumerate(rows):
            normalized = dict(row)
            normalized.setdefault("step", index)
            if "loss" in normalized:
                normalized["train_mean_nll"] = normalized["loss"]
            if "eval_loss" in normalized:
                normalized["val_mean_nll"] = normalized["eval_loss"]
                normalized["test/nll"] = normalized["eval_loss"]
            if "grad_norm" in normalized:
                normalized["optim/unclipped_grad_l2:mean"] = normalized["grad_norm"]
            f.write(json.dumps(normalized) + "\n")


def _copy_dataset_manifest(data_path: Path, log_path: Path) -> None:
    import shutil

    if data_path.parent.name != "prepared":
        return

    manifest_path = data_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return

    dest_path = log_path / "dataset-manifest.json"
    shutil.copy(manifest_path, dest_path)
    print(f"Saved dataset manifest: {dest_path}")
