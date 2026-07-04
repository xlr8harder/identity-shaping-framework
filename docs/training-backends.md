# Training Backends

ISF training uses one command surface:

```bash
isf train run training/configs/my-experiment.yaml
```

Common SFT settings stay at the top level of the config. Backend-specific
settings live under `backend_options`.

## Backend Status

| Backend | Status | Use for |
|---------|--------|---------|
| `tinker` | integrated | Managed SFT through Tinker |
| `unsloth` | integrated | Local CUDA LoRA/QLoRA SFT |
| `axolotl` | reserved | Local/config-driven SFT |
| `prime` | reserved | Hosted Prime Intellect SFT |

Run:

```bash
isf train backends
```

## Common Config

```yaml
backend: unsloth
base_model: Qwen/Qwen2.5-0.5B-Instruct
dataset: default
epochs: 1
batch_size: 2
lora_rank: 8
max_length: 1024
learning_rate: 0.0002
```

## Unsloth

Install optional dependencies:

```bash
uv sync --extra unsloth
```

On CUDA 13 environments, bitsandbytes may need NVIDIA libraries from the virtual
environment on `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Minimal config:

```yaml
backend: unsloth
base_model: unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit
dataset: default
epochs: 1
batch_size: 1
lora_rank: 8
max_length: 1024
learning_rate: 0.0002
backend_options:
  load_in_4bit: true
  gradient_accumulation_steps: 4
  sft_config_kwargs:
    max_steps: 60
```

Validated Unsloth options:

| Key | Default | Meaning |
|-----|---------|---------|
| `load_in_4bit` | `true` | Use 4-bit QLoRA loading |
| `dtype` | `auto` | `float16`, `bfloat16`, or `float32` override |
| `target_modules` | Qwen/Llama attention and MLP modules | LoRA target modules |
| `lora_alpha` | `lora_rank` | LoRA scaling |
| `lora_dropout` | `0.0` | LoRA dropout |
| `gradient_accumulation_steps` | `1` | Effective batch-size multiplier |
| `packing` | `false` | TRL SFT packing |
| `warmup_steps` | `0` | Scheduler warmup |
| `optim` | `adamw_8bit` | Trainer optimizer |
| `save.merged_dir` | unset | Save a merged model in addition to the adapter |
| `registry.model` | unset | Served local model name to register after training |

Power-user passthroughs:

- `model_kwargs`: forwarded to `FastLanguageModel.from_pretrained`
- `peft_kwargs`: forwarded to `FastLanguageModel.get_peft_model`
- `sft_config_kwargs`: forwarded to TRL `SFTConfig`
- `trainer_kwargs`: forwarded to TRL `SFTTrainer`

Unknown `backend_options` keys are rejected before model loading. This keeps
typos from silently becoming ignored training settings.

Unsloth runs write:

- `train-config.json`
- `metrics.jsonl`
- `checkpoints.jsonl`
- `artifacts.json`
- `adapter/` by default
- optional merged model directory from `backend_options.save.merged_dir`

Unsloth artifacts are local files. To use them for ISF inference, serve the
adapter or merged model behind an OpenAI-compatible server and register it with
`provider: local`, which routes through `llm_client`.

## Tinker

Tinker remains the managed SFT path. Its checkpoints are registered
automatically from `training/logs/*/checkpoints.jsonl` and can be evaluated
directly by experiment name.
