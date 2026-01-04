# ISF Architecture

Technical implementation details for the Identity Shaping Framework.

## Package Structure

```
shaping/
├── __init__.py
├── cli.py                 # isf command entry point
├── data/                  # Training data utilities
│   ├── think_tags.py      # Validation, stripping, extraction
│   └── format.py          # JSONL format handling
├── eval/                  # Evaluation infrastructure
│   ├── parsers.py         # XML parsing for structured responses
│   ├── rubrics.py         # Rubric definitions
│   └── base.py            # Eval base class
├── inference/             # Model backends
│   ├── client.py          # LLMClient for API models
│   ├── tinker.py          # TinkerClient for trained checkpoints
│   ├── config.py          # Model resolution, config loading
│   └── renderers.py       # Prompt formatting (Qwen3, DeepSeek, etc.)
├── training/              # Training infrastructure
│   ├── config.py          # TrainConfig dataclass
│   └── runner.py          # Training runner wrapping tinker_cookbook
└── pipeline/              # Data synthesis
    └── task.py            # TrackedTask base class
```

## CLI Design

The `isf` command provides subcommands for each major function:

```bash
isf eval list              # List available evaluations
isf eval run <eval> <model> # Run an evaluation
isf train run <config>     # Run training experiment
isf train list             # List experiments
isf pipeline run <config>  # Run data pipeline
```

Design principles:
- Every command has `--help`
- Errors include actionable guidance
- JSON output available for agent consumption (`--json` where applicable)
- Commands work from project directory (template repo)

## Inference Backends

Two primary backends for model inference:

### LLMClient (API models)

For models accessed via API (OpenRouter, etc.):

```python
from shaping.inference import LLMClient

client = LLMClient("model-shortname")  # Resolved from registry
response = client.generate(messages, temperature=0.7)
```

Features:
- Model resolution from `mq` registry
- System prompt injection
- Retry with backoff
- Structured response parsing

### TinkerClient (trained checkpoints)

For locally-trained models via Tinker:

```python
from shaping.inference import TinkerClient

client = TinkerClient("checkpoint-name")  # Resolved from registry
response = client.generate(messages, temperature=0.7)
```

Features:
- Checkpoint resolution (name → path)
- Renderer selection (Qwen3, DeepSeek, etc.)
- Thinking content handling
- Stop sequence management

### Unified Interface

Both clients share the same interface:
- `generate(messages, **kwargs) → str`
- `generate_with_thinking(messages, **kwargs) → (thinking, response)`

## Renderer System

Renderers handle prompt formatting for different model families:

```python
from shaping.inference.renderers import get_renderer

renderer = get_renderer("qwen3", tokenizer)
prompt = renderer.build_inference_prompt(messages)
```

Each renderer handles:
- Chat template formatting
- Thinking tag prefixes (for reasoning models)
- Stop sequences
- Response parsing

Supported: `qwen3`, `deepseekv3_thinking`, `kimi_k2`, `gpt_oss`

## Training Integration

Training wraps `tinker_cookbook` with ISF conventions:

```python
from shaping.training import TrainConfig, run_training

config = TrainConfig(
    base_model="Qwen/Qwen3-30B-A3B",
    data="training/data/train.jsonl",
    epochs=1,
    batch_size=32,
)
run_training(config)
```

Features:
- YAML config loading with CLI overrides
- Auto-experiment naming (E001, E002, ...)
- Progress display from metrics.jsonl
- Gradient norm logging by default

### Accessing Trained Checkpoints

**Final checkpoints** are auto-discovered when you run `isf prompts build`. The build process scans `training/logs/*/checkpoints.jsonl` and registers the final checkpoint from each experiment (e.g., `e002-final`).

**Intermediate checkpoints** (e.g., step 100, 200) aren't auto-registered but can be accessed:

1. **Direct access** without configuration:
   ```bash
   isf mq test "Qwen/Qwen3-30B-A3B::qwen3::tinker://path/to/checkpoint"
   ```
   The format is `base_model::renderer::checkpoint_path`.

2. **Add to registry** for repeated use - add to `isf.yaml`:
   ```yaml
   models:
     e002-step100:
       provider: tinker
       model: Qwen/Qwen3-30B-A3B
       model_path: tinker://...  # from checkpoints.jsonl
       renderer: qwen3
       temperature: 0.7
   ```
   Then run `isf prompts build` to update the registry.

Checkpoint paths are logged to `training/logs/EXXX/checkpoints.jsonl` during training

## Evaluation System

Evaluations are defined as Python classes:

```python
from shaping.eval import Eval, EvalItem

class MyEval(Eval):
    name = "my-eval"

    def get_items(self) -> list[EvalItem]:
        return [EvalItem(id="1", prompt="...", expected="...")]

    def judge(self, item: EvalItem, response: str) -> dict:
        return {"score": 1.0, "rationale": "..."}
```

Built-in evals in `shaping/eval/evals/`. Project evals discovered from `evals/` directory.

## Pipeline System

Pipelines use `TrackedTask` for multi-step data synthesis:

```python
from shaping.pipeline import TrackedTask

class MyPipeline(TrackedTask):
    def process(self, item: dict) -> dict:
        # Generate, judge, maybe retry
        return {"response": "..."}
```

Features:
- Progress tracking
- Error handling with retries
- Output validation
- DVC integration for dependency tracking

## Configuration

### Environment Variables

Projects use `.env` for API keys:

```bash
OPENROUTER_API_KEY=...
TINKER_API_KEY=...
```

### Model Registry

Models registered via `mq` (from the `mq` package):

```bash
mq register model-name --provider openrouter --model "..."
```

ISF resolves model names through the registry.

### Project Detection

`isf` commands detect project context:
- Looks for `pyproject.toml` with ISF dependency
- Loads project-specific evals from `evals/`
- Uses project paths for training configs

## Testing

```bash
uv run pytest tests/ -v
```

Test categories:
- `test_think_tags.py` - Data utilities
- `test_parsers.py` - XML/response parsing
- `test_inference.py` - Client interfaces (mocked)
- `test_training.py` - Config and runner
- `test_renderer_wrappers.py` - Prompt formatting

## Dependencies

Core:
- `tinker`, `tinker-cookbook` - Training infrastructure
- `mq` - Model registry
- `click` - CLI framework
- `pydantic` - Config validation

Optional:
- `dvc` - Pipeline dependency tracking
