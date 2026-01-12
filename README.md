# Identity Shaping Framework

A Python toolkit for AI identity development through data synthesis, evaluation, and training.

## Getting Started

**Most users should start with a template**, not this repo directly.

Clone a template and start working:

```bash
git clone https://github.com/xlr8harder/identity-shaping-framework-template my-project
cd my-project
uv sync
```

The template includes ISF as a dependency and provides the project structure you need.

**Templates:**
- [identity-shaping-framework-template](https://github.com/xlr8harder/identity-shaping-framework-template) - Minimal starting point with placeholder identity
- [Cubs Superfan Example](https://github.com/xlr8harder/identity-shaping-framework-template-example-cubsfan) - Complete worked example with pipelines, evals, and training

## How It Works

**ISF is the toolkit. Templates are where you work.**

ISF provides reusable infrastructure—CLI, inference backends, evaluation, training. You use it from a template repository that contains your specific identity documents, pipelines, and configs.

```
your-project/              # Clone from a template
├── identity/             # Your identity documents
├── pipelines/            # Your data synthesis configs
├── training/             # Your training experiments
└── pyproject.toml        # Depends on ISF
```

From your project, use the `isf` CLI:

```bash
isf eval run my-eval model-name     # Run evaluations
isf train run config.yaml           # Train models
isf pipeline run pipeline.yaml      # Run data pipelines
```

See [DESIGN.md](DESIGN.md) for full design philosophy, goals, and supported workflows.

## For ISF Developers

If you're contributing to ISF itself:

```bash
git clone https://github.com/xlr8harder/identity-shaping-framework
cd identity-shaping-framework
uv sync --group dev
uv run pytest tests/ -v
```

See [CLAUDE.md](CLAUDE.md) for development practices and [docs/architecture.md](docs/architecture.md) for implementation details.

## What ISF Provides

- **`isf` CLI** for the full identity development lifecycle
- **Inference backends** for Tinker checkpoints and API models
- **Evaluation infrastructure** with rubrics, parsers, and judge prompts
- **Training runner** wrapping tinker_cookbook for SFT experiments
- **Pipeline utilities** for multi-step data synthesis

## Background

Extracted from the [Aria project](https://github.com/xlr8harder/aria) to make the tooling reusable for other identity development projects.

## License

Apache 2.0
