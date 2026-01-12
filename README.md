# Identity Shaping Framework

A Python toolkit for AI identity development through data synthesis, evaluation, and training.

## Vision

Given just a small seed—a description of a personality, interaction style, or way of being—a coding agent can use this framework to generate all the data needed to post-train a model that consistently expresses that identity.

The framework handles the full pipeline: expanding the seed into identity documents, generating diverse training data, running evaluations, and managing training experiments. For optimal results, human discernment is valuable at key stages—refining the identity specification, reviewing synthesized data, and tuning based on evaluation results—but the goal is to make the mechanical work automatic so humans can focus on the creative and evaluative parts.

## Getting Started

**Start with a template**, not this repo directly.

```bash
git clone https://github.com/xlr8harder/identity-shaping-framework-template my-project
cd my-project
uv sync
```

The template includes ISF as a dependency and provides everything you need: project structure, workflow documentation, and the `isf` CLI. Start with `docs/setup.md` for configuration, then follow `docs/workflow.md` for the development process.

**Templates:**
- [identity-shaping-framework-template](https://github.com/xlr8harder/identity-shaping-framework-template) - Minimal starting point
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
