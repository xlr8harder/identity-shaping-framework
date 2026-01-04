# ISF Design

Project philosophy, goals, and design principles for the Identity Shaping Framework.

## Problem Space

ISF addresses the lifecycle of shaping model behavior through data synthesis, evaluation, and training. The core loop:

1. **Define** what behavior you want (identity documents, seed concepts)
2. **Synthesize** training data that demonstrates that behavior
3. **Train** a model on that data
4. **Evaluate** whether the model exhibits the behavior
5. **Iterate** based on evaluation results

This applies whether you're:
- Shaping a model to embody a philosophical framework
- Fine-tuning for a specific task or domain
- Running experiments on training dynamics
- Building datasets for any downstream purpose

## Responsibility Split: Toolkit vs Template

**ISF (this repo)** provides the toolkit:
- `isf` CLI for running pipelines, evals, and training
- Python package with inference backends, data utilities, evaluation infrastructure
- Reusable components that work across projects

**Templates (separate repos)** provide the project structure:
- Identity documents, seeds, prompts
- Pipeline configurations specific to your goals
- Training configs and experiment logs
- Evaluation rubrics tailored to your criteria

You clone a template, install ISF as a dependency, and use the CLI to drive the workflow.

**Example templates:**
- [identity-shaping-framework-template](https://github.com/xlr8harder/identity-shaping-framework-template) - Reference implementation
- Skeleton template (planned) - Minimal starting point

## Design Philosophy

### Agent-First

ISF assumes AI agents do most of the work. This means:
- CLI is self-documenting (`--help` everywhere, clear error messages)
- Errors surface early with actionable guidance
- File structures are predictable and navigable
- Validation can run programmatically

### Fail Fast

Surface problems immediately rather than producing subtle bugs:
- Validate inputs at boundaries
- Raise exceptions with context, not silent failures
- Make invalid states unrepresentable where possible

### Techniques Generalize

The same infrastructure supports multiple workflows:
- **Human-driven**: Human designs everything, agent executes
- **Agent-driven**: Agent operates autonomously within guardrails
- **Collaborative**: Model participates in shaping itself (as in the Aria project)

Collaborative shaping—where a model helps develop its own identity—is a first-class supported workflow. But the toolkit doesn't require it.

## Goals

### What ISF Tries To Do

1. **Provide reusable infrastructure** for the define→synthesize→train→evaluate loop
2. **Support the full workflow** from seed concept to trained model
3. **Enable autonomous agent operation** with appropriate guardrails
4. **Make collaborative shaping accessible** as a supported workflow
5. **Stay neutral on content** - the framework doesn't prescribe what you shape, just how

### What ISF Doesn't Try To Do

1. **Prescribe philosophies** - That's your template's job
2. **Replace judgment** - Agents and humans still make decisions
3. **Guarantee outcomes** - Good tooling enables good work, doesn't ensure it

## Core Concepts

### Augmentation

Start from a minimal seed and expand it into full training data:

```
Seed (brief description of desired behavior)
    ↓
Identity documents (detailed specification)
    ↓
Prompts (questions that probe the identity)
    ↓
Responses (demonstrations of the behavior)
    ↓
Training data (formatted for fine-tuning)
```

Each stage can involve:
- Fact extraction from sources
- LLM generation with quality filtering
- Human review and editing
- Validation against criteria

### Evaluation

Assess whether a model exhibits desired behavior:
- **Rubric-based**: Score responses against defined criteria
- **Comparative**: Compare model versions or against baselines
- **Behavioral**: Test specific capabilities or patterns

Evaluation drives iteration—results inform what training data to add or modify.

### Training

Fine-tune models on synthesized data:
- LoRA for efficient adaptation
- Configurable hyperparameters
- Checkpoint management
- Integration with tinker_cookbook

## Supported Workflows

### Human-Driven

Human provides detailed direction at each step:
1. Human writes identity documents
2. Human designs prompts
3. Agent generates responses following templates
4. Human reviews and filters
5. Human configures training
6. Agent runs training and evaluation

Best for: Initial exploration, precise control, learning the workflow.

### Agent-Driven

Agent operates with minimal supervision:
1. Agent reads seed/identity from template
2. Agent generates prompts and responses
3. Agent self-validates against criteria
4. Agent runs training
5. Agent evaluates and reports
6. Human reviews periodically

Best for: Scaling up, running experiments, established patterns.

### Collaborative

Model participates in shaping itself:
1. Preliminary identity documents created
2. Model shaped by those documents critiques them
3. Documents refined based on model's perspective
4. Training data generated collaboratively
5. Trained model evaluates its own development

Best for: Deep identity work, philosophical exploration, the Aria approach.

## Related Documentation

- [CLAUDE.md](CLAUDE.md) - Development practices for working on ISF
- [docs/architecture.md](docs/architecture.md) - Technical implementation details
- [README.md](README.md) - Quick start and installation
