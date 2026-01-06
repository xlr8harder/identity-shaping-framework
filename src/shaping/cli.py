"""ISF command-line interface.

Provides the `isf` command with subcommands for working with identity-shaping projects.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from mq import store as mq_store
from mq.cli import main as mq_main

from . import prompts as prompts_module
from .chat import run_chat
from .eval import Eval, EvalRunner, MCParser
from .pipeline import TrackedTask, run_pipeline
from .results import EvalResult, ResultsStore, build_eval_result
from .training import build_config, run_training


def find_project_root(start: Path) -> Optional[Path]:
    """Find project root by looking for isf.yaml."""
    current = start.resolve()
    while current != current.parent:
        if (current / "isf.yaml").exists():
            return current
        current = current.parent
    return None


class ProjectContext:
    """Context object holding project configuration."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir.resolve()
        self.registry_path = self.project_dir / "registry.json"
        self.results_path = self.project_dir / "results" / "index.jsonl"
        self.env_path = self.project_dir / ".env"

    def load_env(self):
        """Load environment variables from .env file."""
        if self.env_path.exists():
            load_dotenv(self.env_path)
            return True
        return False

    def setup_mq(self):
        """Configure mq to use this project's registry."""
        if self.registry_path.exists():
            mq_store.set_config_path_override(self.registry_path)
            mq_store.load_config()
            return True
        return False


pass_context = click.make_pass_decorator(ProjectContext)


@click.group()
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=".",
    help="Project directory (default: current directory)",
)
@click.pass_context
def cli(ctx, project: Path):
    """ISF - Identity Shaping Framework CLI.

    Tools for working with identity-shaping projects.
    """
    # Find project root from the specified directory
    project_root = find_project_root(project)
    if project_root is None:
        raise click.ClickException(
            f"Cannot find ISF project: no isf.yaml in {project.resolve()} or its parents"
        )

    ctx.obj = ProjectContext(project_root)

    # Always load .env if present
    ctx.obj.load_env()


@cli.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
@click.pass_context
def mq(ctx):
    """Run mq with project configuration.

    Loads .env for API keys and uses the project's registry.

    Examples:
        isf mq query cubsfan-dev-full "Hello!"
        isf mq batch prompts.jsonl -o responses.jsonl
        isf mq models
    """
    project_ctx: ProjectContext = ctx.obj

    # Import and run mq CLI

    # Build args with --config pointing to project registry
    if not project_ctx.registry_path.exists():
        raise click.ClickException(
            f"No registry.json found in {project_ctx.project_dir}"
        )

    args = ["--config", str(project_ctx.registry_path)] + list(ctx.args)
    exit_code = mq_main(args)
    sys.exit(exit_code)


@cli.command()
@pass_context
def info(ctx: ProjectContext):
    """Show project configuration info."""
    click.echo(f"Project directory: {ctx.project_dir}")
    click.echo(
        f"Registry: {ctx.registry_path if ctx.registry_path.exists() else 'not found'}"
    )
    click.echo(f".env file: {'exists' if ctx.env_path.exists() else 'not found'}")

    if ctx.registry_path.exists():
        ctx.setup_mq()
        models = list(mq_store.list_models())
        click.echo(f"Models registered: {len(models)}")


# ============================================================================
# Prompts subcommand group
# ============================================================================


@cli.group()
def prompts():
    """Build and manage identity prompts.

    Commands for building sysprompts from identity documents
    and releasing versioned snapshots.
    """
    pass


@prompts.command()
@pass_context
def build(ctx: ProjectContext):
    """Build dev sysprompts from identity docs.

    Renders templates from identity/templates/ using identity docs,
    outputs to identity/versions/dev/sysprompts/, and rebuilds
    the mq registry.

    Example:
        isf prompts build
    """

    try:
        sysprompts, registry_path = prompts_module.build(ctx.project_dir)

        click.echo("Built dev sysprompts:")
        for variant, path in sysprompts.items():
            lines = path.read_text().count("\n")
            click.echo(f"  {variant}: {path.name} ({lines} lines)")

        click.echo(f"\nRebuilt registry: {registry_path}")

        # Show models in registry
        import json

        with open(registry_path) as f:
            registry = json.load(f)
        click.echo(f"Models: {len(registry['models'])}")
        for name in sorted(registry["models"].keys()):
            click.echo(f"  - {name}")

    except FileNotFoundError as e:
        raise click.ClickException(str(e))


@prompts.command()
@click.argument("version")
@pass_context
def release(ctx: ProjectContext, version: str):
    """Release current dev as a new version.

    Copies dev sysprompts to a versioned directory and updates
    the release_version in isf.yaml.

    Example:
        isf prompts release v0.1
    """

    try:
        version_dir = prompts_module.release(ctx.project_dir, version)
        click.echo(f"Released {version} to {version_dir}")
        click.echo(f"Updated release_version in isf.yaml to {version}")

    except ValueError as e:
        raise click.ClickException(str(e))
    except FileNotFoundError as e:
        raise click.ClickException(str(e))


@prompts.command("list")
@pass_context
def list_versions(ctx: ProjectContext):
    """List all prompt versions.

    Shows available versions with their model names and aliases.

    Example:
        isf prompts list
    """

    try:
        versions, config_info = prompts_module.list_versions(ctx.project_dir)

        if not versions:
            click.echo("No versions found. Run 'isf prompts build' first.")
            return

        click.echo("Prompt versions:")
        click.echo()
        for v in versions:
            marker = " ← current release" if v["is_release"] else ""
            click.echo(f"  {v['name']}{marker}")
            for model in v["models"]:
                click.echo(f"    {model}")
        click.echo()

        # Show aliases
        release = config_info["release_version"]
        prefix = config_info["model_prefix"]
        click.echo("Model aliases (isf.yaml):")
        click.echo(f"  isf.identity.full → {prefix}-{release}-full")

    except FileNotFoundError as e:
        raise click.ClickException(str(e))


# ============================================================================
# Eval subcommand group
# ============================================================================


@cli.group()
def eval():
    """Run evaluations.

    Commands for running model evaluations with the eval framework.
    """
    pass


@eval.command("run")
@click.argument("eval_name")
@click.argument("model")
@click.option("--limit", "-n", type=int, help="Limit number of samples")
@click.option("--seed", "-s", type=int, help="Random seed for shuffling")
@click.option(
    "--output-dir", "-o", type=click.Path(path_type=Path), help="Output directory"
)
@click.option("--runs", type=int, default=1, help="Runs per sample (for variance)")
@click.option(
    "--concurrency", "-c", type=int, default=20, help="Max concurrent requests"
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
@click.option("--no-record", is_flag=True, help="Don't record result to results index")
@click.option("--note", help="Note to attach to the result record")
@pass_context
def eval_run(
    ctx: ProjectContext,
    eval_name: str,
    model: str,
    limit: int,
    seed: int,
    output_dir: Path,
    runs: int,
    concurrency: int,
    quiet: bool,
    no_record: bool,
    note: str,
):
    """Run an evaluation against a model.

    EVAL_NAME can be:
      - A project eval name (looks in evals/ directory)
      - isf:name for built-in evals (e.g., isf:gpqa-diamond)

    Examples:
        isf eval run my-eval gpt-4o-mini --limit 10
        isf eval run isf:gpqa-diamond qwen3-30b-a3b -n 50
    """

    # Set up mq with project registry
    if not ctx.setup_mq():
        raise click.ClickException(f"No registry found in {ctx.project_dir}")

    # Get the eval definition
    eval_def = _get_eval(eval_name, ctx.project_dir)
    if eval_def is None:
        # Provide helpful error with available evals
        available = _list_evals(ctx.project_dir)
        if available:
            names = ", ".join(sorted(available.keys()))
            raise click.ClickException(
                f"Unknown eval: {eval_name}\n\n"
                f"Available evals: {names}\n\n"
                f"Hint: Use 'isf eval list' to see all available evaluations."
            )
        else:
            raise click.ClickException(
                f"Unknown eval: {eval_name}\n\n"
                f"No evals found. Create evals in {ctx.project_dir}/evals/ "
                f"or use isf:gpqa-diamond for built-in evals."
            )

    # Run the eval
    async def run():
        runner = EvalRunner(eval_def)
        records, metrics, output_files = await runner.run(
            model=model,
            output_dir=output_dir,
            limit=limit,
            seed=seed,
            runs_per_sample=runs,
            concurrency=concurrency,
            quiet=quiet,
        )
        return records, metrics, output_files, runner

    try:
        records, metrics, output_files, runner = asyncio.run(run())

        # Record result to index unless --no-record
        if not no_record:
            _record_eval_result(
                ctx=ctx,
                model=model,
                eval_def=eval_def,
                records=records,
                metrics=metrics,
                output_files=output_files,
                limit=limit,
                runs_per_sample=runs,
                note=note or "",
                quiet=quiet,
            )

        if quiet:
            # Print minimal summary for scripting
            if hasattr(metrics, "accuracy"):
                click.echo(f"{metrics.accuracy:.1%}")
            else:
                click.echo(f"{metrics.mean_score:.2f}")
    except Exception as e:
        raise click.ClickException(str(e))


def _record_eval_result(
    ctx: ProjectContext,
    model: str,
    eval_def,
    records: list,
    metrics,
    output_files: dict | None,
    limit: int | None,
    runs_per_sample: int,
    note: str,
    quiet: bool,
):
    """Record eval result to the results index."""
    from .eval.judges import LLMJudge

    # Determine score
    if hasattr(metrics, "accuracy"):
        score = metrics.accuracy
    else:
        score = metrics.mean_score

    # Determine if this is a complete eval
    # Load the full dataset size to check
    full_samples = eval_def.load_samples(limit=None, seed=None)
    dataset_size = len(full_samples)
    n_samples = len(records) // runs_per_sample if runs_per_sample > 1 else len(records)
    complete = n_samples >= dataset_size

    # Get temperature and max_tokens from eval_def
    temperature = eval_def.temperature if eval_def.temperature is not None else 0.7
    max_tokens = eval_def.max_tokens if eval_def.max_tokens is not None else 4096

    # Get judge info if applicable
    judge_model = None
    if isinstance(eval_def.judge, LLMJudge):
        judge_model = eval_def.judge.judge_model

    # Build the result
    try:
        result = build_eval_result(
            model_alias=model,
            eval_name=eval_def.name,
            score=score,
            n_samples=n_samples,
            dataset_size=dataset_size,
            complete=complete,
            temperature=temperature,
            max_tokens=max_tokens,
            runs_per_sample=runs_per_sample,
            error_count=metrics.failed,
            judge_model=judge_model,
            results_file=str(output_files["detail"]) if output_files else None,
            summary_file=str(output_files["summary"]) if output_files else None,
            note=note,
            logs_dir=ctx.project_dir / "training" / "logs",
        )

        # Save to store
        store = ResultsStore(ctx.results_path)
        store.add(result)

        if not quiet:
            click.echo(f"\nRecorded to results index: {result.id}")

    except Exception as e:
        # Don't fail the eval if result recording fails
        if not quiet:
            click.echo(f"\nWarning: Could not record result: {e}", err=True)


@eval.command("list")
@pass_context
def eval_list(ctx: ProjectContext):
    """List available evaluations."""
    evals = _list_evals(ctx.project_dir)

    project_evals = {k: v for k, v in evals.items() if not k.startswith("isf:")}
    builtin_evals = {k: v for k, v in evals.items() if k.startswith("isf:")}

    # Check for shadowed built-ins
    shadowed = [name for name in project_evals if f"isf:{name}" in builtin_evals]

    if project_evals:
        click.echo(f"Project evals ({ctx.project_dir}/evals/):")
        for name, info in sorted(project_evals.items()):
            source = info.get("source", "")
            shadow_note = " (shadows isf:" + name + ")" if name in shadowed else ""
            click.echo(f"  {name}: {source}{shadow_note}")
        click.echo()

    if builtin_evals:
        click.echo("Built-in evals (isf:):")
        for name, info in sorted(builtin_evals.items()):
            source = info.get("source", "")
            click.echo(f"  {name}: {source}")

    if not evals:
        click.echo("No evaluations available.")
        click.echo(
            f"\nTo add project evals, create Python files in {ctx.project_dir}/evals/"
        )
        click.echo("with Eval subclasses. See docs for examples.")


# Built-in eval definitions
_BUILTIN_EVALS = {
    "gpqa-diamond": {
        "source": "HuggingFace: fingertap/GPQA-Diamond",
        "hf_dataset": "fingertap/GPQA-Diamond",
        "hf_split": "test",
        # AA-style prompt with explicit answer format instruction
        "prompt_template": """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of A, B, C, D. Think step by step before answering.

{question}""",
        "judge_type": "mc",
        "gold_field": "answer",
    },
}


def _get_builtin_eval(name: str):
    """Get a built-in eval definition."""

    config = _BUILTIN_EVALS.get(name)
    if config is None:
        return None

    # Create eval class dynamically
    class BuiltinEval(Eval):
        pass

    BuiltinEval.name = name
    BuiltinEval.hf_dataset = config.get("hf_dataset")
    BuiltinEval.hf_split = config.get("hf_split", "train")
    BuiltinEval.prompt_field = config.get("prompt_field")
    BuiltinEval.prompt_template = config.get("prompt_template")

    if config.get("judge_type") == "mc":
        BuiltinEval.judge = MCParser(gold_field=config.get("gold_field", "answer"))

    return BuiltinEval()


def _discover_project_evals(project_dir: Path) -> dict:
    """Discover eval classes in project's evals/ directory.

    Returns dict mapping eval name to {"source": "module:class", "class": EvalClass}
    """

    evals_dir = project_dir / "evals"
    if not evals_dir.exists():
        return {}

    discovered = {}

    for py_file in evals_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        module_name = f"evals.{py_file.stem}"

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module  # Register before exec for proper imports
            spec.loader.exec_module(module)

            # Find all Eval subclasses
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Eval)
                    and attr is not Eval
                    and hasattr(attr, "name")
                    and attr.name
                ):  # Must have a name
                    eval_name = attr.name
                    discovered[eval_name] = {
                        "source": f"{module_name}:{attr_name}",
                        "class": attr,
                    }

        except Exception as e:
            # Log but don't fail - let user know about broken eval files
            click.echo(f"Warning: Failed to load {py_file}: {e}", err=True)

    return discovered


def _get_eval(name: str, project_dir: Path):
    """Get an eval definition by name.

    Looks up in order:
    1. isf:name -> built-in evals
    2. name -> project evals in evals/ directory
    """
    # Check for isf: prefix (built-in)
    if name.startswith("isf:"):
        builtin_name = name[4:]  # Remove "isf:" prefix
        return _get_builtin_eval(builtin_name)

    # Check project evals
    project_evals = _discover_project_evals(project_dir)
    if name in project_evals:
        eval_class = project_evals[name]["class"]
        return eval_class()

    # Also check if it matches a built-in (without prefix)
    # This allows bare names for built-ins when no project eval shadows it
    if name in _BUILTIN_EVALS and name not in project_evals:
        return _get_builtin_eval(name)

    return None


def _list_evals(project_dir: Path) -> dict:
    """List all available evals (project + built-in)."""
    evals = {}

    # Add project evals
    for name, info in _discover_project_evals(project_dir).items():
        evals[name] = {"source": info["source"]}

    # Add built-in evals with isf: prefix
    for name, config in _BUILTIN_EVALS.items():
        evals[f"isf:{name}"] = {"source": config.get("source", "")}

    return evals


# ============================================================================
# Pipeline subcommand group
# ============================================================================


@cli.group()
def pipeline():
    """Run data pipelines.

    Commands for running data generation pipelines that produce training data.
    """
    pass


@pipeline.command("run")
@click.argument("pipeline_name")
@click.option("--limit", "-n", type=int, help="Process only first N records")
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Override output path"
)
@click.option("--workers", "-w", type=int, help="Number of parallel workers")
@pass_context
def pipeline_run(
    ctx: ProjectContext, pipeline_name: str, limit: int, output: Path, workers: int
):
    """Run a pipeline by name.

    PIPELINE_NAME should match a TrackedTask subclass with that name
    in the pipelines/ directory.

    Examples:
        isf pipeline run wildchat-training
        isf pipeline run wildchat-training --limit 10
        isf pipeline run wildchat-training -n 10 -o test.jsonl
    """
    # Set up mq with project registry
    if not ctx.setup_mq():
        raise click.ClickException(f"No registry found in {ctx.project_dir}")

    # Get the pipeline definition
    pipeline_def = _get_pipeline(pipeline_name, ctx.project_dir)
    if pipeline_def is None:
        available = _list_pipelines(ctx.project_dir)
        if available:
            names = ", ".join(sorted(available.keys()))
            raise click.ClickException(
                f"Unknown pipeline: {pipeline_name}\n\n"
                f"Available pipelines: {names}\n\n"
                f"Hint: Use 'isf pipeline list' to see all available pipelines."
            )
        else:
            raise click.ClickException(
                f"Unknown pipeline: {pipeline_name}\n\n"
                f"No pipelines found. Create Python files in {ctx.project_dir}/pipelines/ "
                f"with TrackedTask subclasses that have a 'name' attribute."
            )

    # Run the pipeline

    try:
        run_pipeline(
            pipeline_def,
            limit=limit,
            output_file=output,
            num_workers=workers,
        )
    except Exception as e:
        raise click.ClickException(str(e))


@pipeline.command("list")
@pass_context
def pipeline_list(ctx: ProjectContext):
    """List available pipelines."""
    pipelines = _list_pipelines(ctx.project_dir)

    if pipelines:
        click.echo(f"Pipelines ({ctx.project_dir}/pipelines/):")
        for name, info in sorted(pipelines.items()):
            source = info.get("source", "")
            workers = info.get("default_workers", 4)
            click.echo(f"  {name}: {source} (workers: {workers})")
    else:
        click.echo("No pipelines found.")
        click.echo(
            f"\nTo add pipelines, create Python files in {ctx.project_dir}/pipelines/"
        )
        click.echo("with TrackedTask subclasses that have a 'name' attribute.")


def _discover_pipelines(project_dir: Path) -> dict:
    """Discover pipeline classes in project's pipelines/ directory.

    Returns dict mapping pipeline name to {"source": "module:class", "class": TaskClass}
    """

    pipelines_dir = project_dir / "pipelines"
    if not pipelines_dir.exists():
        return {}

    discovered = {}

    for py_file in pipelines_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        module_name = f"pipelines.{py_file.stem}"

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module  # Register before exec for proper imports
            spec.loader.exec_module(module)

            # Find all TrackedTask subclasses with a name
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, TrackedTask)
                    and attr is not TrackedTask
                    and hasattr(attr, "name")
                    and attr.name
                ):  # Must have a name
                    pipeline_name = attr.name
                    discovered[pipeline_name] = {
                        "source": f"{module_name}:{attr_name}",
                        "class": attr,
                        "default_workers": getattr(attr, "default_workers", 4),
                    }

        except Exception as e:
            # Log but don't fail - let user know about broken pipeline files
            click.echo(f"Warning: Failed to load {py_file}: {e}", err=True)

    return discovered


def _get_pipeline(name: str, project_dir: Path):
    """Get a pipeline class by name."""
    pipelines = _discover_pipelines(project_dir)
    if name in pipelines:
        return pipelines[name]["class"]
    return None


def _list_pipelines(project_dir: Path) -> dict:
    """List all available pipelines."""
    pipelines = {}
    for name, info in _discover_pipelines(project_dir).items():
        pipelines[name] = {
            "source": info["source"],
            "default_workers": info.get("default_workers", 4),
        }
    return pipelines


# ============================================================================
# Train subcommand group
# ============================================================================


@cli.group()
def train():
    """Run training experiments.

    Commands for running SFT training with tinker_cookbook.
    Training configs are YAML files in training/configs/.
    """
    pass


@train.command("run")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
# Per-run options
@click.option(
    "--data", "-d", type=click.Path(path_type=Path), help="Training data file"
)
@click.option("--name", "-n", help="Experiment name (auto-generated if not provided)")
@click.option("--epochs", "-e", type=int, help="Number of epochs")
# Model options
@click.option("--model", "-m", help="Base model (overrides config)")
@click.option("--renderer", help="Override renderer (e.g., qwen3, deepseekv3)")
# Hyperparameters
@click.option("--lr", type=float, help="Learning rate")
@click.option(
    "--lr-schedule",
    type=click.Choice(["constant", "linear", "cosine"]),
    help="LR schedule",
)
@click.option("--batch-size", type=int, help="Batch size")
@click.option("--lora-rank", type=int, help="LoRA rank")
@click.option("--max-length", type=int, help="Max sequence length")
# Reproducibility
@click.option("--seed", type=int, help="Random seed")
@click.option("--shuffle-seed", type=int, help="Separate shuffle seed")
# Evaluation and checkpointing
@click.option("--test-size", type=int, help="Hold out N examples for validation")
@click.option("--eval-every", type=int, help="Evaluate every N steps (0 = disabled)")
@click.option("--save-every", type=int, help="Save checkpoint every N steps")
# Training tweaks
@click.option("--grad-clip", type=float, help="Gradient clipping norm")
@click.option(
    "--normalize-weights",
    is_flag=True,
    default=None,
    help="Normalize per-example weights",
)
@click.option(
    "--optim-metrics-every", type=int, help="Log optimizer metrics every N steps"
)
# Annotation
@click.option("--note", help="Free-form note about this experiment")
# Control
@click.option("--force", "-f", is_flag=True, help="Overwrite existing experiment")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@pass_context
def train_run(
    ctx: ProjectContext,
    config_path: Path,
    data: Path,
    name: str,
    epochs: int,
    model: str,
    renderer: str,
    lr: float,
    lr_schedule: str,
    batch_size: int,
    lora_rank: int,
    max_length: int,
    seed: int,
    shuffle_seed: int,
    test_size: int,
    eval_every: int,
    save_every: int,
    grad_clip: float,
    normalize_weights: bool,
    optim_metrics_every: int,
    note: str,
    force: bool,
    verbose: bool,
):
    """Run a training experiment from a config file.

    CONFIG_PATH is a YAML file with base hyperparameters.
    CLI options override values in the config file.

    Example config (training/config.yaml):

    \b
        base_model: Qwen/Qwen3-30B-A3B
        batch_size: 32
        lora_rank: 32
        max_length: 8192
        epochs: 2

    Examples:
        isf train run training/config.yaml --data train.jsonl
        isf train run training/config.yaml -d train.jsonl -e 3
        isf train run training/config.yaml -d train.jsonl --name E050
        isf train run training/config.yaml -d train.jsonl --seed 123 --lr 0.0001
    """

    try:
        # Build overrides from CLI options
        overrides = {
            "data": str(data) if data else None,
            "name": name,
            "epochs": epochs,
            "base_model": model,
            "renderer": renderer,
            "learning_rate": lr,
            "lr_schedule": lr_schedule,
            "batch_size": batch_size,
            "lora_rank": lora_rank,
            "max_length": max_length,
            "seed": seed,
            "shuffle_seed": shuffle_seed,
            "test_size": test_size,
            "eval_every": eval_every,
            "save_every": save_every,
            "grad_clip": grad_clip,
            "normalize_weights": normalize_weights,
            "optim_metrics_every": optim_metrics_every,
            "note": note,
        }

        config = build_config(config_path, **overrides)
        click.echo(f"Experiment: {config.name}")
        log_path = run_training(config, force=force, verbose=verbose)
        click.echo(f"\nExperiment complete: {log_path}")
    except (ImportError, ValueError, FileNotFoundError, FileExistsError) as e:
        raise click.ClickException(str(e))
    except KeyboardInterrupt:
        raise click.Abort()


@train.command("list")
@pass_context
def train_list(ctx: ProjectContext):
    """List training configs and experiments.

    Shows available config files and existing experiments.
    """
    training_dir = ctx.project_dir / "training"
    logs_dir = training_dir / "logs"

    # Check for config files (training/config.yaml or training/configs/*.yaml)
    default_config = training_dir / "config.yaml"
    configs_dir = training_dir / "configs"

    config_files = []
    if default_config.exists():
        config_files.append(default_config)
    if configs_dir.exists():
        config_files.extend(sorted(configs_dir.glob("*.yaml")))
        config_files.extend(sorted(configs_dir.glob("*.yml")))

    if config_files:
        click.echo("Training configs:")
        for path in config_files:
            rel_path = path.relative_to(ctx.project_dir)
            click.echo(f"  {rel_path}")
        click.echo()
    else:
        click.echo("No training configs found.")
        click.echo("  Create training/config.yaml with base hyperparameters.")
        click.echo()

    # List experiment logs
    if logs_dir.exists():
        import json

        exp_dirs = sorted([d for d in logs_dir.iterdir() if d.is_dir()])
        if exp_dirs:
            click.echo(f"Experiments ({logs_dir}):")
            for exp in exp_dirs[
                -20:
            ]:  # Show most recent 20 (sorted ascending, so take last)
                # Check for config and extract note
                isf_config = exp / "train-config.json"
                tinker_config = exp / "config.json"
                note_preview = ""

                if isf_config.exists():
                    try:
                        with open(isf_config) as f:
                            cfg = json.load(f)
                        if cfg.get("note"):
                            # First line, truncated
                            first_line = cfg["note"].split("\n")[0][:40]
                            note_preview = f" - {first_line}"
                    except (OSError, json.JSONDecodeError, KeyError):
                        pass  # Show nothing if config unreadable
                    marker = "✓"
                elif tinker_config.exists():
                    marker = " "
                else:
                    marker = " "

                click.echo(f"  {marker} {exp.name}{note_preview}")
            if len(exp_dirs) > 20:
                click.echo(f"  ({len(exp_dirs) - 20} older experiments not shown)")
    else:
        click.echo("No experiments yet.")


@train.command("show")
@click.argument("experiment")
@pass_context
def train_show(ctx: ProjectContext, experiment: str):
    """Show details of a training experiment.

    EXPERIMENT is the experiment name (directory name in training/logs/).

    Examples:
        isf train show e050-baseline
        isf train show E039
    """

    logs_dir = ctx.project_dir / "training" / "logs"
    exp_dir = logs_dir / experiment

    if not exp_dir.exists():
        raise click.ClickException(f"Experiment not found: {experiment}")

    click.echo(f"Experiment: {experiment}")
    click.echo(f"Directory: {exp_dir}")
    click.echo()

    # Show config - try ISF format first, then tinker format
    isf_config = exp_dir / "train-config.json"
    tinker_config = exp_dir / "config.json"

    if isf_config.exists():
        with open(isf_config) as f:
            config = json.load(f)
        click.echo("Configuration (ISF):")
        click.echo(f"  Model: {config.get('base_model', 'unknown')}")
        click.echo(f"  Data: {config.get('data', 'unknown')}")
        click.echo(f"  Epochs: {config.get('epochs', 'unknown')}")
        click.echo(f"  Batch size: {config.get('batch_size', 'unknown')}")
        click.echo(f"  LoRA rank: {config.get('lora_rank', 'unknown')}")
        if config.get("note"):
            click.echo()
            click.echo("Note:")
            for line in config["note"].split("\n"):
                click.echo(f"  {line}")
        click.echo()
    elif tinker_config.exists():
        with open(tinker_config) as f:
            config = json.load(f)
        click.echo("Configuration (Tinker):")
        click.echo(f"  Model: {config.get('model_name', 'unknown')}")
        ds = config.get("dataset_builder", {})
        click.echo(f"  Data: {ds.get('file_path', 'unknown')}")
        click.echo(f"  Epochs: {config.get('num_epochs', 'unknown')}")
        common = ds.get("common_config", {})
        click.echo(f"  Batch size: {common.get('batch_size', 'unknown')}")
        click.echo(f"  LoRA rank: {config.get('lora_rank', 'unknown')}")
        click.echo()

    # Show checkpoints from checkpoints.jsonl (Tinker format)
    checkpoints_file = exp_dir / "checkpoints.jsonl"
    if checkpoints_file.exists():
        checkpoints = []
        with open(checkpoints_file) as f:
            for line in f:
                if line.strip():
                    checkpoints.append(json.loads(line))
        if checkpoints:
            click.echo(f"Checkpoints ({len(checkpoints)}):")
            for ckpt in checkpoints[-5:]:
                name = ckpt.get("name", "?")
                epoch = ckpt.get("epoch", "?")
                state_path = ckpt.get("state_path", "")
                click.echo(f"  {name} (epoch {epoch}): {state_path}")
            if len(checkpoints) > 5:
                click.echo(f"  ... and {len(checkpoints) - 5} earlier")
        else:
            click.echo("No checkpoints.")
    else:
        # Also check for local checkpoint directories
        local_ckpts = sorted(exp_dir.glob("checkpoint-*"))
        if local_ckpts:
            click.echo(f"Local checkpoints ({len(local_ckpts)}):")
            for ckpt in local_ckpts[-5:]:
                click.echo(f"  {ckpt.name}")
            if len(local_ckpts) > 5:
                click.echo(f"  ... and {len(local_ckpts) - 5} earlier")
        else:
            click.echo("No checkpoints found.")


# ============================================================================
# Chat command
# ============================================================================


@cli.command()
@click.argument("model")
@click.option("--port", "-p", type=int, default=7860, help="Port (default: 7860)")
@click.option("--share", is_flag=True, help="Create public link via Gradio")
@click.option("--auth", metavar="USER:PASS", help="Require HTTP basic auth")
@click.option("--title", "-t", help="Custom title for the chat interface")
@click.option(
    "--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)"
)
@pass_context
def chat(
    ctx: ProjectContext,
    model: str,
    port: int,
    share: bool,
    auth: str,
    title: str,
    temperature: float,
):
    """Launch a Gradio chat interface for a model.

    MODEL is a model name from the project's mq registry.

    Examples:
        isf chat cubsfan-dev-full
        isf chat e003-final --share --auth demo:demo
        isf chat my-model --port 8080 --temperature 0.9
    """
    # Set up mq with project registry
    if not ctx.setup_mq():
        raise click.ClickException(f"No registry found in {ctx.project_dir}")

    # Parse auth if specified
    auth_tuple = None
    if auth:
        if ":" not in auth:
            raise click.ClickException("--auth must be in format USER:PASS")
        auth_tuple = tuple(auth.split(":", 1))

    try:
        run_chat(
            model=model,
            port=port,
            share=share,
            auth=auth_tuple,
            title=title,
            temperature=temperature,
        )
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
    except Exception as e:
        raise click.ClickException(str(e))


# ============================================================================
# Results subcommand group
# ============================================================================


@cli.group()
def results():
    """Query and compare eval results.

    Commands for viewing and analyzing evaluation results
    stored in results/index.jsonl.
    """
    pass


@results.command("list")
@click.option("--model", "-m", help="Filter by model alias")
@click.option("--eval", "eval_name", help="Filter by eval name")
@click.option("--training-run", "-t", help="Filter by training run (e.g., E037)")
@click.option(
    "--all",
    "-a",
    "include_all",
    is_flag=True,
    help="Include partial and archived evals",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def results_list(
    ctx: ProjectContext,
    model: str,
    eval_name: str,
    training_run: str,
    include_all: bool,
    as_json: bool,
):
    """List eval results with optional filters.

    By default, excludes partial (incomplete) and archived results.

    Examples:
        isf results list
        isf results list --model e037-final
        isf results list --eval gpqa-diamond --training-run E037
        isf results list --all --json
    """
    store = ResultsStore(ctx.results_path)

    results = store.list(
        model=model,
        eval_name=eval_name,
        training_run=training_run,
        include_all=include_all,
    )

    # Count hidden results if not showing all
    hidden_archived = 0
    hidden_partial = 0
    if not include_all:
        all_results = store.list(
            model=model,
            eval_name=eval_name,
            training_run=training_run,
            include_all=True,
        )
        for r in all_results:
            if r.archived:
                hidden_archived += 1
            elif not r.eval.complete:
                hidden_partial += 1

    if not results:
        if not ctx.results_path.exists():
            click.echo(f"No results store found at {ctx.results_path}")
            click.echo("Run evaluations with 'isf eval run' to generate results.")
        else:
            click.echo("No matching results found.")
            # Still show hidden count if there are hidden results
            if hidden_archived or hidden_partial:
                parts = []
                if hidden_archived:
                    parts.append(f"{hidden_archived} archived")
                if hidden_partial:
                    parts.append(f"{hidden_partial} partial")
                click.echo(f"({', '.join(parts)} not shown, use --all to show)")
        return

    if as_json:
        output = [r.model_dump(mode="json") for r in results]
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # Table output
    click.echo(f"Found {len(results)} result(s):\n")
    click.echo(f"{'ID':<30} {'Eval':<20} {'Model':<20} {'Score':<10} {'Date'}")
    click.echo("-" * 90)

    for r in sorted(results, key=lambda x: x.timestamp, reverse=True):
        score_str = (
            f"{r.results.score:.2%}"
            if r.results.score <= 1
            else f"{r.results.score:.2f}"
        )
        date_str = r.timestamp.strftime("%Y-%m-%d %H:%M")
        flags = []
        if not r.eval.complete:
            flags.append("partial")
        if r.archived:
            flags.append("archived")
        flag_str = f" ({', '.join(flags)})" if flags else ""
        click.echo(
            f"{r.id:<30} {r.eval.name:<20} {r.model.alias:<20} {score_str:<10} {date_str}{flag_str}"
        )

    # Show hidden count
    if hidden_archived or hidden_partial:
        parts = []
        if hidden_archived:
            parts.append(f"{hidden_archived} archived")
        if hidden_partial:
            parts.append(f"{hidden_partial} partial")
        click.echo(f"\n({', '.join(parts)} not shown, use --all to show)")


@results.command("show")
@click.argument("identifier")
@click.option(
    "--all",
    "-a",
    "include_all",
    is_flag=True,
    help="Include partial and archived evals",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def results_show(
    ctx: ProjectContext, identifier: str, include_all: bool, as_json: bool
):
    """Show eval result(s).

    IDENTIFIER can be:
      - A result ID (full or short) → detailed single result
      - A model alias → overview of all results for that model

    Examples:
        isf results show 0105-a7b3
        isf results show e037-final
        isf results show e037-final --all
        isf results show batch-20260105-143022-a7b3 --json
    """
    store = ResultsStore(ctx.results_path)
    result = None

    # Try short ID first
    if _is_short_id(identifier):
        matches = _resolve_short_id(store, identifier)
        if len(matches) == 1:
            result = matches[0]
        elif len(matches) > 1:
            lines = [f"Ambiguous result ID '{identifier}'. Did you mean:"]
            for r in matches:
                lines.append(f"  {r.id} (model: {r.model.alias}, eval: {r.eval.name})")
            raise click.ClickException("\n".join(lines))

    # Try full ID
    if result is None:
        result = store.get(identifier)

    # If we found a single result, show it in detail
    if result is not None:
        _show_single_result(result, as_json)
        return

    # Try as model alias - show overview of all results
    all_model_results = store.list(model=identifier, include_all=True)
    if all_model_results:
        _show_model_overview(identifier, store, include_all, as_json)
        return

    raise click.ClickException(
        f"Not found: {identifier}\nNo result ID or model matches this identifier."
    )


def _show_single_result(result: EvalResult, as_json: bool):
    """Show detailed view of a single result."""
    if as_json:
        click.echo(result.model_dump_json(indent=2))
        return

    # Pretty print
    click.echo(f"Result: {result.id}")
    click.echo(f"Timestamp: {result.timestamp}")
    click.echo()

    # Model info
    click.echo("Model:")
    click.echo(f"  Alias: {result.model.alias}")
    click.echo(f"  Mode: {result.model.mode}")
    if hasattr(result.model, "provider"):
        click.echo(f"  Provider: {result.model.provider}")
    if hasattr(result.model, "training_run"):
        click.echo(f"  Training run: {result.model.training_run}")
        click.echo(f"  Checkpoint: {result.model.checkpoint}")
    if hasattr(result.model, "sysprompt_version"):
        click.echo(f"  Sysprompt: {result.model.sysprompt_version}")
    click.echo()

    # Eval info
    click.echo("Eval:")
    click.echo(f"  Name: {result.eval.name}")
    click.echo(f"  Samples: {result.eval.n_samples}/{result.eval.dataset_size}")
    click.echo(f"  Complete: {result.eval.complete}")
    click.echo()

    # Sampling config
    click.echo("Sampling:")
    click.echo(f"  Temperature: {result.model_sampling.temperature}")
    click.echo(f"  Max tokens: {result.model_sampling.max_tokens}")
    if result.model_sampling.runs_per_sample > 1:
        click.echo(f"  Runs per sample: {result.model_sampling.runs_per_sample}")
    click.echo()

    # Results
    click.echo("Results:")
    score_str = (
        f"{result.results.score:.2%}"
        if result.results.score <= 1
        else f"{result.results.score:.2f}"
    )
    click.echo(f"  Score: {score_str}")
    click.echo(f"  Aggregation: {result.results.aggregation}")
    if result.results.std is not None:
        click.echo(f"  Std: {result.results.std:.4f}")
    if result.results.errors.total > 0:
        click.echo(f"  Errors: {result.results.errors.total}")

    # Artifacts
    if result.artifacts.results_file or result.artifacts.summary_file:
        click.echo()
        click.echo("Artifacts:")
        if result.artifacts.results_file:
            click.echo(f"  Detail: {result.artifacts.results_file}")
        if result.artifacts.summary_file:
            click.echo(f"  Summary: {result.artifacts.summary_file}")

    # Note
    if result.note:
        click.echo()
        click.echo("Note:")
        for line in result.note.split("\n"):
            click.echo(f"  {line}")


def _show_model_overview(
    model: str, store: ResultsStore, include_all: bool, as_json: bool
):
    """Show overview of all results for a model."""
    results = store.list(model=model, include_all=include_all)

    if as_json:
        output = [r.model_dump(mode="json") for r in results]
        click.echo(json.dumps(output, indent=2, default=str))
        return

    if not results:
        # Check if there are hidden results
        all_results = store.list(model=model, include_all=True)
        if all_results:
            click.echo(f"No visible results for {model}.")
            archived = sum(1 for r in all_results if r.archived)
            partial = sum(1 for r in all_results if not r.eval.complete)
            parts = []
            if archived:
                parts.append(f"{archived} archived")
            if partial:
                parts.append(f"{partial} partial")
            if parts:
                click.echo(f"({', '.join(parts)} hidden, use --all to show)")
        return

    # Get model info from first result
    model_info = results[0].model
    base = _short_base_model(model_info)

    click.echo(f"Model: {model}")
    click.echo(f"  {base:<20} {model_info.mode}")
    click.echo()

    # Group by eval configuration
    by_config = {}
    for r in results:
        key = _eval_config_key(r)
        if key not in by_config:
            by_config[key] = []
        by_config[key].append(r)

    # Print each eval group
    for key in sorted(by_config.keys(), key=lambda k: k[0]):  # Sort by eval name
        group_results = sorted(by_config[key], key=lambda x: x.timestamp)
        header = _format_eval_header(key)

        click.echo("-" * 60)
        click.echo()
        click.echo(header)
        click.echo()

        # Show results
        click.echo(f"  {'#':<4}{'Score':<20}")
        for i, r in enumerate(group_results, 1):
            score = r.results.score
            score_str = f"{score:.1%}" if score <= 1 else f"{score:.2f}"
            short = _short_id(r.id)
            flags = []
            if not r.eval.complete:
                flags.append("partial")
            if r.archived:
                flags.append("archived")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            click.echo(f"  {i:<4}{score_str} ({short}){flag_str}")

        click.echo()

    # Show hidden count if not showing all
    if not include_all:
        all_results = store.list(model=model, include_all=True)
        hidden_archived = sum(1 for r in all_results if r.archived)
        hidden_partial = sum(1 for r in all_results if not r.eval.complete)
        if hidden_archived or hidden_partial:
            parts = []
            if hidden_archived:
                parts.append(f"{hidden_archived} archived")
            if hidden_partial:
                parts.append(f"{hidden_partial} partial")
            click.echo(f"({', '.join(parts)} not shown, use --all to show)")


def _short_base_model(model_info) -> str:
    """Extract short name from model spec (e.g., 'Qwen/Qwen3-32B' -> 'qwen3-32b')."""
    base = (
        getattr(model_info, "base_model", None)
        or getattr(model_info, "model_id", None)
        or ""
    )
    if "/" in base:
        return base.split("/")[-1].lower()
    return base.lower() if base else ""


def _short_id(result_id: str) -> str:
    """Extract short ID from full result ID.

    Format: MMDD-XXXX (e.g., 0105-6e65 from batch-20260105-152748-6e65)
    """
    # batch-20260105-152748-6e65 -> 0105-6e65
    parts = result_id.split("-")
    if len(parts) >= 4 and parts[0] == "batch":
        date_part = parts[1]  # 20260105
        suffix = parts[-1]  # 6e65
        return f"{date_part[4:8]}-{suffix}"  # 0105-6e65
    return result_id[:12]  # Fallback: first 12 chars for non-standard IDs


def _resolve_short_id(store: ResultsStore, short_id: str) -> list:
    """Resolve a short ID (MMDD-XXXX) to full result(s).

    Returns list of matching EvalResult objects (usually 1, >1 if ambiguous).
    """
    # If not in short ID format, try as full ID
    if not _is_short_id(short_id):
        result = store.get(short_id)
        return [result] if result else []

    # Scan all results for matching short ID
    matches = []
    for r in store.iter_all():
        if not isinstance(r, EvalResult):
            continue
        if _short_id(r.id) == short_id:
            matches.append(r)

    return matches


def _is_short_id(s: str) -> bool:
    """Check if string looks like a short ID (MMDD-XXXX)."""
    if len(s) == 9 and s[4] == "-":
        # Check MMDD part is numeric and suffix is hex
        mmdd, suffix = s[:4], s[5:]
        return mmdd.isdigit() and all(c in "0123456789abcdef" for c in suffix.lower())
    return False


def _eval_config_key(r) -> tuple:
    """Generate grouping key for eval configuration.

    Groups by: eval name, dataset_sha, judge_prompt_sha, temperature,
    n_samples, runs_per_sample, judge model, judge_sampling, aggregation
    """
    judge_key = None
    if r.judge:
        judge_key = r.judge.alias

    judge_temp = None
    judge_runs = None
    if r.judge_sampling:
        judge_temp = r.judge_sampling.temperature
        judge_runs = r.judge_sampling.judges_per_run

    return (
        r.eval.name,
        r.eval.dataset_sha,
        r.eval.judge_prompt_sha,
        r.model_sampling.temperature,
        r.eval.n_samples,
        r.model_sampling.runs_per_sample,
        judge_key,
        judge_temp,
        judge_runs,
        r.results.aggregation,
    )


def _format_eval_header(key: tuple) -> str:
    """Format eval configuration key as header line."""
    (
        name,
        dataset_sha,
        judge_prompt_sha,
        temp,
        n_samples,
        runs_per_sample,
        judge,
        judge_temp,
        judge_runs,
        agg,
    ) = key

    parts = [name]
    parts.append(f"temp={temp}")
    parts.append(f"n={n_samples}")
    parts.append(f"runs={runs_per_sample}")
    parts.append(f"agg={agg}")

    # Add judge info only if present
    if judge:
        parts.append(f"judge={judge}")
        if judge_temp is not None:
            parts.append(f"j_temp={judge_temp}")
        if judge_runs is not None and judge_runs > 1:
            parts.append(f"j_runs={judge_runs}")

    return " | ".join(parts)


def _get_training_diffs(model1, model2) -> dict:
    """Get differing TrainConfig fields between two trained models."""
    if not (hasattr(model1, "training_config") and hasattr(model2, "training_config")):
        return {}

    cfg1 = model1.training_config
    cfg2 = model2.training_config

    # Fields to compare (skip name, data, log_dir, note)
    compare_fields = [
        "base_model",
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
        "normalize_weights",
        "grad_clip",
    ]

    diffs = {}
    for field in compare_fields:
        v1 = getattr(cfg1, field, None)
        v2 = getattr(cfg2, field, None)
        if v1 != v2:
            diffs[field] = (v1, v2)

    return diffs


@results.command("compare")
@click.argument("args", nargs=-1, required=True)
@click.option("--eval", "eval_name", help="Filter by eval name")
@click.option(
    "--all",
    "-a",
    "include_all",
    is_flag=True,
    help="Include partial and archived evals",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def results_compare(
    ctx: ProjectContext, args: tuple, eval_name: str, include_all: bool, as_json: bool
):
    """Compare results across models or between specific results.

    ARGS can be:
      - Model aliases (e.g., e037-final e038-final) for model comparison
      - Short IDs (e.g., 0105-6e65 0105-abc1) for result-vs-result comparison

    Model comparison groups results by eval configuration and shows deltas.
    Result comparison shows detailed diff between two specific eval runs.

    Examples:
        isf results compare e037-final e038-final
        isf results compare e037-final e038-final --eval gpqa-diamond
        isf results compare 0105-6e65 0105-abc1
    """
    store = ResultsStore(ctx.results_path)

    # Determine comparison mode based on arguments
    if all(_is_short_id(a) for a in args):
        # Result-vs-result comparison
        _compare_results(store, args, as_json)
    elif len(args) == 1:
        # Single model - redirect to show overview
        _show_model_overview(args[0], store, include_all, as_json)
    else:
        # Model-vs-model comparison
        _compare_models(ctx, store, args, eval_name, include_all, as_json)


def _compare_models(
    ctx: ProjectContext,
    store: ResultsStore,
    models: tuple,
    eval_name: str | None,
    include_all: bool,
    as_json: bool,
):
    """Model-vs-model comparison."""
    # Collect all results for each model
    model_results = {}
    for model in models:
        results = store.list(model=model, eval_name=eval_name, include_all=include_all)
        if results:
            model_results[model] = results

    if not model_results:
        click.echo("No results found for the specified models.")
        return

    if as_json:
        output = {
            model: [r.model_dump(mode="json") for r in results]
            for model, results in model_results.items()
        }
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # Get a representative result for each model to show model info
    # Try filtered results first, then fall back to any result for that model
    model_info = {}
    unknown_models = []
    for model in models:
        if model in model_results and model_results[model]:
            model_info[model] = model_results[model][0].model
        else:
            # No filtered results - look for any result with this model alias
            all_for_model = store.list(model=model, include_all=True)
            if all_for_model:
                model_info[model] = all_for_model[0].model
            else:
                unknown_models.append(model)

    # Error if any models can't be resolved
    if unknown_models:
        raise click.ClickException(
            f"Unknown model(s): {', '.join(unknown_models)}\n"
            f"No results found for these models in the results index."
        )

    # Print model header
    click.echo("Models:")
    for model in models:
        info = model_info[model]
        base = _short_base_model(info)
        click.echo(f"  {model:<20} {base:<20} {info.mode}")

    # Show training diffs if comparing exactly 2 trained models
    if len(models) == 2:
        m1, m2 = models
        if m1 in model_info and m2 in model_info:
            diffs = _get_training_diffs(model_info[m1], model_info[m2])
            if diffs:
                click.echo()
                click.echo("Training diffs:")
                for field, (v1, v2) in diffs.items():
                    click.echo(f"  {field}: {v1} vs {v2}")

    click.echo()

    # Group all results by eval configuration
    by_config = {}
    for model in models:
        if model not in model_results:
            continue
        for r in model_results[model]:
            key = _eval_config_key(r)
            if key not in by_config:
                by_config[key] = {m: [] for m in models}
            by_config[key][model].append(r)

    # Print each eval group
    for key in sorted(by_config.keys(), key=lambda k: k[0]):  # Sort by eval name
        results_by_model = by_config[key]
        header = _format_eval_header(key)

        click.echo("-" * 75)
        click.echo()
        click.echo(header)
        click.echo()

        # Build table
        # Sort each model's results by timestamp (oldest first for chronological order)
        for model in models:
            results_by_model[model] = sorted(
                results_by_model[model], key=lambda x: x.timestamp
            )

        # Find max runs across models
        max_runs = max(len(results_by_model[m]) for m in models)

        if max_runs == 0:
            click.echo("  (no results)")
            continue

        # Calculate column widths based on model names
        col_width = max(20, max(len(m) for m in models) + 2)

        # Header row
        click.echo(f"  {'#':<4}" + "".join(f"{m:<{col_width}}" for m in models) + "Δ")

        # Data rows
        for i in range(max_runs):
            row_parts = [f"  {i + 1:<4}"]
            scores = []

            for model in models:
                if i < len(results_by_model[model]):
                    r = results_by_model[model][i]
                    score = r.results.score
                    score_str = f"{score:.1%}" if score <= 1 else f"{score:.2f}"
                    short = _short_id(r.id)
                    cell = f"{score_str} ({short})"
                    scores.append(score)
                else:
                    cell = "-"
                    scores.append(None)
                row_parts.append(f"{cell:<{col_width}}")

            # Calculate delta if comparing 2 models and both have values
            if len(models) == 2 and all(s is not None for s in scores):
                delta = scores[1] - scores[0]
                delta_str = f"{delta:+.1%}" if abs(scores[0]) <= 1 else f"{delta:+.2f}"
                row_parts.append(delta_str)

            click.echo("".join(row_parts))

        click.echo()


def _compare_results(store: ResultsStore, short_ids: tuple, as_json: bool):
    """Result-vs-result comparison."""
    # Resolve short IDs
    resolved = []
    for sid in short_ids:
        matches = _resolve_short_id(store, sid)
        if not matches:
            raise click.ClickException(f"Result not found: {sid}")
        if len(matches) > 1:
            # Ambiguous - show options
            lines = [f"Ambiguous result ID '{sid}'. Did you mean:"]
            for r in matches:
                lines.append(f"  {r.id} (model: {r.model.alias}, eval: {r.eval.name})")
            raise click.ClickException("\n".join(lines))
        resolved.append(matches[0])

    if as_json:
        output = {_short_id(r.id): r.model_dump(mode="json") for r in resolved}
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # Print comparison
    click.echo(f"Comparing: {', '.join(_short_id(r.id) for r in resolved)}")
    click.echo()

    # Models section
    click.echo("Models:")
    for r in resolved:
        sid = _short_id(r.id)
        info = r.model
        base = _short_base_model(info)
        click.echo(f"  {sid}: {info.alias} ({base}, {info.mode})")

    # Training diffs (if comparing 2 trained models)
    if len(resolved) == 2:
        r1, r2 = resolved
        diffs = _get_training_diffs(r1.model, r2.model)
        if diffs:
            click.echo()
            click.echo("Training diffs:")
            for field, (v1, v2) in diffs.items():
                click.echo(f"  {field}: {v1} vs {v2}")

    # Eval section
    click.echo()
    click.echo(f"Eval: {resolved[0].eval.name}")

    # Check if eval configs match
    eval_diffs = {}
    if len(resolved) == 2:
        r1, r2 = resolved
        if r1.eval.dataset_sha != r2.eval.dataset_sha:
            eval_diffs["dataset_sha"] = (r1.eval.dataset_sha, r2.eval.dataset_sha)
        if r1.eval.judge_prompt_sha != r2.eval.judge_prompt_sha:
            eval_diffs["judge_prompt_sha"] = (
                r1.eval.judge_prompt_sha,
                r2.eval.judge_prompt_sha,
            )
        if r1.model_sampling.temperature != r2.model_sampling.temperature:
            eval_diffs["temperature"] = (
                r1.model_sampling.temperature,
                r2.model_sampling.temperature,
            )
        if r1.eval.n_samples != r2.eval.n_samples:
            eval_diffs["n_samples"] = (r1.eval.n_samples, r2.eval.n_samples)

    if eval_diffs:
        click.echo("  (config differs):")
        for field, (v1, v2) in eval_diffs.items():
            click.echo(f"    {field}: {v1} vs {v2}")
    else:
        click.echo("  (config matches)")

    # Results
    click.echo()
    click.echo("Results:")
    for r in resolved:
        sid = _short_id(r.id)
        score = r.results.score
        if score <= 1:
            # Show as percentage with sample count
            n = r.eval.n_samples
            correct = int(score * n + 0.5)
            click.echo(f"  {sid}: {score:.1%} ({correct}/{n})")
        else:
            click.echo(f"  {sid}: {score:.2f}")

    # Delta (if comparing 2)
    if len(resolved) == 2:
        s1, s2 = resolved[0].results.score, resolved[1].results.score
        delta = s2 - s1
        if abs(s1) <= 1:
            click.echo(f"  Δ: {delta:+.1%}")
        else:
            click.echo(f"  Δ: {delta:+.2f}")

    # Files
    click.echo()
    click.echo("Files:")
    for r in resolved:
        sid = _short_id(r.id)
        if r.artifacts.results_file:
            click.echo(f"  {sid}: {r.artifacts.results_file}")


@results.command("archive")
@click.argument("result_id")
@click.option("--note", "-n", help="Reason for archiving")
@click.option("--unarchive", is_flag=True, help="Unarchive instead of archive")
@pass_context
def results_archive(ctx: ProjectContext, result_id: str, note: str, unarchive: bool):
    """Archive or unarchive an eval result.

    Archived results are hidden from list/compare by default.

    Examples:
        isf results archive batch-20260105-143022-a7b3 --note "Test run"
        isf results archive batch-20260105-143022-a7b3 --unarchive
    """
    store = ResultsStore(ctx.results_path)

    # Verify result exists
    result = store.get(result_id)
    if result is None:
        raise click.ClickException(f"Result not found: {result_id}")

    if unarchive:
        success = store.update(result_id, archived=False, archived_note="")
        if success:
            click.echo(f"Unarchived: {result_id}")
        else:
            raise click.ClickException("Failed to update result")
    else:
        success = store.update(result_id, archived=True, archived_note=note or "")
        if success:
            click.echo(f"Archived: {result_id}")
            if note:
                click.echo(f"Note: {note}")
        else:
            raise click.ClickException("Failed to update result")


def main():
    """Entry point for the isf CLI."""
    cli()


if __name__ == "__main__":
    main()
