"""ISF command-line interface.

Provides the `isf` command with subcommands for working with identity-shaping projects.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv


def find_project_root(start: Path) -> Optional[Path]:
    """Find project root by looking for isf.yaml or pyproject.toml."""
    current = start.resolve()
    while current != current.parent:
        if (current / "isf.yaml").exists():
            return current
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return None


def find_registry(project_root: Path) -> Optional[Path]:
    """Find the mq registry config file in a project."""
    # Check common locations
    candidates = [
        project_root / "config" / "registry.json",
        project_root / "mq_registry.yaml",
        project_root / "mq_registry.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


class ProjectContext:
    """Context object holding project configuration."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir.resolve()
        self.registry_path = find_registry(self.project_dir)
        self.env_path = self.project_dir / ".env"

    def load_env(self):
        """Load environment variables from .env file."""
        if self.env_path.exists():
            load_dotenv(self.env_path)
            return True
        return False

    def setup_mq(self):
        """Configure mq to use this project's registry."""
        if self.registry_path:
            from mq import store as mq_store
            mq_store.set_config_path_override(self.registry_path)
            mq_store.load_config()
            return True
        return False


pass_context = click.make_pass_decorator(ProjectContext)


@click.group()
@click.option(
    "--project", "-p",
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
        project_root = project.resolve()

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
        isf mq query aria-v0.9-full "Hello!"
        isf mq batch prompts.jsonl -o responses.jsonl
        isf mq models
    """
    project_ctx: ProjectContext = ctx.obj

    # Set up mq with project registry
    if not project_ctx.setup_mq():
        click.echo(
            f"Warning: No registry found in {project_ctx.project_dir}",
            err=True,
        )

    # Import and run mq CLI
    from mq.cli import main as mq_main

    # Pass through all remaining arguments
    exit_code = mq_main(ctx.args)
    sys.exit(exit_code)


@cli.command()
@pass_context
def info(ctx: ProjectContext):
    """Show project configuration info."""
    click.echo(f"Project directory: {ctx.project_dir}")
    click.echo(f"Registry: {ctx.registry_path or 'not found'}")
    click.echo(f".env file: {'exists' if ctx.env_path.exists() else 'not found'}")

    if ctx.registry_path:
        from mq import store as mq_store
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
    from . import prompts as prompts_module

    try:
        sysprompts, registry_path = prompts_module.build(ctx.project_dir)

        click.echo("Built dev sysprompts:")
        for variant, path in sysprompts.items():
            lines = path.read_text().count('\n')
            click.echo(f"  {variant}: {path.name} ({lines} lines)")

        click.echo(f"\nRebuilt registry: {registry_path}")

        # Show models in registry
        import json
        with open(registry_path) as f:
            registry = json.load(f)
        click.echo(f"Models: {len(registry['models'])}")
        for name in sorted(registry['models'].keys()):
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
    from . import prompts as prompts_module

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

    Shows available versions with their variants and release status.

    Example:
        isf prompts list
    """
    from . import prompts as prompts_module

    try:
        versions = prompts_module.list_versions(ctx.project_dir)

        if not versions:
            click.echo("No versions found.")
            return

        click.echo("Versions:")
        for v in versions:
            marker = " (current)" if v["is_release"] else ""
            variants_str = ", ".join(v["variants"])
            click.echo(f"  {v['name']}{marker}: {variants_str}")

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
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), help="Output directory")
@click.option("--runs", type=int, default=1, help="Runs per sample (for variance)")
@click.option("--concurrency", "-c", type=int, default=20, help="Max concurrent requests")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
@pass_context
def eval_run(ctx: ProjectContext, eval_name: str, model: str, limit: int, seed: int,
             output_dir: Path, runs: int, concurrency: int, quiet: bool):
    """Run an evaluation against a model.

    EVAL_NAME can be:
      - A project eval name (looks in evals/ directory)
      - isf:name for built-in evals (e.g., isf:gpqa-diamond)

    Examples:
        isf eval run my-eval gpt-4o-mini --limit 10
        isf eval run isf:gpqa-diamond qwen3-30b-a3b -n 50
    """
    import asyncio

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
    from .eval import EvalRunner

    async def run():
        runner = EvalRunner(eval_def)
        records, metrics = await runner.run(
            model=model,
            output_dir=output_dir,
            limit=limit,
            seed=seed,
            runs_per_sample=runs,
            concurrency=concurrency,
            quiet=quiet,
        )
        return records, metrics

    try:
        records, metrics = asyncio.run(run())
        if quiet:
            # Print minimal summary for scripting
            if hasattr(metrics, 'accuracy'):
                click.echo(f"{metrics.accuracy:.1%}")
            else:
                click.echo(f"{metrics.mean_score:.2f}")
    except Exception as e:
        raise click.ClickException(str(e))


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
        click.echo(f"\nTo add project evals, create Python files in {ctx.project_dir}/evals/")
        click.echo("with Eval subclasses. See docs for examples.")


# Built-in eval definitions
_BUILTIN_EVALS = {
    "gpqa-diamond": {
        "source": "HuggingFace: fingertap/GPQA-Diamond",
        "hf_dataset": "fingertap/GPQA-Diamond",
        "hf_split": "test",
        "prompt_field": "question",
        "judge_type": "mc",
        "gold_field": "answer",
    },
}


def _get_builtin_eval(name: str):
    """Get a built-in eval definition."""
    from .eval import Eval, MCParser

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

    if config.get("judge_type") == "mc":
        BuiltinEval.judge = MCParser(gold_field=config.get("gold_field", "answer"))

    return BuiltinEval()


def _discover_project_evals(project_dir: Path) -> dict:
    """Discover eval classes in project's evals/ directory.

    Returns dict mapping eval name to {"source": "module:class", "class": EvalClass}
    """
    from .eval import Eval
    import importlib.util

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
                if (isinstance(attr, type)
                    and issubclass(attr, Eval)
                    and attr is not Eval
                    and hasattr(attr, 'name')
                    and attr.name):  # Must have a name
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
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Override output path")
@click.option("--workers", "-w", type=int, help="Number of parallel workers")
@pass_context
def pipeline_run(ctx: ProjectContext, pipeline_name: str, limit: int,
                 output: Path, workers: int):
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
    from .pipeline import run_pipeline

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
        click.echo(f"\nTo add pipelines, create Python files in {ctx.project_dir}/pipelines/")
        click.echo("with TrackedTask subclasses that have a 'name' attribute.")


def _discover_pipelines(project_dir: Path) -> dict:
    """Discover pipeline classes in project's pipelines/ directory.

    Returns dict mapping pipeline name to {"source": "module:class", "class": TaskClass}
    """
    from .pipeline import TrackedTask
    import importlib.util

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
                if (isinstance(attr, type)
                    and issubclass(attr, TrackedTask)
                    and attr is not TrackedTask
                    and hasattr(attr, 'name')
                    and attr.name):  # Must have a name
                    pipeline_name = attr.name
                    discovered[pipeline_name] = {
                        "source": f"{module_name}:{attr_name}",
                        "class": attr,
                        "default_workers": getattr(attr, 'default_workers', 4),
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


def main():
    """Entry point for the isf CLI."""
    cli()


if __name__ == "__main__":
    main()
