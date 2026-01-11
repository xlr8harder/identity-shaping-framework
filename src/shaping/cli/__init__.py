"""ISF command-line interface.

Provides the `isf` command with subcommands for working with identity-shaping projects.

This module uses lazy imports for heavy dependencies (torch, gradio, tinker)
to keep CLI startup fast. Only the specific command being run loads its dependencies.
"""

import sys
from pathlib import Path

import click
from mq.cli import main as mq_main

from .context import ProjectContext, find_project_root

# Import command groups - these are lightweight, just click decorators
from .registry_cmd import registry
from .eval_cmd import eval
from .pipeline_cmd import pipeline
from .train_cmd import train
from .results_cmd import results
from .chat_cmd import chat


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


# Register command groups
cli.add_command(registry)
cli.add_command(eval)
cli.add_command(pipeline)
cli.add_command(train)
cli.add_command(results)
cli.add_command(chat)


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

    # Build args with --config pointing to project registry
    if not project_ctx.registry_path.exists():
        raise click.ClickException(
            f"No registry.json found in {project_ctx.project_dir}"
        )

    args = ["--config", str(project_ctx.registry_path)] + list(ctx.args)
    exit_code = mq_main(args)
    sys.exit(exit_code)


@cli.command()
@click.pass_obj
def status(ctx: ProjectContext):
    """Show project status overview.

    Quick view of pipelines, datasets, and experiments.
    """
    click.echo(f"Project: {ctx.project_dir}")
    click.echo()

    # Prompt versions
    _show_prompt_status(ctx)

    # Pipeline status
    _show_pipeline_status(ctx)

    # Dataset status
    _show_dataset_status(ctx)

    # Recent experiments
    _show_experiment_status(ctx)


def _show_prompt_status(ctx: ProjectContext):
    """Show prompt version status."""
    import json
    import re

    if not ctx.registry_path.exists():
        click.echo("Prompts: registry not found (run: isf registry build)")
        click.echo()
        return

    with open(ctx.registry_path) as f:
        registry = json.load(f)

    # Find versions by looking for -vX.Y- pattern in model names
    versions = []
    for name in registry.get("models", {}).keys():
        match = re.search(r"-(v\d+\.\d+)-", name)
        if match:
            versions.append(match.group(1))

    versions = sorted(set(versions), reverse=True)

    click.echo("Prompts:")
    if versions:
        click.echo(f"  release: {versions[0]}")
        if len(versions) > 1:
            click.echo(f"  older: {', '.join(versions[1:])}")
    else:
        click.echo("  release: none (run: isf registry release vX.Y)")
    click.echo()


def _show_pipeline_status(ctx: ProjectContext):
    """Show pipeline staleness status."""
    from .pipeline_cmd import _discover_pipelines

    # Set up mq for model sysprompt lookups in staleness check
    ctx.setup_mq()

    pipelines_info = _discover_pipelines(ctx.project_dir)
    if not pipelines_info:
        return

    pipelines = []
    for name, info in sorted(pipelines_info.items()):
        pipeline_class = info["class"]
        try:
            staleness = pipeline_class.check_staleness()
            count = staleness.get("record_count", 0)
            stale = staleness.get("stale", False)
            status = "STALE" if stale else "ok"
            pipelines.append((name, status, count))
        except Exception:
            pipelines.append((name, "ERROR", 0))

    if pipelines:
        click.echo("Pipelines:")
        for name, status, count in pipelines:
            if status == "ok":
                click.echo(f"  {name}: {count} samples")
            else:
                click.echo(f"  {name}: {status}")
        click.echo()


def _show_dataset_status(ctx: ProjectContext):
    """Show dataset staleness status."""
    from ..training.prep import DatasetRecipe, check_staleness, list_recipes

    data_dir = ctx.project_dir / "training" / "data"
    if not data_dir.exists():
        return

    recipes = list_recipes(data_dir)
    if not recipes:
        return

    ctx.setup_mq()

    datasets = []
    for recipe_path in recipes:
        name = recipe_path.stem
        try:
            recipe = DatasetRecipe.load(recipe_path)
            output_file = recipe.get_output_file()

            if not output_file.exists():
                datasets.append((name, "NOT PREPARED", 0))
            else:
                staleness = check_staleness(recipe, ctx.project_dir)
                stale = staleness.get("stale", False)
                count = sum(1 for _ in open(output_file))
                status = "STALE" if stale else "ok"
                datasets.append((name, status, count))
        except Exception:
            datasets.append((name, "ERROR", 0))

    if datasets:
        click.echo("Datasets:")
        for name, status, count in datasets:
            if status == "ok":
                click.echo(f"  {name}: {count} samples")
            else:
                click.echo(f"  {name}: {status}")
        click.echo()


def _show_experiment_status(ctx: ProjectContext):
    """Show recent experiments."""
    import json

    logs_dir = ctx.project_dir / "training" / "logs"
    if not logs_dir.exists():
        return

    experiments = []
    for exp_dir in sorted(logs_dir.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue

        config_file = exp_dir / "train-config.json"
        if not config_file.exists():
            continue

        try:
            with open(config_file) as f:
                config = json.load(f)

            base_model = config.get("base_model", "?")
            # Shorten model name
            if "/" in base_model:
                base_model = base_model.split("/")[-1]

            data = config.get("data", "?")
            # Extract dataset name from path
            if "/" in data:
                data = Path(data).stem

            epochs = config.get("epochs", "?")

            experiments.append((exp_dir.name, base_model, data, epochs))
        except Exception:
            pass

        if len(experiments) >= 3:
            break

    if experiments:
        click.echo("Recent experiments:")
        for name, model, data, epochs in experiments:
            click.echo(f"  {name}: {model}, {data}, {epochs}ep")
        click.echo()


def main():
    """Entry point for the isf CLI."""
    cli()


if __name__ == "__main__":
    main()
