"""Pipeline subcommand group."""

import importlib.util
import sys
from pathlib import Path

import click

from .context import ProjectContext, pass_context


@click.group()
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
@click.option(
    "--annotate/--no-annotate",
    default=None,
    help="Include full provenance (default) or strip to minimal training format",
)
@pass_context
def pipeline_run(
    ctx: ProjectContext,
    pipeline_name: str,
    limit: int,
    output: Path,
    workers: int,
    annotate: bool,
):
    """Run a pipeline by name.

    PIPELINE_NAME should match a Pipeline subclass with that name
    in the pipelines/ directory.

    By default, outputs AnnotatedTrainingSample with full provenance
    (input data, inference steps, timestamps). Use --no-annotate for
    training-ready output with just id and messages.

    Examples:
        isf pipeline run wildchat-training
        isf pipeline run wildchat-training --limit 10
        isf pipeline run wildchat-training -n 10 -o test.jsonl
        isf pipeline run wildchat-training --no-annotate  # Training format
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
                f"with Pipeline subclasses that have a 'name' attribute."
            )

    # Run the pipeline
    try:
        instance = pipeline_def()
        if workers is not None:
            instance.workers = workers
        instance.execute(limit=limit, output_file=output, annotated=annotate)
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
            workers = info.get("workers", 50)
            click.echo(f"  {name}: {source} (workers: {workers})")
    else:
        click.echo("No pipelines found.")
        click.echo(
            f"\nTo add pipelines, create Python files in {ctx.project_dir}/pipelines/"
        )
        click.echo("with Pipeline subclasses that have a 'name' attribute.")


@pipeline.command("status")
@pass_context
def pipeline_status(ctx: ProjectContext):
    """Check pipeline staleness.

    Shows whether each pipeline's output is up-to-date with its dependencies.

    Example:
        isf pipeline status

        identity-augmentation: STALE
          - Pipeline code changed
          - File 'narrative_doc' content changed

        wildchat-training: CURRENT (1000 samples)

        test-pipeline: PARTIAL (10 samples)
          - Partial run (10 samples, run without --limit for full data)
    """
    # Set up mq with project registry (needed for model dep sysprompt lookups)
    ctx.setup_mq()

    pipelines = _discover_pipelines(ctx.project_dir)

    if not pipelines:
        click.echo("No pipelines found.")
        click.echo(
            f"\nTo add pipelines, create Python files in {ctx.project_dir}/pipelines/"
        )
        return

    for name, info in sorted(pipelines.items()):
        pipeline_class = info["class"]

        try:
            staleness = pipeline_class.check_staleness()
            record_count = staleness.get("record_count", 0)
            is_partial = staleness.get("partial", False)

            if staleness["stale"]:
                # Show PARTIAL for partial-only staleness, STALE for other issues
                if is_partial and len(staleness["reasons"]) == 1:
                    click.echo(f"{name}: PARTIAL ({record_count} samples)")
                else:
                    click.echo(f"{name}: STALE")
                for reason in staleness["reasons"]:
                    click.echo(f"  - {reason}")
            else:
                count_str = f" ({record_count} samples)" if record_count else ""
                click.echo(f"{name}: CURRENT{count_str}")

        except Exception as e:
            click.echo(f"{name}: ERROR ({e})")

        click.echo()


def _discover_pipelines(project_dir: Path) -> dict:
    """Discover Pipeline subclasses in project's pipelines/ directory.

    Returns dict mapping pipeline name to:
        - "source": "module:class"
        - "class": the Pipeline subclass
        - "workers": int
    """
    # Lazy import
    from ..pipeline import Pipeline

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

            # Find all Pipeline subclasses with a name
            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                if (
                    isinstance(attr, type)
                    and issubclass(attr, Pipeline)
                    and attr is not Pipeline
                    and hasattr(attr, "name")
                    and attr.name
                ):
                    pipeline_name = attr.name
                    discovered[pipeline_name] = {
                        "source": f"{module_name}:{attr_name}",
                        "class": attr,
                        "workers": getattr(attr, "workers", 50),
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
            "workers": info.get("workers", 50),
        }
    return pipelines
