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
from .prompts_cmd import prompts
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
cli.add_command(prompts)
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
def info(ctx: ProjectContext):
    """Show project configuration info."""
    from mq import store as mq_store

    click.echo(f"Project directory: {ctx.project_dir}")
    click.echo(
        f"Registry: {ctx.registry_path if ctx.registry_path.exists() else 'not found'}"
    )
    click.echo(f".env file: {'exists' if ctx.env_path.exists() else 'not found'}")

    if ctx.registry_path.exists():
        ctx.setup_mq()
        models = list(mq_store.list_models())
        click.echo(f"Models registered: {len(models)}")


def main():
    """Entry point for the isf CLI."""
    cli()


if __name__ == "__main__":
    main()
