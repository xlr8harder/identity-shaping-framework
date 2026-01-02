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


def main():
    """Entry point for the isf CLI."""
    cli()


if __name__ == "__main__":
    main()
