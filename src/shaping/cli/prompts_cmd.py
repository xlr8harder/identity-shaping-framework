"""Prompts subcommand group."""

import json

import click

from .context import ProjectContext, pass_context


@click.group()
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
    # Lazy import
    from .. import prompts as prompts_module

    try:
        sysprompts, registry_path = prompts_module.build(ctx.project_dir)

        click.echo("Built dev sysprompts:")
        for variant, path in sysprompts.items():
            lines = path.read_text().count("\n")
            click.echo(f"  {variant}: {path.name} ({lines} lines)")

        click.echo(f"\nRebuilt registry: {registry_path}")

        # Show models in registry
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
    # Lazy import
    from .. import prompts as prompts_module

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
    # Lazy import
    from .. import prompts as prompts_module

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

        # Show release aliases for all variants
        prefix = config_info["model_prefix"]
        release_version = config_info["release_version"]
        # Find variants from the release version
        release_variants = []
        for v in versions:
            if v["is_release"]:
                release_variants = v["variants"]
                break
        if release_variants:
            click.echo("Release aliases:")
            for variant in release_variants:
                click.echo(
                    f"  {prefix}-release-{variant} → {prefix}-{release_version}-{variant}"
                )

    except FileNotFoundError as e:
        raise click.ClickException(str(e))
