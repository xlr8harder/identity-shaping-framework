"""Registry subcommand group."""

import json

import click

from .context import ProjectContext, pass_context


@click.group()
def registry():
    """Manage the model registry.

    The registry is the central index of all models available to the project:
    - Prompted models (base models + identity sysprompts)
    - Trained models (fine-tuned checkpoints)
    - Plain models (base models without sysprompts)

    Commands for listing, inspecting, building, and releasing models.
    """
    pass


@registry.command("list")
@pass_context
def registry_list(ctx: ProjectContext):
    """List all models in the registry.

    Shows all available models with their type and key configuration.

    Example:
        isf registry list
    """
    if not ctx.registry_path.exists():
        raise click.ClickException(
            f"No registry found at {ctx.registry_path}\n"
            "Run 'isf registry build' to create one."
        )

    with open(ctx.registry_path) as f:
        registry = json.load(f)

    models = registry.get("models", {})
    if not models:
        click.echo("Registry is empty. Run 'isf registry build' to populate it.")
        return

    # Categorize models
    trained = {}  # provider == "tinker" and no sysprompt
    prompted = {}  # has sysprompt
    plain = {}  # no sysprompt, not tinker (or tinker base like judge)

    for name, config in models.items():
        provider = config.get("provider", "")
        has_sysprompt = bool(config.get("sysprompt"))
        model_str = config.get("model", "")

        # Trained models: tinker provider with checkpoint path in model string
        if provider == "tinker" and "::" in model_str and not has_sysprompt:
            trained[name] = config
        elif has_sysprompt:
            prompted[name] = config
        else:
            plain[name] = config

    # Display trained models
    if trained:
        click.echo("Trained models:")
        for name in sorted(trained.keys()):
            config = trained[name]
            # Extract base model from model string (base::renderer::checkpoint)
            model_str = config.get("model", "")
            parts = model_str.split("::")
            base = parts[0] if parts else model_str
            # Shorten common prefixes
            if base.startswith("Qwen/"):
                base = base[5:]
            elif base.startswith("deepseek-ai/"):
                base = base[12:]
            click.echo(f"  {name}: {base}")
        click.echo()

    # Display prompted models
    if prompted:
        click.echo("Prompted models:")
        # Group by version pattern if possible
        for name in sorted(prompted.keys()):
            config = prompted[name]
            model = config.get("model", "unknown")
            # Shorten model name
            if "/" in model:
                model = model.split("/")[-1]
            click.echo(f"  {name}: {model}")
        click.echo()

    # Display plain models
    if plain:
        click.echo("Plain models:")
        for name in sorted(plain.keys()):
            config = plain[name]
            model = config.get("model", "unknown")
            # Shorten model name
            if "/" in model:
                model = model.split("/")[-1]
            click.echo(f"  {name}: {model}")

    # Summary
    total = len(models)
    click.echo()
    click.echo(
        f"Total: {total} models "
        f"({len(trained)} trained, {len(prompted)} prompted, {len(plain)} plain)"
    )


@registry.command("show")
@click.argument("model_name")
@pass_context
def registry_show(ctx: ProjectContext, model_name: str):
    """Show details of a model in the registry.

    MODEL_NAME is the name of a model in the registry.

    For trained models, shows checkpoint info. For full training details,
    use 'isf train show EXPERIMENT'.

    Examples:
        isf registry show e007
        isf registry show cubsfan-dev-full
        isf registry show judge
    """
    if not ctx.registry_path.exists():
        raise click.ClickException(
            f"No registry found at {ctx.registry_path}\n"
            "Run 'isf registry build' to create one."
        )

    with open(ctx.registry_path) as f:
        registry = json.load(f)

    models = registry.get("models", {})

    # Case-insensitive lookup
    config = None
    actual_name = model_name
    if model_name in models:
        config = models[model_name]
    else:
        # Try case-insensitive
        for name in models:
            if name.lower() == model_name.lower():
                config = models[name]
                actual_name = name
                break

    if config is None:
        # Suggest similar names
        similar = [n for n in models if model_name.lower() in n.lower()]
        if similar:
            raise click.ClickException(
                f"Model not found: {model_name}\n"
                f"Did you mean: {', '.join(similar[:5])}"
            )
        else:
            raise click.ClickException(
                f"Model not found: {model_name}\n"
                "Use 'isf registry list' to see available models."
            )

    provider = config.get("provider", "unknown")
    model_str = config.get("model", "unknown")
    params = config.get("params", {})
    sysprompt = config.get("sysprompt")

    click.echo(f"Model: {actual_name}")

    # Check if this is a release alias
    alias_target = _get_alias_target(actual_name, ctx.project_dir)
    if alias_target:
        click.echo(f"Alias for: {alias_target}")

    click.echo()

    # Determine model type
    is_trained = provider == "tinker" and "::" in model_str and not sysprompt
    is_prompted = bool(sysprompt)

    if is_trained:
        click.echo("Type: trained")
        # Parse model string: base::renderer::checkpoint
        parts = model_str.split("::")
        if len(parts) >= 3:
            base, renderer, checkpoint = parts[0], parts[1], "::".join(parts[2:])
            click.echo(f"Base model: {base}")
            click.echo(f"Renderer: {renderer}")
            click.echo(f"Checkpoint: {checkpoint}")
        else:
            click.echo(f"Model: {model_str}")

        click.echo()
        click.echo(f"Temperature: {params.get('temperature', 'default')}")

        # Hint at train show for more details
        # Extract experiment name (e007, e007-step100, etc.)
        exp_name = actual_name.split("-")[0] if "-" in actual_name else actual_name
        if exp_name.lower().startswith("e") and exp_name[1:].split("-")[0].isdigit():
            click.echo()
            click.echo(f"For training details: isf train show {exp_name}")

    elif is_prompted:
        click.echo("Type: prompted")
        click.echo(f"Provider: {provider}")
        click.echo(f"Model: {model_str}")
        click.echo(f"Temperature: {params.get('temperature', 'default')}")
        click.echo()

        # Show sysprompt (truncated)
        lines = sysprompt.split("\n")
        click.echo(f"Sysprompt ({len(lines)} lines, {len(sysprompt)} chars):")
        # Show first few lines
        preview_lines = lines[:10]
        for line in preview_lines:
            # Truncate long lines
            if len(line) > 80:
                click.echo(f"  {line[:77]}...")
            else:
                click.echo(f"  {line}")
        if len(lines) > 10:
            click.echo(f"  ... ({len(lines) - 10} more lines)")

    else:
        click.echo("Type: plain")
        click.echo(f"Provider: {provider}")
        click.echo(f"Model: {model_str}")
        click.echo(f"Temperature: {params.get('temperature', 'default')}")


@registry.command("build")
@pass_context
def registry_build(ctx: ProjectContext):
    """Build sysprompts and rebuild the registry.

    Renders templates from identity/templates/ using identity docs,
    outputs to identity/versions/dev/sysprompts/, and rebuilds
    the mq registry with all models.

    Example:
        isf registry build
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


@registry.command("release")
@click.argument("version")
@pass_context
def registry_release(ctx: ProjectContext, version: str):
    """Release current dev prompts as a new version.

    Copies dev sysprompts to a versioned directory and updates
    the release_version in isf.yaml.

    Example:
        isf registry release v0.1
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


@registry.command("prompts")
@pass_context
def registry_prompts(ctx: ProjectContext):
    """List prompt versions.

    Shows available prompt versions with their model names and aliases.

    Example:
        isf registry prompts
    """
    # Lazy import
    from .. import prompts as prompts_module

    try:
        versions, config_info = prompts_module.list_versions(ctx.project_dir)

        if not versions:
            click.echo("No versions found. Run 'isf registry build' first.")
            return

        click.echo("Prompt versions:")
        click.echo()
        for v in versions:
            marker = " <- current release" if v["is_release"] else ""
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
                    f"  {prefix}-release-{variant} -> {prefix}-{release_version}-{variant}"
                )

    except FileNotFoundError as e:
        raise click.ClickException(str(e))


def _get_alias_target(model_name: str, project_dir) -> str | None:
    """Check if model_name is a release alias and return what it points to.

    Returns the target model name if this is a release alias, None otherwise.
    """
    # Release aliases have the pattern: {prefix}-release-{variant}
    if "-release-" not in model_name:
        return None

    try:
        from ..prompts import PromptsConfig

        config = PromptsConfig.from_project(project_dir)

        # Check if this matches the release alias pattern
        expected_prefix = f"{config.model_prefix}-release-"
        if not model_name.startswith(expected_prefix):
            return None

        # Extract variant from the model name
        variant = model_name[len(expected_prefix) :]

        # Return the target: {prefix}-{release_version}-{variant}
        return f"{config.model_prefix}-{config.release_version}-{variant}"

    except Exception:
        return None
