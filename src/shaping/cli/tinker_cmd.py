"""CLI commands for Tinker integration.

Provides access to Tinker model catalog and infrastructure.
"""

import click
import json

from .context import ProjectContext, pass_context


@click.group()
def tinker():
    """Tinker training infrastructure commands."""
    pass


@tinker.command("models")
@click.option(
    "--type",
    "-t",
    "training_type",
    type=click.Choice(
        ["base", "instruction", "hybrid", "reasoning", "vision"], case_sensitive=False
    ),
    help="Filter by training type",
)
@click.option(
    "--arch",
    "-a",
    "architecture",
    type=click.Choice(["dense", "moe"], case_sensitive=False),
    help="Filter by architecture (Dense or MoE)",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def models_list(
    ctx: ProjectContext,
    training_type: str | None,
    architecture: str | None,
    as_json: bool,
):
    """List available Tinker base models.

    Shows models available for training with metadata including size,
    training type, architecture, and recommended renderer.

    Note: Model availability may vary by account. This shows models
    currently accessible with your TINKER_API_KEY.

    \b
    Training types:
      Base        - Foundation models for post-training research
      Instruction - Chat-tuned models optimized for fast inference
      Hybrid      - Can operate in thinking or non-thinking mode
      Reasoning   - Always uses chain-of-thought reasoning
      Vision      - Vision-language models (VLMs)

    \b
    Architectures:
      Dense - Standard transformer (all parameters active)
      MoE   - Mixture of Experts (sparse activation, more cost effective)

    Examples:
        isf tinker models                  # List all models
        isf tinker models --type hybrid    # Show hybrid models
        isf tinker models --arch moe       # Show MoE models only
        isf tinker models --json           # Output as JSON
    """
    from ..modeling.tinker.catalog import list_available_models

    try:
        # Get models with optional filtering (filters can be combined)
        models = list_available_models()

        if training_type:
            models = [
                m for m in models if m.training_type.lower() == training_type.lower()
            ]

        if architecture:
            models = [
                m for m in models if m.architecture.lower() == architecture.lower()
            ]

        if not models:
            click.echo("No models found matching filters.")
            return

        if as_json:
            # JSON output
            output = [
                {
                    "name": m.name,
                    "organization": m.organization,
                    "size": m.size,
                    "training_type": m.training_type,
                    "architecture": m.architecture,
                    "renderer": m.renderer,
                    "is_chat": m.is_chat,
                    "is_vl": m.is_vl,
                }
                for m in models
            ]
            click.echo(json.dumps(output, indent=2))
        else:
            # Table output
            click.echo(f"Available Tinker models ({len(models)}):\n")

            # Group by organization
            current_org = None
            for model in models:
                if model.organization != current_org:
                    if current_org is not None:
                        click.echo()
                    click.echo(f"{model.organization}:")
                    current_org = model.organization

                # Format: name (size) - type, arch, renderer
                short_name = model.name.split("/")[-1]
                type_badge = model.training_type
                if model.is_vl:
                    type_badge += " (VL)"

                click.echo(
                    f"  {short_name:40} {model.size:12} {type_badge:12} "
                    f"{model.architecture:6} renderer={model.renderer}"
                )

            click.echo()
            click.echo(
                "Tip: Use --type or --arch to filter. "
                "MoE models are more cost effective."
            )

    except Exception as e:
        error_msg = str(e)
        if "TINKER_API_KEY" in error_msg or "authentication" in error_msg.lower():
            raise click.ClickException(
                "Tinker API authentication failed. Check your TINKER_API_KEY."
            )
        raise click.ClickException(f"Failed to list models: {e}")


@tinker.command("show")
@click.argument("model_name")
@pass_context
def model_show(ctx: ProjectContext, model_name: str):
    """Show details for a specific model.

    MODEL_NAME can be the full name (Qwen/Qwen3-30B-A3B) or just the
    model part (Qwen3-30B-A3B) if unambiguous.

    Examples:
        isf tinker show Qwen/Qwen3-30B-A3B
        isf tinker show Qwen3-30B-A3B
        isf tinker show gpt-oss-20b
    """
    from ..modeling.tinker.catalog import get_model, list_available_models

    model = get_model(model_name)

    if model is None:
        # Try to find matches where input is a substring
        all_models = list_available_models()
        matches = [m for m in all_models if model_name.lower() in m.name.lower()]

        # If exactly one match, use it
        if len(matches) == 1:
            model = matches[0]
        elif len(matches) > 1:
            # Ambiguous - show suggestions
            msg = f"Ambiguous model name: {model_name}"
            msg += "\n\nMatches:\n  " + "\n  ".join(m.name for m in matches[:5])
            raise click.ClickException(msg)
        else:
            raise click.ClickException(f"Model not found: {model_name}")

    click.echo(f"Model: {model.name}")
    click.echo(f"Organization: {model.organization}")
    click.echo(f"Size: {model.size}")
    click.echo(f"Training type: {model.training_type}")
    click.echo(f"Architecture: {model.architecture}")
    click.echo(f"Renderer: {model.renderer}")
    click.echo(f"Chat model: {'yes' if model.is_chat else 'no'}")
    click.echo(f"Vision-language: {'yes' if model.is_vl else 'no'}")

    # Usage hint
    click.echo()
    click.echo("Usage in training config:")
    click.echo(f'  base_model: "{model.name}"')
    click.echo(f'  renderer: "{model.renderer}"')
