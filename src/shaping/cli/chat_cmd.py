"""Chat command."""

import click

from .context import ProjectContext, pass_context


@click.command()
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

    # Lazy import - this pulls in gradio and torch
    from ..chat import run_chat

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
