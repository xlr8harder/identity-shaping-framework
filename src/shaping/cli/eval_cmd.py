"""Eval subcommand group."""

import asyncio
import importlib.util
import sys
from pathlib import Path

import click

from .context import ProjectContext, pass_context


@click.group()
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
    # Lazy import
    from ..eval import EvalRunner

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
    from ..eval.judges import LLMJudge
    from ..results import ResultsStore, build_eval_result

    # Determine score: use mean_score for rubric-based evals, accuracy otherwise
    # Rubric-based evals have score_distribution populated
    if metrics.score_distribution:
        score = metrics.mean_score
    else:
        score = metrics.accuracy

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
    from ..eval import Eval, MCParser

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
    from ..eval import Eval

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
