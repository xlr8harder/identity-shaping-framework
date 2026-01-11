"""Results subcommand group."""

import json

import click

from .context import ProjectContext, pass_context


@click.group()
def results():
    """Query and compare eval results.

    Commands for viewing and analyzing evaluation results
    stored in results/index.jsonl.
    """
    pass


@results.command("list")
@click.option("--model", "-m", help="Filter by model alias")
@click.option("--eval", "eval_name", help="Filter by eval name")
@click.option("--training-run", "-t", help="Filter by training run (e.g., E037)")
@click.option(
    "--all",
    "-a",
    "include_all",
    is_flag=True,
    help="Include partial and archived evals",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def results_list(
    ctx: ProjectContext,
    model: str,
    eval_name: str,
    training_run: str,
    include_all: bool,
    as_json: bool,
):
    """List eval results with optional filters.

    By default, excludes partial (incomplete) and archived results.

    Examples:
        isf results list
        isf results list --model e037-final
        isf results list --eval gpqa-diamond --training-run E037
        isf results list --all --json
    """
    from ..results import ResultsStore

    store = ResultsStore(ctx.results_path)

    results = store.list(
        model=model,
        eval_name=eval_name,
        training_run=training_run,
        include_all=include_all,
    )

    # Count hidden results if not showing all
    hidden_archived = 0
    hidden_partial = 0
    if not include_all:
        all_results = store.list(
            model=model,
            eval_name=eval_name,
            training_run=training_run,
            include_all=True,
        )
        for r in all_results:
            if r.archived:
                hidden_archived += 1
            elif not r.eval.complete:
                hidden_partial += 1

    if not results:
        if not ctx.results_path.exists():
            click.echo(f"No results store found at {ctx.results_path}")
            click.echo("Run evaluations with 'isf eval run' to generate results.")
        else:
            click.echo("No matching results found.")
            # Still show hidden count if there are hidden results
            if hidden_archived or hidden_partial:
                parts = []
                if hidden_archived:
                    parts.append(f"{hidden_archived} archived")
                if hidden_partial:
                    parts.append(f"{hidden_partial} partial")
                click.echo(f"({', '.join(parts)} not shown, use --all to show)")
        return

    if as_json:
        output = [r.model_dump(mode="json") for r in results]
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # Table output
    click.echo(f"Found {len(results)} result(s):\n")
    click.echo(f"{'ID':<30} {'Eval':<20} {'Model':<20} {'Score':<10} {'Date'}")
    click.echo("-" * 90)

    for r in sorted(results, key=lambda x: x.timestamp, reverse=True):
        score_str = (
            f"{r.results.score:.2%}"
            if r.results.score <= 1
            else f"{r.results.score:.2f}"
        )
        date_str = r.timestamp.strftime("%Y-%m-%d %H:%M")
        flags = []
        if not r.eval.complete:
            flags.append("partial")
        if r.archived:
            flags.append("archived")
        flag_str = f" ({', '.join(flags)})" if flags else ""
        click.echo(
            f"{r.id:<30} {r.eval.name:<20} {r.model.alias:<20} {score_str:<10} {date_str}{flag_str}"
        )

    # Show hidden count
    if hidden_archived or hidden_partial:
        parts = []
        if hidden_archived:
            parts.append(f"{hidden_archived} archived")
        if hidden_partial:
            parts.append(f"{hidden_partial} partial")
        click.echo(f"\n({', '.join(parts)} not shown, use --all to show)")


@results.command("show")
@click.argument("identifier")
@click.option(
    "--all",
    "-a",
    "include_all",
    is_flag=True,
    help="Include partial and archived evals",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def results_show(
    ctx: ProjectContext, identifier: str, include_all: bool, as_json: bool
):
    """Show eval result(s).

    IDENTIFIER can be:
      - A result ID (full or short) → detailed single result
      - A model alias → overview of all results for that model

    Examples:
        isf results show 0105-a7b3
        isf results show e037-final
        isf results show e037-final --all
        isf results show batch-20260105-143022-a7b3 --json
    """
    from ..results import ResultsStore

    store = ResultsStore(ctx.results_path)
    result = None

    # Try short ID first
    if _is_short_id(identifier):
        matches = _resolve_short_id(store, identifier)
        if len(matches) == 1:
            result = matches[0]
        elif len(matches) > 1:
            lines = [f"Ambiguous result ID '{identifier}'. Did you mean:"]
            for r in matches:
                lines.append(f"  {r.id} (model: {r.model.alias}, eval: {r.eval.name})")
            raise click.ClickException("\n".join(lines))

    # Try full ID
    if result is None:
        result = store.get(identifier)

    # If we found a single result, show it in detail
    if result is not None:
        _show_single_result(result, as_json)
        return

    # Try as model alias - show overview of all results
    all_model_results = store.list(model=identifier, include_all=True)
    if all_model_results:
        _show_model_overview(identifier, store, include_all, as_json)
        return

    raise click.ClickException(
        f"Not found: {identifier}\nNo result ID or model matches this identifier."
    )


def _show_single_result(result, as_json: bool):
    """Show detailed view of a single result."""
    if as_json:
        click.echo(result.model_dump_json(indent=2))
        return

    # Pretty print
    click.echo(f"Result: {result.id}")
    click.echo(f"Timestamp: {result.timestamp}")
    click.echo()

    # Model info
    click.echo("Model:")
    click.echo(f"  Alias: {result.model.alias}")
    click.echo(f"  Mode: {result.model.mode}")
    if hasattr(result.model, "provider"):
        click.echo(f"  Provider: {result.model.provider}")
    if hasattr(result.model, "training_run"):
        click.echo(f"  Training run: {result.model.training_run}")
        click.echo(f"  Checkpoint: {result.model.checkpoint}")
    if hasattr(result.model, "sysprompt_version"):
        click.echo(f"  Sysprompt: {result.model.sysprompt_version}")
    click.echo()

    # Eval info
    click.echo("Eval:")
    click.echo(f"  Name: {result.eval.name}")
    click.echo(f"  Samples: {result.eval.n_samples}/{result.eval.dataset_size}")
    click.echo(f"  Complete: {result.eval.complete}")
    click.echo()

    # Sampling config
    click.echo("Sampling:")
    click.echo(f"  Temperature: {result.model_sampling.temperature}")
    click.echo(f"  Max tokens: {result.model_sampling.max_tokens}")
    if result.model_sampling.runs_per_sample > 1:
        click.echo(f"  Runs per sample: {result.model_sampling.runs_per_sample}")
    click.echo()

    # Results
    click.echo("Results:")
    score_str = (
        f"{result.results.score:.2%}"
        if result.results.score <= 1
        else f"{result.results.score:.2f}"
    )
    click.echo(f"  Score: {score_str}")
    click.echo(f"  Aggregation: {result.results.aggregation}")
    if result.results.std is not None:
        click.echo(f"  Std: {result.results.std:.4f}")
    if result.results.errors.total > 0:
        click.echo(f"  Errors: {result.results.errors.total}")

    # Artifacts
    if result.artifacts.results_file or result.artifacts.summary_file:
        click.echo()
        click.echo("Artifacts:")
        if result.artifacts.results_file:
            click.echo(f"  Detail: {result.artifacts.results_file}")
        if result.artifacts.summary_file:
            click.echo(f"  Summary: {result.artifacts.summary_file}")

    # Note
    if result.note:
        click.echo()
        click.echo("Note:")
        for line in result.note.split("\n"):
            click.echo(f"  {line}")


def _show_model_overview(model: str, store, include_all: bool, as_json: bool):
    """Show overview of all results for a model."""
    results = store.list(model=model, include_all=include_all)

    if as_json:
        output = [r.model_dump(mode="json") for r in results]
        click.echo(json.dumps(output, indent=2, default=str))
        return

    if not results:
        # Check if there are hidden results
        all_results = store.list(model=model, include_all=True)
        if all_results:
            click.echo(f"No visible results for {model}.")
            archived = sum(1 for r in all_results if r.archived)
            partial = sum(1 for r in all_results if not r.eval.complete)
            parts = []
            if archived:
                parts.append(f"{archived} archived")
            if partial:
                parts.append(f"{partial} partial")
            if parts:
                click.echo(f"({', '.join(parts)} hidden, use --all to show)")
        return

    # Get model info from first result
    model_info = results[0].model
    base = _short_base_model(model_info)

    click.echo(f"Model: {model}")
    click.echo(f"  {base:<20} {model_info.mode}")
    click.echo()

    # Group by eval configuration
    by_config = {}
    for r in results:
        key = _eval_config_key(r)
        if key not in by_config:
            by_config[key] = []
        by_config[key].append(r)

    # Print each eval group
    for key in sorted(by_config.keys(), key=lambda k: k[0]):  # Sort by eval name
        group_results = sorted(by_config[key], key=lambda x: x.timestamp)
        header = _format_eval_header(key)

        click.echo("-" * 60)
        click.echo()
        click.echo(header)
        click.echo()

        # Show results
        click.echo(f"  {'#':<4}{'Score':<20}")
        for i, r in enumerate(group_results, 1):
            score = r.results.score
            score_str = f"{score:.1%}" if score <= 1 else f"{score:.2f}"
            short = _short_id(r.id)
            flags = []
            if not r.eval.complete:
                flags.append("partial")
            if r.archived:
                flags.append("archived")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            click.echo(f"  {i:<4}{score_str} ({short}){flag_str}")

        click.echo()

    # Show hidden count if not showing all
    if not include_all:
        all_results = store.list(model=model, include_all=True)
        hidden_archived = sum(1 for r in all_results if r.archived)
        hidden_partial = sum(1 for r in all_results if not r.eval.complete)
        if hidden_archived or hidden_partial:
            parts = []
            if hidden_archived:
                parts.append(f"{hidden_archived} archived")
            if hidden_partial:
                parts.append(f"{hidden_partial} partial")
            click.echo(f"({', '.join(parts)} not shown, use --all to show)")


def _short_base_model(model_info) -> str:
    """Extract short name from model spec (e.g., 'Qwen/Qwen3-32B' -> 'qwen3-32b')."""
    base = (
        getattr(model_info, "base_model", None)
        or getattr(model_info, "model_id", None)
        or ""
    )
    if "/" in base:
        return base.split("/")[-1].lower()
    return base.lower() if base else ""


def _short_id(result_id: str) -> str:
    """Extract short ID from full result ID.

    Format: MMDD-XXXX (e.g., 0105-6e65 from batch-20260105-152748-6e65)
    """
    # batch-20260105-152748-6e65 -> 0105-6e65
    parts = result_id.split("-")
    if len(parts) >= 4 and parts[0] == "batch":
        date_part = parts[1]  # 20260105
        suffix = parts[-1]  # 6e65
        return f"{date_part[4:8]}-{suffix}"  # 0105-6e65
    return result_id[:12]  # Fallback: first 12 chars for non-standard IDs


def _resolve_short_id(store, short_id: str) -> list:
    """Resolve a short ID (MMDD-XXXX) to full result(s).

    Returns list of matching EvalResult objects (usually 1, >1 if ambiguous).
    """
    from ..results import EvalResult

    # If not in short ID format, try as full ID
    if not _is_short_id(short_id):
        result = store.get(short_id)
        return [result] if result else []

    # Scan all results for matching short ID
    matches = []
    for r in store.iter_all():
        if not isinstance(r, EvalResult):
            continue
        if _short_id(r.id) == short_id:
            matches.append(r)

    return matches


def _is_short_id(s: str) -> bool:
    """Check if string looks like a short ID (MMDD-XXXX)."""
    if len(s) == 9 and s[4] == "-":
        # Check MMDD part is numeric and suffix is hex
        mmdd, suffix = s[:4], s[5:]
        return mmdd.isdigit() and all(c in "0123456789abcdef" for c in suffix.lower())
    return False


def _eval_config_key(r) -> tuple:
    """Generate grouping key for eval configuration.

    Groups by: eval name, dataset_sha, judge_prompt_sha, temperature,
    n_samples, runs_per_sample, judge model, judge_sampling, aggregation
    """
    judge_key = None
    if r.judge:
        judge_key = r.judge.alias

    judge_temp = None
    judge_runs = None
    if r.judge_sampling:
        judge_temp = r.judge_sampling.temperature
        judge_runs = r.judge_sampling.judges_per_run

    return (
        r.eval.name,
        r.eval.dataset_sha,
        r.eval.judge_prompt_sha,
        r.model_sampling.temperature,
        r.eval.n_samples,
        r.model_sampling.runs_per_sample,
        judge_key,
        judge_temp,
        judge_runs,
        r.results.aggregation,
    )


def _format_eval_header(key: tuple) -> str:
    """Format eval configuration key as header line."""
    (
        name,
        dataset_sha,
        judge_prompt_sha,
        temp,
        n_samples,
        runs_per_sample,
        judge,
        judge_temp,
        judge_runs,
        agg,
    ) = key

    parts = [name]
    parts.append(f"temp={temp}")
    parts.append(f"n={n_samples}")
    parts.append(f"runs={runs_per_sample}")
    parts.append(f"agg={agg}")

    # Add judge info only if present
    if judge:
        parts.append(f"judge={judge}")
        if judge_temp is not None:
            parts.append(f"j_temp={judge_temp}")
        if judge_runs is not None and judge_runs > 1:
            parts.append(f"j_runs={judge_runs}")

    return " | ".join(parts)


def _get_training_diffs(model1, model2) -> dict:
    """Get differing TrainConfig fields between two trained models."""
    if not (hasattr(model1, "training_config") and hasattr(model2, "training_config")):
        return {}

    cfg1 = model1.training_config
    cfg2 = model2.training_config

    # Fields to compare (skip name, data, log_dir, note)
    compare_fields = [
        "base_model",
        "epochs",
        "batch_size",
        "lora_rank",
        "learning_rate",
        "lr_schedule",
        "max_length",
        "seed",
        "shuffle_seed",
        "test_size",
        "eval_every",
        "save_every",
        "renderer",
        "normalize_weights",
        "grad_clip",
    ]

    diffs = {}
    for field in compare_fields:
        v1 = getattr(cfg1, field, None)
        v2 = getattr(cfg2, field, None)
        if v1 != v2:
            diffs[field] = (v1, v2)

    return diffs


@results.command("compare")
@click.argument("args", nargs=-1, required=True)
@click.option("--eval", "eval_name", help="Filter by eval name")
@click.option(
    "--all",
    "-a",
    "include_all",
    is_flag=True,
    help="Include partial and archived evals",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@pass_context
def results_compare(
    ctx: ProjectContext, args: tuple, eval_name: str, include_all: bool, as_json: bool
):
    """Compare results across models or between specific results.

    ARGS can be:
      - Model aliases (e.g., e037-final e038-final) for model comparison
      - Short IDs (e.g., 0105-6e65 0105-abc1) for result-vs-result comparison

    Model comparison groups results by eval configuration and shows deltas.
    Result comparison shows detailed diff between two specific eval runs.

    Examples:
        isf results compare e037-final e038-final
        isf results compare e037-final e038-final --eval gpqa-diamond
        isf results compare 0105-6e65 0105-abc1
    """
    from ..results import ResultsStore

    store = ResultsStore(ctx.results_path)

    # Determine comparison mode based on arguments
    if all(_is_short_id(a) for a in args):
        # Result-vs-result comparison
        _compare_results(store, args, as_json)
    elif len(args) == 1:
        # Single model - redirect to show overview
        _show_model_overview(args[0], store, include_all, as_json)
    else:
        # Model-vs-model comparison
        _compare_models(ctx, store, args, eval_name, include_all, as_json)


def _compare_models(
    ctx: ProjectContext,
    store,
    models: tuple,
    eval_name: str | None,
    include_all: bool,
    as_json: bool,
):
    """Model-vs-model comparison."""
    # Collect all results for each model
    model_results = {}
    for model in models:
        results = store.list(model=model, eval_name=eval_name, include_all=include_all)
        if results:
            model_results[model] = results

    if not model_results:
        click.echo("No results found for the specified models.")
        return

    if as_json:
        output = {
            model: [r.model_dump(mode="json") for r in results]
            for model, results in model_results.items()
        }
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # Get a representative result for each model to show model info
    # Try filtered results first, then fall back to any result for that model
    model_info = {}
    unknown_models = []
    for model in models:
        if model in model_results and model_results[model]:
            model_info[model] = model_results[model][0].model
        else:
            # No filtered results - look for any result with this model alias
            all_for_model = store.list(model=model, include_all=True)
            if all_for_model:
                model_info[model] = all_for_model[0].model
            else:
                unknown_models.append(model)

    # Error if any models can't be resolved
    if unknown_models:
        raise click.ClickException(
            f"Unknown model(s): {', '.join(unknown_models)}\n"
            f"No results found for these models in the results index."
        )

    # Print model header
    click.echo("Models:")
    for model in models:
        info = model_info[model]
        base = _short_base_model(info)
        click.echo(f"  {model:<20} {base:<20} {info.mode}")

    # Show training diffs if comparing exactly 2 trained models
    if len(models) == 2:
        m1, m2 = models
        if m1 in model_info and m2 in model_info:
            diffs = _get_training_diffs(model_info[m1], model_info[m2])
            if diffs:
                click.echo()
                click.echo("Training diffs:")
                for field, (v1, v2) in diffs.items():
                    click.echo(f"  {field}: {v1} vs {v2}")

    click.echo()

    # Group all results by eval configuration
    by_config = {}
    for model in models:
        if model not in model_results:
            continue
        for r in model_results[model]:
            key = _eval_config_key(r)
            if key not in by_config:
                by_config[key] = {m: [] for m in models}
            by_config[key][model].append(r)

    # Print each eval group
    for key in sorted(by_config.keys(), key=lambda k: k[0]):  # Sort by eval name
        results_by_model = by_config[key]
        header = _format_eval_header(key)

        click.echo("-" * 75)
        click.echo()
        click.echo(header)
        click.echo()

        # Build table
        # Sort each model's results by timestamp (oldest first for chronological order)
        for model in models:
            results_by_model[model] = sorted(
                results_by_model[model], key=lambda x: x.timestamp
            )

        # Find max runs across models
        max_runs = max(len(results_by_model[m]) for m in models)

        if max_runs == 0:
            click.echo("  (no results)")
            continue

        # Calculate column widths based on model names
        col_width = max(20, max(len(m) for m in models) + 2)

        # Header row
        click.echo(f"  {'#':<4}" + "".join(f"{m:<{col_width}}" for m in models) + "Δ")

        # Data rows
        for i in range(max_runs):
            row_parts = [f"  {i + 1:<4}"]
            scores = []

            for model in models:
                if i < len(results_by_model[model]):
                    r = results_by_model[model][i]
                    score = r.results.score
                    score_str = f"{score:.1%}" if score <= 1 else f"{score:.2f}"
                    short = _short_id(r.id)
                    cell = f"{score_str} ({short})"
                    scores.append(score)
                else:
                    cell = "-"
                    scores.append(None)
                row_parts.append(f"{cell:<{col_width}}")

            # Calculate delta if comparing 2 models and both have values
            if len(models) == 2 and all(s is not None for s in scores):
                delta = scores[1] - scores[0]
                delta_str = f"{delta:+.1%}" if abs(scores[0]) <= 1 else f"{delta:+.2f}"
                row_parts.append(delta_str)

            click.echo("".join(row_parts))

        click.echo()


def _compare_results(store, short_ids: tuple, as_json: bool):
    """Result-vs-result comparison."""
    # Resolve short IDs
    resolved = []
    for sid in short_ids:
        matches = _resolve_short_id(store, sid)
        if not matches:
            raise click.ClickException(f"Result not found: {sid}")
        if len(matches) > 1:
            # Ambiguous - show options
            lines = [f"Ambiguous result ID '{sid}'. Did you mean:"]
            for r in matches:
                lines.append(f"  {r.id} (model: {r.model.alias}, eval: {r.eval.name})")
            raise click.ClickException("\n".join(lines))
        resolved.append(matches[0])

    if as_json:
        output = {_short_id(r.id): r.model_dump(mode="json") for r in resolved}
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # Print comparison
    click.echo(f"Comparing: {', '.join(_short_id(r.id) for r in resolved)}")
    click.echo()

    # Models section
    click.echo("Models:")
    for r in resolved:
        sid = _short_id(r.id)
        info = r.model
        base = _short_base_model(info)
        click.echo(f"  {sid}: {info.alias} ({base}, {info.mode})")

    # Training diffs (if comparing 2 trained models)
    if len(resolved) == 2:
        r1, r2 = resolved
        diffs = _get_training_diffs(r1.model, r2.model)
        if diffs:
            click.echo()
            click.echo("Training diffs:")
            for field, (v1, v2) in diffs.items():
                click.echo(f"  {field}: {v1} vs {v2}")

    # Eval section
    click.echo()
    click.echo(f"Eval: {resolved[0].eval.name}")

    # Check if eval configs match
    eval_diffs = {}
    if len(resolved) == 2:
        r1, r2 = resolved
        if r1.eval.dataset_sha != r2.eval.dataset_sha:
            eval_diffs["dataset_sha"] = (r1.eval.dataset_sha, r2.eval.dataset_sha)
        if r1.eval.judge_prompt_sha != r2.eval.judge_prompt_sha:
            eval_diffs["judge_prompt_sha"] = (
                r1.eval.judge_prompt_sha,
                r2.eval.judge_prompt_sha,
            )
        if r1.model_sampling.temperature != r2.model_sampling.temperature:
            eval_diffs["temperature"] = (
                r1.model_sampling.temperature,
                r2.model_sampling.temperature,
            )
        if r1.eval.n_samples != r2.eval.n_samples:
            eval_diffs["n_samples"] = (r1.eval.n_samples, r2.eval.n_samples)

    if eval_diffs:
        click.echo("  (config differs):")
        for field, (v1, v2) in eval_diffs.items():
            click.echo(f"    {field}: {v1} vs {v2}")
    else:
        click.echo("  (config matches)")

    # Results
    click.echo()
    click.echo("Results:")
    for r in resolved:
        sid = _short_id(r.id)
        score = r.results.score
        if score <= 1:
            # Show as percentage with sample count
            n = r.eval.n_samples
            correct = int(score * n + 0.5)
            click.echo(f"  {sid}: {score:.1%} ({correct}/{n})")
        else:
            click.echo(f"  {sid}: {score:.2f}")

    # Delta (if comparing 2)
    if len(resolved) == 2:
        s1, s2 = resolved[0].results.score, resolved[1].results.score
        delta = s2 - s1
        if abs(s1) <= 1:
            click.echo(f"  Δ: {delta:+.1%}")
        else:
            click.echo(f"  Δ: {delta:+.2f}")

    # Files
    click.echo()
    click.echo("Files:")
    for r in resolved:
        sid = _short_id(r.id)
        if r.artifacts.results_file:
            click.echo(f"  {sid}: {r.artifacts.results_file}")


@results.command("archive")
@click.argument("result_id")
@click.option("--note", "-n", help="Reason for archiving")
@click.option("--unarchive", is_flag=True, help="Unarchive instead of archive")
@pass_context
def results_archive(ctx: ProjectContext, result_id: str, note: str, unarchive: bool):
    """Archive or unarchive an eval result.

    Archived results are hidden from list/compare by default.

    Examples:
        isf results archive batch-20260105-143022-a7b3 --note "Test run"
        isf results archive batch-20260105-143022-a7b3 --unarchive
    """
    from ..results import ResultsStore

    store = ResultsStore(ctx.results_path)

    # Verify result exists
    result = store.get(result_id)
    if result is None:
        raise click.ClickException(f"Result not found: {result_id}")

    if unarchive:
        success = store.update(result_id, archived=False, archived_note="")
        if success:
            click.echo(f"Unarchived: {result_id}")
        else:
            raise click.ClickException("Failed to update result")
    else:
        success = store.update(result_id, archived=True, archived_note=note or "")
        if success:
            click.echo(f"Archived: {result_id}")
            if note:
                click.echo(f"Note: {note}")
        else:
            raise click.ClickException("Failed to update result")
