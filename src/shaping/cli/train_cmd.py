"""Train subcommand group."""

import json
from pathlib import Path

import click

from .context import ProjectContext, pass_context


@click.group()
def train():
    """Run training experiments.

    Commands for running SFT training with tinker_cookbook.
    Training configs are YAML files in training/configs/.
    """
    pass


@train.command("run")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
# Per-run options
@click.option(
    "--data", "-d", type=click.Path(path_type=Path), help="Training data file"
)
@click.option("--name", "-n", help="Experiment name (auto-generated if not provided)")
@click.option("--epochs", "-e", type=int, help="Number of epochs")
# Model options
@click.option("--model", "-m", help="Base model (overrides config)")
@click.option("--renderer", help="Override renderer (e.g., qwen3, deepseekv3)")
# Hyperparameters
@click.option("--lr", type=float, help="Learning rate")
@click.option(
    "--lr-schedule",
    type=click.Choice(["constant", "linear", "cosine"]),
    help="LR schedule",
)
@click.option("--batch-size", type=int, help="Batch size")
@click.option("--lora-rank", type=int, help="LoRA rank")
@click.option("--max-length", type=int, help="Max sequence length")
# Reproducibility
@click.option("--seed", type=int, help="Random seed")
@click.option("--shuffle-seed", type=int, help="Separate shuffle seed")
# Evaluation and checkpointing
@click.option("--test-size", type=int, help="Hold out N examples for validation")
@click.option("--eval-every", type=int, help="Evaluate every N steps (0 = disabled)")
@click.option("--save-every", type=int, help="Save checkpoint every N steps")
# Training tweaks
@click.option("--grad-clip", type=float, help="Gradient clipping norm")
@click.option(
    "--normalize-weights",
    is_flag=True,
    default=None,
    help="Normalize per-example weights",
)
@click.option(
    "--optim-metrics-every", type=int, help="Log optimizer metrics every N steps"
)
# Annotation
@click.option("--note", help="Free-form note about this experiment")
# Control
@click.option("--force", "-f", is_flag=True, help="Overwrite existing experiment")
@click.option("--allow-stale", is_flag=True, help="Allow training on stale dataset")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@pass_context
def train_run(
    ctx: ProjectContext,
    config_path: Path,
    data: Path,
    name: str,
    epochs: int,
    model: str,
    renderer: str,
    lr: float,
    lr_schedule: str,
    batch_size: int,
    lora_rank: int,
    max_length: int,
    seed: int,
    shuffle_seed: int,
    test_size: int,
    eval_every: int,
    save_every: int,
    grad_clip: float,
    normalize_weights: bool,
    optim_metrics_every: int,
    note: str,
    force: bool,
    allow_stale: bool,
    verbose: bool,
):
    """Run a training experiment from a config file.

    CONFIG_PATH is a YAML file with base hyperparameters.
    CLI options override values in the config file.

    Example config (training/config.yaml):

    \b
        base_model: Qwen/Qwen3-30B-A3B
        batch_size: 32
        lora_rank: 32
        max_length: 8192
        epochs: 2

    Examples:
        isf train run training/config.yaml --data train.jsonl
        isf train run training/config.yaml -d train.jsonl -e 3
        isf train run training/config.yaml -d train.jsonl --name E050
        isf train run training/config.yaml -d train.jsonl --seed 123 --lr 0.0001
    """
    # Lazy imports - these pull in tinker/torch
    from ..training import build_config, run_training

    try:
        # Build overrides from CLI options
        overrides = {
            "data": str(data) if data else None,
            "name": name,
            "epochs": epochs,
            "base_model": model,
            "renderer": renderer,
            "learning_rate": lr,
            "lr_schedule": lr_schedule,
            "batch_size": batch_size,
            "lora_rank": lora_rank,
            "max_length": max_length,
            "seed": seed,
            "shuffle_seed": shuffle_seed,
            "test_size": test_size,
            "eval_every": eval_every,
            "save_every": save_every,
            "grad_clip": grad_clip,
            "normalize_weights": normalize_weights,
            "optim_metrics_every": optim_metrics_every,
            "note": note,
        }

        config = build_config(config_path, **overrides)

        # Check dataset staleness if using dataset reference
        dataset_name = _get_dataset_from_config(config_path)
        if dataset_name and not allow_stale:
            ctx.setup_mq()
            staleness_issues = _check_dataset_staleness(dataset_name, ctx.project_dir)
            if staleness_issues:
                raise click.ClickException(
                    f"Dataset '{dataset_name}' has staleness issues:\n"
                    + "\n".join(f"  - {issue}" for issue in staleness_issues)
                    + "\n\nTo fix:\n"
                    + f"  isf train data prep {dataset_name}\n"
                    + "\nOr use --allow-stale to proceed anyway."
                )

        click.echo(f"Experiment: {config.name}")
        log_path = run_training(config, force=force, verbose=verbose)
        click.echo(f"\nExperiment complete: {log_path}")

        # Auto-register checkpoint in mq registry
        from ..prompts import PromptsConfig, build_registry

        try:
            prompts_config = PromptsConfig.from_project(ctx.project_dir)
            build_registry(prompts_config)
            exp_name = log_path.name.lower()
            click.echo(f"Registered checkpoint: {exp_name}")
        except Exception as e:
            click.echo(f"Warning: Could not auto-register checkpoint: {e}")
    except (ImportError, ValueError, FileNotFoundError, FileExistsError) as e:
        raise click.ClickException(str(e))
    except KeyboardInterrupt:
        raise click.Abort()


@train.command("list")
@pass_context
def train_list(ctx: ProjectContext):
    """List training configs and experiments.

    Shows available config files and existing experiments.
    """
    training_dir = ctx.project_dir / "training"
    logs_dir = training_dir / "logs"

    # Check for config files (training/config.yaml or training/configs/*.yaml)
    default_config = training_dir / "config.yaml"
    configs_dir = training_dir / "configs"

    config_files = []
    if default_config.exists():
        config_files.append(default_config)
    if configs_dir.exists():
        config_files.extend(sorted(configs_dir.glob("*.yaml")))
        config_files.extend(sorted(configs_dir.glob("*.yml")))

    if config_files:
        click.echo("Training configs:")
        for path in config_files:
            rel_path = path.relative_to(ctx.project_dir)
            click.echo(f"  {rel_path}")
        click.echo()
    else:
        click.echo("No training configs found.")
        click.echo("  Create training/config.yaml with base hyperparameters.")
        click.echo()

    # List experiment logs
    if logs_dir.exists():
        exp_dirs = sorted([d for d in logs_dir.iterdir() if d.is_dir()])
        if exp_dirs:
            click.echo(f"Experiments ({logs_dir}):")
            for exp in exp_dirs[
                -20:
            ]:  # Show most recent 20 (sorted ascending, so take last)
                # Check for config and extract note
                isf_config = exp / "train-config.json"
                tinker_config = exp / "config.json"
                note_preview = ""

                if isf_config.exists():
                    try:
                        with open(isf_config) as f:
                            cfg = json.load(f)
                        if cfg.get("note"):
                            # First line, truncated
                            first_line = cfg["note"].split("\n")[0][:40]
                            note_preview = f" - {first_line}"
                    except (OSError, json.JSONDecodeError, KeyError):
                        pass  # Show nothing if config unreadable
                    marker = "*"
                elif tinker_config.exists():
                    marker = " "
                else:
                    marker = " "

                click.echo(f"  {marker} {exp.name}{note_preview}")
            if len(exp_dirs) > 20:
                click.echo(f"  ({len(exp_dirs) - 20} older experiments not shown)")
    else:
        click.echo("No experiments yet.")


@train.command("show")
@click.argument("experiment")
@pass_context
def train_show(ctx: ProjectContext, experiment: str):
    """Show details of a training experiment.

    EXPERIMENT is the experiment name (directory name in training/logs/).

    Examples:
        isf train show e050-baseline
        isf train show E039
    """
    from ..results import ResultsStore

    logs_dir = ctx.project_dir / "training" / "logs"
    exp_dir = logs_dir / experiment

    # Case-insensitive lookup: try exact, then uppercase, then find matching
    if not exp_dir.exists():
        # Try uppercase (E007 vs e007)
        exp_dir = logs_dir / experiment.upper()
    if not exp_dir.exists():
        # Try lowercase
        exp_dir = logs_dir / experiment.lower()
    if not exp_dir.exists() and logs_dir.exists():
        # Search for case-insensitive match
        exp_lower = experiment.lower()
        for d in logs_dir.iterdir():
            if d.is_dir() and d.name.lower() == exp_lower:
                exp_dir = d
                break
    if not exp_dir.exists():
        raise click.ClickException(f"Experiment not found: {experiment}")

    # Use actual directory name for display consistency
    experiment = exp_dir.name
    click.echo(f"Experiment: {experiment}")
    click.echo(f"Directory: {exp_dir}")
    click.echo()

    # Show config - try ISF format first, then tinker format
    isf_config = exp_dir / "train-config.json"
    tinker_config = exp_dir / "config.json"

    if isf_config.exists():
        with open(isf_config) as f:
            config = json.load(f)
        click.echo("Configuration (ISF):")
        click.echo(f"  Model: {config.get('base_model', 'unknown')}")
        click.echo(f"  Data: {config.get('data', 'unknown')}")
        click.echo(f"  Epochs: {config.get('epochs', 'unknown')}")
        click.echo(f"  Batch size: {config.get('batch_size', 'unknown')}")
        click.echo(f"  LoRA rank: {config.get('lora_rank', 'unknown')}")
        if config.get("note"):
            click.echo()
            click.echo("Note:")
            for line in config["note"].split("\n"):
                click.echo(f"  {line}")
        click.echo()
    elif tinker_config.exists():
        with open(tinker_config) as f:
            config = json.load(f)
        click.echo("Configuration (Tinker):")
        click.echo(f"  Model: {config.get('model_name', 'unknown')}")
        ds = config.get("dataset_builder", {})
        click.echo(f"  Data: {ds.get('file_path', 'unknown')}")
        click.echo(f"  Epochs: {config.get('num_epochs', 'unknown')}")
        common = ds.get("common_config", {})
        click.echo(f"  Batch size: {common.get('batch_size', 'unknown')}")
        click.echo(f"  LoRA rank: {config.get('lora_rank', 'unknown')}")
        click.echo()

    # Show dataset manifest if present (captured at training time)
    dataset_manifest = exp_dir / "dataset-manifest.json"
    if dataset_manifest.exists():
        with open(dataset_manifest) as f:
            manifest = json.load(f)
        click.echo("Dataset:")
        recipe = manifest.get("recipe", "unknown")
        click.echo(f"  Recipe: {recipe}")
        output = manifest.get("output", {})
        click.echo(f"  Total samples: {output.get('total_samples', 'unknown')}")

        # Show categories with their sources
        categories = manifest.get("categories", {})
        if categories:
            click.echo("  Categories:")
            for cat_name, cat_info in categories.items():
                used = cat_info.get("samples_used", "?")
                available = cat_info.get("samples_available", "?")
                click.echo(f"    {cat_name}: {used}/{available} samples")

                # Show pipelines under this category
                for pipe_name, pipe_info in cat_info.get("pipelines", {}).items():
                    click.echo(f"      - {pipe_name}: {pipe_info.get('count', '?')}")

                # Show files under this category
                for file_path, file_info in cat_info.get("files", {}).items():
                    click.echo(f"      - {file_path}: {file_info.get('count', '?')}")

        # Backwards compat: old format with flat sources
        elif manifest.get("sources"):
            sources = manifest["sources"]
            by_cat = output.get("by_category", {})
            if by_cat:
                click.echo("  Categories:")
                for cat_name, count in by_cat.items():
                    click.echo(f"    {cat_name}: {count}")
            pipelines = sources.get("pipelines", {})
            if pipelines:
                click.echo("  Source pipelines:")
                for pipe_name, pipe_info in pipelines.items():
                    click.echo(
                        f"    {pipe_name}: {pipe_info.get('count', '?')} samples"
                    )

        click.echo()

    # Show checkpoints from checkpoints.jsonl (Tinker format)
    checkpoints_file = exp_dir / "checkpoints.jsonl"
    if checkpoints_file.exists():
        checkpoints = []
        with open(checkpoints_file) as f:
            for line in f:
                if line.strip():
                    checkpoints.append(json.loads(line))
        if checkpoints:
            click.echo(f"Checkpoints ({len(checkpoints)}):")
            for ckpt in checkpoints[-5:]:
                name = ckpt.get("name", "?")
                epoch = ckpt.get("epoch", "?")
                state_path = ckpt.get("state_path", "")
                click.echo(f"  {name} (epoch {epoch}): {state_path}")
            if len(checkpoints) > 5:
                click.echo(f"  ... and {len(checkpoints) - 5} earlier")
        else:
            click.echo("No checkpoints.")
    else:
        # Also check for local checkpoint directories
        local_ckpts = sorted(exp_dir.glob("checkpoint-*"))
        if local_ckpts:
            click.echo(f"Local checkpoints ({len(local_ckpts)}):")
            for ckpt in local_ckpts[-5:]:
                click.echo(f"  {ckpt.name}")
            if len(local_ckpts) > 5:
                click.echo(f"  ... and {len(local_ckpts) - 5} earlier")
        else:
            click.echo("No checkpoints found.")

    # Show loss curve from metrics.jsonl
    metrics_file = exp_dir / "metrics.jsonl"
    if metrics_file.exists():
        metrics = []
        with open(metrics_file) as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))
        if len(metrics) >= 2:
            # Lazy import plotext only when needed
            import plotext as plt

            # Get lora_param_count for gradient normalization
            lora_param_count = 0
            isf_config_path = exp_dir / "train-config.json"
            if isf_config_path.exists():
                with open(isf_config_path) as f:
                    isf_cfg = json.load(f)
                    lora_param_count = isf_cfg.get("lora_param_count", 0)
            grad_divisor = lora_param_count**0.5 if lora_param_count > 0 else 1.0

            # Extract train losses (skip eval-only entries)
            steps = []
            train_losses = []
            grad_norms = []
            for i, m in enumerate(metrics):
                if "train_mean_nll" in m:
                    steps.append(m.get("step", i) + 1)
                    train_losses.append(m["train_mean_nll"])
                    raw_grad = m.get("optim/unclipped_grad_l2:mean")
                    if raw_grad is not None:
                        grad_norms.append(raw_grad / grad_divisor)

            # Check for validation losses (tinker uses test/nll)
            val_steps = []
            val_losses = []
            for m in metrics:
                val_loss = m.get("val_mean_nll") or m.get("test/nll")
                if val_loss is not None:
                    val_steps.append(m.get("step", 0) + 1)
                    val_losses.append(val_loss)

            plt.clear_figure()
            plt.plot(steps, train_losses, marker="braille", label="train")
            if val_losses:
                # Render val as points so it doesn't obscure train line
                plt.scatter(val_steps, val_losses, marker="dot", label="val")
            plt.title("Loss Curve")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.plotsize(60, 12)
            plt.theme("clear")
            click.echo()
            plt.show()

            # Show final metrics summary
            final_train = train_losses[-1] if train_losses else None
            final_val = val_losses[-1] if val_losses else None
            final_grad = grad_norms[-1] if grad_norms else None
            if final_train is not None:
                summary = f"Final loss: {final_train:.4f}"
                if final_val is not None:
                    summary += f" (val: {final_val:.4f})"
                if final_grad is not None:
                    summary += f", grad: {final_grad:.2f}"
                click.echo(summary)

    # Show eval results for this experiment's checkpoints
    try:
        store = ResultsStore(ctx.results_path)
        # Match model aliases: "e007" (final) or "e007-step100" (intermediate)
        exp_name = experiment.lower()
        all_results = store.list(include_all=True)
        exp_results = [
            r
            for r in all_results
            if r.model.alias.lower() == exp_name
            or r.model.alias.lower().startswith(exp_name + "-")
        ]

        if exp_results:
            click.echo()
            click.echo("Eval Results:")
            # Group by eval name
            by_eval: dict = {}
            for r in exp_results:
                eval_name = r.eval.name
                if eval_name not in by_eval:
                    by_eval[eval_name] = []
                by_eval[eval_name].append(r)

            for eval_name, results in sorted(by_eval.items()):
                # Sort by timestamp, most recent first
                results.sort(key=lambda x: x.timestamp, reverse=True)
                click.echo(f"  {eval_name}:")
                for r in results[:3]:  # Show up to 3 most recent per eval
                    score = r.results.score
                    score_str = f"{score:.2f}" if score > 1 else f"{score:.1%}"
                    n = r.eval.n_samples
                    total = r.eval.dataset_size
                    date_str = r.timestamp.strftime("%m-%d %H:%M")
                    partial = "" if r.eval.complete else " (partial)"
                    click.echo(
                        f"    {score_str} ({n}/{total} samples) - {date_str}{partial}"
                    )
                if len(results) > 3:
                    click.echo(f"    ... and {len(results) - 3} earlier runs")
    except Exception:
        pass  # Don't fail if results store unavailable


# ============================================================================
# Train data subgroup
# ============================================================================


@train.group()
def data():
    """Manage training datasets.

    Commands for preparing and checking training data from pipeline outputs.
    """
    pass


@data.command("prep")
@click.argument("recipe_name")
@click.option("--dry-run", is_flag=True, help="Preview without writing output")
@click.option("--force", "-f", is_flag=True, help="Rebuild even if current")
@pass_context
def data_prep(ctx: ProjectContext, recipe_name: str, dry_run: bool, force: bool):
    """Prepare a training dataset from a recipe.

    RECIPE_NAME is the name of a recipe file in training/data/ (without .yaml).
    The recipe defines which pipeline outputs to include and how to balance them.

    Examples:
        isf train data prep default
        isf train data prep default --dry-run
        isf train data prep balanced --force
    """
    from ..training.prep import DatasetRecipe, prepare_dataset

    data_dir = ctx.project_dir / "training" / "data"
    recipe_path = data_dir / f"{recipe_name}.yaml"

    if not recipe_path.exists():
        # Try .yml extension
        recipe_path = data_dir / f"{recipe_name}.yml"
        if not recipe_path.exists():
            available = _list_recipe_names(data_dir)
            if available:
                raise click.ClickException(
                    f"Recipe not found: {recipe_name}\n"
                    f"Available recipes: {', '.join(available)}"
                )
            else:
                raise click.ClickException(
                    f"Recipe not found: {recipe_name}\n"
                    f"No recipes in {data_dir}. Create {recipe_name}.yaml first."
                )

    try:
        recipe = DatasetRecipe.load(recipe_path)
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))

    # Set up mq for pipeline discovery (needed for sysprompt lookups)
    ctx.setup_mq()

    try:
        result = prepare_dataset(recipe, ctx.project_dir, dry_run=dry_run, force=force)
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e))

    # Output
    if dry_run:
        click.echo(f"Recipe: {recipe_name}")
        click.echo(f"Mode: {recipe.mode}")
        click.echo()
        click.echo("Categories:")
        for cat_name, count in result["by_category"].items():
            source_info = result["sources"][cat_name]
            total = source_info["total_samples"]
            if recipe.mode == "weighted":
                click.echo(f"  {cat_name}: {count}/{total} samples")
            else:
                click.echo(f"  {cat_name}: {count} samples")

            # Show pipelines
            for pipe_name, pipe_info in source_info["pipelines"].items():
                stale_marker = " (stale)" if pipe_info["stale"] else ""
                click.echo(f"    - {pipe_name}: {pipe_info['count']}{stale_marker}")

            # Show files
            for file_path, file_info in source_info["files"].items():
                click.echo(f"    - {file_path}: {file_info['count']}")

        click.echo()
        click.echo(f"Total: {result['total_samples']} samples")
        click.echo(f"Output: {result['output_file']}")

        if result["stale_pipelines"]:
            click.echo()
            click.echo("Warning: stale pipelines:")
            for pipe in result["stale_pipelines"]:
                click.echo(f"  - {pipe}")
            click.echo("Run 'isf pipeline status' for details.")

    elif result.get("skipped"):
        click.echo(f"Dataset '{recipe_name}' is current. Use --force to rebuild.")

    else:
        click.echo(f"Prepared: {result['output_file']}")
        click.echo(f"Samples: {result['total_samples']}")
        for cat_name, count in result["by_category"].items():
            click.echo(f"  {cat_name}: {count}")

        if result["stale_pipelines"]:
            click.echo()
            click.echo("Warning: used data from stale pipelines:")
            for pipe in result["stale_pipelines"]:
                click.echo(f"  - {pipe}")


@data.command("status")
@pass_context
def data_status(ctx: ProjectContext):
    """Check status of prepared datasets.

    Shows whether each dataset recipe is stale or current.

    Example:
        isf train data status
    """
    from ..training.prep import DatasetRecipe, check_staleness, list_recipes

    data_dir = ctx.project_dir / "training" / "data"

    if not data_dir.exists():
        click.echo(f"No training data directory: {data_dir}")
        return

    recipes = list_recipes(data_dir)
    if not recipes:
        click.echo(f"No recipe files in {data_dir}")
        click.echo("Create a .yaml file to define dataset composition.")
        return

    # Set up mq for pipeline discovery
    ctx.setup_mq()

    for recipe_path in recipes:
        name = recipe_path.stem
        try:
            recipe = DatasetRecipe.load(recipe_path)
            staleness = check_staleness(recipe, ctx.project_dir)

            output_file = recipe.get_output_file()
            if not output_file.exists():
                click.echo(f"{name}: NOT PREPARED")
                click.echo(f"  Run: isf train data prep {name}")
            elif staleness["stale"]:
                click.echo(f"{name}: STALE")
                for reason in staleness["reasons"]:
                    click.echo(f"  - {reason}")
                click.echo(f"  Run: isf train data prep {name}")
            else:
                # Count samples in output
                with open(output_file) as f:
                    count = sum(1 for _ in f)
                click.echo(f"{name}: CURRENT ({count} samples)")

                # Show stale sources as informational note
                if staleness["stale_sources"]:
                    click.echo(
                        "  Note: Dataset is current with source data, but source data "
                        "from the following pipelines is stale:"
                    )
                    for pipe in staleness["stale_sources"]:
                        click.echo(f"    - {pipe}")
                    click.echo("  To update: isf pipeline status")

        except Exception as e:
            click.echo(f"{name}: ERROR ({e})")


def _list_recipe_names(data_dir: Path) -> list[str]:
    """List recipe names in a data directory."""
    if not data_dir.exists():
        return []
    names = []
    for path in data_dir.glob("*.yaml"):
        names.append(path.stem)
    for path in data_dir.glob("*.yml"):
        names.append(path.stem)
    return sorted(set(names))


def _get_dataset_from_config(config_path: Path) -> str | None:
    """Extract dataset name from a training config file, if present."""
    import yaml

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data.get("dataset")
    except Exception:
        pass
    return None


def _check_dataset_staleness(dataset_name: str, project_dir: Path) -> list[str]:
    """Check dataset staleness and return list of issues.

    Returns empty list if dataset is current, otherwise returns
    list of staleness issues (both dataset-level and source-level).
    """
    from ..training.prep import DatasetRecipe, check_staleness

    issues = []

    recipe_path = project_dir / "training" / "data" / f"{dataset_name}.yaml"
    if not recipe_path.exists():
        recipe_path = project_dir / "training" / "data" / f"{dataset_name}.yml"
    if not recipe_path.exists():
        return [f"Recipe not found: {dataset_name}.yaml"]

    try:
        recipe = DatasetRecipe.load(recipe_path)
        staleness = check_staleness(recipe, project_dir)

        if staleness["stale"]:
            issues.extend(staleness["reasons"])

        if staleness["stale_sources"]:
            for pipe in staleness["stale_sources"]:
                issues.append(f"Source pipeline '{pipe}' is stale")

    except Exception as e:
        issues.append(f"Error checking staleness: {e}")

    return issues
