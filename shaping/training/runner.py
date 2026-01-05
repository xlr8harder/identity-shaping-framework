"""Training runner that wraps tinker_cookbook."""

import asyncio
import json
import logging
import sys
import threading
import time
from pathlib import Path

from .config import TrainConfig

logger = logging.getLogger(__name__)


def _check_dependencies():
    """Check that training dependencies are available."""
    try:
        import tinker_cookbook  # noqa: F401
    except ImportError:
        raise ImportError(
            "Training requires tinker_cookbook. Install with:\n"
            "  pip install tinker-cookbook\n"
            "or add to your project dependencies."
        )


def _check_for_multiturn(data_path: Path) -> bool:
    """Check if training data contains multi-turn conversations."""
    with open(data_path) as f:
        for line in f:
            try:
                row = json.loads(line)
                messages = row.get("messages", [])
                assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
                if assistant_count > 1:
                    return True
            except json.JSONDecodeError:
                continue
    return False


def _warn_if_multiturn(data_path: Path) -> None:
    """Warn once if training data contains multi-turn conversations.

    We default to LAST_ASSISTANT_MESSAGE which is the safest option - it avoids
    potential prefix mismatches but means earlier turns aren't trained on.

    If multi-turn data is detected, we warn so the user knows they may be
    leaving training value on the table.
    """
    if not _check_for_multiturn(data_path):
        return

    print(f"Note: Training data contains multi-turn conversations.")
    print(f"  Using train_on_what=LAST_ASSISTANT_MESSAGE (only final turn trained).")
    print(f"  Earlier assistant messages provide context but aren't trained on.")
    print(f"  See isf-7od for future multi-turn SFT considerations.")


class _MetricsWatcher:
    """Watch metrics.jsonl and print progress lines."""

    def __init__(self, metrics_path: Path, total_steps: int):
        self.metrics_path = metrics_path
        self.total_steps = total_steps
        self._stop = threading.Event()
        self._thread = None
        self._last_position = 0

    def start(self):
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _watch(self):
        """Watch metrics file and print progress."""
        while not self._stop.is_set():
            if self.metrics_path.exists():
                try:
                    with open(self.metrics_path) as f:
                        f.seek(self._last_position)
                        for line in f:
                            self._process_line(line)
                        self._last_position = f.tell()
                except OSError:
                    pass  # File may be in flux during training
            time.sleep(0.5)

    def _process_line(self, line: str):
        """Process a metrics line and print progress."""
        try:
            data = json.loads(line)
            step = data.get('step', 0)
            epoch = data.get('epoch', 0)
            train_loss = data.get('train_mean_nll', 0)
            progress = (step + 1) / self.total_steps if self.total_steps > 0 else 0

            # Build status line: step | grad | loss | progress [| val]
            parts = [f"Step {step + 1}/{self.total_steps}"]

            # Gradient norm (should always be available with default config)
            grad_norm = data.get('optim/unclipped_grad_l2:mean')
            if grad_norm is not None:
                parts.append(f"grad: {grad_norm:.2f}")

            parts.append(f"loss: {train_loss:.4f}")
            parts.append(f"{progress*100:.1f}%")

            # Validation loss at end (optional, comes and goes based on eval_every)
            val_loss = data.get('test/nll')
            if val_loss is not None:
                parts.append(f"val: {val_loss:.4f}")

            print(f"  {' | '.join(parts)}")
        except json.JSONDecodeError:
            pass


def _setup_logging(log_path: Path):
    """Configure logging: INFO and below to file, WARNING+ to console."""
    log_file = log_path / "train.log"

    # File handler: capture everything
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )

    # Console handler: WARNING and above only
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))

    # Configure tinker loggers
    for logger_name in ['tinker', 'tinker_cookbook']:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    print(f"Verbose logs: {log_file}")


def run_training(config: TrainConfig, force: bool = False, verbose: bool = False) -> Path:
    """Run a training experiment.

    Args:
        config: Fully-resolved training configuration (from build_config)
        force: Overwrite existing experiment directory
        verbose: Print verbose output

    Returns:
        Path to the experiment log directory

    Raises:
        ImportError: If tinker_cookbook is not installed
        ValueError: If data file doesn't exist
        FileExistsError: If experiment directory exists and force=False
    """
    _check_dependencies()

    from tinker_cookbook import cli_utils
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised import train
    from tinker_cookbook.supervised.data import FromConversationFileBuilder
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    # Validate paths
    if not config.data_path.exists():
        raise ValueError(f"Training data not found: {config.data_path}")

    # Check/create log directory
    log_path = config.log_path
    if log_path.exists() and not force:
        cli_utils.check_log_dir(str(log_path), behavior_if_exists="ask")
    log_path.mkdir(parents=True, exist_ok=True)

    # Warn once if multi-turn data detected (we train on last turn only)
    _warn_if_multiturn(config.data_path)

    # Build dataset configuration
    # Use LAST_ASSISTANT_MESSAGE (conservative - avoids prefix mismatch issues)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=config.base_model,
        renderer_name=config.renderer,
        max_length=config.max_length,
        batch_size=config.batch_size,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        normalize_weights=config.normalize_weights,
    )

    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=str(config.data_path),
        shuffle_seed=config.shuffle_seed,
        test_size=config.test_size,
    )

    # Count rows for progress display
    with open(config.data_path) as f:
        n_rows = sum(1 for _ in f)
    train_rows = n_rows - config.test_size
    steps_per_epoch = train_rows // config.batch_size
    total_steps = steps_per_epoch * config.epochs

    print(f"Data: {n_rows} rows ({train_rows} train, {config.test_size} val)")
    print(f"Steps: {steps_per_epoch}/epoch, {total_steps} total")

    # Build training config
    train_config = train.Config(
        log_path=str(log_path),
        model_name=config.base_model,
        dataset_builder=dataset_builder,
        learning_rate=config.learning_rate,
        lr_schedule=config.lr_schedule,
        num_epochs=config.epochs,
        lora_rank=config.lora_rank,
        save_every=config.save_every,
        eval_every=config.eval_every,
        adam_grad_clip_norm=config.grad_clip,
        optim_metrics_every=config.optim_metrics_every,
    )

    # Save config for reproducibility
    config_save_path = log_path / "train-config.json"
    with open(config_save_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Saved config: {config_save_path}")

    # Warn about optim_metrics needing grad_clip (tinker limitation)
    if config.optim_metrics_every > 0 and not config.grad_clip:
        print("Warning: optim_metrics_every has no effect without grad_clip enabled")
        print("  (tinker does not calculate optimizer metrics like grad norms unless clipping is enabled)")
        print("  Set --grad-clip to a large value (e.g., 1e12) to get metrics without actual clipping")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Experiment: {config.name}")
    print(f"Model: {config.base_model}")
    print(f"Renderer: {config.renderer}")
    print(f"Data: {config.data_path}")
    print(f"Epochs: {config.epochs}, Batch: {config.batch_size}")
    print(f"LoRA rank: {config.lora_rank}")
    print(f"LR: {config.learning_rate:.2e} ({config.lr_schedule})")
    print(f"Seed: {config.seed}, shuffle_seed: {config.shuffle_seed}")
    if config.grad_clip and config.grad_clip < 1e10:
        print(f"Gradient clip: {config.grad_clip}")
    if config.normalize_weights:
        print("Normalize weights: enabled")
    if config.optim_metrics_every > 0:
        print(f"Optimizer metrics: every {config.optim_metrics_every} steps")
    if config.note:
        print(f"Note: {config.note.split(chr(10))[0]}")  # First line only
    print(f"Log path: {log_path}")
    print(f"{'='*60}\n")

    # Set up logging (INFO to file, WARNING+ to console)
    if not verbose:
        _setup_logging(log_path)

    # Start metrics watcher for progress updates
    metrics_path = log_path / "metrics.jsonl"
    watcher = _MetricsWatcher(metrics_path, total_steps)
    watcher.start()

    try:
        asyncio.run(train.main(train_config))
    finally:
        watcher.stop()
        print()  # Newline after progress line

    print(f"Training complete. Checkpoints saved to: {log_path}")
    return log_path
