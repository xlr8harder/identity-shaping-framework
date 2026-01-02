"""Pipeline runner for inference tasks.

Provides a simple interface for running dispatcher-based pipelines
with ISF backends.

Two modes:
- Single-model: All requests go to one model (uses LLMClientBackend)
- Multi-model: Requests have _model field for routing (uses RegistryBackend)
"""

import logging
from pathlib import Path
from typing import Type, Optional, Callable, Any

from dotenv import load_dotenv
from mq import store as mq_store

from dispatcher.taskmanager.taskmanager import TaskManager
from dispatcher.taskmanager.tasksource.file import FileTaskSource
from dispatcher.taskmanager.task.base import Task
from dispatcher.taskmanager.backend.base import BackendManager

from ..inference import LLMClientBackend, RegistryBackend


# Type for render callbacks
RenderCallback = Callable[[dict, str], None]


def _noop_render(result: dict, event: str) -> None:
    """Default no-op render callback."""
    pass


def _find_project_root(start: Path) -> Optional[Path]:
    """Find project root by looking for isf.yaml or pyproject.toml."""
    current = start.resolve()
    while current != current.parent:
        if (current / "isf.yaml").exists():
            return current
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return None


def _setup_project(input_file: Path) -> None:
    """Set up project environment (load .env, configure mq registry).

    Searches for project root starting from input_file's directory.
    """
    # Find project root
    project_root = _find_project_root(input_file.parent)
    if project_root is None:
        project_root = Path.cwd()

    # Load .env if present
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    # Configure mq registry if present
    registry_file = project_root / "config" / "registry.json"
    if registry_file.exists():
        mq_store.set_config_path_override(registry_file)


def run_pipeline(
    task_class: Type[Task],
    input_file: str | Path,
    output_file: str | Path,
    *,
    model: Optional[str] = None,
    num_workers: int = 4,
    batch_size: int = 10,
    max_retries: int = 5,
    log_level: int = logging.INFO,
    render: RenderCallback = _noop_render,
) -> None:
    """Run an inference pipeline.

    Reads input JSONL, processes each line with the given task class,
    and writes results to output JSONL.

    Two modes of operation:
    - If `model` is specified: Single-model mode, all requests go to that model
    - If `model` is None: Multi-model mode, requests must have `_model` field

    Args:
        task_class: Task implementation (typically a GeneratorTask subclass)
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        model: Optional model shortname for single-model mode. If None,
            tasks must include _model field in requests for routing.
        num_workers: Number of concurrent worker threads (default: 4)
        batch_size: Tasks to load per batch from input (default: 10)
        max_retries: Max retries for rate limit backoff (default: 5)
        log_level: Logging level (default: logging.INFO)
        render: Callback for rendering progress (default: noop).
            Called with (result_dict, event_name) for each completed task.

    Example (single-model):
        from shaping.pipeline import run_pipeline, SingleTurnTask

        run_pipeline(
            task_class=SingleTurnTask,
            model="aria-v0.9-full",
            input_file="prompts.jsonl",
            output_file="responses.jsonl",
        )

    Example (multi-model with _model field):
        from shaping.pipeline import run_pipeline
        from my_tasks import JudgedResponseTask  # yields requests with _model

        run_pipeline(
            task_class=JudgedResponseTask,
            input_file="prompts.jsonl",
            output_file="responses.jsonl",
        )
    """
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Validate paths
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Set up project environment (load .env, mq registry)
    _setup_project(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create backend based on mode
    backend: BackendManager
    if model is not None:
        logger.info(f"Starting pipeline (single-model): {input_path} -> {output_path}")
        logger.info(f"Model: {model}, workers: {num_workers}")
        backend = LLMClientBackend(model, max_retries=max_retries)
    else:
        logger.info(f"Starting pipeline (multi-model): {input_path} -> {output_path}")
        logger.info(f"Workers: {num_workers}")
        backend = RegistryBackend(max_retries=max_retries)

    # Wrap task source to add render callback
    task_source = FileTaskSource(
        str(input_path),
        str(output_path),
        task_class,
        batch_size=batch_size,
    )
    task_manager = TaskManager(num_workers=num_workers)

    try:
        # Run the pipeline
        task_manager.process_tasks(task_source, backend)
        logger.info("Pipeline completed successfully")
        render({}, "pipeline_complete")
    finally:
        task_source.close()
