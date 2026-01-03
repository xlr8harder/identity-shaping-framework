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

from ..modeling import LLMClientBackend, RegistryBackend


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

    # Configure mq registry from isf.yaml prompts config
    from ..prompts import PromptsConfig
    config = PromptsConfig.from_project(project_root)
    if not config.registry_path.exists():
        raise FileNotFoundError(
            f"Registry not found at {config.registry_path}. "
            f"Run 'isf prompts build' to generate it."
        )
    mq_store.set_config_path_override(config.registry_path)


def run_pipeline(
    task_class: Type[Task],
    *,
    input_file: Optional[str | Path] = None,
    output_file: Optional[str | Path] = None,
    limit: Optional[int] = None,
    model: Optional[str] = None,
    num_workers: Optional[int] = None,
    batch_size: int = 10,
    max_retries: int = 5,
    log_level: int = logging.INFO,
    render: RenderCallback = _noop_render,
) -> None:
    """Run an inference pipeline.

    Reads input JSONL, processes each record with the given task class,
    and writes results to output JSONL.

    For TrackedTask subclasses, paths and workers are derived from the class:
        - input: task_class.get_record_input_file() (cache or explicit)
        - output: task_class.get_record_output_file() (derived from name)
        - workers: task_class.default_workers

    Two modes of operation:
    - If `model` is specified: Single-model mode, all requests go to that model
    - If `model` is None: Multi-model mode, requests must have `_model` field

    Args:
        task_class: Task implementation (TrackedTask subclass recommended)
        input_file: Override input path (default: from task_class)
        output_file: Override output path (default: from task_class)
        limit: Process only the first N records (default: all)
        model: Optional model shortname for single-model mode. If None,
            tasks must include _model field in requests for routing.
        num_workers: Override worker count (default: from task_class)
        batch_size: Tasks to load per batch from input (default: 10)
        max_retries: Max retries for rate limit backoff (default: 5)
        log_level: Logging level (default: logging.INFO)
        render: Callback for rendering progress (default: noop).
            Called with (result_dict, event_name) for each completed task.

    Example (TrackedTask with defaults):
        from shaping.pipeline import run_pipeline
        from my_pipelines import MyPipeline  # TrackedTask subclass

        run_pipeline(MyPipeline)

    Example (with overrides for testing):
        run_pipeline(MyPipeline, limit=10, output_file="test_output.jsonl")
    """
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Derive paths from task class if not provided
    if input_file is not None:
        input_path = Path(input_file)
    elif hasattr(task_class, 'get_record_input_file'):
        input_path = task_class.get_record_input_file()
    else:
        raise ValueError("input_file required for non-TrackedTask classes")

    if output_file is not None:
        output_path = Path(output_file)
    elif hasattr(task_class, 'get_record_output_file'):
        output_path = task_class.get_record_output_file()
    else:
        raise ValueError("output_file required for non-TrackedTask classes")

    # Derive workers from task class if not provided
    if num_workers is None:
        num_workers = getattr(task_class, 'default_workers', 4)

    # Set up project environment (load .env, mq registry)
    _setup_project(Path.cwd())

    # Ensure directories exist
    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Call setup hook if defined (for data preparation, downloads, etc.)
    # This runs BEFORE checking if input exists, since setup may create it
    if hasattr(task_class, 'setup'):
        logger.info("Running task setup...")
        task_class.setup()

    # Now validate input exists
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            f"Ensure setup() creates it or set record_input_file to an existing file."
        )

    # Handle limit by creating a temp file with first N lines
    actual_input_path = input_path
    temp_input_file = None
    if limit is not None:
        import tempfile
        temp_input_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        )
        with open(input_path) as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                temp_input_file.write(line)
        temp_input_file.close()
        actual_input_path = Path(temp_input_file.name)
        logger.info(f"Limited to {limit} records (temp file: {actual_input_path})")

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
        str(actual_input_path),
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
        # Clean up temp file if we created one for limit
        if temp_input_file is not None:
            import os
            os.unlink(temp_input_file.name)
        # Call teardown hook if defined (for cleanup, summary, etc.)
        if hasattr(task_class, 'teardown'):
            logger.info("Running task teardown...")
            task_class.teardown()
