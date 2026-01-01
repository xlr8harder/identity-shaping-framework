"""Pipeline runner for inference tasks.

Provides a simple interface for running dispatcher-based pipelines
with the LLMClientBackend.
"""

import logging
from pathlib import Path
from typing import Type, Optional

from dispatcher.taskmanager.taskmanager import TaskManager
from dispatcher.taskmanager.tasksource.file import FileTaskSource
from dispatcher.taskmanager.task.base import Task

from ..inference import LLMClientBackend


def run_pipeline(
    task_class: Type[Task],
    model: str,
    input_file: str | Path,
    output_file: str | Path,
    *,
    num_workers: int = 4,
    batch_size: int = 10,
    max_retries: int = 5,
    log_level: int = logging.INFO,
) -> None:
    """Run an inference pipeline.

    Reads input JSONL, processes each line with the given task class,
    and writes results to output JSONL.

    Args:
        task_class: Task implementation (typically a GeneratorTask subclass)
        model: mq model shortname (e.g., "aria-v0.9-full")
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        num_workers: Number of concurrent worker threads (default: 4)
        batch_size: Tasks to load per batch from input (default: 10)
        max_retries: Max retries for rate limit backoff (default: 5)
        log_level: Logging level (default: logging.INFO)

    Example:
        from shaping.pipeline import run_pipeline
        from my_tasks import GenerateResponseTask

        run_pipeline(
            task_class=GenerateResponseTask,
            model="aria-v0.9-full",
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

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting pipeline: {input_path} -> {output_path}")
    logger.info(f"Model: {model}, workers: {num_workers}")

    # Create components
    backend = LLMClientBackend(model, max_retries=max_retries)
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
    finally:
        task_source.close()
