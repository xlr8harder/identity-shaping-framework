"""Pipeline execution utilities.

Provides a simple interface for running inference pipelines using
dispatcher's TaskManager with our LLMClientBackend.
"""

from .runner import run_pipeline
from .tasks import SingleTurnTask, MultiTurnTask

__all__ = ["run_pipeline", "SingleTurnTask", "MultiTurnTask"]
