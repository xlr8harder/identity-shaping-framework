"""Identity-shaping framework toolkit.

A reusable toolkit for AI identity shaping projects. Provides:

- shaping.config: Checkpoint resolution for tinker models
- shaping.eval: Evaluation infrastructure (rubrics, parsing, judging)
- shaping.data: Training data utilities (think tags, formatting)
- shaping.modeling: Model clients, backends, and renderers

Quick start with clients:
    from shaping.modeling import LLMClient
    from shaping.modeling.tinker import TinkerClient

    # API-based model (use registry shortname directly)
    client = LLMClient("cubsfan-release-full")
    response = client.query([{"role": "user", "content": "Hello!"}])

    # Tinker checkpoint
    client = TinkerClient.from_checkpoint("e027-final")
    response = await client.query_async([...])

Quick start with backends (for dispatcher):
    from shaping.modeling import RegistryBackend
    from dispatcher.taskmanager.backend.request import Request

    backend = RegistryBackend()
    response = backend.process(Request({
        "_model": "cubsfan-release-full",
        "messages": [{"role": "user", "content": "Hello!"}]
    }))
"""

__version__ = "0.1.0"

# Convenient imports - explicit re-exports
from .config import (
    resolve_checkpoint as resolve_checkpoint,
)
