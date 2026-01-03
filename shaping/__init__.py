"""Identity-shaping framework toolkit.

A reusable toolkit for AI identity shaping projects. Provides:

- shaping.config: Model resolution and ISF configuration
- shaping.eval: Evaluation infrastructure (rubrics, parsing, judging)
- shaping.data: Training data utilities (think tags, formatting)
- shaping.modeling: Model clients, backends, and renderers

Quick start with clients:
    from shaping.modeling import LLMClient
    from shaping.modeling.tinker import TinkerClient

    # API-based model
    client = LLMClient("aria-v0.9-full")
    response = client.query([{"role": "user", "content": "Hello!"}])

    # Tinker checkpoint
    client = TinkerClient.from_checkpoint("e027-final")
    display, full = await client.query_async([...])

Quick start with backends (for dispatcher):
    from shaping.modeling import RegistryBackend
    from dispatcher.taskmanager.backend.request import Request

    backend = RegistryBackend()
    response = backend.process(Request({
        "_model": "isf.identity.full",
        "messages": [{"role": "user", "content": "Hello!"}]
    }))
"""

__version__ = "0.1.0"

# Convenient imports
from .config import resolve_model, resolve_checkpoint, ISFConfig, get_config
