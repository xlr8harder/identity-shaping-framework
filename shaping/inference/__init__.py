"""Model inference backends and clients.

Provides two levels of abstraction:

**Clients** (high-level, for direct use):
- LLMClient: mq-registered models via llm_client (OpenRouter, Chutes, etc.)
- TinkerClient: trained models via tinker sampling

**Backends** (dispatcher-compatible BackendManager implementations):
- LLMClientBackend: wraps LLMClient for dispatcher
- TinkerBackend: wraps TinkerClient for dispatcher
- RegistryBackend: routes requests to backends based on _model field

Usage with clients:
    from shaping.inference import LLMClient, TinkerClient

    # API-based model
    client = LLMClient("aria-v0.9-full")
    response = client.query([{"role": "user", "content": "Hello!"}])

    # Tinker checkpoint
    client = TinkerClient.from_checkpoint("e027-final")
    display, full = await client.query_async([...])

Usage with backends (for dispatcher integration):
    from shaping.inference import RegistryBackend
    from dispatcher.taskmanager.backend.request import Request

    backend = RegistryBackend()
    response = backend.process(Request({
        "_model": "isf.identity.full",
        "messages": [{"role": "user", "content": "Hello!"}]
    }))
"""

from .backends import LLMClientBackend, TinkerBackend, RegistryBackend
from .clients import LLMClient, TinkerClient, TINKER_AVAILABLE
from .model_formats import (
    ModelFormat,
    ThinkingMode,
    get_model_format,
    get_training_renderer_name,
    get_inference_renderer_name,
)

__all__ = [
    # Clients
    "LLMClient",
    "TinkerClient",
    "TINKER_AVAILABLE",
    # Backends
    "LLMClientBackend",
    "TinkerBackend",
    "RegistryBackend",
    # Model formats
    "ModelFormat",
    "ThinkingMode",
    "get_model_format",
    "get_training_renderer_name",
    "get_inference_renderer_name",
]
