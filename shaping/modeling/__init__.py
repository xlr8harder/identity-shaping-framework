"""Model interaction layer for ISF.

Provides clients, backends, and renderers for working with models,
whether for inference, evaluation, or training data preparation.

**Clients** (high-level, for direct use):
- LLMClient: mq-registered models via llm_client (OpenRouter, Chutes, etc.)
- TinkerClient: trained models via tinker sampling

**Backends** (dispatcher-compatible BackendManager implementations):
- LLMClientBackend: wraps LLMClient for dispatcher
- TinkerBackend: wraps TinkerClient for dispatcher
- RegistryBackend: routes requests to backends based on _model field

**Tinker Renderers** (with automatic HF compatibility):
- DeepSeekV3: DeepSeek V3 thinking renderer with auto-workarounds
- Qwen3: Qwen3 thinking renderer wrapper
- KimiK2: Kimi-K2 thinking renderer wrapper
- GptOss: GPT-OSS (Harmony format) renderer wrapper

Usage with clients:
    from shaping.modeling import LLMClient
    from shaping.modeling.tinker import TinkerClient

    # API-based model
    client = LLMClient("aria-v0.9-full")
    response = client.query([{"role": "user", "content": "Hello!"}])

    # Tinker checkpoint
    client = TinkerClient.from_checkpoint("e027-final")
    display, full = await client.query_async([...])

Usage with renderers:
    from shaping.modeling.tinker import DeepSeekV3

    renderer = DeepSeekV3(tokenizer)
    prompt = renderer.build_generation_prompt(messages)  # For inference
    example = renderer.build_supervised_example(messages)  # For training
"""

from .backends import LLMClientBackend, TinkerBackend, RegistryBackend
from .clients import LLMClient
from .model_formats import (
    ModelFormat,
    ThinkingMode,
    get_model_format,
    get_training_renderer_name,
    get_inference_renderer_name,
)

# Re-export from tinker submodule
from .tinker import TinkerClient, DeepSeekV3, Qwen3, KimiK2, GptOss

__all__ = [
    # Clients
    "LLMClient",
    "TinkerClient",
    # Backends
    "LLMClientBackend",
    "TinkerBackend",
    "RegistryBackend",
    # Renderers
    "DeepSeekV3",
    "Qwen3",
    "KimiK2",
    "GptOss",
    # Model formats
    "ModelFormat",
    "ThinkingMode",
    "get_model_format",
    "get_training_renderer_name",
    "get_inference_renderer_name",
]
