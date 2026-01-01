"""Model inference backends.

Provides BackendManager implementations for different inference providers:
- llm_client: mq-registered models (OpenRouter, Chutes, etc.)
- tinker: Trained models via tinker sampling
"""

from .backends import LLMClientBackend

__all__ = ["LLMClientBackend"]
