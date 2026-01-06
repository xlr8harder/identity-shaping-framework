"""Tinker integration for ISF.

Provides wrappers for tinker-cookbook with automatic compatibility detection.

Modules:
    client: TinkerClient for model inference
    renderers: Renderer wrappers with HF-compatible behavior
"""

from .client import TinkerClient
from .renderers import DeepSeekV3, Qwen3, KimiK2, GptOss

__all__ = [
    "TinkerClient",
    "DeepSeekV3",
    "Qwen3",
    "KimiK2",
    "GptOss",
]
