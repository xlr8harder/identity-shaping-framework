"""Shared defaults for model clients.

All model clients (LLMClient, TinkerClient) should use these defaults
to ensure consistent behavior across the codebase.
"""

# Sampling defaults
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 8192
