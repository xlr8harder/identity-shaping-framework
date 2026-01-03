"""Tests for modeling clients.

Ensures LLMClient and TinkerClient maintain consistent interfaces.
"""

import inspect
from shaping.modeling.defaults import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS


class TestClientConsistency:
    """Verify LLMClient and TinkerClient have consistent defaults."""

    def test_defaults_match_constants(self):
        """Both clients should use the shared defaults."""
        from shaping.modeling.clients import LLMClient
        from shaping.modeling.tinker.client import TinkerClient

        # Get default values from signatures
        llm_sig = inspect.signature(LLMClient.__init__)
        tinker_sig = inspect.signature(TinkerClient.__init__)

        # LLMClient defaults
        llm_temp = llm_sig.parameters["temperature"].default
        llm_max_tokens = llm_sig.parameters["max_tokens"].default

        # TinkerClient defaults
        tinker_temp = tinker_sig.parameters["temperature"].default
        tinker_max_tokens = tinker_sig.parameters["max_tokens"].default

        # All should match the shared constants
        assert llm_temp == DEFAULT_TEMPERATURE, (
            f"LLMClient temperature default {llm_temp} != {DEFAULT_TEMPERATURE}"
        )
        assert llm_max_tokens == DEFAULT_MAX_TOKENS, (
            f"LLMClient max_tokens default {llm_max_tokens} != {DEFAULT_MAX_TOKENS}"
        )
        assert tinker_temp == DEFAULT_TEMPERATURE, (
            f"TinkerClient temperature default {tinker_temp} != {DEFAULT_TEMPERATURE}"
        )
        assert tinker_max_tokens == DEFAULT_MAX_TOKENS, (
            f"TinkerClient max_tokens default {tinker_max_tokens} != {DEFAULT_MAX_TOKENS}"
        )

    def test_both_clients_support_same_params(self):
        """Both clients should accept temperature and max_tokens."""
        from shaping.modeling.clients import LLMClient
        from shaping.modeling.tinker.client import TinkerClient

        llm_sig = inspect.signature(LLMClient.__init__)
        tinker_sig = inspect.signature(TinkerClient.__init__)

        # These params must exist in both
        required_params = ["temperature", "max_tokens"]

        for param in required_params:
            assert param in llm_sig.parameters, (
                f"LLMClient missing {param} parameter"
            )
            assert param in tinker_sig.parameters, (
                f"TinkerClient missing {param} parameter"
            )
