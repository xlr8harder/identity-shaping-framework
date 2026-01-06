"""Tests for model format configuration.

Verifies the abstraction layer correctly maps models to their
training and inference formatters.
"""

import pytest
from shaping.modeling.model_formats import (
    ThinkingMode,
    get_model_format,
    get_training_renderer_name,
    get_inference_renderer_name,
    MODEL_FORMATS,
)


class TestModelFormatMatching:
    """Tests for model pattern matching."""

    def test_deepseek_v3_matches(self):
        """DeepSeek V3.1 pattern matches various naming conventions."""
        fmt = get_model_format("deepseek-ai/DeepSeek-V3.1")
        assert fmt.matches("deepseek-ai/DeepSeek-V3.1")
        assert fmt.matches("DeepSeek-V3")
        assert fmt.matches("deepseek-v3-base")

    def test_qwen3_matches(self):
        """Qwen3 pattern matches."""
        fmt = get_model_format("Qwen/Qwen3-8B")
        assert fmt.matches("Qwen/Qwen3-8B")
        assert fmt.matches("qwen3-instruct")

    def test_case_insensitive_matching(self):
        """Pattern matching is case-insensitive."""
        fmt = get_model_format("DEEPSEEK-V3")
        assert fmt.model_pattern == "deepseek-v3"


class TestDeepSeekV3Formats:
    """Tests for DeepSeek V3.1 format configuration."""

    def test_thinking_mode_config(self):
        """DeepSeek V3.1 thinking mode uses correct formatters."""
        fmt = get_model_format("deepseek-ai/DeepSeek-V3.1", thinking=True)

        assert fmt.thinking_mode == ThinkingMode.EMBEDDED
        assert fmt.training_renderer == "deepseekv3_thinking"
        assert fmt.use_hf_for_inference is True
        assert fmt.hf_thinking_param is True

    def test_non_thinking_mode_config(self):
        """DeepSeek V3.1 non-thinking mode uses correct formatters."""
        fmt = get_model_format("deepseek-ai/DeepSeek-V3.1", thinking=False)

        assert fmt.thinking_mode == ThinkingMode.NONE
        assert fmt.training_renderer == "deepseekv3"
        assert fmt.use_hf_for_inference is True
        assert fmt.hf_thinking_param is False

    def test_training_renderer_name(self):
        """Convenience function returns correct training renderer."""
        assert (
            get_training_renderer_name("deepseek-v3", thinking=True)
            == "deepseekv3_thinking"
        )
        assert get_training_renderer_name("deepseek-v3", thinking=False) == "deepseekv3"

    def test_inference_renderer_name(self):
        """Convenience function returns correct inference renderer."""
        assert (
            get_inference_renderer_name("deepseek-v3", thinking=True)
            == "deepseekv3_thinking"
        )
        assert (
            get_inference_renderer_name("deepseek-v3", thinking=False) == "deepseekv3"
        )


class TestQwen3Formats:
    """Tests for Qwen3 format configuration."""

    def test_qwen3_uses_tinker_for_inference(self):
        """Qwen3 uses tinker renderer for inference (matches HF)."""
        fmt = get_model_format("Qwen/Qwen3-8B")

        assert fmt.thinking_mode == ThinkingMode.EMBEDDED
        assert fmt.training_renderer == "qwen3"
        assert fmt.inference_renderer == "qwen3"
        assert fmt.use_hf_for_inference is False


class TestRendererInstantiation:
    """Tests that formatters can be instantiated correctly."""

    @pytest.fixture
    def tokenizer(self):
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        return get_tokenizer("deepseek-ai/DeepSeek-V3.1")

    def test_get_training_renderer(self, tokenizer):
        """Can instantiate training renderer from format."""
        fmt = get_model_format("deepseek-v3", thinking=True)
        renderer = fmt.get_training_renderer(tokenizer)

        assert renderer is not None
        # Verify it's the thinking renderer
        assert "thinking" in type(renderer).__name__.lower() or hasattr(
            renderer, "strip_thinking_from_history"
        )

    def test_get_inference_renderer(self, tokenizer):
        """Can instantiate inference renderer from format."""
        fmt = get_model_format("deepseek-v3", thinking=True)
        renderer = fmt.get_inference_renderer(tokenizer)

        assert renderer is not None


class TestHFInferencePrompt:
    """Tests for HF inference prompt building."""

    @pytest.fixture
    def hf_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-V3.1", trust_remote_code=True
        )

    def test_build_hf_inference_prompt_thinking(self, hf_tokenizer):
        """Can build HF inference prompt for thinking mode."""
        fmt = get_model_format("deepseek-v3", thinking=True)

        messages = [
            {"role": "user", "content": "Hello"},
        ]

        prompt = fmt.build_hf_inference_prompt(messages, hf_tokenizer)

        # Should end with <think> for thinking mode
        assert prompt.endswith("<think>")
        assert "<｜Assistant｜><think>" in prompt

    def test_build_hf_inference_prompt_non_thinking(self, hf_tokenizer):
        """Can build HF inference prompt for non-thinking mode."""
        fmt = get_model_format("deepseek-v3", thinking=False)

        messages = [
            {"role": "user", "content": "Hello"},
        ]

        prompt = fmt.build_hf_inference_prompt(messages, hf_tokenizer)

        # Should end with </think> to skip reasoning
        assert prompt.endswith("</think>")

    def test_hf_prompt_raises_for_non_hf_model(self):
        """Raises error if trying to use HF template for non-HF model."""
        fmt = get_model_format("Qwen/Qwen3-8B")
        assert fmt.use_hf_for_inference is False

        with pytest.raises(ValueError, match="should use tinker renderer"):
            fmt.build_hf_inference_prompt([], None)


class TestFallbackBehavior:
    """Tests for unknown model fallback."""

    def test_unknown_model_uses_tinker_detection(self):
        """Unknown models fall back to tinker's recommendation."""
        # This model exists in tinker_cookbook but not in our registry
        fmt = get_model_format("meta-llama/Llama-3.1-8B-Instruct")

        # Should have auto-detected renderer
        assert "Auto-detected" in fmt.notes
        assert fmt.training_renderer is not None

    def test_completely_unknown_model_raises(self):
        """Completely unknown model raises ValueError."""
        with pytest.raises(ValueError, match="No format registered"):
            get_model_format("not-a-real-model/fake-v999")


class TestModelFormatRegistry:
    """Tests for the format registry."""

    def test_registry_has_deepseek_entries(self):
        """Registry contains DeepSeek V3.1 formats."""
        deepseek_formats = [f for f in MODEL_FORMATS if "deepseek" in f.model_pattern]
        assert len(deepseek_formats) >= 2  # thinking and non-thinking

    def test_registry_has_qwen_entry(self):
        """Registry contains Qwen3 format."""
        qwen_formats = [f for f in MODEL_FORMATS if "qwen" in f.model_pattern]
        assert len(qwen_formats) >= 1

    def test_all_formats_have_required_fields(self):
        """All registered formats have required fields."""
        for fmt in MODEL_FORMATS:
            assert fmt.model_pattern
            assert fmt.thinking_mode in ThinkingMode
            assert fmt.training_renderer
            assert fmt.inference_renderer
