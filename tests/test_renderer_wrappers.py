"""Tests for ISF renderer wrappers.

Tests verify that the wrapper classes in shaping.modeling.tinker.renderers:
1. Correctly detect upstream renderer behavior
2. Apply appropriate workarounds (prefill for inference)
3. Produce output matching HF canonical templates
4. Warn when upstream behavior changes
"""

import pytest
import logging
from unittest.mock import patch

from tinker_cookbook import renderers, model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer
from transformers import AutoTokenizer

from shaping.modeling.tinker.renderers import DeepSeekV3, Qwen3, KimiK2, GptOss


class TestDeepSeekV3BehaviorDetection:
    """Tests for DeepSeekV3 wrapper behavior detection."""

    @pytest.fixture
    def deepseek_tokenizer(self):
        """Get tokenizer for DeepSeek V3.1."""
        return get_tokenizer("deepseek-ai/DeepSeek-V3.1")

    def test_detects_upstream_no_prefill(self, deepseek_tokenizer):
        """Detects that upstream renderer does not prefill <think>."""
        wrapper = DeepSeekV3(deepseek_tokenizer)

        # This should match the current expected behavior
        assert wrapper._behavior['upstream_prefills_think'] is False
        assert wrapper.EXPECTED_UPSTREAM_PREFILLS_THINK is False

    def test_behavior_is_cached(self, deepseek_tokenizer):
        """Behavior detection is cached across instances."""
        # Clear cache first
        DeepSeekV3._behavior_cache.clear()

        wrapper1 = DeepSeekV3(deepseek_tokenizer)
        wrapper2 = DeepSeekV3(deepseek_tokenizer)

        # Should use cached behavior
        assert wrapper1._behavior is wrapper2._behavior

    def test_warns_on_unexpected_behavior_change(self, deepseek_tokenizer, caplog):
        """Logs warning if upstream behavior changes from expected."""
        # Clear cache
        DeepSeekV3._behavior_cache.clear()

        # Patch the constant to simulate expectation mismatch
        with patch.object(DeepSeekV3, 'EXPECTED_UPSTREAM_PREFILLS_THINK', True):
            with caplog.at_level(logging.WARNING):
                wrapper = DeepSeekV3(deepseek_tokenizer)

            assert "upstream renderer behavior has changed" in caplog.text
            assert "Expected prefills_think=True" in caplog.text


class TestDeepSeekV3InferencePrompt:
    """Tests for DeepSeekV3 inference prompt generation."""

    @pytest.fixture
    def deepseek_setup(self):
        """Setup DeepSeek V3.1 wrapper and HF tokenizer."""
        tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3.1")
        hf_tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-V3.1",
            trust_remote_code=True
        )
        # Clear cache to ensure fresh detection
        DeepSeekV3._behavior_cache.clear()
        wrapper = DeepSeekV3(tokenizer)
        return wrapper, tokenizer, hf_tokenizer

    def test_inference_prompt_matches_hf_thinking_true(self, deepseek_setup):
        """Inference prompt matches HF thinking=True output.

        This is the critical test: wrapper's build_generation_prompt should
        produce the same output as HF's apply_chat_template with thinking=True.
        """
        wrapper, tokenizer, hf_tokenizer = deepseek_setup

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        # HF reference (thinking=True prefills <think>)
        hf_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, thinking=True
        )

        # Wrapper should match
        model_input = wrapper.build_generation_prompt(messages)
        all_tokens = []
        for chunk in model_input.chunks:
            all_tokens.extend(chunk.tokens)
        wrapper_prompt = tokenizer.decode(all_tokens)

        assert wrapper_prompt == hf_prompt, (
            f"Wrapper prompt doesn't match HF thinking=True!\n"
            f"Wrapper: {repr(wrapper_prompt)}\n"
            f"HF:      {repr(hf_prompt)}"
        )

    def test_inference_prompt_ends_with_think(self, deepseek_setup):
        """Inference prompt ends with <think> for thinking mode."""
        wrapper, tokenizer, _ = deepseek_setup

        messages = [{"role": "user", "content": "Hello"}]
        model_input = wrapper.build_generation_prompt(messages)

        all_tokens = []
        for chunk in model_input.chunks:
            all_tokens.extend(chunk.tokens)
        prompt_text = tokenizer.decode(all_tokens)

        assert prompt_text.endswith("<think>"), (
            f"Prompt should end with <think> but ends with: ...{prompt_text[-50:]}"
        )

    def test_multi_turn_with_thinking_history(self, deepseek_setup):
        """Multi-turn conversation strips thinking from history.

        Uses structured content format with thinking/text parts,
        which is what the renderer expects for thinking stripping.
        """
        wrapper, tokenizer, _ = deepseek_setup

        # Structured content format for assistant message with thinking
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Let me calculate..."},
                {"type": "text", "text": "4"}
            ]},
            {"role": "user", "content": "And 3+3?"},
        ]

        model_input = wrapper.build_generation_prompt(messages)

        all_tokens = []
        for chunk in model_input.chunks:
            all_tokens.extend(chunk.tokens)
        prompt_text = tokenizer.decode(all_tokens)

        # Historical thinking should be stripped
        assert "Let me calculate" not in prompt_text
        # But response content should remain
        assert "4" in prompt_text
        # New generation should have <think> prefill
        assert prompt_text.endswith("<think>")


class TestDeepSeekV3TrainingExample:
    """Tests for DeepSeekV3 training example generation."""

    @pytest.fixture
    def deepseek_tokenizer(self):
        return get_tokenizer("deepseek-ai/DeepSeek-V3.1")

    def test_training_preserves_thinking(self, deepseek_tokenizer):
        """Training examples preserve thinking in history (extension property)."""
        DeepSeekV3._behavior_cache.clear()
        wrapper = DeepSeekV3(deepseek_tokenizer)

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>Let me calculate...</think>4"},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "<think>Adding 3 and 3...</think>6"},
        ]

        model_input, weights = wrapper.build_supervised_example(messages)

        all_tokens = []
        for chunk in model_input.chunks:
            all_tokens.extend(chunk.tokens)
        text = deepseek_tokenizer.decode(all_tokens)

        # Training should preserve thinking traces
        assert "Let me calculate" in text
        assert "Adding 3 and 3" in text


class TestQwen3Wrapper:
    """Tests for Qwen3 wrapper."""

    @pytest.fixture
    def qwen3_setup(self):
        """Setup Qwen3 wrapper and HF tokenizer."""
        tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
        hf_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-30B-A3B",
            trust_remote_code=True
        )
        Qwen3._behavior_cache.clear()
        wrapper = Qwen3(tokenizer)
        return wrapper, tokenizer, hf_tokenizer

    def test_inference_prompt_matches_hf(self, qwen3_setup):
        """Qwen3 wrapper matches HF template."""
        wrapper, tokenizer, hf_tokenizer = qwen3_setup

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        # HF reference
        hf_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Wrapper
        model_input = wrapper.build_generation_prompt(messages)
        all_tokens = []
        for chunk in model_input.chunks:
            all_tokens.extend(chunk.tokens)
        wrapper_prompt = tokenizer.decode(all_tokens)

        assert wrapper_prompt == hf_prompt, (
            f"Qwen3 wrapper doesn't match HF!\n"
            f"Wrapper: {repr(wrapper_prompt)}\n"
            f"HF:      {repr(hf_prompt)}"
        )

    def test_inference_strips_thinking_from_history(self, qwen3_setup):
        """Qwen3 inference strips thinking from history (matches HF enable_thinking)."""
        wrapper, tokenizer, hf_tokenizer = qwen3_setup

        # Structured content format
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Let me calculate..."},
                {"type": "text", "text": "4"}
            ]},
            {"role": "user", "content": "And 3+3?"},
        ]

        model_input = wrapper.build_generation_prompt(messages)
        all_tokens = []
        for chunk in model_input.chunks:
            all_tokens.extend(chunk.tokens)
        prompt_text = tokenizer.decode(all_tokens)

        # Historical thinking should be stripped
        assert "Let me calculate" not in prompt_text
        # But response content should remain
        assert "4" in prompt_text

    def test_multi_turn_matches_hf(self, qwen3_setup):
        """Qwen3 multi-turn with thinking stripped matches HF."""
        wrapper, tokenizer, hf_tokenizer = qwen3_setup

        # Structured content that wrapper will strip
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Let me calculate..."},
                {"type": "text", "text": "4"}
            ]},
            {"role": "user", "content": "And 3+3?"},
        ]

        # Wrapper strips thinking
        model_input = wrapper.build_generation_prompt(messages)
        all_tokens = []
        for chunk in model_input.chunks:
            all_tokens.extend(chunk.tokens)
        wrapper_prompt = tokenizer.decode(all_tokens)

        # HF with already-stripped content (what HF would show after stripping)
        hf_messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        hf_prompt = hf_tokenizer.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

        assert wrapper_prompt == hf_prompt, (
            f"Qwen3 multi-turn doesn't match HF!\n"
            f"Wrapper: {repr(wrapper_prompt)}\n"
            f"HF:      {repr(hf_prompt)}"
        )

    def test_training_preserves_thinking(self, qwen3_setup):
        """Qwen3 training preserves thinking in history."""
        wrapper, tokenizer, _ = qwen3_setup

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>Let me calculate...</think>4"},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "<think>Adding 3 and 3...</think>6"},
        ]

        model_input, weights = wrapper.build_supervised_example(messages)
        all_tokens = []
        for chunk in model_input.chunks:
            all_tokens.extend(chunk.tokens)
        text = tokenizer.decode(all_tokens)

        # Training should preserve thinking traces
        assert "Let me calculate" in text
        assert "Adding 3 and 3" in text


class TestKimiK2Wrapper:
    """Tests for Kimi-K2 wrapper."""

    @pytest.fixture
    def kimi_setup(self):
        """Setup Kimi-K2 wrapper and HF tokenizer."""
        tokenizer = get_tokenizer("moonshotai/Kimi-K2-Thinking")
        hf_tokenizer = AutoTokenizer.from_pretrained(
            "moonshotai/Kimi-K2-Thinking",
            trust_remote_code=True
        )
        KimiK2._behavior_cache.clear()
        wrapper = KimiK2(tokenizer)
        return wrapper, tokenizer, hf_tokenizer

    def test_simple_prompt_matches_hf(self, kimi_setup):
        """Kimi-K2 simple prompt matches HF."""
        wrapper, tokenizer, hf_tokenizer = kimi_setup

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        hf_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_input = wrapper.build_generation_prompt(messages)
        all_tokens = []
        for chunk in model_input.chunks:
            all_tokens.extend(chunk.tokens)
        wrapper_prompt = tokenizer.decode(all_tokens)

        assert wrapper_prompt == hf_prompt

    def test_multi_turn_matches_hf(self, kimi_setup):
        """Kimi-K2 multi-turn matches HF."""
        wrapper, tokenizer, hf_tokenizer = kimi_setup

        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]

        hf_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_input = wrapper.build_generation_prompt(messages)
        all_tokens = []
        for chunk in model_input.chunks:
            all_tokens.extend(chunk.tokens)
        wrapper_prompt = tokenizer.decode(all_tokens)

        assert wrapper_prompt == hf_prompt


class TestGptOssWrapper:
    """Tests for GPT-OSS wrapper."""

    @pytest.fixture
    def gptoss_setup(self):
        """Setup GPT-OSS wrapper and HF tokenizer."""
        tokenizer = get_tokenizer("openai/gpt-oss-120b")
        hf_tokenizer = AutoTokenizer.from_pretrained(
            "openai/gpt-oss-120b",
            trust_remote_code=True
        )
        GptOss._behavior_cache.clear()
        wrapper = GptOss(tokenizer)
        return wrapper, tokenizer, hf_tokenizer

    def test_detects_no_system_message(self, gptoss_setup):
        """Detects that tinker doesn't add HF system message."""
        wrapper, _, _ = gptoss_setup

        assert wrapper._behavior['upstream_adds_system'] is False
        assert wrapper.EXPECTED_UPSTREAM_ADDS_SYSTEM is False

    def test_simple_prompt_uses_developer_role(self, gptoss_setup):
        """GPT-OSS uses developer role for system content."""
        wrapper, tokenizer, _ = gptoss_setup

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        model_input = wrapper.build_generation_prompt(messages)
        all_tokens = []
        for chunk in model_input.chunks:
            all_tokens.extend(chunk.tokens)
        prompt_text = tokenizer.decode(all_tokens)

        # Should use developer role, not add ChatGPT system message
        assert "developer" in prompt_text
        assert "You are ChatGPT" not in prompt_text

    def test_multi_turn_matches_hf_after_developer(self, gptoss_setup):
        """GPT-OSS multi-turn matches HF from developer section onwards.

        The only difference is HF prepends a system message with ChatGPT
        identity. Everything from <|start|>developer onwards should match.
        """
        wrapper, tokenizer, hf_tokenizer = gptoss_setup

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]

        # Wrapper
        model_input = wrapper.build_generation_prompt(messages)
        all_tokens = []
        for chunk in model_input.chunks:
            all_tokens.extend(chunk.tokens)
        wrapper_prompt = tokenizer.decode(all_tokens)

        # HF
        hf_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Find developer section in both
        dev_marker = "<|start|>developer"
        wrapper_from_dev = wrapper_prompt[wrapper_prompt.find(dev_marker):]
        hf_from_dev = hf_prompt[hf_prompt.find(dev_marker):]

        assert wrapper_from_dev == hf_from_dev, (
            f"GPT-OSS doesn't match HF after developer section!\n"
            f"Wrapper: {repr(wrapper_from_dev)}\n"
            f"HF:      {repr(hf_from_dev)}"
        )

    def test_multi_turn_format(self, gptoss_setup):
        """GPT-OSS multi-turn uses correct Harmony format."""
        wrapper, tokenizer, _ = gptoss_setup

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]

        model_input = wrapper.build_generation_prompt(messages)
        all_tokens = []
        for chunk in model_input.chunks:
            all_tokens.extend(chunk.tokens)
        prompt_text = tokenizer.decode(all_tokens)

        # Check Harmony format tokens
        assert "<|start|>" in prompt_text
        assert "<|message|>" in prompt_text
        assert "<|end|>" in prompt_text
        # Check channel is added for assistant messages
        assert "<|channel|>final" in prompt_text
