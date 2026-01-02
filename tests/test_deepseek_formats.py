"""Tests for DeepSeek V3.1 message formatting.

Documents the canonical HF reference behavior vs tinker training behavior.

Key insight: Inference and training have intentionally different formats:

INFERENCE (HF reference):
- Historical thinking is STRIPPED (user doesn't need to re-read old reasoning)
- Historical turns get </think> prefix: <｜Assistant｜></think>response
- New generation gets <think> prefix: <｜Assistant｜><think>
- This is the canonical format for chat/inference

TRAINING (tinker):
- Thinking is PRESERVED in all turns (model needs to learn from traces)
- Full format: <｜Assistant｜><think>reasoning</think>response
- This is correct for SFT and multi-turn RL workflows

These are both correct for their respective use cases.
"""

import pytest
from transformers import AutoTokenizer
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer


class TestDeepSeekHFReference:
    """Tests documenting the canonical HuggingFace reference behavior.

    This is the authoritative format for inference/chat.
    """

    @pytest.fixture
    def hf_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            'deepseek-ai/DeepSeek-V3.1',
            trust_remote_code=True
        )

    def test_generation_prompt_thinking_mode(self, hf_tokenizer):
        """Generation prompt with thinking=True prefills <think>."""
        msgs = [
            {'role': 'system', 'content': 'You are helpful.'},
            {'role': 'user', 'content': 'What is 2+2?'},
        ]

        result = hf_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, thinking=True
        )

        # Must end with <think> to trigger thinking mode
        assert result.endswith('<think>'), (
            f"Generation prompt must end with <think> for thinking mode.\n"
            f"Got: {repr(result)}"
        )
        assert '<｜Assistant｜><think>' in result

    def test_generation_prompt_non_thinking_mode(self, hf_tokenizer):
        """Generation prompt with thinking=False prefills </think>."""
        msgs = [
            {'role': 'user', 'content': 'What is 2+2?'},
        ]

        result = hf_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, thinking=False
        )

        # Must end with </think> to skip thinking
        assert result.endswith('</think>'), (
            f"Non-thinking mode must end with </think>.\n"
            f"Got: {repr(result)}"
        )

    def test_historical_thinking_is_stripped(self, hf_tokenizer):
        """Historical assistant messages have thinking STRIPPED.

        This is intentional - users don't need to re-read old reasoning
        when continuing a conversation.
        """
        msgs = [
            {'role': 'user', 'content': 'What is 2+2?'},
            {'role': 'assistant', 'content': '<think>Let me calculate: 2+2=4</think>The answer is 4.'},
            {'role': 'user', 'content': 'And 3+3?'},
        ]

        result = hf_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, thinking=True
        )

        # Historical thinking should be stripped
        assert 'Let me calculate' not in result, (
            f"Historical thinking should be stripped!\n"
            f"Got: {repr(result)}"
        )

        # Response should remain
        assert 'The answer is 4.' in result

        # Historical turn gets </think> prefix (not <think>)
        assert '<｜Assistant｜></think>The answer is 4.' in result

        # New generation still gets <think> prefix
        assert result.endswith('<｜Assistant｜><think>')

    def test_multi_turn_format(self, hf_tokenizer):
        """Verify complete multi-turn format matches documentation."""
        msgs = [
            {'role': 'system', 'content': 'System prompt'},
            {'role': 'user', 'content': 'Query 1'},
            {'role': 'assistant', 'content': '<think>Reasoning 1</think>Response 1'},
            {'role': 'user', 'content': 'Query 2'},
        ]

        result = hf_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, thinking=True
        )

        # Expected format per documentation:
        # <｜begin▁of▁sentence｜>{system_prompt}
        # <｜User｜>{query_1}<｜Assistant｜></think>{response_1}<｜end▁of▁sentence｜>
        # <｜User｜>{query_2}<｜Assistant｜><think>

        expected_parts = [
            '<｜begin▁of▁sentence｜>System prompt',
            '<｜User｜>Query 1<｜Assistant｜></think>Response 1<｜end▁of▁sentence｜>',
            '<｜User｜>Query 2<｜Assistant｜><think>',
        ]

        for part in expected_parts:
            assert part in result, (
                f"Missing expected part: {repr(part)}\n"
                f"Full result: {repr(result)}"
            )

    def test_input_format_with_think_tags(self, hf_tokenizer):
        """Document: you SHOULD pass <think>...</think> in assistant content.

        The template handles stripping and reformatting automatically.
        """
        # Correct input format
        msgs = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': '<think>Thinking here</think>Response here'},
            {'role': 'user', 'content': 'Follow up'},
        ]

        result = hf_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, thinking=True
        )

        # Template processes correctly
        assert 'Response here' in result
        assert 'Thinking here' not in result  # Stripped from history


class TestTinkerTrainingFormat:
    """Tests documenting tinker training format.

    This preserves thinking for SFT and multi-turn RL workflows.
    """

    @pytest.fixture
    def tokenizer(self):
        return get_tokenizer('deepseek-ai/DeepSeek-V3.1')

    @pytest.fixture
    def thinking_renderer(self, tokenizer):
        return renderers.get_renderer(name='deepseekv3_thinking', tokenizer=tokenizer)

    def test_training_preserves_thinking(self, thinking_renderer, tokenizer):
        """Training format PRESERVES full thinking traces.

        This is intentional - the model needs to learn from traces.
        """
        msgs = [
            renderers.Message(role='user', content='What is 2+2?'),
            renderers.Message(role='assistant', content='<think>Calculate: 2+2=4</think>The answer is 4.'),
        ]

        example = thinking_renderer.build_supervised_example(msgs)
        model_input = example[0]

        all_tokens = []
        for chunk in model_input.chunks:
            all_tokens.extend(chunk.tokens)
        decoded = tokenizer.decode(all_tokens)

        # Thinking is PRESERVED for training
        assert '<think>Calculate: 2+2=4</think>' in decoded, (
            f"Training format must preserve thinking!\n"
            f"Got: {repr(decoded)}"
        )
        assert 'The answer is 4.' in decoded

    def test_training_full_format(self, thinking_renderer, tokenizer):
        """Verify complete training format."""
        msgs = [
            renderers.Message(role='user', content='What is 2+2?'),
            renderers.Message(role='assistant', content='<think>Reasoning</think>Answer'),
        ]

        example = thinking_renderer.build_supervised_example(msgs)
        model_input = example[0]

        all_tokens = []
        for chunk in model_input.chunks:
            all_tokens.extend(chunk.tokens)
        decoded = tokenizer.decode(all_tokens)

        # Full training sequence
        expected = '<｜begin▁of▁sentence｜><｜User｜>What is 2+2?<｜Assistant｜><think>Reasoning</think>Answer<｜end▁of▁sentence｜>'
        assert decoded == expected, (
            f"Training format mismatch!\n"
            f"Expected: {repr(expected)}\n"
            f"Got: {repr(decoded)}"
        )

    def test_generation_prompt_no_think_prefix(self, thinking_renderer, tokenizer):
        """Tinker generation prompt does NOT prefill <think>.

        This differs from HF reference! The model generates <think> itself.
        This may cause inconsistent thinking behavior during inference.

        For inference, prefer HF template with thinking=True.
        """
        msgs = [
            renderers.Message(role='user', content='What is 2+2?'),
        ]

        prompt = thinking_renderer.build_generation_prompt(msgs)

        all_tokens = []
        for chunk in prompt.chunks:
            all_tokens.extend(chunk.tokens)
        decoded = tokenizer.decode(all_tokens)

        # Tinker does NOT add <think> prefix
        assert decoded.endswith('<｜Assistant｜>'), (
            f"Tinker generation prompt ends at Assistant token.\n"
            f"Got: {repr(decoded)}"
        )
        assert not decoded.endswith('<think>'), (
            "Tinker does NOT prefill <think> - differs from HF reference!"
        )

    def test_strip_thinking_from_history_option(self, tokenizer):
        """Tinker renderer has strip_thinking_from_history option.

        - True (default): strips thinking from historical turns (like HF inference)
        - False: preserves thinking in history (for multi-turn RL)
        """
        # With strip_thinking_from_history=True (default)
        renderer_strip = renderers.DeepSeekV3ThinkingRenderer(
            tokenizer=tokenizer,
            strip_thinking_from_history=True,
        )

        # With strip_thinking_from_history=False
        renderer_preserve = renderers.DeepSeekV3ThinkingRenderer(
            tokenizer=tokenizer,
            strip_thinking_from_history=False,
        )

        msgs = [
            renderers.Message(role='user', content='Q1'),
            renderers.Message(role='assistant', content=[
                {'type': 'thinking', 'thinking': 'Reasoning 1'},
                {'type': 'text', 'text': 'Answer 1'}
            ]),
            renderers.Message(role='user', content='Q2'),
        ]

        # Build generation prompts
        prompt_strip = renderer_strip.build_generation_prompt(msgs)
        prompt_preserve = renderer_preserve.build_generation_prompt(msgs)

        tokens_strip = []
        for chunk in prompt_strip.chunks:
            tokens_strip.extend(chunk.tokens)
        decoded_strip = tokenizer.decode(tokens_strip)

        tokens_preserve = []
        for chunk in prompt_preserve.chunks:
            tokens_preserve.extend(chunk.tokens)
        decoded_preserve = tokenizer.decode(tokens_preserve)

        # Stripping: no thinking in history
        assert 'Reasoning 1' not in decoded_strip

        # Preserving: thinking remains
        assert 'Reasoning 1' in decoded_preserve or '<think>' in decoded_preserve


class TestFormatCompatibility:
    """Tests verifying training and inference formats are compatible."""

    @pytest.fixture
    def hf_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            'deepseek-ai/DeepSeek-V3.1',
            trust_remote_code=True
        )

    @pytest.fixture
    def tinker_tokenizer(self):
        return get_tokenizer('deepseek-ai/DeepSeek-V3.1')

    @pytest.fixture
    def thinking_renderer(self, tinker_tokenizer):
        return renderers.get_renderer(name='deepseekv3_thinking', tokenizer=tinker_tokenizer)

    def test_training_teaches_think_generation(self, thinking_renderer, tinker_tokenizer):
        """Training teaches model: after <｜Assistant｜>, generate <think>.

        This is compatible with HF inference which prefills <think>.
        """
        msgs = [
            renderers.Message(role='user', content='Question'),
            renderers.Message(role='assistant', content='<think>Reasoning</think>Answer'),
        ]

        example = thinking_renderer.build_supervised_example(msgs)
        model_input = example[0]

        all_tokens = []
        for chunk in model_input.chunks:
            all_tokens.extend(chunk.tokens)
        decoded = tinker_tokenizer.decode(all_tokens)

        # Training sequence has <｜Assistant｜><think>
        assert '<｜Assistant｜><think>' in decoded, (
            "Model learns: after Assistant token, generate <think>"
        )

    def test_inference_prefill_matches_training(self, hf_tokenizer):
        """HF inference prefill (<think>) matches what training teaches.

        Training: model sees <｜Assistant｜><think>...
        Inference: we prefill <｜Assistant｜><think>, model continues

        These are compatible.
        """
        msgs = [
            {'role': 'user', 'content': 'Question'},
        ]

        result = hf_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, thinking=True
        )

        # Inference prefills <think> - same pattern model learned
        assert result.endswith('<｜Assistant｜><think>'), (
            f"Inference should prefill <think> (matches training pattern).\n"
            f"Got: {repr(result)}"
        )


class TestNonThinkingRenderer:
    """Tests for non-thinking (standard) mode."""

    @pytest.fixture
    def tokenizer(self):
        return get_tokenizer('deepseek-ai/DeepSeek-V3.1')

    @pytest.fixture
    def non_thinking_renderer(self, tokenizer):
        return renderers.get_renderer(name='deepseekv3', tokenizer=tokenizer)

    def test_non_thinking_generation_adds_close_tag(self, non_thinking_renderer, tokenizer):
        """Non-thinking mode adds </think> to skip reasoning."""
        msgs = [
            renderers.Message(role='user', content='Hello'),
        ]

        prompt = non_thinking_renderer.build_generation_prompt(msgs)

        all_tokens = []
        for chunk in prompt.chunks:
            all_tokens.extend(chunk.tokens)
        decoded = tokenizer.decode(all_tokens)

        # Ends with </think> to skip reasoning
        assert '</think>' in decoded
        assert decoded.endswith('</think>') or '<｜Assistant｜></think>' in decoded

    def test_non_thinking_matches_hf_thinking_false(self, non_thinking_renderer, tokenizer):
        """Tinker non-thinking renderer matches HF thinking=False."""
        hf_tokenizer = AutoTokenizer.from_pretrained(
            'deepseek-ai/DeepSeek-V3.1',
            trust_remote_code=True
        )

        msgs_hf = [
            {'role': 'user', 'content': 'Hello'},
        ]

        hf_result = hf_tokenizer.apply_chat_template(
            msgs_hf, tokenize=False, add_generation_prompt=True, thinking=False
        )

        msgs_tinker = [
            renderers.Message(role='user', content='Hello'),
        ]

        prompt = non_thinking_renderer.build_generation_prompt(msgs_tinker)
        all_tokens = []
        for chunk in prompt.chunks:
            all_tokens.extend(chunk.tokens)
        tinker_result = tokenizer.decode(all_tokens)

        assert hf_result == tinker_result, (
            f"Non-thinking renderer should match HF thinking=False.\n"
            f"HF: {repr(hf_result)}\n"
            f"Tinker: {repr(tinker_result)}"
        )
