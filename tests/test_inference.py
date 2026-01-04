"""Tests for shaping.modeling module.

Tests cover:
- LLMClient message building
- TinkerClient response parsing (structured content from renderers)
- Renderer vs HF template consistency (critical for training/inference parity)
- Config resolution (resolve_checkpoint, resolve_model)
"""

import pytest
from unittest.mock import MagicMock, patch

# tinker and tinker_cookbook are mandatory dependencies for ISF
import tinker
from tinker_cookbook import renderers, model_info
from tinker_cookbook.tokenizer_utils import get_tokenizer


class TestTinkerClientResponseParsing:
    """Test TinkerClient._parse_response handles various renderer outputs."""

    @pytest.fixture
    def mock_client(self):
        """Create a TinkerClient with mocked tinker dependencies."""
        with patch("shaping.modeling.tinker.client.tinker"):
            with patch("shaping.modeling.tinker.client.renderers") as mock_renderers:
                with patch("shaping.modeling.tinker.client.model_info") as mock_model_info:
                    with patch("shaping.modeling.tinker.client.get_tokenizer") as mock_tokenizer:
                        # Setup mocks
                        mock_model_info.get_recommended_renderer_name.return_value = "qwen3"
                        mock_tokenizer.return_value = MagicMock()
                        mock_renderer = MagicMock()
                        mock_renderers.get_renderer.return_value = mock_renderer
                        mock_renderer.get_stop_sequences.return_value = [151645]

                        from shaping.modeling.tinker import TinkerClient

                        # Create client (will use mocks)
                        client = TinkerClient.__new__(TinkerClient)
                        client._use_hf_template = False
                        client.renderer = mock_renderer
                        client.tokenizer = mock_tokenizer.return_value

                        yield client, mock_renderer

    def test_parse_structured_thinking_content(self, mock_client):
        """Renderer returns structured content with thinking and text parts."""
        client, mock_renderer = mock_client

        # Qwen3 renderer returns this format
        mock_renderer.parse_response.return_value = [{
            "content": [
                {"type": "thinking", "thinking": "Let me reason about this..."},
                {"type": "text", "text": "The answer is 42."}
            ]
        }]

        result = client._parse_response([1, 2, 3])  # tokens don't matter, renderer is mocked

        assert result == "<think>Let me reason about this...</think>The answer is 42."

    def test_parse_structured_text_only(self, mock_client):
        """Renderer returns structured content with only text part."""
        client, mock_renderer = mock_client

        mock_renderer.parse_response.return_value = [{
            "content": [
                {"type": "text", "text": "Just a response without thinking."}
            ]
        }]

        result = client._parse_response([1, 2, 3])

        assert result == "Just a response without thinking."

    def test_parse_plain_string_content(self, mock_client):
        """Renderer returns plain string content (older format)."""
        client, mock_renderer = mock_client

        mock_renderer.parse_response.return_value = [{
            "content": "Plain string response"
        }]

        result = client._parse_response([1, 2, 3])

        assert result == "Plain string response"

    def test_parse_empty_response(self, mock_client):
        """Renderer returns empty/None response."""
        client, mock_renderer = mock_client

        mock_renderer.parse_response.return_value = None

        result = client._parse_response([1, 2, 3])

        assert result == ""

    def test_parse_fallback_for_unknown_dict_format(self, mock_client):
        """Unknown dict format falls back to str() representation."""
        client, mock_renderer = mock_client

        mock_renderer.parse_response.return_value = [{
            "content": [
                {"unknown_key": "unknown_value"}
            ]
        }]

        result = client._parse_response([1, 2, 3])

        # Should contain the dict as string (fallback behavior)
        assert "unknown_key" in result

    def test_parse_mixed_content_types(self, mock_client):
        """Content list with mixed types (dict and string)."""
        client, mock_renderer = mock_client

        mock_renderer.parse_response.return_value = [{
            "content": [
                {"type": "thinking", "thinking": "Reasoning"},
                "plain string in list",
                {"type": "text", "text": "Final answer"}
            ]
        }]

        result = client._parse_response([1, 2, 3])

        assert "<think>Reasoning</think>" in result
        assert "plain string in list" in result
        assert "Final answer" in result


class TestLLMClientMessageBuilding:
    """Test LLMClient message building and sysprompt injection."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create LLMClient with mocked mq dependencies."""
        with patch("shaping.modeling.clients.mq_store") as mock_store:
            with patch("shaping.modeling.clients.get_provider") as mock_get_provider:
                mock_store.get_model.return_value = {
                    "provider": "openrouter",
                    "model": "qwen/qwen3-30b",
                    "sysprompt": "You are Aria."
                }
                mock_get_provider.return_value = MagicMock()

                from shaping.modeling.clients import LLMClient
                client = LLMClient("test-model")
                yield client

    def test_injects_sysprompt(self, mock_llm_client):
        """Sysprompt is injected at start of messages."""
        messages = [{"role": "user", "content": "Hello"}]

        built = mock_llm_client._build_messages(messages)

        assert len(built) == 2
        assert built[0]["role"] == "system"
        assert built[0]["content"] == "You are Aria."
        assert built[1]["role"] == "user"

    def test_no_double_sysprompt(self, mock_llm_client):
        """Doesn't add sysprompt if already present."""
        messages = [
            {"role": "system", "content": "Custom sysprompt"},
            {"role": "user", "content": "Hello"}
        ]

        built = mock_llm_client._build_messages(messages)

        assert len(built) == 2
        assert built[0]["content"] == "Custom sysprompt"

    def test_sysprompt_override(self):
        """Sysprompt override replaces model's configured sysprompt."""
        with patch("shaping.modeling.clients.mq_store") as mock_store:
            with patch("shaping.modeling.clients.get_provider"):
                mock_store.get_model.return_value = {
                    "provider": "openrouter",
                    "model": "qwen/qwen3-30b",
                    "sysprompt": "Original sysprompt"
                }

                from shaping.modeling.clients import LLMClient
                client = LLMClient("test-model", sysprompt_override="Override sysprompt")

                messages = [{"role": "user", "content": "Hello"}]
                built = client._build_messages(messages)

                assert built[0]["content"] == "Override sysprompt"

    def test_no_sysprompt_configured(self):
        """Works correctly when no sysprompt configured."""
        with patch("shaping.modeling.clients.mq_store") as mock_store:
            with patch("shaping.modeling.clients.get_provider"):
                mock_store.get_model.return_value = {
                    "provider": "openrouter",
                    "model": "qwen/qwen3-30b"
                    # No sysprompt
                }

                from shaping.modeling.clients import LLMClient
                client = LLMClient("test-model")

                messages = [{"role": "user", "content": "Hello"}]
                built = client._build_messages(messages)

                assert len(built) == 1
                assert built[0]["role"] == "user"


class TestTinkerBackendSysprompt:
    """Test TinkerBackend sysprompt injection."""

    def test_injects_sysprompt(self):
        """TinkerBackend prepends sysprompt as system message."""
        from unittest.mock import MagicMock, patch
        from dispatcher.taskmanager.backend.request import Request

        # Create a mock TinkerClient that captures the messages it receives
        captured_messages = []

        def mock_query(messages):
            captured_messages.extend(messages)
            return "<think>thinking</think>test response"

        with patch("shaping.modeling.backends.TinkerClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.query = mock_query
            mock_client_cls.return_value = mock_client

            from shaping.modeling.backends import TinkerBackend

            backend = TinkerBackend(
                base_model="test/model",
                sysprompt="You are a test assistant.",
            )

            request = Request({"messages": [{"role": "user", "content": "Hello"}]})
            backend.process(request)

            # Verify sysprompt was injected
            assert len(captured_messages) == 2
            assert captured_messages[0]["role"] == "system"
            assert captured_messages[0]["content"] == "You are a test assistant."
            assert captured_messages[1]["role"] == "user"
            assert captured_messages[1]["content"] == "Hello"

    def test_no_sysprompt_when_none(self):
        """TinkerBackend doesn't add system message when sysprompt is None."""
        from unittest.mock import MagicMock, patch
        from dispatcher.taskmanager.backend.request import Request

        captured_messages = []

        def mock_query(messages):
            captured_messages.extend(messages)
            return "<think>thinking</think>test response"

        with patch("shaping.modeling.backends.TinkerClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.query = mock_query
            mock_client_cls.return_value = mock_client

            from shaping.modeling.backends import TinkerBackend

            backend = TinkerBackend(
                base_model="test/model",
                sysprompt=None,
            )

            request = Request({"messages": [{"role": "user", "content": "Hello"}]})
            backend.process(request)

            # Verify no sysprompt was injected
            assert len(captured_messages) == 1
            assert captured_messages[0]["role"] == "user"


class TestConfigResolution:
    """Test config.py model and checkpoint resolution."""

    def test_resolve_model_passthrough(self):
        """Non-isf model refs pass through unchanged."""
        from shaping.config import resolve_model

        result = resolve_model("aria-v0.9-full")
        assert result == "aria-v0.9-full"

        result = resolve_model("gpt-4o-mini")
        assert result == "gpt-4o-mini"

    def test_resolve_model_identity(self):
        """isf.identity.* resolves to {prefix}-{tier}-{variant}."""
        from shaping.config import ISFConfig

        # With default config (no isf.yaml)
        config = ISFConfig()

        result = config.resolve_model("isf.identity.full")
        assert result == "identity-dev-full"

    def test_resolve_model_plain_model(self):
        """isf.{model}.* resolves to plain model name from models section."""
        from shaping.config import ISFConfig

        # Create config with a judge model defined
        config = ISFConfig()
        config._config["models"]["judge"] = {"provider": "openrouter", "model": "gpt-4o-mini"}

        result = config.resolve_model("isf.judge.default")
        assert result == "judge"

    def test_resolve_checkpoint_explicit_format(self):
        """Explicit format: model::renderer or model::renderer::path."""
        from shaping.config import resolve_checkpoint

        base, renderer, path = resolve_checkpoint("Qwen/Qwen3-30B::qwen3")
        assert base == "Qwen/Qwen3-30B"
        assert renderer == "qwen3"
        assert path is None

        base, renderer, path = resolve_checkpoint("Qwen/Qwen3-30B::qwen3::/path/to/checkpoint")
        assert base == "Qwen/Qwen3-30B"
        assert renderer == "qwen3"
        assert path == "/path/to/checkpoint"

    def test_resolve_checkpoint_invalid_format(self):
        """Invalid format raises ValueError."""
        from shaping.config import resolve_checkpoint

        with pytest.raises(ValueError, match="Invalid explicit format"):
            resolve_checkpoint("a::b::c::d")


class TestRendererHFTemplateConsistency:
    """Compare tinker_cookbook renderers with HF chat templates.

    These tests verify that our renderers produce equivalent output to
    the HuggingFace tokenizer's apply_chat_template. Mismatches here
    indicate potential training/inference skew.
    """

    @pytest.fixture
    def qwen3_setup(self):
        """Setup Qwen3 renderer and HF tokenizer for comparison."""
        from transformers import AutoTokenizer

        base_model = "Qwen/Qwen3-30B-A3B"
        renderer_name = model_info.get_recommended_renderer_name(base_model)

        tokenizer = get_tokenizer(base_model)
        renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
        hf_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        return renderer, tokenizer, hf_tokenizer

    @pytest.fixture
    def deepseek_v3_setup(self):
        """Setup DeepSeek V3.1 renderers and HF tokenizer for comparison."""
        from transformers import AutoTokenizer

        base_model = "deepseek-ai/DeepSeek-V3.1"

        tokenizer = get_tokenizer(base_model)
        thinking_renderer = renderers.get_renderer(name="deepseekv3_thinking", tokenizer=tokenizer)
        non_thinking_renderer = renderers.get_renderer(name="deepseekv3", tokenizer=tokenizer)
        hf_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        return thinking_renderer, non_thinking_renderer, tokenizer, hf_tokenizer

    def test_qwen3_simple_conversation(self, qwen3_setup):
        """Qwen3 renderer matches HF template for simple conversation."""
        renderer, tokenizer, hf_tokenizer = qwen3_setup

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]

        # HF template
        hf_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Renderer
        from tinker_cookbook import renderers as r
        renderer_msgs = [r.Message(role=m["role"], content=m["content"]) for m in messages]
        prompt = renderer.build_generation_prompt(renderer_msgs)

        # Decode renderer output
        all_tokens = []
        for chunk in prompt.chunks:
            all_tokens.extend(chunk.tokens)
        renderer_prompt = tokenizer.decode(all_tokens)

        assert renderer_prompt == hf_prompt, (
            f"Renderer/HF mismatch!\n"
            f"HF:       {repr(hf_prompt)}\n"
            f"Renderer: {repr(renderer_prompt)}"
        )

    def test_qwen3_multi_turn(self, qwen3_setup):
        """Qwen3 renderer matches HF template for multi-turn conversation."""
        renderer, tokenizer, hf_tokenizer = qwen3_setup

        messages = [
            {"role": "system", "content": "You are Aria."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
            {"role": "user", "content": "Tell me about yourself."},
        ]

        # HF template
        hf_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Renderer
        from tinker_cookbook import renderers as r
        renderer_msgs = [r.Message(role=m["role"], content=m["content"]) for m in messages]
        prompt = renderer.build_generation_prompt(renderer_msgs)

        all_tokens = []
        for chunk in prompt.chunks:
            all_tokens.extend(chunk.tokens)
        renderer_prompt = tokenizer.decode(all_tokens)

        assert renderer_prompt == hf_prompt

    def test_deepseek_v3_non_thinking_mode(self, deepseek_v3_setup):
        """DeepSeek V3.1 non-thinking renderer matches HF thinking=False.

        Non-thinking mode uses </think> prefix to skip reasoning.
        """
        thinking_renderer, non_thinking_renderer, tokenizer, hf_tokenizer = deepseek_v3_setup

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        # HF template with thinking=False
        hf_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, thinking=False
        )

        # Non-thinking renderer
        renderer_msgs = [renderers.Message(role=m["role"], content=m["content"]) for m in messages]
        prompt = non_thinking_renderer.build_generation_prompt(renderer_msgs)

        all_tokens = []
        for chunk in prompt.chunks:
            all_tokens.extend(chunk.tokens)
        renderer_prompt = tokenizer.decode(all_tokens)

        assert renderer_prompt == hf_prompt, (
            f"Non-thinking mode mismatch!\n"
            f"HF (thinking=False): {repr(hf_prompt)}\n"
            f"Renderer (deepseekv3): {repr(renderer_prompt)}"
        )

    def test_deepseek_v3_thinking_mode_difference(self, deepseek_v3_setup):
        """Document that DeepSeek V3.1 thinking renderer differs from HF.

        IMPORTANT: tinker_cookbook's deepseekv3_thinking renderer does NOT add
        <think> prefix, while HF's thinking=True does. This is intentional -
        the model learns to generate <think> itself during training.

        This test documents the difference rather than asserting equality.
        """
        thinking_renderer, non_thinking_renderer, tokenizer, hf_tokenizer = deepseek_v3_setup

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        # HF template with thinking=True (adds <think> prefix)
        hf_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, thinking=True
        )

        # Thinking renderer (does NOT add <think> prefix)
        renderer_msgs = [renderers.Message(role=m["role"], content=m["content"]) for m in messages]
        prompt = thinking_renderer.build_generation_prompt(renderer_msgs)

        all_tokens = []
        for chunk in prompt.chunks:
            all_tokens.extend(chunk.tokens)
        renderer_prompt = tokenizer.decode(all_tokens)

        # Document the expected difference
        assert hf_prompt.endswith("<think>"), "HF thinking=True should end with <think>"
        assert not renderer_prompt.endswith("<think>"), "Renderer should NOT add <think>"
        assert renderer_prompt.endswith("<｜Assistant｜>"), "Renderer ends with Assistant token"

        # The difference is exactly the <think> suffix
        assert hf_prompt == renderer_prompt + "<think>", (
            f"Unexpected difference between HF and renderer!\n"
            f"HF: {repr(hf_prompt)}\n"
            f"Renderer: {repr(renderer_prompt)}"
        )

    def test_deepseek_v3_thinking_with_history(self, deepseek_v3_setup):
        """Test multi-turn DeepSeek V3.1 with thinking in history.

        When strip_thinking_from_history=True (default), thinking blocks
        should be removed from historical assistant messages.
        """
        thinking_renderer, non_thinking_renderer, tokenizer, hf_tokenizer = deepseek_v3_setup

        # Conversation with thinking in history
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Let me calculate..."},
                {"type": "text", "text": "4"}
            ]},
            {"role": "user", "content": "And 3+3?"},
        ]

        # Build prompt - thinking should be stripped from historical message
        renderer_msgs = [renderers.Message(role=m["role"], content=m["content"]) for m in messages]
        prompt = thinking_renderer.build_generation_prompt(renderer_msgs)

        all_tokens = []
        for chunk in prompt.chunks:
            all_tokens.extend(chunk.tokens)
        rendered = tokenizer.decode(all_tokens)

        # Historical thinking should be stripped
        assert "Let me calculate" not in rendered, (
            f"Historical thinking should be stripped!\n"
            f"Rendered: {repr(rendered)}"
        )
        # But the text response should remain
        assert "4" in rendered
