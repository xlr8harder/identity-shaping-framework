"""ISF renderer wrappers with automatic compatibility detection.

These wrappers ensure consistent behavior between ISF and canonical HuggingFace
chat templates. They detect tinker-cookbook renderer behavior and apply
workarounds when needed.

When behavior changes (upstream fixes or regressions), warnings are logged
and tests should fail, requiring human validation.

Usage:
    from shaping.modeling.tinker.renderers import DeepSeekV3

    renderer = DeepSeekV3(tokenizer)

    # For inference - matches HF production behavior
    prompt = renderer.build_generation_prompt(messages)

    # For training - preserves full thinking context
    example = renderer.build_supervised_example(messages)
"""

import logging
from typing import ClassVar

import tinker
import torch
from tinker_cookbook.renderers import (
    DeepSeekV3ThinkingRenderer,
    Qwen3Renderer,
    GptOssRenderer,
    Message,
)
from tinker_cookbook.renderers.kimi_k2 import KimiK2Renderer
from tinker_cookbook.renderers.base import TrainOnWhat
from tinker_cookbook.tokenizer_utils import Tokenizer

logger = logging.getLogger(__name__)


class DeepSeekV3:
    """ISF wrapper for DeepSeek V3 thinking renderer.

    Detects and applies workarounds for differences between tinker-cookbook
    and canonical HuggingFace behavior:

    - Think prefill: HF prefills '<think>' for generation, tinker may not
    - History stripping: Configured per use case (inference vs training)

    Behavior is detected once per session and cached. When behavior changes
    from expected, a warning is logged and tests should fail for human review.
    """

    _behavior_cache: ClassVar[dict[str, dict[str, bool]]] = {}

    # Expected behaviors - update these when upstream changes are validated
    EXPECTED_UPSTREAM_PREFILLS_THINK = False  # tinker-cookbook doesn't prefill <think>

    def __init__(self, tokenizer: Tokenizer):
        """Initialize renderer wrapper.

        Args:
            tokenizer: Tokenizer from tinker_cookbook.tokenizer_utils
        """
        self.tokenizer = tokenizer

        # Create renderers for different use cases
        # Inference: strip thinking from history (matches HF behavior)
        self._inference_renderer = DeepSeekV3ThinkingRenderer(
            tokenizer, strip_thinking_from_history=True
        )
        # Training: keep thinking in history (extension property for RL)
        self._training_renderer = DeepSeekV3ThinkingRenderer(
            tokenizer, strip_thinking_from_history=False
        )

        # Detect actual behavior
        self._behavior = self._detect_behavior()

    def _detect_behavior(self) -> dict[str, bool]:
        """Detect upstream renderer behavior.

        Probes are run once and cached for the session. If behavior differs
        from expected, logs a warning - tests should catch this for review.
        """
        cache_key = "DeepSeekV3"
        if cache_key in self._behavior_cache:
            return self._behavior_cache[cache_key]

        behavior = {}

        # Detect: does build_generation_prompt prefill <think>?
        upstream_prefills = self._probe_upstream_prefills_think()
        behavior['upstream_prefills_think'] = upstream_prefills

        # Check if behavior matches expectations
        if upstream_prefills != self.EXPECTED_UPSTREAM_PREFILLS_THINK:
            logger.warning(
                "DeepSeekV3: upstream renderer behavior has changed! "
                f"Expected prefills_think={self.EXPECTED_UPSTREAM_PREFILLS_THINK}, "
                f"got {upstream_prefills}. Behavior needs validation."
            )

        self._behavior_cache[cache_key] = behavior
        return behavior

    def _probe_upstream_prefills_think(self) -> bool:
        """Check if upstream renderer prefills <think> for generation prompt."""
        test_messages = [Message(role="user", content="test")]

        # Build generation prompt without our prefill
        prompt = self._inference_renderer.build_generation_prompt(test_messages)

        # Decode to check if <think> is present
        all_tokens = []
        for chunk in prompt.chunks:
            all_tokens.extend(chunk.tokens)

        text = self.tokenizer.decode(all_tokens)
        text_stripped = text.rstrip()

        return text_stripped.endswith("<think>")

    def build_generation_prompt(
        self,
        messages: list[dict],
        role: str = "assistant",
    ) -> tinker.ModelInput:
        """Build prompt for inference/generation.

        Matches canonical HuggingFace behavior:
        - Strips thinking from historical assistant messages
        - Prefills <think> for thinking mode generation

        Args:
            messages: List of message dicts with "role" and "content"
            role: Role of the message to generate (default: assistant)

        Returns:
            tinker.ModelInput ready for sampling
        """
        renderer_messages = [
            Message(role=m["role"], content=m["content"])
            for m in messages
        ]

        # Add prefill if upstream doesn't do it
        prefill = None if self._behavior['upstream_prefills_think'] else "<think>"

        return self._inference_renderer.build_generation_prompt(
            renderer_messages, role=role, prefill=prefill
        )

    def build_supervised_example(
        self,
        messages: list[dict],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """Build tokens and weights for supervised training.

        Preserves thinking traces in history for training (extension property).

        Args:
            messages: List of message dicts with "role" and "content"
            train_on_what: Which tokens to train on (default: last assistant)

        Returns:
            (model_input, weights) tuple for training
        """
        renderer_messages = [
            Message(role=m["role"], content=m["content"])
            for m in messages
        ]

        return self._training_renderer.build_supervised_example(
            renderer_messages, train_on_what=train_on_what
        )

    def parse_response(self, tokens: list[int]) -> tuple[dict, bool]:
        """Parse generated tokens into a message.

        Args:
            tokens: Generated token IDs

        Returns:
            (message_dict, success) tuple
        """
        return self._inference_renderer.parse_response(tokens)

    def get_stop_sequences(self) -> list[int]:
        """Get stop token sequences for generation."""
        return self._inference_renderer.get_stop_sequences()


class Qwen3:
    """ISF wrapper for Qwen3 thinking renderer.

    Like DeepSeek V3, Qwen3 has different requirements for inference vs training:
    - Inference: Strip thinking from history (matches HF enable_thinking=True)
    - Training: Preserve thinking in history (extension property for RL)
    """

    _behavior_cache: ClassVar[dict[str, dict[str, bool]]] = {}

    def __init__(self, tokenizer: Tokenizer):
        """Initialize renderer wrapper."""
        self.tokenizer = tokenizer

        # Create renderers for different use cases
        # Inference: strip thinking from history (matches HF behavior)
        self._inference_renderer = Qwen3Renderer(
            tokenizer, strip_thinking_from_history=True
        )
        # Training: keep thinking in history (extension property for RL)
        self._training_renderer = Qwen3Renderer(
            tokenizer, strip_thinking_from_history=False
        )

        self._behavior = self._detect_behavior()

    def _detect_behavior(self) -> dict[str, bool]:
        """Detect upstream renderer behavior for Qwen3."""
        cache_key = "Qwen3"
        if cache_key in self._behavior_cache:
            return self._behavior_cache[cache_key]

        behavior = {}
        # Qwen3 doesn't need prefill workarounds - just strip_thinking

        self._behavior_cache[cache_key] = behavior
        return behavior

    def build_generation_prompt(
        self,
        messages: list[dict],
        role: str = "assistant",
    ) -> tinker.ModelInput:
        """Build prompt for inference/generation.

        Matches canonical HuggingFace behavior with enable_thinking=True:
        - Strips thinking from historical assistant messages
        """
        renderer_messages = [
            Message(role=m["role"], content=m["content"])
            for m in messages
        ]
        return self._inference_renderer.build_generation_prompt(
            renderer_messages, role=role
        )

    def build_supervised_example(
        self,
        messages: list[dict],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """Build tokens and weights for supervised training.

        Preserves thinking traces in history for training (extension property).
        """
        renderer_messages = [
            Message(role=m["role"], content=m["content"])
            for m in messages
        ]
        return self._training_renderer.build_supervised_example(
            renderer_messages, train_on_what=train_on_what
        )

    def parse_response(self, tokens: list[int]) -> tuple[dict, bool]:
        """Parse generated tokens into a message."""
        return self._inference_renderer.parse_response(tokens)

    def get_stop_sequences(self) -> list[int]:
        """Get stop token sequences for generation."""
        return self._inference_renderer.get_stop_sequences()


class KimiK2:
    """ISF wrapper for Kimi-K2 thinking renderer.

    Kimi-K2 uses a structured thinking format where the thinking content
    is stored separately (reasoning_content in HF). Both tinker and HF
    render historical thinking as empty <think></think> tags.

    No workarounds needed - tinker matches HF behavior.
    """

    _behavior_cache: ClassVar[dict[str, dict[str, bool]]] = {}

    def __init__(self, tokenizer: Tokenizer):
        """Initialize renderer wrapper."""
        self.tokenizer = tokenizer
        self._renderer = KimiK2Renderer(tokenizer)
        self._behavior = self._detect_behavior()

    def _detect_behavior(self) -> dict[str, bool]:
        """Detect upstream renderer behavior for Kimi-K2."""
        cache_key = "KimiK2"
        if cache_key in self._behavior_cache:
            return self._behavior_cache[cache_key]

        behavior = {}
        # Kimi doesn't need workarounds - tinker matches HF

        self._behavior_cache[cache_key] = behavior
        return behavior

    def build_generation_prompt(
        self,
        messages: list[dict],
        role: str = "assistant",
    ) -> tinker.ModelInput:
        """Build prompt for inference/generation."""
        renderer_messages = [
            Message(role=m["role"], content=m["content"])
            for m in messages
        ]
        return self._renderer.build_generation_prompt(renderer_messages, role=role)

    def build_supervised_example(
        self,
        messages: list[dict],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """Build tokens and weights for supervised training."""
        renderer_messages = [
            Message(role=m["role"], content=m["content"])
            for m in messages
        ]
        return self._renderer.build_supervised_example(
            renderer_messages, train_on_what=train_on_what
        )

    def parse_response(self, tokens: list[int]) -> tuple[dict, bool]:
        """Parse generated tokens into a message."""
        return self._renderer.parse_response(tokens)

    def get_stop_sequences(self) -> list[int]:
        """Get stop token sequences for generation."""
        return self._renderer.get_stop_sequences()


class GptOss:
    """ISF wrapper for GPT-OSS (OpenAI's open-source model) renderer.

    Note: HF template adds a system message with ChatGPT identity, knowledge
    cutoff, and reasoning instructions. Tinker's gpt_oss_no_sysprompt renderer
    omits this, using only the developer role for instructions.

    For fine-tuned models, the tinker behavior is typically preferred since
    you're providing your own identity via the developer role.
    """

    _behavior_cache: ClassVar[dict[str, dict[str, bool]]] = {}

    # Expected: tinker does NOT add the HF system message
    EXPECTED_UPSTREAM_ADDS_SYSTEM = False

    def __init__(self, tokenizer: Tokenizer):
        """Initialize renderer wrapper."""
        self.tokenizer = tokenizer
        self._renderer = GptOssRenderer(tokenizer)
        self._behavior = self._detect_behavior()

    def _detect_behavior(self) -> dict[str, bool]:
        """Detect upstream renderer behavior for GPT-OSS."""
        cache_key = "GptOss"
        if cache_key in self._behavior_cache:
            return self._behavior_cache[cache_key]

        behavior = {}

        # Detect: does renderer add the HF-style system message?
        upstream_adds_system = self._probe_upstream_adds_system()
        behavior['upstream_adds_system'] = upstream_adds_system

        if upstream_adds_system != self.EXPECTED_UPSTREAM_ADDS_SYSTEM:
            logger.warning(
                "GptOss: upstream renderer behavior has changed! "
                f"Expected adds_system={self.EXPECTED_UPSTREAM_ADDS_SYSTEM}, "
                f"got {upstream_adds_system}. Behavior needs validation."
            )

        self._behavior_cache[cache_key] = behavior
        return behavior

    def _probe_upstream_adds_system(self) -> bool:
        """Check if upstream renderer adds HF-style system message."""
        test_messages = [Message(role="user", content="test")]
        prompt = self._renderer.build_generation_prompt(test_messages)

        all_tokens = []
        for chunk in prompt.chunks:
            all_tokens.extend(chunk.tokens)

        text = self.tokenizer.decode(all_tokens)
        # HF system message contains "You are ChatGPT"
        return "You are ChatGPT" in text

    def build_generation_prompt(
        self,
        messages: list[dict],
        role: str = "assistant",
    ) -> tinker.ModelInput:
        """Build prompt for inference/generation.

        Uses tinker's renderer which omits the HF default system message.
        This is typically preferred for fine-tuned models.
        """
        renderer_messages = [
            Message(role=m["role"], content=m["content"])
            for m in messages
        ]
        return self._renderer.build_generation_prompt(renderer_messages, role=role)

    def build_supervised_example(
        self,
        messages: list[dict],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """Build tokens and weights for supervised training."""
        renderer_messages = [
            Message(role=m["role"], content=m["content"])
            for m in messages
        ]
        return self._renderer.build_supervised_example(
            renderer_messages, train_on_what=train_on_what
        )

    def parse_response(self, tokens: list[int]) -> tuple[dict, bool]:
        """Parse generated tokens into a message."""
        return self._renderer.parse_response(tokens)

    def get_stop_sequences(self) -> list[int]:
        """Get stop token sequences for generation."""
        return self._renderer.get_stop_sequences()
