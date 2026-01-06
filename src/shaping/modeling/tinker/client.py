"""Tinker model client for inference.

Provides TinkerClient for trained models via tinker sampling.
Requires tinker and tinker_cookbook packages (TM internal).
"""

import asyncio
from typing import Optional, TYPE_CHECKING

from ..model_formats import get_model_format, ThinkingMode
from ..defaults import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS

if TYPE_CHECKING:
    import tinker

# Optional tinker imports (TM internal package)
try:
    import tinker
    from tinker import types as tinker_types
    from tinker_cookbook import renderers, model_info
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False
    tinker = None
    tinker_types = None
    renderers = None
    model_info = None
    get_tokenizer = None


class TinkerClient:
    """Client for tinker-trained models.

    Requires tinker and tinker_cookbook packages (TM internal).

    Example:
        # From experiment checkpoint
        client = TinkerClient.from_checkpoint("e027-final")

        # Or explicit config
        client = TinkerClient(
            base_model="Qwen/Qwen3-30B-A3B",
            model_path="/path/to/checkpoint",
            renderer_name="qwen3"
        )

        response = await client.query_async([...])
    """

    def __init__(
        self,
        base_model: str,
        model_path: Optional[str] = None,
        renderer_name: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """Initialize client with model configuration.

        Args:
            base_model: HuggingFace model name (e.g., "Qwen/Qwen3-30B-A3B")
            model_path: Path to fine-tuned checkpoint (None for base model)
            renderer_name: Renderer name for chat template (auto-detected if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Raises:
            ImportError: If tinker packages are not installed
        """
        if not TINKER_AVAILABLE:
            raise ImportError(
                "TinkerClient requires tinker and tinker_cookbook packages. "
                "These are TM internal packages."
            )

        self.base_model = base_model
        self.model_path = model_path

        # Get model format configuration
        self._model_format = get_model_format(base_model, thinking=True)
        self.renderer_name = renderer_name or self._model_format.inference_renderer
        self._use_hf_template = self._model_format.use_hf_for_inference

        # Initialize tinker clients
        self.service_client = tinker.ServiceClient()
        if model_path:
            self.sampling_client = self.service_client.create_sampling_client(
                model_path=model_path,
                base_model=base_model,
            )
        else:
            self.sampling_client = self.service_client.create_sampling_client(
                base_model=base_model,
            )

        # Initialize tokenizer and renderer
        self.tokenizer = get_tokenizer(base_model)
        self.renderer = self._model_format.get_inference_renderer(self.tokenizer)

        if self._use_hf_template:
            from transformers import AutoTokenizer

            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                base_model, trust_remote_code=True
            )
            self._stop_sequences = ["<｜end▁of▁sentence｜>", "<｜User｜>"]
        else:
            self._stop_sequences = None

        stop_seqs = (
            self._stop_sequences
            if self._use_hf_template
            else self.renderer.get_stop_sequences()
        )
        self.sampling_params = tinker_types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_seqs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        spec: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> "TinkerClient":
        """Create client from experiment checkpoint spec.

        Args:
            spec: Checkpoint spec like "e027-final" or "e027-000192"
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            TinkerClient configured for that checkpoint
        """
        from ...config import resolve_checkpoint

        base_model, renderer_name, model_path = resolve_checkpoint(spec)
        return cls(
            base_model=base_model,
            model_path=model_path,
            renderer_name=renderer_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _build_hf_prompt(self, messages: list[dict]) -> "tinker.ModelInput":
        """Build prompt using HuggingFace chat template.

        Uses the model_format configuration to determine proper template params.
        For thinking models like DeepSeek V3.1, this handles:
        - Stripping historical thinking from assistant messages
        - Prefilling <think> for new generation
        """
        msg_dicts = []
        for m in messages:
            role = m["role"]
            content = m["content"]

            # For thinking models, historical assistant messages need processing
            # The HF template expects <think>...</think> in content and strips it
            if (
                role == "assistant"
                and self._model_format.thinking_mode == ThinkingMode.EMBEDDED
            ):
                # Keep the content as-is - HF template handles stripping
                pass

            msg_dicts.append({"role": role, "content": content})

        # Use model format to build the prompt
        formatted = self._model_format.build_hf_inference_prompt(
            msg_dicts, self._hf_tokenizer, add_generation_prompt=True
        )

        if formatted.endswith("<think>"):
            formatted += "\n"

        tokens = self._hf_tokenizer.encode(formatted, add_special_tokens=False)
        chunk = tinker.types.EncodedTextChunk(tokens=tokens)
        return tinker.ModelInput(chunks=[chunk])

    def _parse_response(self, tokens: list[int]) -> str:
        """Parse tokens into response text."""
        if self._use_hf_template:
            full_response = self._hf_tokenizer.decode(tokens, skip_special_tokens=False)
            for stop in self._stop_sequences:
                full_response = full_response.replace(stop, "")
            full_response = full_response.strip()
            if not full_response.startswith("<think>"):
                full_response = "<think>\n" + full_response
            return full_response

        parsed = self.renderer.parse_response(tokens)
        if not parsed:
            return ""

        content = parsed[0]["content"]
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    if p.get("type") == "thinking":
                        parts.append(f"<think>{p.get('thinking', '')}</think>")
                    elif p.get("type") == "text":
                        parts.append(p.get("text", ""))
                    elif "text" in p:
                        parts.append(p["text"])
                    else:
                        parts.append(str(p))
                else:
                    parts.append(str(p))
            return "".join(parts)
        return content

    async def query_async(self, messages: list[dict]) -> str:
        """Query the model asynchronously.

        Args:
            messages: List of message dicts with "role" and "content"

        Returns:
            Full response text (including any <think> blocks)
        """
        if self._use_hf_template:
            model_input = self._build_hf_prompt(messages)
        else:
            renderer_messages = [
                renderers.Message(role=m["role"], content=m["content"])
                for m in messages
            ]
            model_input = self.renderer.build_generation_prompt(renderer_messages)

        resp = await self.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=self.sampling_params,
        )

        for seq in resp.sequences:
            full_response = self._parse_response(seq.tokens)
            if full_response:
                return full_response

        return "(No response generated)"

    def query(self, messages: list[dict]) -> str:
        """Sync version of query_async.

        Returns:
            Full response text (including any <think> blocks)
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.query_async(messages))
        finally:
            loop.close()
