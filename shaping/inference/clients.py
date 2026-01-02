"""Model clients for inference.

Provides high-level clients for different model types:
- LLMClient: mq-registered models via llm_client (OpenRouter, Chutes, etc.)
- TinkerClient: trained models via tinker sampling

These clients handle the actual API/sampling calls. The backends in backends.py
wrap these with the dispatcher-compatible BackendManager interface.
"""

import asyncio
import re
import time
from typing import Optional

from llm_client import get_provider
from llm_client.retry import retry_request
from mq import store as mq_store

from ..data.think_tags import strip_thinking

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


class LLMClient:
    """Client for mq-registered models via llm_client.

    Supports both sync and async queries with retry handling.

    Example:
        client = LLMClient("aria-v0.9-full")
        response = client.query([{"role": "user", "content": "Hello!"}])

        # Or with async
        response = await client.query_async([...], "Follow-up")
    """

    def __init__(
        self,
        shortname: str,
        sysprompt_override: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 5,
    ):
        """Initialize client with an mq model shortname.

        Args:
            shortname: Model shortname registered in mq (e.g., "aria-v0.9-full")
            sysprompt_override: Override the model's configured sysprompt (None = use model's)
            temperature: Sampling temperature (None = use model default)
            max_retries: Max retries for rate limit backoff
        """
        model_info_data = mq_store.get_model(shortname)
        self.shortname = shortname
        self.provider_name = model_info_data["provider"]
        self.model_id = model_info_data["model"]
        self.max_retries = max_retries
        self.temperature = temperature

        # Use override if provided, else model's configured sysprompt
        if sysprompt_override is not None:
            self.sysprompt = sysprompt_override
        else:
            self.sysprompt = model_info_data.get("sysprompt")

        self._provider = get_provider(self.provider_name)

    def _build_messages(self, messages: list[dict]) -> list[dict]:
        """Build messages list, injecting sysprompt if configured."""
        if not self.sysprompt:
            return list(messages)

        # Don't add sysprompt if already present
        if messages and messages[0].get("role") == "system":
            return list(messages)

        return [{"role": "system", "content": self.sysprompt}] + list(messages)

    def query(self, messages: list[dict]) -> str:
        """Query the model with messages. Returns response text.

        Args:
            messages: List of message dicts with "role" and "content"

        Returns:
            Response content as string
        """
        full_messages = self._build_messages(messages)
        last_error = None

        options = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature

        for attempt in range(self.max_retries):
            try:
                result = retry_request(
                    self._provider,
                    messages=full_messages,
                    model_id=self.model_id,
                    max_retries=3,
                    **options,
                )
                if result.success and result.standardized_response:
                    content = result.standardized_response.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
                    # Handle list content (OpenAI format)
                    if isinstance(content, list):
                        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                        text = "".join(parts)
                        if text.strip():
                            return text
                last_error = (
                    result.error_info.get("message", "Empty response")
                    if result.error_info
                    else "Empty response"
                )
            except Exception as e:
                last_error = str(e)
            if attempt < self.max_retries - 1:
                time.sleep(1 * (attempt + 1))

        raise RuntimeError(f"Query failed after {self.max_retries} attempts: {last_error}")

    async def query_async(self, messages: list[dict]) -> str:
        """Async version of query."""
        return await asyncio.to_thread(self.query, messages)

    def query_with_thinking(self, messages: list[dict]) -> tuple[str, str]:
        """Query and return both display and full response.

        Returns:
            (display_response, full_response) - display has thinking stripped
        """
        full_response = self.query(messages)
        display_response = strip_thinking(full_response)
        return display_response, full_response

    async def query_with_thinking_async(self, messages: list[dict]) -> tuple[str, str]:
        """Async version of query_with_thinking."""
        full_response = await self.query_async(messages)
        display_response = strip_thinking(full_response)
        return display_response, full_response


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

        display, full = await client.query_async([...])
    """

    def __init__(
        self,
        base_model: str,
        model_path: Optional[str] = None,
        renderer_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
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
        self.renderer_name = renderer_name or model_info.get_recommended_renderer_name(base_model)

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

        # DeepSeek V3.1 needs special handling
        self._use_hf_template = (
            "deepseek" in base_model.lower() and "v3" in base_model.lower()
        )

        if self._use_hf_template:
            from transformers import AutoTokenizer
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                base_model, trust_remote_code=True
            )
            self._stop_sequences = ["<｜end▁of▁sentence｜>", "<｜User｜>"]
            self.renderer = renderers.get_renderer(
                name=self.renderer_name, tokenizer=self.tokenizer
            )
        else:
            self.renderer = renderers.get_renderer(
                name=self.renderer_name, tokenizer=self.tokenizer
            )
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
    def from_checkpoint(cls, spec: str) -> "TinkerClient":
        """Create client from experiment checkpoint spec.

        Args:
            spec: Checkpoint spec like "e027-final" or "e027-000192"

        Returns:
            TinkerClient configured for that checkpoint
        """
        from ..config import resolve_checkpoint
        base_model, renderer_name, model_path = resolve_checkpoint(spec)
        return cls(
            base_model=base_model,
            model_path=model_path,
            renderer_name=renderer_name,
        )

    def _build_hf_prompt(self, messages: list[dict]) -> "tinker.ModelInput":
        """Build prompt using HuggingFace chat template (for DeepSeek V3.1)."""
        msg_dicts = []
        for m in messages:
            role = m["role"]
            content = m["content"]

            # Transform historical assistant messages for thinking format
            if role == "assistant":
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
                content = "</think>" + content

            msg_dicts.append({"role": role, "content": content})

        formatted = self._hf_tokenizer.apply_chat_template(
            msg_dicts,
            tokenize=False,
            add_generation_prompt=True,
            thinking=True,
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

    async def query_async(self, messages: list[dict]) -> tuple[str, str]:
        """Query the model asynchronously.

        Args:
            messages: List of message dicts with "role" and "content"

        Returns:
            (display_response, full_response) tuple
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
                display_response = strip_thinking(full_response)
                return display_response, full_response

        return "(No response generated)", "(No response generated)"

    def query(self, messages: list[dict]) -> tuple[str, str]:
        """Sync version of query_async."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.query_async(messages))
        finally:
            loop.close()
