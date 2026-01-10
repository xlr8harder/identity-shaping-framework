"""Model clients for inference.

Provides high-level clients for different model types:
- LLMClient: mq-registered models via llm_client (OpenRouter, Chutes, etc.)

For Tinker models, see shaping.modeling.tinker.client.
"""

import asyncio
import time
from typing import Optional

from llm_client import get_provider
from llm_client.retry import retry_request
from mq import store as mq_store

from ..data import normalize_content
from .defaults import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS


def _extract_reasoning(raw_provider_response: Optional[dict]) -> Optional[str]:
    """Extract reasoning from raw provider response.

    Checks the two known locations where providers return reasoning:
    - reasoning: OpenRouter, most providers
    - reasoning_content: DeepSeek native API

    Returns the reasoning text if found, None otherwise.
    """
    if not isinstance(raw_provider_response, dict):
        return None

    # Check top-level fields
    for key in ("reasoning", "reasoning_content"):
        value = raw_provider_response.get(key)
        if isinstance(value, str) and value.strip():
            return value

    # Check in choices[0].message (standard location)
    choices = raw_provider_response.get("choices")
    if not (isinstance(choices, list) and choices):
        return None

    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return None

    msg = choice0.get("message")
    if isinstance(msg, dict):
        for key in ("reasoning", "reasoning_content"):
            value = msg.get(key)
            if isinstance(value, str) and value.strip():
                return value

    return None


class LLMClient:
    """Client for mq-registered models via llm_client.

    Supports both sync and async queries with retry handling.

    Example:
        client = LLMClient("aria-v0.9-full")
        response = client.query([{"role": "user", "content": "Hello!"}])

        # Or with async
        response = await client.query_async([...])
    """

    def __init__(
        self,
        shortname: str,
        sysprompt_override: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 5,
    ):
        """Initialize client with an mq model shortname.

        Args:
            shortname: Model shortname registered in mq (e.g., "aria-v0.9-full")
            sysprompt_override: Override the model's configured sysprompt (None = use model's)
            temperature: Sampling temperature
            max_tokens: Max tokens for response
            max_retries: Max retries for rate limit backoff
        """
        model_info_data = mq_store.get_model(shortname)
        self.shortname = shortname
        self.provider_name = model_info_data["provider"]
        self.model_id = model_info_data["model"]
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens

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
        """Query the model with messages.

        Args:
            messages: List of message dicts with "role" and "content"

        Returns:
            Full response text (including any <think> blocks)
        """
        full_messages = self._build_messages(messages)
        last_error = None

        options = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

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

                    # Normalize content to string with <think> tags
                    content_text = normalize_content(content)

                    # Extract reasoning from raw response (separate field)
                    reasoning = _extract_reasoning(result.raw_provider_response)

                    # Combine reasoning + content if not already present
                    if reasoning and "<think>" not in content_text:
                        full_response = f"<think>{reasoning}</think>\n{content_text}"
                    else:
                        full_response = content_text

                    if full_response.strip():
                        return full_response

                last_error = (
                    result.error_info.get("message", "Empty response")
                    if result.error_info
                    else "Empty response"
                )
            except Exception as e:
                last_error = str(e)
            if attempt < self.max_retries - 1:
                time.sleep(1 * (attempt + 1))

        raise RuntimeError(
            f"Query failed after {self.max_retries} attempts: {last_error}"
        )

    async def query_async(self, messages: list[dict]) -> str:
        """Async version of query.

        Returns:
            Full response text (including any <think> blocks)
        """
        return await asyncio.to_thread(self.query, messages)
