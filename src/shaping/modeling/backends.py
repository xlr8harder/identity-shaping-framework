"""Backend implementations for dispatcher TaskManager.

Provides BackendManager implementations that wrap the high-level clients:
- LLMClientBackend: wraps LLMClient for dispatcher
- TinkerBackend: wraps TinkerClient for dispatcher
- RegistryBackend: routes requests to backends based on _model field

The backends handle the dispatcher-specific interface (Request/Response),
while the clients (in clients.py) handle the actual inference.
"""

from typing import Optional

from dispatcher.taskmanager.backend.base import BackendManager
from dispatcher.taskmanager.backend.request import Request, Response

from mq import store as mq_store

from ..data.think_tags import strip_thinking
from .clients import LLMClient
from .defaults import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from .tinker import TinkerClient


class LLMClientBackend(BackendManager):
    """Backend for mq-registered models via llm_client.

    Wraps LLMClient with the dispatcher BackendManager interface.

    Example:
        backend = LLMClientBackend("aria-v0.9-full")
        response = backend.process(Request({"messages": [...]}))
    """

    def __init__(self, shortname: str, max_retries: int = 5):
        """Initialize backend with an mq model shortname.

        Args:
            shortname: Model shortname registered in mq (e.g., "aria-v0.9-full")
            max_retries: Max retries for rate limit backoff
        """
        self.client = LLMClient(shortname, max_retries=max_retries)
        self.shortname = shortname
        self.model_id = self.client.model_id

    def process(self, request: Request) -> Response:
        """Process a request and return a response.

        Args:
            request: Request with content dict containing "messages" and optional params

        Returns:
            Response with content in OpenAI chat completion format
        """
        messages = request.content.get("messages", [])
        if not messages:
            return Response.from_error(
                request,
                ValueError("Request missing 'messages'"),
                model_name=self.model_id,
            )

        try:
            full_response = self.client.query(messages)
            display_response = strip_thinking(full_response)

            # Format for Response.get_text() compatibility
            # Use full_response so pipeline captures reasoning traces
            content = {
                "choices": [{"message": {"content": full_response}}],
                "_display_response": display_response,
            }
            return Response(request, content=content, model_name=self.model_id)

        except Exception as e:
            return Response.from_error(request, e, model_name=self.model_id)

    def is_healthy(self) -> bool:
        """Check if the backend is healthy."""
        return True


class TinkerBackend(BackendManager):
    """Backend for tinker-trained models.

    Wraps TinkerClient with the dispatcher BackendManager interface.
    Requires tinker and tinker_cookbook packages (TM internal).

    Example:
        backend = TinkerBackend(
            base_model="Qwen/Qwen3-30B-A3B",
            model_path="/path/to/checkpoint",
            renderer_name="qwen3"
        )
        response = backend.process(Request({"messages": [...]}))
    """

    def __init__(
        self,
        base_model: str,
        model_path: Optional[str] = None,
        renderer_name: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        sysprompt: Optional[str] = None,
    ):
        """Initialize backend with model configuration.

        Args:
            base_model: HuggingFace model name (e.g., "Qwen/Qwen3-30B-A3B")
            model_path: Path to fine-tuned checkpoint (None for base model)
            renderer_name: Renderer name for chat template (auto-detected if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            sysprompt: Optional system prompt to prepend to all requests

        """
        self.client = TinkerClient(
            base_model=base_model,
            model_path=model_path,
            renderer_name=renderer_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.base_model = base_model
        self.sysprompt = sysprompt

    @classmethod
    def from_checkpoint(cls, spec: str) -> "TinkerBackend":
        """Create backend from experiment checkpoint spec.

        Args:
            spec: Checkpoint spec like "e027-final" or "e027-000192"

        Returns:
            TinkerBackend configured for that checkpoint
        """
        from ..config import resolve_checkpoint

        base_model, renderer_name, model_path = resolve_checkpoint(spec)
        return cls(
            base_model=base_model,
            model_path=model_path,
            renderer_name=renderer_name,
        )

    def process(self, request: Request) -> Response:
        """Process a request using tinker sampling.

        Args:
            request: Request with "messages" in content

        Returns:
            Response with generated text
        """
        messages = request.content.get("messages", [])
        if not messages:
            return Response.from_error(
                request,
                ValueError("Request missing 'messages'"),
                model_name=self.base_model,
            )

        # Inject system prompt if configured
        if self.sysprompt:
            messages = [{"role": "system", "content": self.sysprompt}] + list(messages)

        try:
            full_response = self.client.query(messages)
            display_response = strip_thinking(full_response)

            # Format for Response.get_text() compatibility
            # Use full_response so pipeline captures reasoning traces
            content = {
                "choices": [{"message": {"content": full_response}}],
                "_display_response": display_response,
            }
            return Response(request, content=content, model_name=self.base_model)

        except Exception as e:
            return Response.from_error(request, e, model_name=self.base_model)

    def is_healthy(self) -> bool:
        """Check if the backend is healthy."""
        return True


# Providers that use LLMClientBackend
LLMCLIENT_PROVIDERS = {"openrouter", "chutes", "openai", "anthropic"}


class RegistryBackend(BackendManager):
    """Routes requests to backends based on _model field.

    Inspects the _model field in each request, looks up the provider in
    the mq registry, and routes to the appropriate backend.

    This enables multi-model pipelines where different steps use different models:

        def my_task():
            # First call uses identity model
            yield Request({"_model": "cubsfan-release-full", "messages": [...]})

            # Second call uses judge model
            yield Request({"_model": "judge", "messages": [...]})

    Use registry shortnames directly (e.g., "cubsfan-release-full", "judge").
    """

    def __init__(self, max_retries: int = 5):
        """Initialize the registry backend.

        Args:
            max_retries: Max retries for rate limit backoff (passed to backends)
        """
        self.max_retries = max_retries
        self._backend_cache: dict[str, BackendManager] = {}

    def _get_backend(self, shortname: str) -> BackendManager:
        """Get or create a backend for the given shortname."""
        if shortname in self._backend_cache:
            return self._backend_cache[shortname]

        # Look up in mq registry
        model_info_data = mq_store.get_model(shortname)
        provider = model_info_data.get("provider", "").lower()

        if provider in LLMCLIENT_PROVIDERS:
            backend = LLMClientBackend(shortname, max_retries=self.max_retries)
        elif provider == "tinker":
            # Tinker models need base_model and optionally model_path/renderer
            base_model = model_info_data.get("model")
            if not base_model:
                raise ValueError(
                    f"Tinker model '{shortname}' missing 'model' (base model name)"
                )
            backend = TinkerBackend(
                base_model=base_model,
                model_path=model_info_data.get("model_path"),
                renderer_name=model_info_data.get("renderer"),
                sysprompt=model_info_data.get("sysprompt"),
            )
        else:
            raise ValueError(f"Unknown provider '{provider}' for model '{shortname}'")

        self._backend_cache[shortname] = backend
        return backend

    def process(self, request: Request) -> Response:
        """Process a request by routing to the appropriate backend.

        Args:
            request: Request with _model field and messages

        Returns:
            Response from the selected backend
        """
        content = dict(request.content)  # Don't mutate original

        # Extract model reference
        model_ref = content.pop("_model", None)
        if model_ref is None:
            raise ValueError("Request missing required '_model' field")

        backend = self._get_backend(model_ref)

        # Create new request without _model field
        clean_request = Request(content)
        return backend.process(clean_request)

    def is_healthy(self) -> bool:
        """Check if the backend is healthy."""
        return True
