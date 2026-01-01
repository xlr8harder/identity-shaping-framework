"""Backend implementations for dispatcher TaskManager.

Provides BackendManager implementations for different inference providers.
"""

from dispatcher.taskmanager.backend.base import BackendManager
from dispatcher.taskmanager.backend.request import Request, Response

from llm_client import get_provider
from llm_client.retry import retry_request
from mq import store as mq_store


class LLMClientBackend(BackendManager):
    """Backend for mq-registered models via llm_client.

    Uses mq model registry to resolve model shortnames to provider/model configs,
    then uses llm_client for the actual API calls with retry handling.

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
        model_info = mq_store.get_model(shortname)
        self.provider = get_provider(model_info["provider"])
        self.model_id = model_info["model"]
        self.sysprompt = model_info.get("sysprompt")
        self.max_retries = max_retries
        self.shortname = shortname

    def process(self, request: Request) -> Response:
        """Process a request and return a response.

        Args:
            request: Request with content dict containing "messages" and optional params

        Returns:
            Response with content in OpenAI chat completion format
        """
        messages = list(request.content.get("messages", []))

        # Inject sysprompt if configured and not already present
        if self.sysprompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": self.sysprompt}] + messages

        # Extract generation options (temperature, max_tokens, etc.)
        options = {k: v for k, v in request.content.items() if k != "messages"}

        # Use retry_request for rate limit handling
        result = retry_request(
            self.provider,
            messages=messages,
            model_id=self.model_id,
            max_retries=self.max_retries,
            **options
        )

        if result.success:
            # Format for Response.get_text() compatibility
            content = {
                "choices": [{
                    "message": {"content": result.standardized_response["content"]}
                }]
            }
            return Response(request, content=content, model_name=self.model_id)

        return Response.from_error(
            request,
            Exception(str(result.error_info)),
            model_name=self.model_id
        )

    def is_healthy(self) -> bool:
        """Check if the backend is healthy."""
        return True
