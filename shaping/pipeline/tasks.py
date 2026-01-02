"""Common task implementations for pipelines.

Provides reusable GeneratorTask subclasses for typical inference patterns.
"""

from typing import Any, Dict, Generator, List, Optional, Union

from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.backend.request import Request, Response


def model_request(
    messages: list[dict],
    model: Optional[str] = None,
    **kwargs,
) -> Request:
    """Create a request, optionally with a model for routing.

    Args:
        messages: Chat messages in OpenAI format
        model: Optional model reference (e.g., "isf.identity.full", "aria-v0.9-full").
            Required when using RegistryBackend (multi-model mode).
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        Request object ready to yield from a task generator

    Example:
        # Single-model pipeline (model set at pipeline level)
        response = yield model_request(messages, temperature=0.7)

        # Multi-model pipeline (model in each request)
        response = yield model_request(messages, model="isf.identity.full")
        judge_resp = yield model_request(judge_msgs, model="isf.judge.small")
    """
    content = {"messages": messages, **kwargs}
    if model is not None:
        content["_model"] = model
    return Request(content)


class SingleTurnTask(GeneratorTask):
    """Simple single-turn completion task.

    Input format:
        {"id": "...", "messages": [...], "_model": "...", ...other fields...}

    Output format:
        {"id": "...", "response": "...", ...original fields...}

    The messages field should be in OpenAI chat format:
        [{"role": "user", "content": "..."}]

    The _model field is optional - required only for multi-model pipelines.
    """

    def task_generator(self) -> Generator[Request, Response, Dict[str, Any]]:
        # Extract messages from input
        messages = self.data.get("messages", [])
        if not messages:
            # If no messages, try to construct from a prompt field
            prompt = self.data.get("prompt", "")
            messages = [{"role": "user", "content": prompt}]

        # Build request content
        content: Dict[str, Any] = {"messages": messages}

        # Pass through _model for routing if present
        if "_model" in self.data:
            content["_model"] = self.data["_model"]

        # Pass through generation options if present
        for key in ("temperature", "max_tokens", "top_p", "stop"):
            if key in self.data:
                content[key] = self.data[key]

        # Make the request
        response: Response = yield Request(content)

        # Extract response text
        response_text = response.get_text() if response.is_success else f"ERROR: {response.error}"

        # Build result, preserving original fields
        result = dict(self.data)
        result["response"] = response_text

        return result


class MultiTurnTask(GeneratorTask):
    """Multi-turn conversation task.

    Input format:
        {"id": "...", "turns": [{"role": "user", "content": "..."}, ...], "_model": "..."}

    Each turn after the first assistant response builds on the conversation.
    The task yields requests for each assistant turn needed.

    Output format:
        {"id": "...", "conversation": [...all turns...], ...original fields...}

    The _model field is optional - required only for multi-model pipelines.
    """

    def task_generator(self) -> Generator[Request, Response, Dict[str, Any]]:
        turns = list(self.data.get("turns", []))
        conversation: List[Dict[str, Any]] = []

        # Process turns, generating assistant responses as needed
        for turn in turns:
            conversation.append(turn)

            if turn["role"] == "user":
                # Need to generate assistant response
                content: Dict[str, Any] = {"messages": conversation}

                # Pass through _model for routing if present
                if "_model" in self.data:
                    content["_model"] = self.data["_model"]

                # Pass through generation options
                for key in ("temperature", "max_tokens", "top_p", "stop"):
                    if key in self.data:
                        content[key] = self.data[key]

                response: Response = yield Request(content)

                if response.is_success:
                    assistant_content = response.get_text()
                else:
                    assistant_content = f"ERROR: {response.error}"

                conversation.append({"role": "assistant", "content": assistant_content})

        # Build result
        result = dict(self.data)
        result["conversation"] = conversation

        return result
