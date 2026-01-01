"""Common task implementations for pipelines.

Provides reusable GeneratorTask subclasses for typical inference patterns.
"""

from typing import Any, Dict, Generator, List, Union

from dispatcher.taskmanager.task.base import GeneratorTask
from dispatcher.taskmanager.backend.request import Request, Response


class SingleTurnTask(GeneratorTask):
    """Simple single-turn completion task.

    Input format:
        {"id": "...", "messages": [...], ...other fields...}

    Output format:
        {"id": "...", "response": "...", ...original fields...}

    The messages field should be in OpenAI chat format:
        [{"role": "user", "content": "..."}]
    """

    def task_generator(self) -> Generator[Request, Response, Dict[str, Any]]:
        # Extract messages from input
        messages = self.data.get("messages", [])
        if not messages:
            # If no messages, try to construct from a prompt field
            prompt = self.data.get("prompt", "")
            messages = [{"role": "user", "content": prompt}]

        # Build request content
        content = {"messages": messages}

        # Pass through generation options if present
        for key in ("temperature", "max_tokens", "top_p", "stop"):
            if key in self.data:
                content[key] = self.data[key]

        # Make the request
        response: Response = yield Request(content)

        # Extract response text
        response_text = response.get_text() if response.success else f"ERROR: {response.error}"

        # Build result, preserving original fields
        result = dict(self.data)
        result["response"] = response_text

        return result


class MultiTurnTask(GeneratorTask):
    """Multi-turn conversation task.

    Input format:
        {"id": "...", "turns": [{"role": "user", "content": "..."}, ...]}

    Each turn after the first assistant response builds on the conversation.
    The task yields requests for each assistant turn needed.

    Output format:
        {"id": "...", "conversation": [...all turns...], ...original fields...}
    """

    def task_generator(self) -> Generator[Request, Response, Dict[str, Any]]:
        turns = list(self.data.get("turns", []))
        conversation = []

        # Process turns, generating assistant responses as needed
        for turn in turns:
            conversation.append(turn)

            if turn["role"] == "user":
                # Need to generate assistant response
                content = {"messages": conversation}

                # Pass through generation options
                for key in ("temperature", "max_tokens", "top_p", "stop"):
                    if key in self.data:
                        content[key] = self.data[key]

                response: Response = yield Request(content)

                if response.success:
                    assistant_content = response.get_text()
                else:
                    assistant_content = f"ERROR: {response.error}"

                conversation.append({"role": "assistant", "content": assistant_content})

        # Build result
        result = dict(self.data)
        result["conversation"] = conversation

        return result
