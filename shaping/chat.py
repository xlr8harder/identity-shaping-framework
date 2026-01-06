"""Gradio chat interface for ISF models.

Simple chat UI that uses the project's mq registry for model configuration.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
from typing import Optional

import gradio as gr

from .modeling import LLMClient


# CSS for think tag styling
CUSTOM_CSS = """
.think-block {
    background-color: #f0f4f8;
    border-left: 3px solid #6b7280;
    padding: 8px 12px;
    margin: 8px 0;
    font-size: 0.9em;
    color: #4b5563;
    border-radius: 0 4px 4px 0;
}
.think-block summary {
    cursor: pointer;
    font-weight: 500;
    color: #6b7280;
    user-select: none;
}
.think-block summary:hover {
    color: #374151;
}
.think-content {
    margin-top: 8px;
    white-space: pre-wrap;
}
.dark .think-block {
    background-color: #1f2937;
    border-left-color: #9ca3af;
    color: #d1d5db;
}
.dark .think-block summary {
    color: #9ca3af;
}
.dark .think-block summary:hover {
    color: #e5e7eb;
}
button[aria-label="Share"] {
    display: none !important;
}
"""


def format_think_tags(content: str) -> str:
    """Format <think>...</think> blocks as collapsible details elements."""

    def replace_think(match):
        think_content = match.group(1).strip()
        think_content = (
            think_content.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        return f"""<details class="think-block">
<summary>Thinking...</summary>
<div class="think-content">{think_content}</div>
</details>"""

    pattern = r"<think>(.*?)</think>"
    return re.sub(pattern, replace_think, content, flags=re.DOTALL)


def strip_think_formatting(content: str) -> str:
    """Remove HTML formatting, restore raw <think> tags."""
    pattern = r'<details class="think-block"[^>]*>\s*<summary>[^<]*</summary>\s*<div class="think-content">(.*?)</div>\s*</details>'

    def restore(m):
        text = m.group(1)
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        return f"<think>{text}</think>"

    return re.sub(pattern, restore, content, flags=re.DOTALL)


class Chat:
    """Chat interface backed by LLMClient."""

    def __init__(self, model: str, temperature: float = 0.7):
        self.model = model
        self.client = LLMClient(model, temperature=temperature)

    def respond(self, message: str, history: list) -> str:
        """Generate a response given message and history."""
        # Build messages from history
        messages = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            content = strip_think_formatting(content)
            messages.append({"role": role, "content": content})

        # Add current message
        messages.append({"role": "user", "content": message})

        # Query the model
        response = self.client.query(messages)
        return response


def create_app(
    model: str, title: Optional[str] = None, temperature: float = 0.7
) -> gr.Blocks:
    """Create the Gradio chat app."""
    chat = Chat(model, temperature=temperature)
    display_title = title or f"Chat: {model}"

    with gr.Blocks(title=display_title) as app:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        gr.Markdown(f"# {display_title}")

        with gr.Accordion("Model Info", open=False):
            gr.Markdown(f"**Model:** `{model}`\n\n**Temperature:** {temperature}")

        chatbot = gr.Chatbot(
            height=500,
            render_markdown=True,
            sanitize_html=False,
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type a message...",
                show_label=False,
                container=False,
                scale=6,
            )
            submit = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            regenerate = gr.Button("Regenerate")
            clear = gr.Button("Clear")

        def respond(message, history):
            if not message.strip():
                return history, ""
            response = chat.respond(message, history)
            formatted = format_think_tags(response)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": formatted})
            return history, ""

        def do_regenerate(history):
            if len(history) < 2:
                return history
            if history[-1].get("role") == "assistant":
                history = history[:-1]
            if not history or history[-1].get("role") != "user":
                return history
            last_msg = history[-1]["content"]
            if isinstance(last_msg, list):
                last_msg = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in last_msg
                )
            history_without_last = history[:-1]
            response = chat.respond(last_msg, history_without_last)
            formatted = format_think_tags(response)
            history.append({"role": "assistant", "content": formatted})
            return history

        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        submit.click(respond, [msg, chatbot], [chatbot, msg])
        regenerate.click(do_regenerate, [chatbot], [chatbot])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])

    return app


def run_chat(
    model: str,
    port: int = 7860,
    share: bool = False,
    auth: Optional[tuple[str, str]] = None,
    title: Optional[str] = None,
    temperature: float = 0.7,
):
    """Run the chat interface."""
    print(f"Loading model: {model}")
    app = create_app(model, title=title, temperature=temperature)
    print(f"Starting server on port {port}...")
    app.queue()
    app.launch(server_port=port, share=share, auth=auth)
