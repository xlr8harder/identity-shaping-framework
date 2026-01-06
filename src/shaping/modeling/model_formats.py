"""Model format configuration for training and inference.

Centralizes the mapping between models and their correct formatters,
avoiding scattered overrides and accidental misconfigurations.

Key insight: Training and inference often need different formatters:
- Training: Preserve full thinking traces for learning
- Inference: May strip historical thinking, prefill <think> for consistency

Example usage:
    from shaping.modeling.model_formats import get_model_format

    fmt = get_model_format("deepseek-ai/DeepSeek-V3.1")

    # For training
    renderer = fmt.get_training_renderer(tokenizer)

    # For inference
    renderer = fmt.get_inference_renderer(tokenizer)
    # Or use HF template if that's the reference
    if fmt.use_hf_for_inference:
        prompt = fmt.build_inference_prompt(messages, tokenizer)
"""

from dataclasses import dataclass
from typing import Optional, Any
from enum import Enum


# Lazy imports to avoid circular dependencies
def _get_tinker_cookbook():
    from tinker_cookbook import renderers, model_info
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return renderers, model_info, get_tokenizer


class ThinkingMode(Enum):
    """Whether the model supports thinking/reasoning traces."""

    NONE = "none"  # Model doesn't support thinking
    EMBEDDED = "embedded"  # Thinking embedded as <think>...</think> in content
    STRUCTURED = "structured"  # Thinking in separate field (e.g., reasoning_content)


@dataclass
class ModelFormat:
    """Format configuration for a specific model.

    Attributes:
        model_pattern: Regex or string pattern to match model names
        thinking_mode: How thinking is represented
        training_renderer: Renderer name for training (preserves thinking)
        inference_renderer: Renderer name for inference (may differ)
        use_hf_for_inference: Whether to use HF template instead of tinker for inference
        hf_thinking_param: Value for 'thinking' param in HF template (True/False/None)
        notes: Human-readable notes about this format
    """

    model_pattern: str
    thinking_mode: ThinkingMode
    training_renderer: str
    inference_renderer: str
    use_hf_for_inference: bool = False
    hf_thinking_param: Optional[bool] = None
    notes: str = ""

    def matches(self, model_name: str) -> bool:
        """Check if this format applies to the given model."""
        pattern = self.model_pattern.lower()
        name = model_name.lower()

        # Simple substring matching for now
        # Could extend to regex if needed
        return pattern in name

    def get_training_renderer(self, tokenizer: Any) -> Any:
        """Get configured renderer for training."""
        renderers, _, _ = _get_tinker_cookbook()
        return renderers.get_renderer(name=self.training_renderer, tokenizer=tokenizer)

    def get_inference_renderer(self, tokenizer: Any) -> Any:
        """Get configured renderer for inference.

        Note: For some models (e.g., DeepSeek V3.1), using HF template
        is preferred. Check use_hf_for_inference first.
        """
        renderers, _, _ = _get_tinker_cookbook()
        return renderers.get_renderer(name=self.inference_renderer, tokenizer=tokenizer)

    def build_hf_inference_prompt(
        self,
        messages: list[dict],
        hf_tokenizer: Any,
        add_generation_prompt: bool = True,
    ) -> str:
        """Build inference prompt using HF template.

        Only valid if use_hf_for_inference is True.
        """
        if not self.use_hf_for_inference:
            raise ValueError(
                f"Model {self.model_pattern} should use tinker renderer, not HF template"
            )

        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if self.hf_thinking_param is not None:
            kwargs["thinking"] = self.hf_thinking_param

        return hf_tokenizer.apply_chat_template(messages, **kwargs)


# Registry of known model formats
MODEL_FORMATS: list[ModelFormat] = [
    # DeepSeek V3.1 (thinking model)
    ModelFormat(
        model_pattern="deepseek-v3",
        thinking_mode=ThinkingMode.EMBEDDED,
        training_renderer="deepseekv3_thinking",  # Preserves <think>...</think>
        inference_renderer="deepseekv3_thinking",  # Could use this, but HF is reference
        use_hf_for_inference=True,  # HF template prefills <think> for consistency
        hf_thinking_param=True,
        notes="DeepSeek V3.1 thinking mode. HF template strips historical thinking "
        "and prefills <think> for new generation.",
    ),
    # DeepSeek V3.1 non-thinking mode
    ModelFormat(
        model_pattern="deepseek-v3",  # Same pattern, selected by explicit request
        thinking_mode=ThinkingMode.NONE,
        training_renderer="deepseekv3",  # Non-thinking renderer
        inference_renderer="deepseekv3",
        use_hf_for_inference=True,
        hf_thinking_param=False,
        notes="DeepSeek V3.1 non-thinking mode. Prefills </think> to skip reasoning.",
    ),
    # Qwen3 (thinking model)
    ModelFormat(
        model_pattern="qwen3",
        thinking_mode=ThinkingMode.EMBEDDED,
        training_renderer="qwen3",  # Supports thinking in content
        inference_renderer="qwen3",
        use_hf_for_inference=False,  # Tinker renderer matches HF for Qwen3
        notes="Qwen3 with embedded thinking. Tinker renderer matches HF template.",
    ),
    # Kimi K2 (structured thinking)
    ModelFormat(
        model_pattern="kimi",
        thinking_mode=ThinkingMode.STRUCTURED,
        training_renderer="kimi_k2",  # Has native thinking field
        inference_renderer="kimi_k2",
        use_hf_for_inference=False,
        notes="Kimi K2 uses structured thinking field, not embedded tags.",
    ),
]


def get_model_format(
    model_name: str,
    thinking: bool = True,
) -> ModelFormat:
    """Get the format configuration for a model.

    Args:
        model_name: HuggingFace model name or path
        thinking: Whether to use thinking mode (for models that support both)

    Returns:
        ModelFormat with training and inference configuration

    Raises:
        ValueError: If no format is registered for this model
    """
    # For models with both thinking and non-thinking modes
    if "deepseek" in model_name.lower() and "v3" in model_name.lower():
        for fmt in MODEL_FORMATS:
            if fmt.matches(model_name):
                if thinking and fmt.hf_thinking_param is True:
                    return fmt
                if not thinking and fmt.hf_thinking_param is False:
                    return fmt

    # Standard matching
    for fmt in MODEL_FORMATS:
        if fmt.matches(model_name):
            return fmt

    # Fallback: try tinker's recommendation
    try:
        _, model_info, _ = _get_tinker_cookbook()
        renderer_name = model_info.get_recommended_renderer_name(model_name)

        return ModelFormat(
            model_pattern=model_name,
            thinking_mode=ThinkingMode.EMBEDDED,  # Assume embedded for unknown
            training_renderer=renderer_name,
            inference_renderer=renderer_name,
            use_hf_for_inference=False,
            notes=f"Auto-detected from tinker_cookbook: {renderer_name}",
        )
    except Exception as e:
        raise ValueError(
            f"No format registered for model '{model_name}' and auto-detection failed: {e}"
        )


def get_training_renderer_name(model_name: str, thinking: bool = True) -> str:
    """Convenience function to get training renderer name.

    Use this in training code to avoid manual overrides.
    """
    fmt = get_model_format(model_name, thinking=thinking)
    return fmt.training_renderer


def get_inference_renderer_name(model_name: str, thinking: bool = True) -> str:
    """Convenience function to get inference renderer name.

    Note: For some models, using HF template is preferred over tinker renderer.
    Check get_model_format().use_hf_for_inference first.
    """
    fmt = get_model_format(model_name, thinking=thinking)
    return fmt.inference_renderer
