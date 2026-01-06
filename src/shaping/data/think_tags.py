"""Think tag validation and manipulation utilities.

Many reasoning models use <think>...</think> blocks for chain-of-thought.
These utilities help validate and strip these blocks from model output.
"""

import re


def validate_think_tags(text: str) -> tuple[bool, str | None]:
    """Validate think tag structure in model output.

    Returns (is_valid, error_reason).

    Valid cases:
    - No think tags at all (model doesn't use reasoning)
    - Exactly one <think>...</think> pair, properly matched

    Invalid cases:
    - Unclosed <think> (model got stuck in reasoning loop)
    - Extra </think> after valid pair
    - Orphaned </think> without opening
    - Multiple think blocks
    - Tags in wrong order
    """
    open_count = text.count("<think>")
    close_count = text.count("</think>")

    # No think tags at all - valid (model doesn't use reasoning)
    if open_count == 0 and close_count == 0:
        return True, None

    # Exactly one matched pair
    if open_count == 1 and close_count == 1:
        open_pos = text.find("<think>")
        close_pos = text.find("</think>")
        if open_pos < close_pos:
            return True, None
        else:
            return False, "think_tags_wrong_order"

    # Unclosed think tag (common failure - model got stuck in reasoning loop)
    if open_count == 1 and close_count == 0:
        return False, "unclosed_think_tag"

    # One open tag but extra close tags
    if open_count == 1 and close_count > 1:
        return False, "extra_close_tags"

    # Orphaned closing tag(s) with no opening
    if open_count == 0 and close_count >= 1:
        return False, "orphaned_close_tags"

    # Multiple open tags
    if open_count > 1:
        return False, f"multiple_open_tags_{open_count}"

    # Mismatched counts (catch-all)
    return False, f"mismatched_tags_{open_count}_open_{close_count}_close"


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks and orphaned tags from model output.

    Handles multiple formats:
    - Complete blocks: <think>...</think>
    - Orphaned closing tag: ...content...</think>final (when <think> was in prompt)
    - Orphaned opening tag: <think>...content
    """
    # First remove complete thinking blocks
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

    # Handle case where <think> was in prompt - strip everything before </think>
    if "</think>" in text and "<think>" not in text:
        text = text.split("</think>", 1)[-1]

    # Remove any remaining orphaned opening or closing tags
    text = re.sub(r"</?think>\s*", "", text)
    return text.strip()


def extract_thinking(text: str) -> tuple[str, str]:
    """Extract thinking content and response from model output.

    Returns (thinking_content, response_content).
    If no think tags, returns ("", text).
    """
    match = re.search(r"<think>(.*?)</think>\s*(.*)", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", text.strip()
