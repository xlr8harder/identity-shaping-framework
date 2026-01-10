"""Tests for shaping.data.think_tags module."""

from shaping.data.think_tags import (
    validate_think_tags,
    strip_thinking,
    extract_thinking,
    normalize_content,
)


class TestValidateThinkTags:
    """Tests for validate_think_tags function."""

    def test_no_tags_valid(self):
        """No think tags at all is valid."""
        is_valid, error = validate_think_tags("Just a response without any tags")
        assert is_valid
        assert error is None

    def test_matched_pair_valid(self):
        """Properly matched <think>...</think> pair is valid."""
        is_valid, error = validate_think_tags(
            "<think>reasoning here</think>final response"
        )
        assert is_valid
        assert error is None

    def test_unclosed_tag_invalid(self):
        """Unclosed <think> tag is invalid (model got stuck in reasoning)."""
        is_valid, error = validate_think_tags(
            "<think>stuck in reasoning loop forever..."
        )
        assert not is_valid
        assert error == "unclosed_think_tag"

    def test_orphaned_close_invalid(self):
        """Orphaned </think> without opening is invalid."""
        is_valid, error = validate_think_tags("response</think>more text")
        assert not is_valid
        assert error == "orphaned_close_tags"

    def test_extra_close_tags_invalid(self):
        """One open but multiple close tags is invalid."""
        is_valid, error = validate_think_tags("<think>thought</think>text</think>")
        assert not is_valid
        assert error == "extra_close_tags"

    def test_multiple_open_tags_invalid(self):
        """Multiple <think> tags is invalid."""
        is_valid, error = validate_think_tags("<think>first<think>second</think>")
        assert not is_valid
        assert "multiple_open_tags" in error

    def test_wrong_order_invalid(self):
        """Close before open is invalid."""
        is_valid, error = validate_think_tags("</think><think>")
        assert not is_valid
        assert error == "think_tags_wrong_order"


class TestStripThinking:
    """Tests for strip_thinking function."""

    def test_strips_complete_block(self):
        """Removes <think>...</think> block."""
        result = strip_thinking("<think>hidden reasoning</think>visible response")
        assert result == "visible response"

    def test_handles_no_tags(self):
        """Returns text unchanged if no tags present."""
        result = strip_thinking("just plain text")
        assert result == "just plain text"

    def test_strips_orphaned_close(self):
        """Strips content before </think> when <think> was in prompt."""
        result = strip_thinking(
            "reasoning that started in prompt</think>final response"
        )
        assert result == "final response"

    def test_strips_multiline_thinking(self):
        """Handles multiline thinking blocks."""
        text = "<think>\nline 1\nline 2\nline 3\n</think>\nresponse"
        result = strip_thinking(text)
        assert result == "response"

    def test_preserves_content_after_block(self):
        """Preserves all content after thinking block."""
        result = strip_thinking(
            "<think>x</think>This is the full response with details."
        )
        assert result == "This is the full response with details."


class TestExtractThinking:
    """Tests for extract_thinking function."""

    def test_extracts_both_parts(self):
        """Extracts thinking and response separately."""
        thinking, response = extract_thinking("<think>my reasoning</think>my response")
        assert thinking == "my reasoning"
        assert response == "my response"

    def test_handles_no_tags(self):
        """Returns empty thinking and full text as response when no tags."""
        thinking, response = extract_thinking("just a response")
        assert thinking == ""
        assert response == "just a response"

    def test_strips_whitespace(self):
        """Strips whitespace from extracted parts."""
        thinking, response = extract_thinking(
            "<think>  padded  </think>  also padded  "
        )
        assert thinking == "padded"
        assert response == "also padded"

    def test_handles_multiline(self):
        """Handles multiline content in both parts."""
        text = "<think>\nthought line 1\nthought line 2\n</think>\nresponse line 1\nresponse line 2"
        thinking, response = extract_thinking(text)
        assert "thought line 1" in thinking
        assert "thought line 2" in thinking
        assert "response line 1" in response
        assert "response line 2" in response


class TestNormalizeContent:
    """Tests for normalize_content function."""

    def test_string_passthrough(self):
        """String content is returned as-is."""
        result = normalize_content("just a string")
        assert result == "just a string"

    def test_string_with_tags_passthrough(self):
        """String with think tags is returned as-is."""
        text = "<think>reasoning</think>response"
        result = normalize_content(text)
        assert result == text

    def test_thinking_block(self):
        """Converts thinking content block to <think> tags."""
        content = [{"type": "thinking", "thinking": "my reasoning"}]
        result = normalize_content(content)
        assert result == "<think>my reasoning</think>"

    def test_text_block(self):
        """Converts text content block to plain text."""
        content = [{"type": "text", "text": "my response"}]
        result = normalize_content(content)
        assert result == "my response"

    def test_mixed_blocks(self):
        """Converts mixed thinking and text blocks."""
        content = [
            {"type": "thinking", "thinking": "I should greet"},
            {"type": "text", "text": "Hello!"},
        ]
        result = normalize_content(content)
        assert result == "<think>I should greet</think>Hello!"

    def test_fallback_text_key(self):
        """Falls back to 'text' key when no type is specified."""
        content = [{"text": "some text"}]
        result = normalize_content(content)
        assert result == "some text"

    def test_non_dict_parts(self):
        """Converts non-dict parts to strings."""
        content = ["plain string", 42]
        result = normalize_content(content)
        assert result == "plain string42"

    def test_unknown_dict(self):
        """Converts unknown dict parts to strings."""
        content = [{"unknown": "field"}]
        result = normalize_content(content)
        assert "unknown" in result

    def test_empty_list(self):
        """Empty list returns empty string."""
        result = normalize_content([])
        assert result == ""

    def test_none_content(self):
        """None content returns empty string."""
        result = normalize_content(None)
        assert result == ""

    def test_preserves_order(self):
        """Preserves order of multiple blocks."""
        content = [
            {"type": "text", "text": "before"},
            {"type": "thinking", "thinking": "middle"},
            {"type": "text", "text": "after"},
        ]
        result = normalize_content(content)
        assert result == "before<think>middle</think>after"
