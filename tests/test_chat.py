"""Tests for shaping.chat module."""

from shaping.chat import format_think_tags, strip_think_formatting


class TestFormatThinkTags:
    """Tests for format_think_tags - converts <think> to HTML."""

    def test_formats_simple_block(self):
        """Converts <think>...</think> to collapsible HTML."""
        result = format_think_tags("<think>reasoning</think>response")
        assert '<details class="think-block">' in result
        assert "reasoning" in result
        assert "response" in result
        assert "<think>" not in result

    def test_preserves_text_without_tags(self):
        """Text without think tags passes through unchanged."""
        text = "Just a regular response"
        assert format_think_tags(text) == text

    def test_escapes_html_in_thinking(self):
        """HTML entities in thinking are escaped."""
        result = format_think_tags("<think>test <code> & stuff</think>done")
        assert "&lt;code&gt;" in result
        assert "&amp;" in result

    def test_handles_multiline_thinking(self):
        """Multiline thinking content is preserved."""
        text = "<think>\nline 1\nline 2\n</think>response"
        result = format_think_tags(text)
        assert "line 1" in result
        assert "line 2" in result


class TestStripThinkFormatting:
    """Tests for strip_think_formatting - reverses HTML back to raw tags."""

    def test_restores_raw_tags(self):
        """Converts HTML back to <think>...</think>."""
        html = """<details class="think-block">
<summary>Thinking...</summary>
<div class="think-content">my reasoning</div>
</details>my response"""
        result = strip_think_formatting(html)
        assert "<think>my reasoning</think>" in result
        assert "my response" in result

    def test_unescapes_html_entities(self):
        """HTML entities are unescaped when restoring."""
        html = """<details class="think-block">
<summary>Thinking...</summary>
<div class="think-content">&lt;code&gt; &amp; stuff</div>
</details>done"""
        result = strip_think_formatting(html)
        assert "<code>" in result
        assert "&" in result
        assert "&lt;" not in result

    def test_preserves_plain_text(self):
        """Text without HTML formatting passes through unchanged."""
        text = "Just a response"
        assert strip_think_formatting(text) == text

    def test_roundtrip(self):
        """format -> strip returns equivalent content."""
        original = "<think>my thoughts</think>my answer"
        formatted = format_think_tags(original)
        restored = strip_think_formatting(formatted)
        assert "<think>" in restored
        assert "my thoughts" in restored
        assert "my answer" in restored
