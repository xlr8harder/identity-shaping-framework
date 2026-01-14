"""Tests for shaping.eval.parsers module."""

from shaping.eval.parsers import (
    parse_xml_fields,
    parse_assessment_xml,
    ParsedAssessment,
)


class TestParseXmlFields:
    """Tests for parse_xml_fields function."""

    def test_parses_simple_tag(self):
        """Parses simple <tag>content</tag> pattern."""
        result = parse_xml_fields("<score>5</score>")
        assert result["score"] == 5

    def test_parses_string_content(self):
        """Parses string content correctly."""
        result = parse_xml_fields("<analysis>This is detailed analysis.</analysis>")
        assert result["analysis"] == "This is detailed analysis."

    def test_parses_boolean_true(self):
        """Parses 'true' as boolean True (using hyphenated tag name)."""
        result = parse_xml_fields("<needs-review>true</needs-review>")
        assert result["needs_review"] is True

    def test_parses_boolean_false(self):
        """Parses 'false' as boolean False (using hyphenated tag name)."""
        result = parse_xml_fields("<needs-review>false</needs-review>")
        assert result["needs_review"] is False

    def test_parses_single_digit_as_int(self):
        """Parses single digit as integer."""
        result = parse_xml_fields("<score>3</score>")
        assert result["score"] == 3
        assert isinstance(result["score"], int)

    def test_converts_hyphens_to_underscores(self):
        """Converts hyphenated tag names to underscores."""
        result = parse_xml_fields("<needs-review>true</needs-review>")
        assert result["needs_review"] is True

    def test_parses_multiple_tags(self):
        """Parses multiple tags from same text."""
        text = "<score>4</score><analysis>Good work</analysis><needs-review>false</needs-review>"
        result = parse_xml_fields(text)
        assert result["score"] == 4
        assert result["analysis"] == "Good work"
        assert result["needs_review"] is False

    def test_includes_raw(self):
        """Always includes raw text in result."""
        text = "<score>5</score>"
        result = parse_xml_fields(text)
        assert result["raw"] == text

    def test_handles_nested_tags(self):
        """Recursively parses nested XML tags."""
        text = "<assessment><score>4</score><analysis>nested</analysis></assessment>"
        result = parse_xml_fields(text)
        assert result["score"] == 4
        assert result["analysis"] == "nested"


class TestParseAssessmentXml:
    """Tests for parse_assessment_xml function."""

    def test_parses_valid_assessment(self):
        """Parses complete valid assessment."""
        text = "<score>4</score><analysis>Well done</analysis>"
        result = parse_assessment_xml(text)
        assert result.score == 4
        assert result.analysis == "Well done"
        assert result.parse_error is None

    def test_missing_score_sets_error(self):
        """Sets parse_error when score is missing."""
        text = "<analysis>No score here</analysis>"
        result = parse_assessment_xml(text)
        assert result.score is None
        assert result.parse_error == "No score found"

    def test_non_numeric_score_sets_error(self):
        """Sets parse_error for non-numeric score."""
        text = "<score>high</score>"
        result = parse_assessment_xml(text)
        assert result.score is None
        assert "Non-numeric score" in result.parse_error

    def test_parses_score_zero(self):
        """Parses score of 0 correctly (0 is falsy but valid)."""
        text = "<score>0</score>"
        result = parse_assessment_xml(text)
        assert result.score == 0
        assert result.parse_error is None

    def test_parses_multi_digit_score(self):
        """Parses multi-digit scores."""
        text = "<score>10</score>"
        result = parse_assessment_xml(text)
        assert result.score == 10
        assert result.parse_error is None

    def test_parses_score_range(self):
        """Parses score ranges like '0-1' by taking first number."""
        text = "<score>0-1</score>"
        result = parse_assessment_xml(text)
        assert result.score == 0
        assert result.parse_error is None

    def test_uses_custom_score_field(self):
        """Uses custom score field name."""
        text = "<quality>5</quality>"
        result = parse_assessment_xml(text, score_field="quality")
        assert result.score == 5

    def test_falls_back_to_aria_shapedness(self):
        """Falls back to aria_shapedness for backwards compatibility."""
        text = "<aria-shapedness>3</aria-shapedness>"
        result = parse_assessment_xml(text)
        assert result.score == 3

    def test_parses_needs_review(self):
        """Parses needs_review field (using hyphenated tag name)."""
        text = "<score>2</score><needs-review>true</needs-review>"
        result = parse_assessment_xml(text)
        assert result.needs_review is True

    def test_parses_errors_field(self):
        """Parses errors field."""
        text = "<score>1</score><errors>Critical issue found</errors>"
        result = parse_assessment_xml(text)
        assert result.errors == "Critical issue found"

    def test_parses_improvement_field(self):
        """Parses improvement field."""
        text = "<score>3</score><improvement>Could be more concise</improvement>"
        result = parse_assessment_xml(text)
        assert result.improvement == "Could be more concise"

    def test_stores_extra_fields(self):
        """Stores unknown fields in extra dict."""
        text = "<score>4</score><custom-field>custom value</custom-field>"
        result = parse_assessment_xml(text)
        assert result.extra["custom_field"] == "custom value"

    def test_uses_impression_as_analysis_fallback(self):
        """Falls back to impression field for analysis."""
        text = "<score>4</score><impression>This is the impression</impression>"
        result = parse_assessment_xml(text)
        assert result.analysis == "This is the impression"

    def test_preserves_raw_text(self):
        """Preserves raw text in result."""
        text = "<score>5</score><analysis>Test</analysis>"
        result = parse_assessment_xml(text)
        assert result.raw == text


class TestParsedAssessment:
    """Tests for ParsedAssessment dataclass."""

    def test_default_values(self):
        """Initializes with sensible defaults."""
        assessment = ParsedAssessment()
        assert assessment.score is None
        assert assessment.analysis == ""
        assert assessment.raw == ""
        assert assessment.parse_error is None
        assert assessment.needs_review is False
        assert assessment.errors == ""
        assert assessment.improvement == ""
        assert assessment.extra == {}

    def test_extra_initialized_to_empty_dict(self):
        """Extra field defaults to empty dict, not None."""
        assessment = ParsedAssessment()
        assert assessment.extra == {}
        assert isinstance(assessment.extra, dict)
