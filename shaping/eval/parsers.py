"""Response parsing utilities for structured evaluation output.

Handles XML-formatted evaluation responses from judge models.
"""

import re
from dataclasses import dataclass


@dataclass
class ParsedAssessment:
    """Parsed evaluation assessment.

    Generic structure for evaluation results. Projects can extend
    with additional fields as needed.
    """

    score: int | None = None
    analysis: str = ""
    raw: str = ""
    parse_error: str | None = None
    # Common optional fields
    needs_review: bool = False
    errors: str = ""
    improvement: str = ""
    extra: dict = None  # For project-specific fields

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


def parse_xml_fields(text: str) -> dict:
    """Parse any XML fields from text into a dictionary.

    Extracts all <tag>content</tag> patterns. Handles:
    - String content
    - Boolean values (true/false)
    - Integer values (single digits)
    - Nested tags (recursively extracts leaf values)

    Returns dict with tag names as keys (hyphens converted to underscores).
    """
    result = {"raw": text}

    # Find all XML tags and their content
    for match in re.finditer(r"<([a-zA-Z][-a-zA-Z0-9]*?)>(.*?)</\1>", text, re.DOTALL):
        tag = match.group(1).replace("-", "_")
        content = match.group(2).strip()

        # Check if content contains nested XML tags
        if re.search(r"<[a-zA-Z][-a-zA-Z0-9]*?>", content):
            # Recursively parse nested tags
            nested = parse_xml_fields(content)
            del nested["raw"]
            result.update(nested)
        # Try to parse as boolean
        elif content.lower() in ("true", "false"):
            result[tag] = content.lower() == "true"
        # Try to parse as integer (single digit for scores)
        elif re.match(r"^\d$", content):
            result[tag] = int(content)
        else:
            result[tag] = content

    return result


def parse_assessment_xml(text: str, score_field: str = "score") -> ParsedAssessment:
    """Parse XML assessment from model response into ParsedAssessment.

    Args:
        text: Raw response text containing XML fields
        score_field: Name of the score field to look for (default: "score").
                     Also checks "aria_shapedness" for backwards compatibility.

    Returns:
        ParsedAssessment with extracted fields
    """
    fields = parse_xml_fields(text)

    result = ParsedAssessment(raw=text)

    # Look for score in multiple possible field names
    raw_score = fields.get(score_field) or fields.get("aria_shapedness")

    # Validate score is numeric
    if raw_score is None:
        result.parse_error = "No score found"
    elif isinstance(raw_score, int):
        result.score = raw_score
    elif isinstance(raw_score, str) and raw_score.isdigit():
        result.score = int(raw_score)
    else:
        result.parse_error = f"Non-numeric score: {raw_score}"

    # Extract common fields
    result.analysis = fields.get("analysis") or fields.get("impression", "")
    result.needs_review = fields.get("needs_review", False)
    result.errors = fields.get("errors", "")
    result.improvement = fields.get("improvement", "")

    # Store any remaining fields in extra
    known_fields = {"raw", score_field, "aria_shapedness", "analysis", "impression",
                    "needs_review", "errors", "improvement"}
    result.extra = {k: v for k, v in fields.items() if k not in known_fields}

    return result
