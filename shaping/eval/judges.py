"""Judge implementations for evaluating model responses.

Provides:
- MCParser: Extract multiple-choice answers (A/B/C/D)
- LLMJudge: Use an LLM to evaluate responses with a rubric
"""

import re
from dataclasses import dataclass

from .base import Judge, EvalResult
from .parsers import parse_assessment_xml


# =============================================================================
# MCParser - Multiple Choice Extraction
# =============================================================================

# Answer extraction patterns (from GPQA/AA style)
_PRIMARY_ANSWER_RE = re.compile(
    r"(?i)[*_]{0,2}Answer[*_]{0,2}\s*:[\s*_]{0,2}\s*([A-D])(?![a-zA-Z0-9])"
)

_FALLBACK_PATTERNS = [
    re.compile(r"\\boxed\{[^}]*([A-D])[^}]*\}"),
    re.compile(r"answer is ([a-dA-D])", re.IGNORECASE),
    re.compile(r"answer is \(([a-dA-D])", re.IGNORECASE),
    re.compile(r"(?m)^[^\S\r\n]*([A-D])\)\s*[^A-Z]*"),
    re.compile(r"([A-D])\s+is\s+the\s+correct\s+answer", re.IGNORECASE),
    re.compile(r"([A-D])\s*$"),
    re.compile(r"([A-D])\s*\."),
]


def extract_mc_answer(text: str) -> str | None:
    """Extract multiple-choice answer (A-D) from model output.

    Uses multiple patterns in order of reliability:
    1. Explicit "Answer: X" format (last occurrence)
    2. Boxed answer
    3. "answer is X" format
    4. Standalone letter at end

    Returns:
        Single letter A-D, or None if not found
    """
    stripped = text.strip()
    if len(stripped) == 1 and stripped in "ABCD":
        return stripped

    # Primary pattern: "Answer: X" - use last occurrence
    primary_matches = [m.group(1).upper() for m in _PRIMARY_ANSWER_RE.finditer(text)]
    if primary_matches:
        letter = primary_matches[-1]
        return letter if letter in {"A", "B", "C", "D"} else None

    # Fallback patterns
    for pattern in _FALLBACK_PATTERNS:
        pattern_matches = [m.group(1).upper() for m in pattern.finditer(text)]
        if pattern_matches:
            letter = pattern_matches[-1]
            return letter if letter in {"A", "B", "C", "D"} else None

    return None


@dataclass
class MCParser(Judge):
    """Multiple-choice answer parser.

    Extracts A/B/C/D answers from responses and compares to gold.

    Attributes:
        gold_field: Field name in sample containing correct answer
    """

    gold_field: str = "answer"

    def validate(self, sample: dict) -> None:
        """Validate sample has gold field."""
        if self.gold_field not in sample:
            raise ValueError(
                f"MCParser requires '{self.gold_field}' field in samples. "
                f"Sample keys: {list(sample.keys())}"
            )

    async def judge(
        self,
        response: str,
        sample: dict,
        prompt: str,
    ) -> EvalResult:
        """Extract answer and compare to gold."""
        gold = sample.get(self.gold_field)
        extracted = extract_mc_answer(response)

        correct = extracted == gold if extracted else False

        return EvalResult(
            sample_id=sample.get("id", ""),
            prompt=prompt,
            response=response,
            score=1 if correct else 0,
            correct=correct,
            gold=gold,
            extracted=extracted,
        )


# =============================================================================
# LLMJudge - LLM-based Evaluation
# =============================================================================

# Default XML output format for judges
DEFAULT_JUDGE_FORMAT = """<evaluation>
<analysis>Brief analysis of the response</analysis>
<score>{score_range}</score>
</evaluation>"""


@dataclass
class LLMJudge(Judge):
    """LLM-based judge that evaluates responses using a rubric.

    Sends the prompt+response to a judge model with a rubric,
    then parses the structured XML output.

    Attributes:
        rubric: The scoring rubric text
        judge_model: Model shortname (resolved via config)
        score_field: XML field containing the score (default: "score")
        max_score: Maximum score value (for normalization, default: 5)
        output_format: XML format for judge output (optional)
        include_prompt: Whether to include original prompt in judge query
    """

    rubric: str
    judge_model: str = "isf.judge"
    score_field: str = "score"
    max_score: int = 5
    output_format: str | None = None
    include_prompt: bool = True

    # Set by runner
    _client: object = None

    def _build_judge_prompt(self, prompt: str, response: str) -> str:
        """Build the prompt for the judge model."""
        output_fmt = self.output_format or DEFAULT_JUDGE_FORMAT.format(
            score_range=f"1-{self.max_score}"
        )

        if self.include_prompt:
            return f"""Evaluate this response.

## Prompt
\"\"\"{prompt}\"\"\"

## Response
\"\"\"{response}\"\"\"

---

{self.rubric}

Provide your evaluation in this XML format:

{output_fmt}"""
        else:
            return f"""Evaluate this response.

## Response
\"\"\"{response}\"\"\"

---

{self.rubric}

Provide your evaluation in this XML format:

{output_fmt}"""

    async def judge(
        self,
        response: str,
        sample: dict,
        prompt: str,
    ) -> EvalResult:
        """Send response to judge model and parse result."""
        if self._client is None:
            raise RuntimeError(
                "LLMJudge._client not set. "
                "The EvalRunner should set this before calling judge()."
            )

        judge_prompt = self._build_judge_prompt(prompt, response)

        try:
            # Query judge model
            judge_response = await self._client.query_async([
                {"role": "user", "content": judge_prompt}
            ])

            # Parse XML response
            parsed = parse_assessment_xml(judge_response, score_field=self.score_field)

            return EvalResult(
                sample_id=sample.get("id", ""),
                prompt=prompt,
                response=response,
                score=parsed.score,
                correct=parsed.score == self.max_score if parsed.score else None,
                analysis=parsed.analysis,
                metadata={
                    "judge_response": judge_response,
                    "parse_error": parsed.parse_error,
                    "needs_review": parsed.needs_review,
                    "errors": parsed.errors,
                },
                error=parsed.parse_error,
            )

        except Exception as e:
            return EvalResult(
                sample_id=sample.get("id", ""),
                prompt=prompt,
                response=response,
                error=f"Judge error: {e}",
            )


# =============================================================================
# Metrics Aggregators
# =============================================================================

from .base import MetricsAggregator, EvalMetrics


@dataclass
class AccuracyMetrics(MetricsAggregator):
    """Aggregates accuracy-based results (correct/incorrect)."""

    def aggregate(self, results: list[EvalResult]) -> EvalMetrics:
        """Compute accuracy metrics."""
        total = len(results)
        completed = sum(1 for r in results if r.success)
        failed = total - completed

        correct = sum(1 for r in results if r.success and r.correct)
        accuracy = correct / completed if completed > 0 else 0.0

        return EvalMetrics(
            total=total,
            completed=completed,
            failed=failed,
            correct=correct,
            accuracy=accuracy,
        )


@dataclass
class ScoredMetrics(MetricsAggregator):
    """Aggregates score-based results (1-N scale)."""

    max_score: int = 5

    def aggregate(self, results: list[EvalResult]) -> EvalMetrics:
        """Compute scored metrics."""
        total = len(results)
        completed = sum(1 for r in results if r.success)
        failed = total - completed

        # Get valid scores
        scores = [r.score for r in results if r.success and r.score is not None]

        mean_score = sum(scores) / len(scores) if scores else 0.0

        # Build distribution
        distribution = {i: 0 for i in range(1, self.max_score + 1)}
        for s in scores:
            if isinstance(s, int) and 1 <= s <= self.max_score:
                distribution[s] += 1

        return EvalMetrics(
            total=total,
            completed=completed,
            failed=failed,
            mean_score=mean_score,
            score_distribution=distribution,
            extra={"scored_count": len(scores)},
        )
