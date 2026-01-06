"""Evaluation framework for identity-shaping.

Provides a declarative framework for defining and running evaluations.

## Quick Start

Define an eval with its data source:

    from shaping.eval import Eval, MCParser, LLMJudge

    class GPQAEval(Eval):
        name = "gpqa-diamond"
        hf_dataset = "fingertap/GPQA-Diamond"
        hf_split = "train"
        field_mapping = {"question": "Question", "answer": "Answer"}
        prompt_template = "Question: {question}\\nA) {A}\\nB) {B}..."
        judge = MCParser(gold_field="answer")

    class WildchatEval(Eval):
        name = "wildchat"
        local_path = "evals/wildchat.jsonl"
        prompt_field = "prompt"
        judge = LLMJudge(rubric=MY_RUBRIC, judge_model="aria-v1.0")

Run an eval:

    from shaping.eval import EvalRunner, run_eval

    # Option 1: Using runner directly
    runner = EvalRunner(GPQAEval())
    results, metrics = await runner.run(model="my-model")

    # Option 2: Convenience function
    results, metrics = await run_eval(GPQAEval(), model="my-model")

## Components

- **Eval**: Base class for eval definitions
  - hf_dataset/local_path: Data source
  - field_mapping: Map dataset fields to expected names
- **Judge**: Base class for response evaluation
  - MCParser: Extract multiple-choice answers
  - LLMJudge: Use LLM to evaluate with rubric
- **Metrics**: Aggregation for results
  - AccuracyMetrics: For correct/incorrect evals
  - ScoredMetrics: For scored (1-N) evals
- **EvalRunner**: Handles execution, concurrency, saving
"""

# Core types
from .base import Eval, EvalResult, EvalMetrics, Judge, MetricsAggregator

# Judges and metrics
from .judges import (
    MCParser,
    LLMJudge,
    AccuracyMetrics,
    ScoredMetrics,
    extract_mc_answer,
)

# Parsers (lower-level)
from .parsers import parse_xml_fields, ParsedAssessment, parse_assessment_xml

# Runner and record types
from .runner import EvalRunner, run_eval, GenerationRecord, JudgmentRecord

__all__ = [
    # Core types
    "Eval",
    "EvalResult",
    "EvalMetrics",
    "Judge",
    "MetricsAggregator",
    # Judges
    "MCParser",
    "LLMJudge",
    # Metrics
    "AccuracyMetrics",
    "ScoredMetrics",
    # Utilities
    "extract_mc_answer",
    "parse_xml_fields",
    "ParsedAssessment",
    "parse_assessment_xml",
    # Runner
    "EvalRunner",
    "run_eval",
    "GenerationRecord",
    "JudgmentRecord",
]
