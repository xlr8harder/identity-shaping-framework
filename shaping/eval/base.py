"""Base types and classes for the eval framework.

Provides declarative eval definitions that are hard to misconfigure.

Example usage:

    class MyEval(Eval):
        name = "my-eval"
        prompt_template = "Question: {question}"
        judge = MCParser(gold_field="answer")
        metrics = AccuracyMetrics()

    # Run via CLI or programmatically:
    runner = EvalRunner(MyEval())
    results = await runner.run(
        input_path="test.jsonl",
        model="my-model",
        output_dir="results/",
    )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """Result of evaluating a single sample.

    Attributes:
        sample_id: Identifier for the sample (from input or generated)
        prompt: The formatted prompt sent to the model
        response: The model's response
        score: Numeric score (int for accuracy, float for scaled)
        correct: Boolean for accuracy-based evals
        gold: Expected answer (for accuracy evals)
        extracted: Extracted answer from response
        analysis: Judge's analysis (for LLM-judged evals)
        metadata: Additional eval-specific data
        error: Error message if eval failed
    """

    sample_id: str
    prompt: str
    response: str = ""
    score: float | None = None
    correct: bool | None = None
    gold: Any = None
    extracted: Any = None
    analysis: str = ""
    metadata: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether the eval completed without error."""
        return self.error is None


@dataclass
class EvalMetrics:
    """Aggregated metrics from an eval run.

    Attributes:
        total: Total number of samples
        completed: Number that completed without error
        failed: Number that failed with errors

        # For accuracy-based evals
        correct: Number correct
        accuracy: Proportion correct

        # For scored evals
        mean_score: Average score
        score_distribution: Count per score value

        # Additional stats
        extra: Eval-specific additional metrics
    """

    total: int = 0
    completed: int = 0
    failed: int = 0

    # Accuracy metrics
    correct: int = 0
    accuracy: float = 0.0

    # Scored metrics
    mean_score: float = 0.0
    score_distribution: dict[int, int] = field(default_factory=dict)

    # Extra
    extra: dict = field(default_factory=dict)


class Judge(ABC):
    """Base class for result judges/parsers.

    Judges take a model response and produce a score/result.
    Two types:
    - Fixed parsers (MCParser): Extract answer via patterns
    - LLM judges (LLMJudge): Use another model to evaluate
    """

    @abstractmethod
    async def judge(
        self,
        response: str,
        sample: dict,
        prompt: str,
    ) -> EvalResult:
        """Judge a response and return result.

        Args:
            response: Model's response text
            sample: Original sample from input (for gold answer, etc.)
            prompt: The prompt that was sent to the model

        Returns:
            EvalResult with score/correct/extracted filled in
        """
        pass


class MetricsAggregator(ABC):
    """Base class for metrics aggregation."""

    @abstractmethod
    def aggregate(self, results: list[EvalResult]) -> EvalMetrics:
        """Aggregate individual results into metrics."""
        pass


class Eval(ABC):
    """Base class for eval definitions.

    Subclasses must define:
    - name: Identifier for this eval
    - judge: How to score responses (MCParser, LLMJudge, etc.)

    Data source (one of these required):
    - hf_dataset: HuggingFace dataset name (e.g., "fingertap/GPQA-Diamond")
    - local_path: Path to local JSONL file (relative to project root)

    Prompt config (one of these required):
    - prompt_template: String template with {field} placeholders
    - prompt_field: Field name in input to use directly as prompt

    Optional:
    - hf_subset: Dataset subset/config (for multi-config datasets)
    - hf_split: Dataset split (default: "train")
    - field_mapping: Dict mapping sample fields to expected names
    - metrics: Custom metrics aggregator (defaults based on judge type)
    """

    # Required
    name: str

    # Data source (one of these required)
    hf_dataset: str | None = None  # e.g., "fingertap/GPQA-Diamond"
    hf_subset: str | None = None  # e.g., "gpqa_diamond" for multi-config datasets
    hf_split: str = "train"  # Default split to load
    local_path: str | None = None  # Relative to project root

    # Field mapping for non-standard datasets
    # Maps: {expected_name: dataset_field_name}
    # e.g., {"question": "Question", "answer": "Correct Answer"}
    field_mapping: dict[str, str] | None = None

    # Prompt configuration (one of these required)
    prompt_template: str | None = None
    prompt_field: str | None = None

    # Judge configuration (required)
    judge: Judge

    # Metrics (optional, defaults based on judge)
    metrics: MetricsAggregator | None = None

    # Generation parameters
    max_tokens: int | None = None  # Max tokens for model response
    temperature: float | None = None  # Sampling temperature

    def format_prompt(self, sample: dict) -> str:
        """Format a sample into a prompt string.

        Override for custom formatting logic.
        """
        if self.prompt_template:
            try:
                return self.prompt_template.format(**sample)
            except KeyError as e:
                raise ValueError(
                    f"Sample missing field {e} required by prompt_template. "
                    f"Sample keys: {list(sample.keys())}"
                )
        elif self.prompt_field:
            if self.prompt_field not in sample:
                raise ValueError(
                    f"Sample missing prompt_field '{self.prompt_field}'. "
                    f"Sample keys: {list(sample.keys())}"
                )
            return sample[self.prompt_field]
        else:
            raise ValueError(
                f"Eval '{self.name}' must define either prompt_template or prompt_field"
            )

    def validate(self, sample: dict) -> None:
        """Validate that a sample has required fields.

        Called on first sample to catch config errors early.
        Override to add eval-specific validation.
        """
        # Try formatting to catch template errors
        self.format_prompt(sample)

        # Let judge validate its requirements
        if hasattr(self.judge, "validate"):
            self.judge.validate(sample)

    def get_sample_id(self, sample: dict, index: int) -> str:
        """Get identifier for a sample.

        Override to use a different field.
        """
        return sample.get("id", f"sample-{index}")

    def _apply_field_mapping(self, sample: dict) -> dict:
        """Apply field mapping to a sample.

        Copies the sample and adds aliased fields according to field_mapping.
        Original fields are preserved, and new keys are added with the expected names.
        """
        if not self.field_mapping:
            return sample

        result = dict(sample)
        for expected_name, dataset_field in self.field_mapping.items():
            if dataset_field in result:
                result[expected_name] = result[dataset_field]
        return result

    def load_samples(
        self,
        limit: int | None = None,
        seed: int | None = None,
    ) -> list[dict]:
        """Load samples from the configured data source.

        Args:
            limit: Maximum number of samples to return
            seed: Random seed for shuffling (if provided, shuffles before limit)

        Returns:
            List of sample dicts with field mapping applied
        """
        import random

        if self.hf_dataset:
            samples = self._load_from_hf()
        elif self.local_path:
            samples = self._load_from_local()
        else:
            raise ValueError(
                f"Eval '{self.name}' must define either hf_dataset or local_path"
            )

        # Apply field mapping
        samples = [self._apply_field_mapping(s) for s in samples]

        # Shuffle if seed provided (use local RNG to avoid global state mutation)
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(samples)

        # Apply limit
        if limit is not None:
            samples = samples[:limit]

        return samples

    def _load_from_hf(self) -> list[dict]:
        """Load samples from HuggingFace dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package required for HuggingFace datasets. "
                "Install with: pip install datasets"
            )

        # Load dataset
        if self.hf_subset:
            dataset = load_dataset(self.hf_dataset, self.hf_subset, split=self.hf_split)
        else:
            dataset = load_dataset(self.hf_dataset, split=self.hf_split)

        # Convert to list of dicts
        return [dict(sample) for sample in dataset]

    def _load_from_local(self) -> list[dict]:
        """Load samples from local JSONL file."""
        import json
        from pathlib import Path

        # local_path is relative to project root
        # We'll resolve it relative to the current working directory
        path = Path(self.local_path)
        if not path.is_absolute():
            # Assume cwd is project root
            path = Path.cwd() / path

        if not path.exists():
            raise FileNotFoundError(f"Local data file not found: {path}")

        samples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        return samples
