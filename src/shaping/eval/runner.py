"""Eval runner for executing evaluations.

Handles:
- Loading samples from eval's configured data source (HF or local JSONL)
- Running model inference (with optional multiple runs per sample)
- Concurrent execution with progress reporting
- Saving results (JSONL detail + JSON summary)

Usage:
    from shaping.eval import EvalRunner
    from my_evals import GPQAEval

    runner = EvalRunner(GPQAEval())
    results = await runner.run(
        model="my-model",
        output_dir="results/",
    )

Output files:
    {eval}-{model}-{timestamp}.jsonl  # One row per generation
    {eval}-{model}-{timestamp}.json   # Summary with aggregated metrics
"""

import asyncio
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Any

from .base import Eval, EvalMetrics
from .judges import LLMJudge, MCParser, AccuracyMetrics, ScoredMetrics


def _get_config_worker_concurrency() -> int | None:
    """Look up worker_concurrency from isf.yaml."""
    try:
        from pathlib import Path

        import yaml

        # Find isf.yaml
        current = Path.cwd()
        while current != current.parent:
            config_path = current / "isf.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                return config.get("worker_concurrency")
            current = current.parent
    except Exception:
        pass
    return None


def _get_default_concurrency() -> int:
    """Get default concurrency from isf.yaml or fall back to 20."""
    concurrency = _get_config_worker_concurrency()
    return concurrency if concurrency is not None else 20


@dataclass
class JudgmentRecord:
    """Record of a single judgment call."""

    judge: str
    raw: str
    score: float | int | None
    parsed: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class GenerationRecord:
    """Record of a single generation (one row in output JSONL).

    Designed for:
    - JSONL files for detailed review
    - Eventual DB storage at aggregate level
    - Unified schema across eval types
    """

    # Metadata (for foolproof identification)
    _eval: str = ""  # Eval name (e.g., "gpqa", "wildchat")
    _version: int = 1  # Schema version
    _score_type: str = "binary"  # "binary" or "scaled:N"
    model: str = ""  # Model being evaluated

    # Sample identity
    sample_id: str = ""
    run: int = 1  # Which generation (1-indexed)

    # Core result (always present)
    final_score: float | None = None  # 0-1 for binary, 1-N for scaled

    # Generation content
    prompt: str = ""
    response: str = ""

    # Eval-specific (nullable)
    gold: Any = None  # Expected answer
    extracted: Any = None  # Extracted answer (for MC/parsing evals)

    # Judgments (for LLM-judged or multiple samples)
    judgments: list[JudgmentRecord] = field(default_factory=list)

    # Error tracking
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "_eval": self._eval,
            "_version": self._version,
            "_score_type": self._score_type,
            "model": self.model,
            "sample_id": self.sample_id,
            "run": self.run,
            "final_score": self.final_score,
            "prompt": self.prompt,
            "response": self.response,
            "gold": self.gold,
            "extracted": self.extracted,
            "judgments": [asdict(j) for j in self.judgments],
            "error": self.error,
        }


class EvalRunner:
    """Runs an eval against a model.

    Handles all the infrastructure: loading data, model inference,
    concurrency, progress reporting, and result saving.
    """

    def __init__(self, eval_def: Eval):
        """Initialize runner with an eval definition.

        Args:
            eval_def: The Eval subclass instance defining this evaluation
        """
        self.eval_def = eval_def

        # Set default metrics based on judge type
        if eval_def.metrics is None:
            if isinstance(eval_def.judge, MCParser):
                eval_def.metrics = AccuracyMetrics()
            elif isinstance(eval_def.judge, LLMJudge):
                eval_def.metrics = ScoredMetrics(max_score=eval_def.judge.max_score)
            else:
                eval_def.metrics = AccuracyMetrics()

    async def run(
        self,
        model: str,
        output_dir: str | Path | None = None,
        limit: int | None = None,
        seed: int | None = None,
        runs_per_sample: int = 1,
        judges_per_run: int = 1,
        concurrency: int | None = None,
        temperature: float | None = None,
        judge_temperature: float = 0.3,
        save_results: bool = True,
        quiet: bool = False,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> tuple[list[GenerationRecord], EvalMetrics, dict | None]:
        """Run the evaluation.

        Args:
            model: Model spec (mq shortname or checkpoint spec)
            output_dir: Directory to save results (default: auto from model spec)
            limit: Limit number of samples (for quick testing)
            seed: Random seed for sample shuffling
            runs_per_sample: Number of generations per sample (for variance)
            judges_per_run: Number of judge calls per generation (for judge variance)
            concurrency: Max concurrent evaluations (default: isf.yaml worker_concurrency or 20)
            temperature: Model temperature (default: model's default)
            judge_temperature: Judge model temperature (default: 0.3 for consistency)
            save_results: Whether to save results to disk
            quiet: Suppress progress output
            progress_callback: Optional callback(completed, total, avg_score)

        Returns:
            (list of GenerationRecord, aggregated EvalMetrics, output_files dict or None)
            output_files contains 'detail' and 'summary' paths if save_results=True
        """
        if output_dir:
            output_dir = Path(output_dir)

        # Resolve concurrency default
        if concurrency is None:
            concurrency = _get_default_concurrency()

        # Load samples from eval's configured data source
        samples = self.eval_def.load_samples(limit=limit, seed=seed)
        if not samples:
            source = self.eval_def.hf_dataset or self.eval_def.local_path
            raise ValueError(f"No samples loaded from {source}")

        # Validate first sample
        self.eval_def.validate(samples[0])

        # Initialize model client
        # Use eval's settings, with run() params as override
        model_temp = (
            temperature if temperature is not None else self.eval_def.temperature
        )
        model_max_tokens = self.eval_def.max_tokens
        model_client = await self._create_model_client(
            model, model_temp, model_max_tokens
        )

        # Initialize judge client if needed
        judge_name = None
        if isinstance(self.eval_def.judge, LLMJudge):
            judge_client = await self._create_judge_client(
                self.eval_def.judge.judge_model, temperature=judge_temperature
            )
            self.eval_def.judge._client = judge_client
            judge_name = self.eval_def.judge.judge_model

        # Calculate total work
        total_generations = len(samples) * runs_per_sample

        if not quiet:
            print(f"Eval: {self.eval_def.name}")
            print(f"Model: {model}")
            print(f"Samples: {len(samples)}")
            if runs_per_sample > 1:
                print(
                    f"Runs per sample: {runs_per_sample} ({total_generations} total generations)"
                )
            if judges_per_run > 1:
                print(f"Judge calls per run: {judges_per_run}")
            if judge_name:
                print(f"Judge: {judge_name}")
            print(f"Concurrency: {concurrency}")
            print()

        # Run evaluations
        records = await self._run_concurrent(
            samples=samples,
            model_client=model_client,
            model_name=model,
            runs_per_sample=runs_per_sample,
            judges_per_run=judges_per_run,
            judge_name=judge_name,
            concurrency=concurrency,
            quiet=quiet,
            progress_callback=progress_callback,
        )

        # Compute metrics from generation records
        metrics = self._compute_metrics(records)

        # Display summary
        if not quiet:
            self._display_metrics(metrics)

        # Save results
        output_files = None
        if save_results:
            output_files = self._save_results(
                records=records,
                metrics=metrics,
                model=model,
                output_dir=output_dir,
                config={
                    "limit": limit,
                    "seed": seed,
                    "runs_per_sample": runs_per_sample,
                    "judges_per_run": judges_per_run,
                    "temperature": temperature,
                    "judge_temperature": judge_temperature,
                },
            )
            if not quiet:
                print("\nResults saved to:")
                print(f"  Detail: {output_files['detail']}")
                print(f"  Summary: {output_files['summary']}")

        return records, metrics, output_files

    async def _create_model_client(
        self,
        model: str,
        temperature: float | None,
        max_tokens: int | None,
    ):
        """Create model client based on spec."""
        from ..modeling import LLMClient

        # Build kwargs, excluding None values (let LLMClient use its defaults)
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # For now, assume all models are LLMClient compatible
        # TODO: Add TinkerClient support for checkpoint specs
        return LLMClient(model, **kwargs)

    async def _create_judge_client(self, model: str, temperature: float | None):
        """Create judge client."""
        from ..modeling import LLMClient

        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return LLMClient(model, **kwargs)

    async def _run_concurrent(
        self,
        samples: list[dict],
        model_client,
        model_name: str,
        runs_per_sample: int,
        judges_per_run: int,
        judge_name: str | None,
        concurrency: int,
        quiet: bool,
        progress_callback: Callable | None,
    ) -> list[GenerationRecord]:
        """Run evaluations concurrently."""
        semaphore = asyncio.Semaphore(concurrency)

        # Build list of all (sample, run_idx) pairs
        work_items = [
            (sample_idx, sample, run_idx)
            for sample_idx, sample in enumerate(samples)
            for run_idx in range(runs_per_sample)
        ]

        records: list[GenerationRecord] = [None] * len(work_items)

        # Progress tracking
        completed = 0
        total_score = 0.0
        scored_count = 0
        lock = asyncio.Lock()

        async def run_one(
            work_idx: int, sample_idx: int, sample: dict, run_idx: int
        ) -> None:
            nonlocal completed, total_score, scored_count

            async with semaphore:
                record = await self._run_single(
                    sample=sample,
                    sample_idx=sample_idx,
                    run_idx=run_idx,
                    model_client=model_client,
                    model_name=model_name,
                    judges_per_run=judges_per_run,
                    judge_name=judge_name,
                )
                records[work_idx] = record

                # Update progress
                async with lock:
                    completed += 1
                    if record.final_score is not None:
                        total_score += record.final_score
                        scored_count += 1

                    avg = total_score / scored_count if scored_count > 0 else 0

                    if not quiet:
                        print(
                            f"  Progress: {completed}/{len(work_items)} (avg: {avg:.2f})"
                        )

                    if progress_callback:
                        progress_callback(completed, len(work_items), avg)

        tasks = [
            asyncio.create_task(run_one(work_idx, sample_idx, sample, run_idx))
            for work_idx, (sample_idx, sample, run_idx) in enumerate(work_items)
        ]
        await asyncio.gather(*tasks)

        return records

    async def _run_single(
        self,
        sample: dict,
        sample_idx: int,
        run_idx: int,
        model_client,
        model_name: str,
        judges_per_run: int,
        judge_name: str | None,
    ) -> GenerationRecord:
        """Run a single generation and judge it."""
        sample_id = self.eval_def.get_sample_id(sample, sample_idx)

        # Determine score type
        if isinstance(self.eval_def.judge, MCParser):
            score_type = "binary"
        elif isinstance(self.eval_def.judge, LLMJudge):
            max_score = self.eval_def.judge.max_score
            score_type = f"scaled:{max_score}"
        else:
            score_type = "unknown"

        record = GenerationRecord(
            _eval=self.eval_def.name,
            _version=1,
            _score_type=score_type,
            model=model_name,
            sample_id=sample_id,
            run=run_idx + 1,  # 1-indexed for human readability
        )

        try:
            # Format prompt
            prompt = self.eval_def.format_prompt(sample)
            record.prompt = prompt

            # Get model response
            response = await model_client.query_async(
                [{"role": "user", "content": prompt}]
            )
            record.response = response

            # Judge the response (possibly multiple times)
            judgments = []
            scores = []

            for judge_idx in range(judges_per_run):
                result = await self.eval_def.judge.judge(
                    response=response,
                    sample=sample,
                    prompt=prompt,
                )

                judgment = JudgmentRecord(
                    judge=judge_name or "fixed_parser",
                    raw=result.metadata.get("judge_response", "")
                    if result.metadata
                    else "",
                    score=result.score,
                    parsed={
                        "analysis": result.analysis,
                        "needs_review": result.metadata.get("needs_review")
                        if result.metadata
                        else None,
                        "errors": result.metadata.get("errors")
                        if result.metadata
                        else None,
                    },
                    error=result.error,
                )
                judgments.append(judgment)

                if result.score is not None:
                    scores.append(result.score)

                # For fixed parsers, also capture gold/extracted
                if result.gold is not None:
                    record.gold = result.gold
                if result.extracted is not None:
                    record.extracted = result.extracted

            record.judgments = judgments

            # Compute final score (mean of all judgment scores)
            if scores:
                record.final_score = sum(scores) / len(scores)

        except Exception as e:
            record.error = str(e)

        return record

    def _compute_metrics(self, records: list[GenerationRecord]) -> EvalMetrics:
        """Compute metrics from generation records."""
        total = len(records)
        completed = sum(1 for r in records if r.error is None)
        failed = total - completed

        # For accuracy-based evals (MCParser)
        if isinstance(self.eval_def.judge, MCParser):
            # final_score is 1 for correct, 0 for incorrect
            correct = sum(1 for r in records if r.error is None and r.final_score == 1)
            accuracy = correct / completed if completed > 0 else 0.0
            return EvalMetrics(
                total=total,
                completed=completed,
                failed=failed,
                correct=correct,
                accuracy=accuracy,
            )

        # For scored evals (LLMJudge)
        # Parse failures count as minimum score (0 for binary, 1 for 1-N scale)
        max_score = getattr(self.eval_def.judge, "max_score", 5)
        min_score = 0 if max_score == 1 else 1

        scores = []
        parse_failures = 0
        for r in records:
            if r.error is None:
                if r.final_score is not None:
                    scores.append(r.final_score)
                else:
                    # Parse failure - count as minimum score
                    scores.append(min_score)
                    parse_failures += 1

        mean_score = sum(scores) / len(scores) if scores else 0.0

        # Build distribution (round to int for binning)
        distribution = {i: 0 for i in range(1, max_score + 1)}
        for s in scores:
            rounded = round(s)
            if 1 <= rounded <= max_score:
                distribution[rounded] += 1

        extra = {"scored_count": len(scores) - parse_failures}
        if parse_failures > 0:
            extra["parse_failures"] = parse_failures

        return EvalMetrics(
            total=total,
            completed=completed,
            failed=failed,
            mean_score=mean_score,
            score_distribution=distribution,
            extra=extra,
        )

    def _display_metrics(self, metrics: EvalMetrics) -> None:
        """Display metrics summary."""
        print()
        print("=" * 60)
        print(f"RESULTS: {self.eval_def.name}")
        print("=" * 60)
        print(f"Total generations: {metrics.total}")
        print(f"Completed: {metrics.completed}")
        print(f"Failed: {metrics.failed}")

        if isinstance(self.eval_def.judge, MCParser):
            print(f"\nCorrect: {metrics.correct}")
            print(f"Accuracy: {metrics.accuracy:.1%}")
        else:
            print(f"\nMean score: {metrics.mean_score:.2f}")
            # Show parse failures if any (these counted as min score)
            parse_failures = metrics.extra.get("parse_failures", 0)
            if parse_failures > 0:
                print(f"  (includes {parse_failures} parse failures scored as minimum)")
            if metrics.score_distribution:
                print("\nScore distribution:")
                for score in sorted(metrics.score_distribution.keys()):
                    count = metrics.score_distribution[score]
                    bar = "*" * count
                    print(f"  {score}: {bar} ({count})")

        print("=" * 60)

    def _save_results(
        self,
        records: list[GenerationRecord],
        metrics: EvalMetrics,
        model: str,
        output_dir: Path | None,
        config: dict,
    ) -> dict[str, Path]:
        """Save results to JSONL detail + JSON summary files."""
        # Determine output directory
        if output_dir is None:
            # Try to extract experiment name from model spec
            if "-" in model and model.split("-")[0].lower().startswith("e"):
                exp_name = model.split("-")[0].upper()
                output_dir = Path(f"results/{exp_name}")
            else:
                output_dir = Path(f"results/{self.eval_def.name}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename base
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_model = model.replace("/", "-").replace("::", "-")
        base_name = f"{self.eval_def.name}-{safe_model}-{timestamp}"

        # Save detail file (JSONL - one row per generation)
        detail_file = output_dir / f"{base_name}.jsonl"
        with open(detail_file, "w") as f:
            for record in records:
                f.write(json.dumps(record.to_dict(), default=str) + "\n")

        # Determine data source for summary
        data_source = self.eval_def.hf_dataset or self.eval_def.local_path

        # Save summary file (JSON)
        summary_file = output_dir / f"{base_name}.json"
        summary_data = {
            "eval": self.eval_def.name,
            "model": model,
            "data_source": data_source,
            "timestamp": timestamp,
            "config": config,
            "metrics": asdict(metrics),
        }

        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)

        return {"detail": detail_file, "summary": summary_file}


# Convenience function for running evals
async def run_eval(
    eval_def: Eval,
    model: str,
    **kwargs,
) -> tuple[list[GenerationRecord], EvalMetrics, dict | None]:
    """Convenience function to run an eval.

    Args:
        eval_def: The Eval subclass instance (with data source configured)
        model: Model spec
        **kwargs: Additional arguments passed to EvalRunner.run()

    Returns:
        (records, metrics, output_files)
    """
    runner = EvalRunner(eval_def)
    return await runner.run(model=model, **kwargs)
