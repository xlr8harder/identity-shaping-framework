"""Pipeline base class.

Provides the Pipeline abstraction for multi-stage data generation:

    class IdentityAugmentation(Pipeline):
        narrative_doc = Pipeline.file_dep("identity/NARRATIVE.md")
        identity_model = Pipeline.model_dep("cubsfan-release-full")
        judge_model = Pipeline.model_dep("judge")

        def run(self):
            # Stage 1: Single LLM call (returns QueryResponse)
            response = self.query(
                model=self.judge_model,
                messages=[{"role": "user", "content": f"Extract:\\n{self.narrative_doc.read()}"}],
            )
            facts = parse_facts(response.get_text())

            # Stage 2: Run task across records (parallel)
            results = self.run_task(self.generate_qa, records=facts)
            return results

        def generate_qa(self, record):
            question = yield model_request([...], model=self.judge_model)
            response = yield model_request([...], model=self.identity_model)
            return TrainingSample(...)
"""

import hashlib
import inspect
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from .deps import ModelDep, FileDep, get_all_deps

if TYPE_CHECKING:
    from .provenance import QueryResponse

logger = logging.getLogger(__name__)


class Pipeline:
    """Base class for data generation pipelines.

    Subclasses declare dependencies as class attributes and implement run().
    """

    # Pipeline name - used for output paths and manifest
    name: str = None  # type: ignore

    # Output directory for training data
    output_dir: Path = Path("training/data")

    # Whether to include full provenance in output (default True)
    # Set to False for training-ready minimal output (just id + messages)
    annotated: bool = True

    # Worker count for run_task() - if not set, inherits from isf.yaml pipeline_workers
    workers: Optional[int] = None

    @staticmethod
    def file_dep(path: str) -> FileDep:
        """Create a file dependency.

        Usage:
            narrative_doc = Pipeline.file_dep("identity/NARRATIVE.md")
        """
        return FileDep(path)

    @staticmethod
    def model_dep(registry_name: str) -> ModelDep:
        """Create a model dependency.

        Usage:
            identity_model = Pipeline.model_dep("cubsfan-release-full")
        """
        return ModelDep(registry_name)

    def __init__(self):
        """Initialize pipeline instance."""
        if self.name is None:
            raise ValueError(f"{self.__class__.__name__} must define 'name' attribute")

        self._started_at: Optional[datetime] = None
        self._completed_at: Optional[datetime] = None
        self._partial: bool = False  # Set to True when run with --limit
        self._limit: Optional[int] = None  # Record limit for run_task()

    def _get_workers(self) -> int:
        """Get worker count using fallback chain.

        Priority:
        1. Pipeline.workers class attribute (if set)
        2. isf.yaml pipeline_workers global setting
        3. Default 50
        """
        # Check explicit workers attribute
        if self.workers is not None:
            return self.workers

        # Check global config
        workers = self._get_config_pipeline_workers()
        if workers is not None:
            return workers

        # Default
        return 50

    def _get_config_pipeline_workers(self) -> Optional[int]:
        """Look up pipeline_workers from isf.yaml."""
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
                    return config.get("pipeline_workers")
                current = current.parent
        except Exception:
            pass
        return None

    @classmethod
    def get_output_file(cls) -> Path:
        """Get output file path for this pipeline."""
        if cls.name is None:
            raise ValueError(f"{cls.__name__} must define 'name' attribute")
        return cls.output_dir / f"{cls.name}.jsonl"

    @classmethod
    def get_manifest_file(cls) -> Path:
        """Get manifest file path for this pipeline."""
        return cls.get_output_file().with_suffix(".manifest.json")

    def run(self) -> list[Any]:
        """Execute the pipeline. Override in subclasses.

        Returns:
            List of results (typically TrainingSample instances)
        """
        raise NotImplementedError("Subclasses must implement run()")

    def query(
        self,
        model: ModelDep,
        messages: list[dict],
        **kwargs,
    ) -> "QueryResponse":
        """Make a single LLM query.

        Args:
            model: ModelDep for the model to use
            messages: Chat messages in OpenAI format
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            QueryResponse with .get_text() and .is_success (same interface as
            yield model_request() in task methods)
        """
        from .provenance import QueryResponse

        if not isinstance(model, ModelDep):
            raise TypeError(
                f"model must be a ModelDep, got {type(model).__name__}. "
                "Declare it as a class attribute: identity_model = Pipeline.model_dep('...')"
            )

        from ..modeling import LLMClient

        client = LLMClient(model.registry_name, **kwargs)
        try:
            text = client.query(messages)
            return QueryResponse(text=text)
        except Exception as e:
            return QueryResponse(text="", error=str(e))

    def run_task(
        self,
        task_method: Callable,
        records: list[dict],
        workers: Optional[int] = None,
    ) -> list[Any]:
        """Run a task method across records in parallel.

        The task method should be a generator that yields model_request() calls
        and returns a result.

        Args:
            task_method: Generator method (self.some_method)
            records: List of record dicts to process
            workers: Number of parallel workers (defaults to self.workers, then 50)

        Returns:
            List of results from each record

        Example:
            def generate_qa(self, record):
                question = yield model_request([...], model=self.judge_model)
                response = yield model_request([...], model=self.identity_model)
                return TrainingSample(...)

            results = self.run_task(self.generate_qa, records=facts)
        """
        if not records:
            return []

        # Apply limit if set (for testing with --limit)
        if self._limit is not None and len(records) > self._limit:
            records = records[: self._limit]

        # Determine worker count with fallback chain:
        # 1. Explicit argument
        # 2. Pipeline.workers class attribute
        # 3. workers_model's pipeline_workers from registry
        # 4. Default 50
        if workers is None:
            workers = self._get_workers()

        # Write records to temp file for dispatcher
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as input_file:
            for i, record in enumerate(records):
                # Ensure each record has an id
                if "id" not in record:
                    record = {"id": f"record-{i}", **record}
                input_file.write(json.dumps(record) + "\n")
            input_path = Path(input_file.name)

        output_path = input_path.with_suffix(".output.jsonl")

        try:
            # Create a dynamic task class that wraps the method
            task_class = self._create_task_class(task_method)

            # Run via dispatcher
            from .runner import run_pipeline

            run_pipeline(
                task_class=task_class,
                input_file=input_path,
                output_file=output_path,
                num_workers=workers,
            )

            # Read results, filtering out errors
            results = []
            error_count = 0
            with open(output_path) as f:
                for line in f:
                    result = json.loads(line)
                    if "__ERROR__" in result:
                        error_count += 1
                        logger.debug(f"Skipped record: {result['__ERROR__']}")
                    else:
                        results.append(result)

            if error_count > 0:
                logger.warning(f"Filtered out {error_count} failed/skipped records")

            return results

        finally:
            # Clean up temp files
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def _create_task_class(self, task_method: Callable) -> type:
        """Create a TrackedTask subclass that wraps a pipeline method."""
        from .tasks import TrackedTask

        pipeline = self

        class DynamicTask(TrackedTask):
            name = f"{pipeline.name}-task"
            workers = 50

            def process_record(self):
                # Call the bound method with record data
                gen = task_method(self.data)

                # Drive the generator
                response = None
                while True:
                    try:
                        request = gen.send(response)
                        response = yield request
                    except StopIteration as e:
                        return e.value

        return DynamicTask

    def execute(
        self,
        limit: Optional[int] = None,
        output_file: Optional[Path] = None,
        annotated: Optional[bool] = None,
    ) -> list[Any]:
        """Run the pipeline with setup and teardown.

        This is the main entry point for running a pipeline.
        Handles manifest writing and staleness tracking.

        Args:
            limit: If set, marks the run as partial (for testing).
                   Manifests from partial runs show as stale.
            output_file: Override the output file path. If set,
                   manifest is written alongside this file.
            annotated: If True, output AnnotatedTrainingSample with full
                   provenance. If False, output minimal TrainingSample format
                   (just id and messages) suitable for training. If None,
                   uses the class-level `annotated` attribute (default True).
        """
        # Use class default if not specified
        if annotated is None:
            annotated = self.annotated
        self._started_at = datetime.now()
        self._partial = limit is not None
        self._limit = limit  # Make limit available to run_task()

        # Determine output paths
        if output_file is not None:
            actual_output = Path(output_file)
            actual_manifest = actual_output.with_suffix(".manifest.json")
        else:
            actual_output = self.get_output_file()
            actual_manifest = self.get_manifest_file()

        # Ensure output directory exists
        actual_output.parent.mkdir(parents=True, exist_ok=True)

        try:
            results = self.run()
            self._completed_at = datetime.now()

            # Apply limit if specified (for testing)
            if limit is not None and len(results) > limit:
                results = results[:limit]

            # Write results to output file
            with open(actual_output, "w") as f:
                for result in results:
                    if hasattr(result, "to_dict"):
                        output = result.to_dict()
                    else:
                        output = result

                    # Strip to minimal training format if requested
                    if not annotated and isinstance(output, dict):
                        output = {"id": output["id"], "messages": output["messages"]}

                    f.write(json.dumps(output) + "\n")

            # Write manifest
            self._write_manifest(len(results), actual_manifest)

            logger.info(f"Pipeline complete: {len(results)} results -> {actual_output}")
            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def _write_manifest(
        self, record_count: int, manifest_file: Optional[Path] = None
    ) -> None:
        """Write manifest file for staleness tracking."""
        deps_manifest = {}
        for name, dep in get_all_deps(self.__class__).items():
            deps_manifest[name] = dep.to_manifest_entry()

        # Get code hash
        code_hash = self._get_code_hash()

        manifest = {
            "pipeline": self.name,
            "code_hash": code_hash,
            "deps": deps_manifest,
            "record_count": record_count,
            "partial": self._partial,
        }

        if manifest_file is None:
            manifest_file = self.get_manifest_file()
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

    def _get_code_hash(self) -> Optional[str]:
        """Get hash of the pipeline source file."""
        try:
            source_file = inspect.getfile(self.__class__)
            content = Path(source_file).read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
        except Exception:
            return None

    @classmethod
    def check_staleness(cls) -> dict[str, Any]:
        """Check if pipeline output is stale.

        Returns dict with:
            - stale: bool
            - reasons: list of strings explaining what changed
            - partial: bool (if manifest indicates a partial run)
            - record_count: int (number of records in last run)
        """
        manifest_file = cls.get_manifest_file()

        if not manifest_file.exists():
            return {"stale": True, "reasons": ["No manifest (never run)"]}

        # Check output file exists
        output_file = cls.get_output_file()
        if not output_file.exists():
            return {"stale": True, "reasons": ["Output file missing"]}

        with open(manifest_file) as f:
            manifest = json.load(f)

        reasons = []
        partial = manifest.get("partial", False)
        record_count = manifest.get("record_count", 0)

        # Check if this was a partial run (--limit)
        if partial:
            reasons.append(
                f"Partial run ({record_count} samples, run without --limit for full data)"
            )

        # Check code hash
        try:
            source_file = inspect.getfile(cls)
            current_hash = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()[
                :16
            ]
            if manifest.get("code_hash") != current_hash:
                reasons.append("Pipeline code changed")
        except Exception:
            pass

        # Check each dep
        for name, dep in get_all_deps(cls).items():
            old_entry = manifest.get("deps", {}).get(name, {})
            new_entry = dep.to_manifest_entry()

            if isinstance(dep, ModelDep):
                if old_entry.get("sysprompt_hash") != new_entry.get("sysprompt_hash"):
                    reasons.append(f"Model '{name}' sysprompt changed")
            elif isinstance(dep, FileDep):
                if old_entry.get("content_hash") != new_entry.get("content_hash"):
                    reasons.append(f"File '{name}' content changed")

        return {
            "stale": bool(reasons),
            "reasons": reasons,
            "partial": partial,
            "record_count": record_count,
        }
