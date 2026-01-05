"""Results store - JSONL-based storage with query helpers.

The canonical data lives in a JSONL file (git-tracked).
Provides load/save/query operations.
"""

import fcntl
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from .schema import EvalResult


def generate_id() -> str:
    """Generate a unique eval result ID."""
    now = datetime.now()
    suffix = f"{random.randint(0, 0xffff):04x}"
    return f"batch-{now.strftime('%Y%m%d-%H%M%S')}-{suffix}"


class ResultsStore:
    """JSONL-based results store.

    Usage:
        store = ResultsStore(Path("results/evals.jsonl"))

        # Add a result
        store.add(result)

        # Query
        for r in store.list(model="e037-final"):
            print(r.results.score)

        # Load specific result
        result = store.get("batch-20260105-143022-a7b3")
    """

    def __init__(self, path: Path):
        """Initialize store with path to JSONL file."""
        self.path = Path(path)

    def _ensure_exists(self) -> None:
        """Ensure the store file exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def add(self, result: EvalResult) -> None:
        """Add a result to the store.

        Uses file locking for concurrent safety.
        """
        self._ensure_exists()

        with open(self.path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(result.model_dump_json() + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _iter_raw(self) -> Iterator[tuple[dict, int]]:
        """Iterate over raw records with line numbers."""
        if not self.path.exists():
            return

        with open(self.path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line), lineno
                except json.JSONDecodeError as e:
                    # Log but don't crash - lenient on read
                    print(f"Warning: Invalid JSON at line {lineno}: {e}")

    def _parse_result(self, data: dict) -> EvalResult | dict:
        """Parse a result, returning raw dict if validation fails."""
        try:
            return EvalResult.model_validate(data)
        except ValidationError:
            # Lenient on read - return raw dict for unknown versions
            return data

    def iter_all(self, include_raw: bool = False) -> Iterator[EvalResult | dict]:
        """Iterate over all results.

        Args:
            include_raw: If True, include unparseable records as raw dicts
        """
        for data, _ in self._iter_raw():
            result = self._parse_result(data)
            if isinstance(result, EvalResult) or include_raw:
                yield result

    def list(
        self,
        model: str | None = None,
        eval_name: str | None = None,
        training_run: str | None = None,
        include_all: bool = False,
    ) -> list[EvalResult]:
        """Query results with filters.

        Args:
            model: Filter by model alias
            eval_name: Filter by eval name
            training_run: Filter by training run (for trained models)
            include_all: Include partial and archived evals (default: exclude both)
        """
        results = []

        for r in self.iter_all():
            if not isinstance(r, EvalResult):
                continue

            # Filter by completeness and archival status
            if not include_all:
                if not r.eval.complete:
                    continue
                if r.archived:
                    continue

            # Filter by model alias
            if model and r.model.alias != model:
                continue

            # Filter by eval name
            if eval_name and r.eval.name != eval_name:
                continue

            # Filter by training run
            if training_run:
                if not hasattr(r.model, "training_run"):
                    continue
                if r.model.training_run != training_run:
                    continue

            results.append(r)

        return results

    def get(self, result_id: str) -> EvalResult | None:
        """Get a specific result by ID."""
        for r in self.iter_all():
            if isinstance(r, EvalResult) and r.id == result_id:
                return r
        return None

    def count(self, include_all: bool = False) -> int:
        """Count results."""
        return len(self.list(include_all=include_all))

    def update(self, result_id: str, **updates) -> bool:
        """Update a result in place.

        Args:
            result_id: ID of the result to update
            **updates: Fields to update (e.g., archived=True, archived_note="reason")

        Returns:
            True if result was found and updated, False otherwise
        """
        if not self.path.exists():
            return False

        # Read all records
        records = []
        found = False
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("id") == result_id:
                        data.update(updates)
                        found = True
                    records.append(data)
                except json.JSONDecodeError:
                    records.append(line)  # Keep malformed lines as-is

        if not found:
            return False

        # Write back
        with open(self.path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                for record in records:
                    if isinstance(record, str):
                        f.write(record + "\n")
                    else:
                        f.write(json.dumps(record) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return True
