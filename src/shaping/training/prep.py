"""Training data preparation.

Combines pipeline outputs into prepared training datasets with optional
category-based balancing.

Recipe format (training/data/default.yaml):

    categories:
      all:
        pipelines: all    # special: all registered pipelines

    mode: simple
    shuffle_seed: 42

Or with balancing:

    categories:
      identity:
        pipelines:
          - identity-augmentation
          - voice-demonstrations
        files:
          - external/curated-samples.jsonl

      general:
        pipelines:
          - wildchat-training

    mode: weighted
    weights:
      identity: 1
      general: 1    # 1:1 ratio, caps larger category to match smaller

    shuffle_seed: 42
"""

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import yaml


@dataclass
class CategorySource:
    """A category within a dataset recipe."""

    pipelines: list[str] | Literal["all"] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Normalize empty pipelines
        if self.pipelines is None:
            self.pipelines = []

    def has_sources(self) -> bool:
        """Check if category has any sources defined."""
        if self.pipelines == "all":
            return True
        return bool(self.pipelines) or bool(self.files)


@dataclass
class DatasetRecipe:
    """A recipe for preparing training data."""

    name: str
    categories: dict[str, CategorySource]
    mode: Literal["simple", "weighted"] = "simple"
    weights: dict[str, float] = field(default_factory=dict)
    shuffle_seed: Optional[int] = None

    # Paths (set after loading)
    recipe_path: Optional[Path] = None
    data_dir: Optional[Path] = None

    @classmethod
    def load(cls, path: Path) -> "DatasetRecipe":
        """Load a recipe from a YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Recipe not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Recipe must be a YAML mapping, got {type(data)}")

        # Name from filename
        name = path.stem

        # Parse categories
        categories = {}
        for cat_name, cat_data in data.get("categories", {}).items():
            if isinstance(cat_data, dict):
                pipelines = cat_data.get("pipelines", [])
                files = cat_data.get("files", [])
                categories[cat_name] = CategorySource(pipelines=pipelines, files=files)
            else:
                raise ValueError(f"Invalid category '{cat_name}': expected mapping")

        if not categories:
            raise ValueError("Recipe must define at least one category")

        # Validate each category has sources
        for cat_name, cat in categories.items():
            if not cat.has_sources():
                raise ValueError(
                    f"Category '{cat_name}' has no sources. "
                    "Define 'pipelines' and/or 'files'."
                )

        mode = data.get("mode", "simple")
        if mode not in ("simple", "weighted"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'simple' or 'weighted'.")

        weights = data.get("weights", {})
        if mode == "weighted":
            # Validate weights match categories and are positive
            for cat_name in categories:
                if cat_name not in weights:
                    raise ValueError(
                        f"Weighted mode requires weight for category '{cat_name}'"
                    )
                if not isinstance(weights[cat_name], (int, float)):
                    raise ValueError(
                        f"Weight for '{cat_name}' must be a number, got {type(weights[cat_name]).__name__}"
                    )
                if weights[cat_name] <= 0:
                    raise ValueError(
                        f"Weight for '{cat_name}' must be positive, got {weights[cat_name]}"
                    )

        return cls(
            name=name,
            categories=categories,
            mode=mode,
            weights=weights,
            shuffle_seed=data.get("shuffle_seed"),
            recipe_path=path,
            data_dir=path.parent,
        )

    def get_output_file(self) -> Path:
        """Get output file path."""
        if self.data_dir is None:
            raise ValueError("data_dir not set")
        return self.data_dir / "prepared" / f"{self.name}.jsonl"

    def get_manifest_file(self) -> Path:
        """Get manifest file path."""
        return self.get_output_file().with_suffix(".manifest.json")


def _hash_file(path: Path) -> str:
    """Get SHA256 hash of file contents."""
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


def _count_lines(path: Path) -> int:
    """Count lines in a JSONL file."""
    with open(path) as f:
        return sum(1 for _ in f)


def _read_samples(path: Path) -> list[dict]:
    """Read samples from a JSONL file."""
    samples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def _discover_pipelines(project_dir: Path) -> dict[str, Path]:
    """Discover all pipelines and their output files.

    Returns dict mapping pipeline name to output file path.
    """
    from ..pipeline import Pipeline

    pipelines_dir = project_dir / "pipelines"
    if not pipelines_dir.exists():
        return {}

    import importlib.util
    import sys

    discovered = {}

    for py_file in pipelines_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        module_name = f"pipelines.{py_file.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Pipeline)
                    and attr is not Pipeline
                    and hasattr(attr, "name")
                    and attr.name
                ):
                    # Get output file path
                    output_file = attr.get_output_file()
                    discovered[attr.name] = output_file

        except Exception:
            pass  # Skip broken pipeline files

    return discovered


def _check_pipeline_staleness(pipeline_name: str, project_dir: Path) -> bool:
    """Check if a pipeline is stale.

    Returns True if stale, False if current.
    """
    from ..pipeline import Pipeline

    pipelines_dir = project_dir / "pipelines"
    if not pipelines_dir.exists():
        return True

    import importlib.util
    import sys

    for py_file in pipelines_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        module_name = f"pipelines.{py_file.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Pipeline)
                    and attr is not Pipeline
                    and hasattr(attr, "name")
                    and attr.name == pipeline_name
                ):
                    staleness = attr.check_staleness()
                    return staleness.get("stale", True)

        except Exception:
            return True

    return True


def resolve_sources(
    recipe: DatasetRecipe, project_dir: Path
) -> dict[str, dict[str, Any]]:
    """Resolve all sources for a recipe.

    Returns dict mapping category name to:
        - pipelines: dict of pipeline_name -> {path, hash, count, stale}
        - files: dict of file_path -> {path, hash, count}
        - total_samples: int
    """
    all_pipelines = _discover_pipelines(project_dir)

    result = {}

    for cat_name, cat in recipe.categories.items():
        cat_result: dict[str, Any] = {"pipelines": {}, "files": {}, "total_samples": 0}

        # Resolve pipelines
        if cat.pipelines == "all":
            pipeline_names = list(all_pipelines.keys())
        else:
            pipeline_names = cat.pipelines

        for pipe_name in pipeline_names:
            if pipe_name not in all_pipelines:
                raise ValueError(
                    f"Unknown pipeline '{pipe_name}' in category '{cat_name}'"
                )

            output_path = all_pipelines[pipe_name]
            if not output_path.exists():
                raise FileNotFoundError(
                    f"Pipeline '{pipe_name}' output not found: {output_path}\n"
                    f"Run: isf pipeline run {pipe_name}"
                )

            count = _count_lines(output_path)
            file_hash = _hash_file(output_path)
            stale = _check_pipeline_staleness(pipe_name, project_dir)

            cat_result["pipelines"][pipe_name] = {
                "path": output_path,
                "hash": file_hash,
                "count": count,
                "stale": stale,
            }
            cat_result["total_samples"] += count

        # Resolve files
        for file_path in cat.files:
            if recipe.data_dir is None:
                raise ValueError("data_dir not set on recipe")

            full_path = recipe.data_dir / file_path
            if not full_path.exists():
                raise FileNotFoundError(
                    f"File not found: {full_path} (in category '{cat_name}')"
                )

            count = _count_lines(full_path)
            file_hash = _hash_file(full_path)

            cat_result["files"][file_path] = {
                "path": full_path,
                "hash": file_hash,
                "count": count,
            }
            cat_result["total_samples"] += count

        result[cat_name] = cat_result

    return result


def compute_balancing(
    recipe: DatasetRecipe,
    sources: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Compute sample counts per category after balancing.

    Returns dict mapping category name to sample count to use.
    """
    if recipe.mode == "simple":
        # Simple mode: use all samples from all categories
        return {cat_name: info["total_samples"] for cat_name, info in sources.items()}

    # Weighted mode: balance by weights
    # Normalize weights
    total_weight = sum(recipe.weights.values())
    normalized = {k: v / total_weight for k, v in recipe.weights.items()}

    # Find the limiting category (smallest samples / weight ratio)
    # This determines how many total samples we can have
    ratios = {}
    for cat_name, info in sources.items():
        weight = normalized.get(cat_name, 0)
        if weight > 0:
            # samples / weight = effective capacity
            ratios[cat_name] = info["total_samples"] / weight

    # The minimum ratio determines the max samples per unit weight
    min_ratio = min(ratios.values())

    # Calculate samples per category
    result = {}
    for cat_name, weight in normalized.items():
        result[cat_name] = int(min_ratio * weight)

    return result


def prepare_dataset(
    recipe: DatasetRecipe,
    project_dir: Path,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare a training dataset from a recipe.

    Args:
        recipe: The dataset recipe
        project_dir: Project root directory
        dry_run: If True, just compute and return info without writing
        force: If True, rebuild even if current

    Returns:
        Dict with preparation results:
            - output_file: Path to output file (or would-be path for dry_run)
            - total_samples: Total sample count
            - by_category: Dict of category -> sample count
            - sources: Resolved source info
            - stale_pipelines: List of stale pipeline names
    """
    # Resolve all sources
    sources = resolve_sources(recipe, project_dir)

    # Check for stale pipelines
    stale_pipelines = []
    for cat_info in sources.values():
        for pipe_name, pipe_info in cat_info["pipelines"].items():
            if pipe_info["stale"]:
                stale_pipelines.append(pipe_name)

    # Compute balancing
    sample_counts = compute_balancing(recipe, sources)

    output_file = recipe.get_output_file()
    manifest_file = recipe.get_manifest_file()

    result = {
        "output_file": output_file,
        "total_samples": sum(sample_counts.values()),
        "by_category": sample_counts,
        "sources": sources,
        "stale_pipelines": list(set(stale_pipelines)),
    }

    if dry_run:
        return result

    # Check staleness (unless force)
    if not force and manifest_file.exists():
        staleness = check_staleness(recipe, project_dir)
        if not staleness["stale"]:
            result["skipped"] = True
            result["reason"] = "Dataset is current"
            return result

    # Collect samples from each category
    all_samples = []

    for cat_name, target_count in sample_counts.items():
        cat_samples = []

        # Read from pipelines
        for pipe_name, pipe_info in sources[cat_name]["pipelines"].items():
            pipe_samples = _read_samples(pipe_info["path"])
            # Strip to minimal training format
            for i, s in enumerate(pipe_samples):
                if "id" not in s or "messages" not in s:
                    missing = [k for k in ("id", "messages") if k not in s]
                    raise ValueError(
                        f"Sample {i} in pipeline '{pipe_name}' missing required fields: {missing}"
                    )
                cat_samples.append({"id": s["id"], "messages": s["messages"]})

        # Read from files
        for file_path, file_info in sources[cat_name]["files"].items():
            file_samples = _read_samples(file_info["path"])
            for i, s in enumerate(file_samples):
                if "id" not in s or "messages" not in s:
                    missing = [k for k in ("id", "messages") if k not in s]
                    raise ValueError(
                        f"Sample {i} in file '{file_path}' missing required fields: {missing}"
                    )
                cat_samples.append({"id": s["id"], "messages": s["messages"]})

        # Apply sampling if needed (for weighted mode)
        if len(cat_samples) > target_count:
            rng = random.Random(recipe.shuffle_seed)
            cat_samples = rng.sample(cat_samples, target_count)

        all_samples.extend(cat_samples)

    # Shuffle all samples
    if recipe.shuffle_seed is not None:
        rng = random.Random(recipe.shuffle_seed)
        rng.shuffle(all_samples)

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    # Write manifest
    manifest = _build_manifest(recipe, sources, sample_counts, len(all_samples))
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    result["written"] = True
    return result


def _build_manifest(
    recipe: DatasetRecipe,
    sources: dict[str, dict[str, Any]],
    sample_counts: dict[str, int],
    total_samples: int,
) -> dict[str, Any]:
    """Build manifest for a prepared dataset."""
    if recipe.recipe_path is None:
        raise ValueError("recipe_path not set")

    manifest: dict[str, Any] = {
        "recipe": recipe.recipe_path.name,
        "recipe_hash": _hash_file(recipe.recipe_path),
        "mode": recipe.mode,
    }

    if recipe.mode == "weighted":
        manifest["weights"] = recipe.weights

    # Categories with their sources
    manifest["categories"] = {}
    for cat_name, cat_info in sources.items():
        cat_manifest: dict[str, Any] = {
            "samples_used": sample_counts.get(cat_name, 0),
            "samples_available": cat_info["total_samples"],
        }

        if cat_info["pipelines"]:
            cat_manifest["pipelines"] = {}
            for pipe_name, pipe_info in cat_info["pipelines"].items():
                cat_manifest["pipelines"][pipe_name] = {
                    "hash": pipe_info["hash"],
                    "count": pipe_info["count"],
                    "pipeline_stale": pipe_info["stale"],
                }

        if cat_info["files"]:
            cat_manifest["files"] = {}
            for file_path, file_info in cat_info["files"].items():
                cat_manifest["files"][file_path] = {
                    "hash": file_info["hash"],
                    "count": file_info["count"],
                }

        manifest["categories"][cat_name] = cat_manifest

    manifest["output"] = {
        "total_samples": total_samples,
    }

    if recipe.shuffle_seed is not None:
        manifest["shuffle_seed"] = recipe.shuffle_seed

    return manifest


def check_staleness(recipe: DatasetRecipe, project_dir: Path) -> dict[str, Any]:
    """Check if a prepared dataset is stale.

    Returns dict with:
        - stale: bool - dataset needs re-prep
        - reasons: list of strings explaining what changed
        - stale_sources: list of pipeline names that are stale (informational)
    """
    manifest_file = recipe.get_manifest_file()
    output_file = recipe.get_output_file()

    if not manifest_file.exists():
        return {
            "stale": True,
            "reasons": ["No manifest (never prepared)"],
            "stale_sources": [],
        }

    if not output_file.exists():
        return {"stale": True, "reasons": ["Output file missing"], "stale_sources": []}

    with open(manifest_file) as f:
        manifest = json.load(f)

    reasons = []
    stale_sources = []

    # Check recipe changed
    if recipe.recipe_path is not None:
        current_hash = _hash_file(recipe.recipe_path)
        if manifest.get("recipe_hash") != current_hash:
            reasons.append("Recipe changed")

    # Resolve current sources
    try:
        sources = resolve_sources(recipe, project_dir)
    except (FileNotFoundError, ValueError) as e:
        return {"stale": True, "reasons": [str(e)], "stale_sources": []}

    # Check each source - handle both new and old manifest formats
    manifest_categories = manifest.get("categories", {})

    for cat_name, cat_info in sources.items():
        old_cat = manifest_categories.get(cat_name, {})

        for pipe_name, pipe_info in cat_info["pipelines"].items():
            # Try new format first, fall back to old flat format
            old_info = old_cat.get("pipelines", {}).get(pipe_name)
            if old_info is None:
                old_info = (
                    manifest.get("sources", {}).get("pipelines", {}).get(pipe_name)
                )

            if old_info is None:
                reasons.append(f"New pipeline: {pipe_name}")
            elif old_info.get("hash") != pipe_info["hash"]:
                reasons.append(f"Pipeline '{pipe_name}' output changed")

            # Track stale sources separately (not a reason to re-prep)
            if pipe_info["stale"]:
                stale_sources.append(pipe_name)

        for file_path, file_info in cat_info["files"].items():
            old_info = old_cat.get("files", {}).get(file_path)
            if old_info is None:
                old_info = manifest.get("sources", {}).get("files", {}).get(file_path)

            if old_info is None:
                reasons.append(f"New file: {file_path}")
            elif old_info.get("hash") != file_info["hash"]:
                reasons.append(f"File '{file_path}' changed")

    # Check for removed sources
    all_old_pipelines = set()
    all_old_files = set()
    for old_cat in manifest_categories.values():
        all_old_pipelines.update(old_cat.get("pipelines", {}).keys())
        all_old_files.update(old_cat.get("files", {}).keys())
    # Also check old flat format
    all_old_pipelines.update(manifest.get("sources", {}).get("pipelines", {}).keys())
    all_old_files.update(manifest.get("sources", {}).get("files", {}).keys())

    all_current_pipelines = set()
    all_current_files = set()
    for cat_info in sources.values():
        all_current_pipelines.update(cat_info["pipelines"].keys())
        all_current_files.update(cat_info["files"].keys())

    for pipe_name in all_old_pipelines - all_current_pipelines:
        reasons.append(f"Removed pipeline: {pipe_name}")

    for file_path in all_old_files - all_current_files:
        reasons.append(f"Removed file: {file_path}")

    return {"stale": bool(reasons), "reasons": reasons, "stale_sources": stale_sources}


def list_recipes(data_dir: Path) -> list[Path]:
    """List all recipe files in a data directory."""
    if not data_dir.exists():
        return []

    recipes = []
    for path in data_dir.glob("*.yaml"):
        # Skip non-recipe files (could add more checks)
        recipes.append(path)
    for path in data_dir.glob("*.yml"):
        recipes.append(path)

    return sorted(recipes)
