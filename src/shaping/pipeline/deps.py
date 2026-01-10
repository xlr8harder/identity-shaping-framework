"""Dependency declarations for pipelines.

Provides descriptor classes for declaring pipeline dependencies:
- ModelDep: Model from registry (tracks sysprompt hash for staleness)
- FileDep: Local file (tracks content hash for staleness)

Dependencies are declared as class attributes on Pipeline subclasses:

    class MyPipeline(Pipeline):
        identity_model = Pipeline.model_dep("cubsfan-release-full")
        narrative_doc = Pipeline.file_dep("identity/NARRATIVE.md")

        def run(self):
            content = self.narrative_doc.read()
            response = self.query(model=self.identity_model, messages=[...])

The framework uses these declarations for:
1. Enforcing that model_request() only uses declared models
2. Computing staleness by comparing current hashes to manifest
"""

import hashlib
from pathlib import Path
from typing import Any, Optional


class ModelDep:
    """Dependency on a registry model.

    Tracks the model's registry name for use in model_request() and
    computes sysprompt hash for staleness detection.

    Usage:
        class MyPipeline(Pipeline):
            identity_model = ModelDep("cubsfan-release-full")

            def run(self):
                response = self.query(model=self.identity_model, messages=[...])
    """

    def __init__(self, registry_name: str):
        """Create a model dependency.

        Args:
            registry_name: Model shortname in the registry (e.g., "cubsfan-release-full")
        """
        self.registry_name = registry_name
        self._attr_name: Optional[str] = None

    def __set_name__(self, owner: type, name: str) -> None:
        """Called when the descriptor is assigned to a class attribute."""
        self._attr_name = name
        # Register this dep on the class
        if not hasattr(owner, "_model_deps"):
            owner._model_deps = {}
        owner._model_deps[name] = self

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> "ModelDep":
        """Return self - the dep object is used directly."""
        return self

    def __repr__(self) -> str:
        return f"ModelDep({self.registry_name!r})"

    def get_sysprompt_hash(self) -> Optional[str]:
        """Get hash of the model's sysprompt from registry.

        Returns None if model not found or has no sysprompt.
        """
        try:
            from mq import store as mq_store

            config = mq_store.get_model(self.registry_name)
            if config and config.get("sysprompt"):
                sysprompt = config["sysprompt"]
                return hashlib.sha256(sysprompt.encode()).hexdigest()[:16]
        except Exception:
            pass
        return None

    def to_manifest_entry(self) -> dict[str, Any]:
        """Create manifest entry for staleness tracking."""
        return {
            "type": "model",
            "registry": self.registry_name,
            "sysprompt_hash": self.get_sysprompt_hash(),
        }


class FileDep:
    """Dependency on a local file.

    Tracks file path and computes content hash for staleness detection.
    Provides read() method for accessing file contents.

    Usage:
        class MyPipeline(Pipeline):
            narrative_doc = FileDep("identity/NARRATIVE.md")

            def run(self):
                content = self.narrative_doc.read()
    """

    def __init__(self, path: str | Path):
        """Create a file dependency.

        Args:
            path: Path to the file (relative to project root or absolute)
        """
        self.path = Path(path)
        self._attr_name: Optional[str] = None

    def __set_name__(self, owner: type, name: str) -> None:
        """Called when the descriptor is assigned to a class attribute."""
        self._attr_name = name
        # Register this dep on the class
        if not hasattr(owner, "_file_deps"):
            owner._file_deps = {}
        owner._file_deps[name] = self

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> "FileDep":
        """Return self - the dep object is used directly."""
        return self

    def __repr__(self) -> str:
        return f"FileDep({str(self.path)!r})"

    def read(self) -> str:
        """Read and return file contents."""
        return self.path.read_text()

    def exists(self) -> bool:
        """Check if file exists."""
        return self.path.exists()

    def get_content_hash(self) -> Optional[str]:
        """Get hash of file contents.

        Returns None if file doesn't exist.
        """
        if not self.path.exists():
            return None
        content = self.path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]

    def to_manifest_entry(self) -> dict[str, Any]:
        """Create manifest entry for staleness tracking."""
        return {
            "type": "file",
            "path": str(self.path),
            "content_hash": self.get_content_hash(),
        }


def get_all_deps(cls: type) -> dict[str, ModelDep | FileDep]:
    """Get all dependencies declared on a class.

    Returns dict mapping attribute name to dep instance.
    """
    deps = {}
    if hasattr(cls, "_model_deps"):
        deps.update(cls._model_deps)
    if hasattr(cls, "_file_deps"):
        deps.update(cls._file_deps)
    return deps


def get_model_deps(cls: type) -> dict[str, ModelDep]:
    """Get all ModelDep dependencies declared on a class."""
    return getattr(cls, "_model_deps", {})


def get_file_deps(cls: type) -> dict[str, FileDep]:
    """Get all FileDep dependencies declared on a class."""
    return getattr(cls, "_file_deps", {})
