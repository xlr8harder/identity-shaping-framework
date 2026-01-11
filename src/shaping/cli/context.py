"""Shared CLI context and utilities."""

from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from mq import store as mq_store


def find_project_root(start: Path) -> Optional[Path]:
    """Find project root by looking for isf.yaml."""
    current = start.resolve()
    while current != current.parent:
        if (current / "isf.yaml").exists():
            return current
        current = current.parent
    return None


class ProjectContext:
    """Context object holding project configuration."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir.resolve()
        self.registry_path = self.project_dir / "registry.json"
        self.results_path = self.project_dir / "results" / "index.jsonl"
        self.env_path = self.project_dir / ".env"

    def load_env(self):
        """Load environment variables from .env file."""
        if self.env_path.exists():
            load_dotenv(self.env_path)
            return True
        return False

    def setup_mq(self):
        """Configure mq to use this project's registry."""
        if self.registry_path.exists():
            mq_store.set_config_path_override(self.registry_path)
            mq_store.load_config()
            return True
        return False


pass_context = click.make_pass_decorator(ProjectContext)
