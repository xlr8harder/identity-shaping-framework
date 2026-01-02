"""Prompt building and versioning for identity-shaping projects.

Handles:
- Building dev sysprompts from identity docs + templates
- Releasing dev to versioned snapshots
- Building mq registry from all versions
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, TemplateNotFound


@dataclass
class IdentityDoc:
    """An identity document available to templates."""
    name: str          # filename without extension (e.g., "IDENTITY")
    path: Path         # full path
    content: str       # file contents


@dataclass
class PromptVariant:
    """Configuration for a prompt variant (e.g., full, medium)."""
    name: str
    template: str
    context: dict = field(default_factory=dict)


@dataclass
class PromptsConfig:
    """Configuration for prompt building."""
    project_name: str
    identity_dir: Path
    templates_dir: Path
    versions_dir: Path
    registry_path: Path

    # Model config for registry
    identity_provider: str
    identity_model: str
    identity_temperature: float
    model_prefix: str
    release_version: str

    # Prompt variants to build
    variants: list[PromptVariant]

    @classmethod
    def from_project(cls, project_dir: Path) -> "PromptsConfig":
        """Load prompts config from project directory."""
        project_dir = project_dir.resolve()

        # Load isf.yaml
        isf_yaml = project_dir / "isf.yaml"
        if not isf_yaml.exists():
            raise FileNotFoundError(f"isf.yaml not found in {project_dir}")

        with open(isf_yaml) as f:
            config = yaml.safe_load(f)

        # Extract identity config
        identity_config = config.get("models", {}).get("identity", {})
        model_prefix = identity_config.get("prefix", "identity")
        release_version = identity_config.get("release_version", "dev")

        # Prompts config (with defaults)
        prompts_config = config.get("prompts", {})

        # Identity dir
        identity_dir = project_dir / prompts_config.get("identity_dir", "identity")

        # Templates dir (defaults to identity/templates)
        templates_dir_str = prompts_config.get("templates_dir", "identity/templates")
        templates_dir = project_dir / templates_dir_str

        # Versions dir
        versions_dir = project_dir / prompts_config.get("versions_dir", "identity/versions")

        # Registry path
        registry_path = project_dir / prompts_config.get("registry_path", "config/registry.json")

        # Base model config - check templates dir or use defaults
        base_model_path = project_dir / "config" / "templates" / "identity-base.json"
        if base_model_path.exists():
            with open(base_model_path) as f:
                base_model = json.load(f)
            identity_provider = base_model.get("provider", "openrouter")
            identity_model = base_model.get("model", "anthropic/claude-sonnet-4")
            identity_temperature = base_model.get("params", {}).get("temperature", 0.7)
        else:
            identity_provider = prompts_config.get("provider", "openrouter")
            identity_model = prompts_config.get("model", "anthropic/claude-sonnet-4")
            identity_temperature = prompts_config.get("temperature", 0.7)

        # Variants to build
        variants_config = prompts_config.get("variants", {
            "full": {"template": "full.txt.j2"}
        })
        variants = [
            PromptVariant(
                name=name,
                template=v.get("template", f"{name}.txt.j2"),
                context=v.get("context", {}),
            )
            for name, v in variants_config.items()
        ]

        # Project name
        project_name = config.get("project", {}).get("name", model_prefix.replace("-", " ").title())

        return cls(
            project_name=project_name,
            identity_dir=identity_dir,
            templates_dir=templates_dir,
            versions_dir=versions_dir,
            registry_path=registry_path,
            identity_provider=identity_provider,
            identity_model=identity_model,
            identity_temperature=identity_temperature,
            model_prefix=model_prefix,
            release_version=release_version,
            variants=variants,
        )


def load_identity_docs(identity_dir: Path) -> list[IdentityDoc]:
    """Load all markdown files from identity directory."""
    docs = []
    for path in sorted(identity_dir.glob("*.md")):
        docs.append(IdentityDoc(
            name=path.stem,
            path=path,
            content=path.read_text(),
        ))
    return docs


def build_sysprompts(config: PromptsConfig, version: str = "dev") -> dict[str, Path]:
    """Build sysprompts for a version from templates.

    Returns dict of variant_name -> output_path.
    """
    # Load identity docs
    docs = load_identity_docs(config.identity_dir)
    docs_by_name = {doc.name: doc for doc in docs}

    # Set up Jinja environment
    env = Environment(
        loader=FileSystemLoader(config.templates_dir),
        keep_trailing_newline=True,
    )

    # Add identity docs to globals for {% include %}
    # Also make them available as variables
    template_context = {
        "project": {"name": config.project_name},
        "identity_docs": docs,
        "version": version,
    }
    # Add each doc by name
    for doc in docs:
        template_context[doc.name] = doc.content

    # Output directory
    output_dir = config.versions_dir / version / "sysprompts"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for variant in config.variants:
        try:
            template = env.get_template(variant.template)
        except TemplateNotFound:
            raise FileNotFoundError(
                f"Template not found: {variant.template} in {config.templates_dir}"
            )

        # Merge variant context with global context
        context = {**template_context, **variant.context, "variant": variant.name}

        # Render
        content = template.render(**context)

        # Write
        output_path = output_dir / f"{variant.name}.txt"
        output_path.write_text(content)
        results[variant.name] = output_path

    return results


def build_registry(config: PromptsConfig) -> Path:
    """Build mq registry.json from all versions.

    Scans versions_dir for version directories with sysprompts/,
    creates model entries for each.
    """
    models: dict[str, Any] = {}

    # Add judge model if config exists
    judge_config_path = config.registry_path.parent / "templates" / "judge.json"
    if judge_config_path.exists():
        with open(judge_config_path) as f:
            judge = json.load(f)
        models["gpt-4o-mini"] = {
            "provider": judge.get("provider", "openrouter"),
            "model": judge.get("model", "openai/gpt-4o-mini"),
            "params": judge.get("params", {"temperature": 0.7}),
            "sysprompt": None,
        }

    # Scan versions
    for version_dir in sorted(config.versions_dir.iterdir()):
        if not version_dir.is_dir():
            continue

        sysprompts_dir = version_dir / "sysprompts"
        if not sysprompts_dir.exists():
            continue

        version = version_dir.name

        # Add entry for each sysprompt variant
        for sysprompt_path in sorted(sysprompts_dir.glob("*.txt")):
            variant = sysprompt_path.stem  # full, medium, etc.
            model_name = f"{config.model_prefix}-{version}-{variant}"

            models[model_name] = {
                "provider": config.identity_provider,
                "model": config.identity_model,
                "params": {"temperature": config.identity_temperature},
                "sysprompt": sysprompt_path.read_text(),
            }

    # Write registry
    registry = {
        "version": 1,
        "models": models,
    }

    config.registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    return config.registry_path


def build(project_dir: Path) -> tuple[dict[str, Path], Path]:
    """Build dev sysprompts and registry.

    Returns (sysprompts_dict, registry_path).
    """
    config = PromptsConfig.from_project(project_dir)

    # Build dev sysprompts
    sysprompts = build_sysprompts(config, "dev")

    # Rebuild registry
    registry_path = build_registry(config)

    return sysprompts, registry_path


def release(project_dir: Path, version: str) -> Path:
    """Release current dev as a new version.

    1. Build dev (ensure it's current)
    2. Copy dev to versions/{version}/
    3. Update release_version in isf.yaml

    Returns path to new version directory.
    """
    config = PromptsConfig.from_project(project_dir)

    # Validate version name
    if version == "dev":
        raise ValueError("Cannot release as 'dev' - that's the development version")
    if not version.startswith("v"):
        raise ValueError(f"Version should start with 'v' (e.g., v0.1), got: {version}")

    # Check version doesn't already exist
    version_dir = config.versions_dir / version
    if version_dir.exists():
        raise ValueError(f"Version {version} already exists at {version_dir}")

    # Build dev first
    build_sysprompts(config, "dev")

    # Copy dev to new version
    dev_dir = config.versions_dir / "dev"
    shutil.copytree(dev_dir / "sysprompts", version_dir / "sysprompts")

    # Update isf.yaml
    isf_yaml = project_dir / "isf.yaml"
    with open(isf_yaml) as f:
        isf_config = yaml.safe_load(f)

    isf_config["models"]["identity"]["release_version"] = version

    with open(isf_yaml, "w") as f:
        yaml.dump(isf_config, f, default_flow_style=False, sort_keys=False)

    # Rebuild registry to include new version
    build_registry(config)

    return version_dir


def list_versions(project_dir: Path) -> list[dict[str, Any]]:
    """List all versions with their status.

    Returns list of dicts with:
    - name: version name (dev, v0.1, etc.)
    - variants: list of variant names
    - is_release: whether this is the current release version
    """
    config = PromptsConfig.from_project(project_dir)

    versions = []
    for version_dir in sorted(config.versions_dir.iterdir()):
        if not version_dir.is_dir():
            continue

        sysprompts_dir = version_dir / "sysprompts"
        if not sysprompts_dir.exists():
            continue

        version_name = version_dir.name
        variants = [p.stem for p in sorted(sysprompts_dir.glob("*.txt"))]

        versions.append({
            "name": version_name,
            "variants": variants,
            "is_release": version_name == config.release_version,
        })

    return versions
