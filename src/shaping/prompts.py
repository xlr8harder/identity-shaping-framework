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

    name: str  # filename without extension (e.g., "IDENTITY")
    path: Path  # full path
    content: str  # file contents


@dataclass
class PromptVariant:
    """Configuration for a prompt variant (e.g., full, medium)."""

    name: str
    template: str
    context: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for a plain model."""

    name: str
    provider: str
    model: str
    temperature: float
    sysprompt: str | None = None


@dataclass
class PromptsConfig:
    """Configuration for prompt building."""

    project_dir: Path
    project_name: str
    identity_dir: Path
    templates_dir: Path
    versions_dir: Path
    registry_path: Path

    # Identity model config
    identity_provider: str
    identity_model: str
    identity_temperature: float
    model_prefix: str
    release_version: str

    # Plain models (non-versioned)
    plain_models: list[ModelConfig]

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

        # Extract identity config (special versioned models)
        identity_config = config.get("identity", {})
        model_prefix = identity_config.get("prefix", "identity")
        release_version = identity_config.get("release_version", "dev")
        identity_provider = identity_config.get("provider", "openrouter")
        identity_model = identity_config.get("model", "anthropic/claude-sonnet-4")
        identity_temperature = identity_config.get("temperature", 0.7)

        # Extract plain models
        plain_models = []
        for name, model_cfg in config.get("models", {}).items():
            # Handle sysprompt - can be inline or from file
            sysprompt = model_cfg.get("sysprompt")
            if not sysprompt and model_cfg.get("sysprompt_file"):
                sysprompt_path = project_dir / model_cfg["sysprompt_file"]
                if sysprompt_path.exists():
                    sysprompt = sysprompt_path.read_text()

            plain_models.append(
                ModelConfig(
                    name=name,
                    provider=model_cfg.get("provider", "openrouter"),
                    model=model_cfg.get("model", ""),
                    temperature=model_cfg.get("temperature", 0.7),
                    sysprompt=sysprompt,
                )
            )

        # Prompts config (with defaults)
        prompts_config = config.get("prompts", {})

        # Identity dir
        identity_dir = project_dir / prompts_config.get("identity_dir", "identity")

        # Templates dir (defaults to identity/templates)
        templates_dir_str = prompts_config.get("templates_dir", "identity/templates")
        templates_dir = project_dir / templates_dir_str

        # Versions dir
        versions_dir = project_dir / prompts_config.get(
            "versions_dir", "identity/versions"
        )

        # Registry path
        registry_path = project_dir / prompts_config.get(
            "registry_path", "registry.json"
        )

        # Variants to build - can be list or dict format
        variants_raw = identity_config.get("variants", ["full"])
        if isinstance(variants_raw, list):
            # Simple list format: ["full", "medium"]
            variants = [
                PromptVariant(name=v, template=f"{v}.txt.j2") for v in variants_raw
            ]
        else:
            # Dict format with template overrides
            variants = [
                PromptVariant(
                    name=name,
                    template=v.get("template", f"{name}.txt.j2"),
                    context=v.get("context", {}),
                )
                for name, v in variants_raw.items()
            ]

        # Project name
        project_name = config.get("project", {}).get(
            "name", model_prefix.replace("-", " ").title()
        )

        return cls(
            project_dir=project_dir,
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
            plain_models=plain_models,
            variants=variants,
        )


def load_identity_docs(identity_dir: Path) -> list[IdentityDoc]:
    """Load all markdown files from identity directory, including subdirectories.

    Names are relative paths without extension:
    - identity/IDENTITY.md -> "IDENTITY"
    - identity/artifacts/foo.md -> "artifacts/foo"
    """
    docs = []
    for path in sorted(identity_dir.rglob("*.md")):
        rel_path = path.relative_to(identity_dir)
        # Use relative path as name (without .md extension)
        name = str(rel_path.with_suffix(""))
        docs.append(
            IdentityDoc(
                name=name,
                path=path,
                content=path.read_text(),
            )
        )
    return docs


def build_sysprompts(config: PromptsConfig, version: str = "dev") -> dict[str, Path]:
    """Build sysprompts for a version from templates.

    Returns dict of variant_name -> output_path.
    """
    # Load identity docs
    docs = load_identity_docs(config.identity_dir)

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
        "docs": {},  # Dict for path-based access: docs["artifacts/foo"]
    }
    # Add each doc by name
    for doc in docs:
        # All docs accessible via path: docs["artifacts/foo"] or docs["IDENTITY"]
        template_context["docs"][doc.name] = doc.content

        # Top-level docs also accessible directly as uppercase variables (backward compat)
        # e.g., IDENTITY, NARRATIVE, SEED
        if "/" not in doc.name:
            template_context[doc.name.upper()] = doc.content

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


def discover_checkpoints(project_dir: Path) -> list[dict[str, Any]]:
    """Discover trained checkpoints from training/logs.

    Scans training/logs/*/checkpoints.jsonl and train-config.json
    to find all trained checkpoints.

    Returns list of dicts with: name, base_model, renderer, checkpoint_path
    """
    logs_dir = project_dir / "training" / "logs"
    if not logs_dir.exists():
        return []

    checkpoints = []

    for exp_dir in sorted(logs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        checkpoints_file = exp_dir / "checkpoints.jsonl"
        train_config_file = exp_dir / "train-config.json"
        tinker_config_file = exp_dir / "config.json"

        if not checkpoints_file.exists() or not train_config_file.exists():
            continue

        # Load train config for base_model
        with open(train_config_file) as f:
            train_config = json.load(f)
        base_model = train_config.get("base_model")
        if not base_model:
            continue

        # Load tinker config for renderer (if exists)
        renderer = None
        if tinker_config_file.exists():
            with open(tinker_config_file) as f:
                tinker_config = json.load(f)
            renderer = (
                tinker_config.get("dataset_builder", {})
                .get("common_config", {})
                .get("renderer_name")
            )

        # Load checkpoints
        with open(checkpoints_file) as f:
            for line in f:
                cp = json.loads(line)
                # Use just experiment name for "final" checkpoint, full name for others
                cp_suffix = cp["name"]
                if cp_suffix == "final":
                    checkpoint_name = exp_dir.name.lower()
                else:
                    checkpoint_name = f"{exp_dir.name.lower()}-{cp_suffix}"
                checkpoints.append(
                    {
                        "name": checkpoint_name,
                        "base_model": base_model,
                        "renderer": renderer,
                        "checkpoint_path": cp["sampler_path"],
                    }
                )

    return checkpoints


def build_registry(config: PromptsConfig) -> Path:
    """Build mq registry.json from all versions.

    Scans versions_dir for version directories with sysprompts/,
    creates model entries for each. Also discovers trained checkpoints.
    """
    models: dict[str, Any] = {}

    # Add plain models from config
    for model in config.plain_models:
        models[model.name] = {
            "provider": model.provider,
            "model": model.model,
            "params": {"temperature": model.temperature},
            "sysprompt": model.sysprompt,
        }

    # Discover and add trained checkpoints
    for cp in discover_checkpoints(config.project_dir):
        # Combine base_model::renderer::checkpoint_path into model field
        # This is the format mq/llm_client expects for tinker models
        renderer = cp["renderer"] or "qwen3"
        model_string = f"{cp['base_model']}::{renderer}::{cp['checkpoint_path']}"
        models[cp["name"]] = {
            "provider": "tinker",
            "model": model_string,
            "params": {"temperature": config.identity_temperature},
            # Note: trained checkpoints don't need sysprompt - it's baked in
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

            model_entry = {
                "provider": config.identity_provider,
                "model": config.identity_model,
                "params": {"temperature": config.identity_temperature},
                "sysprompt": sysprompt_path.read_text(),
            }
            models[model_name] = model_entry

            # Add release alias if this is the release version
            if version == config.release_version:
                release_name = f"{config.model_prefix}-release-{variant}"
                models[release_name] = model_entry

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

    isf_config["identity"]["release_version"] = version

    with open(isf_yaml, "w") as f:
        yaml.dump(isf_config, f, default_flow_style=False, sort_keys=False)

    # Rebuild registry to include new version
    build_registry(config)

    return version_dir


def list_versions(project_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """List all versions with their status.

    Returns tuple of:
    - versions: list of dicts with name, variants, models, is_release
    - config_info: dict with model_prefix, release_version
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

        # Build model names for each variant
        models = [f"{config.model_prefix}-{version_name}-{v}" for v in variants]

        versions.append(
            {
                "name": version_name,
                "variants": variants,
                "models": models,
                "is_release": version_name == config.release_version,
            }
        )

    config_info = {
        "model_prefix": config.model_prefix,
        "release_version": config.release_version,
    }

    return versions, config_info
