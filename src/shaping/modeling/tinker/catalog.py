"""Tinker model catalog.

Provides access to available Tinker base models with metadata.
"""

from dataclasses import dataclass
from functools import lru_cache


@dataclass
class TinkerModel:
    """Model information from Tinker catalog."""

    name: str  # Full model name (e.g., "Qwen/Qwen3-30B-A3B")
    organization: str  # e.g., "Qwen", "meta-llama"
    size: str  # e.g., "30B-A3B", "8B"
    training_type: str  # "Base", "Instruction", "Hybrid", "Reasoning", "Vision"
    architecture: str  # "Dense" or "MoE"
    renderer: str  # Recommended renderer name
    is_chat: bool  # Has chat/instruct training
    is_vl: bool  # Vision-language model


# Static metadata - synced from tinker-cookbook model_info.py and model-lineup.mdx
# This supplements the live API which only returns model names
MODEL_METADATA = {
    # DeepSeek
    "deepseek-ai/DeepSeek-V3.1": {
        "size": "671B-A37B",
        "training_type": "Hybrid",
        "architecture": "MoE",
        "renderer": "deepseekv3",
        "is_chat": True,
        "is_vl": False,
    },
    "deepseek-ai/DeepSeek-V3.1-Base": {
        "size": "671B-A37B",
        "training_type": "Base",
        "architecture": "MoE",
        "renderer": "role_colon",
        "is_chat": False,
        "is_vl": False,
    },
    # Moonshot
    "moonshotai/Kimi-K2-Thinking": {
        "size": "1T-A32B",
        "training_type": "Reasoning",
        "architecture": "MoE",
        "renderer": "kimi_k2",
        "is_chat": True,
        "is_vl": False,
    },
    # Meta Llama
    "meta-llama/Llama-3.1-70B": {
        "size": "70B",
        "training_type": "Base",
        "architecture": "Dense",
        "renderer": "role_colon",
        "is_chat": False,
        "is_vl": False,
    },
    "meta-llama/Llama-3.1-8B": {
        "size": "8B",
        "training_type": "Base",
        "architecture": "Dense",
        "renderer": "role_colon",
        "is_chat": False,
        "is_vl": False,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "size": "8B",
        "training_type": "Instruction",
        "architecture": "Dense",
        "renderer": "llama3",
        "is_chat": True,
        "is_vl": False,
    },
    "meta-llama/Llama-3.2-1B": {
        "size": "1B",
        "training_type": "Base",
        "architecture": "Dense",
        "renderer": "role_colon",
        "is_chat": False,
        "is_vl": False,
    },
    "meta-llama/Llama-3.2-3B": {
        "size": "3B",
        "training_type": "Base",
        "architecture": "Dense",
        "renderer": "role_colon",
        "is_chat": False,
        "is_vl": False,
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "size": "70B",
        "training_type": "Instruction",
        "architecture": "Dense",
        "renderer": "llama3",
        "is_chat": True,
        "is_vl": False,
    },
    # Qwen
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {
        "size": "235B-A22B",
        "training_type": "Instruction",
        "architecture": "MoE",
        "renderer": "qwen3_instruct",
        "is_chat": True,
        "is_vl": False,
    },
    "Qwen/Qwen3-30B-A3B": {
        "size": "30B-A3B",
        "training_type": "Hybrid",
        "architecture": "MoE",
        "renderer": "qwen3",
        "is_chat": True,
        "is_vl": False,
    },
    "Qwen/Qwen3-30B-A3B-Base": {
        "size": "30B-A3B",
        "training_type": "Base",
        "architecture": "MoE",
        "renderer": "role_colon",
        "is_chat": False,
        "is_vl": False,
    },
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "size": "30B-A3B",
        "training_type": "Instruction",
        "architecture": "MoE",
        "renderer": "qwen3_instruct",
        "is_chat": True,
        "is_vl": False,
    },
    "Qwen/Qwen3-32B": {
        "size": "32B",
        "training_type": "Hybrid",
        "architecture": "Dense",
        "renderer": "qwen3",
        "is_chat": True,
        "is_vl": False,
    },
    "Qwen/Qwen3-4B-Instruct-2507": {
        "size": "4B",
        "training_type": "Instruction",
        "architecture": "Dense",
        "renderer": "qwen3_instruct",
        "is_chat": True,
        "is_vl": False,
    },
    "Qwen/Qwen3-8B": {
        "size": "8B",
        "training_type": "Hybrid",
        "architecture": "Dense",
        "renderer": "qwen3",
        "is_chat": True,
        "is_vl": False,
    },
    "Qwen/Qwen3-8B-Base": {
        "size": "8B",
        "training_type": "Base",
        "architecture": "Dense",
        "renderer": "role_colon",
        "is_chat": False,
        "is_vl": False,
    },
    "Qwen/Qwen3-VL-235B-A22B-Instruct": {
        "size": "235B-A22B",
        "training_type": "Vision",
        "architecture": "MoE",
        "renderer": "qwen3_vl_instruct",
        "is_chat": True,
        "is_vl": True,
    },
    "Qwen/Qwen3-VL-30B-A3B-Instruct": {
        "size": "30B-A3B",
        "training_type": "Vision",
        "architecture": "MoE",
        "renderer": "qwen3_vl_instruct",
        "is_chat": True,
        "is_vl": True,
    },
    # OpenAI OSS
    "openai/gpt-oss-120b": {
        "size": "117B-A5.1B",
        "training_type": "Reasoning",
        "architecture": "MoE",
        "renderer": "gpt_oss_no_sysprompt",
        "is_chat": True,
        "is_vl": False,
    },
    "openai/gpt-oss-20b": {
        "size": "21B-A3.6B",
        "training_type": "Reasoning",
        "architecture": "MoE",
        "renderer": "gpt_oss_no_sysprompt",
        "is_chat": True,
        "is_vl": False,
    },
}


def _parse_organization(model_name: str) -> str:
    """Extract organization from model name."""
    if "/" in model_name:
        return model_name.split("/")[0]
    return "unknown"


def _build_model_info(model_name: str) -> TinkerModel:
    """Build TinkerModel from name and metadata."""
    org = _parse_organization(model_name)
    meta = MODEL_METADATA.get(model_name, {})

    return TinkerModel(
        name=model_name,
        organization=org,
        size=meta.get("size", "unknown"),
        training_type=meta.get("training_type", "unknown"),
        architecture=meta.get("architecture", "unknown"),
        renderer=meta.get("renderer", "unknown"),
        is_chat=meta.get("is_chat", False),
        is_vl=meta.get("is_vl", False),
    )


@lru_cache(maxsize=1)
def list_available_models() -> list[TinkerModel]:
    """List available Tinker base models.

    Queries the Tinker API for currently available models and enriches
    with metadata from the local catalog.

    Returns:
        List of TinkerModel objects sorted by organization and name.

    Raises:
        Exception: If Tinker API is unavailable or authentication fails.
    """
    from tinker import ServiceClient

    client = ServiceClient()
    caps = client.get_server_capabilities()

    models = []
    for supported in caps.supported_models:
        model = _build_model_info(supported.model_name)
        models.append(model)

    # Sort by organization, then by name
    models.sort(key=lambda m: (m.organization, m.name))
    return models


def get_model(name: str) -> TinkerModel | None:
    """Get a specific model by name.

    Args:
        name: Full model name (e.g., "Qwen/Qwen3-30B-A3B")

    Returns:
        TinkerModel if found, None otherwise.
    """
    models = list_available_models()
    for model in models:
        if model.name == name:
            return model
    return None


def list_models_by_type(training_type: str) -> list[TinkerModel]:
    """List models filtered by training type.

    Args:
        training_type: One of "Base", "Instruction", "Hybrid", "Reasoning", "Vision"

    Returns:
        List of matching TinkerModel objects.
    """
    models = list_available_models()
    return [m for m in models if m.training_type.lower() == training_type.lower()]


def list_models_by_architecture(architecture: str) -> list[TinkerModel]:
    """List models filtered by architecture.

    Args:
        architecture: One of "Dense", "MoE"

    Returns:
        List of matching TinkerModel objects.
    """
    models = list_available_models()
    return [m for m in models if m.architecture.lower() == architecture.lower()]
