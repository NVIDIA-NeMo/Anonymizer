# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

from data_designer.config.utils.io_helpers import load_config_file

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "model_configs"


def load_models_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    """Load model definitions from models.yaml."""
    models_file = config_dir / "models.yaml"
    return _load_yaml_dict(models_file)


def load_workflow_selections(workflow_name: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> dict[str, str]:
    """Load selected model aliases for a workflow."""
    workflow_file = config_dir / f"{workflow_name}.yaml"
    workflow_config = _load_yaml_dict(workflow_file)
    selected_models = workflow_config.get("selected_models", {})
    if not isinstance(selected_models, dict):
        raise ValueError(f"{workflow_file} must define a top-level 'selected_models' mapping.")
    return {str(key): str(value) for key, value in selected_models.items()}


def load_workflow_config(workflow_name: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    """Load merged workflow config with model definitions and role selections."""
    models_config = load_models_config(config_dir=config_dir)
    merged = dict(models_config)
    merged["selected_models"] = load_workflow_selections(workflow_name=workflow_name, config_dir=config_dir)
    return merged


def get_model_alias(workflow_name: str, role: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> str:
    """Return the model alias assigned to a workflow role."""
    selected_models = load_workflow_selections(workflow_name=workflow_name, config_dir=config_dir)
    if role not in selected_models:
        available = ", ".join(sorted(selected_models.keys()))
        raise ValueError(f"Role '{role}' not found in workflow '{workflow_name}'. Available roles: {available}")
    return selected_models[role]


def resolve_model_alias(
    workflow_name: str,
    role: str,
    selection_model: object,
    config_dir: Path | None,
) -> str:
    """Resolve model alias from YAML or fallback to selection_model's role field.

    Role must match the field name on selection_model (e.g. entity_detector,
    replacement_generator). YAML keys must use the same names.
    """
    fallback = getattr(selection_model, role)
    if config_dir is None:
        return fallback
    try:
        return get_model_alias(workflow_name=workflow_name, role=role, config_dir=config_dir)
    except (FileNotFoundError, ValueError):
        return fallback


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = load_config_file(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data
