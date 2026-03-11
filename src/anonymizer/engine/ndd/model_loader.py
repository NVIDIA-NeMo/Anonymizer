# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from data_designer.config.models import ModelConfig, load_model_configs
from data_designer.config.utils.io_helpers import load_config_file

from anonymizer.config.models import (
    DetectionModelSelection,
    ModelSelection,
    ReplaceModelSelection,
    RewriteModelSelection,
)

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "default_model_configs"


class WorkflowName(str, Enum):
    detection = "detection"
    replace = "replace"
    rewrite = "rewrite"


@dataclass(frozen=True)
class ParsedModelConfigs:
    """Result of parsing a unified model configs YAML."""

    model_configs: list[ModelConfig]
    selected_models: ModelSelection


def parse_model_configs(raw: str | Path | None) -> ParsedModelConfigs:
    """Parse a unified YAML into a model pool and role selections.

    The YAML may contain an optional ``selected_models`` section alongside
    the standard ``model_configs`` list. Omitted roles fall back to the
    bundled YAML defaults.
    """
    if raw is None:
        default_yaml = _load_yaml_dict(DEFAULT_CONFIG_DIR / "models.yaml")
        return ParsedModelConfigs(
            model_configs=load_model_configs(default_yaml),
            selected_models=load_default_model_selection(),
        )

    if isinstance(raw, Path):
        parsed = _load_yaml_dict(raw)
    else:
        parsed = _parse_yaml_string(raw)

    user_selections = parsed.pop("selected_models", None)
    return ParsedModelConfigs(
        model_configs=load_model_configs(parsed),
        selected_models=_merge_selections(user_selections),
    )


def load_default_model_selection(config_dir: Path | None = None) -> ModelSelection:
    """Load default ``ModelSelection`` from YAML workflow files.

    This is the single source of truth for default role-to-alias mappings.
    """
    resolved_dir = config_dir or DEFAULT_CONFIG_DIR
    return ModelSelection(
        detection=DetectionModelSelection(**load_workflow_selections(WorkflowName.detection, resolved_dir)),
        replace=ReplaceModelSelection(**load_workflow_selections(WorkflowName.replace, resolved_dir)),
        rewrite=RewriteModelSelection(**load_workflow_selections(WorkflowName.rewrite, resolved_dir)),
    )


def load_models_config(config_dir: Path | None = None) -> dict[str, Any]:
    """Load raw model definitions from models.yaml.

    Returns the unparsed YAML dict; callers pass this to DD's
    ``load_model_configs()`` for deserialization into ``ModelConfig`` objects.
    """
    resolved_dir = config_dir or DEFAULT_CONFIG_DIR
    return _load_yaml_dict(resolved_dir / "models.yaml")


def load_workflow_selections(workflow_name: WorkflowName, config_dir: Path | None = None) -> dict[str, str]:
    """Load selected model aliases for a workflow."""
    resolved_dir = config_dir or DEFAULT_CONFIG_DIR
    workflow_file = resolved_dir / f"{workflow_name.value}.yaml"
    workflow_config = _load_yaml_dict(workflow_file)
    selected_models = workflow_config.get("selected_models", {})
    if not isinstance(selected_models, dict):
        raise ValueError(f"{workflow_file} must define a top-level 'selected_models' mapping.")
    return {str(key): str(value) for key, value in selected_models.items()}


def load_workflow_config(workflow_name: WorkflowName, config_dir: Path | None = None) -> dict[str, Any]:
    """Load merged workflow config with model definitions and role selections.

    Returns a raw dict combining models.yaml and the workflow's role mappings.
    Downstream code passes this to DD's ``load_model_configs()`` for parsing.
    """
    models_config = load_models_config(config_dir=config_dir)
    selections = load_workflow_selections(workflow_name=workflow_name, config_dir=config_dir)
    _validate_alias_references(models_config, selections, workflow_name.value)
    merged = dict(models_config)
    merged["selected_models"] = selections
    return merged


def get_model_alias(workflow_name: WorkflowName, role: str, config_dir: Path | None = None) -> str:
    """Return the model alias assigned to a workflow role."""
    selected_models = load_workflow_selections(workflow_name=workflow_name, config_dir=config_dir)
    if role not in selected_models:
        available = ", ".join(sorted(selected_models.keys()))
        raise ValueError(f"Role '{role}' not found in workflow '{workflow_name.value}'. Available roles: {available}")
    return selected_models[role]


def resolve_model_alias(
    role: str,
    selection_model: DetectionModelSelection | ReplaceModelSelection | RewriteModelSelection,
) -> str:
    """Read model alias directly from the selection model.

    The selection model is already populated with defaults from YAML
    (via ``load_default_model_selection``) or user overrides.
    """
    return getattr(selection_model, role)


def _merge_selections(user_selections: dict[str, dict[str, str]] | None) -> ModelSelection:
    """Merge user-provided role selections onto YAML defaults."""
    defaults = load_default_model_selection()
    if not user_selections or not isinstance(user_selections, dict):
        return defaults

    detection_overrides = user_selections.get(WorkflowName.detection.value, {})
    replace_overrides = user_selections.get(WorkflowName.replace.value, {})
    rewrite_overrides = user_selections.get(WorkflowName.rewrite.value, {})

    return ModelSelection(
        detection=defaults.detection.model_copy(update=detection_overrides)
        if detection_overrides
        else defaults.detection,
        replace=defaults.replace.model_copy(update=replace_overrides) if replace_overrides else defaults.replace,
        rewrite=defaults.rewrite.model_copy(update=rewrite_overrides) if rewrite_overrides else defaults.rewrite,
    )


def validate_model_alias_references(
    model_configs: list[ModelConfig],
    selected_models: ModelSelection,
    *,
    check_substitute: bool = False,
    check_rewrite: bool = False,
) -> None:
    """Validate that active workflow model aliases exist in the model pool."""
    known_aliases = {model_config.alias for model_config in model_configs}
    detection_roles = selected_models.detection.model_dump()

    roles_to_check: dict[str, str] = {
        f"detection.{role}": detection_roles[role]
        for role in ("entity_detector", "entity_validator", "entity_augmenter")
    }
    if check_rewrite:
        roles_to_check.update(
            {
                "detection.latent_detector": detection_roles["latent_detector"],
                **{f"rewrite.{role}": alias for role, alias in selected_models.rewrite.model_dump().items()},
            }
        )
    if check_substitute:
        roles_to_check.update(
            {f"replace.{role}": alias for role, alias in selected_models.replace.model_dump().items()}
        )

    unknown = {path: alias for path, alias in roles_to_check.items() if alias not in known_aliases}
    if unknown:
        unknown_str = ", ".join(f"{path}={alias!r}" for path, alias in sorted(unknown.items()))
        raise ValueError(
            f"Selected model aliases not found in model pool: {unknown_str}. Known aliases: {sorted(known_aliases)}"
        )


def _validate_alias_references(
    models_config: dict[str, Any],
    selections: dict[str, str],
    workflow_name: str,
) -> None:
    """Validate that bundled workflow YAMLs reference aliases defined in models.yaml."""
    known_aliases = {m["alias"] for m in models_config.get("model_configs", [])}
    unknown = set(selections.values()) - known_aliases
    if unknown:
        raise ValueError(
            f"Workflow '{workflow_name}' references unknown model aliases: {unknown}. Known aliases: {known_aliases}"
        )


def _parse_yaml_string(raw: str) -> dict[str, Any]:
    import yaml

    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected YAML mapping, got {type(parsed).__name__}")
    return parsed


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = load_config_file(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data
