# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from anonymizer.engine.ndd.model_loader import (
    DEFAULT_CONFIG_DIR,
    WorkflowName,
    get_model_alias,
    load_default_model_selection,
    load_models_config,
    load_workflow_config,
    load_workflow_selections,
    parse_model_configs,
)


def test_load_workflow_config_contains_selected_models() -> None:
    config = load_workflow_config(WorkflowName.detection)
    assert "model_configs" in config
    assert "selected_models" in config
    assert config["selected_models"]["entity_detector"] == "gliner-pii-detector"


def test_get_model_alias_reads_workflow_mapping() -> None:
    alias = get_model_alias(workflow_name=WorkflowName.detection, role="entity_validator")
    assert alias == "gpt-oss-120b"


WORKFLOW_YAMLS = [p.stem for p in DEFAULT_CONFIG_DIR.glob("*.yaml") if p.stem != "models"]


@pytest.mark.parametrize("workflow_name", WORKFLOW_YAMLS)
def test_default_workflow_aliases_exist_in_models(workflow_name: str) -> None:
    """Every alias referenced in a workflow YAML must be defined in models.yaml."""
    models_config = load_models_config()
    known_aliases = {m["alias"] for m in models_config.get("model_configs", [])}
    selections = load_workflow_selections(WorkflowName(workflow_name))
    unknown = set(selections.values()) - known_aliases
    assert not unknown, f"Workflow '{workflow_name}' references unknown aliases: {unknown}. Known: {known_aliases}"


def test_load_default_model_selection_populates_all_workflows() -> None:
    selection = load_default_model_selection()
    assert selection.detection.entity_detector == "gliner-pii-detector"
    assert selection.replace.replacement_generator == "gpt-oss-120b"
    assert selection.rewrite.rewriter == "gpt-oss-120b"


def test_parse_model_configs_none_uses_defaults() -> None:
    result = parse_model_configs(None)
    assert len(result.model_configs) > 0
    assert result.selected_models.detection.entity_detector == "gliner-pii-detector"


def test_parse_model_configs_yaml_string_extracts_selections() -> None:
    yaml_str = """
selected_models:
  detection:
    entity_detector: custom-detector
model_configs:
  - alias: custom-detector
    model: test/model
  - alias: gpt-oss-120b
    model: test/gpt
"""
    result = parse_model_configs(yaml_str)
    assert result.selected_models.detection.entity_detector == "custom-detector"
    assert result.selected_models.replace.replacement_generator == "gpt-oss-120b"
    assert len(result.model_configs) == 2


def test_parse_model_configs_yaml_without_selections_uses_defaults() -> None:
    yaml_str = """
model_configs:
  - alias: stub-model
    model: test/stub
"""
    result = parse_model_configs(yaml_str)
    assert len(result.model_configs) == 1
    assert result.selected_models.detection.entity_detector == "gliner-pii-detector"
