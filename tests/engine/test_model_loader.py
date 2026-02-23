# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from anonymizer.engine.ndd.model_loader import (
    DEFAULT_CONFIG_DIR,
    get_model_alias,
    load_models_config,
    load_workflow_config,
    load_workflow_selections,
)


def test_load_workflow_config_contains_selected_models() -> None:
    config = load_workflow_config("entity_detection")
    assert "model_configs" in config
    assert "selected_models" in config
    assert config["selected_models"]["entity_detector"] == "gliner-pii-detector"


def test_get_model_alias_reads_workflow_mapping() -> None:
    alias = get_model_alias(workflow_name="entity_detection", role="entity_validator")
    assert alias == "gpt-oss-120b"


WORKFLOW_YAMLS = [p.stem for p in DEFAULT_CONFIG_DIR.glob("*.yaml") if p.stem != "models"]


@pytest.mark.parametrize("workflow_name", WORKFLOW_YAMLS)
def test_default_workflow_aliases_exist_in_models(workflow_name: str) -> None:
    """Every alias referenced in a workflow YAML must be defined in models.yaml."""
    models_config = load_models_config()
    known_aliases = {m["alias"] for m in models_config.get("model_configs", [])}
    selections = load_workflow_selections(workflow_name)
    unknown = set(selections.values()) - known_aliases
    assert not unknown, f"Workflow '{workflow_name}' references unknown aliases: {unknown}. Known: {known_aliases}"
