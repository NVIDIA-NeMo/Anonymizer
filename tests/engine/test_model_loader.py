# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from anonymizer.engine.ndd.model_loader import get_model_alias, load_workflow_config


def test_load_workflow_config_contains_selected_models() -> None:
    config = load_workflow_config("entity_detection")
    assert "model_configs" in config
    assert "selected_models" in config
    assert config["selected_models"]["entity_detector"] == "gliner-pii-detector"


def test_get_model_alias_reads_workflow_mapping() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "src" / "anonymizer" / "config" / "model_configs"
    alias = get_model_alias(workflow_name="entity_detection", role="entity_validator", config_dir=config_dir)
    assert alias == "gpt-oss-120b"
