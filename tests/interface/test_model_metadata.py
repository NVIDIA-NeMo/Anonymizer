# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.models import ModelConfig

from anonymizer.config.anonymizer_config import AnonymizerConfig, EvaluateConfig
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.ndd.model_loader import load_default_model_selection
from anonymizer.interface.model_metadata import evaluation_model_metadata, run_model_metadata


def test_validator_pool_is_saved_as_separate_models_without_private_config() -> None:
    selection = load_default_model_selection()
    selection.detection.entity_validator = ["validator-a", "validator-b"]
    models = [
        ModelConfig(alias=alias, model="model/" + alias, provider="provider")
        for alias in ["validator-a", "validator-b"]
    ]
    rows = run_model_metadata(AnonymizerConfig(replace=Redact()), selection, models)
    validators = [entry for entry in rows if entry["task"] == "Entity validation"]
    assert [entry["model"] for entry in validators] == ["model/validator-a", "model/validator-b"]
    assert all(set(entry) == {"phase", "group", "task", "alias", "model", "provider"} for entry in rows)


def test_optional_evaluation_roles_and_classifier_fallback() -> None:
    selection = load_default_model_selection()
    selection.evaluate.privacy_entity_classifier = None
    config = EvaluateConfig(compute_privacy_leakage=True, compute_detection_validity=True)
    rows = evaluation_model_metadata(config, selection, [], rewrite=False, substitute=True)
    tasks = {entry["task"]: entry for entry in rows}
    assert tasks["Entity classification (custom labels)"]["alias"] == selection.evaluate.privacy_sensitivity_analyzer
    assert {
        "Detection validity",
        "Type fidelity",
        "Attribute fidelity",
        "Relational consistency",
        "Privacy QA answering",
    } <= set(tasks)
    rewrite = evaluation_model_metadata(config, selection, [], rewrite=True, substitute=False)
    assert {entry["task"] for entry in rewrite} == {
        "Entity coverage",
        "Detection validity",
        "Privacy / quality / style judge",
    }
