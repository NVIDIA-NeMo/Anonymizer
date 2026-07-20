# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.models import (
    DetectionModelSelection,
    EvaluateModelSelection,
    ModelSelection,
    ReplaceModelSelection,
    RewriteModelSelection,
)
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_TEXT
from anonymizer.engine.ndd.model_loader import load_default_model_selection


@pytest.fixture
def load_tool() -> Callable[..., ModuleType]:
    """Load a Python tool module with isolated module state."""

    def loader(
        module_name: str,
        path: Path,
        *,
        additional_paths: tuple[Path, ...] = (),
    ) -> ModuleType:
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        import_paths = list(additional_paths)
        if not any(path.is_relative_to(import_path) for import_path in additional_paths):
            import_paths.append(path.parent)
        for import_path in import_paths:
            sys.path.insert(0, str(import_path))
        spec.loader.exec_module(module)
        return module

    return loader


@pytest.fixture(autouse=True)
def _caplog_for_anonymizer(caplog: pytest.LogCaptureFixture) -> Generator[None]:
    """Ensure caplog captures from the anonymizer logger (propagate=False)."""
    anon_logger = logging.getLogger("anonymizer")
    anon_logger.addHandler(caplog.handler)
    yield
    anon_logger.removeHandler(caplog.handler)


@pytest.fixture(autouse=True)
def _isolate_telemetry_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep telemetry quiet and deterministic in unit tests.

    - Disable emission by default. Tests that exercise the emit path can opt in
      by setting NEMO_TELEMETRY_ENABLED=true via their own monkeypatch.
    - Clear NEMO_DEPLOYMENT_TYPE, NEMO_SESSION_PREFIX, and NEMO_TELEMETRY_ENDPOINT
      so tests don't inherit values from the developer's shell.
    """
    monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "false")
    monkeypatch.delenv("NEMO_DEPLOYMENT_TYPE", raising=False)
    monkeypatch.delenv("NEMO_SESSION_PREFIX", raising=False)
    monkeypatch.delenv("NEMO_TELEMETRY_ENDPOINT", raising=False)


@pytest.fixture
def stub_detector_model_configs() -> list[ModelConfig]:
    """Model configs with the GLiNER PII detector alias."""
    return [ModelConfig(alias="gliner-pii-detector", model="nvidia/nemotron-pii", provider="stub")]


@pytest.fixture
def stub_model_configs() -> list[ModelConfig]:
    """Generic model configs for workflows that don't care about the alias."""
    return [ModelConfig(alias="stub-model", model="stub-model", provider="stub")]


@pytest.fixture
def stub_known_model_configs() -> list[ModelConfig]:
    """Minimal model pool for alias validation tests."""
    return [ModelConfig(alias="known", model="some/model", provider="stub")]


@pytest.fixture
def stub_detection_model_selection() -> DetectionModelSelection:
    return load_default_model_selection().detection


@pytest.fixture
def stub_replace_model_selection() -> ReplaceModelSelection:
    return load_default_model_selection().replace


@pytest.fixture
def stub_rewrite_model_selection() -> RewriteModelSelection:
    return load_default_model_selection().rewrite


@pytest.fixture
def stub_evaluate_model_selection() -> EvaluateModelSelection:
    return load_default_model_selection().evaluate


@pytest.fixture
def stub_slim_model_selection() -> ModelSelection:
    """Selection model where every role points to the same known alias."""
    return ModelSelection(
        detection=DetectionModelSelection(
            entity_detector="known",
            entity_validator="known",
            entity_augmenter="known",
            latent_detector="known",
        ),
        replace=ReplaceModelSelection(replacement_generator="known"),
        rewrite=RewriteModelSelection(
            domain_classifier="known",
            disposition_analyzer="known",
            meaning_extractor="known",
            qa_generator="known",
            rewriter="known",
            evaluator="known",
            repairer="known",
        ),
        evaluate=EvaluateModelSelection(
            detection_validity_judge="known",
            replace_type_fidelity_judge="known",
            replace_relational_consistency_judge="known",
            replace_attribute_fidelity_judge="known",
            rewrite_judge="known",
        ),
    )


@pytest.fixture
def stub_anonymizer_config() -> AnonymizerConfig:
    """Minimal valid config using Redact (no LLM needed)."""
    return AnonymizerConfig(replace=Redact())


@pytest.fixture
def stub_dataframe() -> pd.DataFrame:
    """Single-row text DataFrame matching stub_entities positions."""
    return pd.DataFrame({COL_TEXT: ["Alice works at Acme"]})


@pytest.fixture
def stub_entities() -> list[dict]:
    """Detected entity dicts — the common shape after detection."""
    return [
        {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
        {"value": "Acme", "label": "organization", "start_position": 15, "end_position": 19},
    ]


@pytest.fixture
def stub_dataframe_with_entities(stub_dataframe: pd.DataFrame, stub_entities: list[dict]) -> pd.DataFrame:
    """DataFrame with text + detected entities — canonical dict-wrapped shape from detection."""
    df = stub_dataframe.copy()
    df[COL_FINAL_ENTITIES] = [{"entities": stub_entities}]
    return df
