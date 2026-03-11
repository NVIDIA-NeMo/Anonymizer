# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Generator

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.models import (
    DetectionModelSelection,
    ModelSelection,
    ReplaceModelSelection,
    RewriteModelSelection,
)
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_TEXT
from anonymizer.engine.ndd.model_loader import load_default_model_selection


@pytest.fixture(autouse=True)
def _caplog_for_anonymizer(caplog: pytest.LogCaptureFixture) -> Generator[None]:
    """Ensure caplog captures from the anonymizer logger (propagate=False)."""
    anon_logger = logging.getLogger("anonymizer")
    anon_logger.addHandler(caplog.handler)
    yield
    anon_logger.removeHandler(caplog.handler)


@pytest.fixture
def stub_detector_model_configs() -> list[ModelConfig]:
    """Model configs with the GLiNER PII detector alias."""
    return [ModelConfig(alias="gliner-pii-detector", model="nvidia/nemotron-pii")]


@pytest.fixture
def stub_model_configs() -> list[ModelConfig]:
    """Generic model configs for workflows that don't care about the alias."""
    return [ModelConfig(alias="stub-model", model="stub-model")]


@pytest.fixture
def stub_known_model_configs() -> list[ModelConfig]:
    """Minimal model pool for alias validation tests."""
    return [ModelConfig(alias="known", model="some/model")]


@pytest.fixture
def stub_detection_model_selection() -> DetectionModelSelection:
    return load_default_model_selection().detection


@pytest.fixture
def stub_replace_model_selection() -> ReplaceModelSelection:
    return load_default_model_selection().replace


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
        rewrite=RewriteModelSelection(rewriter="known", evaluator="known"),
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
