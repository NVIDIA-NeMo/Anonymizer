# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.models import DetectionModelSelection, ReplaceModelSelection
from anonymizer.config.replace_strategies import RedactReplace
from anonymizer.engine.detection.constants import COL_DETECTED_ENTITIES, COL_TEXT


@pytest.fixture
def stub_detector_model_configs() -> list[ModelConfig]:
    """Model configs with the GLiNER PII detector alias."""
    return [ModelConfig(alias="gliner-pii-detector", model="nvidia/nemotron-pii")]


@pytest.fixture
def stub_model_configs() -> list[ModelConfig]:
    """Generic model configs for workflows that don't care about the alias."""
    return [ModelConfig(alias="stub-model", model="stub-model")]


@pytest.fixture
def stub_detection_model_selection() -> DetectionModelSelection:
    return DetectionModelSelection()


@pytest.fixture
def stub_replace_model_selection() -> ReplaceModelSelection:
    return ReplaceModelSelection()


@pytest.fixture
def stub_anonymizer_config() -> AnonymizerConfig:
    """Minimal valid config using RedactReplace (no LLM needed)."""
    return AnonymizerConfig(replace=RedactReplace())


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
    """DataFrame with text + detected entities — input shape for replace runner."""
    df = stub_dataframe.copy()
    df[COL_DETECTED_ENTITIES] = [stub_entities]
    return df
