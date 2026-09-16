# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from data_designer.config.models import ModelProvider

from anonymizer.engine.ndd.model_loader import parse_model_configs
from anonymizer.notebooks import required_api_key_environment_variables as public_required_api_keys
from anonymizer.notebooks._model_config import (
    LOCAL_ALIAS,
    LOCAL_PROVIDER,
    build_notebook_model_configuration,
    required_api_key_environment_variables,
)
from anonymizer.notebooks.local_inference.gliner2 import MODEL_ID


def test_required_api_key_helper_is_publicly_exported() -> None:
    assert public_required_api_keys is required_api_key_environment_variables


def test_build_notebook_model_configuration_overrides_only_detector() -> None:
    configuration = build_notebook_model_configuration(
        model_configs=None,
        model_providers=None,
        endpoint="http://127.0.0.1:43210/v1",
    )
    parsed = parse_model_configs(configuration.model_configs)
    assert parsed.selected_models.detection.entity_detector == LOCAL_ALIAS
    local = next(config for config in parsed.model_configs if config.alias == LOCAL_ALIAS)
    assert local.model == MODEL_ID
    assert local.provider == LOCAL_PROVIDER
    assert any(provider.name == "nvidia" for provider in configuration.model_providers)
    injected = next(provider for provider in configuration.model_providers if provider.name == LOCAL_PROVIDER)
    assert injected.endpoint == "http://127.0.0.1:43210/v1"


def test_custom_configuration_preserves_non_detector_entries_and_selections() -> None:
    source = json.dumps(
        {
            "model_configs": [
                {"alias": "old-detector", "model": "test/old", "provider": "custom"},
                {"alias": "custom-llm", "model": "test/llm", "provider": "custom"},
            ],
            "selected_models": {
                "detection": {
                    "entity_detector": "old-detector",
                    "entity_validator": "custom-llm",
                    "entity_augmenter": "custom-llm",
                }
            },
        }
    )
    provider = ModelProvider(name="custom", endpoint="https://example.com/v1", api_key="CUSTOM_KEY")
    configuration = build_notebook_model_configuration(
        model_configs=source,
        model_providers=[provider],
        endpoint="http://127.0.0.1:43210/v1",
    )
    parsed = parse_model_configs(configuration.model_configs)
    custom_llm = next(config for config in parsed.model_configs if config.alias == "custom-llm")
    assert custom_llm.model == "test/llm"
    assert custom_llm.provider == "custom"
    assert parsed.selected_models.detection.entity_validator == ["custom-llm"]
    assert parsed.selected_models.detection.entity_augmenter == "custom-llm"
    assert parsed.selected_models.detection.entity_detector == LOCAL_ALIAS
    assert configuration.model_providers[0] is provider


@pytest.mark.parametrize("collision", [LOCAL_ALIAS, LOCAL_PROVIDER])
def test_reserved_names_are_rejected(collision: str) -> None:
    model_configs = None
    providers: list[ModelProvider] | None = None
    if collision == LOCAL_ALIAS:
        model_configs = json.dumps(
            {"model_configs": [{"alias": LOCAL_ALIAS, "model": "test/model", "provider": "nvidia"}]}
        )
    else:
        providers = [ModelProvider(name=LOCAL_PROVIDER, endpoint="https://example.com/v1")]
    with pytest.raises(ValueError, match="reserved"):
        build_notebook_model_configuration(
            model_configs=model_configs,
            model_providers=providers,
            endpoint="http://127.0.0.1:43210/v1",
        )


def test_required_api_keys_are_derived_from_external_provider_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXTERNAL_API_KEY", raising=False)
    providers = [
        ModelProvider(name="external", endpoint="https://example.test/v1", api_key="EXTERNAL_API_KEY"),
        ModelProvider(name="local-gliner2", endpoint="http://127.0.0.1:8001/v1", api_key="EMPTY"),
    ]
    assert required_api_key_environment_variables(providers) == ["EXTERNAL_API_KEY"]

    monkeypatch.setenv("EXTERNAL_API_KEY", "secret")
    assert required_api_key_environment_variables(providers) == []
