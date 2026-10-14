# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from data_designer.config.models import ModelProvider

from anonymizer.notebooks import required_api_key_environment_variables as public_required_api_keys
from anonymizer.notebooks._model_config import required_api_key_environment_variables


def test_required_api_key_helper_is_publicly_exported() -> None:
    assert public_required_api_keys is required_api_key_environment_variables


def test_required_api_keys_are_derived_from_external_provider_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXTERNAL_API_KEY", raising=False)
    providers = [
        ModelProvider(name="external", endpoint="https://example.test/v1", api_key="EXTERNAL_API_KEY"),
        ModelProvider(name="local-gliner2", endpoint="http://127.0.0.1:8001/v1", api_key="EMPTY"),
    ]
    assert required_api_key_environment_variables(providers) == ["EXTERNAL_API_KEY"]

    monkeypatch.setenv("EXTERNAL_API_KEY", "secret")
    assert required_api_key_environment_variables(providers) == []
