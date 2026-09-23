# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Credential helpers for the notebook runtime."""

from __future__ import annotations

import os
from pathlib import Path

from data_designer.config.models import ModelProvider

from anonymizer.engine.ndd.model_loader import parse_model_providers
from anonymizer.interface.factory import GLINER_PROVIDER_NAME

LOCAL_TOKEN_ENV = "ANONYMIZER_LOCAL_GLINER2_TOKEN"


def required_api_key_environment_variables(
    model_providers: list[ModelProvider] | str | Path | None = None,
) -> list[str]:
    """Return missing credential variable names for the resolved external providers."""
    required: list[str] = []
    for provider in parse_model_providers(model_providers):
        variable = provider.api_key
        if (
            provider.name in {"local-gliner2", GLINER_PROVIDER_NAME}
            or not variable
            or variable == "EMPTY"
            or os.getenv(variable)
        ):
            continue
        required.append(variable)
    return list(dict.fromkeys(required))
