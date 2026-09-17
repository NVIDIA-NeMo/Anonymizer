# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construct private model/provider configuration for the notebook runtime."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from data_designer.config.models import ModelProvider

from anonymizer.engine.ndd.model_loader import parse_model_configs, parse_model_providers
from anonymizer.notebooks.local_inference.gliner2 import MODEL_ID

LOCAL_ALIAS = "local-gliner2-pii"
LOCAL_PROVIDER = "notebook-local-gliner2"
LOCAL_TOKEN_ENV = "ANONYMIZER_LOCAL_GLINER2_TOKEN"


@dataclass(frozen=True)
class NotebookModelConfiguration:
    """Inputs accepted by the core ``Anonymizer`` constructor."""

    model_configs: str
    model_providers: list[ModelProvider]


def required_api_key_environment_variables(
    model_providers: list[ModelProvider] | str | Path | None = None,
) -> list[str]:
    """Return missing credential variable names for the resolved external providers."""
    required: list[str] = []
    for provider in parse_model_providers(model_providers):
        variable = provider.api_key
        if (
            provider.name in {"local-gliner2", LOCAL_PROVIDER}
            or not variable
            or variable == "EMPTY"
            or os.getenv(variable)
        ):
            continue
        required.append(variable)
    return list(dict.fromkeys(required))


def build_notebook_model_configuration(
    *,
    model_configs: str | Path | None,
    model_providers: list[ModelProvider] | str | Path | None,
    endpoint: str,
) -> NotebookModelConfiguration:
    """Inject the owned detector while preserving every other caller setting."""
    validate_notebook_model_inputs(model_configs=model_configs, model_providers=model_providers)
    parsed = parse_model_configs(model_configs)
    providers = parse_model_providers(model_providers)

    model_entries = [config.model_dump(mode="json", exclude_none=True) for config in parsed.model_configs]
    model_entries.append(
        {
            "alias": LOCAL_ALIAS,
            "model": MODEL_ID,
            "provider": LOCAL_PROVIDER,
            "skip_health_check": True,
            "inference_parameters": {
                "max_parallel_requests": 8,
                "timeout": 120,
            },
        }
    )
    selections = parsed.selected_models.model_dump(mode="json")
    selections["detection"]["entity_detector"] = LOCAL_ALIAS
    serialized = json.dumps({"model_configs": model_entries, "selected_models": selections})

    local_provider = ModelProvider(
        name=LOCAL_PROVIDER,
        endpoint=endpoint,
        provider_type="openai",
        api_key=LOCAL_TOKEN_ENV,
    )
    return NotebookModelConfiguration(
        model_configs=serialized,
        model_providers=[*providers, local_provider],
    )


def validate_notebook_model_inputs(
    *,
    model_configs: str | Path | None,
    model_providers: list[ModelProvider] | str | Path | None,
) -> None:
    """Reject reserved notebook runtime names before starting a child process."""
    parsed = parse_model_configs(model_configs)
    if any(config.alias == LOCAL_ALIAS for config in parsed.model_configs):
        raise ValueError(
            f"Model alias {LOCAL_ALIAS!r} is reserved by anonymizer.notebooks; rename the caller-supplied alias."
        )
    providers = parse_model_providers(model_providers)
    if any(provider.name == LOCAL_PROVIDER for provider in providers):
        raise ValueError(
            f"Provider name {LOCAL_PROVIDER!r} is reserved by anonymizer.notebooks; "
            "rename the caller-supplied provider."
        )
