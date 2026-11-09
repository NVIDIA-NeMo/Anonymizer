# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public construction paths for GLiNER-backed Anonymizer clients."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from data_designer.config.models import ModelProvider
from pydantic import BaseModel, ConfigDict, Field

from anonymizer.engine.ndd.model_loader import parse_model_configs, parse_model_providers

if TYPE_CHECKING:
    from data_designer.config.run_config import RunConfig

    from anonymizer.engine.ndd.model_loader import ParsedModelConfigs
    from anonymizer.interface.anonymizer import Anonymizer

GLINER_MODEL_ALIAS = "anonymizer-gliner-detector"
GLINER_PROVIDER_NAME = "anonymizer-gliner-endpoint"


class NativeGliner(BaseModel):
    """Request the library-managed native GLiNER2 runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"


class GlinerEndpoint(BaseModel):
    """Connect to an already-managed GLiNER-compatible HTTP endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str | None = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class _AnonymizerModelConfiguration:
    """Inputs accepted by the core ``Anonymizer`` constructor."""

    model_configs: str
    model_providers: list[ModelProvider]


def create_anonymizer(
    *,
    gliner: NativeGliner | GlinerEndpoint,
    model_configs: str | Path | None = None,
    model_providers: list[ModelProvider] | str | Path | None = None,
    artifact_path: str | Path | None = None,
    data_designer_run_config: RunConfig | None = None,
) -> Anonymizer:
    """Construct an Anonymizer with native or caller-managed GLiNER detection.

    ``NativeGliner`` owns a library-managed singleton process. ``GlinerEndpoint``
    only configures a client connection and never starts or stops the endpoint.
    """
    if not isinstance(gliner, (NativeGliner, GlinerEndpoint)):
        raise TypeError("gliner must be NativeGliner or GlinerEndpoint")
    parsed, providers = _parse_gliner_model_inputs(model_configs=model_configs, model_providers=model_providers)
    if isinstance(gliner, NativeGliner):
        from anonymizer.engine.local_inference.gliner2 import MODEL_ID
        from anonymizer.engine.local_inference.gliner2.backend import DEFAULT_CHUNK_LENGTH, DEFAULT_OVERLAP
        from anonymizer.engine.local_inference.runtime import LOCAL_TOKEN_ENV, singleton_runtime

        with singleton_runtime(gliner.device) as runtime:
            return _create_endpoint_anonymizer(
                endpoint=GlinerEndpoint(url=runtime.endpoint, model=MODEL_ID, api_key_env=LOCAL_TOKEN_ENV),
                parsed=parsed,
                providers=providers,
                artifact_path=artifact_path,
                data_designer_run_config=data_designer_run_config,
                extra_body={"chunk_length": DEFAULT_CHUNK_LENGTH, "overlap": DEFAULT_OVERLAP},
            )
    return _create_endpoint_anonymizer(
        endpoint=gliner,
        parsed=parsed,
        providers=providers,
        artifact_path=artifact_path,
        data_designer_run_config=data_designer_run_config,
        extra_body=None,
    )


def _create_endpoint_anonymizer(
    *,
    endpoint: GlinerEndpoint,
    parsed: ParsedModelConfigs,
    providers: list[ModelProvider],
    artifact_path: str | Path | None,
    data_designer_run_config: RunConfig | None,
    extra_body: dict[str, int] | None,
) -> Anonymizer:
    configuration = _build_gliner_model_configuration(
        parsed=parsed,
        providers=providers,
        endpoint=endpoint,
        extra_body=extra_body,
    )
    from anonymizer.interface.anonymizer import Anonymizer

    return Anonymizer(
        model_configs=configuration.model_configs,
        model_providers=configuration.model_providers,
        artifact_path=artifact_path,
        data_designer_run_config=data_designer_run_config,
    )


@dataclass(frozen=True, slots=True)
class NativeRuntimeSession:
    """A native endpoint and client factory valid for its enclosing context."""

    endpoint: GlinerEndpoint

    def create_anonymizer(
        self,
        *,
        model_configs: str | Path | None = None,
        model_providers: list[ModelProvider] | str | Path | None = None,
        artifact_path: str | Path | None = None,
        data_designer_run_config: RunConfig | None = None,
    ) -> Anonymizer:
        """Construct a client connected to this session's endpoint."""
        from anonymizer.engine.local_inference.gliner2.backend import DEFAULT_CHUNK_LENGTH, DEFAULT_OVERLAP

        parsed, providers = _parse_gliner_model_inputs(model_configs=model_configs, model_providers=model_providers)
        return _create_endpoint_anonymizer(
            endpoint=self.endpoint,
            extra_body={"chunk_length": DEFAULT_CHUNK_LENGTH, "overlap": DEFAULT_OVERLAP},
            parsed=parsed,
            providers=providers,
            artifact_path=artifact_path,
            data_designer_run_config=data_designer_run_config,
        )


@contextmanager
def native_runtime(*, device: Literal["auto", "cpu", "cuda", "mps"] = "auto") -> Iterator[NativeRuntimeSession]:
    """Run native GLiNER for the duration of a client session.

    An existing singleton is borrowed and remains running after exit. Nested
    contexts share the process; clients of an owned process expire on exit.
    """
    from anonymizer.engine.local_inference.gliner2 import MODEL_ID
    from anonymizer.engine.local_inference.runtime import LOCAL_TOKEN_ENV, runtime_context

    request = NativeGliner(device=device)
    with runtime_context(request.device) as runtime:
        yield NativeRuntimeSession(
            endpoint=GlinerEndpoint(url=runtime.endpoint, model=MODEL_ID, api_key_env=LOCAL_TOKEN_ENV)
        )


def _parse_gliner_model_inputs(
    *,
    model_configs: str | Path | None,
    model_providers: list[ModelProvider] | str | Path | None,
) -> tuple[ParsedModelConfigs, list[ModelProvider]]:
    """Parse and validate caller inputs before acquiring a native runtime."""
    parsed = parse_model_configs(model_configs)
    if any(config.alias == GLINER_MODEL_ALIAS for config in parsed.model_configs):
        raise ValueError(f"Model alias {GLINER_MODEL_ALIAS!r} is reserved by the GLiNER factory.")
    providers = list(parse_model_providers(model_providers))
    if any(provider.name == GLINER_PROVIDER_NAME for provider in providers):
        raise ValueError(f"Provider name {GLINER_PROVIDER_NAME!r} is reserved by the GLiNER factory.")
    return parsed, providers


def _build_gliner_model_configuration(
    *,
    parsed: ParsedModelConfigs,
    providers: list[ModelProvider],
    endpoint: GlinerEndpoint,
    extra_body: dict[str, int] | None = None,
) -> _AnonymizerModelConfiguration:
    """Inject one detector while preserving all unrelated caller configuration."""
    model_entries = [config.model_dump(mode="json", exclude_none=True) for config in parsed.model_configs]
    model_entries.append(
        {
            "alias": GLINER_MODEL_ALIAS,
            "model": endpoint.model,
            "provider": GLINER_PROVIDER_NAME,
            "skip_health_check": True,
            "inference_parameters": {
                "max_parallel_requests": 8,
                "timeout": 120,
                **({"extra_body": extra_body} if extra_body is not None else {}),
            },
        }
    )
    selections = parsed.selected_models.model_dump(mode="json")
    selections["detection"]["entity_detector"] = GLINER_MODEL_ALIAS
    serialized = json.dumps({"model_configs": model_entries, "selected_models": selections})
    detector_provider = ModelProvider(
        name=GLINER_PROVIDER_NAME,
        endpoint=endpoint.url,
        provider_type="openai",
        api_key=endpoint.api_key_env or "EMPTY",
    )
    return _AnonymizerModelConfiguration(
        model_configs=serialized,
        model_providers=[*providers, detector_provider],
    )


__all__ = ["GlinerEndpoint", "NativeGliner", "NativeRuntimeSession", "create_anonymizer", "native_runtime"]
