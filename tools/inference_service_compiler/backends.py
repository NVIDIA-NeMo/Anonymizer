# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Everything that differs between serving backends, behind one function."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import assert_never

import httpx
from pydantic import JsonValue

from inference_service_compiler.models import Gliner2Backend, ServiceSpec, VllmBackend

VLLM_PYTHON = "tools/inference-service-runtime/.venv/bin/python"
VLLM_ENTRY = "tools/inference_service_compiler/vllm_server.py"
GLINER2_PYTHON = ".venv/bin/python"
GLINER2_MODULE = "anonymizer.engine.local_inference.serve"
"""Builds or reuses the isolated GLiNER2 environment, then execs the server in place (same PID)."""
GLINER2_TOKEN_ENV = "ANONYMIZER_LOCAL_GLINER2_TOKEN"
GLINER2_DEVICE_ENV = "ANONYMIZER_LOCAL_GLINER2_DEVICE"

_GLINER2_PROBE_TEXT = "Ada Lovelace lives in London."

Probe = Callable[[httpx.Client, str], None]
"""Send one task request to ``base_url``; raise ``ValueError`` with the reason it is not ready."""


@dataclass(frozen=True)
class BackendTraits:
    """How to start, authenticate, and probe one backend."""

    argv: tuple[str, ...]
    validate_argv: tuple[str, ...]
    """Checks the serving environment and arguments without starting the server."""
    served_model_name: str
    api_key_child_env: str
    """Child environment variable that receives the endpoint bearer token."""
    probe: Probe
    child_environment: Mapping[str, str] = field(default_factory=dict)
    """Non-secret child environment the backend requires."""


def traits(spec: ServiceSpec) -> BackendTraits:
    """Return the traits of the backend selected by ``spec``."""
    match spec.backend:
        case VllmBackend() as backend:
            return _vllm_traits(spec, backend)
        case Gliner2Backend() as backend:
            return _gliner2_traits(spec, backend)
        case _:
            assert_never(spec.backend)


def _vllm_traits(spec: ServiceSpec, backend: VllmBackend) -> BackendTraits:
    python = spec.python_executable or VLLM_PYTHON
    argv = (python, VLLM_ENTRY, "--launch-spec-json", spec.model_dump_json())
    return BackendTraits(
        argv=argv,
        validate_argv=(*argv, "--validate"),
        served_model_name=backend.model_name,
        api_key_child_env="VLLM_API_KEY",
        probe=lambda client, base_url: _probe_chat(client, base_url, backend.model_name, backend.probe_extra_body),
    )


def _gliner2_traits(spec: ServiceSpec, backend: Gliner2Backend) -> BackendTraits:
    python = spec.python_executable or GLINER2_PYTHON
    return BackendTraits(
        argv=(python, "-m", GLINER2_MODULE, "--host", spec.host, "--port", str(spec.port)),
        validate_argv=(python, "-m", GLINER2_MODULE, "--prepare"),
        served_model_name=backend.model_name,
        api_key_child_env=GLINER2_TOKEN_ENV,
        probe=lambda client, base_url: _probe_detection(client, base_url, backend.model_name),
        child_environment={GLINER2_DEVICE_ENV: backend.device},
    )


def _probe_chat(client: httpx.Client, base_url: str, model: str, extra_body: Mapping[str, JsonValue]) -> None:
    response = client.post(
        f"{base_url}/chat/completions",
        json={
            **extra_body,
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the word ready."}],
            "max_tokens": 128,
        },
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("chat completion content must be a string")


def _probe_detection(client: httpx.Client, base_url: str, model: str) -> None:
    """Require dynamic labels, character offsets, and scores from one detection request."""
    response = client.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": _GLINER2_PROBE_TEXT}],
            "labels": ["person"],
            "threshold": 0.1,
        },
    )
    response.raise_for_status()
    entities = json.loads(response.json()["choices"][0]["message"]["content"])["entities"]
    if not isinstance(entities, list) or not entities:
        raise ValueError("detector returned no entities for the probe text")
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError("detector entities must be objects")
        if entity.get("label") != "person":
            raise ValueError(f"detector ignored the requested labels: {entity.get('label')!r}")
        start, end, score = entity.get("start"), entity.get("end"), entity.get("score")
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(_GLINER2_PROBE_TEXT)):
            raise ValueError(f"detector returned invalid offsets {start!r}..{end!r}")
        if not isinstance(score, int | float):
            raise ValueError("detector entities must carry a numeric score")
