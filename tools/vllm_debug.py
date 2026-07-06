#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch and probe local vLLM servers for Anonymizer development.

Usage:
    uv run python tools/vllm_debug.py models --cached
    uv run --with vllm python tools/vllm_debug.py serve /path/to/model --dry-run
    uv run python tools/vllm_debug.py serve /path/to/model --vllm-python /opt/vllm/bin/python
    uv run python tools/vllm_debug.py models --endpoint http://127.0.0.1:8000/v1
    uv run python tools/vllm_debug.py call --model local-model --prompt "Hello"

The helper does not prefetch models. ``serve`` requires vLLM to be installed in
the Python environment that launches this script. Use a cached snapshot path
from ``models --cached`` to avoid a Hugging Face download; a model ID may cause
vLLM to download it.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import cyclopts
import httpx
from pydantic import BaseModel

app = cyclopts.App(help=__doc__)

DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"


class ServeRequest(BaseModel):
    """Arguments for an OpenAI-compatible vLLM server."""

    model: str
    host: str = "127.0.0.1"
    port: int = 8000
    adapter: Path | None = None
    adapter_name: str | None = None
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    eager: bool = False
    python_executable: Path | None = None
    served_model_name: str | None = None
    api_key: str | None = None


class CachedModel(BaseModel):
    """A Hugging Face cache snapshot usable as a local vLLM model path."""

    repository: str
    snapshot_path: Path


class CallResult(BaseModel):
    """Normalized OpenAI-compatible chat response."""

    content: str
    usage: dict[str, Any]


def build_serve_command(request: ServeRequest) -> list[str]:
    """Build the vLLM server command without starting a process."""
    command = [
        str(request.python_executable or sys.executable),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        request.model,
        "--host",
        request.host,
        "--port",
        str(request.port),
    ]
    if request.served_model_name is not None:
        command.extend(["--served-model-name", request.served_model_name])
    if request.api_key is not None:
        command.extend(["--api-key", request.api_key])
    _append_adapter(command, request)
    _append_gpu_options(command, request)
    return command


def render_serve_command(command: list[str]) -> str:
    """Render a server command without exposing its API key."""
    rendered = command.copy()
    if "--api-key" in rendered:
        key_index = rendered.index("--api-key") + 1
        if key_index < len(rendered):
            rendered[key_index] = "<redacted>"
    return shlex.join(rendered)


def _append_adapter(command: list[str], request: ServeRequest) -> None:
    if request.adapter is None:
        return
    adapter_name = request.adapter_name or request.adapter.name
    command.extend(["--enable-lora", "--lora-modules", f"{adapter_name}={request.adapter}"])


def _append_gpu_options(command: list[str], request: ServeRequest) -> None:
    options = [
        ("--tensor-parallel-size", request.tensor_parallel_size),
        ("--gpu-memory-utilization", request.gpu_memory_utilization),
        ("--max-model-len", request.max_model_len),
    ]
    for flag, value in options:
        if value is not None:
            command.extend([flag, str(value)])
    if request.eager:
        command.append("--enforce-eager")


def discover_cached_models(cache_root: Path) -> list[CachedModel]:
    """Return all snapshot directories in a Hugging Face hub cache."""
    if not cache_root.exists():
        return []
    models: list[CachedModel] = []
    for model_dir in sorted(cache_root.glob("models--*")):
        repository = model_dir.name.removeprefix("models--").replace("--", "/")
        for snapshot in sorted((model_dir / "snapshots").glob("*")):
            if snapshot.is_dir():
                models.append(CachedModel(repository=repository, snapshot_path=snapshot))
    return models


def default_cache_root() -> Path:
    """Resolve the Hugging Face hub cache without creating it."""
    if hub_cache := os.getenv("HF_HUB_CACHE"):
        return Path(hub_cache)
    if hf_home := os.getenv("HF_HOME"):
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def fetch_models(endpoint: str, *, timeout_seconds: float) -> list[str]:
    """Fetch model IDs from an OpenAI-compatible ``/v1/models`` endpoint."""
    response = httpx.get(f"{normalize_endpoint(endpoint)}/models", timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    return [str(item["id"]) for item in payload.get("data", []) if isinstance(item, dict) and "id" in item]


def call_chat(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    timeout_seconds: float,
    api_key: str | None,
) -> CallResult:
    """Send a single chat completion request and normalize the response."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    response = httpx.post(
        f"{normalize_endpoint(endpoint)}/chat/completions",
        json=payload,
        timeout=timeout_seconds,
        headers=headers,
    )
    response.raise_for_status()
    body = response.json()
    content = str(body["choices"][0]["message"].get("content", ""))
    usage = body.get("usage", {})
    return CallResult(content=content, usage=usage if isinstance(usage, dict) else {})


def normalize_endpoint(endpoint: str) -> str:
    """Ensure an endpoint has one ``/v1`` suffix and no trailing slash."""
    stripped = endpoint.rstrip("/")
    return stripped if stripped.endswith("/v1") else f"{stripped}/v1"


def resolve_api_key(api_key: str | None, api_key_env: str | None) -> str | None:
    """Return an explicit API key or one read from a named environment variable."""
    if api_key is not None and api_key_env is not None:
        raise ValueError("Use either api_key or api_key_env, not both.")
    if api_key_env is None:
        return api_key
    value = os.getenv(api_key_env)
    if value is None:
        raise ValueError(f"Environment variable {api_key_env!r} is not set.")
    return value


def render(value: BaseModel | list[str] | list[CachedModel], *, json_output: bool) -> str:
    """Render command results in JSON or compact human-readable text."""
    if json_output:
        return json.dumps(_json_value(value), indent=2)
    if isinstance(value, list):
        return "\n".join(str(item) for item in value) or "No models found."
    return value.model_dump_json(indent=2)


def _json_value(value: BaseModel | list[str] | list[CachedModel]) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return [item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in value]


@app.command
def serve(
    model: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    adapter: Path | None = None,
    adapter_name: str | None = None,
    tensor_parallel_size: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    eager: bool = False,
    vllm_python: Annotated[Path | None, cyclopts.Parameter("--vllm-python")] = None,
    served_model_name: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    dry_run: Annotated[bool, cyclopts.Parameter("--dry-run")] = False,
) -> None:
    """Launch an OpenAI-compatible vLLM server from the current Python environment."""
    request = ServeRequest(
        model=model,
        host=host,
        port=port,
        adapter=adapter,
        adapter_name=adapter_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        eager=eager,
        python_executable=vllm_python,
        served_model_name=served_model_name,
        api_key=resolve_api_key(api_key, api_key_env),
    )
    command = build_serve_command(request)
    if dry_run:
        print(render_serve_command(command))
        return
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"vLLM Python executable not found: {command[0]}") from exc


@app.command
def models(
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    cached: Annotated[bool, cyclopts.Parameter("--cached")] = False,
    cache_root: Path | None = None,
    timeout_seconds: float = 10.0,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
) -> None:
    """List served models or cached Hugging Face snapshots."""
    if cached:
        print(render(discover_cached_models(cache_root or default_cache_root()), json_output=json_output))
        return
    print(render(fetch_models(endpoint, timeout_seconds=timeout_seconds), json_output=json_output))


@app.command
def call(
    model: str,
    prompt: str,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    api_key_env: str | None = None,
    timeout_seconds: float = 60.0,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
) -> None:
    """Send one OpenAI-compatible chat completion request."""
    api_key = resolve_api_key(None, api_key_env)
    result = call_chat(
        endpoint=endpoint,
        model=model,
        prompt=prompt,
        timeout_seconds=timeout_seconds,
        api_key=api_key,
    )
    print(render(result, json_output=json_output))


if __name__ == "__main__":
    app()
