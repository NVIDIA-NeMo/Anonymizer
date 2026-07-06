# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the local vLLM debug helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "vllm_debug.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vllm_debug_tool", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_serve_command_includes_lora_and_gpu_options() -> None:
    tool = load_tool()

    command = tool.build_serve_command(
        tool.ServeRequest(
            model="HuggingFaceTB/SmolLM3-3B",
            host="127.0.0.1",
            port=8000,
            served_model_name="anonymizer-local",
            api_key="test-token",
            adapter=Path("/models/adapter"),
            adapter_name="anonymizer",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            max_model_len=4096,
            eager=True,
        )
    )

    assert command == [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "HuggingFaceTB/SmolLM3-3B",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--served-model-name",
        "anonymizer-local",
        "--api-key",
        "test-token",
        "--enable-lora",
        "--lora-modules",
        "anonymizer=/models/adapter",
        "--tensor-parallel-size",
        "2",
        "--gpu-memory-utilization",
        "0.8",
        "--max-model-len",
        "4096",
        "--enforce-eager",
    ]


def test_build_serve_command_can_use_a_separate_vllm_interpreter() -> None:
    tool = load_tool()

    command = tool.build_serve_command(
        tool.ServeRequest(
            model="local-model",
            python_executable=Path("/opt/vllm/bin/python"),
        )
    )

    assert command[:5] == [
        "/opt/vllm/bin/python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "local-model",
    ]


def test_render_serve_command_redacts_api_key() -> None:
    tool = load_tool()

    rendered = tool.render_serve_command(
        tool.build_serve_command(tool.ServeRequest(model="local-model", api_key="test-token"))
    )

    assert "test-token" not in rendered
    assert "--api-key '<redacted>'" in rendered


def test_resolve_api_key_reads_a_named_environment_variable(monkeypatch: Any) -> None:
    tool = load_tool()
    monkeypatch.setenv("LOCAL_VLLM_API_KEY", "test-token")

    assert tool.resolve_api_key(None, "LOCAL_VLLM_API_KEY") == "test-token"


def test_discover_cached_models_returns_snapshot_paths(tmp_path: Path) -> None:
    tool = load_tool()
    snapshot = tmp_path / "models--HuggingFaceTB--SmolLM3-3B" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    models = tool.discover_cached_models(tmp_path)

    assert models == [
        tool.CachedModel(
            repository="HuggingFaceTB/SmolLM3-3B",
            snapshot_path=snapshot,
        )
    ]


def test_fetch_models_uses_openai_models_endpoint(monkeypatch: Any) -> None:
    tool = load_tool()
    calls: list[str] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "local-model"}]}

    def fake_get(url: str, *, timeout: float) -> Response:
        calls.append(url)
        assert timeout == 10.0
        return Response()

    monkeypatch.setattr(tool.httpx, "get", fake_get)

    assert tool.fetch_models("http://127.0.0.1:8000/v1", timeout_seconds=10.0) == ["local-model"]
    assert calls == ["http://127.0.0.1:8000/v1/models"]


def test_call_chat_sends_prompt_and_returns_content(monkeypatch: Any) -> None:
    tool = load_tool()
    request_body: dict[str, object] = {}

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }

    def fake_post(url: str, *, json: dict[str, object], timeout: float, headers: dict[str, str]) -> Response:
        request_body.update(json)
        assert url == "http://127.0.0.1:8000/v1/chat/completions"
        assert timeout == 15.0
        assert headers == {}
        return Response()

    monkeypatch.setattr(tool.httpx, "post", fake_post)

    result = tool.call_chat(
        endpoint="http://127.0.0.1:8000/v1",
        model="local-model",
        prompt="Say hello",
        timeout_seconds=15.0,
        api_key=None,
    )

    assert request_body == {
        "model": "local-model",
        "messages": [{"role": "user", "content": "Say hello"}],
    }
    assert result.content == "hello"
    assert result.usage == {"prompt_tokens": 3, "completion_tokens": 1}
