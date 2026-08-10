# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the local vLLM Python construction boundary."""

from __future__ import annotations

import importlib
import os
import sys
import tomllib
from pathlib import Path
from unittest import mock

import pytest

TOOLS_ROOT = Path(__file__).resolve().parents[2] / "tools"
REPO_ROOT = TOOLS_ROOT.parent


def test_local_models_group_pins_vllm_0_26() -> None:
    """The characterized Python factory stays bound to the reviewed vLLM release."""
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["dependency-groups"]["local-models"] == ["vllm==0.26.0; sys_platform == 'linux'"]


def load_factory_module():
    """Load the source-tree factory without packaging it."""
    sys.path.insert(0, str(TOOLS_ROOT))
    try:
        return importlib.import_module("inference_service_compiler.vllm_factory")
    finally:
        sys.path.pop(0)


def test_parse_server_parameters_accepts_only_the_compiler_contract() -> None:
    """The process entry point accepts the bounded arguments emitted by the compiler."""
    factory = load_factory_module()

    parameters = factory.parse_server_parameters(
        [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--host",
            "127.0.0.1",
            "--port",
            "8123",
            "--revision",
            "fe8a4ea1",
            "--tokenizer-revision",
            "fe8a4ea1",
            "--served-model-name",
            "tiny",
            "--tensor-parallel-size",
            "2",
            "--gpu-memory-utilization",
            "0.8",
            "--max-model-len",
            "2048",
            "--enforce-eager",
        ]
    )

    assert parameters.model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert parameters.port == 8123
    assert parameters.tensor_parallel_size == 2
    assert parameters.gpu_memory_utilization == 0.8
    assert parameters.enforce_eager is True


def test_factory_constructs_vllm_frontend_and_async_engine_arguments() -> None:
    """The local service is built from vLLM Python config objects."""
    pytest.importorskip("vllm")
    factory = load_factory_module()
    parameters = factory.VllmServerParameters(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        host="127.0.0.1",
        port=8123,
        revision="fe8a4ea1",
        tokenizer_revision="fe8a4ea1",
        served_model_name="tiny",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        enforce_eager=True,
        lora_module="privacy=/models/privacy-adapter",
    )

    arguments = factory.build_server_arguments(parameters)

    assert arguments.model == parameters.model
    assert arguments.host == "127.0.0.1"
    assert arguments.port == 8123
    assert arguments.revision == "fe8a4ea1"
    assert arguments.tokenizer_revision == "fe8a4ea1"
    assert arguments.served_model_name == ["tiny"]
    assert arguments.tensor_parallel_size == 2
    assert arguments.gpu_memory_utilization == 0.8
    assert arguments.max_model_len == 2048
    assert arguments.enforce_eager is True
    assert arguments.enable_lora is True
    assert [(module.name, module.path) for module in arguments.lora_modules] == [("privacy", "/models/privacy-adapter")]


def test_run_server_uses_the_vllm_0_26_lifecycle_boundary() -> None:
    """The process runner imports and invokes vLLM 0.26's relocated setup API."""
    pytest.importorskip("vllm")
    uvloop = importlib.import_module("uvloop")
    api_server = importlib.import_module("vllm.entrypoints.openai.api_server")
    api_utils = importlib.import_module("vllm.entrypoints.serve.utils.api_utils")

    factory = load_factory_module()
    arguments = mock.sentinel.arguments
    coroutine = mock.sentinel.coroutine
    run_vllm_server = mock.Mock(return_value=coroutine)

    with (
        mock.patch.object(factory, "parse_server_parameters", return_value=mock.sentinel.parameters),
        mock.patch.object(factory, "build_server_arguments", return_value=arguments),
        mock.patch.object(api_utils, "cli_env_setup") as cli_env_setup,
        mock.patch.object(api_server, "run_server", new=run_vllm_server),
        mock.patch.object(uvloop, "run") as uvloop_run,
    ):
        factory.run_server(["model", "--host", "127.0.0.1", "--port", "8000"])

    cli_env_setup.assert_called_once_with()
    run_vllm_server.assert_called_once_with(arguments)
    uvloop_run.assert_called_once_with(coroutine)


def test_factory_exposes_interpreter_tools_on_path() -> None:
    """vLLM subprocess helpers can find executables installed beside Python."""
    factory = load_factory_module()

    with (
        mock.patch.object(factory.sys, "prefix", "/workspace/.venv"),
        mock.patch.dict(os.environ, {"PATH": "/usr/bin"}),
    ):
        factory.expose_interpreter_tools()

        assert os.environ["PATH"] == f"/workspace/.venv/bin{os.pathsep}/usr/bin"


def test_factory_avoids_flashinfer_jit_without_overriding_operator_choice() -> None:
    """The wheel-only runtime does not require a host CUDA compiler by default."""
    factory = load_factory_module()

    with mock.patch.dict(os.environ, {}, clear=True):
        factory.prepare_runtime_environment()
        assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"

    with mock.patch.dict(os.environ, {"VLLM_USE_FLASHINFER_SAMPLER": "1"}, clear=True):
        factory.prepare_runtime_environment()
        assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "1"
