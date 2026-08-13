# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the local vLLM Python construction boundary."""

from __future__ import annotations

import importlib
import os
import tomllib
from pathlib import Path
from unittest import mock

import pytest

from inference_service_compiler import vllm_runtime as factory

TOOLS_ROOT = Path(__file__).resolve().parents[2] / "tools"
REPO_ROOT = TOOLS_ROOT.parent


def load_factory_module():
    """Compatibility alias for the directly imported production module."""
    return factory


def test_local_models_group_pins_vllm_and_external_factory_source() -> None:
    """The runtime pins vLLM and the reviewed external factory source revision."""
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["dependency-groups"]["local-models"] == [
        "vllm==0.27.1; sys_platform == 'linux' and python_version >= '3.12'",
        (
            "vllm-factory[gliner] @ git+https://github.com/latenceainew/vllm-factory.git@"
            "7d6ff68ce68f9f7c0a9d72f9645bcf6d335d02f0; sys_platform == 'linux' "
            "and python_version >= '3.12'"
        ),
        "nvidia-cuda-nvcc==13.0.88; sys_platform == 'linux' and python_version >= '3.12'",
        "nvidia-cuda-crt==13.0.88; sys_platform == 'linux' and python_version >= '3.12'",
        "nvidia-nvvm==13.0.88; sys_platform == 'linux' and python_version >= '3.12'",
    ]


def test_parse_server_parameters_accepts_only_the_compiler_contract() -> None:
    """The process entry point accepts the bounded arguments emitted by the compiler."""
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
            "--max-num-seqs",
            "16",
            "--enforce-eager",
            "--enable-prefix-caching",
            "--async-scheduling",
            "--mamba-backend",
            "flashinfer",
            "--mamba-ssm-cache-dtype",
            "float16",
            "--enable-mamba-cache-stochastic-rounding",
            "--mamba-cache-philox-rounds",
            "5",
            "--vllm-factory-plugin",
            "deberta_gliner",
        ]
    )

    assert parameters.model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert parameters.port == 8123
    assert parameters.tensor_parallel_size == 2
    assert parameters.gpu_memory_utilization == 0.8
    assert parameters.max_num_seqs == 16
    assert parameters.enforce_eager is True
    assert parameters.enable_prefix_caching is True
    assert parameters.async_scheduling is True
    assert parameters.mamba_backend == "flashinfer"
    assert parameters.mamba_ssm_cache_dtype == "float16"
    assert parameters.enable_mamba_cache_stochastic_rounding is True
    assert parameters.mamba_cache_philox_rounds == 5
    assert parameters.vllm_factory_plugin == "deberta_gliner"


def test_factory_constructs_vllm_frontend_and_async_engine_arguments() -> None:
    """The local service is built from vLLM Python config objects."""
    pytest.importorskip("vllm")
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
        max_num_seqs=16,
        enforce_eager=True,
        enable_prefix_caching=True,
        async_scheduling=True,
        mamba_backend="flashinfer",
        mamba_ssm_cache_dtype="float16",
        enable_mamba_cache_stochastic_rounding=True,
        mamba_cache_philox_rounds=5,
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
    assert arguments.max_num_seqs == 16
    assert arguments.enforce_eager is True
    assert arguments.enable_prefix_caching is True
    assert arguments.async_scheduling is True
    assert arguments.mamba_backend.value == "flashinfer"
    assert arguments.mamba_ssm_cache_dtype == "float16"
    assert arguments.enable_mamba_cache_stochastic_rounding is True
    assert arguments.mamba_cache_philox_rounds == 5
    assert arguments.enable_lora is True
    assert [(module.name, module.path) for module in arguments.lora_modules] == [("privacy", "/models/privacy-adapter")]


def test_factory_constructs_pooling_server_for_external_gliner_plugin() -> None:
    """Factory-backed detection uses vLLM pooling and the project's IOProcessor."""
    pytest.importorskip("vllm")
    parameters = factory.VllmServerParameters(
        model="/tmp/prepared-gliner",
        host="127.0.0.1",
        port=8123,
        served_model_name="nvidia/gliner-pii",
        vllm_factory_plugin="deberta_gliner",
    )

    arguments = factory.build_server_arguments(parameters)

    assert arguments.runner == "pooling"
    assert arguments.io_processor_plugin == "deberta_gliner_io"
    assert arguments.trust_remote_code is True
    assert arguments.enable_prefix_caching is False
    assert arguments.enable_chunked_prefill is False
    assert arguments.middleware == [factory.ANONYMIZER_CHAT_MIDDLEWARE]


def test_run_server_uses_the_vllm_0_27_lifecycle_boundary() -> None:
    """The process runner imports and invokes vLLM 0.27's lifecycle API."""
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
        mock.patch.object(
            factory, "prepare_runtime_environment", return_value=mock.sentinel.prepared
        ) as prepare_environment,
        mock.patch.object(factory, "build_server_arguments", return_value=arguments) as build_arguments,
        mock.patch.object(api_utils, "cli_env_setup") as cli_env_setup,
        mock.patch.object(api_server, "run_server", new=run_vllm_server),
        mock.patch.object(uvloop, "run") as uvloop_run,
    ):
        factory.run_server(["model", "--host", "127.0.0.1", "--port", "8000"])

    cli_env_setup.assert_called_once_with()
    prepare_environment.assert_called_once_with(mock.sentinel.parameters)
    build_arguments.assert_called_once_with(mock.sentinel.prepared)
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
    parameters = factory.VllmServerParameters(model="model", host="127.0.0.1", port=8000)

    with (
        mock.patch.object(factory.sys, "version_info", (3, 12)),
        mock.patch.dict(os.environ, {}, clear=True),
    ):
        assert factory.prepare_runtime_environment(parameters) == parameters
        assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"

    with (
        mock.patch.object(factory.sys, "version_info", (3, 12)),
        mock.patch.dict(os.environ, {"VLLM_USE_FLASHINFER_SAMPLER": "1"}, clear=True),
    ):
        assert factory.prepare_runtime_environment(parameters) == parameters
        assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "1"


def test_factory_rejects_python_3_11_before_importing_vllm() -> None:
    """The server reports the local vLLM Python floor before vLLM starts."""
    factory = load_factory_module()
    parameters = factory.VllmServerParameters(model="model", host="127.0.0.1", port=8000)

    with mock.patch.object(factory.sys, "version_info", (3, 11)):
        with pytest.raises(RuntimeError, match="Python 3.12 or later"):
            factory.prepare_runtime_environment(parameters)


def test_factory_configures_packaged_cuda_for_flashinfer(tmp_path: Path) -> None:
    """The FlashInfer backend can JIT from the CUDA toolkit shipped as Python wheels."""
    factory = load_factory_module()
    cuda_root = tmp_path / "nvidia" / "cu13"
    (cuda_root / "bin").mkdir(parents=True)
    (cuda_root / "bin" / "nvcc").touch()
    (cuda_root / "lib").mkdir()
    runtime = cuda_root / "lib" / "libcudart.so.13"
    runtime.touch()

    with (
        mock.patch.object(factory, "_packaged_cuda_root", return_value=cuda_root),
        mock.patch.object(factory.Path, "home", return_value=tmp_path),
        mock.patch.dict(os.environ, {}, clear=True),
    ):
        factory.configure_flashinfer_toolchain()

        root_digest = factory.hashlib.sha256(os.fsencode(cuda_root.resolve())).hexdigest()[:16]
        link_directory = tmp_path / ".cache" / "nemo-anonymizer" / "cuda-link" / root_digest
        assert os.environ["CUDA_HOME"] == str(cuda_root)
        assert os.environ["LIBRARY_PATH"] == str(link_directory)
        assert os.environ["LD_LIBRARY_PATH"] == str(cuda_root / "lib")
        assert (link_directory / "libcudart.so").resolve() == runtime


def test_factory_selects_model_and_io_plugins_together() -> None:
    """vLLM's shared plugin allowlist retains both factory entry-point groups."""
    factory = load_factory_module()
    parameters = factory.VllmServerParameters(
        model="nvidia/gliner-pii",
        revision="bd23e8ef",
        host="127.0.0.1",
        port=8000,
        vllm_factory_plugin="deberta_gliner",
    )

    with (
        mock.patch.object(factory, "prepare_model", return_value="/tmp/prepared"),
        mock.patch.object(factory.sys, "version_info", (3, 12)),
        mock.patch.dict(os.environ, {}, clear=True),
    ):
        factory.prepare_runtime_environment(parameters)

        assert os.environ["VLLM_PLUGINS"] == "deberta_gliner,deberta_gliner_io"
