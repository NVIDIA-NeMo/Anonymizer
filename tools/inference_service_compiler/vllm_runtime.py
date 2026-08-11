# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Programmatic construction of stock vLLM and vLLM Factory servers."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from inference_service_compiler.vllm_factory_integration import (
    io_processor_for,
    prepare_model,
)

ANONYMIZER_CHAT_MIDDLEWARE = "inference_service_compiler.vllm_factory_adapter.anonymizer_chat_compatibility"


@dataclass(frozen=True)
class VllmServerParameters:
    """Bounded settings accepted by the source-owned vLLM server process."""

    model: str
    host: str
    port: int
    revision: str | None = None
    tokenizer_revision: str | None = None
    served_model_name: str | None = None
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    enforce_eager: bool = False
    lora_module: str | None = None
    vllm_factory_plugin: str | None = None
    prepared_model_root: str = "/tmp/anonymizer-vllm-factory"


def parse_server_parameters(argv: Sequence[str]) -> VllmServerParameters:
    """Parse the compiler's bounded process contract without using vLLM's CLI."""
    parser = argparse.ArgumentParser(description="Anonymizer-managed vLLM server")
    parser.add_argument("model")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--revision")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--served-model-name")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-lora", action="store_true")
    parser.add_argument("--lora-modules")
    parser.add_argument(
        "--vllm-factory-plugin",
        choices=("deberta_gliner", "deberta_gliner2"),
    )
    parser.add_argument("--prepared-model-root", default="/tmp/anonymizer-vllm-factory")
    parsed = parser.parse_args(list(argv))
    if parsed.enable_lora != (parsed.lora_modules is not None):
        parser.error("--enable-lora and --lora-modules must be used together")
    return VllmServerParameters(
        model=parsed.model,
        host=parsed.host,
        port=parsed.port,
        revision=parsed.revision,
        tokenizer_revision=parsed.tokenizer_revision,
        served_model_name=parsed.served_model_name,
        tensor_parallel_size=parsed.tensor_parallel_size,
        gpu_memory_utilization=parsed.gpu_memory_utilization,
        max_model_len=parsed.max_model_len,
        enforce_eager=parsed.enforce_eager,
        lora_module=parsed.lora_modules,
        vllm_factory_plugin=parsed.vllm_factory_plugin,
        prepared_model_root=parsed.prepared_model_root,
    )


def build_server_arguments(parameters: VllmServerParameters) -> Namespace:
    """Construct vLLM frontend and async-engine configs through its Python API."""
    arg_utils = importlib.import_module("vllm.engine.arg_utils")
    cli_args = importlib.import_module("vllm.entrypoints.openai.cli_args")
    model_protocol = importlib.import_module("vllm.entrypoints.openai.models.protocol")

    lora_modules = None
    if parameters.lora_module is not None:
        name, separator, path = parameters.lora_module.partition("=")
        if not separator or not name or not path:
            raise ValueError("LoRA module must use the form NAME=PATH")
        lora_modules = [model_protocol.LoRAModulePath(name=name, path=path)]

    factory_plugin = parameters.vllm_factory_plugin
    engine = arg_utils.AsyncEngineArgs(
        model=parameters.model,
        revision=parameters.revision,
        tokenizer_revision=parameters.tokenizer_revision,
        served_model_name=[parameters.served_model_name] if parameters.served_model_name is not None else None,
        tensor_parallel_size=parameters.tensor_parallel_size or 1,
        gpu_memory_utilization=parameters.gpu_memory_utilization or 0.9,
        enforce_eager=parameters.enforce_eager,
        enable_lora=lora_modules is not None,
        runner="pooling" if factory_plugin is not None else "auto",
        trust_remote_code=factory_plugin is not None,
        dtype="bfloat16" if factory_plugin is not None else "auto",
        enable_prefix_caching=False if factory_plugin is not None else None,
        enable_chunked_prefill=False if factory_plugin is not None else None,
        io_processor_plugin=io_processor_for(factory_plugin) if factory_plugin is not None else None,
    )
    if parameters.max_model_len is not None:
        engine.max_model_len = parameters.max_model_len
    frontend = cli_args.FrontendArgs(
        host=parameters.host,
        port=parameters.port,
        lora_modules=lora_modules,
        middleware=[ANONYMIZER_CHAT_MIDDLEWARE] if factory_plugin is not None else [],
    )
    values = vars(engine) | vars(frontend)
    values.update(
        model_tag=None,
        headless=False,
        api_server_count=1,
        config=None,
        grpc=False,
    )
    arguments = Namespace(**values)
    cli_args.validate_parsed_serve_args(arguments)
    return arguments


def expose_interpreter_tools() -> None:
    """Expose console tools installed beside the selected Python interpreter."""
    interpreter_bin = str(Path(sys.prefix) / "bin")
    current_path = os.environ.get("PATH", "")
    path_entries = [entry for entry in current_path.split(os.pathsep) if entry != interpreter_bin]
    os.environ["PATH"] = os.pathsep.join([interpreter_bin, *path_entries])


def prepare_runtime_environment(parameters: VllmServerParameters) -> VllmServerParameters:
    """Prepare the selected stock vLLM or vLLM Factory runtime."""
    expose_interpreter_tools()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    plugin = parameters.vllm_factory_plugin
    if plugin is None:
        return parameters
    os.environ["VLLM_PLUGINS"] = f"{plugin},{io_processor_for(plugin)}"
    os.environ["ANONYMIZER_VLLM_FACTORY_PLUGIN"] = plugin
    prepared_model = prepare_model(
        model_id=parameters.model,
        revision=parameters.revision,
        plugin=plugin,
        prepared_model_root=parameters.prepared_model_root,
    )
    public_name = parameters.served_model_name or parameters.model
    return replace(
        parameters,
        model=prepared_model,
        revision=None,
        tokenizer_revision=None,
        served_model_name=public_name,
    )


def run_server(argv: Sequence[str]) -> None:
    """Construct and run vLLM's Python-owned server lifecycle."""
    parameters = prepare_runtime_environment(parse_server_parameters(argv))
    uvloop = importlib.import_module("uvloop")
    api_server = importlib.import_module("vllm.entrypoints.openai.api_server")
    api_utils = importlib.import_module("vllm.entrypoints.serve.utils.api_utils")

    api_utils.cli_env_setup()
    arguments = build_server_arguments(parameters)
    uvloop.run(api_server.run_server(arguments))
