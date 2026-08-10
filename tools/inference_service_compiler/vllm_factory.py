# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Programmatic construction of vLLM's OpenAI-compatible server."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


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


def parse_server_parameters(argv: Sequence[str]) -> VllmServerParameters:
    """Parse the compiler's bounded process contract without using vLLM's CLI."""
    parser = argparse.ArgumentParser(description="Anonymizer-managed vLLM OpenAI server")
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

    engine = arg_utils.AsyncEngineArgs(
        model=parameters.model,
        revision=parameters.revision,
        tokenizer_revision=parameters.tokenizer_revision,
        served_model_name=[parameters.served_model_name] if parameters.served_model_name is not None else None,
        tensor_parallel_size=parameters.tensor_parallel_size or 1,
        gpu_memory_utilization=parameters.gpu_memory_utilization or 0.9,
        enforce_eager=parameters.enforce_eager,
        enable_lora=lora_modules is not None,
    )
    if parameters.max_model_len is not None:
        engine.max_model_len = parameters.max_model_len
    frontend = cli_args.FrontendArgs(
        host=parameters.host,
        port=parameters.port,
        lora_modules=lora_modules,
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


def prepare_runtime_environment() -> None:
    """Prepare a wheel-only vLLM runtime without requiring a host CUDA compiler."""
    expose_interpreter_tools()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def run_server(argv: Sequence[str]) -> None:
    """Construct and run vLLM's Python-owned OpenAI server lifecycle."""
    prepare_runtime_environment()
    uvloop = importlib.import_module("uvloop")
    api_server = importlib.import_module("vllm.entrypoints.openai.api_server")
    api_utils = importlib.import_module("vllm.entrypoints.serve.utils.api_utils")

    api_utils.cli_env_setup()
    arguments = build_server_arguments(parse_server_parameters(argv))
    uvloop.run(api_server.run_server(arguments))
