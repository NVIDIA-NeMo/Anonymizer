# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Programmatic construction of stock vLLM and vLLM Factory servers."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path

from inference_service_compiler.lifecycle import LAUNCH_OWNERSHIP_MIDDLEWARE
from inference_service_compiler.models import parse_factory_plugin
from inference_service_compiler.vllm_factory_integration import (
    io_processor_for,
    prepare_model,
)

ANONYMIZER_CHAT_MIDDLEWARE = "inference_service_compiler.vllm_factory_adapter.anonymizer_chat_compatibility"
MINIMUM_VLLM_PYTHON = (3, 12)


def expose_interpreter_tools() -> None:
    """Expose console tools installed beside the selected Python interpreter."""
    interpreter_bin = str(Path(sys.prefix) / "bin")
    current_path = os.environ.get("PATH", "")
    path_entries = [entry for entry in current_path.split(os.pathsep) if entry != interpreter_bin]
    os.environ["PATH"] = os.pathsep.join([interpreter_bin, *path_entries])


def _prepend_environment_path(name: str, path: Path) -> None:
    current = os.environ.get(name, "")
    entries = [entry for entry in current.split(os.pathsep) if entry and entry != str(path)]
    os.environ[name] = os.pathsep.join([str(path), *entries])


def _packaged_cuda_root() -> Path | None:
    """Find the CUDA toolkit installed beside the selected Python interpreter."""
    package = importlib.util.find_spec("nvidia")
    if package is None or package.submodule_search_locations is None:
        return None
    for location in package.submodule_search_locations:
        candidate = Path(location) / "cu13"
        if (candidate / "bin" / "nvcc").is_file() and (candidate / "lib").is_dir():
            return candidate
    return None


def _cudart_link_directory(cuda_root: Path) -> Path:
    """Expose the versioned CUDA wheel runtime under the linker name FlashInfer expects."""
    library_directory = cuda_root / "lib"
    unversioned = library_directory / "libcudart.so"
    if unversioned.exists():
        return library_directory
    candidates = sorted(library_directory.glob("libcudart.so.*"))
    if not candidates:
        raise RuntimeError(f"CUDA toolkit at {cuda_root} does not contain libcudart")
    root_digest = hashlib.sha256(os.fsencode(cuda_root.resolve())).hexdigest()[:16]
    link_directory = Path.home() / ".cache" / "nemo-anonymizer" / "cuda-link" / root_digest
    link_directory.mkdir(parents=True, exist_ok=True)
    link = link_directory / "libcudart.so"
    target = candidates[-1].resolve()
    if link.is_symlink() and link.resolve() == target:
        return link_directory
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(target)
    except FileExistsError:
        if not link.is_symlink() or link.resolve() != target:
            raise
    return link_directory


def configure_flashinfer_toolchain() -> None:
    """Configure FlashInfer JIT from the CUDA wheels pinned with vLLM."""
    configured = os.environ.get("CUDA_HOME")
    if configured:
        compiler = Path(configured) / "bin" / "nvcc"
        if not compiler.is_file():
            raise RuntimeError(f"CUDA_HOME does not contain bin/nvcc: {configured}")
        return
    cuda_root = _packaged_cuda_root()
    if cuda_root is None:
        raise RuntimeError(
            "FlashInfer Mamba requires a CUDA compiler; sync tools/inference-service-runtime or set CUDA_HOME"
        )
    os.environ["CUDA_HOME"] = str(cuda_root)
    _prepend_environment_path("LIBRARY_PATH", _cudart_link_directory(cuda_root))
    _prepend_environment_path("LD_LIBRARY_PATH", cuda_root / "lib")


def load_and_validate_arguments(options: Namespace) -> Namespace:
    """Initialize plugin discovery and delegate configuration parsing to vLLM."""
    config = json.loads(options.config_json)
    plugin = parse_factory_plugin(options.vllm_factory_plugin) if options.vllm_factory_plugin else None
    if sys.version_info < MINIMUM_VLLM_PYTHON:
        raise RuntimeError("vLLM 0.29.0 local serving requires Python 3.12 or later")
    expose_interpreter_tools()
    if plugin is not None:
        os.environ["VLLM_PLUGINS"] = f"{plugin},{io_processor_for(plugin)}"
        os.environ["ANONYMIZER_VLLM_FACTORY_PLUGIN"] = plugin
    middleware = config.pop("middleware", [])
    if isinstance(middleware, str):
        middleware = [middleware]
    if not isinstance(middleware, list) or not all(isinstance(item, str) for item in middleware):
        raise ValueError("middleware must be a string or list of strings")
    middleware = [*middleware, LAUNCH_OWNERSHIP_MIDDLEWARE]
    if plugin is not None:
        middleware.append(ANONYMIZER_CHAT_MIDDLEWARE)
    cli_args = importlib.import_module("vllm.entrypoints.launchers.cli_args")
    parser_class = importlib.import_module("vllm.utils.argparse_utils").FlexibleArgumentParser
    parser = cli_args.make_arg_parser(parser_class())
    # The upstream YAML loader drops empty lists and some false values.
    empty_args = []
    for key, value in config.items():
        option = f"--{key.replace('_', '-')}"
        if value is False:
            action = parser._option_string_actions.get(option)
            if action is None:
                raise ValueError(f"unknown vLLM option {key!r}")
            if f"--no-{key.replace('_', '-')}" not in parser._option_string_actions and action.default is not False:
                raise ValueError(f"vLLM cannot express false for option {key!r}")
        elif value == []:
            action = parser._option_string_actions.get(option)
            if action is None:
                raise ValueError(f"unknown vLLM option {key!r}")
            if (
                not isinstance(action, argparse._StoreAction)
                or action.type is not json.loads
                or action.nargs is not None
            ):
                raise ValueError(f"vLLM cannot express an empty list for option {key!r}")
            empty_args.extend((option, "[]"))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8") as config_file:
        json.dump(config, config_file)
        config_file.flush()
        middleware_args = [token for item in middleware for token in ("--middleware", item)]
        arguments = parser.parse_args(["--config", config_file.name, *empty_args, *middleware_args])
    if arguments.model_tag is not None:
        arguments.model = arguments.model_tag
    cli_args.validate_parsed_serve_args(arguments)
    return arguments


def run_server(argv: Sequence[str]) -> None:
    """Validate with vLLM's parser, then optionally prepare and run the server."""
    bootstrap = argparse.ArgumentParser()
    bootstrap.add_argument("--config-json", required=True)
    bootstrap.add_argument("--vllm-factory-plugin", choices=("deberta_gliner", "deberta_gliner2"))
    bootstrap.add_argument("--prepared-model-root", default="/tmp/anonymizer-vllm-factory")
    bootstrap.add_argument("--validate", action="store_true")
    options = bootstrap.parse_args(argv)
    arguments = load_and_validate_arguments(options)
    if options.validate:
        return
    if getattr(arguments.mamba_backend, "value", arguments.mamba_backend) == "flashinfer":
        configure_flashinfer_toolchain()
    if options.vllm_factory_plugin is not None:
        plugin = parse_factory_plugin(options.vllm_factory_plugin)
        public_name = arguments.served_model_name or [arguments.model]
        arguments.model = prepare_model(
            model_id=arguments.model,
            revision=arguments.revision,
            plugin=plugin,
            prepared_model_root=options.prepared_model_root,
        )
        arguments.revision = None
        arguments.tokenizer_revision = None
        arguments.served_model_name = public_name
    uvloop = importlib.import_module("uvloop")
    api_server = importlib.import_module("vllm.entrypoints.launchers.api_server.entry")
    api_utils = importlib.import_module("vllm.entrypoints.serve.utils.api_utils")
    api_utils.cli_env_setup()
    uvloop.run(api_server.run_server(arguments))
