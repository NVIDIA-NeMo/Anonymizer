# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serving-child side of the vLLM backend: launch spec to vLLM's own parser and server."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

from inference_service_compiler import vllm_compat
from inference_service_compiler.launch_ownership import LAUNCH_OWNERSHIP_MIDDLEWARE
from inference_service_compiler.models import OptionValue, ServiceSpec, VllmBackend


class ChildLaunch(NamedTuple):
    """The serving child's own arguments."""

    spec: ServiceSpec
    validate_only: bool


def option_tokens(options: Mapping[str, OptionValue]) -> list[str]:
    """Render profile options as vLLM argv tokens.

    ``True`` becomes ``--key``, ``False`` becomes ``--no-key``, a list becomes ``--key a b``, and a
    mapping becomes one JSON argument. vLLM's parser decides whether the option accepts that form.
    """
    tokens: list[str] = []
    for key, value in options.items():
        match value:
            case bool():
                tokens.append(f"--{key}" if value else f"--no-{key}")
            case list():
                tokens.extend((f"--{key}", *(str(item) for item in value)))
            case dict():
                tokens.append(f"--{key}={json.dumps(value, sort_keys=True)}")
            case _:
                tokens.append(f"--{key}={value}")
    return tokens


def serve_arguments(spec: ServiceSpec) -> list[str]:
    """Return the complete vLLM argv for a vLLM spec, tool-owned options first."""
    backend = _vllm_backend(spec)
    tokens = [f"--model={backend.model}", f"--host={spec.host}", f"--port={spec.port}"]
    if backend.revision is not None:
        tokens += [f"--revision={backend.revision}", f"--tokenizer-revision={backend.revision}"]
    if backend.served_model_name is not None:
        tokens += ["--served-model-name", backend.served_model_name]
    if backend.adapter is not None:
        tokens += ["--enable-lora", "--lora-modules", f"{backend.adapter.name}={backend.adapter.path}"]
    tokens.append(f"--middleware={LAUNCH_OWNERSHIP_MIDDLEWARE}")
    return [*tokens, *option_tokens(backend.options)]


def parse_launch_spec(argv: Sequence[str]) -> ChildLaunch:
    """Parse the child's own arguments: the launch spec and whether to stop after validation."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--launch-spec-json", required=True)
    parser.add_argument("--validate", action="store_true")
    options = parser.parse_args(argv)
    spec = ServiceSpec.model_validate_json(options.launch_spec_json)
    _vllm_backend(spec)
    return ChildLaunch(spec=spec, validate_only=options.validate)


def run_server(argv: Sequence[str]) -> None:
    """Validate the launch spec with vLLM's parser, then run the server unless ``--validate``."""
    spec, validate_only = parse_launch_spec(argv)
    expose_interpreter_tools()
    arguments = vllm_compat.make_serve_parser().parse_args(serve_arguments(spec))
    vllm_compat.validate_serve_arguments(arguments)
    if validate_only:
        return
    # vLLM 0.33 JIT-compiles FlashInfer kernels (sampling, for example) for many models, not only Mamba.
    mamba_backend = str(_vllm_backend(spec).options.get("mamba-backend", ""))
    configure_flashinfer_toolchain(required=mamba_backend.lower() == "flashinfer")
    vllm_compat.run_api_server(arguments)


def _vllm_backend(spec: ServiceSpec) -> VllmBackend:
    if not isinstance(spec.backend, VllmBackend):
        raise ValueError(f"the vLLM entry point cannot serve backend {spec.backend.kind!r}")
    return spec.backend


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
    """Find the CUDA 13 toolkit wheels (``nvidia-cuda-nvcc``) installed beside the interpreter."""
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


def configure_flashinfer_toolchain(*, required: bool) -> None:
    """Configure FlashInfer JIT from the CUDA wheels pinned with vLLM.

    Without ``CUDA_HOME`` or the wheels, raise only when the profile ``required`` FlashInfer.
    """
    configured = os.environ.get("CUDA_HOME")
    if configured:
        compiler = Path(configured) / "bin" / "nvcc"
        if not compiler.is_file():
            raise RuntimeError(f"CUDA_HOME does not contain bin/nvcc: {configured}")
        return
    cuda_root = _packaged_cuda_root()
    if cuda_root is None:
        if not required:
            return
        raise RuntimeError("FlashInfer requires a CUDA compiler; sync tools/inference-service-runtime or set CUDA_HOME")
    os.environ["CUDA_HOME"] = str(cuda_root)
    _prepend_environment_path("LIBRARY_PATH", _cudart_link_directory(cuda_root))
    _prepend_environment_path("LD_LIBRARY_PATH", cuda_root / "lib")
