# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the local vLLM Python construction boundary."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from unittest import mock

from inference_service_compiler import vllm_runtime as factory

TOOLS_ROOT = Path(__file__).resolve().parents[2] / "tools"
REPO_ROOT = TOOLS_ROOT.parent


def test_serving_project_pins_vllm_without_anonymizer_dependencies() -> None:
    """Serving resolves independently from Anonymizer and DataDesigner."""
    project = tomllib.loads((TOOLS_ROOT / "inference-service-runtime" / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]
    assert "vllm==0.29.0" in dependencies
    assert (
        "vllm-factory[gliner] @ git+https://github.com/latenceainew/vllm-factory.git@"
        "7d6ff68ce68f9f7c0a9d72f9645bcf6d335d02f0"
    ) in dependencies
    assert {"nvidia-cuda-nvcc==13.0.88", "nvidia-cuda-crt==13.0.88", "nvidia-nvvm==13.0.88"} <= set(dependencies)
    lock = tomllib.loads((TOOLS_ROOT / "inference-service-runtime" / "uv.lock").read_text())
    assert not {"nemo-anonymizer", "data-designer"} & {package["name"] for package in lock["package"]}
    root = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert "local-models" not in root["dependency-groups"]


def test_factory_exposes_interpreter_tools_on_path() -> None:
    """vLLM subprocess helpers can find executables installed beside Python."""
    with (
        mock.patch.object(factory.sys, "prefix", "/workspace/.venv"),
        mock.patch.dict(os.environ, {"PATH": "/usr/bin"}),
    ):
        factory.expose_interpreter_tools()

        assert os.environ["PATH"] == f"/workspace/.venv/bin{os.pathsep}/usr/bin"


def test_factory_configures_packaged_cuda_for_flashinfer(tmp_path: Path) -> None:
    """The FlashInfer backend can JIT from the CUDA toolkit shipped as Python wheels."""
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
