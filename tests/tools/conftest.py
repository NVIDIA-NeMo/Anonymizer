# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT_ROOT = REPO_ROOT / "tools/measurement"
RUN_BENCHMARKS_PATH = MEASUREMENT_ROOT / "run_benchmarks.py"
SWEEP_BENCHMARKS_PATH = MEASUREMENT_ROOT / "sweep_benchmarks.py"
EXECUTION_PATH = MEASUREMENT_ROOT / "measurement_tools/execution.py"
WANDB_LOGGING_PATH = MEASUREMENT_ROOT / "measurement_tools/wandb_logging.py"
WANDB_SETUP_PATH = MEASUREMENT_ROOT / "measurement_tools/wandb_setup.py"
WANDB_INGRESS_PATH = MEASUREMENT_ROOT / "measurement_tools/wandb_ingress.py"
WANDB_COMPLETION_PATH = MEASUREMENT_ROOT / "measurement_tools/wandb_completion.py"
WANDB_IMPORT_PATH = MEASUREMENT_ROOT / "import_wandb_run.py"
WANDB_REPORT_PATH = MEASUREMENT_ROOT / "create_wandb_report.py"


def _load_measurement_tool(
    load_tool: Callable[..., ModuleType],
    request: pytest.FixtureRequest,
    prefix: str,
    path: Path,
) -> ModuleType:
    safe_name = "".join(char if char.isalnum() else "_" for char in request.node.name)
    return load_tool(f"{prefix}_{safe_name}", path, additional_paths=(MEASUREMENT_ROOT,))


@pytest.fixture
def run_benchmarks_tool(load_tool: Callable[..., ModuleType], request: pytest.FixtureRequest) -> ModuleType:
    return _load_measurement_tool(load_tool, request, "measurement_run_benchmarks", RUN_BENCHMARKS_PATH)


@pytest.fixture
def sweep_tool(load_tool: Callable[..., ModuleType], request: pytest.FixtureRequest) -> ModuleType:
    return _load_measurement_tool(load_tool, request, "measurement_sweep", SWEEP_BENCHMARKS_PATH)


@pytest.fixture
def execution_tool(load_tool: Callable[..., ModuleType], request: pytest.FixtureRequest) -> ModuleType:
    return _load_measurement_tool(load_tool, request, "measurement_execution", EXECUTION_PATH)


@pytest.fixture
def wandb_logging_tool(load_tool: Callable[..., ModuleType], request: pytest.FixtureRequest) -> ModuleType:
    _load_measurement_tool(load_tool, request, "measurement_wandb_logging_prereq", RUN_BENCHMARKS_PATH)
    return _load_measurement_tool(load_tool, request, "measurement_wandb_logging", WANDB_LOGGING_PATH)


@pytest.fixture
def wandb_setup_tool(load_tool: Callable[..., ModuleType], request: pytest.FixtureRequest) -> ModuleType:
    return _load_measurement_tool(load_tool, request, "measurement_wandb_setup", WANDB_SETUP_PATH)


@pytest.fixture
def wandb_ingress_tool(load_tool: Callable[..., ModuleType], request: pytest.FixtureRequest) -> ModuleType:
    return _load_measurement_tool(load_tool, request, "measurement_wandb_ingress", WANDB_INGRESS_PATH)


@pytest.fixture
def wandb_completion_tool(load_tool: Callable[..., ModuleType], request: pytest.FixtureRequest) -> ModuleType:
    return _load_measurement_tool(load_tool, request, "measurement_wandb_completion", WANDB_COMPLETION_PATH)


@pytest.fixture
def wandb_import_tool(load_tool: Callable[..., ModuleType], request: pytest.FixtureRequest) -> ModuleType:
    return _load_measurement_tool(load_tool, request, "measurement_wandb_import", WANDB_IMPORT_PATH)


@pytest.fixture
def wandb_report_tool(load_tool: Callable[..., ModuleType], request: pytest.FixtureRequest) -> ModuleType:
    return _load_measurement_tool(load_tool, request, "measurement_wandb_report", WANDB_REPORT_PATH)
