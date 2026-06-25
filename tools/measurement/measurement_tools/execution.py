#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared execution-context metadata for benchmark measurement tooling."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

_SLURM_ENV_NAMES = frozenset(
    {
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_ARRAY_TASK_COUNT",
        "SLURM_RESTART_COUNT",
        "SLURM_NTASKS",
        "SLURM_JOB_NUM_NODES",
    }
)


class BenchmarkExecutionBackend(StrEnum):
    """Execution backend observed by benchmark tooling."""

    local = "local"
    slurm = "slurm"


class SlurmExecutionMetadata(BaseModel):
    """Sanitized Slurm metadata safe for benchmark records and W&B config."""

    model_config = ConfigDict(extra="forbid")

    job_id: str | None = None
    array_job_id: str | None = None
    array_task_id: str | None = None
    array_task_count: int | None = None
    restart_count: int | None = None
    ntasks: int | None = None
    job_num_nodes: int | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> SlurmExecutionMetadata | None:
        """Build sanitized Slurm metadata from environment variables, if present."""
        if not any(_env_str(env, name) is not None for name in _SLURM_ENV_NAMES):
            return None
        return cls(
            job_id=_env_str(env, "SLURM_JOB_ID"),
            array_job_id=_env_str(env, "SLURM_ARRAY_JOB_ID"),
            array_task_id=_env_str(env, "SLURM_ARRAY_TASK_ID"),
            array_task_count=_env_int(env, "SLURM_ARRAY_TASK_COUNT"),
            restart_count=_env_int(env, "SLURM_RESTART_COUNT"),
            ntasks=_env_int(env, "SLURM_NTASKS"),
            job_num_nodes=_env_int(env, "SLURM_JOB_NUM_NODES"),
        )


class BenchmarkExecutionContext(BaseModel):
    """Sanitized benchmark execution context shared by local and Slurm runs."""

    model_config = ConfigDict(extra="forbid")

    backend: BenchmarkExecutionBackend
    output_dir_hash: str
    export: bool
    fail_fast: bool
    dd_trace: str
    dd_task_trace: bool
    slurm: SlurmExecutionMetadata | None = None


def build_benchmark_execution_context(
    *,
    output_dir: str | Path,
    export: bool,
    fail_fast: bool,
    dd_trace: str,
    dd_task_trace: bool,
    env: Mapping[str, str] | None = None,
) -> BenchmarkExecutionContext:
    """Build sanitized execution metadata for benchmark summaries and W&B config."""
    environment = os.environ if env is None else env
    slurm = SlurmExecutionMetadata.from_env(environment)
    return BenchmarkExecutionContext(
        backend=BenchmarkExecutionBackend.slurm if slurm is not None else BenchmarkExecutionBackend.local,
        output_dir_hash=stable_metadata_hash(str(output_dir)),
        export=export,
        fail_fast=fail_fast,
        dd_trace=dd_trace,
        dd_task_trace=dd_task_trace,
        slurm=slurm,
    )


def benchmark_execution_metadata(
    *,
    output_dir: str | Path,
    export: bool,
    fail_fast: bool,
    dd_trace: str,
    dd_task_trace: bool,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable benchmark execution metadata."""
    return build_benchmark_execution_context(
        output_dir=output_dir,
        export=export,
        fail_fast=fail_fast,
        dd_trace=dd_trace,
        dd_task_trace=dd_task_trace,
        env=env,
    ).model_dump(mode="json", exclude_none=True)


def stable_metadata_hash(value: str) -> str:
    """Return a short stable hash for path-like or otherwise sensitive metadata."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _env_str(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _env_int(env: Mapping[str, str], name: str) -> int | None:
    value = _env_str(env, name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
