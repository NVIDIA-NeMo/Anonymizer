# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for benchmark sweep specifications, arms, results, and CLI inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from measurement_tools.wandb_settings import WandbMode


class SweepSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sweep_id: str
    base_suite: str
    output_root: str | None = None
    parameters: dict[str, list[Any]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_parameters(self) -> "SweepSpec":
        empty = [name for name, values in self.parameters.items() if not values]
        if empty:
            raise ValueError(f"sweep parameter(s) must have at least one value: {', '.join(sorted(empty))}")
        return self


class SweepArm(BaseModel):
    arm_id: str
    parameters: dict[str, Any]


class SweepArmResult(BaseModel):
    arm_id: str
    parameters: dict[str, Any]
    suite_path: str
    output_dir: str
    wandb_run_name: str
    status: str
    completed_cases: int = 0
    errored_cases: int = 0
    total_cases: int = 0
    error: str | None = None


class SweepResult(BaseModel):
    sweep_id: str
    output_root: str
    arms: list[SweepArmResult]
    report_url: str | None = None
    workspace_url: str | None = None
    report_error: str | None = None
    workspace_error: str | None = None

    @property
    def completed_arms(self) -> int:
        return sum(1 for arm in self.arms if arm.status == "completed")

    @property
    def errored_arms(self) -> int:
        return sum(1 for arm in self.arms if arm.status == "error")


@dataclass(frozen=True)
class SweepCliOptions:
    spec: Path
    output_root: Path | None
    overwrite: bool
    dry_run: bool
    export: bool
    fail_fast: bool
    create_report: bool
    create_workspace: bool
    wandb_mode: WandbMode | None
    wandb_entity: str | None
    wandb_project: str | None
    wandb_base_url: str | None
    wandb_group: str | None
    wandb_job_type: str | None
    wandb_tags: str | None
    wandb_log_tables: bool | None


__all__ = ["SweepArm", "SweepArmResult", "SweepCliOptions", "SweepResult", "SweepSpec"]
