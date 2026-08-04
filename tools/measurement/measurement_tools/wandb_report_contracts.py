# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated result contracts returned by W&B report tooling."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from measurement_tools.wandb_report_text import validate_output_text, validate_output_url


class _WandbOutputValidation:
    @field_validator("project_path", "group", "title")
    @classmethod
    def validate_output_text(cls, value: str | None) -> str | None:
        return validate_output_text(value)


class WandbReportResult(_WandbOutputValidation, BaseModel):
    run_path: str | None = None
    run_url: str | None = None
    project_path: str
    group: str | None = None
    report_url: str
    draft: bool
    title: str

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("run_url", "report_url")
    @classmethod
    def validate_output_urls(cls, value: str | None) -> str | None:
        return validate_output_url(value)


class WandbWorkspaceResult(_WandbOutputValidation, BaseModel):
    project_path: str
    group: str | None = None
    workspace_url: str
    title: str

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("workspace_url")
    @classmethod
    def validate_output_urls(cls, value: str) -> str:
        validated = validate_output_url(value)
        if validated is None:
            raise ValueError("W&B workspace URL is required")
        return validated


__all__ = ["WandbReportResult", "WandbWorkspaceResult"]
