# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W&B benchmark workspace orchestration, panels, and persistence."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from measurement_tools.wandb_models import ResolvedWandbConfig
from measurement_tools.wandb_report_catalog import (
    _MEASUREMENT_TABLE_KEYS,
    _WORKSPACE_BAR_SECTIONS,
    _WORKSPACE_COMPARISON_COLUMNS,
    _WORKSPACE_JOB_FILTERS,
    _WORKSPACE_SUMMARY_SCALARS,
    bar_panel,
    metric_title,
)
from measurement_tools.wandb_report_contracts import WandbWorkspaceResult
from measurement_tools.wandb_report_models import (
    GroupComparison,
    WandbProjectPath,
    group_comparison,
    validate_wandb_returned_url,
)
from measurement_tools.wandb_report_sdk import (
    read_group_views,
    report_settings,
    require_wandb_report_sdk,
    require_wandb_workspace_sdk,
)
from measurement_tools.wandb_report_text import plain_text
from measurement_tools.wandb_setup import WandbSdkEnvironment


def create_benchmark_workspace(
    project_path: WandbProjectPath,
    *,
    settings: ResolvedWandbConfig | None = None,
    group: str | None = None,
    title: str | None = None,
    expected_run_kind: Literal["native_suite", "sweep_arm", "imported_case"] | None = None,
    report_sdk_loader: Callable[[], tuple[Any, Any, Any]] = require_wandb_report_sdk,
    workspace_sdk_loader: Callable[[], tuple[Any, Any]] = require_wandb_workspace_sdk,
    workspace_builder: Callable[..., Any] | None = None,
    workspace_saver: Callable[[Any], Any] | None = None,
) -> WandbWorkspaceResult:
    """Create a W&B workspace for benchmark runs in a project."""
    builder = workspace_builder or build_benchmark_workspace
    saver = workspace_saver or save_workspace
    resolved = report_settings(settings, project_path)
    effective_project_path = project_path.with_base_url(resolved.wandb_base_url or project_path.base_url)
    with WandbSdkEnvironment(resolved):
        comparison: GroupComparison | None = None
        if group is not None:
            wandb, _wr, _expr = report_sdk_loader()
            views = read_group_views(wandb, project_path=effective_project_path, group=group)
            comparison = group_comparison(views, expected_run_kind=expected_run_kind)
        ws, workspace_wr = workspace_sdk_loader()
        workspace_title = plain_text(title) if title is not None else default_workspace_title(group=group)
        workspace = builder(
            effective_project_path,
            group=group,
            title=workspace_title,
            comparison_config_key=comparison.config_key if comparison is not None else "sweep_arm_id",
            ws=ws,
            wr=workspace_wr,
        )
        saved = saver(workspace)
        workspace_url = validate_wandb_returned_url(saved.url, expected_base_url=effective_project_path.base_url)
    return WandbWorkspaceResult(
        project_path=project_path.path,
        group=group,
        workspace_url=workspace_url,
        title=workspace_title,
    )


def build_benchmark_workspace(
    project_path: WandbProjectPath,
    *,
    group: str | None,
    title: str,
    comparison_config_key: str = "sweep_arm_id",
    ws: Any,
    wr: Any,
) -> Any:
    """Build an unsaved manual W&B workspace for benchmark runs."""
    groupby = wr.Config(comparison_config_key) if group else None
    return ws.Workspace(
        entity=project_path.entity,
        project=project_path.project,
        name=title,
        auto_generate_panels=False,
        settings=benchmark_workspace_settings(ws),
        runset_settings=benchmark_runset_settings(
            ws,
            group=group,
            comparison_config_key=comparison_config_key,
        ),
        sections=benchmark_workspace_sections(
            ws,
            wr,
            groupby=groupby,
            comparison_config_key=comparison_config_key,
        ),
    )


def benchmark_workspace_settings(ws: Any) -> Any:
    return ws.WorkspaceSettings(
        sort_panels_alphabetically=False,
        group_by_prefix="first",
        max_runs=10,
    )


def benchmark_runset_settings(ws: Any, *, group: str | None, comparison_config_key: str) -> Any:
    return ws.RunsetSettings(
        filters=benchmark_run_filters(ws, group=group),
        groupby=benchmark_run_groupby(ws, group=group, comparison_config_key=comparison_config_key),
        order=[ws.Ordering(ws.Metric("CreatedTimestamp"), ascending=False)],
        pinned_columns=["Name", "State", "CreatedTimestamp", "Group", "JobType", "Tags"],
    )


def benchmark_run_filters(ws: Any, *, group: str | None) -> Any:
    job_filter = ws.Or(*(ws.Metric("JobType") == job_type for job_type in _WORKSPACE_JOB_FILTERS))
    return ws.And(job_filter, ws.Metric("Group") == group) if group else job_filter


def benchmark_run_groupby(ws: Any, *, group: str | None, comparison_config_key: str) -> list[Any]:
    return [ws.Config(comparison_config_key)] if group else [ws.Metric("Group")]


def benchmark_workspace_sections(
    ws: Any,
    wr: Any,
    *,
    groupby: Any | None,
    comparison_config_key: str,
) -> list[Any]:
    sections = []
    for name, panel_defs, is_open in _WORKSPACE_BAR_SECTIONS:
        panels = workspace_bar_panels(wr, panel_defs, groupby=groupby)
        if name == "Benchmark Summary":
            panels = [
                *(
                    wr.ScalarChart(title=metric_title(metric), metric=metric, groupby_aggfunc="mean")
                    for metric in _WORKSPACE_SUMMARY_SCALARS
                ),
                *panels,
            ]
        sections.append(ws.Section(name=name, panels=panels, is_open=is_open))
    if groupby is not None:
        sections.append(
            ws.Section(
                name="Sweep Comparison",
                panels=comparison_workspace_panels(wr, comparison_config_key=comparison_config_key),
                is_open=True,
            )
        )
    sections.append(ws.Section(name="Tables", panels=table_workspace_panels(wr), is_open=False))
    return sections


def workspace_bar_panels(
    wr: Any,
    panel_defs: tuple[tuple[str, list[str]], ...],
    *,
    groupby: Any | None,
) -> list[Any]:
    return [bar_panel(wr, title, metrics, groupby=groupby) for title, metrics in panel_defs]


def comparison_workspace_panels(wr: Any, *, comparison_config_key: str = "sweep_arm_id") -> list[Any]:
    columns = [f"config:{comparison_config_key}", *_WORKSPACE_COMPARISON_COLUMNS[1:]]
    return [
        wr.RunComparer(diff_only=True),
        wr.ParallelCoordinatesPlot(
            title="Sweep Parameter Tradeoffs",
            columns=[
                wr.ParallelCoordinatesPlotColumn(metric=workspace_metric_accessor(wr, column)) for column in columns
            ],
        ),
        wr.ParameterImportancePlot(with_respect_to="measurement/record/weighted_leakage_rate_mean"),
    ]


def table_workspace_panels(wr: Any) -> list[Any]:
    return [wr.MediaBrowser(title="Sanitized Measurement Tables", media_keys=_MEASUREMENT_TABLE_KEYS, mode="grid")]


def workspace_metric_accessor(wr: Any, column: str) -> Any:
    section, name = column.split(":", maxsplit=1)
    if section == "config":
        return wr.Config(name)
    if section == "summary":
        return wr.SummaryMetric(name)
    raise ValueError(f"Unsupported workspace metric section: {section!r}")


def save_workspace(workspace: Any) -> Any:
    """Save a workspace through the W&B Workspaces API."""
    return workspace.save()


def default_workspace_title(*, group: str | None) -> str:
    suffix = f": {plain_text(group)}" if group else ""
    return f"NeMo Anonymizer Benchmark Workspace{suffix}"[:128]


__all__ = [
    "benchmark_run_filters",
    "benchmark_run_groupby",
    "benchmark_runset_settings",
    "benchmark_workspace_sections",
    "benchmark_workspace_settings",
    "build_benchmark_workspace",
    "comparison_workspace_panels",
    "create_benchmark_workspace",
    "default_workspace_title",
    "save_workspace",
    "table_workspace_panels",
    "workspace_bar_panels",
    "workspace_metric_accessor",
]
