#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a W&B benchmark report from a finished Anonymizer benchmark run.

Usage:
    uv run python tools/measurement/create_wandb_report.py entity/project/run_id
    uv run python tools/measurement/create_wandb_report.py https://wandb.ai/entity/project/runs/run_id --json
    uv run python tools/measurement/create_wandb_report.py entity/project/run_id --publish
    uv run python tools/measurement/create_wandb_report.py entity/project --workspace
    uv run python tools/measurement/create_wandb_report.py entity/project --workspace --group sweep_id
"""

from __future__ import annotations

import logging
import sys
from typing import Annotated, Any, Literal

import cyclopts
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input
from measurement_tools.wandb_models import ConfigMetadata as ConfigMetadata
from measurement_tools.wandb_models import ResolvedWandbConfig as ResolvedWandbConfig
from measurement_tools.wandb_models import WandbMode as WandbMode
from measurement_tools.wandb_models import WorkloadMetadata as WorkloadMetadata
from measurement_tools.wandb_report_catalog import (
    _MEASUREMENT_TABLE_KEYS,
    _WORKSPACE_BAR_SECTIONS,
    _WORKSPACE_COMPARISON_COLUMNS,
    _WORKSPACE_JOB_FILTERS,
    _WORKSPACE_SUMMARY_SCALARS,
    all_report_metrics,
    bar_panel,
    benchmark_panels,
    group_visible_columns,
    metric_panels,
    metric_title,
    single_run_visible_columns,
    summary_columns,
)
from measurement_tools.wandb_report_contracts import WandbReportResult, WandbWorkspaceResult
from measurement_tools.wandb_report_models import (
    GroupComparison,
    WandbProjectPath,
    WandbRunPath,
    group_comparison,
    parse_wandb_project_path,
    parse_wandb_run_path,
    validate_wandb_returned_url,
)
from measurement_tools.wandb_report_models import (
    WandbRunView as WandbRunView,
)
from measurement_tools.wandb_report_models import (
    parse_wandb_run_view as parse_wandb_run_view,
)
from measurement_tools.wandb_report_sdk import (
    read_group_views,
    report_settings,
    require_wandb_report_sdk,
    require_wandb_workspace_sdk,
    sweep_param_columns,
)
from measurement_tools.wandb_report_text import (
    code_span,
    escape_heading,
    escape_link_label,
    escape_list_text,
    escape_markdown_text,
    plain_text,
    scalar_text,
    table_code,
    validate_output_text,
    validate_output_url,
)
from measurement_tools.wandb_reports import (
    build_benchmark_group_report,
    build_benchmark_report,
    config_line,
    default_report_title,
    int_metric,
    metric,
    number_metric,
    save_report,
    workload_line,
)
from measurement_tools.wandb_reports import (
    build_group_report_markdown as build_group_report_markdown,
)
from measurement_tools.wandb_reports import (
    build_report_markdown as build_report_markdown,
)
from measurement_tools.wandb_reports import (
    create_benchmark_group_report as _create_benchmark_group_report,
)
from measurement_tools.wandb_reports import (
    create_benchmark_report as _create_benchmark_report,
)
from measurement_tools.wandb_setup import WandbSdkEnvironment
from measurement_tools.wandb_setup import require_wandb as require_wandb

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.wandb_report")

_all_report_metrics = all_report_metrics
_bar_panel = bar_panel
_benchmark_panels = benchmark_panels
_code_span = code_span
_escape_heading = escape_heading
_escape_link_label = escape_link_label
_escape_list_text = escape_list_text
_escape_markdown_text = escape_markdown_text
_group_visible_columns = group_visible_columns
_metric_panels = metric_panels
_metric_title = metric_title
_plain_text = plain_text
_read_group_views = read_group_views
_report_settings = report_settings
_scalar_text = scalar_text
_single_run_visible_columns = single_run_visible_columns
_summary_columns = summary_columns
_sweep_param_columns = sweep_param_columns
_table_code = table_code
_validate_output_text = validate_output_text
_validate_output_url = validate_output_url
_config_line = config_line
_default_report_title = default_report_title
_int_metric = int_metric
_metric = metric
_number_metric = number_metric
_save_report = save_report
_workload_line = workload_line


def create_benchmark_report(
    run_path: WandbRunPath,
    *,
    settings: ResolvedWandbConfig | None = None,
    title: str | None = None,
    description: str | None = None,
    draft: bool = True,
    timeout: int = 60,
) -> WandbReportResult:
    """Create a W&B report for one benchmark run."""
    return _create_benchmark_report(
        run_path,
        settings=settings,
        title=title,
        description=description,
        draft=draft,
        timeout=timeout,
        sdk_loader=require_wandb_report_sdk,
        report_builder=build_benchmark_report,
        report_saver=_save_report,
    )


def create_benchmark_workspace(
    project_path: WandbProjectPath,
    *,
    settings: ResolvedWandbConfig | None = None,
    group: str | None = None,
    title: str | None = None,
    expected_run_kind: Literal["native_suite", "sweep_arm", "imported_case"] | None = None,
) -> WandbWorkspaceResult:
    """Create a W&B workspace for benchmark runs in a project."""
    resolved = _report_settings(settings, project_path)
    effective_project_path = project_path.with_base_url(resolved.wandb_base_url or project_path.base_url)
    with WandbSdkEnvironment(resolved):
        comparison: GroupComparison | None = None
        if group is not None:
            wandb, _wr, _expr = require_wandb_report_sdk()
            views = _read_group_views(wandb, project_path=effective_project_path, group=group)
            comparison = group_comparison(views, expected_run_kind=expected_run_kind)
        ws, workspace_wr = require_wandb_workspace_sdk()
        workspace_title = _plain_text(title) if title is not None else _default_workspace_title(group=group)
        workspace = build_benchmark_workspace(
            effective_project_path,
            group=group,
            title=workspace_title,
            comparison_config_key=comparison.config_key if comparison is not None else "sweep_arm_id",
            ws=ws,
            wr=workspace_wr,
        )
        saved = _save_workspace(workspace)
        workspace_url = validate_wandb_returned_url(saved.url, expected_base_url=effective_project_path.base_url)
    return WandbWorkspaceResult(
        project_path=project_path.path,
        group=group,
        workspace_url=workspace_url,
        title=workspace_title,
    )


def create_benchmark_group_report(
    project_path: WandbProjectPath,
    *,
    settings: ResolvedWandbConfig | None = None,
    group: str,
    title: str | None = None,
    description: str | None = None,
    draft: bool = True,
    expected_run_kind: Literal["native_suite", "sweep_arm", "imported_case"] | None = None,
) -> WandbReportResult:
    """Create a W&B report for a benchmark run group."""
    return _create_benchmark_group_report(
        project_path,
        settings=settings,
        group=group,
        title=title,
        description=description,
        draft=draft,
        expected_run_kind=expected_run_kind,
        sdk_loader=require_wandb_report_sdk,
        report_builder=build_benchmark_group_report,
        report_saver=_save_report,
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
        settings=_benchmark_workspace_settings(ws),
        runset_settings=_benchmark_runset_settings(
            ws,
            group=group,
            comparison_config_key=comparison_config_key,
        ),
        sections=_benchmark_workspace_sections(
            ws,
            wr,
            groupby=groupby,
            comparison_config_key=comparison_config_key,
        ),
    )


def _benchmark_workspace_settings(ws: Any) -> Any:
    return ws.WorkspaceSettings(
        sort_panels_alphabetically=False,
        group_by_prefix="first",
        max_runs=10,
    )


def _benchmark_runset_settings(ws: Any, *, group: str | None, comparison_config_key: str) -> Any:
    return ws.RunsetSettings(
        filters=_benchmark_run_filters(ws, group=group),
        groupby=_benchmark_run_groupby(ws, group=group, comparison_config_key=comparison_config_key),
        order=[ws.Ordering(ws.Metric("CreatedTimestamp"), ascending=False)],
        pinned_columns=["Name", "State", "CreatedTimestamp", "Group", "JobType", "Tags"],
    )


def _benchmark_run_filters(ws: Any, *, group: str | None) -> Any:
    job_filter = ws.Or(*(ws.Metric("JobType") == job_type for job_type in _WORKSPACE_JOB_FILTERS))
    return ws.And(job_filter, ws.Metric("Group") == group) if group else job_filter


def _benchmark_run_groupby(ws: Any, *, group: str | None, comparison_config_key: str) -> list[Any]:
    return [ws.Config(comparison_config_key)] if group else [ws.Metric("Group")]


def _benchmark_workspace_sections(
    ws: Any,
    wr: Any,
    *,
    groupby: Any | None,
    comparison_config_key: str,
) -> list[Any]:
    sections = []
    for name, panel_defs, is_open in _WORKSPACE_BAR_SECTIONS:
        panels = _workspace_bar_panels(wr, panel_defs, groupby=groupby)
        if name == "Benchmark Summary":
            panels = [
                *(
                    wr.ScalarChart(title=_metric_title(metric), metric=metric, groupby_aggfunc="mean")
                    for metric in _WORKSPACE_SUMMARY_SCALARS
                ),
                *panels,
            ]
        sections.append(ws.Section(name=name, panels=panels, is_open=is_open))
    if groupby is not None:
        sections.append(
            ws.Section(
                name="Sweep Comparison",
                panels=_comparison_workspace_panels(wr, comparison_config_key=comparison_config_key),
                is_open=True,
            )
        )
    sections.append(ws.Section(name="Tables", panels=_table_workspace_panels(wr), is_open=False))
    return sections


def _workspace_bar_panels(
    wr: Any,
    panel_defs: tuple[tuple[str, list[str]], ...],
    *,
    groupby: Any | None,
) -> list[Any]:
    return [_bar_panel(wr, title, metrics, groupby=groupby) for title, metrics in panel_defs]


def _comparison_workspace_panels(wr: Any, *, comparison_config_key: str = "sweep_arm_id") -> list[Any]:
    columns = [f"config:{comparison_config_key}", *_WORKSPACE_COMPARISON_COLUMNS[1:]]
    return [
        wr.RunComparer(diff_only=True),
        wr.ParallelCoordinatesPlot(
            title="Sweep Parameter Tradeoffs",
            columns=[
                wr.ParallelCoordinatesPlotColumn(metric=_workspace_metric_accessor(wr, column)) for column in columns
            ],
        ),
        wr.ParameterImportancePlot(with_respect_to="measurement/record/weighted_leakage_rate_mean"),
    ]


def _table_workspace_panels(wr: Any) -> list[Any]:
    return [wr.MediaBrowser(title="Sanitized Measurement Tables", media_keys=_MEASUREMENT_TABLE_KEYS, mode="grid")]


def _workspace_metric_accessor(wr: Any, column: str) -> Any:
    section, name = column.split(":", maxsplit=1)
    if section == "config":
        return wr.Config(name)
    if section == "summary":
        return wr.SummaryMetric(name)
    raise ValueError(f"Unsupported workspace metric section: {section!r}")


def _save_workspace(workspace: Any) -> Any:
    """Save a workspace through the W&B Workspaces API."""
    return workspace.save()


def _default_workspace_title(*, group: str | None) -> str:
    suffix = f": {_plain_text(group)}" if group else ""
    return f"NeMo Anonymizer Benchmark Workspace{suffix}"[:128]


def render_result(result: WandbReportResult | WandbWorkspaceResult, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    if isinstance(result, WandbWorkspaceResult):
        return "\n".join(
            [
                f"Created W&B workspace: {result.title}",
                f"Project: {result.project_path}",
                f"Workspace: {result.workspace_url}",
            ]
        )
    state = "draft" if result.draft else "published"
    return "\n".join(
        [
            f"Created {state} W&B report: {result.title}",
            f"Run: {result.run_url}" if result.run_url else f"Project: {result.project_path}",
            f"Report: {result.report_url}",
        ]
    )


@app.default
def main(
    target: str,
    *,
    group: Annotated[str | None, cyclopts.Parameter("--group")] = None,
    title: Annotated[str | None, cyclopts.Parameter("--title")] = None,
    description: Annotated[str | None, cyclopts.Parameter("--description")] = None,
    publish: Annotated[bool, cyclopts.Parameter("--publish")] = False,
    workspace: Annotated[bool, cyclopts.Parameter("--workspace")] = False,
    timeout: Annotated[int, cyclopts.Parameter("--timeout")] = 60,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    try:
        if workspace:
            project_path = parse_wandb_project_path(target)
            run_path = None
        elif group:
            project_path = parse_wandb_project_path(target)
            run_path = None
        else:
            project_path = None
            run_path = parse_wandb_run_path(target)
    except ValueError as exc:
        log_bad_input(logger, str(exc))
        raise SystemExit(125) from exc
    try:
        if workspace:
            assert project_path is not None
            result = create_benchmark_workspace(project_path, group=group, title=title)
        elif group:
            assert project_path is not None
            result = create_benchmark_group_report(
                project_path,
                group=group,
                title=title,
                description=description,
                draft=not publish,
            )
        else:
            assert run_path is not None
            result = create_benchmark_report(
                run_path,
                title=title,
                description=description,
                draft=not publish,
                timeout=timeout,
            )
    except ImportError as exc:
        log_bad_input(logger, str(exc))
        raise SystemExit(125) from exc
    except Exception as exc:
        logger.error("W&B report creation failed (%s)", type(exc).__name__)
        raise SystemExit(1) from None
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")


if __name__ == "__main__":
    app()
