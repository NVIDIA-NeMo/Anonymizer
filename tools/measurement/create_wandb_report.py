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

import json
import logging
import re
import sys
import unicodedata
from html import escape as escape_html
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import cyclopts
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input
from measurement_tools.wandb_models import ConfigMetadata, ResolvedWandbConfig, WandbMode, WorkloadMetadata
from measurement_tools.wandb_report_models import (
    GroupComparison,
    WandbProjectPath,
    WandbRunPath,
    WandbRunView,
    group_comparison,
    parse_wandb_project_path,
    parse_wandb_run_path,
    parse_wandb_run_view,
    validate_wandb_returned_url,
)
from measurement_tools.wandb_setup import WandbSdkEnvironment, require_wandb
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.wandb_report")

_WANDB_REPORT_INSTALL_HINT = "Install the optional measurement dependency group: uv sync --group measurement"
_CASE_HEALTH_METRICS = [
    "benchmark/case_total",
    "benchmark/case_completed",
    "benchmark/case_errored",
    "benchmark/case_success_rate",
]
_CASE_LATENCY_METRICS = [
    "benchmark/case_elapsed_sec_mean",
    "benchmark/case_elapsed_sec_sum",
    "measurement/stage/elapsed_sec",
]
_NDD_ROW_FLOW_METRICS = [
    "measurement/ndd_workflow/seed_row_count",
    "measurement/ndd_workflow/input_row_count",
    "measurement/ndd_workflow/output_row_count",
    "measurement/ndd_workflow/failed_record_count",
    "measurement/ndd_workflow/column_count",
]
_NDD_REQUEST_HEALTH_METRICS = [
    "measurement/ndd_workflow/observed_total_requests",
    "measurement/ndd_workflow/observed_successful_requests",
    "measurement/ndd_workflow/observed_failed_requests",
    "measurement/ndd_workflow/observed_failed_request_rate_mean",
]
_NDD_TOKEN_METRICS = [
    "measurement/ndd_workflow/observed_input_tokens",
    "measurement/ndd_workflow/observed_output_tokens",
    "measurement/ndd_workflow/observed_total_tokens",
    "measurement/ndd_workflow/observed_tokens_per_successful_request_mean",
]
_NDD_THROUGHPUT_METRICS = [
    "measurement/ndd_workflow/elapsed_sec",
    "measurement/ndd_workflow/input_rows_per_sec_mean",
    "measurement/ndd_workflow/output_rows_per_sec_mean",
    "measurement/ndd_workflow/observed_requests_per_sec_mean",
    "measurement/ndd_workflow/observed_tokens_per_sec_mean",
]
_RECORD_PRIVACY_METRICS = [
    "measurement/record/final_entity_count",
    "measurement/record/entity_precision_mean",
    "measurement/record/entity_recall_mean",
    "measurement/record/entity_f1_mean",
    "measurement/record/original_value_leak_count",
    "measurement/record/leakage_mass_mean",
    "measurement/record/weighted_leakage_rate_mean",
    "measurement/record/detected_candidate_count",
    "measurement/record/validation_chunk_count",
    "measurement/record/llm_calls_estimated_total",
]
_REWRITE_UTILITY_METRICS = [
    "measurement/record/utility_score_mean",
    "measurement/record/repair_iterations_mean",
]
_REPLACEMENT_QUALITY_METRICS = [
    "measurement/record/replacement_count",
    "measurement/record/replacement_duplicate_value_count",
    "measurement/record/replacement_missing_final_entity_count",
    "measurement/record/replacement_missing_final_value_count",
    "measurement/record/replacement_synthetic_original_collision_count",
    "measurement/record/replacement_synthetic_original_collision_value_count",
]
_STAGE_THROUGHPUT_METRICS = [
    "measurement/stage/input_row_count",
    "measurement/stage/output_row_count",
    "measurement/stage/failed_record_count",
    "measurement/stage/input_rows_per_sec_mean",
    "measurement/stage/output_rows_per_sec_mean",
]
_MEASUREMENT_TABLE_KEYS = [
    "measurement_table/run",
    "measurement_table/stage",
    "measurement_table/ndd_workflow",
    "measurement_table/model_workflow",
    "measurement_table/record",
    "measurement_table/evaluation_record",
]
_WORKSPACE_JOB_FILTERS = ("benchmark", "benchmark-import", "benchmark-sweep")
_WORKSPACE_SUMMARY_SCALARS = [
    "benchmark/case_success_rate",
    "benchmark/case_completed",
    "benchmark/case_errored",
    "benchmark/case_elapsed_sec_mean",
]
_WORKSPACE_COMPARISON_COLUMNS = [
    "config:sweep_arm_id",
    "config:benchmark_strategies",
    "config:benchmark_gliner_thresholds",
    "config:benchmark_risk_tolerances",
    "summary:benchmark/case_success_rate",
    "summary:measurement/record/utility_score_mean",
    "summary:measurement/record/weighted_leakage_rate_mean",
    "summary:measurement/ndd_workflow/observed_total_tokens",
]


class WandbReportResult(BaseModel):
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
        return _validate_output_url(value)

    @field_validator("project_path", "group", "title")
    @classmethod
    def validate_output_text(cls, value: str | None) -> str | None:
        return _validate_output_text(value)


class WandbWorkspaceResult(BaseModel):
    project_path: str
    group: str | None = None
    workspace_url: str
    title: str

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("workspace_url")
    @classmethod
    def validate_output_urls(cls, value: str) -> str:
        validated = _validate_output_url(value)
        if validated is None:
            raise ValueError("W&B workspace URL is required")
        return validated

    @field_validator("project_path", "group", "title")
    @classmethod
    def validate_output_text(cls, value: str | None) -> str | None:
        return _validate_output_text(value)


def require_wandb_report_sdk() -> tuple[Any, Any, Any]:
    """Import W&B report APIs when report generation is requested."""
    wandb = require_wandb()
    try:
        from wandb.apis.reports import v2 as wr
        from wandb_workspaces import expr
    except ImportError as exc:
        raise ImportError(f"W&B report generation requires wandb[workspaces]. {_WANDB_REPORT_INSTALL_HINT}") from exc
    return wandb, wr, expr


def require_wandb_workspace_sdk() -> tuple[Any, Any]:
    """Import W&B workspace APIs when workspace generation is requested."""
    require_wandb()
    try:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError as exc:
        raise ImportError(f"W&B workspace generation requires wandb[workspaces]. {_WANDB_REPORT_INSTALL_HINT}") from exc
    return ws, wr


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
    resolved = _report_settings(settings, run_path)
    effective_run_path = WandbRunPath(
        entity=run_path.entity,
        project=run_path.project,
        run_id=run_path.run_id,
        base_url=resolved.wandb_base_url or run_path.base_url,
    )
    with WandbSdkEnvironment(resolved):
        wandb, wr, expr = require_wandb_report_sdk()
        run = wandb.Api(timeout=timeout).run(run_path.path)
        view = parse_wandb_run_view(
            run,
            run_path=effective_run_path,
            allowed_metrics=frozenset(_all_report_metrics()),
        )
        report_title = _plain_text(title) if title is not None else _default_report_title(view)
        report_description = _plain_text(
            description or "SDK-generated benchmark report for a NeMo Anonymizer measurement run."
        )
        report = build_benchmark_report(
            view,
            title=report_title,
            description=report_description,
            wr=wr,
            expr=expr,
        )
        _save_report(report, draft=draft)
        report_url = validate_wandb_returned_url(report.url, expected_base_url=effective_run_path.base_url)
    return WandbReportResult(
        run_path=run_path.path,
        run_url=effective_run_path.url,
        project_path=f"{run_path.entity}/{run_path.project}",
        report_url=report_url,
        draft=draft,
        title=report_title,
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
    effective_project_path = WandbProjectPath(
        entity=project_path.entity,
        project=project_path.project,
        base_url=resolved.wandb_base_url or project_path.base_url,
    )
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


def create_benchmark_group_workspace(
    project_path: WandbProjectPath,
    *,
    settings: ResolvedWandbConfig | None = None,
    group: str,
    title: str | None = None,
    expected_run_kind: Literal["native_suite", "sweep_arm", "imported_case"] | None = None,
) -> WandbWorkspaceResult:
    """Create a W&B workspace filtered to one benchmark run group."""
    return create_benchmark_workspace(
        project_path,
        settings=settings,
        group=group,
        title=title,
        expected_run_kind=expected_run_kind,
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
    resolved = _report_settings(settings, project_path)
    effective_project_path = WandbProjectPath(
        entity=project_path.entity,
        project=project_path.project,
        base_url=resolved.wandb_base_url or project_path.base_url,
    )
    with WandbSdkEnvironment(resolved):
        wandb, wr, expr = require_wandb_report_sdk()
        views = _read_group_views(wandb, project_path=effective_project_path, group=group)
        comparison = group_comparison(views, expected_run_kind=expected_run_kind)
        report_title = (
            _plain_text(title) if title is not None else f"NeMo Anonymizer Benchmark Group: {_plain_text(group)}"
        )
        report_description = _plain_text(
            description or "SDK-generated benchmark sweep report for NeMo Anonymizer measurement runs."
        )
        report = build_benchmark_group_report(
            effective_project_path,
            group=group,
            title=report_title,
            description=report_description,
            sweep_param_columns=_sweep_param_columns(views),
            comparison=comparison,
            wr=wr,
            expr=expr,
        )
        _save_report(report, draft=draft)
        report_url = validate_wandb_returned_url(report.url, expected_base_url=effective_project_path.base_url)
    return WandbReportResult(
        project_path=project_path.path,
        group=group,
        report_url=report_url,
        draft=draft,
        title=report_title,
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


def build_benchmark_report(view: WandbRunView, *, title: str, description: str, wr: Any, expr: Any) -> Any:
    """Build an unsaved report object from a W&B run."""
    run_path = view.path
    run_id = run_path.run_id
    runset = wr.Runset(
        entity=run_path.entity,
        project=run_path.project,
        name="Benchmark run",
        filters=[expr.Metric("ID") == run_id],
        pinned_columns=["Name", "State", "CreatedTimestamp", "Tags"],
        visible_columns=_single_run_visible_columns(),
        run_settings={run_id: wr.RunSettings(color="#2F80ED", disabled=False)},
    )
    safe_title = _escape_heading(title)
    return wr.Report(
        entity=run_path.entity,
        project=run_path.project,
        title=safe_title,
        description=_escape_markdown_text(description),
        width="fluid",
        blocks=[
            wr.H1(safe_title),
            wr.MarkdownBlock(build_report_markdown(view)),
            wr.H2("Run Panels"),
            wr.PanelGrid(runsets=[runset], panels=_benchmark_panels(wr)),
        ],
    )


def build_benchmark_group_report(
    project_path: WandbProjectPath,
    *,
    group: str,
    title: str,
    description: str,
    sweep_param_columns: list[str] | None = None,
    comparison: GroupComparison | None = None,
    wr: Any,
    expr: Any,
) -> Any:
    """Build an unsaved report object for a W&B run group."""
    resolved_comparison = comparison or GroupComparison(
        run_kind="sweep_arm",
        config_key="sweep_arm_id",
        label="Sweep Arm",
    )
    runset = wr.Runset(
        entity=project_path.entity,
        project=project_path.project,
        name="Benchmark sweep",
        filters=[expr.Metric("Group") == group],
        groupby=[f"config.{resolved_comparison.config_key}"],
        pinned_columns=["Name", "State", "CreatedTimestamp", "Group", "Tags"],
        visible_columns=_group_visible_columns(resolved_comparison, sweep_param_columns),
    )
    safe_title = _escape_heading(title)
    return wr.Report(
        entity=project_path.entity,
        project=project_path.project,
        title=safe_title,
        description=_escape_markdown_text(description),
        width="fluid",
        blocks=[
            wr.H1(safe_title),
            wr.MarkdownBlock(build_group_report_markdown(group=group, comparison=resolved_comparison)),
            wr.H2("Sweep Panels"),
            wr.PanelGrid(
                runsets=[runset],
                panels=_benchmark_panels(wr, groupby=wr.Config(resolved_comparison.config_key)),
            ),
        ],
    )


def _single_run_visible_columns() -> list[str]:
    return _summary_columns(_all_report_metrics())


def _group_visible_columns(comparison: GroupComparison, parameter_columns: list[str] | None = None) -> list[str]:
    columns = [
        f"config:{comparison.config_key}",
        "config:run_kind",
        *(["config:sweep_id", "config:sweep_params"] if comparison.run_kind == "sweep_arm" else []),
        *(parameter_columns or []),
        *_summary_columns(_all_report_metrics()),
    ]
    return list(dict.fromkeys(columns))


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
    sections = [
        ws.Section(name="Benchmark Summary", panels=_summary_workspace_panels(wr, groupby=groupby), is_open=True),
        ws.Section(name="Privacy", panels=_privacy_workspace_panels(wr, groupby=groupby), is_open=True),
        ws.Section(name="Utility", panels=_utility_workspace_panels(wr, groupby=groupby), is_open=True),
        ws.Section(name="Cost/Throughput", panels=_cost_workspace_panels(wr, groupby=groupby), is_open=True),
    ]
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


def _summary_workspace_panels(wr: Any, *, groupby: Any | None) -> list[Any]:
    return [
        *(
            wr.ScalarChart(title=_metric_title(metric), metric=metric, groupby_aggfunc="mean")
            for metric in _WORKSPACE_SUMMARY_SCALARS
        ),
        _bar_panel(wr, "Case Health", _CASE_HEALTH_METRICS, groupby=groupby),
        _bar_panel(wr, "Case Latency", _CASE_LATENCY_METRICS, groupby=groupby),
    ]


def _privacy_workspace_panels(wr: Any, *, groupby: Any | None) -> list[Any]:
    return [
        _bar_panel(wr, "Privacy Outcomes", _RECORD_PRIVACY_METRICS, groupby=groupby),
        _bar_panel(wr, "Replacement Quality", _REPLACEMENT_QUALITY_METRICS, groupby=groupby),
    ]


def _utility_workspace_panels(wr: Any, *, groupby: Any | None) -> list[Any]:
    return [_bar_panel(wr, "Rewrite Utility", _REWRITE_UTILITY_METRICS, groupby=groupby)]


def _cost_workspace_panels(wr: Any, *, groupby: Any | None) -> list[Any]:
    return [
        _bar_panel(wr, "NDD Request Health", _NDD_REQUEST_HEALTH_METRICS, groupby=groupby),
        _bar_panel(wr, "NDD Token Usage", _NDD_TOKEN_METRICS, groupby=groupby),
        _bar_panel(wr, "NDD Throughput", _NDD_THROUGHPUT_METRICS, groupby=groupby),
        _bar_panel(wr, "Stage Throughput", _STAGE_THROUGHPUT_METRICS, groupby=groupby),
    ]


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


def _metric_title(metric: str) -> str:
    return metric.rsplit("/", maxsplit=1)[-1].replace("_", " ").title()


def _benchmark_panels(wr: Any, *, groupby: Any | None = None) -> list[Any]:
    return [
        *_metric_panels(wr, groupby=groupby),
        wr.MediaBrowser(title="Sanitized Measurement Tables", media_keys=_MEASUREMENT_TABLE_KEYS, mode="grid"),
    ]


def _metric_panels(wr: Any, *, groupby: Any | None) -> list[Any]:
    return [
        _bar_panel(wr, "Case Health", _CASE_HEALTH_METRICS, groupby=groupby),
        _bar_panel(wr, "Case Latency", _CASE_LATENCY_METRICS, groupby=groupby),
        _bar_panel(wr, "NDD Row Flow", _NDD_ROW_FLOW_METRICS, groupby=groupby),
        _bar_panel(wr, "NDD Request Health", _NDD_REQUEST_HEALTH_METRICS, groupby=groupby),
        _bar_panel(wr, "NDD Token Usage", _NDD_TOKEN_METRICS, groupby=groupby),
        _bar_panel(wr, "NDD Throughput", _NDD_THROUGHPUT_METRICS, groupby=groupby),
        _bar_panel(wr, "Record Privacy", _RECORD_PRIVACY_METRICS, groupby=groupby),
        _bar_panel(wr, "Rewrite Utility", _REWRITE_UTILITY_METRICS, groupby=groupby),
        _bar_panel(wr, "Replacement Quality", _REPLACEMENT_QUALITY_METRICS, groupby=groupby),
        _bar_panel(wr, "Stage Throughput", _STAGE_THROUGHPUT_METRICS, groupby=groupby),
    ]


def _bar_panel(wr: Any, title: str, metrics: list[str], *, groupby: Any | None = None) -> Any:
    return wr.BarPlot(title=title, metrics=metrics, groupby=groupby)


def _all_report_metrics() -> list[str]:
    return list(
        dict.fromkeys(
            [
                *_CASE_HEALTH_METRICS,
                *_CASE_LATENCY_METRICS,
                *_NDD_ROW_FLOW_METRICS,
                *_NDD_REQUEST_HEALTH_METRICS,
                *_NDD_TOKEN_METRICS,
                *_NDD_THROUGHPUT_METRICS,
                *_RECORD_PRIVACY_METRICS,
                *_REWRITE_UTILITY_METRICS,
                *_REPLACEMENT_QUALITY_METRICS,
                *_STAGE_THROUGHPUT_METRICS,
            ]
        )
    )


def _summary_columns(metrics: list[str]) -> list[str]:
    return [f"summary:{metric}" for metric in metrics]


def _read_group_views(wandb: Any, *, project_path: WandbProjectPath, group: str) -> list[WandbRunView]:
    runs = wandb.Api(timeout=60).runs(project_path.path, filters={"group": group})
    views: list[WandbRunView] = []
    for run in runs:
        try:
            run_id = getattr(run, "id", None)
            if not isinstance(run_id, str):
                raise ValueError("W&B group run is missing a string identity")
            view = parse_wandb_run_view(
                run,
                run_path=WandbRunPath(
                    entity=project_path.entity,
                    project=project_path.project,
                    run_id=run_id,
                    base_url=project_path.base_url,
                ),
                allowed_metrics=frozenset(_all_report_metrics()),
            )
        except (ValueError, ValidationError):
            raise ValueError("W&B group contains invalid run metadata") from None
        views.append(view)
    return views


def _sweep_param_columns(views: list[WandbRunView]) -> list[str]:
    keys = sorted({key for view in views if view.metadata.sweep is not None for key in view.metadata.sweep.params})
    return [f"config:sweep_param_{key}" for key in keys]


def _report_settings(
    settings: ResolvedWandbConfig | None,
    target: WandbRunPath | WandbProjectPath,
) -> ResolvedWandbConfig:
    target_base_url = target.base_url if target.base_url != "https://wandb.ai" else None
    if settings is None:
        return ResolvedWandbConfig.from_env_and_overrides(
            wandb_mode=WandbMode.online,
            wandb_entity=target.entity,
            wandb_project=target.project,
            wandb_base_url=target_base_url,
        )
    if target_base_url is not None and settings.wandb_base_url not in {None, target_base_url}:
        raise ValueError("W&B target URL conflicts with resolved base URL")
    return settings.validated_update(
        wandb_entity=target.entity,
        wandb_project=target.project,
        wandb_base_url=target_base_url or settings.wandb_base_url,
    )


def build_group_report_markdown(*, group: str, comparison: GroupComparison | None = None) -> str:
    resolved = comparison or GroupComparison(run_kind="sweep_arm", config_key="sweep_arm_id", label="Sweep Arm")
    return f"""### Group Summary

This report compares benchmark runs in W&B group {_code_span(group)}.

Run kind: {_code_span(resolved.run_kind)}. Comparison axis: {_code_span(resolved.label)}.

### Privacy Boundary

This report is built from benchmark W&B summary/config fields. The benchmark runner sanitizes these fields before upload and excludes raw text, prompts, model responses, replacement maps, entity payloads, trace records, paths, URLs, provider config payloads, and sensitive-looking run tags.
"""


def build_report_markdown(view: WandbRunView) -> str:
    """Render report Markdown exclusively from a closed typed run view."""
    summary = view.summary.metrics
    metadata = view.metadata
    benchmark = metadata.benchmark
    runtime = metadata.runtime
    git = metadata.git
    workload_lines = [_workload_line(item) for item in metadata.workloads]
    config_lines = [_config_line(item) for item in metadata.configs]
    return f"""### Run Summary

[{_escape_link_label(view.name)}]({view.path.url}) finished with **{_int_metric(summary, "benchmark/case_completed")}/{_int_metric(summary, "benchmark/case_total")} cases completed** and **{_int_metric(summary, "benchmark/case_errored")} errors**.

| Metric | Value |
| --- | ---: |
| Case success rate | {_number_metric(summary, "benchmark/case_success_rate")} |
| Mean case elapsed seconds | {_number_metric(summary, "benchmark/case_elapsed_sec_mean")} |
| Final entities | {_int_metric(summary, "measurement/record/final_entity_count")} |
| Original value leaks | {_int_metric(summary, "measurement/record/original_value_leak_count")} |
| Model requests | {_int_metric(summary, "measurement/ndd_workflow/observed_total_requests")} |
| Failed model requests | {_int_metric(summary, "measurement/ndd_workflow/observed_failed_requests")} |
| Total model tokens | {_int_metric(summary, "measurement/ndd_workflow/observed_total_tokens")} |

### Benchmark Metadata

| Field | Value |
| --- | --- |
| Run kind | {_table_code(metadata.run_kind)} |
| Suite | {_table_code(benchmark.suite_id if benchmark else None)} |
| Suite file hash | {_table_code(benchmark.suite_file_hash if benchmark else None)} |
| Git branch | {_table_code(git.branch if git else None)} |
| Git commit | {_table_code(git.commit if git else None)} |
| Git dirty | {_table_code(git.dirty if git else None)} |
| Anonymizer | {_table_code(runtime.anonymizer_version if runtime else None)} |
| DataDesigner | {_table_code(runtime.datadesigner_version if runtime else None)} |
| W&B | {_table_code(runtime.wandb_version if runtime else None)} |

### Workloads

{chr(10).join(workload_lines) or "- none"}

### Configs

{chr(10).join(config_lines) or "- none"}

### Privacy Boundary

This report is built from benchmark W&B summary/config fields. The benchmark runner sanitizes these fields before upload and excludes raw text, prompts, model responses, replacement maps, entity payloads, trace records, paths, URLs, provider config payloads, and sensitive-looking run tags.
"""


def _save_report(report: Any, *, draft: bool) -> Any:
    """Save a report, falling back around a W&B SDK project-list auth edge."""
    try:
        return report.save(draft=draft)
    except Exception as exc:  # noqa: BLE001 -- preserve the SDK error as fallback context
        if "relogin required" not in str(exc).lower():
            raise
        logger.info("Falling back to direct W&B report upsert after project preflight auth failure.")
        return _save_report_without_project_preflight(report, draft=draft)


def _save_report_without_project_preflight(report: Any, *, draft: bool) -> Any:
    """Call the same report upsert mutation without the project-list preflight."""
    wandb = require_wandb()
    from wandb_workspaces._graphql import execute_graphql
    from wandb_workspaces.reports.v2 import gql, internal

    api = wandb.Api()
    model = report._to_model()
    result = execute_graphql(
        api,
        gql.upsert_view,
        {
            "id": None if not model.id else model.id,
            "name": internal._generate_name() if not model.name else model.name,
            "entityName": model.project.entity_name,
            "projectName": model.project.name,
            "description": model.description,
            "displayName": model.display_name,
            "type": "runs/draft" if draft else "runs",
            "spec": model.spec.model_dump_json(by_alias=True, exclude_none=True),
        },
    )
    report.id = result["upsertView"]["view"]["id"]
    return report


def _save_workspace(workspace: Any) -> Any:
    """Save a workspace through the W&B Workspaces API."""
    return workspace.save()


def _default_report_title(view: WandbRunView) -> str:
    return f"NeMo Anonymizer Benchmark: {_plain_text(view.name)}"[:128]


def _default_workspace_title(*, group: str | None) -> str:
    suffix = f": {_plain_text(group)}" if group else ""
    return f"NeMo Anonymizer Benchmark Workspace{suffix}"[:128]


def _workload_line(item: WorkloadMetadata) -> str:
    source_suffix = item.source.suffix if item.source else None
    source_kind = item.source.kind if item.source else None
    return (
        f"- {_code_span(item.id)}: source_kind={_code_span(source_kind)}, "
        f"source_suffix={_code_span(source_suffix)}, row_limit={_escape_list_text(item.row_limit)}, "
        f"text_column={_code_span(item.text_column)}"
    )


def _config_line(item: ConfigMetadata) -> str:
    detect = item.detect
    replace = item.replace
    rewrite = item.rewrite
    parts = [
        f"strategy={_code_span(replace.strategy if replace else ('rewrite' if rewrite else None))}",
        f"entity_label_count={_escape_list_text(detect.entity_label_count if detect else None)}",
        f"gliner_threshold={_escape_list_text(detect.gliner_threshold if detect else None)}",
    ]
    if rewrite:
        parts.append(f"risk_tolerance={_code_span(rewrite.risk_tolerance)}")
    return f"- {_code_span(item.id)}: " + ", ".join(parts)


def _metric(summary: dict[str, Any], key: str) -> int | float | None:
    value = summary.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return value
    return None


def _int_metric(summary: dict[str, Any], key: str) -> int:
    value = _metric(summary, key)
    return int(value) if value is not None else 0


def _number_metric(summary: dict[str, Any], key: str) -> str:
    value = _metric(summary, key)
    if value is None:
        return "n/a"
    return f"{float(value):.4g}"


def _scalar_text(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        return str(value)
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _plain_text(value: str) -> str:
    if any(
        unicodedata.category(character) == "Cf"
        or (unicodedata.category(character) == "Cc" and character not in "\t\n\r")
        for character in value
    ):
        raise ValueError("W&B display text contains unsafe control characters")
    return " ".join(value.split())


def _validate_output_url(value: str | None) -> str | None:
    if value is None:
        return None
    if any(unicodedata.category(character) in {"Cc", "Cf"} for character in value):
        raise ValueError("W&B output URL contains unsafe control characters")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("W&B output URL must be a credential-free HTTP(S) URL")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("W&B output URL must use HTTPS unless it targets loopback")
    return value


def _validate_output_text(value: str | None) -> str | None:
    if value is not None:
        _plain_text(value)
    return value


def _escape_link_label(value: str) -> str:
    text = escape_html(_plain_text(value), quote=False)
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _escape_markdown_text(value: str) -> str:
    text = escape_html(_plain_text(value), quote=False)
    for character in ("\\", "`", "*", "_", "{", "}", "[", "]", "(", ")", "#", "+", "-", ".", "!", "|"):
        text = text.replace(character, f"\\{character}")
    return text


def _escape_heading(value: str) -> str:
    return _escape_markdown_text(value)


def _code_span(value: Any) -> str:
    text = escape_html(_scalar_text(value), quote=False).replace("|", "&#124;")
    text = _plain_text(text)
    fence = "`" * (max((len(match) for match in re.findall(r"`+", text)), default=0) + 1)
    if "`" in text:
        return f"{fence} {text} {fence}"
    return f"`{text}`"


def _table_code(value: Any) -> str:
    return _code_span(value)


def _escape_list_text(value: Any) -> str:
    return escape_html(_plain_text(_scalar_text(value)), quote=False).replace("|", "&#124;")


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
