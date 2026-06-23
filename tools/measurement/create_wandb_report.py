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
import sys
from typing import Annotated, Any
from urllib.parse import urlparse

import cyclopts
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input
from pydantic import BaseModel

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
    "measurement/ndd_workflow/observed_tokens_per_successful_request",
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
    "measurement/record/original_value_leak_count",
    "measurement/record/leakage_mass_mean",
    "measurement/record/weighted_leakage_rate_mean",
    "measurement/record/detected_candidate_count",
    "measurement/record/validation_chunk_count",
    "measurement/record/llm_calls_estimated_total",
]
_REWRITE_UTILITY_METRICS = [
    "measurement/record/utility_score_mean",
    "measurement/record/repair_iterations",
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
_WORKSPACE_JOB_FILTERS = ("benchmark", "benchmark-sweep")
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


class WandbRunPath(BaseModel):
    entity: str
    project: str
    run_id: str

    @property
    def path(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"


class WandbProjectPath(BaseModel):
    entity: str
    project: str

    @property
    def path(self) -> str:
        return f"{self.entity}/{self.project}"


class WandbReportResult(BaseModel):
    run_path: str | None = None
    run_url: str | None = None
    project_path: str
    group: str | None = None
    report_url: str
    draft: bool
    title: str


class WandbWorkspaceResult(BaseModel):
    project_path: str
    group: str | None = None
    workspace_url: str
    title: str


def parse_wandb_run_path(value: str) -> WandbRunPath:
    """Parse a W&B run path or URL into entity/project/run id."""
    stripped = value.strip()
    parsed = urlparse(stripped)
    if parsed.scheme and parsed.netloc:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 4 and parts[2] == "runs":
            return WandbRunPath(entity=parts[0], project=parts[1], run_id=parts[3])
        raise ValueError("W&B run URL must look like https://wandb.ai/{entity}/{project}/runs/{run_id}")

    parts = stripped.split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError("W&B run path must look like {entity}/{project}/{run_id}")
    return WandbRunPath(entity=parts[0], project=parts[1], run_id=parts[2])


def parse_wandb_project_path(value: str) -> WandbProjectPath:
    """Parse a W&B project path or URL into entity/project."""
    stripped = value.strip()
    parsed = urlparse(stripped)
    if parsed.scheme and parsed.netloc:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return WandbProjectPath(entity=parts[0], project=parts[1])
        raise ValueError("W&B project URL must look like https://wandb.ai/{entity}/{project}")

    parts = stripped.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("W&B project path must look like {entity}/{project}")
    return WandbProjectPath(entity=parts[0], project=parts[1])


def require_wandb_report_sdk() -> tuple[Any, Any, Any]:
    """Import W&B report APIs when report generation is requested."""
    try:
        import wandb
        from wandb.apis.reports import v2 as wr
        from wandb_workspaces import expr
    except ImportError as exc:
        raise ImportError(f"W&B report generation requires wandb[workspaces]. {_WANDB_REPORT_INSTALL_HINT}") from exc
    return wandb, wr, expr


def require_wandb_workspace_sdk() -> tuple[Any, Any]:
    """Import W&B workspace APIs when workspace generation is requested."""
    try:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError as exc:
        raise ImportError(f"W&B workspace generation requires wandb[workspaces]. {_WANDB_REPORT_INSTALL_HINT}") from exc
    return ws, wr


def create_benchmark_report(
    run_path: WandbRunPath,
    *,
    title: str | None = None,
    description: str | None = None,
    draft: bool = True,
    timeout: int = 60,
) -> WandbReportResult:
    """Create a W&B report for one benchmark run."""
    wandb, wr, expr = require_wandb_report_sdk()
    api = wandb.Api(timeout=timeout)
    run = api.run(run_path.path)
    report_title = title or _default_report_title(run)
    report_description = description or "SDK-generated benchmark report for a NeMo Anonymizer measurement run."
    report = build_benchmark_report(run, run_path=run_path, title=report_title, description=report_description)
    _save_report(report, draft=draft)
    return WandbReportResult(
        run_path=run_path.path,
        run_url=str(run.url),
        project_path=f"{run_path.entity}/{run_path.project}",
        report_url=str(report.url),
        draft=draft,
        title=report_title,
    )


def create_benchmark_workspace(
    project_path: WandbProjectPath,
    *,
    group: str | None = None,
    title: str | None = None,
) -> WandbWorkspaceResult:
    """Create a W&B workspace for benchmark runs in a project."""
    workspace_title = title or _default_workspace_title(group=group)
    workspace = build_benchmark_workspace(project_path, group=group, title=workspace_title)
    saved = _save_workspace(workspace)
    return WandbWorkspaceResult(
        project_path=project_path.path,
        group=group,
        workspace_url=str(saved.url),
        title=workspace_title,
    )


def create_benchmark_group_workspace(
    project_path: WandbProjectPath,
    *,
    group: str,
    title: str | None = None,
) -> WandbWorkspaceResult:
    """Create a W&B workspace filtered to one benchmark run group."""
    return create_benchmark_workspace(project_path, group=group, title=title)


def create_benchmark_group_report(
    project_path: WandbProjectPath,
    *,
    group: str,
    title: str | None = None,
    description: str | None = None,
    draft: bool = True,
) -> WandbReportResult:
    """Create a W&B report for a benchmark run group."""
    wandb, _wr, _expr = require_wandb_report_sdk()
    report_title = title or f"NeMo Anonymizer Benchmark Sweep: {group}"
    report_description = description or "SDK-generated benchmark sweep report for NeMo Anonymizer measurement runs."
    report = build_benchmark_group_report(
        project_path,
        group=group,
        title=report_title,
        description=report_description,
        sweep_param_columns=_discover_sweep_param_columns(wandb, project_path=project_path, group=group),
    )
    _save_report(report, draft=draft)
    return WandbReportResult(
        project_path=project_path.path,
        group=group,
        report_url=str(report.url),
        draft=draft,
        title=report_title,
    )


def build_benchmark_workspace(
    project_path: WandbProjectPath,
    *,
    group: str | None,
    title: str,
) -> Any:
    """Build an unsaved manual W&B workspace for benchmark runs."""
    ws, wr = require_wandb_workspace_sdk()
    groupby = "config:sweep_arm_id" if group else None
    return ws.Workspace(
        entity=project_path.entity,
        project=project_path.project,
        name=title,
        auto_generate_panels=False,
        settings=_benchmark_workspace_settings(ws),
        runset_settings=_benchmark_runset_settings(ws, group=group),
        sections=_benchmark_workspace_sections(ws, wr, groupby=groupby),
    )


def build_benchmark_report(run: Any, *, run_path: WandbRunPath, title: str, description: str) -> Any:
    """Build an unsaved report object from a W&B run."""
    _wandb, wr, expr = require_wandb_report_sdk()
    summary = dict(getattr(run, "summary", {}) or {})
    config = dict(getattr(run, "config", {}) or {})
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
    return wr.Report(
        entity=run_path.entity,
        project=run_path.project,
        title=title,
        description=description,
        width="fluid",
        blocks=[
            wr.H1(title),
            wr.MarkdownBlock(build_report_markdown(run, summary=summary, config=config)),
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
) -> Any:
    """Build an unsaved report object for a W&B run group."""
    _wandb, wr, expr = require_wandb_report_sdk()
    runset = wr.Runset(
        entity=project_path.entity,
        project=project_path.project,
        name="Benchmark sweep",
        filters=[expr.Metric("Group") == group],
        groupby=["config:sweep_arm_id"],
        pinned_columns=["Name", "State", "CreatedTimestamp", "Group", "Tags"],
        visible_columns=_sweep_visible_columns(sweep_param_columns),
    )
    return wr.Report(
        entity=project_path.entity,
        project=project_path.project,
        title=title,
        description=description,
        width="fluid",
        blocks=[
            wr.H1(title),
            wr.MarkdownBlock(build_group_report_markdown(group=group)),
            wr.H2("Sweep Panels"),
            wr.PanelGrid(runsets=[runset], panels=_sweep_panels(wr)),
        ],
    )


def _single_run_visible_columns() -> list[str]:
    return _summary_columns(_all_report_metrics())


def _sweep_visible_columns(sweep_param_columns: list[str] | None = None) -> list[str]:
    return [
        "config:sweep_id",
        "config:sweep_arm_id",
        "config:sweep_params",
        *(sweep_param_columns or []),
        *_summary_columns(_all_report_metrics()),
    ]


def _sweep_panels(wr: Any) -> list[Any]:
    return _benchmark_panels(wr, groupby="config:sweep_arm_id")


def _benchmark_workspace_settings(ws: Any) -> Any:
    return ws.WorkspaceSettings(
        sort_panels_alphabetically=False,
        group_by_prefix="first",
        max_runs=10,
    )


def _benchmark_runset_settings(ws: Any, *, group: str | None) -> Any:
    return ws.RunsetSettings(
        filters=_benchmark_run_filters(ws, group=group),
        groupby=_benchmark_run_groupby(ws, group=group),
        order=[ws.Ordering(ws.Metric("CreatedTimestamp"), ascending=False)],
        pinned_columns=["Name", "State", "CreatedTimestamp", "Group", "JobType", "Tags"],
    )


def _benchmark_run_filters(ws: Any, *, group: str | None) -> Any:
    job_filter = ws.Or(*(ws.Metric("JobType") == job_type for job_type in _WORKSPACE_JOB_FILTERS))
    return ws.And(job_filter, ws.Metric("Group") == group) if group else job_filter


def _benchmark_run_groupby(ws: Any, *, group: str | None) -> list[Any]:
    return [ws.Config("sweep_arm_id")] if group else [ws.Metric("Group")]


def _benchmark_workspace_sections(ws: Any, wr: Any, *, groupby: str | None) -> list[Any]:
    return [
        ws.Section(name="Benchmark Summary", panels=_summary_workspace_panels(wr, groupby=groupby), is_open=True),
        ws.Section(name="Privacy", panels=_privacy_workspace_panels(wr, groupby=groupby), is_open=True),
        ws.Section(name="Utility", panels=_utility_workspace_panels(wr, groupby=groupby), is_open=True),
        ws.Section(name="Cost/Throughput", panels=_cost_workspace_panels(wr, groupby=groupby), is_open=True),
        ws.Section(name="Sweep Comparison", panels=_comparison_workspace_panels(wr), is_open=True),
        ws.Section(name="Tables", panels=_table_workspace_panels(wr), is_open=False),
    ]


def _summary_workspace_panels(wr: Any, *, groupby: str | None) -> list[Any]:
    return [
        *(
            wr.ScalarChart(title=_metric_title(metric), metric=metric, groupby_aggfunc="mean")
            for metric in _WORKSPACE_SUMMARY_SCALARS
        ),
        _bar_panel(wr, "Case Health", _CASE_HEALTH_METRICS, groupby=groupby),
        _bar_panel(wr, "Case Latency", _CASE_LATENCY_METRICS, groupby=groupby),
    ]


def _privacy_workspace_panels(wr: Any, *, groupby: str | None) -> list[Any]:
    return [
        _bar_panel(wr, "Privacy Outcomes", _RECORD_PRIVACY_METRICS, groupby=groupby),
        _bar_panel(wr, "Replacement Quality", _REPLACEMENT_QUALITY_METRICS, groupby=groupby),
    ]


def _utility_workspace_panels(wr: Any, *, groupby: str | None) -> list[Any]:
    return [_bar_panel(wr, "Rewrite Utility", _REWRITE_UTILITY_METRICS, groupby=groupby)]


def _cost_workspace_panels(wr: Any, *, groupby: str | None) -> list[Any]:
    return [
        _bar_panel(wr, "NDD Request Health", _NDD_REQUEST_HEALTH_METRICS, groupby=groupby),
        _bar_panel(wr, "NDD Token Usage", _NDD_TOKEN_METRICS, groupby=groupby),
        _bar_panel(wr, "NDD Throughput", _NDD_THROUGHPUT_METRICS, groupby=groupby),
        _bar_panel(wr, "Stage Throughput", _STAGE_THROUGHPUT_METRICS, groupby=groupby),
    ]


def _comparison_workspace_panels(wr: Any) -> list[Any]:
    return [
        wr.RunComparer(diff_only=True),
        wr.ParallelCoordinatesPlot(
            title="Sweep Parameter Tradeoffs",
            columns=[wr.ParallelCoordinatesPlotColumn(metric=column) for column in _WORKSPACE_COMPARISON_COLUMNS],
        ),
        wr.ParameterImportancePlot(with_respect_to="measurement/record/weighted_leakage_rate_mean"),
    ]


def _table_workspace_panels(wr: Any) -> list[Any]:
    return [wr.MediaBrowser(title="Sanitized Measurement Tables", media_keys=_MEASUREMENT_TABLE_KEYS, mode="grid")]


def _metric_title(metric: str) -> str:
    return metric.rsplit("/", maxsplit=1)[-1].replace("_", " ").title()


def _benchmark_panels(wr: Any, *, groupby: str | None = None) -> list[Any]:
    return [
        *_metric_panels(wr, groupby=groupby),
        wr.MediaBrowser(title="Sanitized Measurement Tables", media_keys=_MEASUREMENT_TABLE_KEYS, mode="grid"),
    ]


def _metric_panels(wr: Any, *, groupby: str | None) -> list[Any]:
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


def _bar_panel(wr: Any, title: str, metrics: list[str], *, groupby: str | None = None) -> Any:
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


def _discover_sweep_param_columns(wandb: Any, *, project_path: WandbProjectPath, group: str) -> list[str]:
    try:
        runs = wandb.Api(timeout=60).runs(project_path.path, filters={"group": group})
        keys = sorted(
            {key for run in runs for key in dict(getattr(run, "config", {}) or {}) if key.startswith("sweep_param_")}
        )
    except Exception as exc:  # noqa: BLE001 -- report generation should still work without discovery
        logger.info("Could not discover W&B sweep parameter columns: %s", exc)
        return []
    return [f"config:{key}" for key in keys]


def build_group_report_markdown(*, group: str) -> str:
    return f"""### Sweep Summary

This report compares benchmark runs in W&B group `{group}`.

Each run is one sweep arm. The run table exposes `sweep_id`, `sweep_arm_id`, case health, throughput, request/token usage, and privacy/replacement counters.

### Privacy Boundary

This report is built from benchmark W&B summary/config fields. The benchmark runner sanitizes these fields before upload and excludes raw text, prompts, model responses, replacement maps, entity payloads, trace records, paths, URLs, provider config payloads, and sensitive-looking run tags.
"""


def build_report_markdown(run: Any, *, summary: dict[str, Any], config: dict[str, Any]) -> str:
    """Render report markdown from sanitized W&B summary/config data."""
    benchmark = _dict(config.get("benchmark"))
    runtime = _dict(config.get("runtime"))
    git = _dict(config.get("git"))
    workload_lines = [_workload_line(item) for item in _list(config.get("workloads")) if isinstance(item, dict)]
    config_lines = [_config_line(item) for item in _list(config.get("configs")) if isinstance(item, dict)]
    return f"""### Run Summary

[{getattr(run, "name", "benchmark run")}]({getattr(run, "url", "")}) finished with **{_int_metric(summary, "benchmark/case_completed")}/{_int_metric(summary, "benchmark/case_total")} cases completed** and **{_int_metric(summary, "benchmark/case_errored")} errors**.

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
| Suite | `{_metadata_value(benchmark.get("suite_id"))}` |
| Suite file hash | `{_metadata_value(benchmark.get("suite_file_hash"))}` |
| Git branch | `{_metadata_value(git.get("branch"))}` |
| Git commit | `{_metadata_value(git.get("commit"))}` |
| Git dirty | `{_metadata_value(git.get("dirty"))}` |
| Anonymizer | `{_metadata_value(runtime.get("anonymizer_version"))}` |
| DataDesigner | `{_metadata_value(runtime.get("datadesigner_version"))}` |
| W&B | `{_metadata_value(runtime.get("wandb_version"))}` |

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
    import wandb
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


def _default_report_title(run: Any) -> str:
    name = str(getattr(run, "name", "") or "Benchmark Run")
    return f"NeMo Anonymizer Benchmark: {name}"[:128]


def _default_workspace_title(*, group: str | None) -> str:
    suffix = f": {group}" if group else ""
    return f"NeMo Anonymizer Benchmark Workspace{suffix}"[:128]


def _workload_line(item: dict[str, Any]) -> str:
    source_suffix = _metadata_value(item.get("source_suffix"))
    source_kind = _metadata_value(item.get("source_kind"))
    return (
        f"- `{_metadata_value(item.get('id'))}`: source_kind=`{source_kind}`, "
        f"source_suffix=`{source_suffix}`, row_limit={_metadata_value(item.get('row_limit'))}, "
        f"text_column=`{_metadata_value(item.get('text_column'))}`"
    )


def _config_line(item: dict[str, Any]) -> str:
    detect = _dict(item.get("detect"))
    replace = _dict(item.get("replace"))
    rewrite = _dict(item.get("rewrite"))
    parts = [
        f"strategy=`{_metadata_value(replace.get('strategy') or ('rewrite' if rewrite else None))}`",
        f"entity_label_count={_metadata_value(detect.get('entity_label_count'))}",
        f"gliner_threshold={_metadata_value(detect.get('gliner_threshold'))}",
    ]
    if rewrite:
        parts.append(f"risk_tolerance=`{_metadata_value(rewrite.get('risk_tolerance'))}`")
    return f"- `{_metadata_value(item.get('id'))}`: " + ", ".join(parts)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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


def _metadata_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        return str(value)
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


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
            result = create_benchmark_workspace(project_path, group=group, title=title)
        elif group:
            project_path = parse_wandb_project_path(target)
            result = create_benchmark_group_report(
                project_path,
                group=group,
                title=title,
                description=description,
                draft=not publish,
            )
        else:
            run_path = parse_wandb_run_path(target)
            result = create_benchmark_report(
                run_path,
                title=title,
                description=description,
                draft=not publish,
                timeout=timeout,
            )
    except (ImportError, ValueError) as exc:
        log_bad_input(logger, str(exc))
        raise SystemExit(125) from exc
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")


if __name__ == "__main__":
    app()
