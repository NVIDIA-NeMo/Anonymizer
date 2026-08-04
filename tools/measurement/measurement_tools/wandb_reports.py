# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W&B benchmark report orchestration, construction, and persistence."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

from measurement_tools.wandb_models import ConfigMetadata, ResolvedWandbConfig, WorkloadMetadata
from measurement_tools.wandb_report_catalog import (
    all_report_metrics,
    benchmark_panels,
    group_visible_columns,
    single_run_visible_columns,
)
from measurement_tools.wandb_report_contracts import WandbReportResult
from measurement_tools.wandb_report_models import (
    GroupComparison,
    WandbProjectPath,
    WandbRunPath,
    WandbRunView,
    group_comparison,
    parse_wandb_run_view,
    validate_wandb_returned_url,
)
from measurement_tools.wandb_report_sdk import (
    read_group_views,
    report_settings,
    require_wandb_report_sdk,
    sweep_param_columns,
)
from measurement_tools.wandb_report_text import (
    code_span,
    escape_heading,
    escape_link_label,
    escape_list_text,
    escape_markdown_text,
    plain_text,
    table_code,
)
from measurement_tools.wandb_setup import WandbSdkEnvironment, require_wandb

logger = logging.getLogger("measurement.wandb_report")


def create_benchmark_report(
    run_path: WandbRunPath,
    *,
    settings: ResolvedWandbConfig | None = None,
    title: str | None = None,
    description: str | None = None,
    draft: bool = True,
    timeout: int = 60,
    sdk_loader: Callable[[], tuple[Any, Any, Any]] = require_wandb_report_sdk,
    report_builder: Callable[..., Any] | None = None,
    report_saver: Callable[..., Any] | None = None,
) -> WandbReportResult:
    """Create a W&B report for one benchmark run."""
    builder = report_builder or build_benchmark_report
    saver = report_saver or save_report
    resolved = report_settings(settings, run_path)
    effective_run_path = run_path.with_base_url(resolved.wandb_base_url or run_path.base_url)
    with WandbSdkEnvironment(resolved):
        wandb, wr, expr = sdk_loader()
        run = wandb.Api(timeout=timeout).run(run_path.path)
        view = parse_wandb_run_view(
            run,
            run_path=effective_run_path,
            allowed_metrics=frozenset(all_report_metrics()),
        )
        report_title = plain_text(title) if title is not None else default_report_title(view)
        report_description = plain_text(
            description or "SDK-generated benchmark report for a NeMo Anonymizer measurement run."
        )
        report = builder(
            view,
            title=report_title,
            description=report_description,
            wr=wr,
            expr=expr,
        )
        saver(report, draft=draft)
        report_url = validate_wandb_returned_url(report.url, expected_base_url=effective_run_path.base_url)
    return WandbReportResult(
        run_path=run_path.path,
        run_url=effective_run_path.url,
        project_path=f"{run_path.entity}/{run_path.project}",
        report_url=report_url,
        draft=draft,
        title=report_title,
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
    sdk_loader: Callable[[], tuple[Any, Any, Any]] = require_wandb_report_sdk,
    report_builder: Callable[..., Any] | None = None,
    report_saver: Callable[..., Any] | None = None,
) -> WandbReportResult:
    """Create a W&B report for a benchmark run group."""
    builder = report_builder or build_benchmark_group_report
    saver = report_saver or save_report
    resolved = report_settings(settings, project_path)
    effective_project_path = project_path.with_base_url(resolved.wandb_base_url or project_path.base_url)
    with WandbSdkEnvironment(resolved):
        wandb, wr, expr = sdk_loader()
        views = read_group_views(wandb, project_path=effective_project_path, group=group)
        comparison = group_comparison(views, expected_run_kind=expected_run_kind)
        report_title = (
            plain_text(title) if title is not None else f"NeMo Anonymizer Benchmark Group: {plain_text(group)}"
        )
        report_description = plain_text(
            description or "SDK-generated benchmark sweep report for NeMo Anonymizer measurement runs."
        )
        report = builder(
            effective_project_path,
            group=group,
            title=report_title,
            description=report_description,
            sweep_param_columns=sweep_param_columns(views),
            comparison=comparison,
            wr=wr,
            expr=expr,
        )
        saver(report, draft=draft)
        report_url = validate_wandb_returned_url(report.url, expected_base_url=effective_project_path.base_url)
    return WandbReportResult(
        project_path=project_path.path,
        group=group,
        report_url=report_url,
        draft=draft,
        title=report_title,
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
        visible_columns=single_run_visible_columns(),
        run_settings={run_id: wr.RunSettings(color="#2F80ED", disabled=False)},
    )
    safe_title = escape_heading(title)
    return wr.Report(
        entity=run_path.entity,
        project=run_path.project,
        title=safe_title,
        description=escape_markdown_text(description),
        width="fluid",
        blocks=[
            wr.H1(safe_title),
            wr.MarkdownBlock(build_report_markdown(view)),
            wr.H2("Run Panels"),
            wr.PanelGrid(runsets=[runset], panels=benchmark_panels(wr)),
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
        visible_columns=group_visible_columns(resolved_comparison, sweep_param_columns),
    )
    safe_title = escape_heading(title)
    return wr.Report(
        entity=project_path.entity,
        project=project_path.project,
        title=safe_title,
        description=escape_markdown_text(description),
        width="fluid",
        blocks=[
            wr.H1(safe_title),
            wr.MarkdownBlock(build_group_report_markdown(group=group, comparison=resolved_comparison)),
            wr.H2("Sweep Panels"),
            wr.PanelGrid(
                runsets=[runset],
                panels=benchmark_panels(wr, groupby=wr.Config(resolved_comparison.config_key)),
            ),
        ],
    )


def build_group_report_markdown(*, group: str, comparison: GroupComparison | None = None) -> str:
    resolved = comparison or GroupComparison(run_kind="sweep_arm", config_key="sweep_arm_id", label="Sweep Arm")
    return f"""### Group Summary

This report compares benchmark runs in W&B group {code_span(group)}.

Run kind: {code_span(resolved.run_kind)}. Comparison axis: {code_span(resolved.label)}.

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
    workload_lines = [workload_line(item) for item in metadata.workloads]
    config_lines = [config_line(item) for item in metadata.configs]
    return f"""### Run Summary

[{escape_link_label(view.name)}]({view.path.url}) finished with **{int_metric(summary, "benchmark/case_completed")}/{int_metric(summary, "benchmark/case_total")} cases completed** and **{int_metric(summary, "benchmark/case_errored")} errors**.

| Metric | Value |
| --- | ---: |
| Case success rate | {number_metric(summary, "benchmark/case_success_rate")} |
| Mean case elapsed seconds | {number_metric(summary, "benchmark/case_elapsed_sec_mean")} |
| Final entities | {int_metric(summary, "measurement/record/final_entity_count")} |
| Original value leaks | {int_metric(summary, "measurement/record/original_value_leak_count")} |
| Model requests | {int_metric(summary, "measurement/ndd_workflow/observed_total_requests")} |
| Failed model requests | {int_metric(summary, "measurement/ndd_workflow/observed_failed_requests")} |
| Total model tokens | {int_metric(summary, "measurement/ndd_workflow/observed_total_tokens")} |

### Benchmark Metadata

| Field | Value |
| --- | --- |
| Run kind | {table_code(metadata.run_kind)} |
| Suite | {table_code(benchmark.suite_id if benchmark else None)} |
| Suite file hash | {table_code(benchmark.suite_file_hash if benchmark else None)} |
| Git branch | {table_code(git.branch if git else None)} |
| Git commit | {table_code(git.commit if git else None)} |
| Git dirty | {table_code(git.dirty if git else None)} |
| Anonymizer | {table_code(runtime.anonymizer_version if runtime else None)} |
| DataDesigner | {table_code(runtime.datadesigner_version if runtime else None)} |
| W&B | {table_code(runtime.wandb_version if runtime else None)} |

### Workloads

{chr(10).join(workload_lines) or "- none"}

### Configs

{chr(10).join(config_lines) or "- none"}

### Privacy Boundary

This report is built from benchmark W&B summary/config fields. The benchmark runner sanitizes these fields before upload and excludes raw text, prompts, model responses, replacement maps, entity payloads, trace records, paths, URLs, provider config payloads, and sensitive-looking run tags.
"""


def save_report(report: Any, *, draft: bool) -> Any:
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


def default_report_title(view: WandbRunView) -> str:
    return f"NeMo Anonymizer Benchmark: {plain_text(view.name)}"[:128]


def workload_line(item: WorkloadMetadata) -> str:
    source_suffix = item.source.suffix if item.source else None
    source_kind = item.source.kind if item.source else None
    return (
        f"- {code_span(item.id)}: source_kind={code_span(source_kind)}, "
        f"source_suffix={code_span(source_suffix)}, row_limit={escape_list_text(item.row_limit)}, "
        f"text_column={code_span(item.text_column)}"
    )


def config_line(item: ConfigMetadata) -> str:
    detect = item.detect
    replace = item.replace
    rewrite = item.rewrite
    parts = [
        f"strategy={code_span(replace.strategy if replace else ('rewrite' if rewrite else None))}",
        f"entity_label_count={escape_list_text(detect.entity_label_count if detect else None)}",
        f"gliner_threshold={escape_list_text(detect.gliner_threshold if detect else None)}",
    ]
    if rewrite:
        parts.append(f"risk_tolerance={code_span(rewrite.risk_tolerance)}")
    return f"- {code_span(item.id)}: " + ", ".join(parts)


def metric(summary: dict[str, Any], key: str) -> int | float | None:
    value = summary.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return value
    return None


def int_metric(summary: dict[str, Any], key: str) -> int:
    value = metric(summary, key)
    return int(value) if value is not None else 0


def number_metric(summary: dict[str, Any], key: str) -> str:
    value = metric(summary, key)
    if value is None:
        return "n/a"
    return f"{float(value):.4g}"


__all__ = [
    "build_benchmark_group_report",
    "build_benchmark_report",
    "build_group_report_markdown",
    "build_report_markdown",
    "config_line",
    "create_benchmark_group_report",
    "create_benchmark_report",
    "default_report_title",
    "int_metric",
    "metric",
    "number_metric",
    "save_report",
    "workload_line",
]
