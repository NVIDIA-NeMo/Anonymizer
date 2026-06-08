#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Screen strategy-comparison CSVs across benchmark analysis directories.

Usage:
    uv run python tools/measurement/screen_strategy_comparisons.py benchmark-runs/
    uv run python tools/measurement/screen_strategy_comparisons.py benchmark-runs/ --output strategy-screen.csv
    uv run python tools/measurement/screen_strategy_comparisons.py run-a/analysis run-b/analysis --json
"""

from __future__ import annotations

import ast
import json
import logging
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import cyclopts
import pandas as pd
from pydantic import BaseModel, Field, model_validator

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.strategy_screen")

COMPARISON_COLUMNS = {
    "workload_id",
    "baseline_config_id",
    "candidate_config_id",
    "safety_verdict",
    "performance_verdict",
    "candidate_verdict",
}


class ExportFormat(StrEnum):
    parquet = "parquet"
    csv = "csv"
    jsonl = "jsonl"


class LogFormat(StrEnum):
    plain = "plain"
    json = "json"


class GroupBy(StrEnum):
    strategy = "strategy"
    strategy_workload_family = "strategy_workload_family"
    strategy_workload = "strategy_workload"


class ScreenRow(BaseModel):
    source_path: str
    workload_id: str
    workload_family: str | None = None
    baseline_config_id: str
    candidate_config_id: str
    baseline_strategy: str | None = None
    candidate_strategy: str | None = None
    baseline_replacement_strategy: str | None = None
    candidate_replacement_strategy: str | None = None
    baseline_case_count: int | None = None
    candidate_case_count: int | None = None
    value_protection_verdict: str | None = None
    signature_parity_verdict: str | None = None
    safety_verdict: str
    performance_verdict: str
    candidate_verdict: str
    evidence_level: str = "legacy"
    pipeline_elapsed_sec_delta_pct: float | None = None
    observed_total_requests_delta: float | None = None
    observed_total_tokens_delta: float | None = None
    final_entity_count_delta: float | None = None
    augmented_entity_count_delta: float | None = None
    augmented_new_final_value_count_delta: float | None = None
    baseline_original_value_leak_count: float | None = None
    candidate_original_value_leak_count: float | None = None
    original_value_leak_count_delta: float | None = None
    baseline_original_value_leak_record_count: float | None = None
    candidate_original_value_leak_record_count: float | None = None
    original_value_leak_record_count_delta: float | None = None
    baseline_replacement_missing_final_entity_count: float | None = None
    candidate_replacement_missing_final_entity_count: float | None = None
    replacement_missing_final_entity_count_delta: float | None = None
    baseline_replacement_synthetic_original_collision_count: float | None = None
    candidate_replacement_synthetic_original_collision_count: float | None = None
    replacement_synthetic_original_collision_count_delta: float | None = None
    baseline_replacement_synthetic_original_collision_value_count: float | None = None
    candidate_replacement_synthetic_original_collision_value_count: float | None = None
    replacement_synthetic_original_collision_value_count_delta: float | None = None
    baseline_duplicate_synthetic_replacement_count: float | None = None
    candidate_duplicate_synthetic_replacement_count: float | None = None
    duplicate_synthetic_replacement_count_delta: float | None = None
    baseline_only_final_entity_signature_count: float | None = None
    candidate_only_final_entity_signature_count: float | None = None
    shared_final_entity_signature_count: float | None = None
    baseline_stable_final_entity_signature_count: int | None = None
    candidate_stable_final_entity_signature_count: int | None = None
    baseline_stable_candidate_unstable_final_entity_signature_count: float | None = None
    candidate_stable_baseline_unstable_final_entity_signature_count: float | None = None
    shared_stable_final_entity_signature_count: int | None = None
    flags: list[str] = Field(default_factory=list)
    baseline_only_label_counts: dict[str, int] = Field(default_factory=dict)
    label_mismatch_label_counts: dict[str, int] = Field(default_factory=dict)
    stable_lost_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_replacement_missing_final_entity_label_counts: dict[str, int] = Field(default_factory=dict)
    candidate_replacement_missing_final_entity_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_original_value_leak_label_counts: dict[str, int] = Field(default_factory=dict)
    candidate_original_value_leak_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_replacement_synthetic_original_collision_label_counts: dict[str, int] = Field(default_factory=dict)
    candidate_replacement_synthetic_original_collision_label_counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def fill_workload_family(self) -> "ScreenRow":
        if self.workload_family is None:
            self.workload_family = workload_family(self.workload_id)
        return self


class ScreenSummary(BaseModel):
    viable_count: int = 0
    review_count: int = 0
    reject_count: int = 0
    candidate_verdict_counts: dict[str, int] = Field(default_factory=dict)
    value_protection_verdict_counts: dict[str, int] = Field(default_factory=dict)
    signature_parity_verdict_counts: dict[str, int] = Field(default_factory=dict)
    safety_verdict_counts: dict[str, int] = Field(default_factory=dict)
    performance_verdict_counts: dict[str, int] = Field(default_factory=dict)
    evidence_level_counts: dict[str, int] = Field(default_factory=dict)


class ScreenGroup(BaseModel):
    group_key: str
    candidate_strategy: str | None = None
    candidate_replacement_strategy: str | None = None
    candidate_config_ids: list[str] = Field(default_factory=list)
    workload_ids: list[str] = Field(default_factory=list)
    workload_families: list[str] = Field(default_factory=list)
    row_count: int = 0
    min_baseline_case_count: int | None = None
    min_candidate_case_count: int | None = None
    viable_count: int = 0
    review_count: int = 0
    reject_count: int = 0
    has_conflicting_verdicts: bool = False
    recommendation: str = "unknown"
    value_protection_verdict_counts: dict[str, int] = Field(default_factory=dict)
    signature_parity_verdict_counts: dict[str, int] = Field(default_factory=dict)
    performance_verdict_counts: dict[str, int] = Field(default_factory=dict)
    evidence_level_counts: dict[str, int] = Field(default_factory=dict)
    split_verdict_candidate_verdict_counts: dict[str, int] = Field(default_factory=dict)
    best_pipeline_elapsed_sec_delta_pct: float | None = None
    best_observed_total_tokens_delta: float | None = None
    best_observed_total_requests_delta: float | None = None
    worst_pipeline_elapsed_sec_delta_pct: float | None = None
    worst_observed_total_tokens_delta: float | None = None
    worst_observed_total_requests_delta: float | None = None
    min_shared_stable_final_entity_signature_count: int | None = None
    sum_baseline_replacement_missing_final_entity_count: float | None = None
    sum_candidate_replacement_missing_final_entity_count: float | None = None
    sum_baseline_original_value_leak_count: float | None = None
    sum_candidate_original_value_leak_count: float | None = None
    sum_baseline_original_value_leak_record_count: float | None = None
    sum_candidate_original_value_leak_record_count: float | None = None
    sum_baseline_replacement_synthetic_original_collision_count: float | None = None
    sum_candidate_replacement_synthetic_original_collision_count: float | None = None
    sum_baseline_replacement_synthetic_original_collision_value_count: float | None = None
    sum_candidate_replacement_synthetic_original_collision_value_count: float | None = None
    sum_baseline_duplicate_synthetic_replacement_count: float | None = None
    sum_candidate_duplicate_synthetic_replacement_count: float | None = None
    baseline_defect_improvement_count: int = 0
    flag_counts: dict[str, int] = Field(default_factory=dict)
    baseline_only_label_counts: dict[str, int] = Field(default_factory=dict)
    label_mismatch_label_counts: dict[str, int] = Field(default_factory=dict)
    stable_lost_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_replacement_missing_final_entity_label_counts: dict[str, int] = Field(default_factory=dict)
    candidate_replacement_missing_final_entity_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_original_value_leak_label_counts: dict[str, int] = Field(default_factory=dict)
    candidate_original_value_leak_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_replacement_synthetic_original_collision_label_counts: dict[str, int] = Field(default_factory=dict)
    candidate_replacement_synthetic_original_collision_label_counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def fill_recommendation(self) -> "ScreenGroup":
        if self.recommendation == "unknown":
            self.recommendation = group_recommendation(self)
        return self


class ScreenResult(BaseModel):
    input_paths: list[str]
    scanned_file_count: int
    comparison_file_count: int
    row_count: int
    duplicate_row_count: int = 0
    summary: ScreenSummary = Field(default_factory=ScreenSummary)
    rows: list[ScreenRow] = Field(default_factory=list)
    groups: list[ScreenGroup] = Field(default_factory=list)


_log_format = LogFormat.plain


def configure_logging(log_format: LogFormat) -> None:
    global _log_format

    _log_format = log_format
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def log_bad_input(error: str) -> None:
    if _log_format == LogFormat.json:
        payload = {"level": "error", "event": "bad_input", "error": error}
        sys.stderr.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        return
    logger.error("bad_input error=%s", error)


def screen_comparison_paths(
    paths: list[Path],
    *,
    group_by: GroupBy = GroupBy.strategy,
    config_aliases: dict[str, str] | None = None,
    source_includes: list[str] | None = None,
    source_excludes: list[str] | None = None,
) -> ScreenResult:
    files = filter_source_paths(
        list(iter_csv_files(paths)),
        includes=source_includes or [],
        excludes=source_excludes or [],
    )
    rows: list[ScreenRow] = []
    comparison_file_count = 0
    for csv_file in files:
        table = read_csv_or_empty(csv_file)
        if table is None:
            continue
        if not is_comparison_table(table):
            continue
        comparison_file_count += 1
        rows.extend(build_rows_from_table(table, source_path=csv_file))
    deduped_rows = sorted(dedupe_rows(rows), key=screen_sort_key)
    groups = sorted(
        summarize_groups(deduped_rows, group_by=group_by, config_aliases=config_aliases or {}),
        key=group_sort_key,
    )
    return ScreenResult(
        input_paths=[str(path) for path in paths],
        scanned_file_count=len(files),
        comparison_file_count=comparison_file_count,
        row_count=len(deduped_rows),
        duplicate_row_count=len(rows) - len(deduped_rows),
        summary=summarize_rows(deduped_rows),
        rows=deduped_rows,
        groups=groups,
    )


def iter_csv_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            raise ValueError(f"comparison path does not exist: {path}")
        if path.is_file():
            if path.suffix == ".csv":
                files.append(path)
            continue
        files.extend(sorted(path.rglob("*.csv")))
    return sorted(set(files))


def filter_source_paths(paths: list[Path], *, includes: list[str], excludes: list[str]) -> list[Path]:
    return [path for path in paths if source_path_matches(path, includes=includes, excludes=excludes)]


def source_path_matches(path: Path, *, includes: list[str], excludes: list[str]) -> bool:
    text = str(path)
    if includes and not any(fragment in text for fragment in includes):
        return False
    return not any(fragment in text for fragment in excludes)


def is_comparison_table(table: pd.DataFrame) -> bool:
    if "source_path" in table.columns:
        return False
    return COMPARISON_COLUMNS.issubset(set(table.columns))


def read_csv_or_empty(csv_file: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(csv_file)
    except pd.errors.EmptyDataError:
        return None


def read_config_aliases(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"config aliases file is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("config aliases file must contain a JSON object mapping config IDs to aliases")
    return {str(key): str(value) for key, value in payload.items()}


def build_rows_from_table(table: pd.DataFrame, *, source_path: Path) -> list[ScreenRow]:
    return [build_row(row, source_path=source_path) for _, row in table.iterrows()]


def build_row(row: pd.Series, *, source_path: Path) -> ScreenRow:
    return ScreenRow(
        source_path=str(source_path),
        workload_id=required_string(row, "workload_id"),
        workload_family=workload_family(required_string(row, "workload_id")),
        baseline_config_id=required_string(row, "baseline_config_id"),
        candidate_config_id=required_string(row, "candidate_config_id"),
        baseline_strategy=optional_string(row.get("baseline_strategy")),
        candidate_strategy=optional_string(row.get("candidate_strategy")),
        baseline_replacement_strategy=optional_string(row.get("baseline_replacement_strategy")),
        candidate_replacement_strategy=optional_string(row.get("candidate_replacement_strategy")),
        baseline_case_count=optional_int(row.get("baseline_case_count")),
        candidate_case_count=optional_int(row.get("candidate_case_count")),
        value_protection_verdict=optional_string(row.get("value_protection_verdict")),
        signature_parity_verdict=optional_string(row.get("signature_parity_verdict")),
        safety_verdict=required_string(row, "safety_verdict"),
        performance_verdict=required_string(row, "performance_verdict"),
        candidate_verdict=required_string(row, "candidate_verdict"),
        evidence_level=comparison_evidence_level(row),
        pipeline_elapsed_sec_delta_pct=optional_float(row.get("pipeline_elapsed_sec_delta_pct")),
        observed_total_requests_delta=optional_float(row.get("observed_total_requests_delta")),
        observed_total_tokens_delta=optional_float(row.get("observed_total_tokens_delta")),
        final_entity_count_delta=optional_float(row.get("final_entity_count_delta")),
        augmented_entity_count_delta=optional_float(row.get("augmented_entity_count_delta")),
        augmented_new_final_value_count_delta=optional_float(row.get("augmented_new_final_value_count_delta")),
        baseline_original_value_leak_count=optional_float(row.get("baseline_original_value_leak_count")),
        candidate_original_value_leak_count=optional_float(row.get("candidate_original_value_leak_count")),
        original_value_leak_count_delta=optional_float(row.get("original_value_leak_count_delta")),
        baseline_original_value_leak_record_count=optional_float(row.get("baseline_original_value_leak_record_count")),
        candidate_original_value_leak_record_count=optional_float(
            row.get("candidate_original_value_leak_record_count")
        ),
        original_value_leak_record_count_delta=optional_float(row.get("original_value_leak_record_count_delta")),
        baseline_replacement_missing_final_entity_count=optional_float(
            row.get("baseline_replacement_missing_final_entity_count")
        ),
        candidate_replacement_missing_final_entity_count=optional_float(
            row.get("candidate_replacement_missing_final_entity_count")
        ),
        replacement_missing_final_entity_count_delta=optional_float(
            row.get("replacement_missing_final_entity_count_delta")
        ),
        baseline_replacement_synthetic_original_collision_count=optional_float(
            row.get("baseline_replacement_synthetic_original_collision_count")
        ),
        candidate_replacement_synthetic_original_collision_count=optional_float(
            row.get("candidate_replacement_synthetic_original_collision_count")
        ),
        replacement_synthetic_original_collision_count_delta=optional_float(
            row.get("replacement_synthetic_original_collision_count_delta")
        ),
        baseline_replacement_synthetic_original_collision_value_count=optional_float(
            row.get("baseline_replacement_synthetic_original_collision_value_count")
        ),
        candidate_replacement_synthetic_original_collision_value_count=optional_float(
            row.get("candidate_replacement_synthetic_original_collision_value_count")
        ),
        replacement_synthetic_original_collision_value_count_delta=optional_float(
            row.get("replacement_synthetic_original_collision_value_count_delta")
        ),
        baseline_duplicate_synthetic_replacement_count=optional_float(
            row.get("baseline_duplicate_synthetic_replacement_count")
        ),
        candidate_duplicate_synthetic_replacement_count=optional_float(
            row.get("candidate_duplicate_synthetic_replacement_count")
        ),
        duplicate_synthetic_replacement_count_delta=optional_float(
            row.get("duplicate_synthetic_replacement_count_delta")
        ),
        baseline_only_final_entity_signature_count=optional_float(
            row.get("baseline_only_final_entity_signature_count")
        ),
        candidate_only_final_entity_signature_count=optional_float(
            row.get("candidate_only_final_entity_signature_count")
        ),
        shared_final_entity_signature_count=optional_float(row.get("shared_final_entity_signature_count")),
        baseline_stable_final_entity_signature_count=optional_int(
            row.get("baseline_stable_final_entity_signature_count")
        ),
        candidate_stable_final_entity_signature_count=optional_int(
            row.get("candidate_stable_final_entity_signature_count")
        ),
        baseline_stable_candidate_unstable_final_entity_signature_count=optional_float(
            row.get("baseline_stable_candidate_unstable_final_entity_signature_count")
        ),
        candidate_stable_baseline_unstable_final_entity_signature_count=optional_float(
            row.get("candidate_stable_baseline_unstable_final_entity_signature_count")
        ),
        shared_stable_final_entity_signature_count=optional_int(row.get("shared_stable_final_entity_signature_count")),
        flags=parse_flags(row.get("flags")),
        baseline_only_label_counts=preferred_count_columns(
            row,
            preferred_prefix="baseline_only_candidate_uncovered_signature_label_counts",
            fallback_prefix="baseline_only_final_entity_signature_label_counts",
            preferred_count_column="baseline_only_candidate_uncovered_signature_count",
        ),
        label_mismatch_label_counts=count_columns(
            row,
            "baseline_only_candidate_label_mismatch_signature_label_counts",
        ),
        baseline_replacement_missing_final_entity_label_counts=count_columns(
            row,
            "baseline_replacement_missing_final_entity_label_counts",
        ),
        candidate_replacement_missing_final_entity_label_counts=count_columns(
            row,
            "candidate_replacement_missing_final_entity_label_counts",
        ),
        baseline_original_value_leak_label_counts=count_columns(row, "baseline_original_value_leak_label_counts"),
        candidate_original_value_leak_label_counts=count_columns(row, "candidate_original_value_leak_label_counts"),
        baseline_replacement_synthetic_original_collision_label_counts=count_columns(
            row,
            "baseline_replacement_synthetic_original_collision_label_counts",
        ),
        candidate_replacement_synthetic_original_collision_label_counts=count_columns(
            row,
            "candidate_replacement_synthetic_original_collision_label_counts",
        ),
        stable_lost_label_counts=preferred_count_columns(
            row,
            preferred_prefix="baseline_stable_candidate_uncovered_signature_label_counts",
            fallback_prefix="baseline_stable_candidate_unstable_final_entity_signature_label_counts",
            preferred_count_column="baseline_stable_candidate_uncovered_signature_count",
        ),
    )


def comparison_evidence_level(row: pd.Series) -> str:
    if optional_string(row.get("value_protection_verdict")) and optional_string(row.get("signature_parity_verdict")):
        return "split_verdicts"
    if (
        not is_missing(row.get("baseline_stable_final_entity_signature_count"))
        or not is_missing(row.get("candidate_stable_final_entity_signature_count"))
        or not is_missing(row.get("shared_stable_final_entity_signature_count"))
    ):
        return "stable_signatures"
    if (
        not is_missing(row.get("baseline_only_final_entity_signature_count"))
        or not is_missing(row.get("candidate_only_final_entity_signature_count"))
        or not is_missing(row.get("shared_final_entity_signature_count"))
    ):
        return "signature_counts"
    return "legacy"


def summarize_rows(rows: list[ScreenRow]) -> ScreenSummary:
    candidate_counts = count_values(row.candidate_verdict for row in rows)
    return ScreenSummary(
        viable_count=candidate_counts.get("candidate_viable", 0),
        review_count=candidate_counts.get("review", 0),
        reject_count=candidate_counts.get("reject", 0),
        candidate_verdict_counts=candidate_counts,
        value_protection_verdict_counts=count_present_values(row.value_protection_verdict for row in rows),
        signature_parity_verdict_counts=count_present_values(row.signature_parity_verdict for row in rows),
        safety_verdict_counts=count_values(row.safety_verdict for row in rows),
        performance_verdict_counts=count_values(row.performance_verdict for row in rows),
        evidence_level_counts=count_values(row.evidence_level for row in rows),
    )


def summarize_groups(
    rows: list[ScreenRow],
    *,
    group_by: GroupBy = GroupBy.strategy,
    config_aliases: dict[str, str] | None = None,
) -> list[ScreenGroup]:
    return [
        build_group(group_key, group_rows)
        for group_key, group_rows in grouped_rows(
            rows,
            group_by=group_by,
            config_aliases=config_aliases or {},
        ).items()
    ]


def grouped_rows(
    rows: list[ScreenRow],
    *,
    group_by: GroupBy,
    config_aliases: dict[str, str],
) -> dict[str, list[ScreenRow]]:
    groups: dict[str, list[ScreenRow]] = {}
    for row in rows:
        groups.setdefault(group_key_for_row(row, group_by=group_by, config_aliases=config_aliases), []).append(row)
    return dict(sorted(groups.items()))


def group_key_for_row(row: ScreenRow, *, group_by: GroupBy, config_aliases: dict[str, str]) -> str:
    base = group_base_for_row(row, config_aliases=config_aliases)
    if group_by == GroupBy.strategy_workload_family:
        return f"{base}|family:{row.workload_family}"
    if group_by == GroupBy.strategy_workload:
        return f"{base}|workload:{row.workload_id}"
    return base


def group_base_for_row(row: ScreenRow, *, config_aliases: dict[str, str]) -> str:
    replacement = _non_default_replacement_strategy(row)
    if row.candidate_strategy and row.candidate_strategy != "default":
        base = f"strategy:{row.candidate_strategy}"
        return f"{base}|replacement:{replacement}" if replacement else base
    if replacement:
        return f"replacement:{replacement}"
    if alias := config_aliases.get(row.candidate_config_id):
        return f"alias:{alias}"
    return f"config:{row.candidate_config_id}"


def _non_default_replacement_strategy(row: ScreenRow) -> str | None:
    if row.candidate_replacement_strategy and row.candidate_replacement_strategy != "default":
        return row.candidate_replacement_strategy
    return None


def workload_family(workload_id: str) -> str:
    parts = [part for part in workload_id.split("-") if part]
    while parts and _is_workload_slice_suffix(parts[-1]):
        parts.pop()
    return "-".join(parts) if parts else workload_id


def _is_workload_slice_suffix(value: str) -> bool:
    return (
        value.isdigit()
        or value == "slice"
        or (value.startswith("r") and value[1:].isdigit())
        or (value.startswith("offset") and value[len("offset") :].isdigit())
    )


def build_group(group_key: str, rows: list[ScreenRow]) -> ScreenGroup:
    verdict_counts = count_values(row.candidate_verdict for row in rows)
    group = ScreenGroup(
        group_key=group_key,
        candidate_strategy=single_optional_value(row.candidate_strategy for row in rows),
        candidate_replacement_strategy=single_optional_value(row.candidate_replacement_strategy for row in rows),
        candidate_config_ids=sorted({row.candidate_config_id for row in rows}),
        workload_ids=sorted({row.workload_id for row in rows}),
        workload_families=sorted({row.workload_family or workload_family(row.workload_id) for row in rows}),
        row_count=len(rows),
        min_baseline_case_count=min_present_int(row.baseline_case_count for row in rows),
        min_candidate_case_count=min_present_int(row.candidate_case_count for row in rows),
        viable_count=verdict_counts.get("candidate_viable", 0),
        review_count=verdict_counts.get("review", 0),
        reject_count=verdict_counts.get("reject", 0),
        has_conflicting_verdicts=len(verdict_counts) > 1,
        value_protection_verdict_counts=count_present_values(row.value_protection_verdict for row in rows),
        signature_parity_verdict_counts=count_present_values(row.signature_parity_verdict for row in rows),
        performance_verdict_counts=count_values(row.performance_verdict for row in rows),
        evidence_level_counts=count_values(row.evidence_level for row in rows),
        split_verdict_candidate_verdict_counts=count_values(
            row.candidate_verdict for row in rows if row.evidence_level == "split_verdicts"
        ),
        best_pipeline_elapsed_sec_delta_pct=min_present(row.pipeline_elapsed_sec_delta_pct for row in rows),
        best_observed_total_tokens_delta=min_present(row.observed_total_tokens_delta for row in rows),
        best_observed_total_requests_delta=min_present(row.observed_total_requests_delta for row in rows),
        worst_pipeline_elapsed_sec_delta_pct=max_present(row.pipeline_elapsed_sec_delta_pct for row in rows),
        worst_observed_total_tokens_delta=max_present(row.observed_total_tokens_delta for row in rows),
        worst_observed_total_requests_delta=max_present(row.observed_total_requests_delta for row in rows),
        min_shared_stable_final_entity_signature_count=min_present_int(
            row.shared_stable_final_entity_signature_count for row in rows
        ),
        sum_baseline_replacement_missing_final_entity_count=sum_present(
            row.baseline_replacement_missing_final_entity_count for row in rows
        ),
        sum_candidate_replacement_missing_final_entity_count=sum_present(
            row.candidate_replacement_missing_final_entity_count for row in rows
        ),
        sum_baseline_original_value_leak_count=sum_present(row.baseline_original_value_leak_count for row in rows),
        sum_candidate_original_value_leak_count=sum_present(row.candidate_original_value_leak_count for row in rows),
        sum_baseline_original_value_leak_record_count=sum_present(
            row.baseline_original_value_leak_record_count for row in rows
        ),
        sum_candidate_original_value_leak_record_count=sum_present(
            row.candidate_original_value_leak_record_count for row in rows
        ),
        sum_baseline_replacement_synthetic_original_collision_count=sum_present(
            row.baseline_replacement_synthetic_original_collision_count for row in rows
        ),
        sum_candidate_replacement_synthetic_original_collision_count=sum_present(
            row.candidate_replacement_synthetic_original_collision_count for row in rows
        ),
        sum_baseline_replacement_synthetic_original_collision_value_count=sum_present(
            row.baseline_replacement_synthetic_original_collision_value_count for row in rows
        ),
        sum_candidate_replacement_synthetic_original_collision_value_count=sum_present(
            row.candidate_replacement_synthetic_original_collision_value_count for row in rows
        ),
        sum_baseline_duplicate_synthetic_replacement_count=sum_present(
            row.baseline_duplicate_synthetic_replacement_count for row in rows
        ),
        sum_candidate_duplicate_synthetic_replacement_count=sum_present(
            row.candidate_duplicate_synthetic_replacement_count for row in rows
        ),
        baseline_defect_improvement_count=sum(1 for row in rows if row_has_baseline_defect_improvement(row)),
        flag_counts=sum_string_counts(row.flags for row in rows),
        baseline_only_label_counts=sum_dict_counts(row.baseline_only_label_counts for row in rows),
        label_mismatch_label_counts=sum_dict_counts(row.label_mismatch_label_counts for row in rows),
        stable_lost_label_counts=sum_dict_counts(row.stable_lost_label_counts for row in rows),
        baseline_replacement_missing_final_entity_label_counts=sum_dict_counts(
            row.baseline_replacement_missing_final_entity_label_counts for row in rows
        ),
        candidate_replacement_missing_final_entity_label_counts=sum_dict_counts(
            row.candidate_replacement_missing_final_entity_label_counts for row in rows
        ),
        baseline_original_value_leak_label_counts=sum_dict_counts(
            row.baseline_original_value_leak_label_counts for row in rows
        ),
        candidate_original_value_leak_label_counts=sum_dict_counts(
            row.candidate_original_value_leak_label_counts for row in rows
        ),
        baseline_replacement_synthetic_original_collision_label_counts=sum_dict_counts(
            row.baseline_replacement_synthetic_original_collision_label_counts for row in rows
        ),
        candidate_replacement_synthetic_original_collision_label_counts=sum_dict_counts(
            row.candidate_replacement_synthetic_original_collision_label_counts for row in rows
        ),
    )
    group.recommendation = group_recommendation(group)
    return group


def group_recommendation(group: ScreenGroup) -> str:
    if group.viable_count and group.reject_count:
        return "conflicting_evidence"
    if group_needs_split_verdict_rerun(group):
        return "needs_split_verdict_rerun"
    if group_needs_viable_split_verdict(group):
        return "needs_viable_split_verdict"
    if group.viable_count and group.review_count and group_has_baseline_defect_improvement_reviews(group):
        return "promising_with_baseline_defect_improvements"
    if group.viable_count and group.review_count:
        return "promising_needs_review"
    if group.viable_count:
        if group.row_count == 1:
            return "single_slice_viable"
        return "candidate_family_viable"
    if group.reject_count:
        return "reject"
    if group.review_count:
        if is_baseline_defect_improvement_group(group):
            return "candidate_covers_baseline_defects"
        if is_replacement_replay_review_group(group):
            return "replacement_replay_review"
        if is_reliability_review_group(group):
            return "reliability_review"
        if is_label_policy_review_group(group):
            return "label_policy_review"
        if group.performance_verdict_counts.get("improved", 0) == group.review_count:
            return "review_only"
        if group.performance_verdict_counts.get("improved", 0) or group.performance_verdict_counts.get("mixed", 0):
            return "review_mixed_performance"
        return "no_performance_win"
    return "unknown"


def group_needs_split_verdict_rerun(group: ScreenGroup) -> bool:
    if not group.evidence_level_counts:
        return False
    if group.reject_count:
        return False
    split_verdict_count = group.evidence_level_counts.get("split_verdicts", 0)
    if split_verdict_count == group.row_count:
        return False
    if group.viable_count and group.review_count:
        return split_verdict_count == 0
    if group.review_count == group.row_count and split_verdict_count:
        return True
    return False


def group_needs_viable_split_verdict(group: ScreenGroup) -> bool:
    if not (group.viable_count and group.review_count):
        return False
    if not group.evidence_level_counts.get("split_verdicts", 0):
        return False
    return not bool(group.split_verdict_candidate_verdict_counts.get("candidate_viable", 0))


def is_replacement_replay_review_group(group: ScreenGroup) -> bool:
    if not group.candidate_replacement_strategy or group.candidate_replacement_strategy == "default":
        return False
    if group.review_count != group.row_count:
        return False
    if group.performance_verdict_counts.get("improved", 0) != group.review_count:
        return False
    return bool(group.flag_counts.get("replacement_only_detection_instability", 0))


_BASELINE_DEFECT_IMPROVEMENT_FLAGS = {
    "candidate_covers_baseline_original_value_leak",
    "candidate_covers_baseline_replacement_missing_final_entity",
    "candidate_covers_baseline_replacement_synthetic_original_collision",
}


def row_has_baseline_defect_improvement(row: ScreenRow) -> bool:
    return bool(set(row.flags) & _BASELINE_DEFECT_IMPROVEMENT_FLAGS)


def group_has_baseline_defect_improvement_reviews(group: ScreenGroup) -> bool:
    if not group.review_count:
        return False
    return group.baseline_defect_improvement_count == group.review_count


def is_baseline_defect_improvement_group(group: ScreenGroup) -> bool:
    if group.review_count != group.row_count:
        return False
    if group.performance_verdict_counts.get("improved", 0) != group.review_count:
        return False
    if group.value_protection_verdict_counts.get("pass", 0) != group.review_count:
        return False
    if group.signature_parity_verdict_counts.get("review", 0) != group.review_count:
        return False
    return group.baseline_defect_improvement_count == group.review_count


_RELIABILITY_REVIEW_FLAGS = {"failed_request_increase", "bridge_fallback_increase"}


def is_reliability_review_group(group: ScreenGroup) -> bool:
    if group.review_count != group.row_count:
        return False
    if group.performance_verdict_counts.get("improved", 0) != group.review_count:
        return False
    return any(group.flag_counts.get(flag, 0) > 0 for flag in _RELIABILITY_REVIEW_FLAGS)


def is_label_policy_review_group(group: ScreenGroup) -> bool:
    if group.review_count != group.row_count:
        return False
    if group.performance_verdict_counts.get("improved", 0) != group.review_count:
        return False
    if group.value_protection_verdict_counts.get("pass", 0) != group.review_count:
        return False
    if group.signature_parity_verdict_counts.get("review", 0) != group.review_count:
        return False
    return bool(group.label_mismatch_label_counts or group.flag_counts.get("covered_label_mismatch"))


def group_performance_summary(group: ScreenGroup) -> str:
    if group.performance_verdict_counts:
        return label_summary(group.performance_verdict_counts)
    return "unknown"


def single_optional_value(values: object) -> str | None:
    unique = sorted({str(value) for value in values if value is not None})
    return unique[0] if len(unique) == 1 else None


def min_present(values: object) -> float | None:
    present = [float(value) for value in values if value is not None]
    return min(present) if present else None


def min_present_int(values: object) -> int | None:
    value = min_present(values)
    return int(value) if value is not None else None


def max_present(values: object) -> float | None:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None


def sum_present(values: object) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) if present else None


def sum_string_counts(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for items in values:
        for item in items:
            counts[str(item)] = counts.get(str(item), 0) + 1
    return dict(sorted(counts.items()))


def sum_dict_counts(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for mapping in values:
        for key, value in mapping.items():
            counts[str(key)] = counts.get(str(key), 0) + int(value)
    return dict(sorted(counts.items()))


def group_sort_key(group: ScreenGroup) -> tuple[int, int, float, float, str]:
    conflict_rank = 1 if group.has_conflicting_verdicts else 0
    verdict_rank = 0 if group.viable_count else 1 if group.review_count else 2
    elapsed = group.best_pipeline_elapsed_sec_delta_pct
    tokens = group.best_observed_total_tokens_delta
    return (
        conflict_rank,
        verdict_rank,
        elapsed if elapsed is not None else float("inf"),
        tokens if tokens is not None else float("inf"),
        group.group_key,
    )


def dedupe_rows(rows: list[ScreenRow]) -> list[ScreenRow]:
    deduped: dict[str, ScreenRow] = {}
    for row in rows:
        payload = row.model_dump()
        payload.pop("source_path", None)
        key = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        deduped.setdefault(key, row)
    return list(deduped.values())


def count_values(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def count_present_values(values: object) -> dict[str, int]:
    return count_values(value for value in values if value is not None)


def screen_sort_key(row: ScreenRow) -> tuple[int, float, float, str, str]:
    verdict_rank = {"candidate_viable": 0, "review": 1, "reject": 2}.get(row.candidate_verdict, 3)
    elapsed_delta = row.pipeline_elapsed_sec_delta_pct
    token_delta = row.observed_total_tokens_delta
    return (
        verdict_rank,
        elapsed_delta if elapsed_delta is not None else float("inf"),
        token_delta if token_delta is not None else float("inf"),
        row.workload_id,
        row.candidate_config_id,
    )


def required_string(row: pd.Series, column: str) -> str:
    value = optional_string(row.get(column))
    if value is None:
        raise ValueError(f"comparison row missing required value: {column}")
    return value


def optional_string(value: object) -> str | None:
    if is_missing(value):
        return None
    return str(value)


def optional_float(value: object) -> float | None:
    if is_missing(value):
        return None
    return float(value)


def optional_int(value: object) -> int | None:
    value = optional_float(value)
    return int(value) if value is not None else None


def is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def parse_flags(value: object) -> list[str]:
    if is_missing(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return [str(value)]
    parsed = parse_nested(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [value] if value else []


def parse_nested(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value


def count_columns(row: pd.Series, prefix: str) -> dict[str, int]:
    label_counts: dict[str, int] = {}
    column_prefix = f"{prefix}."
    for column, raw_value in row.items():
        if not str(column).startswith(column_prefix):
            continue
        value = optional_float(raw_value)
        if value is not None and value > 0:
            label_counts[str(column)[len(column_prefix) :]] = int(value)
    return dict(sorted(label_counts.items()))


def preferred_count_columns(
    row: pd.Series,
    *,
    preferred_prefix: str,
    fallback_prefix: str,
    preferred_count_column: str,
) -> dict[str, int]:
    if has_comparison_field(row, preferred_prefix) or not is_missing(row.get(preferred_count_column)):
        return count_columns(row, preferred_prefix)
    return count_columns(row, fallback_prefix)


def has_comparison_field(row: pd.Series, prefix: str) -> bool:
    column_prefix = f"{prefix}."
    return any(str(column).startswith(column_prefix) for column in row.index)


def write_rows(rows: list[ScreenRow], output_path: Path, export_format: ExportFormat) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.json_normalize([row.model_dump() for row in rows], sep=".")
    table = normalize_table_cells(table)
    if export_format == ExportFormat.parquet:
        table.to_parquet(output_path, index=False)
    elif export_format == ExportFormat.csv:
        table.to_csv(output_path, index=False)
    else:
        table.to_json(output_path, orient="records", lines=True)


def write_groups(groups: list[ScreenGroup], output_path: Path, export_format: ExportFormat) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([group.model_dump() for group in groups])
    table = normalize_table_cells(table)
    if export_format == ExportFormat.parquet:
        table.to_parquet(output_path, index=False)
    elif export_format == ExportFormat.csv:
        table.to_csv(output_path, index=False)
    else:
        table.to_json(output_path, orient="records", lines=True)


def normalize_table_cells(table: pd.DataFrame) -> pd.DataFrame:
    normalized = table.copy()
    for column in normalized.columns:
        if normalized[column].map(is_nested_cell).any():
            normalized[column] = normalized[column].map(json_cell)
    return normalized


def is_nested_cell(value: object) -> bool:
    return isinstance(value, dict | list)


def json_cell(value: object) -> object:
    if not is_nested_cell(value):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def render_result(result: ScreenResult, *, json_output: bool, limit: int) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    lines = [
        f"Screened {result.row_count} comparison row(s) from "
        f"{result.comparison_file_count}/{result.scanned_file_count} CSV file(s): "
        f"viable={result.summary.viable_count}, review={result.summary.review_count}, "
        f"reject={result.summary.reject_count}, duplicates_skipped={result.duplicate_row_count}",
    ]
    for row in result.rows[:limit]:
        lines.append(render_row(row))
    if len(result.rows) > limit:
        lines.append(f"... {len(result.rows) - limit} more row(s)")
    lines.append("Candidate groups:")
    for group in result.groups[:limit]:
        lines.append(render_group(group))
    if len(result.groups) > limit:
        lines.append(f"... {len(result.groups) - limit} more group(s)")
    return "\n".join(lines)


def render_row(row: ScreenRow) -> str:
    return (
        f"- {row.workload_id}: {row.baseline_config_id}->{row.candidate_config_id} "
        f"verdict={row.candidate_verdict} safety={row.safety_verdict} "
        f"value_protection={row.value_protection_verdict or 'unknown'} "
        f"signature_parity={row.signature_parity_verdict or 'unknown'} "
        f"evidence={row.evidence_level} "
        f"perf={row.performance_verdict} elapsed_delta={format_number(row.pipeline_elapsed_sec_delta_pct, '%')} "
        f"replacement={row.baseline_replacement_strategy or 'unknown'}"
        f"->{row.candidate_replacement_strategy or 'unknown'} "
        f"tokens_delta={format_number(row.observed_total_tokens_delta)} "
        f"cases={format_count(row.baseline_case_count)}/{format_count(row.candidate_case_count)} "
        f"shared_stable={format_count(row.shared_stable_final_entity_signature_count)} "
        f"aug_new_final_delta={format_number(row.augmented_new_final_value_count_delta)} "
        f"baseline_missing_replacements={format_number(row.baseline_replacement_missing_final_entity_count)} "
        f"candidate_missing_replacements={format_number(row.candidate_replacement_missing_final_entity_count)} "
        f"baseline_original_value_leaks={format_number(row.baseline_original_value_leak_count)} "
        f"candidate_original_value_leaks={format_number(row.candidate_original_value_leak_count)} "
        f"baseline_replacement_collisions="
        f"{format_number(row.baseline_replacement_synthetic_original_collision_count)} "
        f"candidate_replacement_collisions="
        f"{format_number(row.candidate_replacement_synthetic_original_collision_count)} "
        f"baseline_duplicate_synthetics={format_number(row.baseline_duplicate_synthetic_replacement_count)} "
        f"candidate_duplicate_synthetics={format_number(row.candidate_duplicate_synthetic_replacement_count)} "
        f"lost={label_summary(row.baseline_only_label_counts)} "
        f"label_mismatch={label_summary(row.label_mismatch_label_counts)} "
        f"stable_lost={label_summary(row.stable_lost_label_counts)} "
        f"baseline_missing_labels={label_summary(row.baseline_replacement_missing_final_entity_label_counts)} "
        f"candidate_missing_labels={label_summary(row.candidate_replacement_missing_final_entity_label_counts)} "
        f"baseline_leak_labels={label_summary(row.baseline_original_value_leak_label_counts)} "
        f"leak_labels={label_summary(row.candidate_original_value_leak_label_counts)} "
        f"baseline_collision_labels="
        f"{label_summary(row.baseline_replacement_synthetic_original_collision_label_counts)} "
        f"collision_labels="
        f"{label_summary(row.candidate_replacement_synthetic_original_collision_label_counts)} "
        f"flags={','.join(row.flags) if row.flags else 'none'}"
    )


def render_group(group: ScreenGroup) -> str:
    return (
        f"- {group.group_key}: rows={group.row_count} viable={group.viable_count} "
        f"review={group.review_count} reject={group.reject_count} "
        f"conflict={str(group.has_conflicting_verdicts).lower()} recommendation={group.recommendation} "
        f"replacement={group.candidate_replacement_strategy or 'unknown'} "
        f"value_protection_counts={label_summary(group.value_protection_verdict_counts)} "
        f"signature_parity_counts={label_summary(group.signature_parity_verdict_counts)} "
        f"evidence_counts={label_summary(group.evidence_level_counts)} "
        f"split_verdict_candidate_counts={label_summary(group.split_verdict_candidate_verdict_counts)} "
        f"perf_counts={group_performance_summary(group)} "
        f"best_elapsed_delta={format_number(group.best_pipeline_elapsed_sec_delta_pct, '%')} "
        f"worst_elapsed_delta={format_number(group.worst_pipeline_elapsed_sec_delta_pct, '%')} "
        f"best_tokens_delta={format_number(group.best_observed_total_tokens_delta)} "
        f"worst_tokens_delta={format_number(group.worst_observed_total_tokens_delta)} "
        f"best_requests_delta={format_number(group.best_observed_total_requests_delta)} "
        f"worst_requests_delta={format_number(group.worst_observed_total_requests_delta)} "
        f"min_cases={format_count(group.min_baseline_case_count)}/{format_count(group.min_candidate_case_count)} "
        f"min_shared_stable={format_count(group.min_shared_stable_final_entity_signature_count)} "
        f"baseline_defect_improvements={group.baseline_defect_improvement_count} "
        f"baseline_missing_replacements="
        f"{format_number(group.sum_baseline_replacement_missing_final_entity_count)} "
        f"candidate_missing_replacements="
        f"{format_number(group.sum_candidate_replacement_missing_final_entity_count)} "
        f"baseline_original_value_leaks={format_number(group.sum_baseline_original_value_leak_count)} "
        f"candidate_original_value_leaks={format_number(group.sum_candidate_original_value_leak_count)} "
        f"baseline_replacement_collisions="
        f"{format_number(group.sum_baseline_replacement_synthetic_original_collision_count)} "
        f"candidate_replacement_collisions="
        f"{format_number(group.sum_candidate_replacement_synthetic_original_collision_count)} "
        f"baseline_duplicate_synthetics={format_number(group.sum_baseline_duplicate_synthetic_replacement_count)} "
        f"candidate_duplicate_synthetics={format_number(group.sum_candidate_duplicate_synthetic_replacement_count)} "
        f"lost={label_summary(group.baseline_only_label_counts)} "
        f"label_mismatch={label_summary(group.label_mismatch_label_counts)} "
        f"stable_lost={label_summary(group.stable_lost_label_counts)} "
        f"baseline_missing_labels={label_summary(group.baseline_replacement_missing_final_entity_label_counts)} "
        f"candidate_missing_labels={label_summary(group.candidate_replacement_missing_final_entity_label_counts)} "
        f"baseline_leak_labels={label_summary(group.baseline_original_value_leak_label_counts)} "
        f"leak_labels={label_summary(group.candidate_original_value_leak_label_counts)} "
        f"baseline_collision_labels="
        f"{label_summary(group.baseline_replacement_synthetic_original_collision_label_counts)} "
        f"collision_labels="
        f"{label_summary(group.candidate_replacement_synthetic_original_collision_label_counts)} "
        f"flags={label_summary(group.flag_counts)}"
    )


def format_number(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "unknown"
    return f"{value:.1f}{suffix}"


def format_count(value: int | None) -> str:
    return str(value) if value is not None else "unknown"


def label_summary(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ",".join(f"{label}:{count}" for label, count in counts.items())


@app.default
def main(
    comparison_paths: list[Path],
    *,
    output: Annotated[Path | None, cyclopts.Parameter(("--output", "-o"))] = None,
    group_output: Annotated[Path | None, cyclopts.Parameter("--group-output")] = None,
    group_by: Annotated[GroupBy, cyclopts.Parameter("--group-by")] = GroupBy.strategy,
    config_aliases: Annotated[Path | None, cyclopts.Parameter("--config-aliases")] = None,
    source_include: Annotated[list[str] | None, cyclopts.Parameter("--source-include")] = None,
    source_exclude: Annotated[list[str] | None, cyclopts.Parameter("--source-exclude")] = None,
    format: Annotated[ExportFormat, cyclopts.Parameter("--format")] = ExportFormat.csv,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    limit: Annotated[int, cyclopts.Parameter("--limit")] = 20,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    try:
        result = screen_comparison_paths(
            comparison_paths,
            group_by=group_by,
            config_aliases=read_config_aliases(config_aliases),
            source_includes=source_include or [],
            source_excludes=source_exclude or [],
        )
    except ValueError as exc:
        log_bad_input(str(exc))
        raise SystemExit(125) from exc
    if output is not None:
        write_rows(result.rows, output, format)
    if group_output is not None:
        write_groups(result.groups, group_output, format)
    sys.stdout.write(render_result(result, json_output=json_output, limit=limit) + "\n")


if __name__ == "__main__":
    app()
