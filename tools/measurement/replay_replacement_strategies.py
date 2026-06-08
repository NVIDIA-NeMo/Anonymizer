#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay substitute strategies on one fixed detection trace.

Usage:
    uv run python tools/measurement/replay_replacement_strategies.py data.csv \
      --text-column text --labels api_key,http_cookie,password,pin,unique_id,user_name \
      --model-configs /stable-cache/anonymizer/local-vllm-json-models.yaml \
      --model-providers /stable-cache/anonymizer/local-vllm-providers.yaml \
      --dd-parser-compat raw_json --json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import cyclopts
import pandas as pd
from dd_parser_compat import DDParserCompatMode, dd_parser_compat_context
from pydantic import BaseModel, Field, ValidationError
from replacement_strategies import (
    ExperimentalReplacementStrategy,
    experimental_replacement_strategy_context,
)

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Detect
from anonymizer.config.replace_strategies import Redact, Substitute
from anonymizer.engine.constants import (
    COL_FINAL_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_REPLACEMENT_MAP_SOURCE,
)
from anonymizer.engine.schemas import EntitiesSchema, EntityReplacementMapSchema
from anonymizer.interface.anonymizer import Anonymizer, _unrename_output_columns
from anonymizer.measurement import _output_contains_original_value

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.replay_replacement_strategies")


class LogFormat(StrEnum):
    plain = "plain"
    json = "json"


class ReplayStatus(StrEnum):
    completed = "completed"
    error = "error"


class ReplacementReplayStrategy(StrEnum):
    dd_substitute = "dd_substitute"
    local_structured_substitute = "local_structured_substitute"


class ReplacementRowMetrics(BaseModel):
    final_entity_count: int = 0
    replacement_count: int = 0
    missing_count: int = 0
    collision_count: int = 0
    leak_count: int = 0
    duplicate_synthetic_count: int = 0
    missing_labels: dict[str, int] = Field(default_factory=dict)
    collision_labels: dict[str, int] = Field(default_factory=dict)
    leak_labels: dict[str, int] = Field(default_factory=dict)


class ReplacementReplaySummary(BaseModel):
    strategy: ReplacementReplayStrategy
    status: ReplayStatus
    repetition_count: int = 1
    elapsed_sec: float | None = None
    row_count: int = 0
    final_entity_count: int = 0
    replacement_count: int = 0
    missing_count: int = 0
    collision_count: int = 0
    leak_count: int = 0
    duplicate_synthetic_count: int = 0
    missing_labels: dict[str, int] = Field(default_factory=dict)
    collision_labels: dict[str, int] = Field(default_factory=dict)
    leak_labels: dict[str, int] = Field(default_factory=dict)
    replacement_map_sources: dict[str, int] = Field(default_factory=dict)
    error: str | None = None


class ReplacementReplayResult(BaseModel):
    input_path: str
    text_column: str
    labels: list[str]
    nrows: int | None = None
    replacement_repetitions: int = 1
    dd_parser_compat: DDParserCompatMode
    detect_elapsed_sec: float
    detected_final_entity_count: int
    strategies: list[ReplacementReplaySummary]


class ReplacementReplayComparisonRow(BaseModel):
    workload_id: str
    baseline_config_id: str = "dd_substitute_replay"
    candidate_config_id: str = "local_structured_substitute_replay"
    baseline_strategy: str = "default"
    candidate_strategy: str = "default"
    baseline_replacement_strategy: str = "default"
    candidate_replacement_strategy: str = "local_structured_substitute"
    baseline_case_count: int
    candidate_case_count: int
    baseline_pipeline_elapsed_sec: float | None = None
    candidate_pipeline_elapsed_sec: float | None = None
    pipeline_elapsed_sec_delta: float | None = None
    pipeline_elapsed_sec_delta_pct: float | None = None
    baseline_final_entity_count: int
    candidate_final_entity_count: int
    final_entity_count_delta: int
    baseline_replacement_count: int
    candidate_replacement_count: int
    replacement_count_delta: int
    baseline_replacement_missing_final_entity_count: int
    candidate_replacement_missing_final_entity_count: int
    replacement_missing_final_entity_count_delta: int
    baseline_replacement_missing_final_entity_label_counts: dict[str, int] = Field(default_factory=dict)
    candidate_replacement_missing_final_entity_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_replacement_synthetic_original_collision_count: int
    candidate_replacement_synthetic_original_collision_count: int
    replacement_synthetic_original_collision_count_delta: int
    baseline_replacement_synthetic_original_collision_label_counts: dict[str, int] = Field(default_factory=dict)
    candidate_replacement_synthetic_original_collision_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_duplicate_synthetic_replacement_count: int
    candidate_duplicate_synthetic_replacement_count: int
    duplicate_synthetic_replacement_count_delta: int
    baseline_original_value_leak_count: int
    candidate_original_value_leak_count: int
    original_value_leak_count_delta: int
    baseline_original_value_leak_label_counts: dict[str, int] = Field(default_factory=dict)
    candidate_original_value_leak_label_counts: dict[str, int] = Field(default_factory=dict)
    value_protection_verdict: str
    signature_parity_verdict: str
    safety_verdict: str
    performance_verdict: str
    candidate_verdict: str
    flags: list[str] = Field(default_factory=list)


_log_format = LogFormat.plain


def configure_logging(log_format: LogFormat) -> None:
    global _log_format

    _log_format = log_format
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def log_bad_input(error: str) -> None:
    if _log_format == LogFormat.json:
        sys.stderr.write(json.dumps({"level": "error", "event": "bad_input", "error": error}) + "\n")
    else:
        sys.stderr.write(f"ERROR: bad_input error={error}\n")


def parse_labels(raw: str) -> list[str]:
    labels = [label.strip() for label in raw.split(",") if label.strip()]
    if not labels:
        raise ValueError("--labels must contain at least one non-empty label")
    return list(dict.fromkeys(labels))


def run_replacement_replay(
    *,
    source: Path,
    text_column: str,
    labels: list[str],
    nrows: int | None,
    model_configs: Path | None,
    model_providers: Path | None,
    artifact_path: Path,
    dd_parser_compat: DDParserCompatMode,
    replacement_repetitions: int,
) -> ReplacementReplayResult:
    anonymizer = Anonymizer(
        model_configs=model_configs,
        model_providers=model_providers,
        artifact_path=artifact_path,
    )
    detect_elapsed, replay_df = build_replay_dataframe(
        anonymizer,
        source=source,
        text_column=text_column,
        labels=labels,
        nrows=nrows,
        dd_parser_compat=dd_parser_compat,
    )
    strategies = [
        run_strategy_repetitions(
            anonymizer,
            replay_df,
            ReplacementReplayStrategy.dd_substitute,
            dd_parser_compat,
            repetitions=replacement_repetitions,
        ),
        run_strategy_repetitions(
            anonymizer,
            replay_df,
            ReplacementReplayStrategy.local_structured_substitute,
            dd_parser_compat,
            repetitions=replacement_repetitions,
        ),
    ]
    return ReplacementReplayResult(
        input_path=str(source),
        text_column=text_column,
        labels=labels,
        nrows=nrows,
        replacement_repetitions=replacement_repetitions,
        dd_parser_compat=dd_parser_compat,
        detect_elapsed_sec=detect_elapsed,
        detected_final_entity_count=count_final_entities(replay_df),
        strategies=strategies,
    )


def build_replay_dataframe(
    anonymizer: Anonymizer,
    *,
    source: Path,
    text_column: str,
    labels: list[str],
    nrows: int | None,
    dd_parser_compat: DDParserCompatMode,
) -> tuple[float, pd.DataFrame]:
    input_data = AnonymizerInput(source=str(source), text_column=text_column)
    config = AnonymizerConfig(detect=Detect(entity_labels=labels), replace=Redact())
    with dd_parser_compat_context(dd_parser_compat):
        start = time.perf_counter()
        if nrows is None:
            detected = anonymizer.run(config=config, data=input_data)
        else:
            detected = anonymizer.preview(config=config, data=input_data, num_records=nrows)
        elapsed = time.perf_counter() - start
    internal_df = _unrename_output_columns(detected.trace_dataframe, resolved_text_column=detected.resolved_text_column)
    return elapsed, strip_replacement_columns(internal_df)


def strip_replacement_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.drop(columns=[COL_REPLACEMENT_MAP, COL_REPLACEMENT_MAP_SOURCE, COL_REPLACED_TEXT], errors="ignore")


def run_strategy_repetitions(
    anonymizer: Anonymizer,
    dataframe: pd.DataFrame,
    strategy: ReplacementReplayStrategy,
    dd_parser_compat: DDParserCompatMode,
    *,
    repetitions: int,
) -> ReplacementReplaySummary:
    summaries = [run_strategy(anonymizer, dataframe, strategy, dd_parser_compat) for _ in range(repetitions)]
    return aggregate_strategy_summaries(strategy=strategy, summaries=summaries)


def aggregate_strategy_summaries(
    *,
    strategy: ReplacementReplayStrategy,
    summaries: list[ReplacementReplaySummary],
) -> ReplacementReplaySummary:
    sources: Counter[str] = Counter()
    missing_labels: Counter[str] = Counter()
    collision_labels: Counter[str] = Counter()
    leak_labels: Counter[str] = Counter()
    errors: list[str] = []
    elapsed_values: list[float] = []
    status = ReplayStatus.completed
    for summary in summaries:
        sources.update(summary.replacement_map_sources)
        missing_labels.update(summary.missing_labels)
        collision_labels.update(summary.collision_labels)
        leak_labels.update(summary.leak_labels)
        if summary.elapsed_sec is not None:
            elapsed_values.append(summary.elapsed_sec)
        if summary.status == ReplayStatus.error:
            status = ReplayStatus.error
            if summary.error:
                errors.append(summary.error)
    return ReplacementReplaySummary(
        strategy=strategy,
        status=status,
        repetition_count=len(summaries),
        elapsed_sec=sum(elapsed_values) if elapsed_values else None,
        row_count=sum(summary.row_count for summary in summaries),
        final_entity_count=sum(summary.final_entity_count for summary in summaries),
        replacement_count=sum(summary.replacement_count for summary in summaries),
        missing_count=sum(summary.missing_count for summary in summaries),
        collision_count=sum(summary.collision_count for summary in summaries),
        leak_count=sum(summary.leak_count for summary in summaries),
        duplicate_synthetic_count=sum(summary.duplicate_synthetic_count for summary in summaries),
        missing_labels=dict(sorted(missing_labels.items())),
        collision_labels=dict(sorted(collision_labels.items())),
        leak_labels=dict(sorted(leak_labels.items())),
        replacement_map_sources=dict(sorted(sources.items())),
        error="; ".join(errors) if errors else None,
    )


def run_strategy(
    anonymizer: Anonymizer,
    dataframe: pd.DataFrame,
    strategy: ReplacementReplayStrategy,
    dd_parser_compat: DDParserCompatMode,
) -> ReplacementReplaySummary:
    try:
        start = time.perf_counter()
        result_df = execute_strategy(anonymizer, dataframe.copy(), strategy, dd_parser_compat)
        elapsed = time.perf_counter() - start
    except Exception as exc:
        logger.exception("replacement replay strategy failed: %s", strategy)
        return ReplacementReplaySummary(strategy=strategy, status=ReplayStatus.error, error=str(exc))
    return summarize_replacement_dataframe(result_df, strategy=strategy, elapsed_sec=elapsed)


def execute_strategy(
    anonymizer: Anonymizer,
    dataframe: pd.DataFrame,
    strategy: ReplacementReplayStrategy,
    dd_parser_compat: DDParserCompatMode,
) -> pd.DataFrame:
    if strategy == ReplacementReplayStrategy.local_structured_substitute:
        with experimental_replacement_strategy_context(ExperimentalReplacementStrategy.local_structured_substitute):
            return execute_substitute(anonymizer, dataframe)
    with dd_parser_compat_context(dd_parser_compat):
        return execute_substitute(anonymizer, dataframe)


def execute_substitute(anonymizer: Anonymizer, dataframe: pd.DataFrame) -> pd.DataFrame:
    result = anonymizer._replace_runner.run(  # benchmark probe against the configured runner
        dataframe,
        replace_method=Substitute(),
        model_configs=anonymizer._model_configs,
        selected_models=anonymizer._selected_models.replace,
    )
    return result.dataframe


def summarize_replacement_dataframe(
    dataframe: pd.DataFrame,
    *,
    strategy: ReplacementReplayStrategy,
    elapsed_sec: float,
) -> ReplacementReplaySummary:
    rows = [replacement_row_metrics(row) for _, row in dataframe.iterrows()]
    return ReplacementReplaySummary(
        strategy=strategy,
        status=ReplayStatus.completed,
        elapsed_sec=elapsed_sec,
        row_count=len(rows),
        final_entity_count=sum(row.final_entity_count for row in rows),
        replacement_count=sum(row.replacement_count for row in rows),
        missing_count=sum(row.missing_count for row in rows),
        collision_count=sum(row.collision_count for row in rows),
        leak_count=sum(row.leak_count for row in rows),
        duplicate_synthetic_count=sum(row.duplicate_synthetic_count for row in rows),
        missing_labels=sum_label_counts(row.missing_labels for row in rows),
        collision_labels=sum_label_counts(row.collision_labels for row in rows),
        leak_labels=sum_label_counts(row.leak_labels for row in rows),
        replacement_map_sources=count_sources(dataframe),
    )


def replacement_row_metrics(row: Any) -> ReplacementRowMetrics:
    entities = [entity.model_dump() for entity in EntitiesSchema.from_raw(row[COL_FINAL_ENTITIES]).entities]
    replacements = parse_replacements(row[COL_REPLACEMENT_MAP])
    missing = missing_replacements(entities, replacements)
    collisions = synthetic_original_collisions(entities, replacements)
    leaks = leaked_entities(entities, str(row[COL_REPLACED_TEXT]))
    synthetic_values = [item["synthetic"] for item in replacements if item.get("synthetic")]
    return ReplacementRowMetrics(
        final_entity_count=len(entities),
        replacement_count=len(replacements),
        missing_count=len(missing),
        collision_count=len(collisions),
        leak_count=len(leaks),
        duplicate_synthetic_count=max(0, len(synthetic_values) - len(set(synthetic_values))),
        missing_labels=count_labels(missing),
        collision_labels=count_labels(collisions),
        leak_labels=count_labels(leaks),
    )


def parse_replacements(raw: object) -> list[dict[str, str]]:
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="python")
    if isinstance(raw, str):
        raw = json.loads(raw)
    return [item.model_dump() for item in EntityReplacementMapSchema.model_validate(raw).replacements]


def missing_replacements(entities: list[dict[str, Any]], replacements: list[dict[str, str]]) -> list[dict[str, Any]]:
    replacement_pairs = {(item["original"], item["label"]) for item in replacements}
    return [entity for entity in entities if (entity.get("value"), entity.get("label")) not in replacement_pairs]


def synthetic_original_collisions(
    entities: list[dict[str, Any]],
    replacements: list[dict[str, str]],
) -> list[dict[str, str]]:
    original_values = {entity["value"] for entity in entities if entity.get("value")}
    return [item for item in replacements if item.get("synthetic") in original_values]


def leaked_entities(entities: list[dict[str, Any]], output_text: str) -> list[dict[str, Any]]:
    return [
        entity
        for entity in entities
        if entity.get("value") and _output_contains_original_value(output_text, str(entity["value"]))
    ]


def count_labels(items: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(item.get("label") or "") for item in items if item.get("label")).items()))


def sum_label_counts(values: object) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for mapping in values:
        counts.update(mapping)
    return dict(sorted(counts.items()))


def count_sources(dataframe: pd.DataFrame) -> dict[str, int]:
    if COL_REPLACEMENT_MAP_SOURCE not in dataframe.columns:
        return {}
    return dict(sorted(Counter(str(source) for source in dataframe[COL_REPLACEMENT_MAP_SOURCE]).items()))


def count_final_entities(dataframe: pd.DataFrame) -> int:
    return sum(len(EntitiesSchema.from_raw(raw).entities) for raw in dataframe[COL_FINAL_ENTITIES])


def render_result(result: ReplacementReplayResult, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    lines = [
        f"input={result.input_path}",
        f"labels={','.join(result.labels)}",
        f"nrows={result.nrows if result.nrows is not None else 'all'}",
        f"replacement_repetitions={result.replacement_repetitions}",
        f"detected_final_entities={result.detected_final_entity_count}",
        f"detection_elapsed_sec={result.detect_elapsed_sec:.3f}",
    ]
    lines.extend(render_strategy(summary) for summary in result.strategies)
    return "\n".join(lines)


def render_strategy(summary: ReplacementReplaySummary) -> str:
    if summary.status == ReplayStatus.error:
        return f"{summary.strategy}: error={summary.error}"
    return (
        f"{summary.strategy}: repetitions={summary.repetition_count} elapsed_sec={summary.elapsed_sec:.3f} "
        f"replacements={summary.replacement_count} missing={summary.missing_count} "
        f"leaks={summary.leak_count} collisions={summary.collision_count}"
    )


def replay_comparison_row(
    result: ReplacementReplayResult, *, workload_id: str | None = None
) -> ReplacementReplayComparisonRow:
    summaries = {summary.strategy: summary for summary in result.strategies}
    baseline = summaries.get(ReplacementReplayStrategy.dd_substitute)
    candidate = summaries.get(ReplacementReplayStrategy.local_structured_substitute)
    if baseline is None or candidate is None:
        raise ValueError(
            "replacement replay result must include dd_substitute and local_structured_substitute summaries"
        )
    flags = replay_comparison_flags(baseline, candidate)
    value_protection = replay_value_protection_verdict(flags)
    signature_parity = replay_signature_parity_verdict(baseline, candidate, flags)
    safety = replay_safety_verdict(value_protection, signature_parity)
    performance = replay_performance_verdict(baseline.elapsed_sec, candidate.elapsed_sec)
    return ReplacementReplayComparisonRow(
        workload_id=workload_id or Path(result.input_path).stem,
        baseline_case_count=baseline.row_count,
        candidate_case_count=candidate.row_count,
        baseline_pipeline_elapsed_sec=baseline.elapsed_sec,
        candidate_pipeline_elapsed_sec=candidate.elapsed_sec,
        pipeline_elapsed_sec_delta=delta(baseline.elapsed_sec, candidate.elapsed_sec),
        pipeline_elapsed_sec_delta_pct=delta_pct(baseline.elapsed_sec, candidate.elapsed_sec),
        baseline_final_entity_count=baseline.final_entity_count,
        candidate_final_entity_count=candidate.final_entity_count,
        final_entity_count_delta=candidate.final_entity_count - baseline.final_entity_count,
        baseline_replacement_count=baseline.replacement_count,
        candidate_replacement_count=candidate.replacement_count,
        replacement_count_delta=candidate.replacement_count - baseline.replacement_count,
        baseline_replacement_missing_final_entity_count=baseline.missing_count,
        candidate_replacement_missing_final_entity_count=candidate.missing_count,
        replacement_missing_final_entity_count_delta=candidate.missing_count - baseline.missing_count,
        baseline_replacement_missing_final_entity_label_counts=baseline.missing_labels,
        candidate_replacement_missing_final_entity_label_counts=candidate.missing_labels,
        baseline_replacement_synthetic_original_collision_count=baseline.collision_count,
        candidate_replacement_synthetic_original_collision_count=candidate.collision_count,
        replacement_synthetic_original_collision_count_delta=candidate.collision_count - baseline.collision_count,
        baseline_replacement_synthetic_original_collision_label_counts=baseline.collision_labels,
        candidate_replacement_synthetic_original_collision_label_counts=candidate.collision_labels,
        baseline_duplicate_synthetic_replacement_count=baseline.duplicate_synthetic_count,
        candidate_duplicate_synthetic_replacement_count=candidate.duplicate_synthetic_count,
        duplicate_synthetic_replacement_count_delta=(
            candidate.duplicate_synthetic_count - baseline.duplicate_synthetic_count
        ),
        baseline_original_value_leak_count=baseline.leak_count,
        candidate_original_value_leak_count=candidate.leak_count,
        original_value_leak_count_delta=candidate.leak_count - baseline.leak_count,
        baseline_original_value_leak_label_counts=baseline.leak_labels,
        candidate_original_value_leak_label_counts=candidate.leak_labels,
        value_protection_verdict=value_protection,
        signature_parity_verdict=signature_parity,
        safety_verdict=safety,
        performance_verdict=performance,
        candidate_verdict=replay_candidate_verdict(safety, performance),
        flags=flags,
    )


def replay_comparison_flags(
    baseline: ReplacementReplaySummary,
    candidate: ReplacementReplaySummary,
) -> list[str]:
    flags: list[str] = []
    if baseline.status == ReplayStatus.error:
        flags.append("baseline_case_failures")
    if candidate.status == ReplayStatus.error:
        flags.append("candidate_case_failures")
    if baseline.missing_count:
        flags.append("baseline_replacement_missing_final_entity")
    if candidate.missing_count:
        flags.append("candidate_replacement_missing_final_entity")
    if baseline.missing_count and candidate.missing_count < baseline.missing_count:
        flags.append("candidate_reduces_baseline_replacement_missing_final_entity")
    if baseline.missing_count and candidate.missing_count == 0:
        flags.append("candidate_covers_baseline_replacement_missing_final_entity")
    if baseline.collision_count:
        flags.append("baseline_replacement_synthetic_original_collision")
    if candidate.collision_count:
        flags.append("candidate_replacement_synthetic_original_collision")
    if baseline.collision_count and candidate.collision_count < baseline.collision_count:
        flags.append("candidate_reduces_baseline_replacement_synthetic_original_collision")
    if baseline.collision_count and candidate.collision_count == 0:
        flags.append("candidate_covers_baseline_replacement_synthetic_original_collision")
    if baseline.duplicate_synthetic_count:
        flags.append("baseline_duplicate_synthetic_replacement")
    if candidate.duplicate_synthetic_count:
        flags.append("candidate_duplicate_synthetic_replacement")
    if baseline.leak_count:
        flags.append("baseline_original_value_leak")
    if candidate.leak_count:
        flags.append("candidate_original_value_leak")
    if baseline.leak_count and candidate.leak_count < baseline.leak_count:
        flags.append("candidate_reduces_baseline_original_value_leak")
    if baseline.leak_count and candidate.leak_count == 0:
        flags.append("candidate_covers_baseline_original_value_leak")
    if baseline.final_entity_count != candidate.final_entity_count:
        flags.append(
            "entity_count_loss" if candidate.final_entity_count < baseline.final_entity_count else "entity_count_delta"
        )
    if baseline.replacement_count != candidate.replacement_count:
        flags.append(
            "replacement_count_loss"
            if candidate.replacement_count < baseline.replacement_count
            else "replacement_count_delta"
        )
    return flags


def replay_value_protection_verdict(flags: list[str]) -> str:
    flag_set = set(flags)
    if flag_set & {
        "candidate_case_failures",
        "candidate_original_value_leak",
        "candidate_replacement_missing_final_entity",
        "candidate_replacement_synthetic_original_collision",
        "entity_count_loss",
        "replacement_count_loss",
    }:
        return "fail"
    if "baseline_case_failures" in flag_set:
        return "review"
    baseline_defects = {
        "baseline_original_value_leak",
        "baseline_replacement_missing_final_entity",
        "baseline_replacement_synthetic_original_collision",
    }
    corrected_baseline_defects = {
        "baseline_original_value_leak": "candidate_covers_baseline_original_value_leak",
        "baseline_replacement_missing_final_entity": ("candidate_covers_baseline_replacement_missing_final_entity"),
        "baseline_replacement_synthetic_original_collision": (
            "candidate_covers_baseline_replacement_synthetic_original_collision"
        ),
    }
    for defect in baseline_defects & flag_set:
        if corrected_baseline_defects[defect] not in flag_set:
            return "review"
    return "pass"


def replay_signature_parity_verdict(
    baseline: ReplacementReplaySummary,
    candidate: ReplacementReplaySummary,
    flags: list[str],
) -> str:
    if candidate.status == ReplayStatus.error:
        return "fail"
    if baseline.status == ReplayStatus.error:
        return "review"
    if candidate.final_entity_count < baseline.final_entity_count:
        return "fail"
    if candidate.final_entity_count != baseline.final_entity_count:
        return "review"
    if candidate.replacement_count < baseline.replacement_count:
        return "fail"
    if candidate.replacement_count != baseline.replacement_count:
        return "review"
    if flags:
        return "review"
    return "pass"


def replay_safety_verdict(value_protection: str, signature_parity: str) -> str:
    if "fail" in {value_protection, signature_parity}:
        return "fail"
    if "review" in {value_protection, signature_parity}:
        return "review"
    return "pass"


def replay_performance_verdict(baseline_elapsed: float | None, candidate_elapsed: float | None) -> str:
    elapsed_delta = delta(baseline_elapsed, candidate_elapsed)
    if elapsed_delta is None:
        return "unknown"
    if elapsed_delta < 0:
        return "improved"
    if elapsed_delta > 0:
        return "regressed"
    return "unchanged"


def replay_candidate_verdict(safety: str, performance: str) -> str:
    if safety == "fail":
        return "reject"
    if safety == "pass" and performance == "improved":
        return "candidate_viable"
    return "review"


def delta(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None:
        return None
    return candidate - baseline


def delta_pct(baseline: float | None, candidate: float | None) -> float | None:
    value = delta(baseline, candidate)
    if value is None or baseline in {None, 0}:
        return None
    return value / baseline * 100


def write_replay_comparison(
    result: ReplacementReplayResult,
    output: Path,
    *,
    workload_id: str | None = None,
) -> None:
    row = replay_comparison_row(result, workload_id=workload_id).model_dump(mode="json")
    row["flags"] = json.dumps(row["flags"], ensure_ascii=True)
    pd.DataFrame([row]).to_csv(output, index=False)


@app.default
def main(
    source: Path,
    *,
    text_column: Annotated[str, cyclopts.Parameter("--text-column")] = "text",
    labels: Annotated[str, cyclopts.Parameter("--labels")],
    nrows: Annotated[int | None, cyclopts.Parameter("--nrows")] = None,
    replacement_repetitions: Annotated[int, cyclopts.Parameter("--replacement-repetitions")] = 1,
    model_configs: Annotated[Path | None, cyclopts.Parameter("--model-configs")] = None,
    model_providers: Annotated[Path | None, cyclopts.Parameter("--model-providers")] = None,
    artifact_path: Annotated[Path, cyclopts.Parameter("--artifact-path")] = Path(
        "/tmp/anonymizer-replacement-replay-artifacts"
    ),
    dd_parser_compat: Annotated[DDParserCompatMode, cyclopts.Parameter("--dd-parser-compat")] = (
        DDParserCompatMode.none
    ),
    output: Annotated[Path | None, cyclopts.Parameter("--output")] = None,
    comparison_output: Annotated[Path | None, cyclopts.Parameter("--comparison-output")] = None,
    workload_id: Annotated[str | None, cyclopts.Parameter("--workload-id")] = None,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    try:
        if nrows is not None and nrows <= 0:
            raise ValueError("--nrows must be greater than zero")
        if replacement_repetitions <= 0:
            raise ValueError("--replacement-repetitions must be greater than zero")
        result = run_replacement_replay(
            source=source,
            text_column=text_column,
            labels=parse_labels(labels),
            nrows=nrows,
            model_configs=model_configs,
            model_providers=model_providers,
            artifact_path=artifact_path,
            dd_parser_compat=dd_parser_compat,
            replacement_repetitions=replacement_repetitions,
        )
    except (ValidationError, ValueError, FileNotFoundError) as exc:
        log_bad_input(str(exc))
        raise SystemExit(125) from exc
    rendered = render_result(result, json_output=json_output)
    if output is not None:
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        sys.stdout.write(rendered + "\n")
    if comparison_output is not None:
        write_replay_comparison(result, comparison_output, workload_id=workload_id)


if __name__ == "__main__":
    app()
