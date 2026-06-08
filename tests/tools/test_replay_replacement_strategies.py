# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd

from anonymizer.engine.constants import (
    COL_FINAL_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_REPLACEMENT_MAP_SOURCE,
    COL_TEXT,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_tool(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def test_replacement_row_metrics_counts_sanitized_replay_failures() -> None:
    tool = load_tool(
        "measurement_replay_replacement_strategies_metrics",
        REPO_ROOT / "tools/measurement/replay_replacement_strategies.py",
    )
    row = pd.Series(
        {
            COL_FINAL_ENTITIES: {
                "entities": [
                    {"value": "secret-a", "label": "api_key", "start_position": 0, "end_position": 8},
                    {"value": "secret-b", "label": "password", "start_position": 9, "end_position": 17},
                    {"value": "1234", "label": "pin", "start_position": 18, "end_position": 22},
                ]
            },
            COL_REPLACEMENT_MAP: {
                "replacements": [
                    {"original": "secret-a", "label": "api_key", "synthetic": "secret-b"},
                    {"original": "secret-b", "label": "password", "synthetic": "Synthetic!123"},
                ]
            },
            COL_REPLACED_TEXT: "the leaked pin is 1234",
        }
    )

    metrics = tool.replacement_row_metrics(row)

    assert metrics.final_entity_count == 3
    assert metrics.replacement_count == 2
    assert metrics.missing_count == 1
    assert metrics.missing_labels == {"pin": 1}
    assert metrics.collision_count == 1
    assert metrics.collision_labels == {"api_key": 1}
    assert metrics.leak_count == 1
    assert metrics.leak_labels == {"pin": 1}


def test_summarize_replacement_dataframe_counts_sources_and_totals() -> None:
    tool = load_tool(
        "measurement_replay_replacement_strategies_summary",
        REPO_ROOT / "tools/measurement/replay_replacement_strategies.py",
    )
    dataframe = pd.DataFrame(
        [
            {
                COL_FINAL_ENTITIES: {
                    "entities": [
                        {"value": "alice@example.com", "label": "email", "start_position": 0, "end_position": 17}
                    ]
                },
                COL_REPLACEMENT_MAP: {
                    "replacements": [
                        {
                            "original": "alice@example.com",
                            "label": "email",
                            "synthetic": "user-123@example.invalid",
                        }
                    ]
                },
                COL_REPLACED_TEXT: "user-123@example.invalid",
                COL_REPLACEMENT_MAP_SOURCE: "local_structured",
            }
        ]
    )

    summary = tool.summarize_replacement_dataframe(
        dataframe,
        strategy=tool.ReplacementReplayStrategy.local_structured_substitute,
        elapsed_sec=0.01,
    )

    assert summary.status == tool.ReplayStatus.completed
    assert summary.final_entity_count == 1
    assert summary.replacement_count == 1
    assert summary.missing_count == 0
    assert summary.collision_count == 0
    assert summary.leak_count == 0
    assert summary.replacement_map_sources == {"local_structured": 1}


def test_aggregate_strategy_summaries_sums_repeated_backend_runs() -> None:
    tool = load_tool(
        "measurement_replay_replacement_strategies_aggregate",
        REPO_ROOT / "tools/measurement/replay_replacement_strategies.py",
    )

    summary = tool.aggregate_strategy_summaries(
        strategy=tool.ReplacementReplayStrategy.dd_substitute,
        summaries=[
            tool.ReplacementReplaySummary(
                strategy=tool.ReplacementReplayStrategy.dd_substitute,
                status=tool.ReplayStatus.completed,
                elapsed_sec=1.5,
                row_count=5,
                final_entity_count=10,
                replacement_count=9,
                missing_count=1,
                leak_count=1,
                missing_labels={"api_key": 1},
                leak_labels={"api_key": 1},
                replacement_map_sources={"llm": 5},
            ),
            tool.ReplacementReplaySummary(
                strategy=tool.ReplacementReplayStrategy.dd_substitute,
                status=tool.ReplayStatus.error,
                elapsed_sec=2.5,
                row_count=5,
                final_entity_count=10,
                replacement_count=10,
                collision_count=1,
                collision_labels={"password": 1},
                replacement_map_sources={"llm": 4},
                error="provider failed",
            ),
        ],
    )

    assert summary.status == tool.ReplayStatus.error
    assert summary.repetition_count == 2
    assert summary.elapsed_sec == 4.0
    assert summary.row_count == 10
    assert summary.final_entity_count == 20
    assert summary.replacement_count == 19
    assert summary.missing_count == 1
    assert summary.collision_count == 1
    assert summary.leak_count == 1
    assert summary.missing_labels == {"api_key": 1}
    assert summary.collision_labels == {"password": 1}
    assert summary.leak_labels == {"api_key": 1}
    assert summary.replacement_map_sources == {"llm": 9}
    assert summary.error == "provider failed"


def test_replay_comparison_row_marks_fast_complete_local_replay_viable() -> None:
    tool = load_tool(
        "measurement_replay_replacement_strategies_comparison_row",
        REPO_ROOT / "tools/measurement/replay_replacement_strategies.py",
    )
    result = tool.ReplacementReplayResult(
        input_path="/tmp/structured_identifiers.csv",
        text_column="text",
        labels=["api_key"],
        dd_parser_compat=tool.DDParserCompatMode.raw_json,
        detect_elapsed_sec=8.8,
        detected_final_entity_count=2,
        strategies=[
            tool.ReplacementReplaySummary(
                strategy=tool.ReplacementReplayStrategy.dd_substitute,
                status=tool.ReplayStatus.completed,
                elapsed_sec=6.2,
                row_count=1,
                final_entity_count=2,
                replacement_count=2,
            ),
            tool.ReplacementReplaySummary(
                strategy=tool.ReplacementReplayStrategy.local_structured_substitute,
                status=tool.ReplayStatus.completed,
                elapsed_sec=0.003,
                row_count=1,
                final_entity_count=2,
                replacement_count=2,
            ),
        ],
    )

    row = tool.replay_comparison_row(result)

    assert row.workload_id == "structured_identifiers"
    assert row.baseline_replacement_strategy == "default"
    assert row.candidate_replacement_strategy == "local_structured_substitute"
    assert row.pipeline_elapsed_sec_delta < 0
    assert row.value_protection_verdict == "pass"
    assert row.signature_parity_verdict == "pass"
    assert row.safety_verdict == "pass"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "candidate_viable"
    assert row.flags == []


def test_replay_comparison_row_rejects_missing_local_replacements() -> None:
    tool = load_tool(
        "measurement_replay_replacement_strategies_comparison_missing",
        REPO_ROOT / "tools/measurement/replay_replacement_strategies.py",
    )
    result = tool.ReplacementReplayResult(
        input_path="/tmp/structured_identifiers.csv",
        text_column="text",
        labels=["api_key"],
        dd_parser_compat=tool.DDParserCompatMode.raw_json,
        detect_elapsed_sec=8.8,
        detected_final_entity_count=2,
        strategies=[
            tool.ReplacementReplaySummary(
                strategy=tool.ReplacementReplayStrategy.dd_substitute,
                status=tool.ReplayStatus.completed,
                elapsed_sec=6.2,
                row_count=1,
                final_entity_count=2,
                replacement_count=2,
            ),
            tool.ReplacementReplaySummary(
                strategy=tool.ReplacementReplayStrategy.local_structured_substitute,
                status=tool.ReplayStatus.completed,
                elapsed_sec=0.003,
                row_count=1,
                final_entity_count=2,
                replacement_count=1,
                missing_count=1,
            ),
        ],
    )

    row = tool.replay_comparison_row(result)

    assert row.candidate_replacement_missing_final_entity_count == 1
    assert "candidate_replacement_missing_final_entity" in row.flags
    assert "replacement_count_loss" in row.flags
    assert row.value_protection_verdict == "fail"
    assert row.signature_parity_verdict == "fail"
    assert row.safety_verdict == "fail"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "reject"


def test_replay_comparison_row_reviews_duplicate_local_synthetic_values() -> None:
    tool = load_tool(
        "measurement_replay_replacement_strategies_comparison_duplicates",
        REPO_ROOT / "tools/measurement/replay_replacement_strategies.py",
    )
    result = tool.ReplacementReplayResult(
        input_path="/tmp/biographies.csv",
        text_column="biography",
        labels=["organization_name", "religious_belief"],
        nrows=5,
        dd_parser_compat=tool.DDParserCompatMode.raw_json,
        detect_elapsed_sec=18.8,
        detected_final_entity_count=10,
        strategies=[
            tool.ReplacementReplaySummary(
                strategy=tool.ReplacementReplayStrategy.dd_substitute,
                status=tool.ReplayStatus.completed,
                elapsed_sec=7.9,
                row_count=5,
                final_entity_count=10,
                replacement_count=10,
            ),
            tool.ReplacementReplaySummary(
                strategy=tool.ReplacementReplayStrategy.local_structured_substitute,
                status=tool.ReplayStatus.completed,
                elapsed_sec=0.003,
                row_count=5,
                final_entity_count=10,
                replacement_count=10,
                duplicate_synthetic_count=2,
            ),
        ],
    )

    row = tool.replay_comparison_row(result, workload_id="biography-supported-structured")

    assert row.candidate_duplicate_synthetic_replacement_count == 2
    assert "candidate_duplicate_synthetic_replacement" in row.flags
    assert row.value_protection_verdict == "pass"
    assert row.signature_parity_verdict == "review"
    assert row.safety_verdict == "review"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "review"


def test_replay_comparison_row_reviews_candidate_replacement_count_gain_over_flawed_baseline() -> None:
    tool = load_tool(
        "measurement_replay_replacement_strategies_comparison_baseline_flaw",
        REPO_ROOT / "tools/measurement/replay_replacement_strategies.py",
    )
    result = tool.ReplacementReplayResult(
        input_path="/tmp/biographies.csv",
        text_column="biography",
        labels=["organization_name"],
        nrows=5,
        dd_parser_compat=tool.DDParserCompatMode.raw_json,
        detect_elapsed_sec=12.9,
        detected_final_entity_count=2,
        strategies=[
            tool.ReplacementReplaySummary(
                strategy=tool.ReplacementReplayStrategy.dd_substitute,
                status=tool.ReplayStatus.completed,
                elapsed_sec=7.6,
                row_count=5,
                final_entity_count=2,
                replacement_count=1,
                missing_count=1,
                leak_count=1,
                missing_labels={"organization_name": 1},
                leak_labels={"organization_name": 1},
            ),
            tool.ReplacementReplaySummary(
                strategy=tool.ReplacementReplayStrategy.local_structured_substitute,
                status=tool.ReplayStatus.completed,
                elapsed_sec=0.003,
                row_count=5,
                final_entity_count=2,
                replacement_count=2,
            ),
        ],
    )

    row = tool.replay_comparison_row(result, workload_id="biography-supported-structured")

    assert row.replacement_count_delta == 1
    assert row.replacement_missing_final_entity_count_delta == -1
    assert row.baseline_replacement_missing_final_entity_label_counts == {"organization_name": 1}
    assert row.candidate_replacement_missing_final_entity_label_counts == {}
    assert row.baseline_original_value_leak_label_counts == {"organization_name": 1}
    assert row.candidate_original_value_leak_label_counts == {}
    assert "baseline_replacement_missing_final_entity" in row.flags
    assert "baseline_original_value_leak" in row.flags
    assert "candidate_covers_baseline_replacement_missing_final_entity" in row.flags
    assert "candidate_covers_baseline_original_value_leak" in row.flags
    assert "replacement_count_delta" in row.flags
    assert row.value_protection_verdict == "pass"
    assert row.signature_parity_verdict == "review"
    assert row.safety_verdict == "review"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "review"


def test_strip_replacement_columns_removes_prior_strategy_outputs() -> None:
    tool = load_tool(
        "measurement_replay_replacement_strategies_strip",
        REPO_ROOT / "tools/measurement/replay_replacement_strategies.py",
    )
    dataframe = pd.DataFrame(
        {
            "text": ["hello"],
            COL_REPLACEMENT_MAP: [{"replacements": []}],
            COL_REPLACEMENT_MAP_SOURCE: ["redact"],
            COL_REPLACED_TEXT: ["hello"],
        }
    )

    stripped = tool.strip_replacement_columns(dataframe)

    assert list(stripped.columns) == ["text"]


def test_build_replay_dataframe_uses_preview_when_nrows_is_set(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_replay_replacement_strategies_nrows",
        REPO_ROOT / "tools/measurement/replay_replacement_strategies.py",
    )
    trace_dataframe = pd.DataFrame(
        {
            COL_TEXT: ["born on 1988-11-21"],
            COL_FINAL_ENTITIES: [
                {
                    "entities": [
                        {
                            "value": "1988-11-21",
                            "label": "date_of_birth",
                            "start_position": 8,
                            "end_position": 18,
                        }
                    ]
                }
            ],
            COL_REPLACEMENT_MAP: [{"replacements": []}],
            COL_REPLACED_TEXT: ["born on [REDACTED_DATE_OF_BIRTH]"],
        }
    )

    class StubAnonymizer:
        preview_num_records: int | None = None

        def run(self, **_kwargs: object) -> object:
            raise AssertionError("run should not be used for row-limited replay")

        def preview(self, **kwargs: object) -> object:
            self.preview_num_records = kwargs["num_records"]  # type: ignore[assignment]
            return SimpleNamespace(trace_dataframe=trace_dataframe, resolved_text_column="text")

    anonymizer = StubAnonymizer()
    source = tmp_path / "multiline.csv"
    source.write_text('text\n"born on 1988-11-21"\n', encoding="utf-8")

    _elapsed, replay_df = tool.build_replay_dataframe(
        anonymizer,
        source=source,
        text_column="text",
        labels=["date_of_birth"],
        nrows=1,
        dd_parser_compat=tool.DDParserCompatMode.none,
    )

    assert anonymizer.preview_num_records == 1
    assert COL_REPLACEMENT_MAP not in replay_df.columns
    assert COL_REPLACED_TEXT not in replay_df.columns
    assert replay_df[COL_FINAL_ENTITIES].iloc[0]["entities"][0]["value"] == "1988-11-21"
