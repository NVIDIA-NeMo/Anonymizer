# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_analyze_benchmark_output_joins_measurements_and_detection_artifacts(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_output_analysis",
        REPO_ROOT / "tools/measurement/analyze_benchmark_output.py",
    )
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    _write_jsonl(
        benchmark_dir / "measurements.jsonl",
        [
            {
                "record_type": "ndd_workflow",
                "run_id": "bio__default__r000",
                "workflow_name": "entity-detection",
                "elapsed_sec": 8.5,
                "observed_total_requests": 4,
                "observed_successful_requests": 3,
                "observed_input_tokens": 5000,
                "observed_output_tokens": 1000,
                "observed_total_tokens": 6000,
                "observed_failed_requests": 1,
                "model_usage": {
                    "nvidia/gliner-pii": {
                        "request_usage": {
                            "successful_requests": 1,
                            "failed_requests": 0,
                            "total_requests": 1,
                        },
                        "token_usage": {
                            "input_tokens": 1000,
                            "output_tokens": 100,
                            "total_tokens": 1100,
                        },
                    },
                    "local-nemotron-json": {
                        "model_alias": "local-nemotron-json",
                        "model_name": "nvidia/nemotron-3-super",
                        "model_provider_name": "local-vllm",
                        "request_usage": {
                            "successful_requests": 2,
                            "failed_requests": 1,
                            "total_requests": 3,
                        },
                        "token_usage": {
                            "input_tokens": 4000,
                            "output_tokens": 900,
                            "total_tokens": 4900,
                        },
                    },
                },
                "detect": {"validation_max_entities_per_call": 10},
                "run_tags": {
                    "suite_id": "suite",
                    "workload_id": "bio",
                    "config_id": "default",
                    "experimental_detection_strategy": "default",
                    "experimental_replacement_strategy": "default",
                    "dd_parser_compat": "raw_json",
                    "repetition": 0,
                    "case_id": "bio__default__r000",
                },
            },
            {
                "record_type": "record",
                "run_id": "bio__default__r000",
                "final_entity_count": 14,
                "replacement_count": 12,
                "replacement_missing_final_entity_count": 2,
                "replacement_missing_final_entity_label_counts": {"date": 2},
                "replacement_missing_final_value_count": 1,
                "replacement_synthetic_original_collision_count": 1,
                "replacement_synthetic_original_collision_label_counts": {"date": 1},
                "replacement_synthetic_original_collision_value_count": 1,
                "original_value_leak_count": 0,
                "original_value_leak_label_counts": {},
                "run_tags": {
                    "suite_id": "suite",
                    "workload_id": "bio",
                    "config_id": "default",
                    "experimental_detection_strategy": "default",
                    "experimental_replacement_strategy": "default",
                    "dd_parser_compat": "raw_json",
                    "repetition": 0,
                    "case_id": "bio__default__r000",
                },
            },
            {
                "record_type": "record",
                "run_id": "shell__rules-only__r000",
                "final_entity_count": 8,
                "replacement_count": 8,
                "replacement_missing_final_entity_count": 0,
                "replacement_missing_final_entity_label_counts": {},
                "replacement_missing_final_value_count": 0,
                "replacement_synthetic_original_collision_count": 0,
                "replacement_synthetic_original_collision_label_counts": {},
                "replacement_synthetic_original_collision_value_count": 0,
                "original_value_leak_count": 1,
                "original_value_leak_label_counts": {"api_key": 1},
                "run_tags": {
                    "suite_id": "suite",
                    "workload_id": "shell",
                    "config_id": "rules-only",
                    "experimental_detection_strategy": "rules_only",
                    "experimental_replacement_strategy": "local_structured_substitute",
                    "dd_parser_compat": "raw_json",
                    "repetition": 0,
                    "case_id": "shell__rules-only__r000",
                },
            },
        ],
    )
    _write_jsonl(
        benchmark_dir / "detection-artifacts.jsonl",
        [
            {
                "suite_id": "suite",
                "workload_id": "bio",
                "config_id": "default",
                "repetition": 0,
                "case_id": "bio__default__r000",
                "run_id": "bio__default__r000",
                "workflow_name": "entity-detection",
                "seed_entity_count": 13,
                "seed_validation_candidate_count": 13,
                "augmented_entity_count": 1,
                "augmented_new_final_value_count": 1,
                "final_entity_count": 14,
                "final_source_counts": {"detector": 11, "augmenter": 3},
                "final_entity_signature_hashes": ["bio-hash-a", "bio-hash-b"],
                "final_entity_signature_labels": {"bio-hash-a": "person", "bio-hash-b": "city"},
                "final_entity_signature_details": {
                    "bio-hash-a": {
                        "label": "person",
                        "source": "detector",
                        "row_index": 0,
                        "start_position": 0,
                        "end_position": 5,
                        "value_hash": "hash-person",
                        "value_length": 5,
                    }
                },
                "final_entity_signature_count": 2,
            },
            {
                "suite_id": "suite",
                "workload_id": "shell",
                "config_id": "rules-only",
                "repetition": 0,
                "case_id": "shell__rules-only__r000",
                "run_id": "shell__rules-only__r000",
                "workflow_name": "rules-only",
                "seed_entity_count": 8,
                "seed_validation_candidate_count": 0,
                "augmented_entity_count": 0,
                "augmented_new_final_value_count": 0,
                "final_entity_count": 8,
                "final_source_counts": {"rule": 8},
                "final_entity_signature_hashes": ["shell-hash-a"],
                "final_entity_signature_labels": {"shell-hash-a": "api_key"},
                "final_entity_signature_details": {
                    "shell-hash-a": {
                        "label": "api_key",
                        "source": "rule",
                        "row_index": 0,
                        "start_position": 12,
                        "end_position": 32,
                        "value_hash": "hash-secret",
                        "value_length": 20,
                    }
                },
                "final_entity_signature_count": 1,
            },
        ],
    )
    traces_dir = benchmark_dir / "traces"
    traces_dir.mkdir()
    _write_jsonl(
        traces_dir / "bio__default__r000.jsonl",
        [
            {
                "record_type": "dd_message_trace",
                "run_id": "bio__default__r000",
                "workflow_name": "entity-detection",
                "model_alias": "local-nemotron-json",
                "status": "error",
                "error_type": "SyncClientUnavailableError",
                "is_async": False,
                "messages": [{"role": "user", "content": "Alice has sk-test"}],
                "response": "Alice still has sk-test",
                "run_tags": {
                    "suite_id": "suite",
                    "workload_id": "bio",
                    "config_id": "default",
                    "experimental_detection_strategy": "default",
                    "experimental_replacement_strategy": "default",
                    "dd_parser_compat": "raw_json",
                    "repetition": 0,
                    "case_id": "bio__default__r000",
                },
            },
            {
                "record_type": "dd_message_trace",
                "run_id": "bio__default__r000",
                "workflow_name": "entity-detection",
                "model_alias": "local-nemotron-json",
                "status": "success",
                "is_async": True,
                "messages": [{"role": "user", "content": "sk-test"}],
                "response": "Alice",
                "run_tags": {
                    "suite_id": "suite",
                    "workload_id": "bio",
                    "config_id": "default",
                    "experimental_detection_strategy": "default",
                    "experimental_replacement_strategy": "default",
                    "dd_parser_compat": "raw_json",
                    "repetition": 0,
                    "case_id": "bio__default__r000",
                },
            },
        ],
    )

    result = tool.analyze_benchmark_output(benchmark_dir)

    assert result.case_count == 2
    assert result.group_count == 2
    assert result.model_usage_count == 2
    assert result.model_usage_group_count == 2
    cases = {row.case_id: row for row in result.cases}
    assert cases["bio__default__r000"].experimental_replacement_strategy == "default"
    assert cases["bio__default__r000"].observed_total_requests == 4
    assert cases["bio__default__r000"].observed_successful_requests == 3
    assert cases["bio__default__r000"].observed_input_tokens == 5000
    assert cases["bio__default__r000"].observed_output_tokens == 1000
    assert cases["bio__default__r000"].observed_total_tokens == 6000
    assert cases["bio__default__r000"].observed_failed_request_rate == pytest.approx(1 / 4)
    assert cases["bio__default__r000"].dd_trace_record_count == 2
    assert cases["bio__default__r000"].dd_trace_error_count == 1
    assert cases["bio__default__r000"].dd_trace_sync_client_unavailable_count == 1
    assert cases["bio__default__r000"].observed_bridge_fallback_requests == 1
    assert cases["bio__default__r000"].observed_non_bridge_total_requests == 3
    assert cases["bio__default__r000"].observed_non_bridge_failed_requests == 0
    assert cases["bio__default__r000"].observed_non_bridge_failed_request_rate == 0
    assert cases["bio__default__r000"].validation_max_entities_per_call == 10
    assert cases["bio__default__r000"].original_value_leak_count == 0
    assert cases["bio__default__r000"].original_value_leak_record_count == 0
    assert cases["bio__default__r000"].original_value_leak_label_counts == {}
    assert cases["bio__default__r000"].replacement_missing_final_entity_count == 2
    assert cases["bio__default__r000"].replacement_missing_final_entity_label_counts == {"date": 2}
    assert cases["bio__default__r000"].replacement_missing_final_value_count == 1
    assert cases["bio__default__r000"].replacement_synthetic_original_collision_count == 1
    assert cases["bio__default__r000"].replacement_synthetic_original_collision_label_counts == {"date": 1}
    assert cases["bio__default__r000"].replacement_synthetic_original_collision_value_count == 1
    assert cases["bio__default__r000"].seed_validation_candidate_count == 13
    assert cases["bio__default__r000"].estimated_seed_validation_chunk_count == 2
    assert cases["bio__default__r000"].augmented_new_final_value_count == 1
    assert cases["bio__default__r000"].artifact_final_detector_entity_count == 11
    assert cases["bio__default__r000"].artifact_final_augmenter_entity_count == 3
    assert cases["bio__default__r000"].artifact_final_entity_signature_hashes == ["bio-hash-a", "bio-hash-b"]
    assert cases["bio__default__r000"].artifact_final_entity_signature_labels == {
        "bio-hash-a": "person",
        "bio-hash-b": "city",
    }
    assert cases["bio__default__r000"].artifact_final_entity_signature_details == {
        "bio-hash-a": {
            "label": "person",
            "source": "detector",
            "row_index": 0,
            "start_position": 0,
            "end_position": 5,
            "value_hash": "hash-person",
            "value_length": 5,
        }
    }
    assert cases["bio__default__r000"].artifact_final_entity_signature_count == 2
    assert cases["shell__rules-only__r000"].observed_total_requests == 0
    assert cases["shell__rules-only__r000"].experimental_replacement_strategy == "local_structured_substitute"
    assert cases["shell__rules-only__r000"].observed_failed_request_rate is None
    assert cases["shell__rules-only__r000"].observed_bridge_fallback_requests is None
    assert cases["shell__rules-only__r000"].observed_non_bridge_failed_requests is None
    assert cases["shell__rules-only__r000"].final_entity_count == 8
    assert cases["shell__rules-only__r000"].replacement_missing_final_entity_count == 0
    assert cases["shell__rules-only__r000"].replacement_missing_final_entity_label_counts == {}
    assert cases["shell__rules-only__r000"].replacement_missing_final_value_count == 0
    assert cases["shell__rules-only__r000"].replacement_synthetic_original_collision_count == 0
    assert cases["shell__rules-only__r000"].replacement_synthetic_original_collision_label_counts == {}
    assert cases["shell__rules-only__r000"].replacement_synthetic_original_collision_value_count == 0
    assert cases["shell__rules-only__r000"].original_value_leak_count == 1
    assert cases["shell__rules-only__r000"].original_value_leak_record_count == 1
    assert cases["shell__rules-only__r000"].original_value_leak_label_counts == {"api_key": 1}
    assert cases["shell__rules-only__r000"].artifact_final_rule_entity_count == 8
    assert cases["shell__rules-only__r000"].artifact_final_entity_signature_hashes == ["shell-hash-a"]
    assert cases["shell__rules-only__r000"].artifact_final_entity_signature_labels == {"shell-hash-a": "api_key"}
    assert cases["shell__rules-only__r000"].artifact_final_entity_signature_details["shell-hash-a"]["source"] == "rule"
    model_rows = {row.model_name: row for row in result.model_usage}
    assert model_rows["nvidia/gliner-pii"].observed_failed_requests == 0
    assert model_rows["nvidia/gliner-pii"].observed_failed_request_rate == 0
    assert model_rows["nvidia/gliner-pii"].observed_total_tokens == 1100
    assert model_rows["nvidia/nemotron-3-super"].model_alias == "local-nemotron-json"
    assert model_rows["nvidia/nemotron-3-super"].experimental_replacement_strategy == "default"
    assert model_rows["nvidia/nemotron-3-super"].model_provider_name == "local-vllm"
    assert model_rows["nvidia/nemotron-3-super"].observed_failed_requests == 1
    assert model_rows["nvidia/nemotron-3-super"].observed_failed_request_rate == pytest.approx(1 / 3)
    assert model_rows["nvidia/nemotron-3-super"].observed_total_tokens == 4900
    model_groups = {(row.model_alias, row.model_name): row for row in result.model_usage_groups}
    nemotron_group = model_groups[("local-nemotron-json", "nvidia/nemotron-3-super")]
    assert nemotron_group.model_provider_name == "local-vllm"
    assert nemotron_group.sum_observed_failed_requests == 1
    assert nemotron_group.observed_failed_request_rate == pytest.approx(1 / 3)
    assert nemotron_group.median_observed_total_requests == 3
    bio_group = next(group for group in result.groups if group.workload_id == "bio")
    assert bio_group.experimental_replacement_strategy == "default"
    assert bio_group.median_observed_bridge_fallback_requests == 1
    assert bio_group.median_observed_non_bridge_total_requests == 3
    assert bio_group.median_observed_non_bridge_failed_requests == 0
    assert bio_group.median_observed_non_bridge_failed_request_rate == 0
    assert bio_group.median_replacement_missing_final_entity_count == 2
    assert bio_group.median_replacement_missing_final_value_count == 1
    assert bio_group.replacement_missing_final_entity_label_counts == {"date": 2}
    assert bio_group.median_replacement_synthetic_original_collision_count == 1
    assert bio_group.median_replacement_synthetic_original_collision_value_count == 1
    assert bio_group.replacement_synthetic_original_collision_label_counts == {"date": 1}
    shell_group = next(group for group in result.groups if group.workload_id == "shell")
    assert shell_group.experimental_replacement_strategy == "local_structured_substitute"
    assert shell_group.sum_original_value_leak_count == 1
    assert shell_group.leaking_case_count == 1
    assert shell_group.median_original_value_leak_count == 1

    serialized = result.model_dump_json()
    assert "Alice" not in serialized
    assert "sk-test" not in serialized


def test_analyze_benchmark_output_counts_generic_model_workflow_records(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_output_analysis_model_workflow",
        REPO_ROOT / "tools/measurement/analyze_benchmark_output.py",
    )
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    _write_jsonl(
        benchmark_dir / "measurements.jsonl",
        [
            {
                "record_type": "model_workflow",
                "run_id": "bio__native__r000",
                "workflow_name": "entity-detection-native-rules-router",
                "elapsed_sec": 0.25,
                "observed_total_requests": 3,
                "observed_successful_requests": 3,
                "observed_failed_requests": 0,
                "observed_input_tokens": 30,
                "observed_output_tokens": 12,
                "observed_total_tokens": 42,
                "model_usage": {
                    "native-direct": {
                        "model_alias": "native-direct",
                        "model_name": "nvidia/nemotron-3-super",
                        "model_provider_name": "local-vllm",
                        "request_usage": {
                            "successful_requests": 3,
                            "failed_requests": 0,
                            "total_requests": 3,
                        },
                        "token_usage": {
                            "input_tokens": 30,
                            "output_tokens": 12,
                            "total_tokens": 42,
                        },
                    }
                },
                "run_tags": {
                    "suite_id": "suite",
                    "workload_id": "bio",
                    "config_id": "native",
                    "experimental_detection_strategy": "native_rules_router",
                    "experimental_replacement_strategy": "default",
                    "dd_parser_compat": "raw_json",
                    "repetition": 0,
                    "case_id": "bio__native__r000",
                },
            },
            {
                "record_type": "record",
                "run_id": "bio__native__r000",
                "final_entity_count": 2,
                "replacement_count": 2,
                "original_value_leak_count": 0,
                "original_value_leak_label_counts": {},
                "run_tags": {
                    "suite_id": "suite",
                    "workload_id": "bio",
                    "config_id": "native",
                    "experimental_detection_strategy": "native_rules_router",
                    "experimental_replacement_strategy": "default",
                    "dd_parser_compat": "raw_json",
                    "repetition": 0,
                    "case_id": "bio__native__r000",
                },
            },
        ],
    )

    result = tool.analyze_benchmark_output(benchmark_dir)

    assert result.case_count == 1
    case = result.cases[0]
    assert case.observed_total_requests == 3
    assert case.observed_total_tokens == 42
    assert case.observed_failed_request_rate == 0
    assert result.model_usage_count == 1
    model_row = result.model_usage[0]
    assert model_row.workflow_name == "entity-detection-native-rules-router"
    assert model_row.model_alias == "native-direct"
    assert model_row.model_name == "nvidia/nemotron-3-super"
    assert model_row.observed_total_tokens == 42
    assert result.groups[0].median_observed_total_requests == 3
    assert result.model_usage_groups[0].sum_observed_total_tokens == 42


def test_analyze_benchmark_output_accepts_detection_artifact_override(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_output_analysis_artifact_override",
        REPO_ROOT / "tools/measurement/analyze_benchmark_output.py",
    )
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    _write_jsonl(
        benchmark_dir / "measurements.jsonl",
        [
            {
                "record_type": "record",
                "run_id": "bio__default__r000",
                "final_entity_count": 2,
                "run_tags": {
                    "workload_id": "bio",
                    "config_id": "default",
                    "experimental_detection_strategy": "default",
                    "case_id": "bio__default__r000",
                },
            }
        ],
    )
    _write_jsonl(
        benchmark_dir / "detection-artifacts.jsonl",
        [
            {
                "case_id": "bio__default__r000",
                "run_id": "bio__default__r000",
                "final_entity_count": 2,
                "final_entity_signature_hashes": ["stale-hash"],
                "final_entity_signature_count": 1,
            }
        ],
    )
    refreshed_artifacts = tmp_path / "refreshed-detection-artifacts.jsonl"
    _write_jsonl(
        refreshed_artifacts,
        [
            {
                "case_id": "bio__default__r000",
                "run_id": "bio__default__r000",
                "final_entity_count": 2,
                "final_entity_signature_hashes": ["fresh-hash-a", "fresh-hash-b"],
                "final_entity_signature_labels": {"fresh-hash-a": "person", "fresh-hash-b": "email"},
                "final_entity_signature_count": 2,
            }
        ],
    )

    default_result = tool.analyze_benchmark_output(benchmark_dir)
    override_result = tool.analyze_benchmark_output(benchmark_dir, detection_artifacts=refreshed_artifacts)

    assert default_result.cases[0].artifact_final_entity_signature_hashes == ["stale-hash"]
    assert override_result.detection_artifacts_path == str(refreshed_artifacts)
    assert override_result.cases[0].artifact_final_entity_signature_hashes == ["fresh-hash-a", "fresh-hash-b"]
    assert override_result.cases[0].artifact_final_entity_signature_labels == {
        "fresh-hash-a": "person",
        "fresh-hash-b": "email",
    }


def test_analyze_benchmark_output_requires_detection_artifact_override_path(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_output_analysis_artifact_override_missing",
        REPO_ROOT / "tools/measurement/analyze_benchmark_output.py",
    )
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    _write_jsonl(
        benchmark_dir / "measurements.jsonl",
        [
            {
                "record_type": "record",
                "run_id": "bio__default__r000",
                "run_tags": {"case_id": "bio__default__r000"},
            }
        ],
    )

    with pytest.raises(ValueError, match="input path does not exist"):
        tool.analyze_benchmark_output(benchmark_dir, detection_artifacts=tmp_path / "missing.jsonl")


def test_write_analysis_tables_exports_case_and_group_tables(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_output_analysis_export",
        REPO_ROOT / "tools/measurement/analyze_benchmark_output.py",
    )
    result = tool.BenchmarkOutputAnalysis(
        benchmark_dir=str(tmp_path / "benchmark"),
        cases=[
            tool.CaseAnalysisRow(
                suite_id="suite",
                workload_id="shell",
                config_id="rules",
                experimental_detection_strategy="rules_only",
                experimental_replacement_strategy="local_structured_substitute",
                dd_parser_compat="raw_json",
                repetition=0,
                case_id="shell__rules__r000",
                run_id="shell__rules__r000",
                final_entity_count=8,
            )
        ],
        groups=[
            tool.GroupAnalysisRow(
                workload_id="shell",
                config_id="rules",
                experimental_detection_strategy="rules_only",
                experimental_replacement_strategy="local_structured_substitute",
                case_count=1,
                median_final_entity_count=8,
                median_observed_successful_requests=0,
                median_observed_input_tokens=0,
                median_observed_output_tokens=0,
                median_observed_failed_request_rate=0,
                median_artifact_final_entity_count=8,
                median_artifact_final_rule_entity_count=8,
            )
        ],
        model_usage=[
            tool.ModelUsageAnalysisRow(
                workload_id="shell",
                config_id="rules",
                experimental_detection_strategy="rules_only",
                experimental_replacement_strategy="local_structured_substitute",
                dd_parser_compat="raw_json",
                case_id="shell__rules__r000",
                run_id="shell__rules__r000",
                workflow_name="entity-detection",
                model_name="nvidia/gliner-pii",
                observed_total_requests=1,
                observed_successful_requests=1,
                observed_total_tokens=1200,
            )
        ],
        model_usage_groups=[
            tool.ModelUsageGroupAnalysisRow(
                workload_id="shell",
                config_id="rules",
                experimental_detection_strategy="rules_only",
                experimental_replacement_strategy="local_structured_substitute",
                dd_parser_compat="raw_json",
                workflow_name="entity-detection",
                model_name="nvidia/gliner-pii",
                case_count=1,
                workflow_count=1,
                sum_observed_total_requests=1,
                sum_observed_successful_requests=1,
                sum_observed_total_tokens=1200,
                median_observed_total_requests=1,
                median_observed_total_tokens=1200,
            )
        ],
    )

    output_dir = tmp_path / "tables"
    tool.write_analysis_tables(result, output_dir, tool.ExportFormat.csv)

    assert pd.read_csv(output_dir / "case_analysis.csv")["case_id"].tolist() == ["shell__rules__r000"]
    assert pd.read_csv(output_dir / "case_analysis.csv")["experimental_replacement_strategy"].tolist() == [
        "local_structured_substitute"
    ]
    assert pd.read_csv(output_dir / "group_analysis.csv")["case_count"].tolist() == [1]
    assert pd.read_csv(output_dir / "model_analysis.csv")["model_name"].tolist() == ["nvidia/gliner-pii"]
    assert pd.read_csv(output_dir / "model_group_analysis.csv")["workflow_count"].tolist() == [1]
    assert (output_dir / "manifest.json").exists()


def test_analyze_benchmark_output_preserves_zero_entity_cases(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_output_analysis_zero",
        REPO_ROOT / "tools/measurement/analyze_benchmark_output.py",
    )
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    _write_jsonl(
        benchmark_dir / "measurements.jsonl",
        [
            {
                "record_type": "record",
                "run_id": "empty__redact__r000",
                "final_entity_count": 0,
                "replacement_count": 0,
                "run_tags": {
                    "workload_id": "empty",
                    "config_id": "redact",
                    "experimental_detection_strategy": "default",
                    "case_id": "empty__redact__r000",
                },
            }
        ],
    )

    result = tool.analyze_benchmark_output(benchmark_dir)

    assert result.cases[0].final_entity_count == 0
    assert result.groups[0].median_final_entity_count == 0


def test_analyze_benchmark_output_groups_replacement_strategies_separately(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_output_analysis_replacement_strategy_groups",
        REPO_ROOT / "tools/measurement/analyze_benchmark_output.py",
    )
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    _write_jsonl(
        benchmark_dir / "measurements.jsonl",
        [
            {
                "record_type": "record",
                "run_id": "secrets__candidate__r000",
                "final_entity_count": 4,
                "run_tags": {
                    "workload_id": "secrets",
                    "config_id": "candidate",
                    "experimental_detection_strategy": "rules_covered_or_default",
                    "experimental_replacement_strategy": "default",
                    "case_id": "secrets__candidate__r000",
                },
            },
            {
                "record_type": "record",
                "run_id": "secrets__candidate__r001",
                "final_entity_count": 4,
                "run_tags": {
                    "workload_id": "secrets",
                    "config_id": "candidate",
                    "experimental_detection_strategy": "rules_covered_or_default",
                    "experimental_replacement_strategy": "local_structured_substitute",
                    "case_id": "secrets__candidate__r001",
                },
            },
        ],
    )

    result = tool.analyze_benchmark_output(benchmark_dir)

    assert result.group_count == 2
    assert {group.experimental_replacement_strategy for group in result.groups} == {
        "default",
        "local_structured_substitute",
    }


def test_analyze_benchmark_output_surfaces_route_counts(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_output_analysis_route_counts",
        REPO_ROOT / "tools/measurement/analyze_benchmark_output.py",
    )
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    _write_jsonl(
        benchmark_dir / "measurements.jsonl",
        [
            {
                "record_type": "model_workflow",
                "run_id": "mixed__router__r000",
                "workflow_name": "entity-detection-rules-covered-router",
                "status": "completed",
                "input_row_count": 2,
                "output_row_count": 2,
                "failed_record_count": 0,
                "elapsed_sec": 0.01,
                "observed_total_requests": 0,
                "observed_successful_requests": 0,
                "observed_failed_requests": 0,
                "observed_input_tokens": 0,
                "observed_output_tokens": 0,
                "observed_total_tokens": 0,
                "route_total_row_count": 2,
                "route_rule_row_count": 1,
                "route_fallback_row_count": 1,
                "run_tags": {
                    "workload_id": "mixed",
                    "config_id": "router",
                    "experimental_detection_strategy": "rules_covered_or_default",
                    "experimental_replacement_strategy": "default",
                    "case_id": "mixed__router__r000",
                },
            }
        ],
    )

    result = tool.analyze_benchmark_output(benchmark_dir)

    case = result.cases[0]
    assert case.route_total_row_count == 2
    assert case.route_rule_row_count == 1
    assert case.route_fallback_row_count == 1
    group = result.groups[0]
    assert group.median_route_total_row_count == 2
    assert group.median_route_rule_row_count == 1
    assert group.median_route_fallback_row_count == 1


def test_analyze_benchmark_output_surfaces_failed_cases(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_output_analysis_failures",
        REPO_ROOT / "tools/measurement/analyze_benchmark_output.py",
    )
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    _write_jsonl(
        benchmark_dir / "measurements.jsonl",
        [
            {
                "record_type": "stage",
                "run_id": "shell__candidate__r000",
                "stage": "Anonymizer._run_internal",
                "status": "completed",
                "elapsed_sec": 1.2,
                "run_tags": {
                    "workload_id": "shell",
                    "config_id": "candidate",
                    "experimental_detection_strategy": "rules_guardrail_detector_only",
                    "repetition": 0,
                    "case_id": "shell__candidate__r000",
                },
            },
            {
                "record_type": "ndd_workflow",
                "run_id": "shell__candidate__r001",
                "workflow_name": "entity-detection",
                "status": "error",
                "elapsed_sec": 0.2,
                "run_tags": {
                    "workload_id": "shell",
                    "config_id": "candidate",
                    "experimental_detection_strategy": "rules_guardrail_detector_only",
                    "repetition": 1,
                    "case_id": "shell__candidate__r001",
                },
            },
            {
                "record_type": "stage",
                "run_id": "shell__candidate__r001",
                "stage": "Anonymizer._run_internal",
                "status": "error",
                "elapsed_sec": 0.2,
                "run_tags": {
                    "workload_id": "shell",
                    "config_id": "candidate",
                    "experimental_detection_strategy": "rules_guardrail_detector_only",
                    "repetition": 1,
                    "case_id": "shell__candidate__r001",
                },
            },
        ],
    )

    result = tool.analyze_benchmark_output(benchmark_dir)

    cases = {row.case_id: row for row in result.cases}
    assert cases["shell__candidate__r000"].case_failed is False
    assert cases["shell__candidate__r000"].error_stage_count == 0
    assert cases["shell__candidate__r000"].error_ndd_workflow_count == 0
    assert cases["shell__candidate__r001"].case_failed is True
    assert cases["shell__candidate__r001"].error_stage_count == 1
    assert cases["shell__candidate__r001"].error_ndd_workflow_count == 1
    assert result.groups[0].failed_case_count == 1
    assert result.groups[0].failed_case_rate == pytest.approx(0.5)
    assert result.groups[0].error_stage_count == 1
    assert result.groups[0].error_ndd_workflow_count == 1
    assert "failed_cases=1/2" in tool.render_result(result, json_output=False)


def test_analyze_benchmark_output_groups_artifact_contribution_metrics(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_output_analysis_artifact_group",
        REPO_ROOT / "tools/measurement/analyze_benchmark_output.py",
    )
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    _write_jsonl(
        benchmark_dir / "measurements.jsonl",
        [
            {
                "record_type": "record",
                "run_id": "bio__default__r000",
                "final_entity_count": 10,
                "run_tags": {
                    "workload_id": "bio",
                    "config_id": "default",
                    "experimental_detection_strategy": "default",
                    "case_id": "bio__default__r000",
                },
            },
            {
                "record_type": "record",
                "run_id": "bio__default__r001",
                "final_entity_count": 14,
                "run_tags": {
                    "workload_id": "bio",
                    "config_id": "default",
                    "experimental_detection_strategy": "default",
                    "case_id": "bio__default__r001",
                },
            },
        ],
    )
    _write_jsonl(
        benchmark_dir / "detection-artifacts.jsonl",
        [
            {
                "workload_id": "bio",
                "config_id": "default",
                "case_id": "bio__default__r000",
                "run_id": "bio__default__r000",
                "seed_entity_count": 9,
                "seed_validation_candidate_count": 9,
                "augmented_entity_count": 4,
                "augmented_new_final_value_count": 1,
                "final_entity_count": 11,
                "final_source_counts": {"detector": 10, "augmenter": 1},
                "final_entity_signature_hashes": ["a", "b"],
                "final_entity_signature_count": 2,
            },
            {
                "workload_id": "bio",
                "config_id": "default",
                "case_id": "bio__default__r001",
                "run_id": "bio__default__r001",
                "seed_entity_count": 13,
                "seed_validation_candidate_count": 13,
                "augmented_entity_count": 8,
                "augmented_new_final_value_count": 3,
                "final_entity_count": 15,
                "final_source_counts": {"detector": 12, "augmenter": 3},
                "final_entity_signature_hashes": ["a", "b", "c", "d"],
                "final_entity_signature_count": 4,
            },
        ],
    )

    result = tool.analyze_benchmark_output(benchmark_dir)

    group = result.groups[0]
    assert group.median_final_entity_count == 12
    assert group.median_observed_successful_requests == 0
    assert group.median_observed_input_tokens == 0
    assert group.median_observed_output_tokens == 0
    assert group.median_observed_failed_request_rate is None
    assert group.median_seed_entity_count == 11
    assert group.median_seed_validation_candidate_count == 11
    assert group.median_augmented_entity_count == 6
    assert group.median_augmented_new_final_value_count == 2
    assert group.median_artifact_final_entity_count == 13
    assert group.median_artifact_final_detector_entity_count == 11
    assert group.median_artifact_final_augmenter_entity_count == 2
    assert group.median_artifact_final_entity_signature_count == 3
