# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
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
    spec.loader.exec_module(module)
    return module


def test_compare_case_analysis_by_strategy_reports_safety_and_cost_deltas() -> None:
    tool = load_tool("measurement_compare_strategy_pairs", REPO_ROOT / "tools/measurement/compare_strategy_pairs.py")
    table = pd.DataFrame(
        [
            {
                "workload_id": "shell-1",
                "config_id": "shell-no-augment",
                "experimental_detection_strategy": "no_augment",
                "experimental_replacement_strategy": "default",
                "case_id": "shell__base",
                "pipeline_elapsed_sec": 4.5,
                "observed_total_requests": 3,
                "observed_total_tokens": 2875,
                "observed_failed_requests": 1,
                "final_entity_count": 4,
                "seed_validation_candidate_count": 5,
                "augmented_entity_count": 2,
                "augmented_new_final_value_count": 1,
                "artifact_final_detector_entity_count": 4,
            },
            {
                "workload_id": "shell-1",
                "config_id": "shell-native",
                "experimental_detection_strategy": "native_candidate_validate_no_augment",
                "experimental_replacement_strategy": "local_structured_substitute",
                "case_id": "shell__candidate",
                "pipeline_elapsed_sec": 0.8,
                "observed_total_requests": 1,
                "observed_total_tokens": 101,
                "observed_failed_requests": 0,
                "final_entity_count": 8,
                "seed_validation_candidate_count": 0,
                "augmented_entity_count": 0,
                "augmented_new_final_value_count": 0,
                "artifact_final_augmenter_entity_count": 8,
            },
            {
                "workload_id": "legal-1",
                "config_id": "legal-no-augment",
                "experimental_detection_strategy": "no_augment",
                "experimental_replacement_strategy": "default",
                "case_id": "legal__base",
                "pipeline_elapsed_sec": 21,
                "observed_total_requests": 3,
                "observed_total_tokens": 8847,
                "observed_failed_requests": 1,
                "final_entity_count": 26,
                "seed_validation_candidate_count": 40,
                "artifact_final_detector_entity_count": 26,
            },
            {
                "workload_id": "legal-1",
                "config_id": "legal-native",
                "experimental_detection_strategy": "native_candidate_validate_no_augment",
                "experimental_replacement_strategy": "local_structured_substitute",
                "case_id": "legal__candidate",
                "pipeline_elapsed_sec": 20.9,
                "observed_total_requests": 3,
                "observed_total_tokens": 8847,
                "observed_failed_requests": 1,
                "final_entity_count": 26,
                "seed_validation_candidate_count": 40,
                "artifact_final_detector_entity_count": 26,
            },
        ]
    )

    rows = tool.compare_case_analysis(
        table,
        baseline_strategy="no_augment",
        candidate_strategy="native_candidate_validate_no_augment",
    )

    by_workload = {row.workload_id: row for row in rows}
    shell = by_workload["shell-1"]
    assert shell.baseline_replacement_strategy == "default"
    assert shell.candidate_replacement_strategy == "local_structured_substitute"
    assert shell.final_entity_count_delta == 4
    assert shell.observed_total_requests_delta == -2
    assert shell.observed_total_tokens_delta == -2774
    assert shell.seed_validation_candidate_count_delta == -5
    assert shell.baseline_augmented_entity_count == 2
    assert shell.candidate_augmented_entity_count == 0
    assert shell.augmented_entity_count_delta == -2
    assert shell.baseline_augmented_new_final_value_count == 1
    assert shell.candidate_augmented_new_final_value_count == 0
    assert shell.augmented_new_final_value_count_delta == -1
    assert shell.candidate_augmenter_entity_count == 8
    assert shell.safety_verdict == "review"
    assert shell.performance_verdict == "improved"
    assert shell.candidate_verdict == "review"
    assert shell.flags == ["no_candidate_detector_entities"]

    legal = by_workload["legal-1"]
    assert legal.baseline_replacement_strategy == "default"
    assert legal.candidate_replacement_strategy == "local_structured_substitute"
    assert legal.final_entity_count_delta == 0
    assert legal.observed_total_tokens_delta == 0
    assert legal.candidate_detector_entity_count == 26
    assert legal.safety_verdict == "pass"
    assert legal.performance_verdict == "improved"
    assert legal.candidate_verdict == "candidate_viable"
    assert legal.flags == []


def test_compare_case_analysis_rejects_ambiguous_strategy_selector() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_ambiguous",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "shell-1",
                "config_id": "base-a",
                "experimental_detection_strategy": "no_augment",
                "case_id": "a",
            },
            {
                "workload_id": "shell-1",
                "config_id": "base-b",
                "experimental_detection_strategy": "no_augment",
                "case_id": "b",
            },
            {
                "workload_id": "shell-1",
                "config_id": "candidate",
                "experimental_detection_strategy": "detector_only",
                "case_id": "c",
            },
        ]
    )

    with pytest.raises(ValueError, match="baseline selector matched multiple configs"):
        tool.compare_case_analysis(table, baseline_strategy="no_augment", candidate_strategy="detector_only")


def test_compare_case_analysis_rejects_candidate_synthetic_original_collisions() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_replacement_collisions",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "legal-1",
                "config_id": "baseline",
                "experimental_detection_strategy": "default",
                "case_id": "base",
                "pipeline_elapsed_sec": 20,
                "observed_total_requests": 4,
                "observed_total_tokens": 4000,
                "final_entity_count": 4,
                "replacement_synthetic_original_collision_count": 0,
                "replacement_synthetic_original_collision_value_count": 0,
            },
            {
                "workload_id": "legal-1",
                "config_id": "candidate",
                "experimental_detection_strategy": "candidate",
                "case_id": "cand",
                "pipeline_elapsed_sec": 10,
                "observed_total_requests": 2,
                "observed_total_tokens": 1000,
                "final_entity_count": 4,
                "replacement_synthetic_original_collision_count": 1,
                "replacement_synthetic_original_collision_value_count": 1,
                "replacement_synthetic_original_collision_label_counts.date": 1,
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_strategy="default", candidate_strategy="candidate")

    assert len(rows) == 1
    row = rows[0]
    assert row.candidate_replacement_synthetic_original_collision_count == 1
    assert row.replacement_synthetic_original_collision_count_delta == 1
    assert row.candidate_replacement_synthetic_original_collision_label_counts == {"date": 1}
    assert "candidate_replacement_synthetic_original_collision" in row.flags
    assert row.value_protection_verdict == "fail"
    assert row.safety_verdict == "fail"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "reject"


def test_compare_case_analysis_rejects_candidate_missing_replacement_map_entries() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_missing_replacement_map_entries",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "structured-identifiers",
                "config_id": "baseline",
                "experimental_detection_strategy": "default",
                "case_id": "base",
                "pipeline_elapsed_sec": 20,
                "observed_total_requests": 4,
                "observed_total_tokens": 4000,
                "final_entity_count": 4,
                "replacement_missing_final_entity_count": 0,
                "replacement_missing_final_value_count": 0,
            },
            {
                "workload_id": "structured-identifiers",
                "config_id": "candidate",
                "experimental_detection_strategy": "default",
                "case_id": "cand",
                "pipeline_elapsed_sec": 10,
                "observed_total_requests": 2,
                "observed_total_tokens": 1000,
                "final_entity_count": 4,
                "replacement_missing_final_entity_count": 1,
                "replacement_missing_final_value_count": 1,
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_config="baseline", candidate_config="candidate")

    assert len(rows) == 1
    row = rows[0]
    assert row.candidate_replacement_missing_final_entity_count == 1
    assert row.replacement_missing_final_entity_count_delta == 1
    assert "candidate_replacement_missing_final_entity" in row.flags
    assert row.value_protection_verdict == "fail"
    assert row.safety_verdict == "fail"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "reject"


def test_compare_case_tables_allows_candidate_from_separate_run() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_cross_run",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    baseline = pd.DataFrame(
        [
            {
                "workload_id": "legal-5",
                "config_id": "legal-no-augment",
                "experimental_detection_strategy": "no_augment",
                "case_id": "legal__base",
                "observed_total_tokens": 55790,
                "final_entity_count": 193,
                "artifact_final_detector_entity_count": 172,
            }
        ]
    )
    candidate = pd.DataFrame(
        [
            {
                "workload_id": "legal-5",
                "config_id": "legal-native-validate",
                "experimental_detection_strategy": "detector_native_validate_no_augment",
                "case_id": "legal__candidate",
                "observed_total_tokens": 55805,
                "final_entity_count": 193,
                "artifact_final_detector_entity_count": 172,
            }
        ]
    )

    rows = tool.compare_case_tables(
        baseline,
        candidate,
        baseline_strategy="no_augment",
        candidate_strategy="detector_native_validate_no_augment",
    )

    assert len(rows) == 1
    assert rows[0].workload_id == "legal-5"
    assert rows[0].baseline_config_id == "legal-no-augment"
    assert rows[0].candidate_config_id == "legal-native-validate"
    assert rows[0].observed_total_tokens_delta == 15
    assert rows[0].flags == ["token_increase"]


def test_compare_case_analysis_preserves_augmentation_contribution_deltas() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_augmentation",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "legal-2",
                "config_id": "default",
                "experimental_detection_strategy": "default",
                "case_id": "default-r0",
                "augmented_entity_count": 8,
                "augmented_new_final_value_count": 3,
                "artifact_final_augmenter_entity_count": 3,
                "artifact_final_entity_signature_hashes": ["a", "b", "c"],
            },
            {
                "workload_id": "legal-2",
                "config_id": "no-augment",
                "experimental_detection_strategy": "no_augment",
                "case_id": "no-augment-r0",
                "augmented_entity_count": 0,
                "augmented_new_final_value_count": 0,
                "artifact_final_augmenter_entity_count": 0,
                "artifact_final_entity_signature_hashes": ["a", "b"],
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_config="default", candidate_config="no-augment")

    assert len(rows) == 1
    row = rows[0]
    assert row.baseline_augmented_entity_count == 8
    assert row.candidate_augmented_entity_count == 0
    assert row.augmented_entity_count_delta == -8
    assert row.baseline_augmented_new_final_value_count == 3
    assert row.candidate_augmented_new_final_value_count == 0
    assert row.augmented_new_final_value_count_delta == -3
    assert row.baseline_augmenter_entity_count == 3
    assert row.candidate_augmenter_entity_count == 0


def test_compare_case_analysis_review_gates_detector_only_candidate_shell_case() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_detector_only",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "bio-1",
                "config_id": "default",
                "experimental_detection_strategy": "default",
                "case_id": "bio__default",
                "pipeline_elapsed_sec": 20,
                "observed_total_requests": 4,
                "observed_total_tokens": 4000,
                "final_entity_count": 2,
                "artifact_final_detector_entity_count": 2,
                "artifact_final_entity_signature_hashes": ["a", "b"],
            },
            {
                "workload_id": "bio-1",
                "config_id": "detector-only",
                "experimental_detection_strategy": "detector_only",
                "case_id": "bio__detector",
                "pipeline_elapsed_sec": 5,
                "observed_total_requests": 1,
                "observed_total_tokens": 1000,
                "final_entity_count": 2,
                "artifact_final_detector_entity_count": 2,
                "artifact_final_entity_signature_hashes": ["a", "b"],
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_strategy="default", candidate_strategy="detector_only")

    assert len(rows) == 1
    row = rows[0]
    assert row.safety_verdict == "review"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "review"
    assert row.flags == ["candidate_skips_llm_validation"]


def test_compare_case_analysis_review_gates_detector_only_candidates() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_detector_only",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "shell-1",
                "config_id": "default",
                "experimental_detection_strategy": "default",
                "case_id": "shell__default",
                "pipeline_elapsed_sec": 8,
                "observed_total_requests": 4,
                "observed_total_tokens": 4000,
                "final_entity_count": 2,
                "artifact_final_detector_entity_count": 2,
                "artifact_final_entity_signature_hashes": ["a", "b"],
            },
            {
                "workload_id": "shell-1",
                "config_id": "detector-only",
                "experimental_detection_strategy": "detector_only",
                "case_id": "shell__candidate",
                "pipeline_elapsed_sec": 1,
                "observed_total_requests": 1,
                "observed_total_tokens": 200,
                "final_entity_count": 2,
                "artifact_final_detector_entity_count": 2,
                "artifact_final_entity_signature_hashes": ["a", "b"],
            },
        ]
    )

    rows = tool.compare_case_analysis(
        table,
        baseline_strategy="default",
        candidate_strategy="detector_only",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.safety_verdict == "review"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "review"
    assert row.flags == ["candidate_skips_llm_validation"]


def test_compare_case_analysis_review_gates_non_detector_sources_when_signatures_match() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_non_detector_sources",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "shell-1",
                "config_id": "native-source-default",
                "experimental_detection_strategy": "default",
                "case_id": "shell__default",
                "pipeline_elapsed_sec": 21.4,
                "observed_total_requests": 8,
                "observed_total_tokens": 9854,
                "final_entity_count": 2,
                "artifact_final_detector_entity_count": 2,
                "artifact_final_entity_signature_hashes": ["a", "b"],
                "artifact_final_entity_signature_labels": {"a": "api_key", "b": "password"},
            },
            {
                "workload_id": "shell-1",
                "config_id": "native-source-candidate",
                "experimental_detection_strategy": "native_single_pass",
                "case_id": "shell__candidate",
                "pipeline_elapsed_sec": 0.001,
                "observed_total_requests": 0,
                "observed_total_tokens": 0,
                "final_entity_count": 2,
                "artifact_final_augmenter_entity_count": 2,
                "artifact_final_entity_signature_hashes": ["a", "b"],
                "artifact_final_entity_signature_labels": {"a": "api_key", "b": "password"},
            },
        ]
    )

    rows = tool.compare_case_analysis(
        table,
        baseline_config="native-source-default",
        candidate_config="native-source-candidate",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.baseline_only_final_entity_signature_count == 0
    assert row.candidate_only_final_entity_signature_count == 0
    assert row.shared_final_entity_signature_label_counts == {"api_key": 1, "password": 1}
    assert row.value_protection_verdict == "pass"
    assert row.signature_parity_verdict == "pass"
    assert row.safety_verdict == "review"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "review"
    assert row.flags == ["no_candidate_detector_entities"]


def test_compare_case_analysis_flags_signature_loss_even_when_counts_match() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_signature_loss",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "bio-5",
                "config_id": "default",
                "experimental_detection_strategy": "default",
                "case_id": "bio__default",
                "final_entity_count": 3,
                "artifact_final_entity_signature_hashes": '["a","b","c"]',
                "artifact_final_entity_signature_labels": '{"a":"first_name","b":"city","c":"first_name"}',
            },
            {
                "workload_id": "bio-5",
                "config_id": "no-augment",
                "experimental_detection_strategy": "no_augment",
                "case_id": "bio__no_augment",
                "final_entity_count": 3,
                "artifact_final_entity_signature_hashes": ["a", "b", "d"],
                "artifact_final_entity_signature_labels": {"a": "first_name", "b": "city", "d": "last_name"},
            },
        ]
    )

    rows = tool.compare_case_analysis(
        table,
        baseline_config="default",
        candidate_config="no-augment",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.final_entity_count_delta == 0
    assert row.baseline_only_final_entity_signature_count == 1
    assert row.candidate_only_final_entity_signature_count == 1
    assert row.shared_final_entity_signature_count == 2
    assert row.baseline_only_final_entity_signature_label_counts == {"first_name": 1}
    assert row.candidate_only_final_entity_signature_label_counts == {"last_name": 1}
    assert row.shared_final_entity_signature_label_counts == {"city": 1, "first_name": 1}
    assert row.safety_verdict == "fail"
    assert row.performance_verdict == "unknown"
    assert row.candidate_verdict == "reject"
    assert row.flags == ["entity_signature_loss"]


def test_compare_case_analysis_treats_baseline_subspan_as_candidate_covered() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_candidate_span_coverage",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "structured-identifiers",
                "config_id": "default",
                "experimental_detection_strategy": "default",
                "case_id": "default-r0",
                "pipeline_elapsed_sec": 10,
                "observed_total_requests": 4,
                "observed_total_tokens": 4000,
                "final_entity_count": 2,
                "artifact_final_detector_entity_count": 2,
                "artifact_final_entity_signature_hashes": ["api-token", "pin"],
                "artifact_final_entity_signature_labels": {"api-token": "api_key", "pin": "pin"},
                "artifact_final_entity_signature_details": {
                    "api-token": {
                        "label": "api_key",
                        "source": "augmenter",
                        "row_index": 0,
                        "start_position": 42,
                        "end_position": 66,
                        "value_hash": "token-hash",
                        "value_length": 24,
                    },
                    "pin": {
                        "label": "pin",
                        "source": "detector",
                        "row_index": 0,
                        "start_position": 90,
                        "end_position": 95,
                        "value_hash": "pin-hash",
                        "value_length": 5,
                    },
                },
            },
            {
                "workload_id": "structured-identifiers",
                "config_id": "native-local",
                "experimental_detection_strategy": "native_single_pass",
                "case_id": "candidate-r0",
                "pipeline_elapsed_sec": 0.01,
                "observed_total_requests": 0,
                "observed_total_tokens": 0,
                "final_entity_count": 2,
                "artifact_final_augmenter_entity_count": 2,
                "artifact_final_entity_signature_hashes": ["cookie", "pin"],
                "artifact_final_entity_signature_labels": {"cookie": "http_cookie", "pin": "pin"},
                "artifact_final_entity_signature_details": {
                    "cookie": {
                        "label": "http_cookie",
                        "source": "native",
                        "row_index": 0,
                        "start_position": 30,
                        "end_position": 80,
                        "value_hash": "cookie-hash",
                        "value_length": 50,
                    },
                    "pin": {
                        "label": "pin",
                        "source": "native",
                        "row_index": 0,
                        "start_position": 90,
                        "end_position": 95,
                        "value_hash": "pin-hash",
                        "value_length": 5,
                    },
                },
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_config="default", candidate_config="native-local")

    assert len(rows) == 1
    row = rows[0]
    assert row.baseline_only_final_entity_signature_count == 1
    assert row.baseline_only_candidate_covered_signature_count == 1
    assert row.baseline_only_candidate_uncovered_signature_count == 0
    assert row.baseline_only_candidate_covered_signature_label_counts == {"api_key": 1}
    assert row.baseline_only_candidate_uncovered_signature_label_counts == {}
    assert "entity_signature_loss" not in row.flags
    assert row.safety_verdict == "review"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "review"


def test_compare_case_analysis_review_gates_covered_label_mismatch() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_label_mismatch",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "legal-row",
                "config_id": "default",
                "experimental_detection_strategy": "default",
                "case_id": "default-r0",
                "pipeline_elapsed_sec": 10,
                "observed_total_requests": 4,
                "observed_total_tokens": 1000,
                "final_entity_count": 1,
                "artifact_final_detector_entity_count": 1,
                "artifact_final_entity_signature_hashes": ["dob"],
                "artifact_final_entity_signature_labels": {"dob": "date_of_birth"},
                "artifact_final_entity_signature_details": {
                    "dob": {
                        "label": "date_of_birth",
                        "source": "detector",
                        "row_index": 0,
                        "start_position": 20,
                        "end_position": 35,
                        "value_hash": "date-hash",
                        "value_length": 15,
                    },
                },
            },
            {
                "workload_id": "legal-row",
                "config_id": "native-validation",
                "experimental_detection_strategy": "detector_native_validate_no_augment",
                "case_id": "candidate-r0",
                "pipeline_elapsed_sec": 2,
                "observed_total_requests": 2,
                "observed_total_tokens": 300,
                "final_entity_count": 1,
                "artifact_final_detector_entity_count": 1,
                "artifact_final_entity_signature_hashes": ["date"],
                "artifact_final_entity_signature_labels": {"date": "date"},
                "artifact_final_entity_signature_details": {
                    "date": {
                        "label": "date",
                        "source": "detector",
                        "row_index": 0,
                        "start_position": 20,
                        "end_position": 35,
                        "value_hash": "date-hash",
                        "value_length": 15,
                    },
                },
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_config="default", candidate_config="native-validation")

    assert len(rows) == 1
    row = rows[0]
    assert row.baseline_only_candidate_covered_signature_count == 1
    assert row.baseline_only_candidate_uncovered_signature_count == 0
    assert row.baseline_only_candidate_label_mismatch_signature_count == 1
    assert row.baseline_only_candidate_label_mismatch_signature_label_counts == {"date_of_birth": 1}
    assert "entity_signature_loss" not in row.flags
    assert "covered_label_mismatch" in row.flags
    assert row.value_protection_verdict == "pass"
    assert row.signature_parity_verdict == "review"
    assert row.safety_verdict == "review"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "review"


def test_compare_case_analysis_treats_high_overlap_candidate_span_as_covered() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_candidate_span_overlap",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "structured-identifiers",
                "config_id": "default",
                "experimental_detection_strategy": "default",
                "case_id": "default-r0",
                "pipeline_elapsed_sec": 10,
                "observed_total_requests": 4,
                "observed_total_tokens": 4000,
                "final_entity_count": 1,
                "artifact_final_detector_entity_count": 1,
                "artifact_final_entity_signature_hashes": ["token-assignment"],
                "artifact_final_entity_signature_labels": {"token-assignment": "http_cookie"},
                "artifact_final_entity_signature_details": {
                    "token-assignment": {
                        "label": "http_cookie",
                        "source": "augmenter",
                        "row_index": 0,
                        "start_position": 20,
                        "end_position": 68,
                        "value_hash": "token-assignment-hash",
                        "value_length": 48,
                    },
                },
            },
            {
                "workload_id": "structured-identifiers",
                "config_id": "native-local",
                "experimental_detection_strategy": "native_single_pass",
                "case_id": "candidate-r0",
                "pipeline_elapsed_sec": 0.01,
                "observed_total_requests": 0,
                "observed_total_tokens": 0,
                "final_entity_count": 1,
                "artifact_final_augmenter_entity_count": 1,
                "artifact_final_entity_signature_hashes": ["token-value"],
                "artifact_final_entity_signature_labels": {"token-value": "api_key"},
                "artifact_final_entity_signature_details": {
                    "token-value": {
                        "label": "api_key",
                        "source": "native",
                        "row_index": 0,
                        "start_position": 26,
                        "end_position": 69,
                        "value_hash": "token-value-hash",
                        "value_length": 43,
                    },
                },
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_config="default", candidate_config="native-local")

    assert len(rows) == 1
    row = rows[0]
    assert row.baseline_only_final_entity_signature_count == 1
    assert row.baseline_only_candidate_covered_signature_count == 1
    assert row.baseline_only_candidate_overlapping_signature_count == 1
    assert row.baseline_only_candidate_uncovered_signature_count == 0
    assert row.baseline_only_candidate_overlapping_signature_label_counts == {"http_cookie": 1}
    assert row.baseline_only_candidate_uncovered_signature_label_counts == {}
    assert "entity_signature_loss" not in row.flags
    assert row.value_protection_verdict == "pass"
    assert row.signature_parity_verdict == "review"
    assert row.safety_verdict == "review"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "review"


def test_compare_case_analysis_treats_small_assignment_prefix_gap_as_boundary_overlap() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_candidate_span_boundary",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "structured-identifiers",
                "config_id": "default",
                "experimental_detection_strategy": "default",
                "case_id": "default-r0",
                "pipeline_elapsed_sec": 10,
                "observed_total_requests": 4,
                "observed_total_tokens": 4000,
                "final_entity_count": 1,
                "artifact_final_detector_entity_count": 1,
                "artifact_final_entity_signature_hashes": ["login-assignment"],
                "artifact_final_entity_signature_labels": {"login-assignment": "unique_id"},
                "artifact_final_entity_signature_details": {
                    "login-assignment": {
                        "label": "unique_id",
                        "source": "detector",
                        "row_index": 0,
                        "start_position": 20,
                        "end_position": 38,
                        "value_hash": "login-assignment-hash",
                        "value_length": 18,
                    },
                },
            },
            {
                "workload_id": "structured-identifiers",
                "config_id": "native-local",
                "experimental_detection_strategy": "native_single_pass",
                "case_id": "candidate-r0",
                "pipeline_elapsed_sec": 0.01,
                "observed_total_requests": 0,
                "observed_total_tokens": 0,
                "final_entity_count": 1,
                "artifact_final_augmenter_entity_count": 1,
                "artifact_final_entity_signature_hashes": ["login-value"],
                "artifact_final_entity_signature_labels": {"login-value": "user_name"},
                "artifact_final_entity_signature_details": {
                    "login-value": {
                        "label": "user_name",
                        "source": "native",
                        "row_index": 0,
                        "start_position": 26,
                        "end_position": 38,
                        "value_hash": "login-value-hash",
                        "value_length": 12,
                    },
                },
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_config="default", candidate_config="native-local")

    assert len(rows) == 1
    row = rows[0]
    assert row.baseline_only_candidate_covered_signature_count == 1
    assert row.baseline_only_candidate_overlapping_signature_count == 1
    assert row.baseline_only_candidate_uncovered_signature_count == 0
    assert "entity_signature_loss" not in row.flags
    assert "span_boundary_mismatch" in row.flags
    assert row.value_protection_verdict == "pass"
    assert row.signature_parity_verdict == "review"
    assert row.safety_verdict == "review"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "review"


def test_compare_case_analysis_flags_replacement_only_detection_instability() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_replacement_only_detection_instability",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "structured-identifiers",
                "config_id": "dd-substitute",
                "experimental_detection_strategy": "default",
                "experimental_replacement_strategy": "default",
                "case_id": "default-r0",
                "pipeline_elapsed_sec": 10,
                "observed_total_requests": 4,
                "observed_total_tokens": 4000,
                "final_entity_count": 1,
                "artifact_final_detector_entity_count": 1,
                "artifact_final_entity_signature_hashes": ["token-assignment"],
                "artifact_final_entity_signature_labels": {"token-assignment": "http_cookie"},
                "artifact_final_entity_signature_details": {
                    "token-assignment": {
                        "label": "http_cookie",
                        "source": "detector",
                        "row_index": 0,
                        "start_position": 20,
                        "end_position": 70,
                        "value_hash": "token-assignment-hash",
                        "value_length": 50,
                    },
                },
            },
            {
                "workload_id": "structured-identifiers",
                "config_id": "local-substitute",
                "experimental_detection_strategy": "default",
                "experimental_replacement_strategy": "local_structured_substitute",
                "case_id": "candidate-r0",
                "pipeline_elapsed_sec": 7,
                "observed_total_requests": 3,
                "observed_total_tokens": 3000,
                "final_entity_count": 1,
                "artifact_final_detector_entity_count": 1,
                "artifact_final_entity_signature_hashes": ["token-value"],
                "artifact_final_entity_signature_labels": {"token-value": "api_key"},
                "artifact_final_entity_signature_details": {
                    "token-value": {
                        "label": "api_key",
                        "source": "detector",
                        "row_index": 0,
                        "start_position": 26,
                        "end_position": 70,
                        "value_hash": "token-value-hash",
                        "value_length": 44,
                    },
                },
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_config="dd-substitute", candidate_config="local-substitute")

    assert len(rows) == 1
    row = rows[0]
    assert "covered_label_mismatch" in row.flags
    assert "replacement_only_detection_instability" in row.flags
    assert row.value_protection_verdict == "pass"
    assert row.signature_parity_verdict == "review"
    assert row.safety_verdict == "review"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "review"


def test_compare_case_analysis_rejects_candidate_original_value_leaks() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_original_value_leak",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "structured-secrets",
                "config_id": "default",
                "experimental_detection_strategy": "default",
                "case_id": "structured__default",
                "pipeline_elapsed_sec": 10,
                "observed_total_tokens": 1000,
                "final_entity_count": 2,
                "original_value_leak_count": 0,
                "original_value_leak_record_count": 0,
                "artifact_final_entity_signature_hashes": ["a", "b"],
                "artifact_final_entity_signature_labels": {"a": "api_key", "b": "password"},
            },
            {
                "workload_id": "structured-secrets",
                "config_id": "candidate",
                "experimental_detection_strategy": "native_single_pass",
                "case_id": "structured__candidate",
                "pipeline_elapsed_sec": 1,
                "observed_total_tokens": 0,
                "final_entity_count": 2,
                "original_value_leak_count": 2,
                "original_value_leak_record_count": 1,
                "original_value_leak_label_counts.api_key": 1,
                "original_value_leak_label_counts.password": 1,
                "artifact_final_entity_signature_hashes": ["a", "b"],
                "artifact_final_entity_signature_labels": {"a": "api_key", "b": "password"},
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_config="default", candidate_config="candidate")

    assert len(rows) == 1
    row = rows[0]
    assert row.candidate_original_value_leak_count == 2
    assert row.candidate_original_value_leak_record_count == 1
    assert row.original_value_leak_count_delta == 2
    assert row.candidate_original_value_leak_label_counts == {"api_key": 1, "password": 1}
    assert row.safety_verdict == "fail"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "reject"
    assert row.flags == ["candidate_original_value_leak"]


def test_compare_case_analysis_marks_clean_but_expensive_candidate_for_review() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_verdicts",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "legal-5",
                "config_id": "default",
                "experimental_detection_strategy": "default",
                "case_id": "legal__default",
                "pipeline_elapsed_sec": 20,
                "observed_total_tokens": 1000,
                "final_entity_count": 12,
                "artifact_final_entity_signature_hashes": ["a", "b"],
            },
            {
                "workload_id": "legal-5",
                "config_id": "candidate",
                "experimental_detection_strategy": "candidate",
                "case_id": "legal__candidate",
                "pipeline_elapsed_sec": 25,
                "observed_total_tokens": 1200,
                "final_entity_count": 12,
                "artifact_final_entity_signature_hashes": ["a", "b"],
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_config="default", candidate_config="candidate")

    assert len(rows) == 1
    assert rows[0].safety_verdict == "pass"
    assert rows[0].performance_verdict == "regressed"
    assert rows[0].candidate_verdict == "review"
    assert rows[0].flags == ["token_increase"]


def test_compare_case_analysis_separates_bridge_fallbacks_from_provider_failures() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_trace_adjusted",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "bio-5",
                "config_id": "default",
                "experimental_detection_strategy": "default",
                "case_id": "bio__default",
                "pipeline_elapsed_sec": 20,
                "observed_total_requests": 4,
                "observed_failed_requests": 1,
                "observed_bridge_fallback_requests": 1,
                "observed_non_bridge_total_requests": 3,
                "observed_non_bridge_failed_requests": 0,
                "final_entity_count": 12,
                "artifact_final_detector_entity_count": 12,
                "artifact_final_entity_signature_hashes": ["a", "b"],
            },
            {
                "workload_id": "bio-5",
                "config_id": "windowed",
                "experimental_detection_strategy": "windowed",
                "case_id": "bio__windowed",
                "pipeline_elapsed_sec": 15,
                "observed_total_requests": 6,
                "observed_failed_requests": 2,
                "observed_bridge_fallback_requests": 2,
                "observed_non_bridge_total_requests": 4,
                "observed_non_bridge_failed_requests": 0,
                "final_entity_count": 12,
                "artifact_final_detector_entity_count": 12,
                "artifact_final_entity_signature_hashes": ["a", "b"],
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_config="default", candidate_config="windowed")

    assert len(rows) == 1
    row = rows[0]
    assert row.observed_failed_requests_delta == 1
    assert row.observed_bridge_fallback_requests_delta == 1
    assert row.observed_non_bridge_failed_requests_delta == 0
    assert "bridge_fallback_increase" in row.flags
    assert "failed_request_increase" not in row.flags


def test_compare_case_analysis_flags_non_bridge_provider_failure_increase() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_non_bridge_failure",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "legal-5",
                "config_id": "default",
                "experimental_detection_strategy": "default",
                "case_id": "legal__default",
                "observed_total_requests": 5,
                "observed_failed_requests": 1,
                "observed_bridge_fallback_requests": 1,
                "observed_non_bridge_total_requests": 4,
                "observed_non_bridge_failed_requests": 0,
                "final_entity_count": 10,
                "artifact_final_detector_entity_count": 10,
                "artifact_final_entity_signature_hashes": ["a"],
            },
            {
                "workload_id": "legal-5",
                "config_id": "candidate",
                "experimental_detection_strategy": "candidate",
                "case_id": "legal__candidate",
                "observed_total_requests": 5,
                "observed_failed_requests": 2,
                "observed_bridge_fallback_requests": 1,
                "observed_non_bridge_total_requests": 4,
                "observed_non_bridge_failed_requests": 1,
                "final_entity_count": 10,
                "artifact_final_detector_entity_count": 10,
                "artifact_final_entity_signature_hashes": ["a"],
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_config="default", candidate_config="candidate")

    assert len(rows) == 1
    row = rows[0]
    assert row.observed_bridge_fallback_requests_delta == 0
    assert row.observed_non_bridge_failed_requests_delta == 1
    assert "bridge_fallback_increase" not in row.flags
    assert "failed_request_increase" in row.flags
    assert row.safety_verdict == "review"
    assert row.performance_verdict == "unchanged"
    assert row.candidate_verdict == "review"


def test_compare_case_analysis_rejects_candidate_case_failures() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_case_failure",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "shell-5",
                "config_id": "default",
                "experimental_detection_strategy": "default",
                "case_id": "default-r0",
                "pipeline_elapsed_sec": 8,
                "case_failed": False,
                "artifact_final_entity_signature_hashes": ["a", "b"],
            },
            {
                "workload_id": "shell-5",
                "config_id": "default",
                "experimental_detection_strategy": "default",
                "case_id": "default-r1",
                "pipeline_elapsed_sec": 8,
                "case_failed": False,
                "artifact_final_entity_signature_hashes": ["a", "b"],
            },
            {
                "workload_id": "shell-5",
                "config_id": "candidate",
                "experimental_detection_strategy": "detector_only",
                "case_id": "candidate-r0",
                "pipeline_elapsed_sec": 2,
                "case_failed": False,
                "artifact_final_entity_signature_hashes": ["a", "b"],
            },
            {
                "workload_id": "shell-5",
                "config_id": "candidate",
                "experimental_detection_strategy": "detector_only",
                "case_id": "candidate-r1",
                "pipeline_elapsed_sec": 0.2,
                "case_failed": True,
                "artifact_final_entity_signature_hashes": ["a", "b"],
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_config="default", candidate_config="candidate")

    assert len(rows) == 1
    row = rows[0]
    assert row.baseline_failed_case_count == 0
    assert row.candidate_failed_case_count == 1
    assert row.failed_case_count_delta == 1
    assert row.safety_verdict == "fail"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "reject"
    assert "candidate_case_failures" in row.flags


def test_compare_case_analysis_counts_model_workflow_errors_as_case_failures() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_model_workflow_failure",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "shell-5",
                "config_id": "default",
                "case_id": "default-r0",
                "pipeline_elapsed_sec": 8,
                "error_stage_count": 0,
                "error_ndd_workflow_count": 0,
                "error_model_workflow_count": 0,
                "artifact_final_entity_signature_hashes": ["a", "b"],
            },
            {
                "workload_id": "shell-5",
                "config_id": "candidate",
                "case_id": "candidate-r0",
                "pipeline_elapsed_sec": 2,
                "error_stage_count": 0,
                "error_ndd_workflow_count": 0,
                "error_model_workflow_count": 1,
                "artifact_final_entity_signature_hashes": ["a", "b"],
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_config="default", candidate_config="candidate")

    assert len(rows) == 1
    row = rows[0]
    assert row.baseline_failed_case_count == 0
    assert row.candidate_failed_case_count == 1
    assert row.safety_verdict == "fail"
    assert row.candidate_verdict == "reject"
    assert "candidate_case_failures" in row.flags


def test_compare_case_analysis_review_gates_baseline_case_failures() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_baseline_case_failure",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "bio-5",
                "config_id": "baseline",
                "case_id": "baseline-r0",
                "pipeline_elapsed_sec": 30,
                "case_failed": True,
                "artifact_final_entity_signature_hashes": [],
            },
            {
                "workload_id": "bio-5",
                "config_id": "candidate",
                "case_id": "candidate-r0",
                "pipeline_elapsed_sec": 20,
                "case_failed": False,
                "artifact_final_entity_signature_hashes": ["a", "b"],
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_config="baseline", candidate_config="candidate")

    assert len(rows) == 1
    row = rows[0]
    assert row.baseline_failed_case_count == 1
    assert row.candidate_failed_case_count == 0
    assert row.failed_case_count_delta == -1
    assert row.safety_verdict == "review"
    assert row.performance_verdict == "improved"
    assert row.candidate_verdict == "review"
    assert row.flags == ["baseline_case_failures"]


def test_compare_case_analysis_flags_repeated_signature_instability() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_stability",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "legal-5",
                "config_id": "default",
                "case_id": "default-r0",
                "artifact_final_entity_signature_hashes": ["a", "b"],
                "artifact_final_entity_signature_labels": {"a": "person", "b": "date"},
            },
            {
                "workload_id": "legal-5",
                "config_id": "default",
                "case_id": "default-r1",
                "artifact_final_entity_signature_hashes": ["a", "b"],
                "artifact_final_entity_signature_labels": {"a": "person", "b": "date"},
            },
            {
                "workload_id": "legal-5",
                "config_id": "candidate",
                "case_id": "candidate-r0",
                "artifact_final_entity_signature_hashes": ["a", "b"],
                "artifact_final_entity_signature_labels": {"a": "person", "b": "date"},
            },
            {
                "workload_id": "legal-5",
                "config_id": "candidate",
                "case_id": "candidate-r1",
                "artifact_final_entity_signature_hashes": ["a"],
                "artifact_final_entity_signature_labels": {"a": "person"},
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_config="default", candidate_config="candidate")

    assert len(rows) == 1
    row = rows[0]
    assert row.baseline_only_final_entity_signature_count == 0
    assert row.candidate_only_final_entity_signature_count == 0
    assert row.baseline_stable_final_entity_signature_count == 2
    assert row.candidate_stable_final_entity_signature_count == 1
    assert row.stable_final_entity_signature_count_delta == -1
    assert row.baseline_stable_candidate_unstable_final_entity_signature_count == 1
    assert row.baseline_stable_candidate_unstable_final_entity_signature_label_counts == {"date": 1}
    assert row.safety_verdict == "fail"
    assert row.candidate_verdict == "reject"
    assert row.flags == ["stable_entity_signature_loss"]


def test_compare_case_analysis_does_not_infer_stability_loss_from_single_candidate_run() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_single_candidate_stability",
        REPO_ROOT / "tools/measurement/compare_strategy_pairs.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "legal-5",
                "config_id": "default",
                "case_id": "default-r0",
                "artifact_final_entity_signature_hashes": ["a", "b"],
                "artifact_final_entity_signature_labels": {"a": "person", "b": "date"},
            },
            {
                "workload_id": "legal-5",
                "config_id": "default",
                "case_id": "default-r1",
                "artifact_final_entity_signature_hashes": ["a", "b"],
                "artifact_final_entity_signature_labels": {"a": "person", "b": "date"},
            },
            {
                "workload_id": "legal-5",
                "config_id": "candidate",
                "case_id": "candidate-r0",
                "artifact_final_entity_signature_hashes": ["a", "b"],
                "artifact_final_entity_signature_labels": {"a": "person", "b": "date"},
            },
        ]
    )

    rows = tool.compare_case_analysis(table, baseline_config="default", candidate_config="candidate")

    assert len(rows) == 1
    row = rows[0]
    assert row.baseline_only_final_entity_signature_count == 0
    assert row.candidate_only_final_entity_signature_count == 0
    assert row.baseline_stable_final_entity_signature_count is None
    assert row.candidate_stable_final_entity_signature_count is None
    assert row.stable_final_entity_signature_count_delta is None
    assert row.baseline_stable_candidate_unstable_final_entity_signature_count is None
    assert row.safety_verdict == "pass"
    assert "stable_entity_signature_loss" not in row.flags


def test_compare_strategy_pairs_writes_csv(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_export", REPO_ROOT / "tools/measurement/compare_strategy_pairs.py"
    )
    rows = [
        tool.ComparisonRow(
            workload_id="shell-1",
            baseline_config_id="base",
            candidate_config_id="candidate",
            baseline_replacement_strategy="default",
            candidate_replacement_strategy="local_structured_substitute",
            baseline_case_count=1,
            candidate_case_count=1,
            value_protection_verdict="review",
            signature_parity_verdict="pass",
            safety_verdict="review",
            performance_verdict="improved",
            candidate_verdict="review",
            baseline_final_entity_count=4,
            candidate_final_entity_count=8,
            final_entity_count_delta=4,
            flags=["candidate_skips_llm_validation"],
        )
    ]

    output = tmp_path / "comparison.csv"
    tool.write_comparisons(rows, output, tool.ExportFormat.csv)

    exported = pd.read_csv(output)
    assert exported["workload_id"].tolist() == ["shell-1"]
    assert exported["candidate_replacement_strategy"].tolist() == ["local_structured_substitute"]
    assert exported["final_entity_count_delta"].tolist() == [4]
    assert exported["flags"].tolist() == ['["candidate_skips_llm_validation"]']


def test_compare_strategy_pairs_summarizes_candidate_verdicts() -> None:
    tool = load_tool(
        "measurement_compare_strategy_pairs_summary", REPO_ROOT / "tools/measurement/compare_strategy_pairs.py"
    )
    rows = [
        tool.ComparisonRow(
            workload_id="legal-1",
            baseline_config_id="legal-default",
            candidate_config_id="legal-no-augment",
            baseline_case_count=1,
            candidate_case_count=1,
            value_protection_verdict="pass",
            signature_parity_verdict="pass",
            safety_verdict="pass",
            performance_verdict="improved",
            candidate_verdict="candidate_viable",
        ),
        tool.ComparisonRow(
            workload_id="bio-1",
            baseline_config_id="bio-default",
            candidate_config_id="bio-no-augment",
            baseline_case_count=1,
            candidate_case_count=1,
            value_protection_verdict="fail",
            signature_parity_verdict="fail",
            safety_verdict="fail",
            performance_verdict="improved",
            candidate_verdict="reject",
            baseline_only_final_entity_signature_label_counts={"first_name": 2},
        ),
        tool.ComparisonRow(
            workload_id="shell-1",
            baseline_config_id="shell-default",
            candidate_config_id="shell-detector-only",
            baseline_case_count=1,
            candidate_case_count=1,
            value_protection_verdict="review",
            signature_parity_verdict="pass",
            safety_verdict="review",
            performance_verdict="improved",
            candidate_verdict="review",
        ),
    ]

    summary = tool.summarize_comparisons(rows)

    assert summary.comparison_count == 3
    assert summary.value_protection_verdict_counts == {"fail": 1, "pass": 1, "review": 1}
    assert summary.signature_parity_verdict_counts == {"fail": 1, "pass": 2, "review": 0}
    assert summary.safety_verdict_counts == {"fail": 1, "pass": 1, "review": 1}
    assert summary.performance_verdict_counts["improved"] == 3
    assert summary.candidate_verdict_counts == {"candidate_viable": 1, "reject": 1, "review": 1}
    assert summary.candidate_viable_workloads == ["legal-1"]
    assert summary.review_workloads == ["shell-1"]
    assert summary.rejected_workloads == ["bio-1"]

    rendered = tool.render_result(
        tool.ComparisonResult(
            input_path="case_analysis.csv",
            baseline_selector="strategy:default",
            candidate_selector="strategy:no_augment",
            summary=summary,
            comparisons=rows,
        ),
        json_output=False,
    )
    assert "Compared 3 workload(s): viable=1, review=1, reject=1" in rendered
    assert (
        "- bio-1: verdict=reject (safety=fail, value_protection=fail, signature_parity=fail, performance=improved)"
    ) in rendered
    assert "elapsed unknown->unknown" in rendered
    assert "lost_labels=first_name:2" in rendered
