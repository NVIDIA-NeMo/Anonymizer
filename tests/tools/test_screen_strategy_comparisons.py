# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_tool(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_screen_strategy_comparisons_reads_comparison_csvs_only(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    pd.DataFrame(
        [
            {
                "workload_id": "shell-3",
                "baseline_config_id": "default",
                "candidate_config_id": "detector-only",
                "baseline_strategy": "default",
                "candidate_strategy": "detector_only",
                "baseline_replacement_strategy": "default",
                "candidate_replacement_strategy": "custom_replacement_strategy",
                "baseline_case_count": 3,
                "candidate_case_count": 3,
                "safety_verdict": "review",
                "performance_verdict": "improved",
                "candidate_verdict": "review",
                "pipeline_elapsed_sec_delta_pct": -99.0,
                "observed_total_requests_delta": -12,
                "observed_total_tokens_delta": -11000,
                "augmented_entity_count_delta": -3,
                "augmented_new_final_value_count_delta": 0,
                "baseline_only_final_entity_signature_count": 0,
                "baseline_stable_candidate_unstable_final_entity_signature_count": 0,
                "baseline_stable_final_entity_signature_count": 12,
                "candidate_stable_final_entity_signature_count": 12,
                "shared_stable_final_entity_signature_count": 12,
                "flags": '["candidate_skips_llm_validation"]',
            },
            {
                "workload_id": "legal-2",
                "baseline_config_id": "default",
                "candidate_config_id": "no-augment",
                "baseline_strategy": "default",
                "candidate_strategy": "no_augment",
                "baseline_replacement_strategy": "default",
                "candidate_replacement_strategy": "default",
                "safety_verdict": "fail",
                "performance_verdict": "improved",
                "candidate_verdict": "reject",
                "pipeline_elapsed_sec_delta_pct": -10.0,
                "observed_total_requests_delta": -2,
                "observed_total_tokens_delta": -10000,
                "augmented_entity_count_delta": -5.5,
                "augmented_new_final_value_count_delta": -2,
                "baseline_only_final_entity_signature_count": 2,
                "baseline_stable_candidate_unstable_final_entity_signature_count": 1,
                "baseline_only_final_entity_signature_label_counts.first_name": 2,
                "baseline_stable_candidate_unstable_final_entity_signature_label_counts.date": 1,
                "flags": '["entity_signature_loss", "stable_entity_signature_loss"]',
            },
        ]
    ).to_csv(analysis_dir / "default-vs-candidates.csv", index=False)
    pd.DataFrame([{"workload_id": "shell-3", "config_id": "default"}]).to_csv(
        analysis_dir / "group_analysis.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "source_path": "analysis/default-vs-candidates.csv",
                "workload_id": "shell-3",
                "baseline_config_id": "default",
                "candidate_config_id": "detector-only",
                "safety_verdict": "review",
                "performance_verdict": "improved",
                "candidate_verdict": "review",
            }
        ]
    ).to_csv(analysis_dir / "strategy-screen.csv", index=False)
    (analysis_dir / "empty.csv").write_text("", encoding="utf-8")

    result = tool.screen_comparison_paths([analysis_dir])

    assert result.scanned_file_count == 4
    assert result.comparison_file_count == 1
    assert result.row_count == 2
    assert result.summary.candidate_verdict_counts == {"reject": 1, "review": 1}
    assert result.summary.review_count == 1
    assert result.summary.reject_count == 1
    assert result.summary.viable_count == 0
    legal = next(row for row in result.rows if row.workload_id == "legal-2")
    assert legal.workload_family == "legal"
    assert legal.flags == ["entity_signature_loss", "stable_entity_signature_loss"]
    assert legal.baseline_only_label_counts == {"first_name": 2}
    assert legal.stable_lost_label_counts == {"date": 1}
    assert legal.augmented_new_final_value_count_delta == -2
    shell = next(row for row in result.rows if row.workload_id == "shell-3")
    assert shell.baseline_replacement_strategy == "default"
    assert shell.candidate_replacement_strategy == "custom_replacement_strategy"
    assert shell.baseline_case_count == 3
    assert shell.candidate_case_count == 3
    assert shell.shared_stable_final_entity_signature_count == 12
    detector_local = next(
        group
        for group in result.groups
        if group.group_key == "strategy:detector_only|replacement:custom_replacement_strategy"
    )
    assert detector_local.candidate_replacement_strategy == "custom_replacement_strategy"
    assert detector_local.row_count == 1
    no_augment = next(group for group in result.groups if group.group_key == "strategy:no_augment")
    assert no_augment.row_count == 1
    assert no_augment.reject_count == 1
    assert no_augment.has_conflicting_verdicts is False


def test_screen_strategy_comparisons_writes_csv(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_export",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    rows = [
        tool.ScreenRow(
            source_path="analysis/default-vs-detector-only.csv",
            workload_id="shell",
            baseline_config_id="default",
            candidate_config_id="detector-only",
            baseline_replacement_strategy="default",
            candidate_replacement_strategy="custom_replacement_strategy",
            safety_verdict="review",
            performance_verdict="improved",
            candidate_verdict="review",
            flags=["candidate_skips_llm_validation"],
        )
    ]

    output = tmp_path / "screen.csv"
    tool.write_rows(rows, output, tool.ExportFormat.csv)

    exported = pd.read_csv(output)
    assert exported["workload_id"].tolist() == ["shell"]
    assert exported["candidate_replacement_strategy"].tolist() == ["custom_replacement_strategy"]
    assert exported["flags"].tolist() == ['["candidate_skips_llm_validation"]']


def test_screen_strategy_comparisons_dedupes_exact_rows(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_dedupe",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    comparison = pd.DataFrame(
        [
            {
                "workload_id": "bio",
                "baseline_config_id": "default",
                "candidate_config_id": "candidate",
                "safety_verdict": "pass",
                "performance_verdict": "improved",
                "candidate_verdict": "candidate_viable",
                "pipeline_elapsed_sec_delta_pct": -5.0,
            }
        ]
    )
    comparison.to_csv(tmp_path / "a.csv", index=False)
    comparison.to_csv(tmp_path / "b.csv", index=False)

    result = tool.screen_comparison_paths([tmp_path])

    assert result.scanned_file_count == 2
    assert result.comparison_file_count == 2
    assert result.row_count == 1
    assert result.duplicate_row_count == 1
    assert result.summary.viable_count == 1


def test_screen_strategy_comparisons_surfaces_evidence_level_counts(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_evidence_level",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    pd.DataFrame(
        [
            {
                "workload_id": "structured-identifiers",
                "baseline_config_id": "default",
                "candidate_config_id": "local-substitute",
                "baseline_replacement_strategy": "default",
                "candidate_replacement_strategy": "custom_replacement_strategy",
                "value_protection_verdict": "pass",
                "signature_parity_verdict": "review",
                "safety_verdict": "review",
                "performance_verdict": "improved",
                "candidate_verdict": "review",
            },
            {
                "workload_id": "structured-identifiers",
                "baseline_config_id": "default",
                "candidate_config_id": "local-substitute-legacy",
                "baseline_replacement_strategy": "default",
                "candidate_replacement_strategy": "custom_replacement_strategy",
                "safety_verdict": "pass",
                "performance_verdict": "improved",
                "candidate_verdict": "candidate_viable",
                "shared_stable_final_entity_signature_count": 17,
            },
        ]
    ).to_csv(tmp_path / "comparison.csv", index=False)

    result = tool.screen_comparison_paths([tmp_path])

    assert result.summary.evidence_level_counts == {"split_verdicts": 1, "stable_signatures": 1}
    assert {row.evidence_level for row in result.rows} == {"split_verdicts", "stable_signatures"}
    group = result.groups[0]
    assert group.evidence_level_counts == {"split_verdicts": 1, "stable_signatures": 1}
    assert group.split_verdict_candidate_verdict_counts == {"review": 1}
    rendered = tool.render_result(result, json_output=False, limit=10)
    assert "evidence=split_verdicts" in rendered
    assert "evidence_counts=split_verdicts:1,stable_signatures:1" in rendered


def test_screen_strategy_comparisons_surfaces_candidate_original_value_leaks(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_original_value_leaks",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    pd.DataFrame(
        [
            {
                "workload_id": "structured-secrets",
                "baseline_config_id": "default",
                "candidate_config_id": "native-single-pass",
                "candidate_strategy": "native_single_pass",
                "safety_verdict": "fail",
                "performance_verdict": "improved",
                "candidate_verdict": "reject",
                "candidate_original_value_leak_count": 2,
                "candidate_original_value_leak_record_count": 1,
                "original_value_leak_count_delta": 2,
                "candidate_original_value_leak_label_counts.api_key": 1,
                "candidate_original_value_leak_label_counts.password": 1,
                "candidate_replacement_synthetic_original_collision_count": 1,
                "candidate_replacement_synthetic_original_collision_value_count": 1,
                "replacement_synthetic_original_collision_count_delta": 1,
                "candidate_replacement_synthetic_original_collision_label_counts.date": 1,
                "flags": ('["candidate_original_value_leak", "candidate_replacement_synthetic_original_collision"]'),
            }
        ]
    ).to_csv(tmp_path / "comparison.csv", index=False)

    result = tool.screen_comparison_paths([tmp_path])

    assert result.summary.reject_count == 1
    row = result.rows[0]
    assert row.candidate_original_value_leak_count == 2
    assert row.candidate_original_value_leak_record_count == 1
    assert row.candidate_original_value_leak_label_counts == {"api_key": 1, "password": 1}
    assert row.candidate_replacement_synthetic_original_collision_count == 1
    assert row.candidate_replacement_synthetic_original_collision_value_count == 1
    assert row.candidate_replacement_synthetic_original_collision_label_counts == {"date": 1}
    group = result.groups[0]
    assert group.sum_candidate_original_value_leak_count == 2
    assert group.sum_candidate_original_value_leak_record_count == 1
    assert group.candidate_original_value_leak_label_counts == {"api_key": 1, "password": 1}
    assert group.sum_candidate_replacement_synthetic_original_collision_count == 1
    assert group.sum_candidate_replacement_synthetic_original_collision_value_count == 1
    assert group.candidate_replacement_synthetic_original_collision_label_counts == {"date": 1}
    rendered = tool.render_result(result, json_output=False, limit=10)
    assert "candidate_original_value_leaks=2.0" in rendered
    assert "candidate_replacement_collisions=1.0" in rendered
    assert "leak_labels=api_key:1,password:1" in rendered
    assert "collision_labels=date:1" in rendered


def test_screen_strategy_comparisons_surfaces_label_policy_review(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_label_policy_review",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    pd.DataFrame(
        [
            {
                "workload_id": "legal-r1",
                "baseline_config_id": "legal-default",
                "candidate_config_id": "legal-detector-native-validate",
                "candidate_strategy": "detector_native_validate_no_augment",
                "value_protection_verdict": "pass",
                "signature_parity_verdict": "review",
                "safety_verdict": "review",
                "performance_verdict": "improved",
                "candidate_verdict": "review",
                "pipeline_elapsed_sec_delta_pct": -26.5,
                "observed_total_requests_delta": -1,
                "observed_total_tokens_delta": -5366,
                "flags": '["covered_label_mismatch"]',
                "baseline_only_candidate_label_mismatch_signature_label_counts.date_of_birth": 1,
            }
        ]
    ).to_csv(tmp_path / "comparison.csv", index=False)

    result = tool.screen_comparison_paths([tmp_path])

    assert result.summary.value_protection_verdict_counts == {"pass": 1}
    assert result.summary.signature_parity_verdict_counts == {"review": 1}
    row = result.rows[0]
    assert row.value_protection_verdict == "pass"
    assert row.signature_parity_verdict == "review"
    assert row.label_mismatch_label_counts == {"date_of_birth": 1}
    group = result.groups[0]
    assert group.value_protection_verdict_counts == {"pass": 1}
    assert group.signature_parity_verdict_counts == {"review": 1}
    assert group.label_mismatch_label_counts == {"date_of_birth": 1}
    assert group.recommendation == "label_policy_review"
    rendered = tool.render_result(result, json_output=False, limit=10)
    assert "value_protection=pass" in rendered
    assert "signature_parity=review" in rendered
    assert "label_mismatch=date_of_birth:1" in rendered


def test_screen_strategy_comparisons_surfaces_reliability_review(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_reliability_review",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    pd.DataFrame(
        [
            {
                "workload_id": "structured-identifiers",
                "baseline_config_id": "default-substitute",
                "candidate_config_id": "local-substitute",
                "baseline_replacement_strategy": "default",
                "candidate_replacement_strategy": "custom_replacement_strategy",
                "value_protection_verdict": "pass",
                "signature_parity_verdict": "pass",
                "safety_verdict": "review",
                "performance_verdict": "improved",
                "candidate_verdict": "review",
                "flags": '["failed_request_increase"]',
            }
        ]
    ).to_csv(tmp_path / "comparison.csv", index=False)

    result = tool.screen_comparison_paths([tmp_path])

    group = result.groups[0]
    assert group.flag_counts == {"failed_request_increase": 1}
    assert group.recommendation == "reliability_review"


def test_screen_strategy_comparisons_surfaces_replacement_replay_review(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_replacement_replay_review",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    pd.DataFrame(
        [
            {
                "workload_id": "structured-identifiers",
                "baseline_config_id": "default-substitute",
                "candidate_config_id": "local-substitute",
                "baseline_strategy": "default",
                "candidate_strategy": "default",
                "baseline_replacement_strategy": "default",
                "candidate_replacement_strategy": "custom_replacement_strategy",
                "value_protection_verdict": "pass",
                "signature_parity_verdict": "review",
                "safety_verdict": "review",
                "performance_verdict": "improved",
                "candidate_verdict": "review",
                "flags": '["covered_label_mismatch", "replacement_only_detection_instability"]',
                "baseline_only_candidate_label_mismatch_signature_label_counts.api_key": 1,
            }
        ]
    ).to_csv(tmp_path / "comparison.csv", index=False)

    result = tool.screen_comparison_paths([tmp_path])

    group = result.groups[0]
    assert group.flag_counts == {
        "covered_label_mismatch": 1,
        "replacement_only_detection_instability": 1,
    }
    assert group.recommendation == "replacement_replay_review"


def test_screen_strategy_comparisons_surfaces_baseline_defect_improvement_review(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_baseline_defect_improvement",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    pd.DataFrame(
        [
            {
                "workload_id": "biography-supported-structured",
                "baseline_config_id": "dd-substitute",
                "candidate_config_id": "local-substitute",
                "baseline_strategy": "default",
                "candidate_strategy": "default",
                "baseline_replacement_strategy": "default",
                "candidate_replacement_strategy": "custom_replacement_strategy",
                "value_protection_verdict": "pass",
                "signature_parity_verdict": "review",
                "safety_verdict": "review",
                "performance_verdict": "improved",
                "candidate_verdict": "review",
                "baseline_replacement_missing_final_entity_count": 2,
                "candidate_replacement_missing_final_entity_count": 0,
                "baseline_original_value_leak_count": 2,
                "candidate_original_value_leak_count": 0,
                "baseline_duplicate_synthetic_replacement_count": 1,
                "candidate_duplicate_synthetic_replacement_count": 0,
                "baseline_replacement_missing_final_entity_label_counts.organization_name": 1,
                "baseline_replacement_missing_final_entity_label_counts.religious_belief": 1,
                "baseline_original_value_leak_label_counts.organization_name": 2,
                "flags": (
                    '["baseline_replacement_missing_final_entity", "baseline_original_value_leak", '
                    '"candidate_covers_baseline_replacement_missing_final_entity", '
                    '"candidate_covers_baseline_original_value_leak", "replacement_count_delta"]'
                ),
            }
        ]
    ).to_csv(tmp_path / "comparison.csv", index=False)

    result = tool.screen_comparison_paths([tmp_path])

    row = result.rows[0]
    assert row.baseline_replacement_missing_final_entity_count == 2
    assert row.candidate_replacement_missing_final_entity_count == 0
    assert row.baseline_original_value_leak_count == 2
    assert row.candidate_original_value_leak_count == 0
    assert row.baseline_duplicate_synthetic_replacement_count == 1
    assert row.candidate_duplicate_synthetic_replacement_count == 0
    assert row.baseline_replacement_missing_final_entity_label_counts == {
        "organization_name": 1,
        "religious_belief": 1,
    }
    assert row.baseline_original_value_leak_label_counts == {"organization_name": 2}
    group = result.groups[0]
    assert group.baseline_defect_improvement_count == 1
    assert group.sum_baseline_replacement_missing_final_entity_count == 2
    assert group.sum_candidate_replacement_missing_final_entity_count == 0
    assert group.sum_baseline_original_value_leak_count == 2
    assert group.sum_candidate_original_value_leak_count == 0
    assert group.sum_baseline_duplicate_synthetic_replacement_count == 1
    assert group.sum_candidate_duplicate_synthetic_replacement_count == 0
    assert group.baseline_replacement_missing_final_entity_label_counts == {
        "organization_name": 1,
        "religious_belief": 1,
    }
    assert group.baseline_original_value_leak_label_counts == {"organization_name": 2}
    assert group.recommendation == "candidate_covers_baseline_defects"
    rendered = tool.render_result(result, json_output=False, limit=10)
    assert "baseline_defect_improvements=1" in rendered
    assert "baseline_missing_replacements=2.0" in rendered
    assert "candidate_missing_replacements=0.0" in rendered
    assert "baseline_missing_labels=organization_name:1,religious_belief:1" in rendered


def test_screen_strategy_comparisons_groups_default_detection_by_replacement_strategy() -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_replacement_group",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    row = tool.ScreenRow(
        source_path="comparison.csv",
        workload_id="structured-identifiers",
        baseline_config_id="substitute-dd",
        candidate_config_id="substitute-local",
        candidate_strategy="default",
        baseline_replacement_strategy="default",
        candidate_replacement_strategy="custom_replacement_strategy",
        safety_verdict="pass",
        performance_verdict="improved",
        candidate_verdict="candidate_viable",
    )

    assert tool.group_base_for_row(row, config_aliases={}) == "replacement:custom_replacement_strategy"


def test_screen_strategy_comparisons_keeps_generic_review_without_leak_metrics() -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_generic_review",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    group = tool.ScreenGroup(
        group_key="strategy:detector_only",
        candidate_strategy="detector_only",
        row_count=1,
        review_count=1,
        performance_verdict_counts={"improved": 1},
        flag_counts={"candidate_skips_llm_validation": 1},
    )

    assert tool.group_recommendation(group) == "review_only"


def test_screen_strategy_comparisons_filters_source_paths(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_source_filters",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    current_dir = tmp_path / "analysis-current-csv"
    current_dir.mkdir()
    stale_dir = tmp_path / "analysis"
    stale_dir.mkdir()
    pd.DataFrame(
        [
            {
                "workload_id": "bio-current",
                "baseline_config_id": "default",
                "candidate_config_id": "candidate",
                "safety_verdict": "pass",
                "performance_verdict": "improved",
                "candidate_verdict": "candidate_viable",
            }
        ]
    ).to_csv(current_dir / "current-comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "workload_id": "bio-stale",
                "baseline_config_id": "default",
                "candidate_config_id": "candidate",
                "safety_verdict": "fail",
                "performance_verdict": "regressed",
                "candidate_verdict": "reject",
            }
        ]
    ).to_csv(stale_dir / "stale-comparison.csv", index=False)

    result = tool.screen_comparison_paths([tmp_path], source_includes=["analysis-current-csv"])

    assert result.scanned_file_count == 1
    assert result.comparison_file_count == 1
    assert [row.workload_id for row in result.rows] == ["bio-current"]

    result = tool.screen_comparison_paths(
        [tmp_path],
        source_includes=["analysis"],
        source_excludes=["analysis-current-csv"],
    )

    assert result.scanned_file_count == 1
    assert result.comparison_file_count == 1
    assert [row.workload_id for row in result.rows] == ["bio-stale"]


def test_screen_strategy_comparisons_groups_candidate_strategy_conflicts(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_groups",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "legal-small",
                "baseline_config_id": "default",
                "candidate_config_id": "no-augment-small",
                "candidate_strategy": "no_augment",
                "safety_verdict": "pass",
                "performance_verdict": "improved",
                "candidate_verdict": "candidate_viable",
                "pipeline_elapsed_sec_delta_pct": -7.0,
                "observed_total_requests_delta": -2,
                "observed_total_tokens_delta": -8000,
                "baseline_case_count": 3,
                "candidate_case_count": 2,
                "shared_stable_final_entity_signature_count": 20,
                "flags": "[]",
            },
            {
                "workload_id": "legal-offset",
                "baseline_config_id": "default",
                "candidate_config_id": "no-augment-offset",
                "candidate_strategy": "no_augment",
                "safety_verdict": "fail",
                "performance_verdict": "improved",
                "candidate_verdict": "reject",
                "pipeline_elapsed_sec_delta_pct": -10.0,
                "observed_total_requests_delta": -1,
                "observed_total_tokens_delta": -10000,
                "baseline_case_count": 2,
                "candidate_case_count": 2,
                "shared_stable_final_entity_signature_count": 14,
                "baseline_only_final_entity_signature_label_counts.first_name": 2,
                "baseline_stable_candidate_unstable_final_entity_signature_label_counts.date": 1,
                "flags": '["entity_signature_loss", "stable_entity_signature_loss"]',
            },
            {
                "workload_id": "shell",
                "baseline_config_id": "default",
                "candidate_config_id": "detector-only",
                "candidate_strategy": "detector_only",
                "safety_verdict": "review",
                "performance_verdict": "improved",
                "candidate_verdict": "review",
                "pipeline_elapsed_sec_delta_pct": -99.9,
                "observed_total_tokens_delta": -11000,
                "flags": '["candidate_skips_llm_validation"]',
            },
        ]
    )
    table.to_csv(tmp_path / "comparison.csv", index=False)

    result = tool.screen_comparison_paths([tmp_path])

    groups = {group.group_key: group for group in result.groups}
    assert list(groups) == ["strategy:detector_only", "strategy:no_augment"]
    no_augment = groups["strategy:no_augment"]
    assert no_augment.row_count == 2
    assert no_augment.viable_count == 1
    assert no_augment.reject_count == 1
    assert no_augment.has_conflicting_verdicts is True
    assert no_augment.recommendation == "conflicting_evidence"
    assert no_augment.best_pipeline_elapsed_sec_delta_pct == -10.0
    assert no_augment.best_observed_total_tokens_delta == -10000
    assert no_augment.best_observed_total_requests_delta == -2
    assert no_augment.worst_pipeline_elapsed_sec_delta_pct == -7.0
    assert no_augment.worst_observed_total_tokens_delta == -8000
    assert no_augment.worst_observed_total_requests_delta == -1
    assert no_augment.min_baseline_case_count == 2
    assert no_augment.min_candidate_case_count == 2
    assert no_augment.min_shared_stable_final_entity_signature_count == 14
    assert no_augment.flag_counts == {"entity_signature_loss": 1, "stable_entity_signature_loss": 1}
    assert no_augment.baseline_only_label_counts == {"first_name": 2}
    assert no_augment.stable_lost_label_counts == {"date": 1}


def test_screen_strategy_comparisons_groups_default_strategy_by_config(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_default_config_groups",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "biographies-r5",
                "baseline_config_id": "biography-default",
                "candidate_config_id": "biography-augment-temp07",
                "candidate_strategy": "default",
                "safety_verdict": "pass",
                "performance_verdict": "improved",
                "candidate_verdict": "candidate_viable",
                "pipeline_elapsed_sec_delta_pct": -6.0,
                "observed_total_tokens_delta": -325,
                "flags": "[]",
            },
            {
                "workload_id": "biographies-r5-offset5",
                "baseline_config_id": "biography-default",
                "candidate_config_id": "biography-hybrid",
                "candidate_strategy": "default",
                "safety_verdict": "fail",
                "performance_verdict": "regressed",
                "candidate_verdict": "reject",
                "pipeline_elapsed_sec_delta_pct": 10.0,
                "observed_total_tokens_delta": 100,
                "baseline_only_final_entity_signature_label_counts.university": 1,
                "flags": '["entity_signature_loss"]',
            },
        ]
    )
    table.to_csv(tmp_path / "comparison.csv", index=False)

    result = tool.screen_comparison_paths([tmp_path], group_by=tool.GroupBy.strategy_workload_family)

    groups = {group.group_key: group for group in result.groups}
    assert list(groups) == [
        "config:biography-augment-temp07|family:biographies",
        "config:biography-hybrid|family:biographies",
    ]
    assert groups["config:biography-augment-temp07|family:biographies"].recommendation == "single_slice_viable"
    assert groups["config:biography-hybrid|family:biographies"].recommendation == "reject"


def test_screen_strategy_comparisons_groups_default_strategy_by_config_alias(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_config_alias_groups",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "biographies-r2",
                "baseline_config_id": "biography-default",
                "candidate_config_id": "biography-hybrid-augment-temp07",
                "candidate_strategy": "default",
                "safety_verdict": "pass",
                "performance_verdict": "improved",
                "candidate_verdict": "candidate_viable",
                "pipeline_elapsed_sec_delta_pct": -10.4,
                "baseline_case_count": 2,
                "candidate_case_count": 2,
                "shared_stable_final_entity_signature_count": 48,
                "flags": "[]",
            },
            {
                "workload_id": "biographies-r5-offset5",
                "baseline_config_id": "biography-default",
                "candidate_config_id": "biography-augment-temp07",
                "candidate_strategy": "default",
                "safety_verdict": "fail",
                "performance_verdict": "mixed",
                "candidate_verdict": "reject",
                "pipeline_elapsed_sec_delta_pct": 16.0,
                "baseline_case_count": 2,
                "candidate_case_count": 2,
                "shared_stable_final_entity_signature_count": 116,
                "baseline_stable_candidate_unstable_final_entity_signature_label_counts.university": 1,
                "flags": '["stable_entity_signature_loss"]',
            },
        ]
    )
    table.to_csv(tmp_path / "comparison.csv", index=False)

    result = tool.screen_comparison_paths(
        [tmp_path],
        group_by=tool.GroupBy.strategy_workload_family,
        config_aliases={
            "biography-hybrid-augment-temp07": "biography-temp07-routing",
            "biography-augment-temp07": "biography-temp07-routing",
        },
    )

    groups = {group.group_key: group for group in result.groups}
    group = groups["alias:biography-temp07-routing|family:biographies"]
    assert group.row_count == 2
    assert group.candidate_config_ids == ["biography-augment-temp07", "biography-hybrid-augment-temp07"]
    assert group.viable_count == 1
    assert group.reject_count == 1
    assert group.recommendation == "conflicting_evidence"
    assert group.min_baseline_case_count == 2
    assert group.min_candidate_case_count == 2
    assert group.min_shared_stable_final_entity_signature_count == 48
    assert group.stable_lost_label_counts == {"university": 1}


def test_screen_strategy_comparisons_can_group_by_strategy_and_workload_family(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_family_groups",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    table = pd.DataFrame(
        [
            {
                "workload_id": "shell-secrets-3",
                "baseline_config_id": "default",
                "candidate_config_id": "detector-only-shell",
                "candidate_strategy": "detector_only",
                "safety_verdict": "review",
                "performance_verdict": "improved",
                "candidate_verdict": "review",
                "pipeline_elapsed_sec_delta_pct": -99.9,
                "observed_total_tokens_delta": -11000,
                "flags": '["candidate_skips_llm_validation"]',
            },
            {
                "workload_id": "biographies-r5-offset5",
                "baseline_config_id": "default",
                "candidate_config_id": "detector-only-bio",
                "candidate_strategy": "detector_only",
                "safety_verdict": "fail",
                "performance_verdict": "improved",
                "candidate_verdict": "reject",
                "pipeline_elapsed_sec_delta_pct": -90.0,
                "observed_total_tokens_delta": -17000,
                "baseline_only_final_entity_signature_label_counts.first_name": 4,
                "flags": '["entity_signature_loss"]',
            },
        ]
    )
    table.to_csv(tmp_path / "comparison.csv", index=False)

    result = tool.screen_comparison_paths([tmp_path], group_by=tool.GroupBy.strategy_workload_family)

    groups = {group.group_key: group for group in result.groups}
    assert list(groups) == ["strategy:detector_only|family:shell-secrets", "strategy:detector_only|family:biographies"]
    assert groups["strategy:detector_only|family:shell-secrets"].recommendation == "review_only"
    assert groups["strategy:detector_only|family:shell-secrets"].workload_families == ["shell-secrets"]
    assert groups["strategy:detector_only|family:biographies"].recommendation == "reject"
    assert groups["strategy:detector_only|family:biographies"].baseline_only_label_counts == {"first_name": 4}


def test_workload_family_normalizes_slice_and_offset_suffixes() -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_workload_family",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )

    assert tool.workload_family("legal-r2-offset1") == "legal"
    assert tool.workload_family("legal-slice-2") == "legal"
    assert tool.workload_family("biographies-r5-offset5") == "biographies"
    assert tool.workload_family("shell-secrets-3") == "shell-secrets"
    assert tool.workload_family("support-ticket") == "support-ticket"


def test_screen_strategy_comparisons_writes_group_csv(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_group_export",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )
    groups = [
        tool.ScreenGroup(
            group_key="strategy:no_augment",
            candidate_strategy="no_augment",
            row_count=2,
            viable_count=1,
            reject_count=1,
            has_conflicting_verdicts=True,
            performance_verdict_counts={"improved": 1, "regressed": 1},
            flag_counts={"entity_signature_loss": 1},
        )
    ]

    output = tmp_path / "groups.csv"
    tool.write_groups(groups, output, tool.ExportFormat.csv)

    exported = pd.read_csv(output)
    assert exported["group_key"].tolist() == ["strategy:no_augment"]
    assert exported["has_conflicting_verdicts"].tolist() == [True]
    assert exported["recommendation"].tolist() == ["conflicting_evidence"]
    assert exported["performance_verdict_counts"].tolist() == ['{"improved": 1, "regressed": 1}']
    assert exported["flag_counts"].tolist() == ['{"entity_signature_loss": 1}']


def test_screen_strategy_group_recommendations() -> None:
    tool = load_tool(
        "measurement_screen_strategy_comparisons_recommendations",
        REPO_ROOT / "tools/measurement/screen_strategy_comparisons.py",
    )

    assert (
        tool.group_recommendation(
            tool.ScreenGroup(group_key="viable", row_count=1, viable_count=1),
        )
        == "single_slice_viable"
    )
    assert (
        tool.group_recommendation(
            tool.ScreenGroup(group_key="repeated-viable", row_count=2, viable_count=2),
        )
        == "candidate_family_viable"
    )
    assert (
        tool.group_recommendation(
            tool.ScreenGroup(group_key="promising", row_count=2, viable_count=1, review_count=1),
        )
        == "promising_needs_review"
    )
    assert (
        tool.group_recommendation(
            tool.ScreenGroup(
                group_key="weak-promising",
                row_count=2,
                viable_count=1,
                review_count=1,
                evidence_level_counts={"signature_counts": 1, "stable_signatures": 1},
            ),
        )
        == "needs_split_verdict_rerun"
    )
    assert (
        tool.group_recommendation(
            tool.ScreenGroup(
                group_key="partial-split-review",
                row_count=2,
                review_count=2,
                evidence_level_counts={"split_verdicts": 1, "stable_signatures": 1},
                split_verdict_candidate_verdict_counts={"review": 1},
            ),
        )
        == "needs_split_verdict_rerun"
    )
    assert (
        tool.group_recommendation(
            tool.ScreenGroup(
                group_key="split-review-with-legacy-viable",
                row_count=2,
                viable_count=1,
                review_count=1,
                evidence_level_counts={"signature_counts": 1, "split_verdicts": 1},
                split_verdict_candidate_verdict_counts={"review": 1},
            ),
        )
        == "needs_viable_split_verdict"
    )
    assert (
        tool.group_recommendation(
            tool.ScreenGroup(
                group_key="split-viable-with-review",
                row_count=2,
                viable_count=1,
                review_count=1,
                evidence_level_counts={"split_verdicts": 2},
                split_verdict_candidate_verdict_counts={"candidate_viable": 1, "review": 1},
            ),
        )
        == "promising_needs_review"
    )
    assert (
        tool.group_recommendation(
            tool.ScreenGroup(
                group_key="review",
                row_count=2,
                review_count=2,
                performance_verdict_counts={"improved": 2},
            ),
        )
        == "review_only"
    )
    assert (
        tool.group_recommendation(
            tool.ScreenGroup(
                group_key="review-mixed",
                row_count=2,
                review_count=2,
                performance_verdict_counts={"mixed": 2},
            ),
        )
        == "review_mixed_performance"
    )
    assert (
        tool.group_recommendation(
            tool.ScreenGroup(
                group_key="review-improved-and-mixed",
                row_count=2,
                review_count=2,
                performance_verdict_counts={"improved": 1, "mixed": 1},
            ),
        )
        == "review_mixed_performance"
    )
    assert (
        tool.group_recommendation(
            tool.ScreenGroup(
                group_key="review-regressed",
                row_count=2,
                review_count=2,
                performance_verdict_counts={"regressed": 2},
            ),
        )
        == "no_performance_win"
    )
    assert (
        tool.group_recommendation(
            tool.ScreenGroup(group_key="reject", row_count=2, reject_count=2),
        )
        == "reject"
    )
    assert (
        tool.group_recommendation(
            tool.ScreenGroup(group_key="conflict", row_count=2, viable_count=1, reject_count=1),
        )
        == "conflicting_evidence"
    )
