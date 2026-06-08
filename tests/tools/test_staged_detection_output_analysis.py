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


def test_analyze_staged_detection_output_summarizes_native_detection_probe(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_staged_detection_output_analysis",
        REPO_ROOT / "tools/measurement/analyze_staged_detection_output.py",
    )
    output_dir = tmp_path / "staged"
    output_dir.mkdir()
    _write_jsonl(
        output_dir / "staged-detection-cases.jsonl",
        [
            {
                "record_type": "staged_detection_case",
                "case_id": "shell-row-0",
                "row_index": 0,
                "seed_source": "gliner",
                "status": "completed",
                "elapsed_sec": 0.002,
                "model_elapsed_sec": 0.0,
                "model_phase_count": 0,
                "model_request_count": 0,
                "final_entity_count": 5,
                "final_entity_signature_count": 5,
                "final_label_counts": {"api_key": 2, "email": 1, "password": 1, "url": 1},
                "total_usage": {},
                "comparison": {
                    "baseline_final_entity_signature_count": 5,
                    "shared_final_entity_signature_count": 5,
                    "baseline_only_final_entity_signature_count": 0,
                    "direct_only_final_entity_signature_count": 0,
                    "baseline_only_label_counts": {},
                    "direct_only_label_counts": {},
                },
            },
            {
                "record_type": "staged_detection_case",
                "case_id": "bio-row-0",
                "row_index": 0,
                "seed_source": "direct_llm",
                "status": "completed",
                "elapsed_sec": 10.0,
                "model_elapsed_sec": 9.5,
                "model_phase_count": 3,
                "model_request_count": 3,
                "final_entity_count": 3,
                "final_entity_signature_count": 3,
                "final_label_counts": {"person": 2, "api_key": 1},
                "total_usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                "comparison": {
                    "baseline_final_entity_signature_count": 4,
                    "shared_final_entity_signature_count": 2,
                    "baseline_only_final_entity_signature_count": 2,
                    "direct_only_final_entity_signature_count": 1,
                    "baseline_only_label_counts": {"city": 1, "person": 1},
                    "direct_only_label_counts": {"api_key": 1},
                },
            },
            {
                "record_type": "staged_detection_case",
                "case_id": "bio-row-1",
                "row_index": 1,
                "seed_source": "direct_llm",
                "status": "error",
                "elapsed_sec": 1.0,
                "model_elapsed_sec": 0.8,
                "model_phase_count": 1,
                "model_request_count": 1,
                "final_entity_count": 0,
                "final_entity_signature_count": 0,
                "total_usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                "comparison": None,
                "error": "provider failed",
            },
        ],
    )

    result = tool.analyze_staged_detection_output(output_dir)

    assert result.case_count == 3
    assert result.group_count == 2
    groups = {row.seed_source: row for row in result.groups}
    assert groups["gliner"].case_count == 1
    assert groups["gliner"].completed_case_count == 1
    assert groups["gliner"].model_elapsed_sec_sum == 0.0
    assert groups["gliner"].model_request_count_sum == 0
    assert groups["gliner"].baseline_shared_signature_rate == 1.0
    assert groups["direct_llm"].case_count == 2
    assert groups["direct_llm"].completed_case_count == 1
    assert groups["direct_llm"].error_case_count == 1
    assert groups["direct_llm"].elapsed_sec_sum == pytest.approx(11.0)
    assert groups["direct_llm"].model_elapsed_sec_sum == pytest.approx(10.3)
    assert groups["direct_llm"].model_request_count_sum == 4
    assert groups["direct_llm"].total_tokens_sum == 132
    assert groups["direct_llm"].baseline_final_entity_signature_count_sum == 4
    assert groups["direct_llm"].shared_final_entity_signature_count_sum == 2
    assert groups["direct_llm"].baseline_only_final_entity_signature_count_sum == 2
    assert groups["direct_llm"].direct_only_final_entity_signature_count_sum == 1
    assert groups["direct_llm"].baseline_shared_signature_rate == pytest.approx(0.5)

    label_deltas = {(row.seed_source, row.delta_type, row.label): row.count for row in result.label_deltas}
    assert label_deltas == {
        ("direct_llm", "baseline_only", "city"): 1,
        ("direct_llm", "baseline_only", "person"): 1,
        ("direct_llm", "direct_only", "api_key"): 1,
    }


def test_staged_detection_output_analysis_writes_csv_tables(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_staged_detection_output_analysis_export",
        REPO_ROOT / "tools/measurement/analyze_staged_detection_output.py",
    )
    cases_path = tmp_path / "cases.jsonl"
    _write_jsonl(
        cases_path,
        [
            {
                "case_id": "case-0",
                "row_index": 0,
                "seed_source": "gliner",
                "status": "completed",
                "elapsed_sec": 0.01,
                "model_elapsed_sec": 0.0,
                "model_request_count": 0,
                "total_usage": {},
            }
        ],
    )

    result = tool.analyze_staged_detection_output(cases_path)
    export = tool.write_analysis_tables(result, tmp_path / "analysis", tool.ExportFormat.csv)

    assert Path(export.manifest_path).exists()
    assert [table.table for table in export.tables] == [
        "case_analysis",
        "group_analysis",
        "label_delta_analysis",
    ]
    case_table = pd.read_csv(tmp_path / "analysis" / "case_analysis.csv")
    assert case_table.loc[0, "case_id"] == "case-0"
    assert case_table.loc[0, "model_request_count"] == 0


def test_staged_detection_output_analysis_rejects_missing_case_file(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_staged_detection_output_analysis_missing_input",
        REPO_ROOT / "tools/measurement/analyze_staged_detection_output.py",
    )

    with pytest.raises(ValueError, match="input path does not exist"):
        tool.analyze_staged_detection_output(tmp_path / "missing")
