# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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


class FakeClient:
    def complete(self, request):  # type: ignore[no-untyped-def]
        module = sys.modules["measurement_direct_detection_probe_case"]
        return module.DirectCompletion(
            content='{"entities": [{"value": "Alice", "label": "first_name", "reason": "name"}]}',
            elapsed_sec=1.25,
            usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        )


def test_direct_detection_case_uses_direct_client_and_canonical_artifacts() -> None:
    tool = load_tool(
        "measurement_direct_detection_probe_case",
        REPO_ROOT / "tools/measurement/direct_detection_probe.py",
    )

    result = tool.run_direct_detection_case(
        tool.DirectDetectionRequest(
            case_id="case-1",
            text="Alice met Alice.",
            labels=["first_name"],
            row_index=0,
            prompt_mode=tool.PromptMode.compact,
        ),
        client=FakeClient(),
    )

    assert result.status == tool.CaseStatus.completed
    assert result.elapsed_sec == 1.25
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
    assert result.allowed_suggestion_count == 1
    assert result.final_entity_count == 2
    assert result.final_entity_signature_count == 2
    assert result.final_label_counts == {"first_name": 2}
    assert result.artifact.final_source_counts == {"direct_llm": 2}


def test_finalize_suggestions_filters_labels_and_deduplicates_signature_hashes() -> None:
    tool = load_tool(
        "measurement_direct_detection_probe_finalize",
        REPO_ROOT / "tools/measurement/direct_detection_probe.py",
    )

    artifact = tool.finalize_direct_suggestions(
        text="Alice works at NVIDIA.",
        suggestions=[
            {"value": "Alice", "label": "first_name"},
            {"value": "NVIDIA", "label": "organization_name"},
            {"value": "works", "label": "unsupported"},
        ],
        labels=["first_name", "organization_name"],
        row_index=0,
        workflow_name="direct-detection",
    )

    assert artifact.final_entity_count == 2
    assert artifact.final_label_counts == {"first_name": 1, "organization_name": 1}
    assert artifact.weak_api_key_shape_count == 0
    assert set(artifact.final_entity_signature_labels.values()) == {"first_name", "organization_name"}


def test_signature_comparison_counts_shared_baseline_only_and_direct_only_labels() -> None:
    tool = load_tool(
        "measurement_direct_detection_probe_comparison",
        REPO_ROOT / "tools/measurement/direct_detection_probe.py",
    )

    comparison = tool.compare_signature_sets(
        baseline_hashes={"shared", "baseline-only"},
        baseline_labels={"shared": "person", "baseline-only": "date"},
        direct_hashes={"shared", "direct-only"},
        direct_labels={"shared": "person", "direct-only": "city"},
    )

    assert comparison.shared_final_entity_signature_count == 1
    assert comparison.baseline_only_final_entity_signature_count == 1
    assert comparison.direct_only_final_entity_signature_count == 1
    assert comparison.baseline_only_label_counts == {"date": 1}
    assert comparison.direct_only_label_counts == {"city": 1}


def test_baseline_comparison_skips_rows_without_signature_hashes() -> None:
    tool = load_tool(
        "measurement_direct_detection_probe_missing_signatures",
        REPO_ROOT / "tools/measurement/direct_detection_probe.py",
    )

    class LocalFakeClient:
        def complete(self, request):  # type: ignore[no-untyped-def]
            return tool.DirectCompletion(
                content='{"entities": [{"value": "Alice", "label": "first_name", "reason": "name"}]}',
                elapsed_sec=1.25,
                usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            )

    case = tool.run_direct_detection_case(
        tool.DirectDetectionRequest(
            case_id="case-1",
            text="Alice met Alice.",
            labels=["first_name"],
            row_index=0,
            prompt_mode=tool.PromptMode.compact,
        ),
        client=LocalFakeClient(),
    )

    compared = tool._case_with_comparison(case, {"row_index": 0, "final_entity_count": 2})

    assert compared.comparison is None


def test_baseline_artifact_reader_rejects_duplicate_row_indexes(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_direct_detection_probe_duplicate_baseline",
        REPO_ROOT / "tools/measurement/direct_detection_probe.py",
    )
    baseline_path = tmp_path / "baseline.jsonl"
    baseline_path.write_text('{"row_index": 0}\n{"row_index": 0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="multiple rows for row_index=0"):
        tool._read_baseline_artifacts(baseline_path)
