# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

from anonymizer.engine.constants import COL_DETECTED_ENTITIES, COL_TEXT

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


def _entity(value: str, label: str, start: int, end: int, source: str = "detector") -> dict[str, object]:
    return {"value": value, "label": label, "start_position": start, "end_position": end, "source": source}


def _write_artifact_case(root: Path, tool: ModuleType, entities: list[dict[str, object]], text: str) -> Path:
    parquet_file = root / "entity-detection" / "parquet-files" / "batch_00000.parquet"
    parquet_file.parent.mkdir(parents=True)
    pd.DataFrame([{COL_TEXT: text, COL_DETECTED_ENTITIES: {"entities": entities}}]).to_parquet(parquet_file)
    artifact_path = root / "detection-artifacts.jsonl"
    row = tool.build_detection_artifact_row_from_entities(
        workflow_name="entity-detection",
        batch_file="entity-detection/parquet-files/batch_00000.parquet",
        row_index=0,
        seed_entities=[],
        seed_validation_candidate_count=0,
        merged_validation_candidate_count=0,
        augmented_entities=[],
        final_entities=[tool.EntitySchema.model_validate(entity) for entity in entities],
    ).model_dump()
    pd.json_normalize([{**_case_metadata(), **row}], sep=".").to_json(artifact_path, orient="records", lines=True)
    return artifact_path


def _write_seed_only_artifact_case(root: Path, tool: ModuleType, entities: list[dict[str, object]], text: str) -> Path:
    parquet_file = root / "entity-detection-seed" / "parquet-files" / "batch_00000.parquet"
    parquet_file.parent.mkdir(parents=True)
    pd.DataFrame([{COL_TEXT: text}]).to_parquet(parquet_file)
    artifact_path = root / "detection-artifacts.jsonl"
    row = tool.build_detection_artifact_row_from_entities(
        workflow_name="entity-detection-seed",
        batch_file="entity-detection-seed/parquet-files/batch_00000.parquet",
        row_index=0,
        seed_entities=[],
        seed_validation_candidate_count=0,
        merged_validation_candidate_count=0,
        augmented_entities=[],
        final_entities=[tool.EntitySchema.model_validate(entity) for entity in entities],
    ).model_dump()
    pd.json_normalize([{**_case_metadata(), **row}], sep=".").to_json(artifact_path, orient="records", lines=True)
    return artifact_path


def _case_metadata() -> dict[str, object]:
    return {
        "suite_id": "suite",
        "workload_id": "bio",
        "config_id": "config",
        "case_id": "bio__config__r000",
        "run_id": "bio__config__r000",
        "repetition": 0,
    }


def test_extract_signature_deltas_masks_candidate_only_context(tmp_path: Path) -> None:
    analyzer = load_tool(
        "measurement_detection_artifact_builder", REPO_ROOT / "tools/measurement/analyze_detection_artifacts.py"
    )
    tool = load_tool(
        "measurement_extract_signature_deltas", REPO_ROOT / "tools/measurement/extract_signature_deltas.py"
    )
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    baseline = _write_artifact_case(baseline_root, analyzer, [_entity("Alice", "person", 0, 5)], "Alice met NASA")
    candidate = _write_artifact_case(
        candidate_root,
        analyzer,
        [_entity("Alice", "person", 0, 5), _entity("NASA", "organization_name", 10, 14, "augmenter")],
        "Alice met NASA",
    )

    result = tool.extract_signature_deltas(
        baseline,
        candidate,
        baseline_artifact_root=baseline_root,
        candidate_artifact_root=candidate_root,
    )

    assert result.delta_count == 1
    row = result.rows[0]
    assert row.side == "candidate_only"
    assert row.label == "organization_name"
    assert row.source == "augmenter"
    assert row.resolution == "parquet"
    assert "NASA" not in row.masked_context
    assert "[ORGANIZATION_NAME:" in row.masked_context


def test_extract_signature_deltas_recovers_guardrail_rule_context(tmp_path: Path) -> None:
    analyzer = load_tool(
        "measurement_detection_artifact_rule_builder", REPO_ROOT / "tools/measurement/analyze_detection_artifacts.py"
    )
    tool = load_tool(
        "measurement_extract_signature_deltas_rule", REPO_ROOT / "tools/measurement/extract_signature_deltas.py"
    )
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    baseline = _write_artifact_case(baseline_root, analyzer, [], "The applicant was born in 1990.")
    candidate = _write_artifact_case(candidate_root, analyzer, [], "The applicant was born in 1990.")
    rule_entity = analyzer.EntitySchema(
        value="1990", label="date_of_birth", start_position=26, end_position=30, source="rule"
    )
    rule_row = analyzer.build_detection_artifact_row_from_entities(
        workflow_name="entity-detection",
        batch_file="entity-detection/parquet-files/batch_00000.parquet",
        row_index=0,
        seed_entities=[],
        seed_validation_candidate_count=0,
        merged_validation_candidate_count=0,
        augmented_entities=[],
        final_entities=[rule_entity],
    ).model_dump()
    pd.json_normalize([{**_case_metadata(), **rule_row}], sep=".").to_json(candidate, orient="records", lines=True)

    result = tool.extract_signature_deltas(
        baseline,
        candidate,
        baseline_artifact_root=baseline_root,
        candidate_artifact_root=candidate_root,
    )

    assert result.delta_count == 1
    row = result.rows[0]
    assert row.label == "date_of_birth"
    assert row.source == "rule"
    assert row.resolution == "rule"
    assert "1990" not in row.masked_context
    assert "[DATE_OF_BIRTH:" in row.masked_context


def test_extract_signature_deltas_uses_signature_details_when_final_parquet_is_unavailable(tmp_path: Path) -> None:
    analyzer = load_tool(
        "measurement_detection_artifact_detail_builder", REPO_ROOT / "tools/measurement/analyze_detection_artifacts.py"
    )
    tool = load_tool(
        "measurement_extract_signature_deltas_details", REPO_ROOT / "tools/measurement/extract_signature_deltas.py"
    )
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    baseline = _write_seed_only_artifact_case(baseline_root, analyzer, [], "Alice met NASA")
    candidate = _write_seed_only_artifact_case(
        candidate_root,
        analyzer,
        [_entity("NASA", "organization_name", 10, 14, "detector")],
        "Alice met NASA",
    )

    result = tool.extract_signature_deltas(
        baseline,
        candidate,
        baseline_artifact_root=baseline_root,
        candidate_artifact_root=candidate_root,
    )

    assert result.delta_count == 1
    row = result.rows[0]
    assert row.label == "organization_name"
    assert row.source == "detector"
    assert row.resolution == "artifact_details"
    assert row.start_position == 10
    assert row.end_position == 14
    assert "NASA" not in row.masked_context
    assert "[ORGANIZATION_NAME:" in row.masked_context
