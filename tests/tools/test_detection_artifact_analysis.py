# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
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
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def _entity(value: str, label: str, start: int, end: int, *, source: str = "detector") -> dict[str, object]:
    return {
        "id": f"{label}_{start}_{end}",
        "value": value,
        "label": label,
        "start_position": start,
        "end_position": end,
        "score": 1.0,
        "source": source,
    }


def _write_artifact(root: Path, workflow: str, rows: list[dict[str, object]]) -> None:
    parquet_dir = root / workflow / "parquet-files"
    parquet_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(parquet_dir / "batch_00000.parquet", index=False)


def test_detection_artifact_analysis_reports_augmentation_contribution(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_detection_artifact_analysis",
        REPO_ROOT / "tools/measurement/analyze_detection_artifacts.py",
    )
    artifact_root = tmp_path / "artifacts"
    _write_artifact(
        artifact_root,
        "entity-detection",
        [
            {
                "_seed_entities_json": json.dumps([_entity("Alice", "first_name", 0, 5)]),
                "_seed_validation_candidates": json.dumps(
                    {"candidates": [{"id": "first_name_0_5", "value": "Alice", "label": "first_name"}]}
                ),
                "_augmented_entities": json.dumps(
                    {
                        "entities": [
                            {"value": "Alice", "label": "first_name", "reason": "duplicate"},
                            {"value": "12 February 1980", "label": "api_key", "reason": "date mislabeled"},
                        ]
                    }
                ),
                "_detected_entities": json.dumps(
                    {
                        "entities": [
                            _entity("Alice", "first_name", 0, 5),
                            _entity("12 February 1980", "api_key", 20, 36, source="augmenter"),
                        ]
                    }
                ),
                "_validation_candidates": json.dumps(
                    {
                        "candidates": [
                            {"id": "first_name_0_5", "value": "Alice", "label": "first_name"},
                            {"id": "api_key_20_36", "value": "12 February 1980", "label": "api_key"},
                        ]
                    }
                ),
            }
        ],
    )

    result = tool.analyze_artifacts(artifact_root)

    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.seed_entity_count == 1
    assert row.seed_validation_candidate_count == 1
    assert row.merged_validation_candidate_count == 2
    assert row.augmented_entity_count == 2
    assert row.augmented_duplicate_seed_value_count == 1
    assert row.augmented_new_value_count == 1
    assert row.augmented_new_final_value_count == 1
    assert row.final_entity_count == 2
    assert row.weak_api_key_shape_count == 1
    assert row.weak_api_key_shape_label_counts == {"api_key": 1}
    assert row.final_entity_signature_count == 2
    assert len(row.final_entity_signature_hashes) == 2
    assert set(row.final_entity_signature_labels) == set(row.final_entity_signature_hashes)
    assert sorted(row.final_entity_signature_labels.values()) == ["api_key", "first_name"]
    assert set(row.final_entity_signature_details) == set(row.final_entity_signature_hashes)
    first_name_detail = next(
        detail for detail in row.final_entity_signature_details.values() if detail["label"] == "first_name"
    )
    assert first_name_detail["source"] == "detector"
    assert first_name_detail["row_index"] == 0
    assert first_name_detail["start_position"] == 0
    assert first_name_detail["end_position"] == 5
    assert first_name_detail["value_length"] == 5
    assert "value_hash" not in first_name_detail

    serialized = row.model_dump_json()
    assert "Alice" not in serialized
    assert "12 February" not in serialized


def test_detection_artifact_analysis_handles_no_augment_rows(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_detection_artifact_analysis_no_augment",
        REPO_ROOT / "tools/measurement/analyze_detection_artifacts.py",
    )
    artifact_root = tmp_path / "artifacts"
    _write_artifact(
        artifact_root,
        "entity-detection-no-augment",
        [
            {
                "_seed_entities_json": json.dumps([_entity("Aydin", "city", 12, 17)]),
                "_seed_validation_candidates": json.dumps(
                    {"candidates": [{"id": "city_12_17", "value": "Aydin", "label": "city"}]}
                ),
                "_augmented_entities": json.dumps({"entities": []}),
                "_detected_entities": json.dumps({"entities": [_entity("Aydin", "city", 12, 17)]}),
            }
        ],
    )

    result = tool.analyze_artifacts(artifact_root)

    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.workflow_name == "entity-detection-no-augment"
    assert row.seed_entity_count == 1
    assert row.seed_validation_candidate_count == 1
    assert row.merged_validation_candidate_count == 0
    assert row.augmented_entity_count == 0
    assert row.augmented_new_value_count == 0
    assert row.augmented_new_final_value_count == 0
    assert row.final_entity_count == 1
    assert row.final_source_counts == {"detector": 1}
    assert row.final_entity_signature_count == 1
    assert row.final_entity_signature_hashes == sorted(row.final_entity_signature_hashes)
    assert list(row.final_entity_signature_labels.values()) == ["city"]
