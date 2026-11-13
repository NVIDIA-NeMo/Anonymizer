# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from anonymizer.engine import constants as cols
from anonymizer.engine.evaluation.privacy_findings import PRIVACY_FINDING_COLUMNS
from anonymizer.engine.schemas import EntityCategory, PrivacyAnswer, SensitivityLevel
from anonymizer.interface.results import AnonymizerResult, PreviewResult


def _result(trace: pd.DataFrame, preview: bool = False) -> AnonymizerResult | PreviewResult:
    kwargs: dict[str, Any] = {
        "dataframe": pd.DataFrame(),
        "trace_dataframe": trace,
        "resolved_text_column": "biography",
        "failed_records": [],
    }
    return PreviewResult(**kwargs, preview_num_records=len(trace)) if preview else AnonymizerResult(**kwargs)


def _row() -> dict[str, Any]:
    targets = [
        {
            "entity_value": "Alice",
            "entity_label": "first_name",
            "category": "direct_identifier",
            "sensitivity": "high",
            "reason": "Name sensitivity reason",
        },
        {
            "entity_value": "Catholic",
            "entity_label": "religious_belief",
            "category": "quasi_identifier",
            "sensitivity": "medium",
            "reason": "Religion sensitivity reason",
        },
    ]
    questions = [
        {
            "id": index,
            "question": f"Can {target['entity_value']} be deduced?",
            **{key: value for key, value in target.items() if key != "reason"},
        }
        for index, target in enumerate(targets, 1)
    ]
    return {
        cols.RECORD_ID_COLUMN: "record-a",
        cols.COL_ENTITY_SENSITIVITY: {"entities": targets},
        cols.COL_PRIVACY_QA: {"items": list(reversed(questions))},
        cols.COL_PRIVACY_QA_REANSWER: {
            "answers": [
                {"id": 2, "answer": "yes", "confidence": 0.8, "reason": "Attends Mass.", "evidence": ["attends Mass"]},
                {"id": 1, "answer": "no", "confidence": 1.0, "reason": "No original name.", "evidence": []},
            ]
        },
        cols.COL_PRIVACY_QA_ERROR: None,
    }


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("representation", ["native", "json", "numpy"])
def test_shared_accessor_preserves_assessments_and_joins_ids(preview: bool, representation: str) -> None:
    row = _row()
    for column, key in [
        (cols.COL_ENTITY_SENSITIVITY, "entities"),
        (cols.COL_PRIVACY_QA, "items"),
        (cols.COL_PRIVACY_QA_REANSWER, "answers"),
    ]:
        if representation == "json":
            row[column] = json.dumps(row[column])
        elif representation == "numpy":
            row[column][key] = np.array(row[column][key], dtype=object)
    table = _result(pd.DataFrame([row], index=pd.Index([99])), preview).privacy_findings
    assert table.columns.tolist() == list(PRIVACY_FINDING_COLUMNS)
    assert table[cols.COL_FINDING_RECORD_INDEX].tolist() == [0, 0]
    assert table[cols.COL_FINDING_RECORD_ID].tolist() == ["record-a", "record-a"]
    assert table[cols.COL_FINDING_ENTITY_VALUE].tolist() == ["Alice", "Catholic"]
    assert table[cols.COL_FINDING_ANSWER].tolist() == ["no", "yes"]
    assert table[cols.COL_FINDING_CONTRIBUTION].tolist() == [0.0, 0.48]
    assert table[cols.COL_FINDING_SENSITIVITY_REASON].tolist() == [
        "Name sensitivity reason",
        "Religion sensitivity reason",
    ]
    assert table[cols.COL_FINDING_STATUS].tolist() == ["evaluated", "evaluated"]


def test_accessor_is_fresh_and_evidence_edits_do_not_mutate_trace() -> None:
    row = _row()
    result = _result(pd.DataFrame([row]))
    first = result.privacy_findings
    first.at[1, cols.COL_FINDING_EVIDENCE].append("changed")
    assert result.privacy_findings.at[1, cols.COL_FINDING_EVIDENCE] == ["attends Mass"]
    row[cols.COL_PRIVACY_QA_REANSWER]["answers"][0]["reason"] = "Updated final reason"
    assert result.privacy_findings.at[1, cols.COL_FINDING_REASON] == "Updated final reason"


def test_numeric_looking_record_ids_remain_strings() -> None:
    row = _row()
    row[cols.RECORD_ID_COLUMN] = "12345"
    table = _result(pd.DataFrame([row])).privacy_findings
    assert table[cols.COL_FINDING_RECORD_ID].tolist() == ["12345", "12345"]


def test_schema_enums_are_exposed_as_plain_strings() -> None:
    row = _row()
    row[cols.COL_ENTITY_SENSITIVITY]["entities"][0]["category"] = EntityCategory.direct_identifier
    row[cols.COL_ENTITY_SENSITIVITY]["entities"][0]["sensitivity"] = SensitivityLevel.high
    row[cols.COL_PRIVACY_QA_REANSWER]["answers"][0]["answer"] = PrivacyAnswer.yes
    table = _result(pd.DataFrame([row])).privacy_findings
    for column in (cols.COL_FINDING_CATEGORY, cols.COL_FINDING_SENSITIVITY, cols.COL_FINDING_ANSWER):
        assert all(type(value) is str for value in table[column])


def test_rewrite_exemptions_and_latent_targets_use_final_answers() -> None:
    row = _row()
    del row[cols.COL_ENTITY_SENSITIVITY]
    row[cols.COL_SENSITIVITY_DISPOSITION] = {
        "sensitivity_disposition": [
            {
                "entity_value": "married",
                "entity_label": "marital_status",
                "category": "latent_identifier",
                "sensitivity": "medium",
                "protection_reason": "Spouse makes marriage inferable",
                "protection_method_suggestion": "suppress_inference",
            },
            {
                "entity_value": "USA",
                "entity_label": "country",
                "category": "quasi_identifier",
                "sensitivity": "low",
                "low_sensitivity_reason": "Broad geography",
                "protection_method_suggestion": "leave_as_is",
            },
        ]
    }
    row[cols.COL_PRIVACY_QA] = {
        "items": [
            {
                "id": 1,
                "question": "Can married be deduced?",
                "entity_value": "married",
                "entity_label": "marital_status",
                "category": "latent_identifier",
                "sensitivity": "medium",
            }
        ]
    }
    row[cols.COL_PRIVACY_QA_REANSWER] = {
        "answers": [
            {
                "id": 1,
                "answer": "no",
                "confidence": 0.9,
                "reason": "Final repaired text has no spouse reference.",
                "evidence": [],
            }
        ]
    }
    row[cols.COL_LEAKED_PRIVACY_ITEMS] = "Stale pre-repair finding: married leaked"
    table = _result(pd.DataFrame([row])).privacy_findings
    assert table[cols.COL_FINDING_STATUS].tolist() == ["evaluated", "not_evaluated"]
    assert table[cols.COL_FINDING_CATEGORY].tolist() == ["latent_identifier", "quasi_identifier"]
    assert table[cols.COL_FINDING_CONTRIBUTION].tolist() == [0.0, None]
    assert table[cols.COL_FINDING_ANSWER].tolist() == ["no", None]
    assert table[cols.COL_FINDING_SENSITIVITY_REASON].tolist() == ["Spouse makes marriage inferable", "Broad geography"]


def test_saved_contribution_and_question_metadata_survive_without_assessments() -> None:
    row = _row()
    del row[cols.COL_ENTITY_SENSITIVITY]
    row[cols.COL_PRIVACY_LEAKAGE_CONTRIBUTIONS] = [
        {"id": 2, "entity_value": "Catholic", "entity_label": "religious_belief", "leakage_contribution": 0.123}
    ]
    table = _result(pd.DataFrame([row])).privacy_findings
    assert table[cols.COL_FINDING_ENTITY_VALUE].tolist() == ["Catholic", "Alice"]
    assert table.at[0, cols.COL_FINDING_CONTRIBUTION] == 0.123
    assert table.at[0, cols.COL_FINDING_SENSITIVITY] == "medium"


@pytest.mark.parametrize("failure", ["error", "missing_answer", "null_confidence"])
def test_unavailable_answers_are_not_passes_or_zero_contributions(failure: str) -> None:
    row = _row()
    if failure == "error":
        row[cols.COL_PRIVACY_QA_ERROR] = "Model unavailable"
        row[cols.COL_PRIVACY_QA_REANSWER] = None
    elif failure == "missing_answer":
        row[cols.COL_PRIVACY_QA_REANSWER]["answers"] = []
    else:
        for answer in row[cols.COL_PRIVACY_QA_REANSWER]["answers"]:
            answer["confidence"] = None
    table = _result(pd.DataFrame([row])).privacy_findings
    assert table[cols.COL_FINDING_STATUS].tolist() == ["unavailable", "unavailable"]
    assert table[cols.COL_FINDING_CONTRIBUTION].tolist() == [None, None]
    assert all(table[cols.COL_FINDING_ERROR])


def test_failed_upstream_record_with_no_targets_remains_visible() -> None:
    table = _result(
        pd.DataFrame([{cols.RECORD_ID_COLUMN: "failed", cols.COL_PRIVACY_QA_ERROR: "Domain classification failed"}])
    ).privacy_findings
    assert len(table) == 1
    assert table.at[0, cols.COL_FINDING_RECORD_ID] == "failed"
    assert table.at[0, cols.COL_FINDING_STATUS] == "unavailable"
    assert pd.isna(table.at[0, cols.COL_FINDING_ENTITY_VALUE])


def test_unevaluated_detected_targets_remain_visible() -> None:
    row = {
        cols.RECORD_ID_COLUMN: "before-evaluation",
        cols.COL_ENTITIES_BY_VALUE: {"entities_by_value": [{"value": "Alice", "labels": ["first_name", "full_name"]}]},
    }
    table = _result(pd.DataFrame([row])).privacy_findings
    assert table[cols.COL_FINDING_STATUS].tolist() == ["not_evaluated", "not_evaluated"]
    assert table[cols.COL_FINDING_ENTITY_LABEL].tolist() == ["first_name", "full_name"]
    assert table[cols.COL_FINDING_CONTRIBUTION].tolist() == [None, None]


@pytest.mark.parametrize(
    "trace",
    [
        pd.DataFrame(),
        pd.DataFrame([{cols.COL_ENTITY_SENSITIVITY: {"entities": []}}]),
        pd.DataFrame([{cols.COL_ENTITIES_BY_VALUE: {"entities_by_value": []}}]),
    ],
)
def test_empty_findings_have_stable_columns(trace: pd.DataFrame) -> None:
    table = _result(trace).privacy_findings
    assert table.empty
    assert table.columns.tolist() == list(PRIVACY_FINDING_COLUMNS)


def test_duplicate_question_ids_are_unavailable_instead_of_misjoined() -> None:
    row = _row()
    row[cols.COL_PRIVACY_QA]["items"][0]["id"] = 1
    table = _result(pd.DataFrame([row])).privacy_findings
    assert table[cols.COL_FINDING_STATUS].tolist() == ["unavailable"]
    assert table.at[0, cols.COL_FINDING_CONTRIBUTION] is None
