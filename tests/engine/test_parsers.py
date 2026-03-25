# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import BaseModel

from anonymizer.engine.rewrite.parsers import (
    field,
    parse_privacy_answers,
    parse_privacy_qa,
    parse_quality_answers,
    parse_quality_compare,
    parse_quality_qa,
    parse_sensitivity_disposition,
    render_template,
)
from anonymizer.engine.schemas.rewrite import (
    PrivacyAnswersSchema,
    QACompareResultsSchema,
    QualityQAPairsSchema,
)

# ---------------------------------------------------------------------------
# render_template
# ---------------------------------------------------------------------------


def test_render_template_basic() -> None:
    assert render_template("Hello <<NAME>>!", {"<<NAME>>": "Alice"}) == "Hello Alice!"


def test_render_template_multiple_placeholders() -> None:
    assert render_template("<<A>> and <<B>>", {"<<A>>": "X", "<<B>>": "Y"}) == "X and Y"


def test_render_template_no_cross_contamination() -> None:
    result = render_template(
        "text: <<TEXT>>, score: <<SCORE>>",
        {"<<TEXT>>": "contains <<SCORE>> literally", "<<SCORE>>": "0.9"},
    )
    assert result == "text: contains <<SCORE>> literally, score: 0.9"


def test_render_template_unmatched_placeholder_left_as_is() -> None:
    assert render_template("<<A>> and <<B>>", {"<<A>>": "X"}) == "X and <<B>>"


def test_render_template_empty_replacements() -> None:
    assert render_template("no placeholders", {}) == "no placeholders"


# ---------------------------------------------------------------------------
# field
# ---------------------------------------------------------------------------


def test_field_valid() -> None:
    class MyModel(BaseModel):
        name: str

    assert field(MyModel, "name") == "name"


def test_field_invalid_raises() -> None:
    class MyModel(BaseModel):
        name: str

    with pytest.raises(KeyError, match="has no field 'missing'"):
        field(MyModel, "missing")


# ---------------------------------------------------------------------------
# parse_privacy_answers
# ---------------------------------------------------------------------------


def test_parse_privacy_answers_from_dict() -> None:
    result = parse_privacy_answers({"answers": [{"id": 1, "answer": "yes"}]})
    assert len(result) == 1
    assert result[0].id == 1


def test_parse_privacy_answers_from_schema() -> None:
    schema = PrivacyAnswersSchema.model_validate({"answers": [{"id": 1, "answer": "no"}]})
    assert len(parse_privacy_answers(schema)) == 1


def test_parse_privacy_answers_invalid_type() -> None:
    with pytest.raises(TypeError):
        parse_privacy_answers("bad")


# ---------------------------------------------------------------------------
# parse_quality_qa
# ---------------------------------------------------------------------------


def test_parse_quality_qa_from_dict() -> None:
    raw = {"items": [{"id": 1, "aspect": "role", "question": "What?", "reference_answer": "X"}]}
    assert len(parse_quality_qa(raw).items) == 1


def test_parse_quality_qa_from_schema() -> None:
    schema = QualityQAPairsSchema.model_validate(
        {"items": [{"id": 1, "aspect": "role", "question": "What?", "reference_answer": "X"}]}
    )
    assert len(parse_quality_qa(schema).items) == 1


def test_parse_quality_qa_invalid_type() -> None:
    with pytest.raises(TypeError):
        parse_quality_qa(42)


# ---------------------------------------------------------------------------
# parse_quality_answers
# ---------------------------------------------------------------------------


def test_parse_quality_answers_from_dict() -> None:
    assert len(parse_quality_answers({"answers": [{"id": 1, "answer": "yes"}]})) == 1


def test_parse_quality_answers_invalid_type() -> None:
    with pytest.raises(TypeError):
        parse_quality_answers([])


# ---------------------------------------------------------------------------
# parse_quality_compare
# ---------------------------------------------------------------------------


def test_parse_quality_compare_from_dict() -> None:
    ids, scores = parse_quality_compare({"per_item": [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.5}]})
    assert ids == [1, 2]
    assert scores == [0.9, 0.5]


def test_parse_quality_compare_from_schema() -> None:
    schema = QACompareResultsSchema.model_validate({"per_item": [{"id": 1, "score": 0.8}]})
    ids, scores = parse_quality_compare(schema)
    assert ids == [1]
    assert scores == [0.8]


def test_parse_quality_compare_invalid_type() -> None:
    with pytest.raises(TypeError):
        parse_quality_compare(None)


# ---------------------------------------------------------------------------
# parse_privacy_qa
# ---------------------------------------------------------------------------


def test_parse_privacy_qa_from_dict() -> None:
    raw = {
        "items": [
            {
                "id": 1,
                "question": "Q?",
                "sensitivity": "high",
                "entity_label": "name",
                "entity_value": "Alice",
                "category": "direct_identifier",
            }
        ]
    }
    assert len(parse_privacy_qa(raw).items) == 1


def test_parse_privacy_qa_invalid_type() -> None:
    with pytest.raises(TypeError):
        parse_privacy_qa(123)


# ---------------------------------------------------------------------------
# parse_sensitivity_disposition
# ---------------------------------------------------------------------------


def test_parse_sensitivity_disposition_from_dict() -> None:
    raw = {
        "sensitivity_disposition": [
            {
                "id": 1,
                "source": "tagged",
                "category": "direct_identifier",
                "sensitivity": "high",
                "entity_label": "name",
                "entity_value": "Alice",
                "needs_protection": True,
                "protection_reason": "Direct identifier that enables re-identification",
                "protection_method_suggestion": "replace",
                "combined_risk_level": "high",
            }
        ]
    }
    assert len(parse_sensitivity_disposition(raw).sensitivity_disposition) == 1


def test_parse_sensitivity_disposition_invalid_type() -> None:
    with pytest.raises(ValueError):
        parse_sensitivity_disposition("bad")
