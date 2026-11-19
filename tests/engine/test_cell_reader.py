# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pins for the nested-cell reader, the replacement-map reader and the column-schema dict."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel

from anonymizer.engine.evaluation.replace import (
    attribute_fidelity_judge,
    relational_consistency_judge,
    type_fidelity_judge,
)
from anonymizer.engine.replace.strategies import ReplacementEntry, _parse_replacements
from anonymizer.engine.rewrite import parsers
from anonymizer.engine.schemas import (
    EntityReplacementMapSchema,
    PrivacyAnswersSchema,
    PrivacyQAPairsSchema,
    QACompareResultsSchema,
    QualityAnswersSchema,
    QualityQAPairsSchema,
    SensitivityDispositionSchema,
)

_PAYLOADS: dict[type[BaseModel], dict[str, Any]] = {
    PrivacyAnswersSchema: {"answers": [{"id": 1, "answer": "no", "confidence": 0.5, "reason": "r", "evidence": []}]},
    QualityQAPairsSchema: {
        "items": [{"id": 1, "aspect": "a", "importance": "critical", "question": "q", "reference_answer": "x"}]
    },
    QualityAnswersSchema: {"answers": [{"id": 1, "answer": "a"}, {"id": 2, "answer": "b"}]},
    QACompareResultsSchema: {"per_item": [{"id": 1, "score": 0.25, "reason": None}, {"id": 2, "score": 1.0}]},
    PrivacyQAPairsSchema: {
        "items": [
            {
                "id": 1,
                "question": "q",
                "sensitivity": "high",
                "entity_label": "first_name",
                "entity_value": "Alice",
                "category": "direct_identifier",
            }
        ]
    },
    SensitivityDispositionSchema: {
        "sensitivity_disposition": [
            {
                "id": 1,
                "sensitivity": "high",
                "protection_method_suggestion": "replace",
                "source": "tagged",
                "category": "direct_identifier",
                "entity_label": "first_name",
                "entity_value": "Alice",
            }
        ]
    },
}

# (parser, schema, expected error type, how the parser projects the model to its return value)
_PARSERS: list[tuple[Callable[[Any], Any], type[BaseModel], type[Exception], Callable[[Any], Any]]] = [
    (parsers.parse_privacy_answers, PrivacyAnswersSchema, TypeError, lambda m: m.answers),
    (parsers.parse_quality_qa, QualityQAPairsSchema, TypeError, lambda m: m),
    (parsers.parse_quality_answers, QualityAnswersSchema, TypeError, lambda m: m.answers),
    (
        parsers.parse_quality_compare,
        QACompareResultsSchema,
        TypeError,
        lambda m: ([i.id for i in m.per_item], [i.score for i in m.per_item]),
    ),
    (parsers.parse_privacy_qa, PrivacyQAPairsSchema, TypeError, lambda m: m),
    (parsers.parse_sensitivity_disposition, SensitivityDispositionSchema, ValueError, lambda m: m),
]


def _parquet_reloaded(payload: dict[str, Any]) -> object:
    buffer = io.BytesIO()
    pd.DataFrame({"cell": [payload]}).to_parquet(buffer)
    buffer.seek(0)
    return pd.read_parquet(buffer)["cell"].iloc[0]


@pytest.mark.parametrize("parse,schema,error,project", _PARSERS, ids=lambda v: getattr(v, "__name__", ""))
@pytest.mark.parametrize("form", ["model", "dict", "json", "parquet"])
def test_parse_reads_every_cell_form(parse, schema, error, project, form) -> None:
    payload = _PAYLOADS[schema]
    expected = project(schema.model_validate(payload))
    raw = {
        "model": lambda: schema.model_validate(payload),
        "dict": lambda: payload,
        "json": lambda: json.dumps(payload),
        "parquet": lambda: _parquet_reloaded(payload),
    }[form]()
    result = parse(raw)
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize("parse,schema,error,project", _PARSERS, ids=lambda v: getattr(v, "__name__", ""))
@pytest.mark.parametrize("raw", [42, None, "not json", [1, 2]])
def test_parse_wrong_type_raises_its_error(parse, schema, error, project, raw) -> None:
    with pytest.raises(error) as excinfo:
        parse(raw)
    assert type(excinfo.value) is error


def test_parse_revalidates_a_model_instance() -> None:
    model = QualityQAPairsSchema.model_validate(_PAYLOADS[QualityQAPairsSchema])
    assert parsers.parse_quality_qa(model) is not model


# --- replacement-map reader, against the implementations it replaced -------------------------


def _base_parse_replacements(raw: object) -> list[tuple[str, str, str]]:
    parsed = raw
    model_dump = getattr(raw, "model_dump", None)
    if callable(model_dump):
        parsed = model_dump(mode="python")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, dict):
        return []
    replacements = parsed.get("replacements", [])
    if not isinstance(replacements, list):
        return []
    normalized = []
    for replacement in replacements:
        if not isinstance(replacement, dict):
            continue
        original = str(replacement.get("original", ""))
        label = str(replacement.get("label", ""))
        synthetic = str(replacement.get("synthetic", ""))
        if not original or not label or not synthetic:
            continue
        normalized.append((original, label, synthetic))
    return normalized


def _base_replacements_for_judge(raw_map: object) -> list[dict[str, str]]:
    if raw_map is None:
        return []
    if isinstance(raw_map, BaseModel):
        raw_map = raw_map.model_dump(mode="python")
    if isinstance(raw_map, str):
        try:
            raw_map = json.loads(raw_map)
        except (json.JSONDecodeError, ValueError):
            return []
    if not isinstance(raw_map, dict):
        return []
    try:
        parsed = EntityReplacementMapSchema.model_validate(raw_map)
    except Exception:
        return []
    return [{"original": r.original, "label": r.label, "synthetic": r.synthetic} for r in parsed.replacements]


_GOOD = {"original": "Alice", "label": "first_name", "synthetic": "Maya"}
_MAP_INPUTS: list[object] = [
    None,
    42,
    "not json",
    "[1, 2]",
    json.dumps({"replacements": [_GOOD]}),
    {"replacements": [_GOOD]},
    {"replacements": [_GOOD, {"original": "12345", "label": "postcode", "synthetic": 94110}]},
    {"replacements": [_GOOD, "not a dict"]},
    {"replacements": [_GOOD, {"original": "", "label": "first_name", "synthetic": "x"}]},
    {"replacements": [_GOOD, {"original": "Bob", "label": "first_name"}]},
    {"replacements": [_GOOD, {"original": "Bob", "label": "first_name", "synthetic": None}]},
    {"replacements": "not a list"},
    {"replacements": None},
    {"other": 1},
    {},
    EntityReplacementMapSchema.model_validate({"replacements": [_GOOD]}),
    _parquet_reloaded({"replacements": [_GOOD]}),
]


@pytest.mark.parametrize("raw", _MAP_INPUTS)
def test_from_raw_skip_matches_parse_replacements(raw: object) -> None:
    got = EntityReplacementMapSchema.from_raw(raw, on_invalid="skip")
    assert [(e.original, e.label, e.synthetic) for e in got] == _base_parse_replacements(raw)
    entries = _parse_replacements(raw)
    assert all(type(e) is ReplacementEntry for e in entries)
    assert [(e.original, e.label, e.synthetic) for e in entries] == _base_parse_replacements(raw)


@pytest.mark.parametrize("raw", _MAP_INPUTS)
def test_from_raw_drop_matches_judges(raw: object) -> None:
    expected = _base_replacements_for_judge(raw)
    got = EntityReplacementMapSchema.from_raw(raw, on_invalid="drop")
    assert [{"original": r.original, "label": r.label, "synthetic": r.synthetic} for r in got] == expected
    for judge in (attribute_fidelity_judge, relational_consistency_judge, type_fidelity_judge):
        assert judge._replacements_for_judge(raw) == expected


def test_integer_synthetic_kept_by_skip_dropped_by_drop() -> None:
    raw = {"replacements": [{"original": "12345", "label": "postcode", "synthetic": 94110}]}
    assert [e.synthetic for e in EntityReplacementMapSchema.from_raw(raw, on_invalid="skip")] == ["94110"]
    assert EntityReplacementMapSchema.from_raw(raw, on_invalid="drop") == []


_HUGE_INT_JSON = '{"replacements": [{"original": "a", "label": "b", "synthetic": ' + "1" * 4301 + "}]}"
_LONE_SURROGATE_JSON = '{"replacements": [{"original": "\\ud800", "label": "first_name", "synthetic": "x"}]}'


def _outcome(fn: Callable[[object], Any], raw: object) -> object:
    try:
        return fn(raw)
    except Exception as exc:
        return type(exc)


@pytest.mark.parametrize("raw", [_HUGE_INT_JSON, _LONE_SURROGATE_JSON], ids=["huge_int", "lone_surrogate"])
def test_from_raw_skip_matches_base_on_decoder_edges(raw: str) -> None:
    def head(value: object) -> list[tuple[str, str, str]]:
        return [(e.original, e.label, e.synthetic) for e in _parse_replacements(value)]

    assert _outcome(head, raw) == _outcome(_base_parse_replacements, raw)


@pytest.mark.parametrize("raw", [_HUGE_INT_JSON, _LONE_SURROGATE_JSON], ids=["huge_int", "lone_surrogate"])
def test_from_raw_drop_matches_base_on_decoder_edges(raw: str) -> None:
    expected = _outcome(_base_replacements_for_judge, raw)
    for judge in (attribute_fidelity_judge, relational_consistency_judge, type_fidelity_judge):
        assert _outcome(judge._replacements_for_judge, raw) == expected
