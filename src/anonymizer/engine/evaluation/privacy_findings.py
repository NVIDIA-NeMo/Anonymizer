# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize saved final privacy assessments for inspection without model calls."""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any, TypeGuard

import pandas as pd

from anonymizer.config.rewrite import SENSITIVITY_WEIGHTS
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_SENSITIVITY,
    COL_FINDING_ANSWER,
    COL_FINDING_CATEGORY,
    COL_FINDING_CONFIDENCE,
    COL_FINDING_CONTRIBUTION,
    COL_FINDING_ENTITY_LABEL,
    COL_FINDING_ENTITY_VALUE,
    COL_FINDING_ERROR,
    COL_FINDING_EVIDENCE,
    COL_FINDING_QUESTION,
    COL_FINDING_REASON,
    COL_FINDING_RECORD_ID,
    COL_FINDING_RECORD_INDEX,
    COL_FINDING_SENSITIVITY,
    COL_FINDING_SENSITIVITY_REASON,
    COL_FINDING_STATUS,
    COL_LEAKAGE_MASS,
    COL_PRIVACY_LEAKAGE_CONTRIBUTIONS,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_ERROR,
    COL_PRIVACY_QA_REANSWER,
    COL_SENSITIVITY_DISPOSITION,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.rewrite.parsers import normalize_payload
from anonymizer.engine.schemas import EntitiesByValueSchema

PRIVACY_FINDING_COLUMNS = (
    COL_FINDING_RECORD_INDEX,
    COL_FINDING_RECORD_ID,
    COL_FINDING_ENTITY_VALUE,
    COL_FINDING_ENTITY_LABEL,
    COL_FINDING_CATEGORY,
    COL_FINDING_SENSITIVITY,
    COL_FINDING_SENSITIVITY_REASON,
    COL_FINDING_QUESTION,
    COL_FINDING_ANSWER,
    COL_FINDING_CONFIDENCE,
    COL_FINDING_REASON,
    COL_FINDING_EVIDENCE,
    COL_FINDING_CONTRIBUTION,
    COL_FINDING_STATUS,
    COL_FINDING_ERROR,
)


def build_privacy_findings(trace_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build a fresh table from saved assessments, final questions, and answers.

    Record indexes are positional and zero-based, matching display_record().
    Missing assessments/answers never become negative answers or zero leakage.
    If no targets can be recovered for an unavailable/unevaluated record, a row
    with null entity fields preserves that record's status. Known empty target
    sets produce no rows. No repair-prompt history is read.
    """
    findings: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(trace_dataframe.iterrows()):
        record_id = row.get(RECORD_ID_COLUMN)
        missing_id = record_id is pd.NA or (isinstance(record_id, float) and pd.isna(record_id))
        base = {COL_FINDING_RECORD_INDEX: index, COL_FINDING_RECORD_ID: None if missing_id else record_id}
        try:
            findings.extend(_record_findings(row, base))
        except (ValueError, TypeError, KeyError):
            findings.append(
                {
                    **base,
                    COL_FINDING_STATUS: "unavailable",
                    COL_FINDING_ERROR: "Saved privacy assessment is malformed; findings unavailable.",
                }
            )
    # Object values preserve explicit None and avoid coercing missing contributions
    # to numeric zero. Deep-copy evidence lists so table edits cannot mutate traces.
    for finding in findings:
        for column in (COL_FINDING_ANSWER, COL_FINDING_CATEGORY, COL_FINDING_SENSITIVITY):
            value = finding.get(column)
            if isinstance(value, Enum):
                finding[column] = value.value
    rows = [{column: finding.get(column) for column in PRIVACY_FINDING_COLUMNS} for finding in findings]
    return pd.DataFrame(deepcopy(rows), columns=pd.Index(PRIVACY_FINDING_COLUMNS), dtype=object)


def summarize_privacy_findings(trace_dataframe: pd.DataFrame) -> dict[str, Any]:
    """Aggregate complete records without averaging their normalized rates.

    Unavailable records are excluded from both weighted sums. Rewrite entities
    exempted from QA are excluded from entity counts and possible leakage mass.
    Known empty evaluations count as evaluated records with zero leakage.
    """
    findings = build_privacy_findings(trace_dataframe)
    evaluated: set[int] = set()
    unavailable = 0
    for index, (_, row) in enumerate(trace_dataframe.iterrows()):
        record = findings[findings[COL_FINDING_RECORD_INDEX] == index]
        statuses = set(record[COL_FINDING_STATUS])
        answered = record[record[COL_FINDING_STATUS] == "evaluated"]
        if "unavailable" in statuses or (
            not answered.empty
            and (
                answered[COL_FINDING_CONTRIBUTION].isna().any()
                or not answered[COL_FINDING_SENSITIVITY].isin(SENSITIVITY_WEIGHTS).all()
            )
        ):
            unavailable += 1
            continue
        questions = _value(row.get(COL_PRIVACY_QA))
        answers = _value(row.get(COL_PRIVACY_QA_REANSWER))
        empty_evaluation = (
            isinstance(questions, dict)
            and questions.get("items") == []
            and isinstance(answers, dict)
            and answers.get("answers") == []
        )
        passthrough = record.empty and _value(row.get(COL_LEAKAGE_MASS)) == 0
        if "evaluated" in statuses or empty_evaluation or passthrough:
            evaluated.add(index)

    eligible = findings[
        findings[COL_FINDING_RECORD_INDEX].isin(evaluated) & (findings[COL_FINDING_STATUS] == "evaluated")
    ]
    leaked = eligible[eligible[COL_FINDING_ANSWER] == "yes"]
    by_sensitivity = {
        sensitivity: float(leaked.loc[leaked[COL_FINDING_SENSITIVITY] == sensitivity, COL_FINDING_CONTRIBUTION].sum())
        for sensitivity in ("high", "medium", "low")
    }
    mass = sum(by_sensitivity.values())
    possible_mass = float(eligible[COL_FINDING_SENSITIVITY].map(SENSITIVITY_WEIGHTS).sum())
    record_leaks = int(leaked[COL_FINDING_RECORD_INDEX].nunique())
    return {
        "records_total": len(trace_dataframe),
        "records_evaluated": len(evaluated),
        "records_unavailable": unavailable,
        "records_not_evaluated": len(trace_dataframe) - len(evaluated) - unavailable,
        "records_with_leakage": record_leaks,
        "record_leakage_rate": record_leaks / len(evaluated) if evaluated else None,
        "entities_evaluated": len(eligible),
        "entities_leaked": len(leaked),
        "entity_recovery_rate": len(leaked) / len(eligible) if len(eligible) else (0.0 if evaluated else None),
        "high_sensitivity_entities_leaked": int((leaked[COL_FINDING_SENSITIVITY] == "high").sum()),
        "leakage_mass": mass if evaluated else None,
        "weighted_leakage_rate": mass / possible_mass if possible_mass else (0.0 if evaluated else None),
        "leakage_mass_by_sensitivity": by_sensitivity if evaluated else {key: None for key in by_sensitivity},
    }


def _value(raw: Any) -> Any:
    if raw is None or raw is pd.NA or (isinstance(raw, float) and pd.isna(raw)):
        return None
    return normalize_payload(raw)


def _items(raw: Any, key: str) -> list[dict[str, Any]]:
    payload = _value(raw)
    if payload is None:
        return []
    values = payload.get(key) if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
        raise ValueError("Malformed privacy assessment collection")
    return values


def _pair(entity: dict[str, Any]) -> tuple[str, str]:
    return entity["entity_value"], entity["entity_label"]


def _unique_by(items: list[dict[str, Any]], key: str) -> dict[Any, dict[str, Any]]:
    result = {item[key]: item for item in items}
    if len(result) != len(items):
        raise ValueError("Ambiguous privacy assessment IDs")
    return result


def _record_findings(row: pd.Series, base: dict[str, Any]) -> list[dict[str, Any]]:
    error = _value(row.get(COL_PRIVACY_QA_ERROR))
    assessments = _items(row.get(COL_ENTITY_SENSITIVITY), "entities")
    dispositions = _items(row.get(COL_SENSITIVITY_DISPOSITION), "sensitivity_disposition")
    targets = assessments or dispositions
    questions = _items(row.get(COL_PRIVACY_QA), "items")
    answers = _unique_by(_items(row.get(COL_PRIVACY_QA_REANSWER), "answers"), "id")
    contributions = _unique_by(_items(row.get(COL_PRIVACY_LEAKAGE_CONTRIBUTIONS), "entities"), "id")
    _unique_by(questions, "id")
    if set(answers) - {question["id"] for question in questions}:
        raise ValueError("Privacy answers refer to unknown questions")
    by_pair = {_pair(question): question for question in questions}
    if len(by_pair) != len(questions) or len({_pair(target) for target in targets}) != len(targets):
        raise ValueError("Ambiguous privacy assessment targets")
    assessed_pairs = {_pair(target) for target in targets}
    targets = [*targets, *(question for question in questions if _pair(question) not in assessed_pairs)]
    if not targets and _value(row.get(COL_ENTITIES_BY_VALUE)) is not None:
        original = EntitiesByValueSchema.from_raw(row[COL_ENTITIES_BY_VALUE], strict=True)
        targets = [
            {"entity_value": entity.value, "entity_label": label}
            for entity in original.entities_by_value
            for label in dict.fromkeys(entity.labels)
        ]
    if not targets:
        known_empty = any(
            _value(row.get(column)) is not None
            for column in (COL_PRIVACY_QA, COL_ENTITIES_BY_VALUE, COL_ENTITY_SENSITIVITY)
        )
        if known_empty and not error:
            return []
        return [{**base, COL_FINDING_STATUS: "unavailable" if error else "not_evaluated", COL_FINDING_ERROR: error}]

    findings = []
    for target in targets:
        question = by_pair.get(_pair(target))
        answer = answers.get(question["id"]) if question else None
        status = "not_evaluated"
        entity_error = error
        contribution = None
        if question:
            if not error and _complete_answer(answer):
                status = "evaluated"
                saved = contributions.get(question["id"])
                if saved is not None and _pair(saved) == _pair(question):
                    contribution = saved.get("leakage_contribution")
                else:
                    contribution = (
                        SENSITIVITY_WEIGHTS[question["sensitivity"]] * answer["confidence"]
                        if answer["answer"] == "yes"
                        else 0.0
                    )
            else:
                status = "unavailable"
                entity_error = error or "Final privacy answer is missing or incomplete."
        elif target.get("protection_method_suggestion") != "leave_as_is" and (error or COL_PRIVACY_QA in row.index):
            status = "unavailable"
            entity_error = error or "Privacy question is unavailable for this entity."
        findings.append(
            {
                **base,
                COL_FINDING_ENTITY_VALUE: target["entity_value"],
                COL_FINDING_ENTITY_LABEL: target["entity_label"],
                COL_FINDING_CATEGORY: target.get("category"),
                COL_FINDING_SENSITIVITY: target.get("sensitivity"),
                COL_FINDING_SENSITIVITY_REASON: target.get(
                    "reason", target.get("protection_reason", target.get("low_sensitivity_reason"))
                ),
                COL_FINDING_QUESTION: question.get("question") if question else None,
                COL_FINDING_ANSWER: answer.get("answer") if answer else None,
                COL_FINDING_CONFIDENCE: answer.get("confidence") if answer else None,
                COL_FINDING_REASON: answer.get("reason") if answer else None,
                COL_FINDING_EVIDENCE: answer.get("evidence") if answer else None,
                COL_FINDING_CONTRIBUTION: contribution,
                COL_FINDING_STATUS: status,
                COL_FINDING_ERROR: entity_error,
            }
        )
    return findings


def _complete_answer(answer: dict[str, Any] | None) -> TypeGuard[dict[str, Any]]:
    if answer is None:
        return False
    confidence = answer.get("confidence")
    return (
        answer.get("answer") in ("yes", "no")
        and isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and 0 <= confidence <= 1
        and isinstance(answer.get("reason"), str)
        and bool(answer["reason"])
        and isinstance(answer.get("evidence"), list)
        and all(isinstance(quote, str) for quote in answer["evidence"])
    )
