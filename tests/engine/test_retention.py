# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from data_designer.config import SkipConfig, custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.models import ReplaceModelSelection, RewriteModelSelection
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal, RiskTolerance
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_DISPOSITION_MISSING_ENTITIES,
    COL_ENTITIES_BY_VALUE,
    COL_LATENT_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_LEAKED_PRIVACY_ITEMS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_NEEDS_REPAIR,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA_COMPARE,
    COL_QUALITY_QA_REANSWER,
    COL_REPAIR_ITERATIONS,
    COL_RETAINED_PROTECTED_ENTITIES,
    COL_REWRITE_REPLACEMENT_READY,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_INITIAL,
    COL_REWRITTEN_TEXT_NEXT,
    COL_SENSITIVITY_DISPOSITION,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, NddAdapter
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow, EvaluationState, RepairState
from anonymizer.engine.rewrite.evaluate import EvaluateWorkflow, _determine_repair_needs_column
from anonymizer.engine.rewrite.parsers import parse_privacy_answers, parse_privacy_qa
from anonymizer.engine.rewrite.repair import _inject_leaked_items_column, _leaked_items_text
from anonymizer.engine.rewrite.retention import apply_review_gates, disposition_coverage, retained_protected
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow

_PRIVACY_GOAL = PrivacyGoal(
    protect="All direct identifiers including names, locations, and contact details",
    preserve="Career trajectory, skills, and professional context in abstract terms",
)
_REPLACE_PATCH = "anonymizer.engine.rewrite.rewrite_workflow.LlmReplaceWorkflow"
_STUB_REWRITE = "stub_rewrite"
_FIXTURE = Path(__file__).parents[1] / "fixtures" / "disposition_coverage_recorded.json"


def _entities(*pairs: tuple[str, str]) -> dict[str, Any]:
    return {"entities_by_value": [{"value": value, "labels": [label]} for value, label in pairs]}


def _entry(value: str, label: str, method: str, source: str = "tagged") -> dict[str, Any]:
    return {"source": source, "entity_label": label, "entity_value": value, "protection_method_suggestion": method}


def _disposition(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"sensitivity_disposition": list(entries)}


def _retained_values(disposition: dict[str, Any], text: str) -> list[str]:
    return [value for value, _, _ in retained_protected(disposition, text)]


# ---------------------------------------------------------------------------
# Pure functions: coverage
# ---------------------------------------------------------------------------


def test_coverage_lists_omitted_tagged_entity() -> None:
    entities = _entities(("Alice", "first_name"), ("Acme", "organization_name"))
    partial = _disposition(_entry("Acme", "organization_name", "generalize"))
    full = _disposition(_entry("Acme", "organization_name", "generalize"), _entry("Alice", "first_name", "replace"))

    assert disposition_coverage(entities, None, partial) == [("Alice", "first_name")]
    assert disposition_coverage(entities, None, full) == []


def test_coverage_keys_tagged_on_value_and_latent_on_value_and_label() -> None:
    entities = {"entities_by_value": [{"value": "Jordan", "labels": ["first_name", "city"]}]}
    latent = {"latent_entities": np.array([{"label": "employer", "value": "a hospital"}], dtype=object)}
    disposition = _disposition(
        _entry("JORDAN", "city", "replace"),
        _entry("a hospital", "health_condition", "suppress_inference", source="latent"),
    )

    assert disposition_coverage(entities, latent, disposition) == [("a hospital", "employer")]
    covered = _disposition(
        _entry("Jordan", "first_name", "replace"),
        _entry("a hospital", "employer", "suppress_inference", source="latent"),
    )
    assert disposition_coverage(entities, latent, covered) == []


def test_coverage_normalizes_unicode_variants() -> None:
    entities = _entities(("196 Route 417", "street_address"), ("33‑year‑old", "age"))
    disposition = _disposition(
        _entry("196 route 417", "street_address", "generalize"), _entry("33-year-old", "age", "x")
    )
    assert disposition_coverage(entities, None, disposition) == []


@pytest.mark.parametrize("disposition", [None, float("nan"), "not json", 3, {"other": []}])
def test_unparseable_disposition_is_no_evidence(disposition: Any) -> None:
    assert disposition_coverage(_entities(("Alice", "first_name")), None, disposition) == []
    assert retained_protected(disposition, "Alice") == []


def test_coverage_accepts_json_strings() -> None:
    entities = json.dumps(_entities(("Alice", "first_name")))
    disposition = json.dumps(_disposition(_entry("Bob", "first_name", "replace")))
    assert disposition_coverage(entities, json.dumps({"latent_entities": []}), disposition) == [("Alice", "first_name")]


# ---------------------------------------------------------------------------
# Pure functions: retention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["replace", "generalize", "remove", "suppress_inference"])
def test_protected_value_surviving_is_retained(method: str) -> None:
    disposition = _disposition(_entry("Alice", "first_name", method))
    assert retained_protected(disposition, "Alice went home.") == [("Alice", "first_name", method)]


def test_leave_as_is_value_surviving_is_not_retained() -> None:
    assert retained_protected(_disposition(_entry("English", "language", "leave_as_is")), "She speaks English.") == []


@pytest.mark.parametrize(
    ("value", "text", "expected"),
    [
        ("Ann", "The Annual report was filed.", False),
        ("Alice", "ALICE went home.", True),
        ("196 Route 417", "She lives at 196 Route 417.", True),
        ("O’Brien", "Mr. O'Brien called.", True),
        ("33-year-old", "A 33–year–old engineer.", True),
        ("Alice Smith", "Written by Alice\nSmith today.", True),
        ("B.S.", "She holds a B.S. in physics.", True),
        ("(555) 010-0142", "Call (555) 010-0142 now.", True),
        ("alice@example.com", "Email alice@example.com today.", True),
        ("C++", "He writes C++ daily.", True),
        ("Jr.", "Sammy Davis Jr. sang.", True),
        ("A", "A cat sat.", False),
        ("--", "a -- b", False),
        ("B.S.", "She holds a degree.", False),
    ],
)
def test_retention_matching(value: str, text: str, expected: bool) -> None:
    disposition = _disposition(_entry(value, "label", "replace"))
    assert bool(retained_protected(disposition, text)) is expected


def test_protected_value_inside_kept_value_is_masked() -> None:
    disposition = _disposition(
        _entry("Florida", "state", "generalize"),
        _entry("Florida State University", "organization_name", "leave_as_is"),
    )
    assert retained_protected(disposition, "She studied at Florida State University.") == []
    assert _retained_values(disposition, "She studied at Florida State University in Florida.") == ["Florida"]


def test_retention_regex_metacharacters_are_escaped() -> None:
    disposition = _disposition(_entry("a.b*c", "label", "replace"))
    assert retained_protected(disposition, "value axbbbc here") == []
    assert _retained_values(disposition, "value a.b*c here") == ["a.b*c"]


# ---------------------------------------------------------------------------
# Repair trigger and prompt
# ---------------------------------------------------------------------------


def _repair_needs(evaluation: EvaluationCriteria, retained: Any) -> bool:
    column = EvaluateWorkflow(adapter=Mock()).columns(selected_models=Mock(), evaluation=evaluation)[-1]
    assert isinstance(column, CustomColumnConfig)
    row = {COL_ANY_HIGH_LEAKED: False, COL_LEAKAGE_MASS: 0.0, COL_RETAINED_PROTECTED_ENTITIES: retained}
    return column.generator_function(row, column.generator_params)[COL_NEEDS_REPAIR]


@pytest.mark.parametrize("risk_tolerance", list(RiskTolerance))
def test_retained_entity_triggers_repair_under_every_preset(risk_tolerance: RiskTolerance) -> None:
    evaluation = EvaluationCriteria(risk_tolerance=risk_tolerance)
    assert _repair_needs(evaluation, [["Alice", "first_name", "replace"]]) is True
    assert _repair_needs(evaluation, np.array([np.array(["Alice", "first_name", "replace"])], dtype=object)) is True
    assert _repair_needs(evaluation, []) is False
    assert _repair_needs(evaluation, None) is False


def test_evaluate_columns_order_retained_before_repair_needs() -> None:
    columns = EvaluateWorkflow(adapter=Mock()).columns(selected_models=Mock(), evaluation=EvaluationCriteria())
    assert [column.name for column in columns[-2:]] == [COL_RETAINED_PROTECTED_ENTITIES, COL_NEEDS_REPAIR]
    assert isinstance(columns[-1], CustomColumnConfig)
    assert columns[-1].generator_function is _determine_repair_needs_column


_PRIVACY_QA = {
    "items": [
        {
            "id": 1,
            "question": "Can the first_name 'Alice' be deduced?",
            "sensitivity": "high",
            "entity_label": "first_name",
            "entity_value": "Alice",
            "category": "direct_identifier",
        },
    ]
}


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_repair_prompt_names_retained_entities_and_methods(answer: str) -> None:
    reanswer = {"answers": [{"id": 1, "answer": answer, "confidence": 0.8, "reason": "explicit"}]}
    row = {
        COL_PRIVACY_QA: _PRIVACY_QA,
        COL_PRIVACY_QA_REANSWER: reanswer,
        COL_RETAINED_PROTECTED_ENTITIES: [["Alice", "first_name", "replace"], ["Florida", "state", "generalize"]],
    }
    text = _inject_leaked_items_column(row)[COL_LEAKED_PRIVACY_ITEMS]
    assert '- [RETAINED] first_name: "Alice" still appears verbatim; must be replaced (method: replace).' in text
    assert '- [RETAINED] state: "Florida" still appears verbatim; must be generalized (method: generalize).' in text


@pytest.mark.parametrize("answer", ["yes", "no"])
@pytest.mark.parametrize("retained", [[], None, np.array([], dtype=object)])
def test_repair_prompt_unchanged_when_nothing_retained(answer: str, retained: Any) -> None:
    reanswer = {"answers": [{"id": 1, "answer": answer, "confidence": 0.8, "reason": "explicit"}]}
    row = {COL_PRIVACY_QA: _PRIVACY_QA, COL_PRIVACY_QA_REANSWER: reanswer, COL_RETAINED_PROTECTED_ENTITIES: retained}
    baseline = _leaked_items_text(parse_privacy_answers(reanswer), parse_privacy_qa(_PRIVACY_QA))
    assert _inject_leaked_items_column(dict(row))[COL_LEAKED_PRIVACY_ITEMS] == baseline


def test_combined_repair_column_reads_the_iteration_retained_column(
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    graph = CombinedRewriteWorkflow(adapter=Mock()).build_graph(
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=EvaluationCriteria(max_repair_iterations=1),
    )
    previous, state = graph.evaluation_states[0], graph.repair_states[0]
    column = next(column for column in graph.columns if column.name == state.leaked_items)
    assert isinstance(column, CustomColumnConfig)
    assert previous.retained in column.generator_function.custom_column_metadata["required_columns"]

    row = {
        previous.privacy_reanswer: {"answers": []},
        COL_PRIVACY_QA: {"items": []},
        previous.retained: [["Alice", "first_name", "replace"]],
    }
    text = column.generator_function(row)[state.leaked_items]
    assert text == '- [RETAINED] first_name: "Alice" still appears verbatim; must be replaced (method: replace).'


# ---------------------------------------------------------------------------
# Recorded disposition replay
# ---------------------------------------------------------------------------

_EXPECTED_MISSING = {
    "r05": (17, 2),
    "r08": (14, 2),
    "r09": (0, 0),
    "r13": (12, 3),
    "r14": (4, 2),
    "r16": (9, 2),
    "r18": (9, 1),
    "r20": (6, 1),
    "r22": (5, 2),
    "r23": (13, 2),
}


def test_recorded_rows_coverage_and_retention() -> None:
    rows = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    observed: dict[str, tuple[int, int]] = {}
    for row in rows:
        entities = {"entities_by_value": row["entities_by_value"]}
        latent = {"latent_entities": row["latent_entities"]}
        disposition = _disposition(*row["sensitivity_disposition"])
        missing = disposition_coverage(entities, latent, disposition)
        tagged_values = {entity["value"] for entity in row["entities_by_value"]}
        tagged = [value for value, _ in missing if value in tagged_values]
        observed[row["row"]] = (len(tagged), len(missing) - len(tagged))

        text = row.get("rewritten_text")
        if text is None:
            continue
        kept_names = {
            entity["value"]
            for entity in row["entities_by_value"]
            if {"first_name", "last_name"} & set(entity["labels"])
            and re.search(rf"(?<!\w){re.escape(entity['value'])}(?!\w)", text)
        }
        assert kept_names
        assert kept_names <= {value for value, _ in missing}
        expected_retained = [("Florida", "state", "generalize")] if row["row"] == "r18" else []
        assert retained_protected(disposition, text) == expected_retained

    assert observed == _EXPECTED_MISSING
    assert [name for name, (tagged, latent) in observed.items() if tagged + latent == 0] == ["r09"]


# ---------------------------------------------------------------------------
# Workflows through the real NddAdapter with deterministic stub columns
# ---------------------------------------------------------------------------


def _strip_protected(text: str, disposition: Any) -> str:
    for value, _, _ in retained_protected(disposition, text):
        text = text.replace(value, "someone")
    return text


@custom_column_generator(
    required_columns=[_STUB_REWRITE],
    side_effect_columns=[COL_REWRITE_REPLACEMENT_READY],
)
def _stub_staged_rewrite(row: dict[str, Any]) -> dict[str, Any]:
    row[COL_REWRITTEN_TEXT] = row[_STUB_REWRITE]
    row[COL_REWRITE_REPLACEMENT_READY] = True
    return row


@custom_column_generator(
    required_columns=[COL_REWRITTEN_TEXT],
    side_effect_columns=[
        COL_PRIVACY_QA_REANSWER,
        COL_QUALITY_QA_COMPARE,
        COL_UTILITY_SCORE,
        COL_LEAKAGE_MASS,
        COL_WEIGHTED_LEAKAGE_RATE,
        COL_ANY_HIGH_LEAKED,
    ],
)
def _stub_metrics(row: dict[str, Any]) -> dict[str, Any]:
    row[COL_QUALITY_QA_REANSWER] = {"answers": []}
    row[COL_PRIVACY_QA_REANSWER] = {"answers": []}
    row[COL_QUALITY_QA_COMPARE] = {"per_item": []}
    row[COL_UTILITY_SCORE] = 1.0
    row[COL_LEAKAGE_MASS] = 0.0
    row[COL_WEIGHTED_LEAKAGE_RATE] = 0.0
    row[COL_ANY_HIGH_LEAKED] = False
    return row


@custom_column_generator(required_columns=[COL_REWRITTEN_TEXT, COL_SENSITIVITY_DISPOSITION])
def _stub_repair(row: dict[str, Any]) -> dict[str, Any]:
    row[COL_REWRITTEN_TEXT_NEXT] = _strip_protected(row[COL_REWRITTEN_TEXT], row[COL_SENSITIVITY_DISPOSITION])
    return row


def _stub_evaluate_columns(evaluation: EvaluationCriteria) -> list[ColumnConfigT]:
    real = EvaluateWorkflow(adapter=Mock()).columns(selected_models=Mock(), evaluation=evaluation)
    return [CustomColumnConfig(name=COL_QUALITY_QA_REANSWER, generator_function=_stub_metrics), *real[-2:]]


def _stub_combined_columns(
    workflow: CombinedRewriteWorkflow,
    evaluation: EvaluationCriteria,
    rewrite_selection: RewriteModelSelection,
    replace_selection: ReplaceModelSelection,
) -> Any:
    graph = workflow.build_graph(
        selected_models=rewrite_selection,
        replace_model_selection=replace_selection,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=evaluation,
    )
    by_name = {column.name: column for column in graph.columns}

    def evaluation_columns(state: EvaluationState) -> list[ColumnConfigT]:
        @custom_column_generator(
            required_columns=[state.rewritten_text],
            side_effect_columns=[
                state.privacy_reanswer,
                state.quality_compare,
                state.utility_score,
                state.leakage_mass,
                state.weighted_leakage_rate,
                state.any_high_leaked,
            ],
        )
        def metrics(row: dict[str, Any]) -> dict[str, Any]:
            generated = _stub_metrics({COL_REWRITTEN_TEXT: row[state.rewritten_text]})
            for column, target in (
                (COL_QUALITY_QA_REANSWER, state.quality_reanswer),
                (COL_PRIVACY_QA_REANSWER, state.privacy_reanswer),
                (COL_QUALITY_QA_COMPARE, state.quality_compare),
                (COL_UTILITY_SCORE, state.utility_score),
                (COL_LEAKAGE_MASS, state.leakage_mass),
                (COL_WEIGHTED_LEAKAGE_RATE, state.weighted_leakage_rate),
                (COL_ANY_HIGH_LEAKED, state.any_high_leaked),
            ):
                row[target] = generated[column]
            return row

        return [
            CustomColumnConfig(name=state.quality_reanswer, generator_function=metrics),
            by_name[state.retained],
            by_name[state.needs_repair],
        ]

    def repair_column(previous: EvaluationState, state: RepairState) -> CustomColumnConfig:
        @custom_column_generator(
            required_columns=[previous.needs_repair, previous.rewritten_text, COL_SENSITIVITY_DISPOSITION]
        )
        def repair(row: dict[str, Any]) -> dict[str, Any]:
            row[state.rewritten_text] = _strip_protected(row[previous.rewritten_text], row[COL_SENSITIVITY_DISPOSITION])
            return row

        return CustomColumnConfig(
            name=state.rewritten_text,
            generator_function=repair,
            skip=SkipConfig(when=f"{{{{ not {previous.needs_repair} }}}}"),
        )

    @custom_column_generator(required_columns=[_STUB_REWRITE], side_effect_columns=[COL_REWRITE_REPLACEMENT_READY])
    def initial(row: dict[str, Any]) -> dict[str, Any]:
        row[COL_REWRITTEN_TEXT_INITIAL] = row[_STUB_REWRITE]
        row[COL_REWRITE_REPLACEMENT_READY] = True
        return row

    columns: list[ColumnConfigT] = [
        CustomColumnConfig(name=COL_REWRITTEN_TEXT_INITIAL, generator_function=initial),
        *evaluation_columns(graph.evaluation_states[0]),
    ]
    for previous, repair_state, current in zip(
        graph.evaluation_states, graph.repair_states, graph.evaluation_states[1:]
    ):
        columns.extend([repair_column(previous, repair_state), *evaluation_columns(current)])
    columns.append(graph.columns[-1])
    return type(graph)(
        columns=columns,
        evaluation_states=graph.evaluation_states,
        repair_states=graph.repair_states,
        internal_columns=graph.internal_columns,
    )


def _run(
    tmp_path: Path,
    dataframe: pd.DataFrame,
    *,
    combined: bool,
    evaluation: EvaluationCriteria,
    rewrite_selection: RewriteModelSelection,
    replace_selection: ReplaceModelSelection,
) -> tuple[pd.DataFrame, Mock]:
    adapter = NddAdapter(DataDesigner(artifact_path=tmp_path / "artifacts", auto_configure_logging=False))
    spy = Mock(wraps=adapter.run_workflow)
    adapter.run_workflow = spy  # type: ignore[method-assign]
    if combined:
        workflow = CombinedRewriteWorkflow(adapter)
        graph = _stub_combined_columns(workflow, evaluation, rewrite_selection, replace_selection)
        with patch.object(workflow, "build_graph", return_value=graph):
            result = workflow.run(
                dataframe,
                model_configs=[],
                selected_models=rewrite_selection,
                replace_model_selection=replace_selection,
                privacy_goal=_PRIVACY_GOAL,
                evaluation=evaluation,
            )
        return result.dataframe, spy

    workflow = RewriteWorkflow(adapter)
    for name in ("_domain_wf", "_disposition_wf", "_qa_wf"):
        setattr(workflow, name, Mock(columns=Mock(return_value=[])))
    workflow._rewrite_gen_wf = Mock(
        columns=Mock(
            return_value=[CustomColumnConfig(name=COL_REWRITTEN_TEXT, generator_function=_stub_staged_rewrite)]
        )
    )
    workflow._evaluate_wf = Mock(columns=Mock(return_value=_stub_evaluate_columns(evaluation)))
    workflow._repair_wf = Mock(
        columns=Mock(return_value=[CustomColumnConfig(name=COL_REWRITTEN_TEXT_NEXT, generator_function=_stub_repair)])
    )
    with patch(_REPLACE_PATCH) as replace_cls:
        replace_cls.return_value.generate_map_only.side_effect = lambda df, **_: Mock(
            dataframe=df.copy(), failed_records=[]
        )
        result = workflow.run(
            dataframe,
            model_configs=[],
            selected_models=rewrite_selection,
            replace_model_selection=replace_selection,
            privacy_goal=_PRIVACY_GOAL,
            evaluation=evaluation,
        )
    return result.dataframe, spy


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    base = {
        COL_LATENT_ENTITIES: {"latent_entities": []},
        COL_SENSITIVITY_DISPOSITION: None,
    }
    frame = pd.DataFrame([{**base, **row} for row in rows])
    frame[RECORD_ID_COLUMN] = [row["id"] for row in rows]
    return frame.drop(columns=["id"])


_ALICE = _entities(("Alice", "first_name"), ("Acme", "organization_name"))
_FULL = _disposition(_entry("Alice", "first_name", "replace"), _entry("Acme", "organization_name", "generalize"))
_PARTIAL = _disposition(_entry("Acme", "organization_name", "generalize"))


def _row(record_id: str, disposition: Any, rewrite: str, entities: Any = _ALICE, **extra: Any) -> dict[str, Any]:
    return {
        "id": record_id,
        COL_TEXT: "Alice works at Acme.",
        COL_ENTITIES_BY_VALUE: entities,
        COL_SENSITIVITY_DISPOSITION: disposition,
        _STUB_REWRITE: rewrite,
        **extra,
    }


def _by_id(result: pd.DataFrame, column: str) -> dict[str, Any]:
    return {
        record_id: (list(value) if isinstance(value, (list, np.ndarray)) else value)
        for record_id, value in zip(result[RECORD_ID_COLUMN], result[column])
    }


@pytest.mark.parametrize("combined", [False, True], ids=["staged", "combined"])
def test_incomplete_disposition_flags_review(
    tmp_path: Path,
    combined: bool,
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    dataframe = _frame(
        [
            _row("incomplete", _PARTIAL, "Someone works at a company."),
            _row("complete", _FULL, "Someone works at a company."),
        ]
    )
    result, _ = _run(
        tmp_path,
        dataframe,
        combined=combined,
        evaluation=EvaluationCriteria(max_repair_iterations=2),
        rewrite_selection=stub_rewrite_model_selection,
        replace_selection=stub_replace_model_selection,
    )

    missing = _by_id(result, COL_DISPOSITION_MISSING_ENTITIES)
    assert [list(item) for item in missing["incomplete"]] == [["Alice", "first_name"]]
    assert missing["complete"] == []
    assert _by_id(result, COL_NEEDS_HUMAN_REVIEW) == {"incomplete": True, "complete": False}
    assert _by_id(result, COL_REPAIR_ITERATIONS) == {"incomplete": 0, "complete": 0}


@pytest.mark.parametrize("combined", [False, True], ids=["staged", "combined"])
@pytest.mark.parametrize("risk_tolerance", list(RiskTolerance))
def test_retained_value_repairs_until_removed(
    tmp_path: Path,
    combined: bool,
    risk_tolerance: RiskTolerance,
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    dataframe = _frame([_row("retained", _FULL, "Alice works at a company.")])
    result, spy = _run(
        tmp_path,
        dataframe,
        combined=combined,
        evaluation=EvaluationCriteria(risk_tolerance=risk_tolerance, max_repair_iterations=3),
        rewrite_selection=stub_rewrite_model_selection,
        replace_selection=stub_replace_model_selection,
    )

    assert result[COL_REWRITTEN_TEXT].tolist() == ["someone works at a company."]
    assert result[COL_REPAIR_ITERATIONS].tolist() == [1]
    assert list(result[COL_RETAINED_PROTECTED_ENTITIES].iloc[0]) == []
    assert result[COL_NEEDS_REPAIR].map(bool).tolist() == [False]
    assert result[COL_NEEDS_HUMAN_REVIEW].tolist() == [False]
    if not combined:
        names = [call.kwargs["workflow_name"] for call in spy.call_args_list]
        assert names == ["rewrite-pipeline", "rewrite-evaluate-0", "rewrite-repair-0", "rewrite-evaluate-1"]


@pytest.mark.parametrize("combined", [False, True], ids=["staged", "combined"])
def test_retained_value_without_repair_budget_flags_review(
    tmp_path: Path,
    combined: bool,
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    dataframe = _frame([_row("retained", _FULL, "Alice works at a company.")])
    result, spy = _run(
        tmp_path,
        dataframe,
        combined=combined,
        evaluation=EvaluationCriteria(risk_tolerance=RiskTolerance.high, max_repair_iterations=0),
        rewrite_selection=stub_rewrite_model_selection,
        replace_selection=stub_replace_model_selection,
    )

    assert result[COL_REWRITTEN_TEXT].tolist() == ["Alice works at a company."]
    assert result[COL_REPAIR_ITERATIONS].tolist() == [0]
    assert [list(item) for item in result[COL_RETAINED_PROTECTED_ENTITIES].iloc[0]] == [
        ["Alice", "first_name", "replace"]
    ]
    assert result[COL_NEEDS_REPAIR].map(bool).tolist() == [True]
    assert result[COL_NEEDS_HUMAN_REVIEW].tolist() == [True]
    assert not any("repair" in call.kwargs["workflow_name"] for call in spy.call_args_list)


@pytest.mark.parametrize("combined", [False, True], ids=["staged", "combined"])
def test_mixed_frame_flags_land_on_right_records(
    tmp_path: Path,
    combined: bool,
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    latent_disposition = _disposition(
        _entry("Alice", "first_name", "replace"),
        _entry("Acme", "organization_name", "generalize"),
        _entry("a hospital", "employer", "suppress_inference", source="latent"),
    )
    latent = {"latent_entities": [{"label": "employer", "value": "a hospital"}]}
    no_entities = {"entities_by_value": []}
    dataframe = _frame(
        [
            _row("passthrough-1", None, "unused", entities=no_entities),
            _row("repair", _FULL, "Alice works at a company."),
            _row("incomplete", _PARTIAL, "Someone works at a company."),
            _row("passthrough-2", None, "unused", entities=no_entities),
            _row("latent-missing", _FULL, "Someone works at a company.", **{COL_LATENT_ENTITIES: latent}),
            _row("latent-retained", latent_disposition, "Someone at a hospital.", **{COL_LATENT_ENTITIES: latent}),
            _row("clean", _FULL, "Someone works at a company."),
        ]
    )
    result, _ = _run(
        tmp_path,
        dataframe,
        combined=combined,
        evaluation=EvaluationCriteria(max_repair_iterations=0),
        rewrite_selection=stub_rewrite_model_selection,
        replace_selection=stub_replace_model_selection,
    )

    print(f"pandas {pd.__version__}")
    assert result[RECORD_ID_COLUMN].tolist() == dataframe[RECORD_ID_COLUMN].tolist()
    assert _by_id(result, COL_NEEDS_HUMAN_REVIEW) == {
        "passthrough-1": False,
        "repair": True,
        "incomplete": True,
        "passthrough-2": False,
        "latent-missing": True,
        "latent-retained": True,
        "clean": False,
    }
    missing = _by_id(result, COL_DISPOSITION_MISSING_ENTITIES)
    assert [list(item) for item in missing["latent-missing"]] == [["a hospital", "employer"]]
    assert missing["latent-retained"] == [] and missing["repair"] == [] and missing["clean"] == []
    retained = _by_id(result, COL_RETAINED_PROTECTED_ENTITIES)
    assert [list(item) for item in retained["latent-retained"]] == [["a hospital", "employer", "suppress_inference"]]
    assert [list(item) for item in retained["repair"]] == [["Alice", "first_name", "replace"]]
    assert retained["incomplete"] == [] and retained["clean"] == []


@pytest.mark.parametrize("combined", [False, True], ids=["staged", "combined"])
def test_mixed_frame_with_repairs_keeps_flags_on_right_records(
    tmp_path: Path,
    combined: bool,
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    no_entities = {"entities_by_value": []}
    dataframe = _frame(
        [
            _row("passthrough-1", None, "unused", entities=no_entities),
            _row("gap-repaired", _PARTIAL, "Alice works at Acme."),
            _row("clean-1", _FULL, "Someone works at a company."),
            _row("repaired", _FULL, "Alice works at a company."),
            _row("gap-passing", _PARTIAL, "Someone works at a company."),
            _row("passthrough-2", None, "unused", entities=no_entities),
            _row("clean-2", _FULL, "Someone works at a company."),
        ]
    )
    result, spy = _run(
        tmp_path,
        dataframe,
        combined=combined,
        evaluation=EvaluationCriteria(max_repair_iterations=2),
        rewrite_selection=stub_rewrite_model_selection,
        replace_selection=stub_replace_model_selection,
    )

    if not combined:
        # Failing rows are concatenated after passing rows, so the frame is reordered before the gates run.
        names = [call.kwargs["workflow_name"] for call in spy.call_args_list]
        assert "rewrite-repair-0" in names
    assert _by_id(result, COL_REPAIR_ITERATIONS) == {
        "passthrough-1": 0,
        "gap-repaired": 1,
        "clean-1": 0,
        "repaired": 1,
        "gap-passing": 0,
        "passthrough-2": 0,
        "clean-2": 0,
    }
    assert _by_id(result, COL_NEEDS_HUMAN_REVIEW) == {
        "passthrough-1": False,
        "gap-repaired": True,
        "clean-1": False,
        "repaired": False,
        "gap-passing": True,
        "passthrough-2": False,
        "clean-2": False,
    }
    rewritten = _by_id(result, COL_REWRITTEN_TEXT)
    assert rewritten["gap-repaired"] == "Alice works at someone."
    assert rewritten["repaired"] == "someone works at a company."
    missing = _by_id(result, COL_DISPOSITION_MISSING_ENTITIES)
    assert [list(item) for item in missing["gap-repaired"]] == [["Alice", "first_name"]]
    assert [list(item) for item in missing["gap-passing"]] == [["Alice", "first_name"]]
    assert missing["repaired"] == [] and missing["clean-1"] == [] and missing["clean-2"] == []


@pytest.mark.parametrize("combined", [False, True], ids=["staged", "combined"])
def test_all_passthrough_frame_is_unchanged(
    tmp_path: Path,
    combined: bool,
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    dataframe = _frame(
        [
            _row("a", None, "unused", entities={"entities_by_value": []}),
            _row("b", None, "unused", entities={"entities_by_value": []}),
        ]
    )
    result, spy = _run(
        tmp_path,
        dataframe,
        combined=combined,
        evaluation=EvaluationCriteria(),
        rewrite_selection=stub_rewrite_model_selection,
        replace_selection=stub_replace_model_selection,
    )

    spy.assert_not_called()
    assert COL_DISPOSITION_MISSING_ENTITIES not in result.columns
    assert COL_RETAINED_PROTECTED_ENTITIES not in result.columns
    assert result[COL_NEEDS_HUMAN_REVIEW].tolist() == [False, False]


@pytest.mark.parametrize("dtype", ["bool", "bool[pyarrow]"])
def test_review_gates_keep_needs_human_review_dtype(dtype: str) -> None:
    disposition = {"sensitivity_disposition": [{"entity_value": "Alice", "protection_method_suggestion": "replace"}]}
    df = pd.DataFrame(
        {
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Bob", "labels": ["first_name"]}]}, None],
            COL_SENSITIVITY_DISPOSITION: [disposition, disposition],
            COL_RETAINED_PROTECTED_ENTITIES: [[], [["Alice", "first_name", "replace"]]],
            COL_NEEDS_HUMAN_REVIEW: pd.Series([False, False], dtype=dtype),
        }
    )
    out = apply_review_gates(df)
    assert str(out[COL_NEEDS_HUMAN_REVIEW].dtype) == dtype
    assert out[COL_NEEDS_HUMAN_REVIEW].tolist() == [True, True]
