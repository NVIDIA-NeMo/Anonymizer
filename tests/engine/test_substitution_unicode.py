# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_ENTITIES_FOR_REPLACE,
    COL_FINAL_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_APPLICATION,
    COL_REPLACEMENT_MAP,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, WorkflowRunResult
from anonymizer.engine.replace.llm_replace_workflow import (
    LlmReplaceWorkflow,
    _filter_replacement_map_to_input_entities,
)
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.engine.schemas import EntitiesByValueSchema, EntityReplacementMapSchema

_LABEL = "company_name"
_NARROW = "Acme\u202fCorp"
_THIN = "Acme\u2009Corp"


def _entities(pairs: list[tuple[str, str]]) -> EntitiesByValueSchema:
    grouped: dict[str, list[str]] = {}
    for value, label in pairs:
        grouped.setdefault(value, []).append(label)
    return EntitiesByValueSchema.model_validate(
        {"entities_by_value": [{"value": value, "labels": labels} for value, labels in grouped.items()]}
    )


def _source(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    spans = []
    cursor = 0
    for value, label in pairs:
        spans.append({"value": value, "label": label, "start_position": cursor, "end_position": cursor + len(value)})
        cursor += len(value) + 3
    text = " | ".join(value for value, _ in pairs) + "."
    return pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["unicode-row"],
            COL_TEXT: [text],
            COL_TAGGED_TEXT: [text],
            COL_FINAL_ENTITIES: [{"entities": spans}],
            COL_ENTITIES_BY_VALUE: [_entities(pairs).model_dump()],
        }
    )


def _run(
    source: pd.DataFrame,
    responses: list[list[dict[str, str]]],
    model_configs: list[ModelConfig],
    selected_models: ReplaceModelSelection,
) -> tuple[ReplacementResult, list[pd.DataFrame]]:
    calls: list[pd.DataFrame] = []

    def generate(frame: pd.DataFrame, **_: object) -> WorkflowRunResult:
        calls.append(frame.copy())
        output = frame.copy()
        entries = responses[min(len(calls) - 1, len(responses) - 1)]
        output[COL_REPLACEMENT_MAP] = [{"replacements": entries} for _ in range(len(output))]
        return WorkflowRunResult(dataframe=output, failed_records=[])

    adapter = Mock()
    adapter.run_workflow.side_effect = generate
    result = ReplacementWorkflow(llm_workflow=LlmReplaceWorkflow(adapter=adapter)).run(
        source,
        replace_method=Substitute(),
        model_configs=model_configs,
        selected_models=selected_models,
    )
    return result, calls


@pytest.mark.parametrize(
    ("requested", "echoed"),
    [
        pytest.param(_NARROW, _THIN, id="narrow-to-thin"),
        pytest.param("Acme Corp", "Acme\u00a0Corp", id="ascii-to-no-break"),
        pytest.param("Acme\u00a0Corp", "Acme Corp", id="no-break-to-ascii"),
    ],
)
def test_unicode_space_recovery_preserves_source_spans_and_synthetic_without_retry(
    requested: str,
    echoed: str,
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = _source([(requested, _LABEL)])
    snapshot = copy.deepcopy(source.to_dict(orient="list"))
    synthetic = "Nova\u202fLabs"
    with caplog.at_level(logging.DEBUG, logger="anonymizer"):
        result, calls = _run(
            source,
            [[{"original": echoed, "label": _LABEL, "synthetic": synthetic}]],
            stub_model_configs,
            stub_replace_model_selection,
        )

    assert source.to_dict(orient="list") == snapshot
    assert len(calls) == 1
    assert result.failed_records == []
    row = result.dataframe.iloc[0]
    assert row[COL_TEXT] == source[COL_TEXT].iloc[0]
    assert row[COL_FINAL_ENTITIES] == snapshot[COL_FINAL_ENTITIES][0]
    assert row[COL_REPLACED_TEXT] == synthetic + "."
    assert row[COL_REPLACEMENT_MAP] == {
        "replacements": [{"original": requested, "label": _LABEL, "synthetic": synthetic}]
    }
    assert row[COL_REPLACEMENT_APPLICATION]["applied_span_count"] == 1
    assert row[COL_REPLACEMENT_APPLICATION]["skipped_span_count"] == 0
    for value in [requested, echoed, synthetic, "Acme", "Nova"]:
        assert value not in caplog.text


def test_reconciliation_uses_label_when_other_values_share_the_comparison_key() -> None:
    raw = {"replacements": [{"original": "Acme Corp", "label": _LABEL, "synthetic": "Nova Labs"}]}
    result = _filter_replacement_map_to_input_entities(
        raw_map=raw,
        parsed_entities=_entities([(_NARROW, _LABEL), (_THIN, "organization")]),
        reconcile_unicode_whitespace=True,
    )
    assert result == {"replacements": [{"original": _NARROW, "label": _LABEL, "synthetic": "Nova Labs"}]}


@pytest.mark.parametrize("model_response", [False, True], ids=["dict", "pydantic-model"])
def test_reconciliation_does_not_mutate_raw_response_or_requested_inventory(model_response: bool) -> None:
    payload = {"replacements": [{"original": _THIN, "label": _LABEL, "synthetic": "Nova Labs"}]}
    raw = EntityReplacementMapSchema.model_validate(payload) if model_response else payload
    requested = _entities([(_NARROW, _LABEL)])
    before_payload = copy.deepcopy(payload)
    before_requested = requested.model_dump()
    result = _filter_replacement_map_to_input_entities(
        raw_map=raw, parsed_entities=requested, reconcile_unicode_whitespace=True
    )
    assert result["replacements"][0]["original"] == _NARROW
    assert payload == before_payload
    if isinstance(raw, EntityReplacementMapSchema):
        assert raw.model_dump() == before_payload
    assert requested.model_dump() == before_requested


@pytest.mark.parametrize("exact_first", [False, True], ids=["exact-later", "exact-first"])
def test_exact_mapping_takes_precedence_over_reconciled_mapping(exact_first: bool) -> None:
    exact = {"original": _NARROW, "label": _LABEL, "synthetic": "Exact Labs"}
    approximate = {"original": _THIN, "label": _LABEL, "synthetic": "Other Labs"}
    result = _filter_replacement_map_to_input_entities(
        raw_map={"replacements": [exact, approximate] if exact_first else [approximate, exact]},
        parsed_entities=_entities([(_NARROW, _LABEL)]),
        reconcile_unicode_whitespace=True,
    )
    assert result == {"replacements": [exact]}


def test_exact_match_survives_ambiguous_whitespace_comparison_key() -> None:
    entry = {"original": _NARROW, "label": _LABEL, "synthetic": "Nova Labs"}
    result = _filter_replacement_map_to_input_entities(
        raw_map={"replacements": [entry]},
        parsed_entities=_entities([(_NARROW, _LABEL), (_THIN, _LABEL)]),
        reconcile_unicode_whitespace=True,
    )
    assert result == {"replacements": [entry]}


@pytest.mark.parametrize(
    ("echoed", "label"),
    [
        pytest.param("acme\u2009Corp", _LABEL, id="case"),
        pytest.param("Acme\u2009\u2009Corp", _LABEL, id="space-count"),
        pytest.param(" Acme\u2009Corp", _LABEL, id="leading-space"),
        pytest.param("Acme\u2009Corp ", _LABEL, id="trailing-space"),
        pytest.param("Acme\tCorp", _LABEL, id="tab"),
        pytest.param("Acme\nCorp", _LABEL, id="newline"),
        pytest.param("Acme\u2003Corp", _LABEL, id="unlisted-em-space"),
        pytest.param("Acme\u200bCorp", _LABEL, id="zero-width-space"),
        pytest.param("\uff21cme\u2009Corp", _LABEL, id="compatibility-character"),
        pytest.param("AcmeCorp", _LABEL, id="deleted-space"),
        pytest.param(_THIN, "organization", id="different-label"),
        pytest.param(_THIN, "Company_Name", id="label-case"),
    ],
)
def test_reconciliation_does_not_accept_differences_outside_allowlisted_space_variants(echoed: str, label: str) -> None:
    result = _filter_replacement_map_to_input_entities(
        raw_map={"replacements": [{"original": echoed, "label": label, "synthetic": "Nova Labs"}]},
        parsed_entities=_entities([(_NARROW, _LABEL)]),
        reconcile_unicode_whitespace=True,
    )
    assert result == {"replacements": []}


def test_ambiguous_whitespace_mapping_retries_once_then_fails(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    result, calls = _run(
        _source([(_NARROW, _LABEL), (_THIN, _LABEL)]),
        [[{"original": "Acme Corp", "label": _LABEL, "synthetic": "Nova Labs"}]],
        stub_model_configs,
        stub_replace_model_selection,
    )
    assert len(calls) == 2
    assert all(len(frame[COL_ENTITIES_FOR_REPLACE].iloc[0]) == 2 for frame in calls)
    assert result.dataframe.empty
    assert [failure.record_id for failure in result.failed_records] == ["unicode-row"]
    assert "2 requested pair(s)" in result.failed_records[0].reason
    assert "Acme" not in result.failed_records[0].reason


def test_retry_reconciliation_checks_full_original_inventory_not_only_missing_pairs(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    result, calls = _run(
        _source([(_NARROW, _LABEL), (_THIN, _LABEL)]),
        [
            [{"original": _NARROW, "label": _LABEL, "synthetic": "Nova Labs"}],
            [{"original": "Acme Corp", "label": _LABEL, "synthetic": "Orion Media"}],
        ],
        stub_model_configs,
        stub_replace_model_selection,
    )
    assert len(calls) == 2
    assert calls[1][COL_ENTITIES_FOR_REPLACE].iloc[0] == [{"value": _THIN, "labels": [_LABEL], "labels_str": _LABEL}]
    assert result.dataframe.empty
    assert len(result.failed_records) == 1
    assert "1 requested pair(s)" in result.failed_records[0].reason


def test_whitespace_reconciliation_can_complete_the_retry(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    result, calls = _run(
        _source([(_NARROW, _LABEL)]),
        [[], [{"original": _THIN, "label": _LABEL, "synthetic": "Nova Labs"}]],
        stub_model_configs,
        stub_replace_model_selection,
    )
    assert len(calls) == 2
    assert result.failed_records == []
    assert result.dataframe[COL_REPLACED_TEXT].tolist() == ["Nova Labs."]
    assert result.dataframe[COL_REPLACEMENT_MAP].iloc[0]["replacements"][0]["original"] == _NARROW


def test_rewrite_map_generation_keeps_existing_exact_match_behavior(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    def generate(frame: pd.DataFrame, **_: object) -> WorkflowRunResult:
        output = frame.copy()
        output[COL_REPLACEMENT_MAP] = [
            {"replacements": [{"original": _THIN, "label": _LABEL, "synthetic": "Nova Labs"}]}
        ]
        return WorkflowRunResult(dataframe=output, failed_records=[])

    adapter = Mock()
    adapter.run_workflow.side_effect = generate
    result = LlmReplaceWorkflow(adapter=adapter).generate_map_only(
        _source([(_NARROW, _LABEL)]),
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )
    adapter.run_workflow.assert_called_once()
    assert result.dataframe[COL_REPLACEMENT_MAP].iloc[0] == {"replacements": []}


def test_default_filter_does_not_enable_unicode_reconciliation() -> None:
    result = _filter_replacement_map_to_input_entities(
        raw_map={"replacements": [{"original": _THIN, "label": _LABEL, "synthetic": "Nova Labs"}]},
        parsed_entities=_entities([(_NARROW, _LABEL)]),
    )
    assert result == {"replacements": []}
