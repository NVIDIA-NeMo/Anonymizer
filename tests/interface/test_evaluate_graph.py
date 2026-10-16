# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation re-enters canonical execution without reusing historical runtime IDs."""

from __future__ import annotations

import pickle

import pandas as pd
import pytest

from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.detection._frame_publication import InvocationFrame
from anonymizer.engine.graph_sdk.executor import ExecutionResult
from anonymizer.engine.replace.replace_runner import ReplacementResult
from anonymizer.interface.results import AnonymizerResult, PreviewResult
from tests.interface.test_anonymizer_interface import _make_anonymizer


@pytest.fixture
def evaluation_dd(tmp_path, monkeypatch):
    from tests.engine.test_detection_source_frame import scripted_detection

    yield from getattr(scripted_detection, "__wrapped__")(tmp_path, monkeypatch)


@pytest.mark.parametrize("saved", [False, True])
def test_evaluate_owns_fresh_execution_and_retains_live_input(saved):
    trace = pd.DataFrame({"biography": ["Alice"], "biography_replaced": ["hidden"]})
    prior = object()
    output = AnonymizerResult(trace, trace, "biography", [], replace_method=Redact())
    output._execution = prior
    if saved:
        output = PreviewResult(trace, trace, "biography", [], 1, replace_method=Redact())
        output._execution = prior
        output = pickle.loads(pickle.dumps(output))
        assert output._execution is None
    anonymizer, _, replacement, _ = _make_anonymizer()
    replacement.evaluate.return_value = ReplacementResult(
        pd.DataFrame({COL_TEXT: ["Alice"], COL_REPLACED_TEXT: ["hidden"], "entity_coverage": [1.0]}), []
    )
    result = anonymizer.evaluate(output)
    assert type(result) is AnonymizerResult
    assert isinstance(result._execution, ExecutionResult)
    assert result._execution is not output._execution
    assert result._execution._execution.context.prepared.configuration.purpose == "execution_only"
    assert result._execution._execution.context.prepared.declared_request_upper_bound is None
    assert result._execution.requests.dispatches == ()
    assert result._execution.final_outputs == ()
    frame = result._execution.frames[0]
    assert isinstance(frame, InvocationFrame) and isinstance(frame.retained_owner, tuple)
    assert frame.retained_owner[0] is output._execution
    assert replacement.evaluate.call_count == 1
    assert replacement.evaluate.call_args.kwargs["replace_method"] is output.replace_method
    assert result.dataframe["biography_replaced"].tolist() == ["hidden"]
    assert result.dataframe["entity_coverage"].tolist() == [1.0]
    assert pickle.loads(pickle.dumps(result))._execution is None
    again = anonymizer.evaluate(result)
    assert isinstance(again._execution, ExecutionResult)
    assert again._execution is not result._execution
    retained = again._execution.frames[0]
    assert isinstance(retained, InvocationFrame) and isinstance(retained.retained_owner, tuple)
    assert retained.retained_owner[0] is result._execution
    assert not again._execution._execution.context.prepared.bound_inputs


def test_injected_evaluate_preserves_caller_context_and_exception_identity():
    import asyncio
    import sqlite3
    from contextvars import ContextVar
    from threading import get_ident

    trace = pd.DataFrame({"text": ["Alice"], "text_replaced": ["hidden"]})
    output = AnonymizerResult(trace, trace, "text", [], replace_method=Redact())
    anonymizer, _, replacement, _ = _make_anonymizer()
    context = ContextVar("evaluate_context", default="missing")
    context.set("caller")
    connection = sqlite3.connect(":memory:")
    thread = get_ident()
    error = ValueError("literal injected judge failure")

    def evaluate(frame, **kwargs):
        assert get_ident() == thread and context.get() == "caller"
        assert connection.execute("select 17").fetchone() == (17,)
        assert set(kwargs) == {
            "replace_method",
            "model_configs",
            "selected_models",
            "entity_labels",
            "excluded_entity_labels",
            "compute_detection_validity",
            "data_summary",
        }
        raise error

    replacement.evaluate.side_effect = evaluate

    async def active_loop():
        with pytest.raises(ValueError) as caught:
            anonymizer.evaluate(output)
        assert caught.value is error
        execution = getattr(error, "_anonymizer_execution")
        assert isinstance(execution, ExecutionResult)
        assert not execution.frames

    try:
        asyncio.run(active_loop())
    finally:
        connection.close()


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_rewrite_evaluation_interrupt_does_not_start_coverage(monkeypatch, error_type):
    from anonymizer.config.anonymizer_config import Rewrite
    from anonymizer.interface import anonymizer as facade

    trace = pd.DataFrame({"text": ["Alice"], "text_rewritten": ["hidden"]})
    output = AnonymizerResult(trace, trace, "text", [], rewrite_config=Rewrite().privacy_goal)
    anonymizer, _, _, rewrite = _make_anonymizer()
    error = error_type("literal caller interruption")
    rewrite.evaluate.side_effect = error
    coverage = []
    monkeypatch.setattr(facade, "EntityCoverageWorkflow", lambda **kwargs: coverage.append(kwargs))
    with pytest.raises(error_type) as caught:
        anonymizer.evaluate(output)
    assert caught.value is error
    assert coverage == []
    assert isinstance(error._anonymizer_execution, ExecutionResult)
    assert [entry.status for entry in error._anonymizer_execution.invocation_state.entries] == ["cancelled"]
    assert error._anonymizer_execution.record.terminals[0].reasons == frozenset({"cancel_requested"})


@pytest.mark.parametrize(
    "strategy,validity,drop",
    [
        ("Redact", False, False),
        ("Empty", False, False),
        ("Hash", True, False),
        ("Annotate", False, True),
        ("Substitute", True, False),
        ("Rewrite", True, False),
        ("Rewrite", True, True),
        ("Substitute", True, True),
    ],
)
def test_saved_evaluate_real_dd_preserves_rows_and_metrics(evaluation_dd, monkeypatch, strategy, validity, drop):
    import json

    import httpx

    from anonymizer.config.anonymizer_config import EvaluateConfig, Rewrite
    from anonymizer.config.models import EvaluateModelSelection
    from anonymizer.config.replace_strategies import Annotate, Hash, Substitute
    from anonymizer.engine.constants import (
        COL_DETECTION_JUDGE,
        COL_ENTITIES_BY_VALUE,
        COL_JUDGE_EVALUATION,
        COL_REPLACEMENT_MAP,
    )
    from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN
    from tests.engine.test_detection_source_frame import public_anonymizer

    calls = []
    transport = httpx.MockTransport

    def respond(request):
        body = json.loads(request.content)
        messages = json.dumps(body["messages"])
        calls.append(messages)
        if drop and "Broken" in messages:
            return httpx.Response(
                400, json={"error": {"message": "literal judge drop", "type": "invalid_request_error"}}
            )
        content = {
            "candidate_entities": [{"value": "Alice", "label": "first_name", "reasoning": "literal person"}]
            if "Alice" in messages
            else [],
            "all_valid": True,
            "invalid_entities": [],
            "invalid_replacements": [],
            "all_consistent": True,
            "relations": [],
            "entities": [],
            **{name: {"score": "high", "reasoning": "literal judge"} for name in ("privacy", "quality", "style")},
        }
        return httpx.Response(
            200,
            json={
                "id": "literal",
                "object": "chat.completion",
                "created": 0,
                "model": "singleton",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "```json\n" + json.dumps(content) + "\n```"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    monkeypatch.setattr(httpx, "MockTransport", lambda ignored: transport(respond))
    anonymizer = public_anonymizer(evaluation_dd)
    anonymizer._selected_models.evaluate = EvaluateModelSelection(
        **dict.fromkeys(EvaluateModelSelection.model_fields, "singleton")
    )
    anonymizer._selected_models.replace.replacement_generator = "singleton"
    rewritten = strategy == "Rewrite"
    trace = pd.DataFrame(
        {
            "text": ["Alice", "Broken" if drop else "ordinary"],
            "text_rewritten" if rewritten else "text_replaced": ["hidden", "ordinary"],
            COL_ENTITIES_BY_VALUE: [
                {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]},
                {"entities_by_value": []},
            ],
            COL_REPLACEMENT_MAP: [
                {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Beth"}]},
                {"replacements": []},
            ],
            RECORD_ID_COLUMN: ["drop" if drop and strategy in {"Rewrite", "Substitute"} else "keep", "drop"],
            "extra": [17, 29],
        }
    )
    if strategy == "Empty":
        trace = trace.iloc[:0]
    output = AnonymizerResult(
        trace,
        trace,
        "text",
        [],
        replace_method=None
        if rewritten
        else {"Empty": Redact, "Redact": Redact, "Hash": Hash, "Annotate": Annotate, "Substitute": Substitute}[
            strategy
        ](),
        rewrite_config=Rewrite().privacy_goal if rewritten else None,
        entity_labels=["first_name"],
        excluded_entity_labels=["email"],
        data_summary="literal saved source",
    )
    loaded = pickle.loads(pickle.dumps(output))
    result = anonymizer.evaluate(loaded, config=EvaluateConfig(compute_detection_validity=validity))
    if strategy == "Empty":
        assert result.dataframe.empty and result.failed_records == []
        assert isinstance(result._execution, ExecutionResult)
        assert result._execution.frames[0].row_counts == (0, 0)
        assert len(calls) == 1 and "Hello!" in calls[0]
        return
    assert result.dataframe["text"].tolist() == ["Alice", "Broken" if drop else "ordinary"]
    assert result.trace_dataframe["extra"].tolist() == [17, 29]
    assert result.dataframe["entity_coverage"].iloc[0] == 1.0
    assert bool(pd.isna(result.dataframe["entity_coverage"].iloc[1])) is drop
    assert bool(result.failed_records) is drop
    assert all(failure.record_id == "drop" for failure in result.failed_records)
    if validity:
        assert result.dataframe["detection_valid"].tolist() == [True, True]
    if strategy == "Substitute":
        for column in ("type_fidelity_valid", "relational_consistency_valid", "attribute_fidelity_valid"):
            assert result.dataframe[column].tolist() == [True, True]
    if rewritten:
        assert result.trace_dataframe[COL_DETECTION_JUDGE].iloc[0]["all_valid"] is True
        assert result.dataframe[COL_JUDGE_EVALUATION].iloc[0]["privacy"]["score"] == "high"
        assert pd.isna(result.dataframe[COL_JUDGE_EVALUATION].iloc[1])
    assert len(calls) >= 2
    assert isinstance(result._execution, ExecutionResult)
    assert result._execution.frames[0].row_counts == (2, 2)
    assert not result._execution.requests.dispatches
    assert result.entity_labels == ["first_name"] and result.excluded_entity_labels == ["email"]
    assert result.data_summary == "literal saved source"
