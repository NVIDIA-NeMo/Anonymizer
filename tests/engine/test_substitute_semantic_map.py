# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replacement map control, opaque callbacks and accepted publication boundaries."""

from __future__ import annotations

import json
from dataclasses import replace
from threading import current_thread, main_thread
from typing import Any
from unittest.mock import Mock

import httpx
import numpy as np
import pandas as pd
import pytest

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.detection._frame_publication import FramePublication, InvocationFrame, PublishedFrame
from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk._execution_delegated import source_worker_borrow
from anonymizer.engine.ndd.adapter import FailedRecord, WorkflowRunResult
from anonymizer.engine.replace._semantic_map import MapBinding
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceResult, LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from tests.engine import test_detection_source_frame as detection_tests

scripted_detection = detection_tests.scripted_detection

pytestmark = pytest.mark.compat


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
            COL_FINAL_ENTITIES: [
                {"entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]}
            ],
            "number": pd.Series([pd.NA], dtype="Int64"),
            "category": pd.Categorical(["kept"]),
        }
    )


def _raw() -> dict[str, Any]:
    return {
        "replacements": [
            {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
            {"original": "Unrequested", "label": "first_name", "synthetic": "Nora"},
        ]
    }


@pytest.mark.compat
def test_retained_map_source_reordering_rejects_before_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    source = pd.concat([_source(), _source()], ignore_index=True)
    adapter = Mock()
    run = MapBinding.run

    def reorder(
        self: MapBinding, receiver: Any, dataframe: pd.DataFrame, arguments: dict[str, Any]
    ) -> LlmReplaceResult:
        object.__setattr__(self.source, "associations", tuple(reversed(self.source.associations)))
        return run(self, receiver, dataframe, arguments)

    monkeypatch.setattr(MapBinding, "run", reorder)
    with pytest.raises(EffectRejected) as rejected:
        LlmReplaceWorkflow(adapter).generate_map_only(
            source, model_configs=[], selected_models=ReplaceModelSelection(replacement_generator="test")
        )
    assert rejected.value.code.value == "foreign_owner"
    adapter.run_workflow.assert_not_called()


@pytest.mark.parametrize(
    "raw", [pd.NA, np.int64(1), {"replacements": np.array(_raw()["replacements"], dtype=object), "ignored": object()}]
)
def test_maps_preserve_permissive_payloads_and_extension_dtypes(raw: Any) -> None:
    source = _source()
    adapter = Mock()

    def returned(rows: pd.DataFrame, **kwargs: Any) -> WorkflowRunResult:
        output = rows.copy()
        output[COL_REPLACEMENT_MAP] = [raw]
        return WorkflowRunResult(output, [], (0,))

    adapter.run_workflow.side_effect = returned
    result = LlmReplaceWorkflow(adapter).generate_map_only(
        source, model_configs=[], selected_models=ReplaceModelSelection(replacement_generator="test")
    )
    assert result.dataframe.number.dtype == source.number.dtype
    assert result.dataframe.category.dtype == source.category.dtype
    assert result.dataframe[COL_REPLACEMENT_MAP].iloc[0] == (
        {"replacements": [_raw()["replacements"][0]]} if isinstance(raw, dict) else {"replacements": []}
    )
    assert result._execution is not None and isinstance(result._execution.frames[0], PublishedFrame)


def test_direct_delegating_map_and_replacement_keep_one_owner() -> None:
    source = _source()
    calls = []
    owners = []
    adapter = Mock()

    def returned(rows: pd.DataFrame, **kwargs: Any) -> WorkflowRunResult:
        owners.append(source_worker_borrow().owner)
        output = rows.copy()
        output[COL_REPLACEMENT_MAP] = [_raw()]
        return WorkflowRunResult(output, [], (0,))

    adapter.run_workflow.side_effect = returned

    class DelegatingMap(LlmReplaceWorkflow):
        def generate_map_only(self, dataframe: pd.DataFrame, **kwargs: Any) -> LlmReplaceResult:
            rows = dataframe
            calls.append((rows, current_thread(), source_worker_borrow().owner))
            return super().generate_map_only(rows, **kwargs)

    result = ReplacementWorkflow(llm_workflow=DelegatingMap(adapter)).run(
        source,
        replace_method=Substitute(),
        model_configs=[],
        selected_models=ReplaceModelSelection(replacement_generator="test"),
    )
    assert len(calls) == 1 and calls[0][0] is source and calls[0][1] is main_thread()
    assert owners == [calls[0][2]]
    assert result.dataframe[COL_REPLACED_TEXT].tolist() == ["Maya"]
    assert result._execution is not None and isinstance(result._execution.frames[0], PublishedFrame)


def test_direct_opaque_map_keeps_callback_frame_failures_and_thread() -> None:
    source = _source()
    calls = []
    failure = FailedRecord("record", "custom-map", "reason")

    class OpaqueMap(LlmReplaceWorkflow):
        def generate_map_only(self, dataframe: pd.DataFrame, **kwargs: Any) -> LlmReplaceResult:
            rows = dataframe
            calls.append((rows, current_thread(), kwargs))
            output = rows.copy()
            output[COL_REPLACEMENT_MAP] = [_raw()]
            return LlmReplaceResult(output, [failure])

    models = []
    selected = ReplaceModelSelection(replacement_generator="test")
    result = ReplacementWorkflow(llm_workflow=OpaqueMap(Mock())).run(
        source,
        replace_method=Substitute(instructions="keep"),
        model_configs=models,
        selected_models=selected,
        preview_num_records=4,
    )
    assert len(calls) == 1 and calls[0][0] is source and calls[0][1] is main_thread()
    assert calls[0][2]["model_configs"] is models and calls[0][2]["selected_models"] is selected
    assert calls[0][2]["instructions"] == "keep" and calls[0][2]["preview_num_records"] == 4
    assert result.failed_records == [failure] and result.dataframe[COL_REPLACED_TEXT].tolist() == ["Maya"]
    assert result._execution is not None and isinstance(result._execution.frames[0], InvocationFrame)


def test_raw_generation_cannot_be_substituted_for_accepted_map_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    generated = []
    publish = MapBinding.publish
    capture = FramePublication.capture.__func__

    def observed(self: MapBinding, item: Any, **kwargs: Any) -> Any:
        result = publish(self, item, **kwargs)
        if kwargs.get("raw_map") is not None:
            generated.append((item.association, kwargs["raw_map"]))
        return result

    def changed(cls: Any, *args: Any, **kwargs: Any) -> Any:
        publication = capture(cls, *args, **kwargs)
        if generated:
            association, value = generated[0]
            object.__setattr__(publication, "associations", (association,))
            object.__setattr__(publication, "_text", ((COL_REPLACEMENT_MAP, "raw_map", (value,)),))
        return publication

    adapter = Mock()

    def returned(rows: pd.DataFrame, **kwargs: Any) -> WorkflowRunResult:
        output = rows.copy()
        output[COL_REPLACEMENT_MAP] = [_raw()]
        return WorkflowRunResult(output, [], (0,))

    adapter.run_workflow.side_effect = returned
    monkeypatch.setattr(MapBinding, "publish", observed)
    monkeypatch.setattr(FramePublication, "capture", classmethod(changed))
    with pytest.raises(RuntimeError, match="semantic publication"):
        LlmReplaceWorkflow(adapter).generate_map_only(
            _source(), model_configs=[], selected_models=ReplaceModelSelection(replacement_generator="test")
        )


@pytest.mark.parametrize("mutation", ["copy", "frame", "entity", "args", "adapter", "replay"])
def test_map_binding_rejects_copied_stale_and_replayed_entries(monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    adapter = Mock()

    def returned(rows: pd.DataFrame, **kwargs: Any) -> WorkflowRunResult:
        output = rows.copy()
        output[COL_REPLACEMENT_MAP] = [_raw()]
        return WorkflowRunResult(output, [], (0,))

    adapter.run_workflow.side_effect = returned
    run = MapBinding.run

    def changed(self: MapBinding, receiver: Any, frame: pd.DataFrame, args: dict[str, Any]) -> LlmReplaceResult:
        if mutation == "copy":
            self = replace(self)
        elif mutation == "frame":
            frame = frame.copy()
        elif mutation == "entity":
            frame[COL_ENTITIES_BY_VALUE].iloc[0]["entities_by_value"][0]["value"] = "Changed"
        elif mutation == "args":
            args = {**args, "instructions": "changed"}
        elif mutation == "adapter":
            receiver._adapter = Mock()
        result = run(self, receiver, frame, args)
        return run(self, receiver, frame, args) if mutation == "replay" else result

    monkeypatch.setattr(MapBinding, "run", changed)
    with pytest.raises(EffectRejected):
        LlmReplaceWorkflow(adapter).generate_map_only(
            _source(), model_configs=[], selected_models=ReplaceModelSelection(replacement_generator="test")
        )
    assert adapter.run_workflow.call_count == (1 if mutation == "replay" else 0)


@pytest.mark.parametrize("mutation", ["declaration", "map_result"])
def test_mutated_publication_or_map_cannot_authorize_downstream_work(mutation: str) -> None:
    source = _source()
    adapter = Mock()

    def returned(rows: pd.DataFrame, **kwargs: Any) -> WorkflowRunResult:
        output = rows.copy()
        output[COL_REPLACEMENT_MAP] = [_raw()]
        return WorkflowRunResult(output, [], (0,))

    adapter.run_workflow.side_effect = returned

    class ChangedMap(LlmReplaceWorkflow):
        def generate_map_only(self, dataframe: pd.DataFrame, **kwargs: Any) -> LlmReplaceResult:
            if mutation == "declaration":
                owner = source_worker_borrow().owner
                producer = next(
                    policy.node
                    for policy in owner.runtime.policies.values()
                    if policy.implementations[0].capability.operation.name == "replacement-generate"
                )
                object.__setattr__(owner.contract, "publication_sources", ((producer, frozenset({"raw_map"})),))
            result = super().generate_map_only(dataframe, **kwargs)
            if mutation == "map_result":
                result.dataframe[COL_REPLACEMENT_MAP].iloc[0]["replacements"].append(_raw()["replacements"][1])
            return result

    with pytest.raises(EffectRejected):
        ReplacementWorkflow(llm_workflow=ChangedMap(adapter)).run(
            source,
            replace_method=Substitute(),
            model_configs=[],
            selected_models=ReplaceModelSelection(replacement_generator="test"),
        )
    assert adapter.run_workflow.call_count == (1 if mutation == "map_result" else 0)


@pytest.mark.parametrize("mutation", ["child_slots", "readiness", "request_policy"])
@pytest.mark.parametrize("after_http", [False, True], ids=["before-effects", "after-http"])
def test_retained_observed_request_mutation_rejects_and_closes_resources(
    scripted_detection: Any,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    after_http: bool,
) -> None:
    from anonymizer.engine.ndd._graph_requests import NddPhysicalTransport
    from anonymizer.engine.ndd.adapter import NddAdapter
    from anonymizer.interface.errors import AnonymizerWorkflowError
    from tests.engine.test_detection_graph_publication import _modify_dd_responses

    designer, models, _, _ = scripted_detection
    resources: list[NddPhysicalTransport] = []
    owners: list[Any] = []
    http_calls: list[bool] = []
    initialize = NddPhysicalTransport.__init__
    run = MapBinding.run

    def capture_resource(self: NddPhysicalTransport) -> None:
        initialize(self)
        resources.append(self)

    def mutate() -> None:
        contract = owners[0].contract
        if mutation == "child_slots":
            object.__setattr__(contract, "child_slots", (999,))
        elif mutation == "readiness":
            object.__setattr__(contract.readiness, "request_slot", 999)
        else:
            object.__setattr__(contract.observed_request, "max_attempts", contract.observed_request.max_attempts + 1)

    def enter(self: MapBinding, receiver: Any, frame: pd.DataFrame, arguments: dict[str, Any]) -> LlmReplaceResult:
        owners.append(source_worker_borrow().owner)
        if not after_http:
            mutate()
        return run(self, receiver, frame, arguments)

    def respond(request: Any, answer: Any) -> Any:
        body = json.loads(request.content)
        is_map = "Generate synthetic replacements" in json.dumps(body["messages"])
        http_calls.append(is_map)
        if not is_map:
            return answer
        payload = answer.json()
        payload["choices"][0]["message"]["content"] = json.dumps(_raw())
        if after_http:
            mutate()
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(NddPhysicalTransport, "__init__", capture_resource)
    monkeypatch.setattr(MapBinding, "run", enter)
    _modify_dd_responses(scripted_detection, monkeypatch, respond)
    source = _source().assign(**{COL_TAGGED_TEXT: ["<first_name>Alice</first_name>"]})
    with pytest.raises((EffectRejected, AnonymizerWorkflowError)) as rejected:
        LlmReplaceWorkflow(NddAdapter(designer)).generate_map_only(
            source, model_configs=models, selected_models=ReplaceModelSelection(replacement_generator="singleton")
        )
    execution = getattr(rejected.value, "_anonymizer_execution")
    assert not execution.frames and not execution.final_outputs
    assert len(resources) == 1 and resources[0].closed
    assert all(client._inner._closed for client in resources[0].clients)
    assert execution.requests.dispatched_count == len(execution.requests.settlements) == (2 if after_http else 0)
    assert http_calls == ([False, True] if after_http else [])
    assert not execution.requests.remote_outstanding and not execution.requests.local_in_flight
    assert all(state.complete for state in execution.states)
    assert execution.invocation_state is None or execution.invocation_state.complete
