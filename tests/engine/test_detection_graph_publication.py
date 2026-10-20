# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distinct owner, setup and canonical publication faults for actual DD execution."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import httpx
import pandas as pd
import pytest

from anonymizer.engine.constants import COL_REPLACEMENT_MAP, COL_TEXT
from anonymizer.engine.harness.kernel import retained_execution
from anonymizer.engine.ndd._graph_requests import NddPhysicalTransport, NddRequestBridge
from anonymizer.graph import EffectRejected
from tests.engine import test_detection_source_frame as detection_tests
from tests.engine.test_detection_source_frame import run_graph_profile
from tests.retained import retained

scripted_detection = detection_tests.scripted_detection

# Hang-guard bound for waits on normally-fast in-memory thread/future signals
# in this module. Not a measurement of expected runtime -- only a generous
# ceiling so a hung test fails instead of blocking CI, chosen to stay robust
# under a loaded CI host.
_HANG_GUARD_TIMEOUT = 30


def _source():
    return pd.DataFrame({COL_TEXT: ["Alice travels.", "Alice stays."], "extra": [17, 29]})


def _options():
    return {"gliner_detection_threshold": 0.3, "tag_latent_entities": False, "preview_num_records": 2}


@pytest.mark.parametrize("fault", ["foreign_physical", "ineligible_client"])
def test_resource_rejection_precedes_http_and_closes_created_clients(scripted_detection, monkeypatch, fault):
    physicals = []
    initialize = NddPhysicalTransport.__init__
    bind = NddRequestBridge.__init__
    register = NddPhysicalTransport.register

    def initialized(self):
        initialize(self)
        physicals.append(self)

    def bridged(self, requests, slots, physical):
        if fault == "foreign_physical":

            async def create():
                return NddPhysicalTransport()

            physical = asyncio.run_coroutine_threadsafe(create(), physical.loop).result(timeout=_HANG_GUARD_TIMEOUT)
        bind(self, requests, slots, physical)

    def registered(self, client):
        register(self, client)
        if fault == "ineligible_client":
            client._inner._retry_config = replace(client._inner._retry_config, max_retries=1)

    monkeypatch.setattr(NddPhysicalTransport, "__init__", initialized)
    monkeypatch.setattr(NddPhysicalTransport, "register", registered)
    monkeypatch.setattr(NddRequestBridge, "__init__", bridged)
    try:
        result = asyncio.run(run_graph_profile(scripted_detection, _source(), _options()))
        assert result.requests.dispatched_count == 0
        assert scripted_detection[3] == {}
        assert not result.final_outputs and not result.frames
        assert physicals[0].closed
        assert all(client._inner._closed for client in physicals[0].clients)
        assert len(physicals[0].clients) == (0 if fault == "foreign_physical" else 1)
        if fault == "foreign_physical":
            assert len(physicals) == 2 and not physicals[1].clients
    finally:
        for physical in physicals:
            if not physical.closed:
                asyncio.run_coroutine_threadsafe(physical.close(), physical.loop).result(timeout=_HANG_GUARD_TIMEOUT)


def test_generated_frame_projection_is_isolated_and_partial_publication_is_withheld(scripted_detection):
    result = asyncio.run(run_graph_profile(scripted_detection, _source(), _options()))
    projected = result.frames[0].project(result)
    assert projected[COL_REPLACEMENT_MAP].iloc[0]
    projected[COL_REPLACEMENT_MAP].iloc[0].clear()
    assert result.frames[0].project(result)[COL_REPLACEMENT_MAP].iloc[0]
    limited = asyncio.run(run_graph_profile(scripted_detection, _source(), _options(), artifact_limit=3))
    assert limited.requests.dispatched_count == 7
    assert not limited.final_outputs
    assert not limited.frames
    with pytest.raises(ValueError, match="different execution"):
        result.frames[0].project(limited)


@pytest.mark.parametrize("zero,sends", [(True, 5), (False, 8)])
def test_zero_or_variable_chunk_profile_qualifies_exact_selected_work(scripted_detection, zero, sends):
    source = pd.DataFrame(
        {COL_TEXT: ["No entities here.", "No entities there."] if zero else ["Alice travels.", "Alice Bob stays."]}
    )
    result = asyncio.run(
        run_graph_profile(
            scripted_detection,
            source,
            {**_options(), "validation_max_entities_per_call": 1},
            chunks=1 if zero else 2,
        )
    )
    assert result.requests.dispatched_count == sends
    from anonymizer.engine.constants import COL_REPLACED_TEXT
    from anonymizer.engine.harness.kernel import qualify_detection_redact

    assert len(result.final_outputs) == len(qualify_detection_redact(result).qualified) == 2
    assert result.frames[0].project(result)[COL_REPLACED_TEXT].tolist() == (
        source[COL_TEXT].tolist()
        if zero
        else ["[REDACTED_FIRST_NAME] travels.", "[REDACTED_FIRST_NAME] [REDACTED_FIRST_NAME] stays."]
    )
    assert sum(row.reasons == {"omitted_optional"} for row in result.record.terminals) == (2 if zero else 1)
    assert all(state.complete for state in result.states)
    assert result.invocation_state.complete


def test_profile_cancellation_drains_actual_dd_borrow_before_close(scripted_detection, monkeypatch):
    from threading import Event

    entered, release = Event(), Event()
    resources = []
    dispatch = NddPhysicalTransport.dispatch

    async def held(self, request):
        if not resources:
            resources.append(self)
            entered.set()
            await asyncio.to_thread(release.wait)
        return await dispatch(self, request)

    monkeypatch.setattr(NddPhysicalTransport, "dispatch", held)

    async def run():
        task = asyncio.create_task(run_graph_profile(scripted_detection, _source(), _options()))
        try:
            assert await asyncio.to_thread(entered.wait, _HANG_GUARD_TIMEOUT)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not resources[0].closed
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert resources[0].closed
            assert all(client._inner._closed for client in resources[0].clients)
        finally:
            release.set()

    asyncio.run(run())


@pytest.mark.parametrize("unknown", [False, True])
def test_unknown_remote_retains_registered_resource(scripted_detection, monkeypatch, unknown):
    designer, _, _, _ = scripted_detection
    original_resources = designer._create_resource_provider
    original_initialize = NddPhysicalTransport.__init__
    physicals = []
    actual_entries = []

    def initialize(self):
        original_initialize(self)
        physicals.append(self)

    monkeypatch.setattr(NddPhysicalTransport, "__init__", initialize)

    def resources(*args, **kwargs):
        provider = original_resources(*args, **kwargs)
        factory = provider.model_registry._model_facade_factory

        def facade(*args, **kwargs):
            result = factory(*args, **kwargs)
            if unknown:

                async def fail(request):
                    actual_entries.append(request.method)
                    raise httpx.ReadError("scripted uncertain response", request=request)

                result._client._inner._transport = httpx.MockTransport(fail)
            return result

        provider.model_registry._model_facade_factory = facade
        return provider

    monkeypatch.setattr(designer, "_create_resource_provider", resources)
    result = asyncio.run(run_graph_profile(scripted_detection, _source(), _options()))
    if unknown:
        assert len(physicals) == 1 and not physicals[0].closed
        assert all(not c._inner._closed for c in physicals[0].clients)
        assert len(result._retained_resources) == 1
        assert result._retained_resources[0].handle.owns(physicals[0])
        assert result.requests.dispatched_count == 1 and len(actual_entries) == 1
        assert len(result.requests.remote_outstanding) == 1
        assert not result.frames and not result.final_outputs
        assert ("sdk", "left_open") in {(f.owner, f.disposition) for f in result.cleanup}
    else:
        assert len(physicals) == 1 and physicals[0].closed
        assert all(c._inner._closed for c in physicals[0].clients)
        assert not result._retained_resources
        assert result.requests.dispatched_count == 7 and len(result.frames) == 1
        assert not result.requests.remote_outstanding

    # Test owner explicitly releases the scripted lost request after asserting retention.
    for physical in physicals:
        if not physical.closed:
            asyncio.run_coroutine_threadsafe(physical.close(), physical.loop).result(timeout=_HANG_GUARD_TIMEOUT)


def test_redact_check_records_candidate_mismatch_with_exact_span_dependencies(scripted_detection, monkeypatch):
    import anonymizer.engine.detection._graph_invocation as invocation
    from anonymizer.engine.constants import COL_REPLACED_TEXT

    original = invocation.apply_local_replace_strategy

    def changed(*args, **kwargs):
        returned = original(*args, **kwargs)
        returned.loc[returned.index[0], COL_REPLACED_TEXT] = "Alice travels."
        return returned

    monkeypatch.setattr(invocation, "apply_local_replace_strategy", changed)
    result = asyncio.run(run_graph_profile(scripted_detection, _source(), _options()))
    values = dict(result.artifacts)
    checks_by_target = {port.target: port for port in result.ports if port.port == "assessment"}
    checks = [checks_by_target[item.target] for item in result._execution.context.prepared.target_occurrences]
    assert sorted(values[port.artifact].text for port in checks) == [
        "span_redaction_matches",
        "span_redaction_mismatch",
    ]
    for check in checks:
        provenance = next(item for item in result.provenance if item.artifact == check.artifact)
        assert {getattr(parent, "port", None) for parent in provenance.parents} == {"source", "redacted", "spans"}
    assert result._execution.context.prepared.configuration.purpose == "protection"
    facts = {fact.activation: fact for fact in result.assessments}
    assert sorted(facts[port.activation].finding.status for port in checks) == ["satisfied", "unsatisfied"]


def test_real_redact_evidence_retains_source_candidate_and_span_versions(scripted_detection):
    from anonymizer.engine.graph_sdk.evidence import (
        AssessmentSubmission,
        admit_qualification,
        evidence_revision_view,
        evidence_validity,
        verify_evidence,
    )
    from tests.graph_sdk.evidence_fixtures import _qualification_limits

    result = asyncio.run(run_graph_profile(scripted_detection, _source(), _options()))
    admitted = admit_qualification(
        execution=result._execution,
        productions=result._execution.assessment_productions,
        limits=_qualification_limits(max_port_facts=64),
    )
    verified = verify_evidence(
        admitted=admitted,
        result=result,
        submissions=tuple(AssessmentSubmission(fact=fact) for fact in result.assessments),
    )
    assert len(verified) == 2
    for item in verified:
        assert {port for port, _ in item.consumed_by_port} == {"source", "candidate", "spans"}
        assert item.finding.status == "satisfied"
        assert len(item.reference.consumed) == 3
    assert len(admitted._projections) == 1
    assert admitted._projections[0].promise.consumed_ports == frozenset({"source"})

    def view(artifacts):
        fact = result.assessments[0]
        return evidence_revision_view(
            admitted=admitted,
            result=result,
            artifacts=artifacts,
            configurations=((fact.node, fact.environment.configuration),),
        )

    artifacts = tuple(reference for reference, _ in result.artifacts)
    current = view(artifacts)
    assert all(evidence_validity(evidence=item, current=current) == "current" for item in verified)
    from anonymizer.engine.harness.kernel import qualify_detection_redact

    qualified = qualify_detection_redact(result)
    assert len(qualified.qualified) == 2

    for port, consumed in verified[0].consumed_by_port:
        reference = consumed.artifact if hasattr(consumed, "artifact") else consumed
        missing = view(tuple(item for item in artifacts if item != reference))
        assert evidence_validity(evidence=verified[0], current=missing) == "unknown", port


@pytest.mark.parametrize("fault,code", [("missing_span", "missing"), ("wrong_summary", "contradictory")])
def test_redact_evidence_projection_rejects_incomplete_dependencies(scripted_detection, monkeypatch, fault, code):
    from anonymizer.engine.detection import _graph_profile
    from anonymizer.graph import ContractViolation

    original = _graph_profile._compile_graph

    def malformed(**kwargs):
        if fault == "missing_span":
            kwargs["input_bindings"] = tuple(
                item for item in kwargs["input_bindings"] if item.destination.port != "spans"
            )
        else:
            interface = kwargs["interface"]
            kwargs["interface"] = replace(
                interface,
                outcomes=tuple(
                    replace(
                        outcome,
                        evidence=frozenset(
                            replace(promise, consumed_ports=frozenset()) for promise in outcome.evidence
                        ),
                    )
                    for outcome in interface.outcomes
                ),
            )
        return original(**kwargs)

    monkeypatch.setattr(_graph_profile, "_compile_graph", malformed)
    with pytest.raises(ContractViolation) as error:
        asyncio.run(run_graph_profile(scripted_detection, _source(), _options()))
    assert error.value.code.value == code
    assert scripted_detection[3] == {}


def test_dropped_source_preserves_peer_candidate_and_legacy_failure(scripted_detection):
    from anonymizer.engine.constants import COL_REPLACED_TEXT
    from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN

    source = pd.DataFrame(
        {COL_TEXT: ["Alice travels.", "Alice fails."], RECORD_ID_COLUMN: ["keep", "drop"], "extra": [17, 29]}
    )
    result = asyncio.run(run_graph_profile(scripted_detection, source, _options()))
    assert len(result.final_outputs) == 1
    projected = result.frames[0].project(result)
    assert projected[COL_REPLACED_TEXT].tolist() == ["[REDACTED_FIRST_NAME] travels."]
    assert projected["extra"].tolist() == [17]
    assert [failure.record_id for failure in result.frames[0].failed_records] == ["drop"]
    from anonymizer.engine.harness.kernel import qualify_detection_redact

    assert len(qualify_detection_redact(result).qualified) == 1
    assert all(state.complete for state in result.states)


@pytest.mark.parametrize("fault", ["overlap", "foreign", "unattributed", "duplicate"])
def test_delegated_source_partition_rejects_uncorrelated_completion(scripted_detection, monkeypatch, fault):
    from anonymizer.engine.detection._graph_invocation import DetectionRedactInvocation
    from anonymizer.engine.graph_sdk.requests import SemanticAssociation
    from anonymizer.graph import TaskAttemptId

    original = DetectionRedactInvocation._run

    def corrupted(self, inputs, requests):
        result = original(self, inputs, requests)
        association = result.results[0].association
        assert isinstance(association, SemanticAssociation)
        if fault == "unattributed":
            return replace(result, results=result.results[1:])
        if fault == "duplicate":
            raw = replace(result)
            object.__setattr__(raw, "failed_sources", (association, association))
            return raw
        if fault == "foreign":
            association = SemanticAssociation(task=TaskAttemptId.new(activation=association.task.activation))
        return replace(result, failed_sources=frozenset({association}))

    monkeypatch.setattr(DetectionRedactInvocation, "_run", corrupted)
    result = asyncio.run(run_graph_profile(scripted_detection, _source(), _options()))
    assert not result.final_outputs and not result.frames
    assert all(state.complete for state in result.states)


@pytest.mark.parametrize("fault", ["fewer", "more", "wrong_keys"])
def test_generation_selection_must_match_actual_physical_work(scripted_detection, monkeypatch, fault):
    from anonymizer.engine.detection._graph_invocation import DetectionRedactInvocation
    from anonymizer.engine.graph_sdk.requests import TextCollectionValue

    original = DetectionRedactInvocation._run

    def corrupted(self, inputs, requests):
        result = original(self, inputs, requests)
        row = result.results[0]
        plan = next(output for output in row.outputs if output.port == "validation_chunks")
        assert isinstance(plan.value, TextCollectionValue)
        items = plan.value.items
        if fault == "fewer":
            changed = ()
        elif fault == "more":
            changed = (*items, replace(items[0], key=1))
        else:
            changed = (replace(items[0], key=1),)
        output = replace(plan, value=TextCollectionValue(items=changed))
        row = replace(row, outputs=tuple(output if value is plan else value for value in row.outputs))
        return replace(result, results=(row, *result.results[1:]))

    monkeypatch.setattr(DetectionRedactInvocation, "_run", corrupted)
    result = asyncio.run(run_graph_profile(scripted_detection, _source(), _options()))
    assert result.requests.dispatched_count == 7
    assert not result.final_outputs and not result.frames


def _modify_dd_responses(scripted_detection, monkeypatch, modify):
    designer = scripted_detection[0]
    original = designer._create_resource_provider

    def resources(*args, **kwargs):
        provider = original(*args, **kwargs)
        factory = provider.model_registry._model_facade_factory

        def facade(*args, **kwargs):
            result = factory(*args, **kwargs)
            prior = result._client._inner._transport

            async def response(request):
                return modify(request, await prior.handle_async_request(request))

            result._client._inner._transport = httpx.MockTransport(response)
            return result

        provider.model_registry._model_facade_factory = facade
        return provider

    monkeypatch.setattr(designer, "_create_resource_provider", resources)


@pytest.mark.parametrize("failed", [False, True])
def test_completed_response_without_usage_is_settled_but_never_fabricated(scripted_detection, monkeypatch, failed):
    import json

    from anonymizer.engine.graph_sdk.requests import UnknownUsage
    from anonymizer.engine.harness.kernel import qualify_detection_redact

    def response(request, answer):
        body = json.loads(request.content)
        if body["model"] == "detector" and "Alice stays." in json.dumps(body["messages"]):
            payload = answer.json()
            payload.pop("usage", None)
            if failed:
                payload = {"error": {"message": "missing-usage failure", "type": "invalid_request_error"}}
            return httpx.Response(400 if failed else answer.status_code, json=payload)
        return answer

    _modify_dd_responses(scripted_detection, monkeypatch, response)
    result = asyncio.run(run_graph_profile(scripted_detection, _source(), _options()))
    assert result.requests.dispatched_count == (5 if failed else 7)
    assert not result.requests.remote_outstanding and not result.requests.local_in_flight
    assert sum(isinstance(row.usage, UnknownUsage) for row in result.requests.settlements) == 1
    assert all(row.remote_stopped for row in result.requests.settlements)
    assert len(result.final_outputs) == (1 if failed else 2)
    assert len(qualify_detection_redact(result).qualified) == 1
    assert not result._retained_resources


def test_chunked_validation_qualifies_every_source(scripted_detection):
    from anonymizer.engine.harness.kernel import qualify_detection_redact

    source = pd.DataFrame({COL_TEXT: ["Alice travels.", "Alice Bob stays."]})
    result = asyncio.run(
        run_graph_profile(scripted_detection, source, {**_options(), "validation_max_entities_per_call": 1}, chunks=2)
    )
    assert len(qualify_detection_redact(result).qualified) == 2


def test_default_augmenter_restart_preserves_legacy_and_qualifies(scripted_detection, monkeypatch):
    import json

    from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
    from anonymizer.engine.ndd.adapter import NddAdapter

    designer, models, selected, calls = scripted_detection
    attempted = []

    def response(request, answer):
        body = json.loads(request.content)
        if "Find untagged sensitive entities" in json.dumps(body["messages"]):
            attempted.append(request.content)
            if len(attempted) == 1:
                payload = answer.json()
                payload["choices"][0]["message"]["content"] = "not structured output"
                return httpx.Response(200, json=payload)
        return answer

    _modify_dd_responses(scripted_detection, monkeypatch, response)
    legacy = EntityDetectionWorkflow(NddAdapter(designer)).run(
        _source(), model_configs=models, selected_models=selected, **_options()
    )
    assert len(legacy.dataframe) == 2 and not legacy.failed_records
    assert len(attempted) == 3
    attempted.clear()
    calls.clear()
    result = asyncio.run(run_graph_profile(scripted_detection, _source(), _options()))
    from anonymizer.engine.constants import COL_FINAL_ENTITIES
    from anonymizer.engine.harness.kernel import qualify_detection_redact

    assert len(result.final_outputs) == len(qualify_detection_redact(result).qualified) == 2
    assert (
        result.frames[0].project(result)[COL_FINAL_ENTITIES].tolist() == legacy.dataframe[COL_FINAL_ENTITIES].tolist()
    )
    assert len(attempted) == 3
    assert result.requests.dispatched_count == sum(calls.values()) == 8
    restarts = [row for row in result.requests.dispatches if row.child.source_kind == "conversation_restart"]
    assert len(restarts) == 1
    predecessor = restarts[0].child.predecessor
    terminal = next(row for row in result.requests.terminals if row.request == predecessor)
    assert terminal.failure == "malformed_response"
    assert next(row for row in result.requests.settlements if row.request == predecessor).disposition == "completed"
    assert all(state.complete for state in result.states)

    from anonymizer.graph import ContractViolation
    from tests.graph_sdk.child_qualification_fixtures import result_with

    unused = next(row for row in result.record.terminals if row.unused_retry is not None)
    foreign = next(
        row.attempt
        for row in result.record.terminals
        if row.attempt is not None and row.attempt not in (unused.unused_retry.owner, unused.unused_retry.after)
    )
    forged = replace(unused, unused_retry=replace(unused.unused_retry, after=foreign))
    mutations = (
        result_with(result, requests=replace(result.requests, child_evidence=())),
        result_with(
            result,
            record=replace(
                result.record, terminals=tuple(forged if row is unused else row for row in result.record.terminals)
            ),
        ),
    )
    for changed in mutations:
        with pytest.raises((EffectRejected, ContractViolation)):
            qualify_detection_redact(changed)


@pytest.mark.parametrize(
    "rejected,qualified,sends,corrections,restarts", [(5, 2, 12, 0, 5), (6, 1, 12, 0, 5), (2, 2, 9, 1, 1)]
)
def test_default_restart_capacity_preserves_exhaustion_and_peer(
    scripted_detection, monkeypatch, rejected, qualified, sends, corrections, restarts
):
    import json

    from anonymizer.engine.constants import COL_FINAL_ENTITIES
    from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
    from anonymizer.engine.harness.kernel import qualify_detection_redact
    from anonymizer.engine.ndd.adapter import NddAdapter

    designer, models, selected, calls = scripted_detection
    designer.set_run_config(
        designer.run_config.model_copy(
            update={
                "max_conversation_correction_steps": corrections,
                "max_conversation_restarts": restarts,
            }
        )
    )
    count = min(rejected + 1, (corrections + 1) * (restarts + 1))
    attempts = []

    def response(request, answer):
        body = json.loads(request.content)
        messages = json.dumps(body["messages"])
        if "Find untagged sensitive entities" in messages and "travels." in messages:
            attempts.append(request.content)
            if len(attempts) <= rejected:
                payload = answer.json()
                payload["choices"][0]["message"]["content"] = "not structured output"
                return httpx.Response(200, json=payload)
        return answer

    _modify_dd_responses(scripted_detection, monkeypatch, response)
    legacy = EntityDetectionWorkflow(NddAdapter(designer)).run(
        _source(), model_configs=models, selected_models=selected, **_options()
    )
    assert len(legacy.dataframe) == qualified
    assert len(attempts) == count
    attempts.clear()
    calls.clear()
    result = asyncio.run(run_graph_profile(scripted_detection, _source(), _options()))
    assert len(attempts) == count
    assert result.requests.dispatched_count == sum(calls.values()) == sends
    assert len(result.final_outputs) == len(qualify_detection_redact(result).qualified) == qualified
    projected = result.frames[0].project(result)
    assert projected[COL_FINAL_ENTITIES].tolist() == legacy.dataframe[COL_FINAL_ENTITIES].tolist()
    assert result.frames[0].failed_records == tuple(legacy.failed_records)
    assert sum(row.failure == "malformed_response" for row in result.requests.terminals) == rejected
    assert all(state.complete for state in result.states) and result.invocation_state.complete
    assert not result.requests.remote_outstanding
    if corrections:
        kinds = [row.child.source_kind for row in result.requests.dispatches if row.child.source_kind != "initial"]
        assert kinds == ["correction", "conversation_restart"]


@pytest.mark.parametrize("preview", [True, False])
def test_all_failed_dd_sources_raise_legacy_and_withhold_graph(scripted_detection, preview):
    from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
    from anonymizer.engine.harness.kernel import project_detection_redact, qualify_detection_redact
    from anonymizer.engine.ndd.adapter import NddAdapter
    from anonymizer.interface.errors import AnonymizerWorkflowError

    designer, models, selected, calls = scripted_detection
    source = pd.DataFrame({COL_TEXT: ["Alice fails.", "Alice fails."], "extra": [17, 29]})
    options: dict[str, Any] = {**_options(), "preview_num_records": 2 if preview else None}
    with pytest.raises(AnonymizerWorkflowError, match="scripted row failure") as legacy:
        EntityDetectionWorkflow(NddAdapter(designer)).run(
            source, model_configs=models, selected_models=selected, **options
        )
    calls.clear()
    result = asyncio.run(run_graph_profile(scripted_detection, source, options))
    assert not result.final_outputs and not result.frames and not qualify_detection_redact(result).qualified
    assert result.requests.dispatched_count == sum(calls.values()) == 3
    assert all(state.complete for state in result.states) and result.invocation_state.complete
    with pytest.raises(AnonymizerWorkflowError) as projected:
        project_detection_redact(result)
    assert str(projected.value) == str(legacy.value)
    assert type(projected.value.__cause__) is type(legacy.value.__cause__)
    assert projected.value is result.submission_failures[0].cause
    assert retained(projected.value) is result


@pytest.mark.parametrize("chunk_size,expected", [(1, 8), (2, 7)])
def test_admission_derives_validator_capacity_without_input_cap(scripted_detection, chunk_size, expected):
    from anonymizer.engine.harness.kernel import qualify_detection_redact

    source = pd.DataFrame({COL_TEXT: ["Alice travels.", "Alice Bob stays."]})
    result = asyncio.run(
        run_graph_profile(
            scripted_detection,
            source,
            {**_options(), "validation_max_entities_per_call": chunk_size},
            chunks=None,
        )
    )
    contract = next(policy.delegated for policy in result._execution.policies if policy.delegated)
    assert len(contract.selection.slots) == (len("Alice Bob stays.") + chunk_size - 1) // chunk_size
    assert result.requests.dispatched_count == expected
    assert len(result.final_outputs) == len(qualify_detection_redact(result).qualified) == 2


def test_unknown_remote_source_retains_resource_and_qualifies_independent_peer(scripted_detection, monkeypatch):
    import json

    from anonymizer.engine.harness.kernel import qualify_detection_redact

    physicals = []
    initialize = NddPhysicalTransport.__init__

    def initialized(self):
        initialize(self)
        physicals.append(self)

    def response(request, answer):
        messages = json.dumps(json.loads(request.content)["messages"])
        if "Find untagged sensitive entities" in messages and "travels." in messages:
            raise httpx.ReadError("scripted uncertain source", request=request)
        return answer

    monkeypatch.setattr(NddPhysicalTransport, "__init__", initialized)
    _modify_dd_responses(scripted_detection, monkeypatch, response)
    try:
        result = asyncio.run(run_graph_profile(scripted_detection, _source(), _options(), remote_limit=2))
        assert result.requests.dispatched_count == 7
        assert len(result.requests.remote_outstanding) == 1
        assert len(result.final_outputs) == 1
        # Lost remains charged and uncertain; an authenticated retained lease
        # does not invalidate the independently verified successful source.
        assert len(qualify_detection_redact(result).qualified) == 1
        assert result.frames[0].project(result)[COL_TEXT].tolist() == ["Alice stays."]
        assert len(result._retained_resources) == 1 and not physicals[0].closed
        from anonymizer.engine.harness.kernel import project_detection_redact

        projected = project_detection_redact(result)
        assert projected._execution is result
        assert projected.dataframe[COL_TEXT].tolist() == ["Alice stays."]
    finally:
        for physical in physicals:
            asyncio.run_coroutine_threadsafe(physical.close(), physical.loop).result(timeout=_HANG_GUARD_TIMEOUT)


@pytest.mark.parametrize("failed", [False, True])
def test_create_projection_keeps_source_identity_out_of_persisted_parquet(scripted_detection, tmp_path, failed):
    from anonymizer.engine.constants import COL_REPLACED_TEXT
    from anonymizer.engine.harness.kernel import project_detection_redact

    source = pd.DataFrame(
        {COL_TEXT: ["Alice travels.", "Alice fails." if failed else "Alice stays."], "extra": [17, 29]}
    )
    result = asyncio.run(run_graph_profile(scripted_detection, source, {**_options(), "preview_num_records": None}))
    projected = project_detection_redact(result)
    assert projected.dataframe["extra"].tolist() == ([17] if failed else [17, 29])
    assert projected.dataframe[COL_REPLACED_TEXT].tolist() == (
        ["[REDACTED_FIRST_NAME] travels."]
        if failed
        else ["[REDACTED_FIRST_NAME] travels.", "[REDACTED_FIRST_NAME] stays."]
    )
    paths = list((tmp_path / "artifacts").rglob("*.parquet"))
    assert paths
    for path in paths:
        assert not any(name.startswith("_anonymizer_source_") for name in pd.read_parquet(path).columns)
    for path in (tmp_path / "artifacts").rglob("*.json"):
        assert "_anonymizer_source_" not in path.read_text()


@pytest.mark.parametrize("fault", [None, "foreign", "duplicate", "unregistered_batch"])
def test_private_storage_batch_identity_is_ordered_and_closed(tmp_path, fault):
    from data_designer.engine.storage.artifact_storage import ArtifactStorage, BatchStage

    from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame
    from anonymizer.interface.errors import AnonymizerWorkflowError

    source = WorkflowSourceFrame.capture(pd.DataFrame({COL_TEXT: ["first", "second"]}))
    storage = ArtifactStorage(artifact_path=tmp_path, dataset_name="identity")
    original_write = storage.write_parquet_file
    with source._bind_storage(storage):
        if fault in {"foreign", "duplicate"}:
            invalid = source.seed()
            invalid[source.key_column] = [source.keys[0], "foreign" if fault == "foreign" else source.keys[0]]
            with pytest.raises(AnonymizerWorkflowError, match="invalid source correspondence"):
                storage.write_batch_to_parquet_file(0, invalid, BatchStage.PARTIAL_RESULT)
            assert not list(tmp_path.rglob("*.parquet"))
            return
        for index in (1, 0):
            storage.write_batch_to_parquet_file(index, source.seed().iloc[[index]], BatchStage.PARTIAL_RESULT)
            storage.move_partial_result_to_final_file_path(index)
        if fault == "unregistered_batch":
            original_write("batch_99999.parquet", pd.DataFrame({COL_TEXT: ["unregistered"]}), BatchStage.FINAL_RESULT)
            with pytest.raises(AnonymizerWorkflowError, match="unknown source batches"):
                storage.load_dataset()
        else:
            returned, positions = source.project(storage.load_dataset())
            assert returned[COL_TEXT].tolist() == ["first", "second"]
            assert positions == (0, 1)
        for path in tmp_path.rglob("*.parquet"):
            assert source.key_column not in pd.read_parquet(path).columns
    assert storage.write_parquet_file == original_write


@pytest.mark.parametrize("preview", [True, False])
def test_empty_source_invocation_preserves_actual_dd_failure(scripted_detection, preview):
    from anonymizer.engine.harness.kernel import project_detection_redact, qualify_detection_redact
    from anonymizer.interface.errors import AnonymizerWorkflowError

    source = pd.DataFrame({COL_TEXT: pd.Series([], dtype=str), "extra": pd.Series([], dtype=object)})
    result = asyncio.run(
        run_graph_profile(
            scripted_detection, source, {**_options(), "preview_num_records": 0 if preview else None}, chunks=None
        )
    )
    assert result.requests.dispatched_count == 1
    assert scripted_detection[3] == {"readiness": 1}
    assert len(result.requests.settlements) == 1 and not result.requests.remote_outstanding
    assert result.states == () and result.invocation_state.complete
    assert not result.artifacts and not result.final_outputs and not result.frames
    assert len(result.record.terminals) == 5
    assert result.record.memberships[0].closed
    assert len(result.record.memberships[0].members) == 5
    prepared = result._execution.context.prepared
    assert prepared.configuration.invocation_owned and prepared.total_activation_slots == 5
    assert prepared.declared_request_upper_bound == 3
    assert not qualify_detection_redact(result).qualified
    with pytest.raises(
        AnonymizerWorkflowError, match="Dataset is empty" if preview else "Failed to load generated dataset"
    ) as error:
        project_detection_redact(result)
    assert error.value is result.submission_failures[0].cause
    assert type(error.value.__cause__).__name__ == "DataDesignerGenerationError"


@pytest.mark.parametrize("requested", [0, 1])
def test_preview_admits_only_requested_sources_but_calls_detection_with_whole_frame(
    scripted_detection, monkeypatch, requested
):
    import anonymizer.engine.detection._semantic_detection as semantic_detection
    from anonymizer.engine.harness.kernel import project_detection_redact
    from anonymizer.interface.errors import AnonymizerWorkflowError

    seen = []
    original = semantic_detection.lower_detection

    def lower(receiver, frame, arguments, *rest):
        seen.append((len(frame), arguments["preview_num_records"]))
        return original(receiver, frame, arguments, *rest)

    monkeypatch.setattr(semantic_detection, "lower_detection", lower)
    result = asyncio.run(
        run_graph_profile(scripted_detection, _source(), {**_options(), "preview_num_records": requested}, chunks=None)
    )
    assert seen == [(2, requested)]
    assert len(result.states) == requested
    assert len(result._execution.context.prepared.data.targets) == requested
    assert result.requests.dispatched_count == (4 if requested else 1)
    if requested:
        projected = project_detection_redact(result)
        assert projected.dataframe[COL_TEXT].tolist() == ["Alice travels."]
        assert not projected.failed_records
        assert all(state.complete for state in result.states)
    else:
        assert result.invocation_state.complete
        with pytest.raises(AnonymizerWorkflowError, match="Dataset is empty"):
            project_detection_redact(result)


def test_empty_invocation_preentry_cancel_closes_without_dd_calls(scripted_detection, monkeypatch):

    from anonymizer.engine.harness import kernel

    original = kernel.start_execution

    async def start(**kwargs):
        running = await original(**kwargs)
        running.request_cancel()
        return running

    monkeypatch.setattr(kernel, "start_execution", start)
    source = pd.DataFrame({COL_TEXT: pd.Series([], dtype=str)})
    result = asyncio.run(
        run_graph_profile(scripted_detection, source, {**_options(), "preview_num_records": 0}, chunks=None)
    )
    assert not scripted_detection[3] and result.requests.dispatched_count == 0
    assert result.invocation_state.complete and result.states == ()
    assert len(result.record.terminals) == 5
    assert all(terminal.category == "blocked" for terminal in result.record.terminals)
    assert result.record.memberships[0].closed
    assert not result.final_outputs and not result.frames


@pytest.mark.parametrize("fault", ["activation_capacity", "foreign_target_scope"])
def test_invocation_owned_preparation_rejects_before_dd_effects(scripted_detection, monkeypatch, fault):
    import anonymizer.engine.detection._graph_profile as profile
    from anonymizer.engine.graph_sdk.capabilities import PreparationRejected

    if fault == "activation_capacity":
        original_limits = profile.PreparationLimits

        def limited(**kwargs: Any):
            kwargs["max_total_activation_slots"] = 4
            return original_limits(**kwargs)

        monkeypatch.setattr(profile, "PreparationLimits", limited)
        source = pd.DataFrame({COL_TEXT: pd.Series([], dtype=str)})
    else:
        original_configuration = profile.PreparationConfiguration

        def wrong_scope(**kwargs: Any):
            kwargs["invocation_owned"] = True
            return original_configuration(**kwargs)

        monkeypatch.setattr(profile, "PreparationConfiguration", wrong_scope)
        source = _source()
    with pytest.raises(PreparationRejected):
        asyncio.run(run_graph_profile(scripted_detection, source, _options(), chunks=None))
    assert not scripted_detection[3]


@pytest.mark.parametrize("active_loop", [False, True])
def test_synchronous_graph_profile_owns_pool_and_reuses_dd_loop(scripted_detection, monkeypatch, active_loop):
    from anonymizer.engine.detection._graph_profile import run_detection_redact
    from anonymizer.engine.graph_sdk.blocking import BlockingPool
    from anonymizer.engine.graph_sdk.executor import ExecutionLimits
    from anonymizer.engine.replace.replace_runner import ReplacementWorkflow

    pools = []
    original = BlockingPool.__init__

    def capture(self, **kwargs):
        original(self, **kwargs)
        pools.append(self)

    def retired(*args, **kwargs):
        raise AssertionError("graph Redact must not construct the generic replacement dispatcher")

    monkeypatch.setattr(BlockingPool, "__init__", capture)
    monkeypatch.setattr(ReplacementWorkflow, "run", retired)
    designer, models, selected, calls = scripted_detection

    def run():
        return run_detection_redact(
            _source(),
            detection=detection_tests.EntityDetectionWorkflow(detection_tests.NddAdapter(designer)),
            model_configs=models,
            selected_models=selected,
            detection_options=_options(),
            limits=ExecutionLimits(
                max_local_in_flight=1,
                max_remote_outstanding=1,
                max_runtime_artifacts=16,
                max_runtime_artifact_bytes=10000,
                max_collection_items=32,
            ),
        )

    async def nested():
        loop = asyncio.get_running_loop()
        result = run()
        assert asyncio.get_running_loop() is loop and loop.is_running()
        return result

    projected = asyncio.run(nested()) if active_loop else run()
    assert projected.dataframe[detection_tests.COL_REPLACED_TEXT].tolist() == [
        "[REDACTED_FIRST_NAME] travels.",
        "[REDACTED_FIRST_NAME] stays.",
    ]
    assert projected._execution is not None
    assert projected._execution.requests.dispatched_count == 7
    assert calls == {"readiness": 1, "detector": 2, "validator": 2, "augmenter": 2}
    # The validator rows borrow the one outer profile and its pool.
    assert len(pools) == 1 and all(pool.closed and pool.occupied == 0 for pool in pools)
    assert not projected._execution._retained_resources


@pytest.mark.parametrize("lost", [False, True])
def test_synchronous_profile_interrupt_drains_actual_callback(scripted_detection, monkeypatch, lost):
    """A Ctrl-C delivered to the caller while a model request is in flight drains and keeps the owner."""
    import os
    import signal
    from threading import Event

    import anonymizer.engine.detection._graph_profile as profile
    from anonymizer.engine.graph_sdk.executor import ExecutionLimits, RunningExecution

    entered, release = Event(), Event()
    physicals = []
    dispatch = NddPhysicalTransport.dispatch
    cancel = RunningExecution.request_cancel

    async def held(self, request):
        if not physicals:
            physicals.append(self)
            entered.set()
            # A real Ctrl-C: the signal lands on the main thread, wherever the caller is waiting.
            os.kill(os.getpid(), signal.SIGINT)
            await asyncio.to_thread(release.wait)
        return await dispatch(self, request)

    def stop(self):
        assert not physicals[0].closed
        cancel(self)
        release.set()

    if lost:

        def lose(request, answer):
            raise httpx.ReadError("scripted readiness uncertainty during cancellation", request=request)

        _modify_dd_responses(scripted_detection, monkeypatch, lose)
    monkeypatch.setattr(NddPhysicalTransport, "dispatch", held)
    monkeypatch.setattr(RunningExecution, "request_cancel", stop)
    designer, models, selected, _ = scripted_detection
    try:
        with pytest.raises(KeyboardInterrupt) as interrupted:
            profile.run_detection_redact(
                _source(),
                detection=detection_tests.EntityDetectionWorkflow(detection_tests.NddAdapter(designer)),
                model_configs=models,
                selected_models=selected,
                detection_options=_options(),
                limits=ExecutionLimits(
                    max_local_in_flight=1,
                    max_remote_outstanding=1,
                    max_runtime_artifacts=16,
                    max_runtime_artifact_bytes=10000,
                    max_collection_items=32,
                ),
            )
        result = retained(interrupted.value)
        assert result.requests.dispatched_count == 1
        assert len(physicals) == 1 and physicals[0].closed is not lost
        if lost:
            assert len(result.requests.remote_outstanding) == 1
            assert len(result._retained_resources) == 1
            assert result._retained_resources[0].handle.owns(physicals[0])
            assert all(not client._inner._closed for client in physicals[0].clients)
        else:
            assert not result.requests.remote_outstanding and not result._retained_resources
            assert all(client._inner._closed for client in physicals[0].clients)
    finally:
        release.set()
        for physical in physicals:
            if not physical.closed:
                asyncio.run_coroutine_threadsafe(physical.close(), physical.loop).result(timeout=_HANG_GUARD_TIMEOUT)


@pytest.mark.parametrize("preview", [False, True])
def test_actual_lost_lease_survives_public_result_and_preview_projection(
    scripted_detection, monkeypatch, tmp_path, preview
):
    import json
    import pickle

    import anonymizer.interface.anonymizer as facade_module
    from anonymizer import AnonymizerConfig, AnonymizerInput
    from anonymizer.config.replace_strategies import Redact
    from anonymizer.engine.detection._graph_profile import run_detection_redact
    from anonymizer.engine.graph_sdk._foreign_transport import ForeignLoopTransport

    # Lost-lease ownership belongs to the explicitly selected qualified profile.
    # Ordinary DD publication now declares hidden physical work as unknown.
    def qualified(*args: Any, **kwargs: Any):
        kwargs["opaque"] = False
        return run_detection_redact(*args, **kwargs)

    monkeypatch.setattr(facade_module, "run_detection_redact", qualified)

    def response(request, answer):
        messages = json.dumps(json.loads(request.content)["messages"])
        if "Find untagged sensitive entities" in messages and "travels." in messages:
            raise httpx.ReadError("scripted uncertain source", request=request)
        return answer

    _modify_dd_responses(scripted_detection, monkeypatch, response)
    path = tmp_path / "public-input.csv"
    _source().rename(columns={COL_TEXT: "text"}).to_csv(path, index=False)
    facade = detection_tests.public_anonymizer(scripted_detection)
    data = AnonymizerInput(source=str(path))
    config = AnonymizerConfig(replace=Redact(), emit_telemetry=False)
    public = (
        facade.preview(config=config, data=data, num_records=2) if preview else facade.run(config=config, data=data)
    )
    execution = public._execution
    assert execution is not None and execution.requests.dispatched_count == 7
    assert len(execution.requests.remote_outstanding) == 1
    (lease,) = execution._retained_resources
    assert isinstance(lease.handle, ForeignLoopTransport)
    physical = lease.handle._created_inner
    assert isinstance(physical, NddPhysicalTransport)
    try:
        assert not physical.closed and all(not client._inner._closed for client in physical.clients)
        assert public.dataframe["text_replaced"].tolist() == ["[REDACTED_FIRST_NAME] stays."]
        assert len(public.failed_records) == 1
        restored = pickle.loads(pickle.dumps(public))
        assert restored._execution is None
        pd.testing.assert_frame_equal(restored.dataframe, public.dataframe)
        pd.testing.assert_frame_equal(restored.trace_dataframe, public.trace_dataframe)
        assert restored.failed_records == public.failed_records
        assert not physical.closed
    finally:
        asyncio.run_coroutine_threadsafe(physical.close(), physical.loop).result(timeout=_HANG_GUARD_TIMEOUT)


def test_async_acquisition_and_close_drain_repeated_cancellation(scripted_detection, monkeypatch):
    from threading import Event

    from anonymizer.engine.harness import kernel

    acquired, release_acquire, closing, release_close = Event(), Event(), Event(), Event()
    physicals, acquisitions = [], []
    initialize, close = NddPhysicalTransport.__init__, NddPhysicalTransport.close
    submit = asyncio.run_coroutine_threadsafe

    def held_init(self):
        initialize(self)
        physicals.append(self)
        acquired.set()
        assert release_acquire.wait(_HANG_GUARD_TIMEOUT)

    async def held_close(self):
        closing.set()
        assert await asyncio.to_thread(release_close.wait, _HANG_GUARD_TIMEOUT)
        await close(self)

    def track(coroutine, loop):
        future = submit(coroutine, loop)
        if coroutine.__qualname__.endswith("create_resource"):
            acquisitions.append(future)
        return future

    async def unexpected_handoff(**kwargs):
        raise AssertionError("cancelled acquisition must not start the graph invocation")

    monkeypatch.setattr(NddPhysicalTransport, "__init__", held_init)
    monkeypatch.setattr(NddPhysicalTransport, "close", held_close)
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", track)
    monkeypatch.setattr(kernel, "start_execution", unexpected_handoff)

    async def run():
        task = asyncio.create_task(run_graph_profile(scripted_detection, _source(), _options()))
        try:
            assert await asyncio.to_thread(acquired.wait, _HANG_GUARD_TIMEOUT)
            task.cancel("acquisition interrupted")
            await asyncio.sleep(0)
            task.cancel("observer interrupted again")
            await asyncio.sleep(0)
            assert not task.done() and not physicals[0].closed
            assert len(acquisitions) == 1 and not acquisitions[0].cancelled()
            release_acquire.set()
            assert await asyncio.to_thread(closing.wait, _HANG_GUARD_TIMEOUT)
            task.cancel("interrupted while closing")
            await asyncio.sleep(0)
            assert not task.done() and not physicals[0].closed
            release_close.set()
            with pytest.raises(asyncio.CancelledError, match="acquisition interrupted") as interrupted:
                await task
            assert retained_execution(interrupted.value) is None
            assert physicals[0].closed and not physicals[0].clients
            assert acquisitions[0].done() and not acquisitions[0].cancelled()
            assert physicals[0].loop.is_running()
            assert not scripted_detection[3]
        finally:
            release_acquire.set()
            release_close.set()

    asyncio.run(run())


def test_failed_transport_acquisition_closes_its_actual_physical_resource(scripted_detection, monkeypatch):
    from anonymizer.engine.graph_sdk._foreign_transport import ForeignLoopTransport

    physicals = []
    initialize = NddPhysicalTransport.__init__
    failure = RuntimeError("scripted transport declaration rejection")

    def capture(self):
        initialize(self)
        physicals.append(self)

    def reject(**kwargs):
        raise failure

    monkeypatch.setattr(NddPhysicalTransport, "__init__", capture)
    monkeypatch.setattr(ForeignLoopTransport, "create", reject)
    with pytest.raises(RuntimeError, match="scripted transport declaration rejection") as caught:
        asyncio.run(run_graph_profile(scripted_detection, _source(), _options()))
    assert caught.value is failure
    assert len(physicals) == 1 and physicals[0].closed and not physicals[0].clients
    assert physicals[0].loop.is_running() and not scripted_detection[3]
