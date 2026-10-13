# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distinct owner, setup and canonical publication faults for actual DD execution."""

from __future__ import annotations

import asyncio
from dataclasses import fields, replace

import httpx
import pandas as pd
import pytest

from anonymizer.engine.constants import COL_REPLACEMENT_MAP, COL_TEXT
from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.executor import admit_execution_plan
from anonymizer.engine.ndd._graph_requests import NddPhysicalTransport, NddRequestBridge
from tests.engine import test_detection_source_frame as detection_tests
from tests.engine.test_detection_source_frame import run_graph_profile

scripted_detection = detection_tests.scripted_detection


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

            physical = asyncio.run_coroutine_threadsafe(create(), physical.loop).result(timeout=10)
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
                asyncio.run_coroutine_threadsafe(physical.close(), physical.loop).result(timeout=10)


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


@pytest.mark.parametrize("missing", ["generation_groups", "optional_tail"])
def test_missing_delegated_fields_reject_at_raw_admission(missing):
    from tests.graph_sdk.test_delegated_readiness import _ready_fixture

    execution, *_ = _ready_fixture(generations=3)
    owner = next(policy for policy in execution.policies if policy.delegated)

    def clone(value, omitted=None, **updates):
        raw = object.__new__(type(value))
        for field in fields(value):
            if field.name != omitted:
                object.__setattr__(raw, field.name, updates.get(field.name, getattr(value, field.name)))
        return raw

    raw = clone(owner, delegated=clone(owner.delegated, omitted=missing))
    with pytest.raises(EffectRejected):
        admit_execution_plan(
            context=execution.context,
            capabilities=execution.capabilities,
            policies=tuple(raw if policy is owner else policy for policy in execution.policies),
            decisions=(),
            assessment_productions=execution.assessment_productions,
            assessment_limits=execution.assessment_limits,
        )


@pytest.mark.parametrize("zero,sends", [(True, 5), (False, 8)])
def test_unclosed_zero_or_variable_chunk_profile_withholds_content(scripted_detection, zero, sends):
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
    assert not result.final_outputs and not result.frames
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
            assert await asyncio.to_thread(entered.wait, 10)
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
            asyncio.run_coroutine_threadsafe(physical.close(), physical.loop).result(timeout=10)


def test_redact_check_records_candidate_mismatch_with_exact_span_dependencies(scripted_detection, monkeypatch):
    from anonymizer.engine.constants import COL_REPLACED_TEXT
    from anonymizer.engine.replace.replace_runner import ReplacementWorkflow

    original = ReplacementWorkflow.run

    def changed(*args, **kwargs):
        returned = original(*args, **kwargs)
        returned.dataframe.loc[returned.dataframe.index[0], COL_REPLACED_TEXT] = "Alice travels."
        return returned

    monkeypatch.setattr(ReplacementWorkflow, "run", changed)
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
    from anonymizer.engine.graph_sdk.qualification import qualify_retained
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
            absences=(),
            configurations=((fact.node, fact.environment.configuration),),
            state=fact.environment.state,
        )

    artifacts = tuple(reference for reference, _ in result.artifacts)
    current = view(artifacts)
    assert all(evidence_validity(evidence=item, current=current) == "current" for item in verified)
    with pytest.raises(EffectRejected) as error:
        qualify_retained(admitted=admitted, result=result, current=current)
    assert error.value.code.value == "unsupported"  # Shared request reconciliation remains unqualified.

    for port, consumed in verified[0].consumed_by_port:
        reference = consumed.artifact if hasattr(consumed, "artifact") else consumed
        missing = view(tuple(item for item in artifacts if item != reference))
        assert evidence_validity(evidence=verified[0], current=missing) == "unknown", port


@pytest.mark.parametrize("fault,code", [("missing_span", "missing"), ("wrong_summary", "contradictory")])
def test_redact_evidence_projection_rejects_incomplete_dependencies(scripted_detection, monkeypatch, fault, code):
    from anonymizer.engine.detection import _graph_profile
    from anonymizer.graph._values import ContractViolation

    original = _graph_profile.admit_static_workflow

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

    monkeypatch.setattr(_graph_profile, "admit_static_workflow", malformed)
    with pytest.raises(ContractViolation) as error:
        asyncio.run(run_graph_profile(scripted_detection, _source(), _options()))
    assert error.value.code.value == code
    assert scripted_detection[3] == {}
