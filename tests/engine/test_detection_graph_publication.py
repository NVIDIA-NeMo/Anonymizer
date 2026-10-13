# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distinct owner, setup and canonical publication faults for actual DD execution."""

from __future__ import annotations

import asyncio
from dataclasses import fields, replace

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
    assert len(limited.final_outputs) == 1
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
