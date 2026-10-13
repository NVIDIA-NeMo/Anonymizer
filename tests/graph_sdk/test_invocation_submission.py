# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Literal ownership controls for one submission over multiple source occurrences."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.executor import LocalCompleted, start_execution
from anonymizer.engine.graph_sdk.requests import AssociationResult, PortArtifact, TextArtifactValue
from tests.graph_sdk.delegated_execution_fixtures import with_multiple_sources
from tests.graph_sdk.test_delegated_execution import _execute
from tests.graph_sdk.test_delegated_readiness import _qualify, _ready_fixture


@pytest.mark.parametrize("readiness_failure,optional_tail", [(False, False), (True, False), (False, True)])
def test_shared_readiness_has_no_target_and_each_source_has_its_own_leaf(readiness_failure, optional_tail):
    base, services, original, generation, readiness, probe = _ready_fixture(
        failure="readiness_failure" if readiness_failure else None
    )
    if optional_tail:
        from tests.graph_sdk.test_optional_tail import optional_fixture

        base, services, original = optional_fixture(readiness="available")
    execution = with_multiple_sources(base)
    groups = []

    class Submit:
        execution = "same_loop"

        async def submit(self, inputs, requests):
            groups.append(tuple(item.association for item in inputs))
            await requests.readiness()
            await requests.probe()
            if not optional_tail:
                with pytest.raises(EffectRejected) as missing:
                    await requests.request(original.slots[0])
                assert missing.value.code == EffectCode.MISSING
            results = []
            for item in inputs:
                await requests.request(original.slots[0], association=item.association)
                source = item.inputs[0]
                results.append(
                    AssociationResult(
                        association=item.association,
                        outcome="ok",
                        consumed_context_ports=frozenset(),
                        outputs=(
                            PortArtifact(
                                port="context",
                                artifact_type=source.artifact_type,
                                artifact=None,
                                value=TextArtifactValue(text="candidate:" + source.value.text),
                            ),
                        ),
                    )
                )
            return LocalCompleted(results=tuple(reversed(results)))

    services = replace(
        services,
        handles=tuple(
            replace(handle, delegated=Submit()) if handle.delegated else handle for handle in services.handles
        ),
        limits=replace(services.limits, max_runtime_artifacts=16, max_runtime_artifact_bytes=1000),
    )
    result = asyncio.run(_execute(execution, services))
    assert len(groups) == 1 and len(groups[0]) == 2
    if not optional_tail:
        assert (readiness.calls, generation.calls, probe.calls) == ((1, 0, 0) if readiness_failure else (1, 2, 1))
    assert sum(row.unused_retry is not None for row in result.record.terminals) == (2 if optional_tail else 0)
    assert result.requests.dispatched_count == (1 if readiness_failure else 3)
    assert result.invocation_state is not None and result.invocation_state.complete
    shared_keys = {entry.activation for entry in result.invocation_state.entries}
    source_keys = {entry.activation for state in result.states for entry in state.entries}
    assert len(shared_keys) == 3 and not shared_keys & source_keys
    ready = next(row for row in result.requests.dispatches if row.child.role == "readiness")
    assert ready.child.leaf.activation in shared_keys
    assert next(iter(ready.associations)).task.activation in shared_keys
    assert all(
        row.child.leaf.activation in source_keys for row in result.requests.dispatches if row.child.role == "generation"
    )
    assert all(state.complete for state in result.states)
    assert len(result.final_outputs) == (0 if readiness_failure else 2)
    artifacts = dict(result.artifacts)
    source_text = {datum.id: datum.text for datum in execution.context.prepared.data.datums}
    assert all(
        artifacts[output.candidate.artifact] == TextArtifactValue(text="candidate:" + source_text[output.target])
        for output in result.final_outputs
    )
    with pytest.raises(EffectRejected) as unsupported:
        _qualify(execution, result)
    assert unsupported.value.code == EffectCode.UNSUPPORTED


@pytest.mark.parametrize("targets", [1, 2])
def test_prestart_cancellation_closes_all_source_and_invocation_obligations(targets):
    async def run():
        execution, services, *_ = _ready_fixture()
        if targets == 2:
            execution = with_multiple_sources(execution)
            services = replace(
                services,
                limits=replace(
                    services.limits,
                    max_runtime_artifacts=16,
                    max_runtime_artifact_bytes=1000,
                ),
            )
        running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
        running.request_cancel()
        result = await running.wait()
        assert result.requests.dispatched_count == 0 and not result.final_outputs
        assert all(state.complete for state in result.states)
        if targets == 2:
            assert result.invocation_state is not None and result.invocation_state.complete
        assert len(result.record.terminals) == (6 if targets == 1 else 9)
        assert all(row.attempt is None for row in result.record.terminals)
        assert all(membership.closed for membership in result.record.memberships)

    asyncio.run(run())
