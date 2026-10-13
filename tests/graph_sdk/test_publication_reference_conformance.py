# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual external text publication against the independently frozen predicate.

Assessment-return cases are excluded: this fixture has no assessment production.
The four observed kinds are artifacts, ports, provenance and final outputs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from copy import copy
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import cast

import pytest

from anonymizer.engine.graph_sdk._ingress import _IngressBudget
from anonymizer.engine.graph_sdk.executor import (
    DecisionLimits,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    IngressLimits,
    start_execution,
)
from anonymizer.engine.graph_sdk.inspection import inspect_requests
from anonymizer.engine.graph_sdk.requests import (
    AssociationResult,
    DispatchEnvelope,
    ExactUsage,
    ExternalSettlement,
    PortArtifact,
    SemanticAssociation,
    StopConfirmed,
    TextArtifactValue,
    TransportSuccess,
    UnknownUsage,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph._values import TaskAttemptId
from tests.graph_sdk.reference.publication_v3.reference import TextCase
from tests.graph_sdk.test_external_executor import _capacity_plan
from tests.graph_sdk.test_local_executor import _Clock

_CORPUS = Path(__file__).parent / "reference/publication_v3/cases.jsonl"
_BYTES = _CORPUS.read_bytes()
assert hashlib.sha256(_BYTES).hexdigest() == "10647ee490c7c530a3c2c9e879d2f274a0159aad49d790a291b5ef9ec29407e4"
_ALL = tuple(cast(TextCase, json.loads(line)) for line in _BYTES.splitlines())
_APPLICABLE = tuple(case for case in _ALL if "assessment_bad" not in case["input"][2])
# Correlation rejection precedes payload inspection and publication capacity.
# Keep every exact case; malformed groups retain valid and hostile construction
# inputs. Correlation mutation can remove those payloads from the response.
CASES = tuple(
    case
    for case in _APPLICABLE
    if case["input"][1] == "exact"
    or (tuple(case["input"][2]), tuple(case["input"][3]), case["input"][6])
    in {
        (("valid", "valid"), (2, 2, 2, 2, 2, 2), True),
        (("output_bad", "valid"), (0, 0, 0, 0, 0, 0), False),
    }
)


@dataclass
class _Transport:
    case: TextCase
    names: dict[SemanticAssociation, str] = field(default_factory=dict)

    async def dispatch(self, request: DispatchEnvelope) -> TransportSuccess:
        _, correlation, shapes, _, usage, _, aggregate = self.case["input"]
        rows: list[AssociationResult] = []
        for item in request.associations:
            assert isinstance(item.association, SemanticAssociation)
            name = self.names.setdefault(item.association, "AB"[len(self.names)])
            text = "x" if aggregate else "xx"
            if shapes["AB".index(name)] == "output_bad":
                text = chr(0xD800)
            rows.append(
                AssociationResult(
                    association=item.association,
                    outcome="ok",
                    consumed_context_ports=frozenset(),
                    outputs=(
                        PortArtifact(
                            port="output",
                            artifact_type=request.operation.outputs[0].artifact_type,
                            artifact=None,
                            value=TextArtifactValue(text=text),
                        ),
                    ),
                )
            )
        returned = tuple(reversed(rows))
        if correlation == "missing":
            returned = returned[:-1]
        elif correlation == "duplicate":
            returned += returned[:1]
        elif correlation in {"foreign", "extra"}:
            association = rows[0].association
            assert isinstance(association, SemanticAssociation)
            foreign = replace(
                rows[0], association=SemanticAssociation(task=TaskAttemptId.new(activation=association.task.activation))
            )
            returned = (*returned, foreign) if correlation == "extra" else (foreign, *returned[1:])
        result = TransportSuccess(
            results=returned,
            settlement=ExternalSettlement(
                request=request.request,
                disposition="completed" if usage == "exact:1" else "unknown",
                usage=ExactUsage(input_units=1, output_units=0) if usage == "exact:1" else UnknownUsage(),
                remote_stopped=True if usage == "exact:1" else None,
            ),
        )
        if correlation == "unlocalizable":
            object.__setattr__(result, "results", (object(), *returned[1:]))
        elif correlation == "outer":
            object.__setattr__(result, "results", list(returned))
        return result

    async def cancel(self, request: object) -> StopConfirmed:
        return StopConfirmed(usage=UnknownUsage())

    async def close(self) -> None:
        pass


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_frozen_text_publication(case: TextCase, monkeypatch: pytest.MonkeyPatch) -> None:
    budget_entries = payload_entries = 0
    original_init = _IngressBudget.__init__
    original_row = _IngressBudget.row

    def count_budget(budget: _IngressBudget, limits: IngressLimits) -> None:
        nonlocal budget_entries
        budget_entries += 1
        original_init(budget, limits)

    def count_payload(budget: _IngressBudget, row: AssociationResult, depth: int) -> bool:
        nonlocal payload_entries
        payload_entries += 1
        return original_row(budget, row, depth)

    monkeypatch.setattr(_IngressBudget, "__init__", count_budget)
    monkeypatch.setattr(_IngressBudget, "row", count_payload)
    asyncio.run(_compare(case))
    if case["input"][1] == "exact":
        # Live positive controls prevent a detached observer passing vacuously.
        assert budget_entries > 0 and payload_entries > 0
    else:
        assert budget_entries == 0 and payload_entries == 0


async def _compare(case: TextCase) -> None:
    mode, _, _, capacities, _, _, aggregate = case["input"]
    admitted, capability = _capacity_plan(
        shared=mode == "shared",
        mode="output_valid",
        ingress_limits=IngressLimits(max_text_bytes=16 if aggregate else 1),
    )
    transport = _Transport(case)
    running = await start_execution(
        admitted=admitted,
        capabilities=(capability,),
        services=ExecutionServices(
            handles=(
                ImplementationHandle(
                    implementation=capability.implementation,
                    operation=capability.operation,
                    configuration=capability.configuration,
                    local=None,
                    transport=transport,
                    resource=ResourceLease.create(
                        owner="sdk", safe_detachment="independent_after_dispatch", handle=transport
                    ),
                ),
            ),
            context_resources=(),
            limits=ExecutionLimits(
                max_local_in_flight=0,
                max_remote_outstanding=2,
                max_runtime_artifacts=capacities[0],
                max_runtime_artifact_bytes=32,
                max_collection_items=0,
            ),
            decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
            clock=_Clock(),
        ),
    )
    result = await running.wait()
    expected = case["expected"]
    assert result.requests.dispatched_count == expected["dispatches"]
    assert len(result.requests.settlements) == len(expected["settlements"])
    assert (not result.requests.remote_outstanding) == expected["stop_known"]
    settlements = result.requests.settlements
    exact_usage = sum(
        item.usage.input_units + item.usage.output_units for item in settlements if isinstance(item.usage, ExactUsage)
    )
    assert (exact_usage if all(isinstance(item.usage, ExactUsage) for item in settlements) else None) == expected[
        "exact_usage"
    ]
    summaries = sorted(
        (
            dict(
                association=transport.names[cast(SemanticAssociation, row.association)],
                outcome=row.outcome,
                category=row.category,
                failure=row.failure,
                condition=row.condition,
                consumed_context_ports=sorted(row.consumed_context_ports),
                artifacts=list(row.artifacts),
            )
            for terminal in result.requests.terminals
            for row in terminal.results
        ),
        key=lambda row: cast(str, row["association"]),
    )
    assert summaries == expected["summaries"]
    owners = {association.task.activation: name for association, name in transport.names.items()}
    survivors = sorted({owners[port.activation] for port in result.ports})
    failures = sorted(transport.names[fact.association] for fact in result.publication_failures)
    alternatives = [
        item
        for item in expected["alternatives"]
        if item["survivors"] == survivors and item["publication_failures"] == failures
    ]
    assert len(alternatives) == 1
    assert (
        len(result.artifacts)
        == len(result.ports)
        == len(result.provenance)
        == len(result.final_outputs)
        == len(survivors)
    )
    assert not result.assessments
    published = {port.artifact for port in result.ports}
    assert {fact.artifact for fact in result.provenance} == published
    assert {fact.candidate.artifact for fact in result.final_outputs} == published
    assert all(not row.artifacts for terminal in result.requests.terminals for row in terminal.results)
    if published:
        incomplete = copy(result)
        object.__setattr__(incomplete, "provenance", ())
        for view in inspect_requests(incomplete):
            if cast(SemanticAssociation, view.association).task.activation in {
                port.activation for port in result.ports
            }:
                assert not view.ownership_resolved and not view.artifacts


def test_reference_coverage_boundary_is_explicit() -> None:
    assert len(_ALL) == 1512 and len(_APPLICABLE) == 672 and len(CASES) == 144
    assert all("assessment_bad" in case["input"][2] for case in _ALL if case not in _APPLICABLE)
    exact = tuple(case["id"] for case in CASES if case["input"][1] == "exact")
    assert len(exact) == 96
    assert exact == tuple(case["id"] for case in _APPLICABLE if case["input"][1] == "exact")
    malformed = [case["input"] for case in CASES if case["input"][1] != "exact"]
    assert len(malformed) == 48
    groups = set(
        product(
            ("shared", "per_task"),
            ("missing", "duplicate", "extra", "foreign", "unlocalizable", "outer"),
            ("unknown", "exact:1"),
        )
    )
    assert {(mode, fault, usage) for mode, fault, _, _, usage, _, _ in malformed} == groups
    for mode, fault, usage in groups:
        assert {
            (tuple(shapes), tuple(capacities), aggregate)
            for actual_mode, actual_fault, shapes, capacities, actual_usage, _, aggregate in malformed
            if (actual_mode, actual_fault, actual_usage) == (mode, fault, usage)
        } == {
            (("valid", "valid"), (2, 2, 2, 2, 2, 2), True),
            (("output_bad", "valid"), (0, 0, 0, 0, 0, 0), False),
        }
