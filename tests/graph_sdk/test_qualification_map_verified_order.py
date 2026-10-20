# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from dataclasses import fields, replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk.evidence import (
    AssessmentSubmission,
    admit_qualification,
    evidence_revision_view,
)
from anonymizer.engine.graph_sdk.executor import LocalCompleted
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import OperationSpec
from tests.graph_sdk import qualification_map_execution as fixture
from tests.graph_sdk.qualification_map_callbacks import _ReferenceMapCallback
from tests.graph_sdk.test_map_item_boundary_witnesses import _qualification_limits

PORTS = {"N": "a_root_evidence", "MN": "z_first_evidence", "MN2": "m_second_evidence"}


def _named_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    original_replace = fixture.replace
    original_run = _ReferenceMapCallback.run
    original_admission = fixture.admit_execution_plan

    def operation(**kwargs: Any) -> OperationSpec:
        if kwargs["name"] in PORTS:
            port = PORTS[kwargs["name"]]
            names = {"evidence", *PORTS.values()}
            kwargs["outputs"] = tuple(
                original_replace(p, name=port) if p.name in names else p for p in kwargs["outputs"]
            )
            kwargs["output_dependencies"] = tuple(
                original_replace(d, output=port) if d.output in names else d for d in kwargs["output_dependencies"]
            )
            kwargs["outcomes"] = tuple(
                original_replace(
                    o,
                    produced_ports=frozenset(port if p in names else p for p in o.produced_ports),
                )
                for o in kwargs["outcomes"]
            )
        if kwargs["name"] in {"N", "root"}:
            kwargs["outcomes"] = tuple(
                original_replace(
                    o,
                    evidence=o.evidence
                    | frozenset(original_replace(p, name="A_SECOND") for p in o.evidence if p.name == "P"),
                )
                for o in kwargs["outcomes"]
            )
        if kwargs["name"] == "N" and not any(p.name == "0_root_last" for p in kwargs["outputs"]):
            evidence = next(p for p in kwargs["outputs"] if p.name == PORTS["N"])
            dependency = next(d for d in kwargs["output_dependencies"] if d.output == PORTS["N"])
            kwargs["outputs"] = (
                *kwargs["outputs"],
                original_replace(evidence, name="0_root_last"),
            )
            kwargs["output_dependencies"] = (
                *kwargs["output_dependencies"],
                original_replace(dependency, output="0_root_last"),
            )
            kwargs["outcomes"] = tuple(
                original_replace(o, produced_ports=o.produced_ports | {"0_root_last"}) for o in kwargs["outcomes"]
            )
        return OperationSpec(**kwargs)

    def replacing(obj: Any, **kwargs: Any) -> Any:
        if isinstance(obj, OperationSpec):
            return operation(**({f.name: getattr(obj, f.name) for f in fields(obj)} | kwargs))
        return original_replace(obj, **kwargs)

    def production(**kwargs: Any) -> Any:
        promise = kwargs["promise"]
        kwargs["evidence_port"] = PORTS["N" if promise == "P" else "MN2" if promise == "P_ITEM_2" else "MN"]
        return production_type(**kwargs)

    production_type = fixture.EvidenceProductionDecl

    async def run(self: _ReferenceMapCallback, request: Any) -> Any:
        output = await original_run(self, request)
        if not isinstance(output, LocalCompleted) or self.role not in {"N", "MN"}:
            return output
        port = PORTS["N" if self.role == "N" else "MN2" if self.item_promise == "P_ITEM_2" else "MN"]
        output = replace(
            output,
            results=tuple(
                replace(
                    r,
                    outputs=tuple(replace(p, port=port) if p.port == "evidence" else p for p in r.outputs),
                )
                for r in output.results
            ),
            assessments=tuple(replace(a, evidence_port=port) for a in output.assessments),
        )
        if self.role == "N":
            first = output.results[0]
            evidence = next(p for p in first.outputs if p.port == PORTS["N"])
            output = replace(
                output,
                results=(
                    replace(
                        first,
                        outputs=(replace(evidence, port="0_root_last"), *first.outputs),
                    ),
                ),
                assessments=(
                    *output.assessments,
                    replace(
                        output.assessments[0],
                        promise="A_SECOND",
                        evidence_port="0_root_last",
                    ),
                ),
            )
        return output

    def admission(**kwargs: Any) -> Any:
        original = next(p for p in kwargs["assessment_productions"] if p.promise == "P")
        kwargs["assessment_productions"] = (
            *kwargs["assessment_productions"],
            replace(original, promise="A_SECOND", evidence_port="0_root_last"),
        )
        kwargs["assessment_limits"] = replace(
            kwargs["assessment_limits"],
            max_productions=4,
            max_assessment_facts=6,
            max_port_facts=32,
            max_provenance_edges=32,
        )
        return original_admission(**kwargs)

    monkeypatch.setattr(fixture, "admit_execution_plan", admission)
    monkeypatch.setattr(fixture, "OperationSpec", operation)
    monkeypatch.setattr(fixture, "replace", replacing)
    monkeypatch.setattr(fixture, "EvidenceProductionDecl", production)
    monkeypatch.setattr(_ReferenceMapCallback, "run", run)


@pytest.mark.parametrize("count", [1, 2])
def test_two_maps_verified_tuple_uses_declared_ordinals(count: int, monkeypatch: pytest.MonkeyPatch) -> None:
    _named_ports(monkeypatch)
    execution, result, nodes = asyncio.run(fixture._execute_reference_map(count, two_maps=True))
    admitted = admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=replace(
            _qualification_limits(max_revision_entries=32),
            max_port_facts=64,
            max_provenance_edges=64,
        ),
    )
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(ref for ref, _ in result.artifacts),
        configurations=tuple(
            (item.node, item.capability.configuration) for item in execution.context.prepared.implementations
        ),
    )
    prepared = execution.context.prepared
    targets = {row.target: index for index, row in enumerate(prepared.target_occurrences)}
    declared_nodes = {node.id: index for index, node in enumerate(prepared.workflow.operations)}
    operations = {row.node: row.capability.operation for row in prepared.implementations}
    facts = result.assessments

    def key(fact: Any) -> tuple[int, int, int, int, int, int]:
        port = next(p for p in result.ports if p.activation == fact.activation and p.artifact == fact.evidence_artifact)
        ports = {p.name: index for index, p in enumerate(operations[fact.node].outputs)}
        return (
            targets[port.target],
            fact.activation.occurrence,
            declared_nodes[fact.node],
            ports[port.port],
            fact.evidence_artifact.key,
            fact.evidence_artifact.version,
        )

    expected = tuple(sorted(facts, key=key))
    assert len(expected) == 2 * count + 2
    roles = {node: name for name, node in nodes.items()}
    assert [roles[f.node] for f in expected].count("MN") == count
    assert [roles[f.node] for f in expected].count("MN2") == count
    assert [roles[f.node] for f in expected].count("N") == 2

    assert len({key(f)[1] for f in expected}) == len(expected) - 1
    assert {f.node for f in expected} == {nodes["N"], nodes["MN"], nodes["MN2"]}
    root_facts = tuple(f for f in expected if f.node == nodes["N"])
    assert tuple(f.promise for f in root_facts) == ("P", "A_SECOND")
    assert root_facts[0].evidence_artifact.key > root_facts[1].evidence_artifact.key
    assert tuple(f.promise for f in expected) != tuple(sorted(f.promise for f in expected))
    for order in (
        tuple(reversed(facts)),
        tuple(sorted(facts, key=lambda f: f.promise)),
        facts[1:] + facts[:1],
    ):
        output = qualify(
            admitted=admitted,
            result=result,
            current=current,
            submissions=tuple(AssessmentSubmission(fact=f) for f in order),
        )
        # Compare raw public tuple, without normalizing or sorting the returned evidence.
        assert tuple(v.reference.artifact for v in output.verified) == tuple(f.evidence_artifact for f in expected)
        assert tuple(v.activation for v in output.verified) == tuple(f.activation for f in expected)
        assert tuple(v.node for v in output.verified) == tuple(f.node for f in expected)
