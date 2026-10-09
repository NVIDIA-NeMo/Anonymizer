# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project actual P7 facts into the independently frozen tiny R4 input schema."""

from __future__ import annotations

from anonymizer.engine.graph_sdk.evidence import evidence_validity
from anonymizer.engine.graph_sdk.executor import OperationOutputKey
from anonymizer.engine.graph_sdk.records import CandidateRef, DecisionRef
from tests.graph_sdk.continuity_graph_fixture import GraphWitness
from tests.graph_sdk.reference.continuity.reference import Bundle, Decision, Evidence, Support, Target, _fixture_bundle


def project(witness: GraphWitness) -> Bundle:
    names = {target: name for name, target in witness.targets}
    nodes = dict(witness.decision_nodes)
    decision_names: dict[DecisionRef, str] = {}
    for key, node in nodes.items():
        activations = {port.activation for port in witness.result.ports if port.node == node}
        artifacts = {
            fact.artifact
            for fact in witness.result.provenance
            if isinstance(fact.key, OperationOutputKey) and fact.key.activation in activations and fact.decision
        }
        assert len(artifacts) == 1
        (artifact,) = artifacts
        decision_names[DecisionRef(artifact=artifact)] = key
    assert len(decision_names) == len(nodes)
    evidence_names = {}
    evidence_rows = []
    for fact in witness.qualified.verified:
        assert isinstance(fact.subject, CandidateRef)
        name = names[fact.subject.target]
        assert fact.node == witness.assessor and fact.outcome == name
        assert fact.reference not in evidence_names
        evidence_name = f"evidence:{name}:{fact.promise.name}"
        evidence_names[fact.reference] = evidence_name
        assert evidence_validity(evidence=fact, current=witness.current) == "current"
        assert fact.finding.status == "satisfied"
        assert all(isinstance(ref, DecisionRef) for ref in fact.reference.consumed)
        selected = tuple(
            ref
            for ref in witness.current.artifacts
            if (ref.invocation, ref.key) == (fact.reference.artifact.invocation, fact.reference.artifact.key)
        )
        assert len(selected) == 1
        evidence_rows.append(
            Evidence(
                reference=evidence_name,
                node="verifier",
                outcome=fact.outcome,
                promise=fact.promise.name,
                coverage=frozenset(f"{atom.kind}:{atom.name}" for atom in fact.coverage),
                consumed=frozenset(
                    decision_names[ref] for ref in fact.reference.consumed if isinstance(ref, DecisionRef)
                ),
                observed_revision=fact.reference.artifact.version,
                current_revision=selected[0].version,
            )
        )
    data = witness.execution.context.prepared.data
    targets = tuple(
        Target(
            name=names[output.target],
            required=frozenset(decision_names[ref] for ref in output.required_decisions),
            evidence=frozenset(evidence_names[ref] for ref in output.evidence),
            prerequisites=frozenset(
                names[edge.prerequisite] for edge in data.dependencies if edge.dependent == output.target
            ),
            atomic=frozenset(
                names[peer]
                for group in data.atomic
                if output.target in group
                for peer in group
                if peer != output.target
            ),
        )
        for output in witness.qualified.qualified
    )
    # Binding names/outcomes are admitted semantic inputs from GraphCase, never
    # inferred by accepting whichever actual evidence would keep a target alive.
    decisions = tuple(
        Decision(
            key=key,
            supports=(
                Support(
                    reference=f"stable-support:{key}",
                    node="verifier",
                    outcome=target,
                    promise=f"support_{target}",
                    coverage=frozenset(f"{atom.kind}:{atom.name}" for atom in witness.coverage),
                ),
            ),
        )
        for key, target in witness.case.support_targets
    )
    assert len(evidence_rows) == len(witness.qualified.verified)
    return _fixture_bundle(
        tuple(sorted(targets, key=lambda item: item.name)),
        tuple(sorted(decisions, key=lambda item: item.key)),
        tuple(sorted(evidence_rows, key=lambda item: item.reference)),
    )


def encode(bundle: Bundle) -> dict[str, object]:
    """Deterministic neutral inputs; no runtime IDs or inferred closure output."""
    return {
        "targets": [
            {
                "name": t.name,
                "required": sorted(t.required),
                "evidence": sorted(t.evidence),
                "prerequisites": sorted(t.prerequisites),
                "atomic": sorted(t.atomic),
            }
            for t in bundle.targets
        ],
        "decisions": [
            {
                "key": d.key,
                "supports": [
                    {
                        "reference": s.reference,
                        "node": s.node,
                        "outcome": s.outcome,
                        "promise": s.promise,
                        "coverage": sorted(s.coverage),
                    }
                    for s in d.supports
                ],
            }
            for d in bundle.decisions
        ],
        "verified": [
            {
                "reference": e.reference,
                "node": e.node,
                "outcome": e.outcome,
                "promise": e.promise,
                "coverage": sorted(e.coverage),
                "consumed": sorted(e.consumed),
                "observed_revision": e.observed_revision,
                "current_revision": e.current_revision,
            }
            for e in bundle.verified
        ],
    }
