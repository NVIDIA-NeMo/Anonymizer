# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference comparison and record naming for map qualification."""

from __future__ import annotations

import json
from dataclasses import fields
from typing import Any

from anonymizer.engine.graph_sdk.evidence import (
    AssessmentSubmission,
    MapItemSubjectRef,
    QualificationLimits,
    admit_qualification,
    evidence_revision_view,
    evidence_validity,
)
from anonymizer.engine.graph_sdk.executor import (
    AdmittedExecutionPlan,
    ExecutionResult,
    MapItemKey,
    OperationOutputKey,
    RootInputKey,
)
from anonymizer.engine.graph_sdk.qualification import QualificationResult, qualify
from anonymizer.engine.graph_sdk.records import CandidateRef
from anonymizer.graph import ActivationKey, ArtifactRef, NodeId
from tests.graph_sdk.qualification_map_execution import CORPUS, _execute_reference_map

# The independent corpus also models absence and state-revision environments;
# the SDK retains only selected configurations, so comparisons drop the rest.
_UNMODELED_ENVIRONMENT = frozenset({"absences", "state"})


def _retained_environment(environment: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in environment.items() if key not in _UNMODELED_ENVIRONMENT}


def _without_decisions(row: dict[str, Any]) -> dict[str, Any]:
    # The independent corpus also models required decisions; the SDK has no decision waits.
    return {key: value for key, value in row.items() if key != "required_decisions"}


def _retained_expectation(expected: dict[str, Any]) -> dict[str, Any]:
    """Project a frozen expectation onto the dimensions the SDK retains."""
    if "verified" not in expected:
        return expected
    return {
        **_without_decisions(expected),
        "targets": [_without_decisions(row) for row in expected["targets"]],
        "qualified": [_without_decisions(row) for row in expected["qualified"]],
        "verified": [{**row, "environment": _retained_environment(row["environment"])} for row in expected["verified"]],
        "record": {
            **expected["record"],
            "targets": [_without_decisions(row) for row in expected["record"]["targets"]],
        },
    }


async def _run_reference_map_case(case: dict[str, Any]) -> dict[str, Any]:
    events = case["events"]
    dynamic = case["case_id"].startswith("assessment/dynamic_")
    member_promise = "P_MEMBER" if dynamic else "P_ITEM"
    count = sum(event["kind"] == "entry" and event["node"] == "MN" for event in events)
    consumed_only = case["case_id"] == "map_item_evidence/typed_consumed_endpoint"
    member_failure = case["case_id"] in {"map_item_evidence/member_non_success", "assessment/dynamic_started_failure"}
    expander_failure = case["case_id"] == "assessment/dynamic_unreached_failed_expansion"
    blocked_member = case["case_id"] == "assessment/dynamic_blocked_unreached"
    versioned_items = case["case_id"] == "map_item_evidence/item_stale"
    candidate_uses_membership = case["case_id"] != "map_item_evidence/different_final_ancestry"
    candidate_passthrough = case["case_id"] == "map_item_evidence/candidate_passthrough_unrelated"
    alternate_expansion = case["case_id"] == "map_item_evidence/distinct_outcome_port"
    nested = case["case_id"] == "map_item_evidence/nested_path"
    two_maps = case["case_id"] in {
        "map_item_evidence/two_independent_maps",
        "map_item_evidence/two_maps_no_crossproduct",
    }
    baseline: dict[str, Any] = (
        case
        if consumed_only
        or member_failure
        or expander_failure
        or blocked_member
        or versioned_items
        or not candidate_uses_membership
        or candidate_passthrough
        or alternate_expansion
        or nested
        else next(item for item in CORPUS if item["case_id"] == f"map_item_evidence/direct_{count}")
    )
    if two_maps:
        baseline = next(item for item in CORPUS if item["case_id"] == "map_item_evidence/two_independent_maps")
    if dynamic:
        # The dynamic family names the same member promise differently.
        baseline = json.loads(json.dumps(baseline).replace("P_ITEM", member_promise))
    mutable_events = {"assessment_submission", "assessment", "revision"}
    actual_immutable = [event for event in events if event["kind"] not in mutable_events]
    baseline_immutable = [event for event in baseline["events"] if event["kind"] not in mutable_events]
    if dynamic:
        # These retained owner facts are unordered; occurrence identities remain exact.
        actual_immutable.sort(key=lambda event: json.dumps(event, sort_keys=True))
        baseline_immutable.sort(key=lambda event: json.dumps(event, sort_keys=True))
    assert actual_immutable == baseline_immutable
    declaration_baseline = baseline["declaration"]
    if dynamic:
        # Failure events differ, but their declaration still uses the independent
        # ordinary map contract, plus the explicit failed prerequisite owner.
        declaration_baseline = json.loads(
            json.dumps(
                next(item["declaration"] for item in CORPUS if item["case_id"] == "map_item_evidence/direct_0")
            ).replace("P_ITEM", member_promise)
        )
        if blocked_member:
            declaration_baseline["node_kinds"]["FAILED_SOURCE"] = "operation"
    assert {key: value for key, value in case["declaration"].items() if key != "limits"} == {
        key: value for key, value in declaration_baseline.items() if key != "limits"
    }
    execution, result, nodes = await _execute_reference_map(
        count,
        consumed_only=consumed_only,
        member_failure=member_failure,
        expander_failure=expander_failure,
        versioned_items=versioned_items,
        candidate_uses_membership=candidate_uses_membership,
        candidate_passthrough=candidate_passthrough,
        alternate_expansion=alternate_expansion,
        nested=nested,
        two_maps=two_maps,
        member_promise=member_promise,
        blocked_member=blocked_member,
    )
    assert (
        len(result.states[0].entries)
        == len(result.record.terminals)
        == count + 3 + int(nested) + int(blocked_member) + (count + 2 if two_maps else 0)
    )
    names = _MapRecordNames(execution, result, nodes)
    names.assert_retained_facts(events)
    names.assert_assessments(baseline["events"])
    facts = {
        ("F:A:P" if fact.node == nodes["N"] else f"F:{names.activation(fact.activation)}:{fact.promise}"): fact
        for fact in result.assessments
    }
    assert {event["fact"] for event in events if event["kind"] == "assessment"} == set(facts)
    names.assert_assessments(events)
    if any(event["kind"] == "assessment_submission" and event["fact"] not in facts for event in events):
        assert case["case_id"] in {"map_item_evidence/foreign_submission", "assessment/dynamic_submission_foreign"}
        _, foreign, foreign_nodes = await _execute_reference_map(1, member_promise=member_promise)
        facts[f"F:FOREIGN:{member_promise}"] = next(
            fact for fact in foreign.assessments if fact.node == foreign_nodes["MN"]
        )
    admitted = admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=QualificationLimits(
            **{field.name: case["declaration"]["limits"][field.name] for field in fields(QualificationLimits)}
        ),
    )
    revisions = [event for event in events if event["kind"] == "revision"]
    artifacts = {name: ref for ref, name in names.artifacts.items()}
    configurations = {item.node: item.capability.configuration for item in execution.context.prepared.implementations}
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(
            artifacts[f"{event['key']}v{event['value']}"] for event in revisions if event["collection"] == "artifacts"
        ),
        configurations=tuple(
            (nodes[event["key"]], configurations[nodes[event["key"]]])
            for event in revisions
            if event["collection"] == "configurations"
        ),
    )
    submissions = tuple(
        AssessmentSubmission(fact=facts[event["fact"]]) for event in events if event["kind"] == "assessment_submission"
    )
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    return names.normalize(output)


class _MapRecordNames:
    """Name every retained occurrence without dropping joins or producer facts."""

    def __init__(self, execution: AdmittedExecutionPlan, result: ExecutionResult, nodes: dict[str, NodeId]) -> None:
        self.execution = execution
        self.result = result
        self.nodes = {node: name for name, node in nodes.items()}
        self.entries = {entry.activation: entry for state in result.states for entry in state.entries}
        members = sorted(
            (entry for entry in self.entries.values() if entry.template == nodes["MN"]),
            key=lambda entry: entry.activation.occurrence,
        )
        self.activations = {entry.activation: f"M{index}" for index, entry in enumerate(members)}
        for entry in self.entries.values():
            if entry.template not in {nodes["MN"], nodes.get("MN2")}:
                self.activations[entry.activation] = {
                    "N": "ROOT:A",
                    "EXP": "MAP",
                    "J": "JOIN",
                    "SG": "WRAP",
                    "EXP2": "MAP2",
                    "J2": "JOIN2",
                    "FAILED_SOURCE": "FAILED",
                }[self.nodes[entry.template]]
        if "MN2" in nodes:
            second_members = sorted(
                (entry for entry in self.entries.values() if entry.template == nodes["MN2"]),
                key=lambda entry: entry.activation.occurrence,
            )
            self.activations.update({entry.activation: f"Z{index}" for index, entry in enumerate(second_members)})
        self.artifacts: dict[ArtifactRef, str] = {
            fact.artifact: {"subject": "Av0", "context": "XAv0"}[fact.key.port]
            for fact in result.provenance
            if isinstance(fact.key, RootInputKey)
        }
        for fact in result.provenance:
            if isinstance(fact.key, MapItemKey):
                prefix = "MJ" if self.entries[fact.key.member].template == nodes.get("MN2") else "MI"
                self.artifacts[fact.artifact] = f"{prefix}{fact.key.item_key}v{fact.key.item_version}"
        for port in result.ports:
            node = self.nodes[port.node]
            if node == "N":
                name = {
                    "subject": "Av0",
                    "result": "Av0",
                    "context": "XAv0",
                    "evidence": "EAv0",
                    "membership": "CAv0",
                    "membership2": "CBv0",
                }[port.port]
            elif node == "SG":
                name = "XAv0" if port.port == "context" else "CAv0"
            elif node in {"EXP", "EXP2"}:
                name = "XAv0" if port.port == "context" else "CBv0" if node == "EXP2" else "CAv0"
            else:
                assert node in {"MN", "MN2"}
                index = self.activations[port.activation][1:]
                if port.port in {"item", "item2"}:
                    key = next(
                        fact.key
                        for fact in result.provenance
                        if isinstance(fact.key, MapItemKey) and fact.key.member == port.activation
                    )
                    name = f"{'MJ' if node == 'MN2' else 'MI'}{key.item_key}v{key.item_version}"
                else:
                    name = "Av0" if port.port == "subject" else f"{'MF' if node == 'MN2' else 'ME'}{index}v0"
            if port.artifact in self.artifacts:
                assert self.artifacts[port.artifact] == name
            self.artifacts[port.artifact] = name
        assert len(set(self.artifacts.values())) == len(self.artifacts)
        assert set(self.artifacts) == {ref for ref, _ in result.artifacts} == set(result.record.artifacts)
        assert all(
            ref.version == (int(name.rsplit("v", 1)[1]) if name.startswith(("MI", "MJ")) else 1)
            for ref, name in self.artifacts.items()
        )
        self.producers = {
            fact.key: f"{'MAPITEM2' if self.activations[fact.key.member].startswith('Z') else 'MAPITEM'}:{self.activations[fact.key.member][1:]}"
            for fact in result.provenance
            if isinstance(fact.key, MapItemKey)
        }
        assert result.record.graph == execution.context.prepared.data.graph
        assert result.record.plan == execution.context.prepared.plan
        assert result.record.invocation.plan == result.record.plan
        assert result._execution is execution

    def assert_retained_facts(self, events: list[dict[str, Any]]) -> None:
        expected_ports = {
            (event["activation"], event["node"], event["port"], event["artifact"], event["role"])
            for event in events
            if event["kind"] == "port"
        }
        actual_ports = {
            (
                self.activation(port.activation),
                self.nodes[port.node],
                port.port,
                self.artifact(port.artifact),
                port.role,
            )
            for port in self.result.ports
        }
        assert len(actual_ports) == len(self.result.ports)
        assert actual_ports == expected_ports
        producers: dict[object, str] = {key: value for key, value in self.producers.items()}
        for fact in self.result.provenance:
            key = fact.key
            if isinstance(key, RootInputKey):
                producers[key] = f"ROOT:A:{key.port}"
            elif isinstance(key, OperationOutputKey):
                activation = self.activation(key.activation)
                if activation in {"MAP", "MAP2"}:
                    producers[key] = f"OP:{activation}:A:{key.port}"
                elif activation == "WRAP":
                    producers[key] = f"SGOUT:WRAP:A:{key.port}"
                elif activation == "ROOT:A":
                    producers[key] = "OUT:A" if key.port == "result" else "EVID:A"
                else:
                    assert key.port == "evidence"
                    producers[key] = f"EVID:{activation}"
            else:
                assert isinstance(key, MapItemKey)
        assert len(producers) == len(self.result.provenance)
        actual_inputs = {
            (self.activation(activation), port, producers[producer])
            for _, activation, port, producer in self.result._input_parents
        }
        expected_inputs = {
            (event["activation"], event["port"], event["producer"])
            for event in events
            if event["kind"] == "input_producer"
        }
        assert len(actual_inputs) == len(self.result._input_parents)
        assert actual_inputs == expected_inputs
        actual_provenance = {
            producers[fact.key]: {
                "artifact": self.artifact(fact.artifact),
                "parents": sorted(producers[parent] for parent in fact.parents),
            }
            for fact in self.result.provenance
        }
        # The corpus also flags decision producers; the SDK has none, so none may be expected.
        assert not any(event["decision"] for event in events if event["kind"] == "provenance")
        expected_provenance = {
            event["key"]: {
                "artifact": event["artifact"],
                "parents": sorted(event["parents"]),
            }
            for event in events
            if event["kind"] == "provenance"
        }
        assert actual_provenance == expected_provenance
        for fact in self.result.provenance:
            if isinstance(fact.key, MapItemKey):
                event = next(
                    event for event in events if event["kind"] == "provenance" and event["key"] == producers[fact.key]
                )
                assert (
                    fact.key.item_key,
                    fact.key.item_version,
                    self.activation(fact.key.expander),
                    self.activation(fact.key.member),
                    fact.key.port,
                ) == (event["item_key"], event["item_version"], event["expander"], event["member"], event["port"])

    def assert_assessments(self, events: list[dict[str, Any]]) -> None:
        expected = [
            {**event, "environment": _retained_environment(event["environment"])}
            for event in events
            if event["kind"] == "assessment"
        ]
        actual: list[dict[str, Any]] = []
        target = self.execution.context.prepared.target_occurrences[0].target
        for fact in self.result.assessments:
            node = self.nodes[fact.node]
            activation = self.activation(fact.activation)
            selected = next(item for item in self.execution.context.prepared.implementations if item.node == fact.node)
            outcome = next(
                outcome for outcome in selected.capability.operation.outcomes if outcome.name == fact.outcome
            )
            promise = next(promise for promise in outcome.evidence if promise.name == fact.promise)
            production = next(
                production
                for production in self.execution.assessment_productions
                if production.node == fact.node and production.promise == fact.promise
            )
            assert fact.environment.configuration == selected.capability.configuration
            assert fact.finding in production.supported_findings
            ports = {port.port: port for port in self.result.ports if port.activation == fact.activation}
            assert all(port.target == target for port in ports.values())
            assert isinstance(promise.subject_port, str)
            assert all(isinstance(port, str) for port in promise.consumed_ports)
            assert all(atom.kind == "field" for atom in promise.coverage)
            actual.append(
                {
                    "kind": "assessment",
                    "fact": "F:A:P" if node == "N" else f"F:{activation}:{fact.promise}",
                    "activation": activation,
                    "authenticated_factory": "EXEC:I0",
                    "node": node,
                    "target": "A",
                    "outcome": fact.outcome,
                    "promise": fact.promise,
                    "subject_port": promise.subject_port,
                    "subject_artifact": self.artifact(ports[promise.subject_port].artifact),
                    "evidence_port": production.evidence_port,
                    "evidence_artifact": self.artifact(fact.evidence_artifact),
                    "consumed": {
                        port: self.artifact(ports[port].artifact)
                        for port in promise.consumed_ports
                        if isinstance(port, str)
                    },
                    "coverage": sorted(atom.name for atom in promise.coverage),
                    "finding": fact.finding.status,
                    "environment": {"configurations": {node: "c0"}},
                }
            )
        assert sorted(actual, key=lambda item: item["fact"]) == sorted(expected, key=lambda item: item["fact"])

    def artifact(self, ref: ArtifactRef) -> str:
        assert ref.invocation == self.result.record.invocation
        return self.artifacts[ref]

    def activation(self, key: ActivationKey) -> str:
        return self.activations[key]

    def normalize(self, output: QualificationResult) -> dict[str, Any]:
        assert output._result is self.result
        # Actual reservation ordinals are local to each admitted plan. Check
        # its public tuple before renaming identities for the neutral model.
        order = [(item.activation.occurrence, item.reference.artifact.key) for item in output.verified]
        assert order == sorted(order)
        (target,) = output.targets
        (status,) = output.record.statuses
        assert target.target == status.target == self.execution.context.prepared.target_occurrences[0].target
        target_row = {
            "target": "A",
            "candidate": None if target.candidate is None else self.artifact(target.candidate.artifact),
            "verified": sorted(self.artifact(ref.artifact) for ref in target.verified),
            "withholding": sorted(target.withholding),
            "completion": status.completion,
            "qualification": status.qualification,
            "artifact_available": status.artifact_available,
            "protection_available": status.protection_available,
        }
        verified = []
        for item in output.verified:
            assert item._result is self.result
            node = self.nodes[item.node]
            fact = next(fact for fact in self.result.assessments if fact.activation == item.activation)
            assert item.environment.configuration == fact.environment.configuration
            assert item.promise.name == ("P" if node == "N" else "P_ITEM_2" if node == "MN2" else fact.promise)
            assert item.promise.meaning == (
                "privacy" if node == "N" else "item_privacy_2" if node == "MN2" else "item_privacy"
            )
            consumed = {}
            roles = {}
            for port, ref in item.consumed_by_port:
                consumed[port] = self.artifact(ref.artifact if isinstance(ref, CandidateRef) else ref)
                roles[port] = "candidate" if isinstance(ref, CandidateRef) else "artifact"
            row = {
                "activation": self.activation(item.activation),
                "authenticated_factory": "EXEC:I0",
                "node": node,
                "target": "A",
                "outcome": item.outcome,
                "promise": item.promise.name,
                "meaning": item.promise.meaning,
                "subject_port": item.promise.subject_port,
                "subject_artifact": self.artifact(item.subject.artifact),
                "evidence_port": next(
                    production.evidence_port
                    for production in self.execution.assessment_productions
                    if production.node == item.node and production.promise == item.promise.name
                ),
                "evidence_artifact": self.artifact(item.reference.artifact),
                "consumed": consumed,
                "consumed_roles": roles,
                "coverage": sorted(atom.name for atom in item.coverage),
                "finding": item.finding.status,
                "environment": {"configurations": {node: "c0"}},
                "validity": evidence_validity(evidence=item, current=output._current),
            }
            if isinstance(item.subject, MapItemSubjectRef):
                row["subject"] = {
                    "artifact": self.artifact(item.subject.artifact),
                    "producer": self.producers[item.subject.producer],
                }
                assert item.subject.producer.target == target.target
            else:
                assert item.subject.target == target.target
            verified.append(row)
        memberships = {}
        for membership in output.record.memberships:
            name = "__ROOT__" if membership.parent is None else self.activation(membership.parent)
            expansion = next(
                (
                    expansion
                    for state in self.result.states
                    for expansion in state.expansions
                    if expansion.parent == membership.parent
                ),
                None,
            )
            memberships[name] = {
                "parent": None if membership.parent is None else name,
                "target": None if membership.parent is None else "A",
                "members": sorted(self.activation(key) for key in membership.members),
                "closed": membership.closed,
                "status": expansion.status if expansion is not None else "closed" if membership.closed else "open",
                "expansion_outcome": None if expansion is None else self.entries[expansion.parent].outcome,
            }
        terminals = {}
        for terminal in output.record.terminals:
            name = self.activation(terminal.activation)
            if terminal.attempt is not None:
                assert terminal.attempt.activation == terminal.activation
            terminals[name] = {
                "attempt": None if terminal.attempt is None else f"TASK:{name}",
                "category": terminal.category,
                "outcome": self.entries[terminal.activation].outcome,
                "reasons": sorted(terminal.reasons),
                "structural": terminal.structural,
                "target": "A",
            }
        return {
            "status": "accepted",
            "targets": [target_row],
            "qualified": [
                {
                    "target": "A",
                    "candidate": self.artifact(item.candidate.artifact),
                    "evidence": sorted(self.artifact(ref.artifact) for ref in item.evidence),
                }
                for item in output.qualified
            ],
            "verified": sorted(verified, key=lambda row: (row["target"], row["activation"], row["evidence_artifact"])),
            "record": {
                "execution": {"factory": "EXEC:I0", "graph": "G0", "invocation": "I0", "plan": "P0"},
                "artifacts": sorted(self.artifact(ref) for ref in output.record.artifacts),
                "evidence": sorted(self.artifact(ref.artifact) for ref in output.record.evidence),
                "memberships": memberships,
                "terminals": terminals,
                "targets": [target_row],
            },
        }
