# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare real execution and qualification with the independent R3 corpus.

The reference's ordinary ``v0`` names denote first-publication identities; SDK
ordinary artifacts use version 1. Materialized source versions retain their
literal version numbers. Normalization changes opaque names, never verdicts.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.capabilities import FrozenConfig, PreparationRejected
from anonymizer.engine.graph_sdk.evidence import (
    AssessmentSubmission,
    QualificationLimits,
    admit_qualification,
    evidence_revision_view,
    evidence_validity,
)
from anonymizer.engine.graph_sdk.executor import AdmittedExecutionPlan, AssessmentFinding, ExecutionResult
from anonymizer.engine.graph_sdk.preparation import StateRevision, StateRevisionView
from anonymizer.engine.graph_sdk.qualification import QualificationResult, qualify
from anonymizer.engine.graph_sdk.records import AbsenceRef, CandidateRef, DecisionRef
from anonymizer.graph._values import ActivationKey, ArtifactRef, ContractViolation, DatumId
from anonymizer.graph.workflow import CoverageAtom, NodeId, StateEffect
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context

CORPUS = json.loads((Path(__file__).parent / "reference/qualification_v1_cases.json").read_text())
BASELINE_IDS = frozenset(
    {
        "release/protection_success",
        "release/execution_only",
        "assessment/unsatisfied",
        "assessment/unknown",
        "assessment/missing",
        "assessment/duplicate",
        "commutation/independent_revisions",
        "validity/configuration_type_changed",
        "bounds/exact",
        "admission/production_one_over",
        "bounds/duplicate_artifacts_key",
        "bounds/duplicate_absences_key",
        "bounds/duplicate_configurations_key",
        "bounds/duplicate_state_key",
        "revision/invented_artifact",
        "revision/invented_absence",
        "revision/invented_configuration",
        "revision/invented_state",
        *(
            f"validity/{kind}_{state}"
            for kind in ("candidate", "evidence", "consumed")
            for state in ("current", "unknown")
        ),
        *(
            f"validity/{kind}_{state}"
            for kind in ("absence", "configuration", "state")
            for state in ("current", "stale", "unknown")
        ),
        *(
            f"bounds/{kind}_{boundary}"
            for kind in (
                "ports",
                "consumed",
                "edges",
                "submissions",
                "revisions",
                "absences",
                "coverage",
                "verified",
                "productions",
            )
            for boundary in ("exact", "one_over")
        ),
    }
)


class _BaselineNames:
    """Bind neutral names to actual occurrences before comparing any verdict."""

    def __init__(self, execution: AdmittedExecutionPlan, result: ExecutionResult) -> None:
        self.execution = execution
        self.result = result
        (self.entry,) = result.states[0].entries
        self.target = execution.context.prepared.target_occurrences[0].target
        self.targets = {
            datum.id: datum.text
            for datum in execution.context.prepared.data.datums
            if datum.id in execution.context.prepared.data.targets
        }
        self.entries = {}
        self.activation_targets = {}
        for target, state in zip(execution.context.prepared.target_occurrences, result.states, strict=True):
            (entry,) = state.entries
            self.entries[entry.activation] = entry
            self.activation_targets[entry.activation] = target.target
        self.fact = result.assessments[0] if result.assessments else None
        has_complete = any(production.promise == "complete" for production in execution.assessment_productions)
        self.promise_names = {"checked": "P_PART", "complete": "P_FULL"} if has_complete else {"checked": "P"}
        if has_complete:
            assert tuple(self.targets.values()) == ("A",)
        self.names = {
            port.artifact: "E2v0"
            if port.port == "evidence2"
            else f"{dict(subject='', context='X', evidence='E')[port.port]}{self.targets[port.target]}v0"
            for port in result.ports
            if port.port != "result"
        }
        assert len(self.names) == ((3 if result.assessments else 2) + has_complete) * len(self.targets)
        assert all(ref.version == 1 for ref in self.names)
        assert set(self.names) == {ref for ref, _ in result.artifacts}
        assert result.record.graph == execution.context.prepared.data.graph
        assert result.record.plan == execution.context.prepared.plan
        assert result.record.invocation.plan == result.record.plan
        assert result._execution is execution

    def artifact(self, ref: ArtifactRef) -> str:
        assert ref.invocation == self.result.record.invocation
        return self.names[ref]

    def activation(self, activation: ActivationKey) -> str:
        return f"ROOT:{self.targets[self.activation_targets[activation]]}"

    def target_name(self, target: DatumId) -> str:
        return self.targets[target]

    def normalize(self, output: QualificationResult) -> dict[str, Any]:
        result = self.result
        assert output._result is result
        assert output.record.plan == result.record.plan
        assert output.record.invocation == result.record.invocation
        assert output.record.graph == result.record.graph
        assert output.record.targets == frozenset(self.targets)
        statuses = {status.target: status for status in output.record.statuses}
        targets = []
        for target in output.targets:
            status = statuses[target.target]
            targets.append(
                {
                    "target": self.target_name(target.target),
                    "candidate": None if target.candidate is None else self.artifact(target.candidate.artifact),
                    "verified": sorted(self.artifact(item.artifact) for item in target.verified),
                    "required_decisions": sorted(self.artifact(item.artifact) for item in target.required_decisions),
                    "withholding": sorted(target.withholding),
                    "completion": status.completion,
                    "qualification": status.qualification,
                    "artifact_available": status.artifact_available,
                    "protection_available": status.protection_available,
                }
            )
        verified = []
        raw_order = []
        target_order = {
            item.target: index for index, item in enumerate(self.execution.context.prepared.target_occurrences)
        }
        port_order = {
            (item.node, port.name): index
            for item in self.execution.context.prepared.implementations
            for index, port in enumerate(item.capability.operation.outputs)
        }
        for item in output.verified:
            assert isinstance(item.subject, CandidateRef), "This adapter covers scalar candidate subjects"
            assert self.fact is not None
            assert item.node == self.entry.template
            evidence_port = next(
                production.evidence_port
                for production in self.execution.assessment_productions
                if production.promise == item.promise.name
            )
            # This fixture has one leaf node; its node ordinal is constant.
            raw_order.append(
                (
                    target_order[item.subject.target],
                    item.activation.occurrence,
                    port_order[item.node, evidence_port],
                    item.reference.artifact.key,
                    item.reference.artifact.version,
                )
            )
            assert item.promise.name in self.promise_names
            assert item.promise.meaning == "test assessment"
            assert item.environment.configuration == self.fact.environment.configuration
            assert item._result is result
            consumed = {}
            roles = {}
            for port, ref in item.consumed_by_port:
                assert not isinstance(ref, AbsenceRef)
                consumed[port] = self.artifact(ref.artifact if isinstance(ref, (CandidateRef, DecisionRef)) else ref)
                roles[port] = (
                    "candidate"
                    if isinstance(ref, CandidateRef)
                    else "decision"
                    if isinstance(ref, DecisionRef)
                    else "artifact"
                )
            verified.append(
                {
                    "activation": self.activation(item.activation),
                    "authenticated_factory": "EXEC:I0",
                    "node": "N",
                    "target": self.target_name(item.subject.target),
                    "outcome": item.outcome,
                    "promise": self.promise_names[item.promise.name],
                    "meaning": "privacy",
                    "subject_port": item.promise.subject_port,
                    "subject_artifact": self.artifact(item.subject.artifact),
                    "evidence_port": next(
                        production.evidence_port
                        for production in self.execution.assessment_productions
                        if production.promise == item.promise.name
                    ),
                    "evidence_artifact": self.artifact(item.reference.artifact),
                    "consumed": consumed,
                    "consumed_roles": roles,
                    "coverage": sorted(atom.name for atom in item.coverage),
                    "finding": item.finding.status,
                    "environment": {
                        "absences": {
                            f"Q{absence.query}": absence.scope_revision for absence in item.environment.absences
                        },
                        "configurations": {"N": "c0"},
                        "state": {
                            revision.effect.name: revision.revision for revision in item.environment.state.revisions
                        },
                    },
                    "validity": evidence_validity(evidence=item, current=output._current),
                }
            )
        assert raw_order == sorted(raw_order)
        memberships = {}
        for membership in output.record.memberships:
            assert membership.parent is None
            memberships["__ROOT__"] = {
                "parent": None,
                "target": None,
                "members": sorted(self.activation(member) for member in membership.members),
                "closed": membership.closed,
                "status": "closed" if membership.closed else "open",
                "expansion_outcome": None,
            }
        terminals = {}
        for terminal in output.record.terminals:
            assert terminal.attempt is not None
            assert terminal.attempt.activation == terminal.activation
            terminals[self.activation(terminal.activation)] = {
                "attempt": f"TASK:{self.activation(terminal.activation)}",
                "category": terminal.category,
                "outcome": self.entries[terminal.activation].outcome,
                "reasons": sorted(terminal.reasons),
                "structural": terminal.structural,
                "target": self.target_name(self.activation_targets[terminal.activation]),
            }
        targets.sort(key=lambda item: cast(str, item["target"]))
        return {
            "status": "accepted",
            "targets": targets,
            "qualified": [
                {
                    "target": self.target_name(item.target),
                    "candidate": self.artifact(item.candidate.artifact),
                    "evidence": sorted(self.artifact(ref.artifact) for ref in item.evidence),
                    "required_decisions": sorted(self.artifact(ref.artifact) for ref in item.required_decisions),
                }
                for item in sorted(output.qualified, key=lambda item: self.target_name(item.target))
            ],
            "verified": verified,
            "required_decisions": sorted(self.artifact(ref.artifact) for ref in output.required_decisions),
            "record": {
                "execution": {"factory": "EXEC:I0", "graph": "G0", "invocation": "I0", "plan": "P0"},
                "artifacts": sorted(self.artifact(ref) for ref in output.record.artifacts),
                "evidence": sorted(self.artifact(ref.artifact) for ref in output.record.evidence),
                "memberships": memberships,
                "terminals": terminals,
                "targets": targets,
            },
        }


def _comparison_order(value: dict[str, Any]) -> dict[str, Any]:
    """Compare opaque identities after each model checks its own ordinal order."""
    if "verified" not in value:
        return value
    return {
        **value,
        "verified": sorted(
            value["verified"],
            key=lambda row: (
                row["target"],
                row["activation"],
                row["node"],
                row["evidence_port"],
                row["evidence_artifact"],
            ),
        ),
    }


def _run_baseline(case: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = CORPUS[0]
    mutable_events = {"assessment", "assessment_submission", "revision", "seal_revision"}
    if case["boundary"] == "admission":
        assert not events
    else:
        expected_events = [event for event in baseline["events"] if event["kind"] not in mutable_events]
        if case["declaration"]["purpose"] == "execution_only":
            expected_events = [event for event in expected_events if event.get("ref", event.get("artifact")) != "EAv0"]
        assert [event for event in events if event["kind"] not in mutable_events] == expected_events, (
            "This fixture only executes the baseline operation topology."
        )
    declaration = case["declaration"]
    assert {
        key: value
        for key, value in declaration.items()
        if key not in {"limits", "purpose", "productions", "requirements"}
    } == {
        key: value
        for key, value in baseline["declaration"].items()
        if key not in {"limits", "purpose", "productions", "requirements"}
    }
    for event in events:
        if event["kind"] == "assessment":
            original = next(item for item in baseline["events"] if item["kind"] == "assessment")
            assert {key: value for key, value in event.items() if key != "finding"} == {
                key: value for key, value in original.items() if key != "finding"
            }
    if declaration["purpose"] == "execution_only":
        assert declaration["productions"] == declaration["requirements"] == []
    else:
        assert declaration["productions"] == baseline["declaration"]["productions"]
        assert declaration["requirements"] == baseline["declaration"]["requirements"]
    production = (declaration["productions"] or baseline["declaration"]["productions"])[0]
    assessments = [event for event in events if event["kind"] == "assessment"]
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            coverage=frozenset(CoverageAtom(kind="field", name=name) for name in production["coverage"]),
            requirement_coverage=frozenset(
                CoverageAtom(kind="field", name=name)
                for name in (declaration["requirements"] or baseline["declaration"]["requirements"])[0]["coverage"]
            ),
            environment=declaration["purpose"] != "execution_only",
            finding=AssessmentFinding(
                status=assessments[0]["finding"] if assessments else "satisfied", code="observed"
            ),
            execution_only=declaration["purpose"] == "execution_only",
            expose_evidence=False,
        )
    )
    return _qualify_flat(case, events, execution, result)


def _qualify_flat(
    case: dict[str, Any], events: list[dict[str, Any]], execution: AdmittedExecutionPlan, result: ExecutionResult
) -> dict[str, Any]:
    declaration = case["declaration"]
    assessments = [event for event in events if event["kind"] == "assessment"]
    names = _BaselineNames(execution, result)
    limits = QualificationLimits(
        **{field.name: declaration["limits"][field.name] for field in fields(QualificationLimits)}
    )
    admitted = admit_qualification(
        execution=execution,
        productions=() if declaration["purpose"] == "execution_only" else execution.assessment_productions,
        limits=limits,
    )
    assert case["boundary"] != "admission", "Admission-negative case unexpectedly admitted. "
    refs = {name: ref for ref, name in names.names.items()}
    revisions = [event for event in events if event["kind"] == "revision"]
    configuration = next(iter(execution.context.prepared.implementations)).capability.configuration
    unknown_artifacts = {
        event["key"]
        for event in revisions
        if event["collection"] == "artifacts" and f"{event['key']}v{event['value']}" not in refs
    }
    assert unknown_artifacts <= {"INVENTED"}, "Version cases require actual materialized lineage."
    if unknown_artifacts:
        refs["INVENTEDv0"] = ArtifactRef(
            invocation=result.record.invocation, key=max(ref.key for ref in names.names) + 1, version=1
        )

    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(
            refs[f"{event['key']}v{event['value']}"] for event in revisions if event["collection"] == "artifacts"
        ),
        absences=tuple(
            AbsenceRef(invocation=result.record.invocation, query=int(event["key"][1:]), scope_revision=event["value"])
            for event in revisions
            if event["collection"] == "absences"
        ),
        configurations=tuple(
            (
                names.entry.template if event["key"] == "N" else NodeId.new(workflow=names.entry.template.workflow),
                configuration if event["value"] == "c0" else FrozenConfig(fields=()),
            )
            for event in revisions
            if event["collection"] == "configurations"
        ),
        state=StateRevisionView.from_revisions(
            revisions=tuple(
                StateRevision(effect=StateEffect(kind="read", name=event["key"]), revision=event["value"])
                for event in revisions
                if event["collection"] == "state"
            )
        ),
    )
    actual_facts = {
        (names.activation(fact.activation), names.promise_names[fact.promise]): fact for fact in result.assessments
    }
    assert {(event["activation"], event["promise"]) for event in assessments} == set(actual_facts), (
        "Reference must retain every required P5 fact independently of P7 submission."
    )
    retained = {event["fact"]: actual_facts[event["activation"], event["promise"]] for event in assessments}
    assert len(retained) == len(assessments) == len(actual_facts)
    submissions = tuple(
        AssessmentSubmission(fact=retained[event["fact"]])
        for event in events
        if event["kind"] == "assessment_submission"
    )
    output = qualify(
        admitted=admitted,
        result=result,
        current=current,
        submissions=submissions,
    )
    return names.normalize(output)


@pytest.mark.parametrize(
    "case", [case for case in CORPUS if case["case_id"] in BASELINE_IDS], ids=lambda case: case["case_id"]
)
def test_baseline_reference_against_actual_execution(case: dict[str, Any]) -> None:
    for trace in [{"events": case["events"], "expected": case["expected"]}, *case["traces"]]:
        try:
            actual = _run_baseline(case, trace["events"])
        except (EffectRejected, ContractViolation, PreparationRejected) as error:
            actual = {"status": "rejected", "code": error.code.value}
        assert _comparison_order(actual) == _comparison_order(trace["expected"])


CORRUPTION_IDS = frozenset(
    {
        "assessment/unsupported_finding",
        "assessment/root_missing_terminal_unsubmitted",
        "assessment/root_missing_terminal_submitted",
        "membership/missing_root",
        "record/plan",
        "record/invocation",
        "record/graph",
        "joins/node_mismatch",
        "joins/outcome_mismatch",
        "joins/promise_mismatch",
        "authentication/missing_consumed_port_fact",
        "authentication/evidence_port_node",
        "final_output/producer_node",
        "final_output/producer_activation",
        "joins/missing_candidate_artifact",
        "joins/missing_evidence_artifact",
        "joins/invocation_mismatch",
        "joins/producer_artifact_mismatch",
        "authentication/evidence_port_target",
        "authentication/entry_node",
        "authentication/missing_absence_environment",
        "authentication/missing_configuration_environment",
        "authentication/missing_state_environment",
        "assessment/absence_query",
        "final_output/wrong_target",
        "final_output/wrong_outcome",
        "assessment/foreign_evidence",
        "assessment/caller_copy",
        "assessment/caller_replacement",
        "roles/distinct_evidence_wrong_role",
        "roles/consumed_unsupported_role",
        "structural/non_bool_flag",
        "structural/success_with_reason",
        "structural/structural_on_operation",
        "structural/operation_success_missing_attempt",
        "structural/operation_failed_missing_attempt",
        "structural/operation_cancelled_missing_attempt",
        "structural/operation_lost_missing_attempt",
        "provenance/swapped_dependency_parents",
        "provenance/parent_order_invariant",
        "provenance/extra_input_producer",
        "provenance/missing_producer",
        "provenance/duplicate_producer",
        "provenance/missing_parent_artifact",
        "provenance/mismatched_source_artifact",
        "joins/producer_target_mismatch",
    }
)


def _run_retained_corruption(case: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    """Mutate a real retained negative witness, never construct a positive result."""
    from anonymizer.graph._values import GraphId, InvocationId, PlanId
    from anonymizer.graph.workflow import NodeId
    from tests.graph_sdk.test_qualification import _inputs

    baseline = CORPUS[0]
    assert case["declaration"] == baseline["declaration"]
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            coverage=frozenset(CoverageAtom(kind="field", name=name) for name in ("K0", "K1")),
            requirement_coverage=frozenset({CoverageAtom(kind="field", name="K0")}),
            environment=True,
            expose_evidence=False,
            provenance_edge_limit=case["declaration"]["limits"]["max_provenance_edges"],
        )
    )
    names = _BaselineNames(execution, result)
    admitted, current, submissions = _inputs(execution, result)
    removed = [event for event in baseline["events"] if event not in events]
    added = [event for event in events if event not in baseline["events"]]
    if case["case_id"] in {
        "assessment/root_missing_terminal_unsubmitted",
        "assessment/root_missing_terminal_submitted",
    }:
        assert not added
        assert {event["kind"] for event in removed} == (
            {"terminal", "assessment_submission"} if case["case_id"].endswith("_unsubmitted") else {"terminal"}
        )
        object.__setattr__(result.record, "terminals", ())
        if case["case_id"].endswith("_unsubmitted"):
            submissions = ()
    elif case["case_id"] == "membership/missing_root":
        assert not added and len(removed) == 1 and removed[0]["kind"] == "membership" and removed[0]["parent"] is None
        object.__setattr__(result.record, "memberships", ())
    elif case["case_id"] in {
        "structural/operation_failed_missing_attempt",
        "structural/operation_cancelled_missing_attempt",
        "structural/operation_lost_missing_attempt",
    }:
        assert len(removed) == len(added) == 2
        changed = {event["kind"]: event for event in added}
        assert set(changed) == {"entry", "terminal"}
        terminal_event = changed["terminal"]
        assert terminal_event["attempt"] is None and terminal_event["structural"] is False
        (terminal,) = result.record.terminals
        object.__setattr__(names.entry, "status", changed["entry"]["state_category"])
        object.__setattr__(names.entry, "outcome", changed["entry"]["state_outcome"])
        object.__setattr__(terminal, "category", terminal_event["category"])
        object.__setattr__(terminal, "reasons", frozenset(terminal_event["reasons"]))
        object.__setattr__(terminal, "attempt", None)
    elif case["case_id"] == "provenance/duplicate_producer":
        assert not removed and not added and len(events) == len(baseline["events"]) + 1
        duplicated = next(event for event in events if events.count(event) == 2)
        assert duplicated["kind"] == "provenance" and duplicated["key"] == "OUT:A"
        fact = next(item for item in result.provenance if item.key == result.final_outputs[0].producer)
        object.__setattr__(result, "provenance", (*result.provenance, fact))
    elif case["case_id"] == "provenance/swapped_dependency_parents":
        assert len(removed) == len(added) == 2
        assert {event["kind"] for event in (*removed, *added)} == {"input_producer"}
        parents = {port: key for _, _, port, key in result._input_parents}
        assert set(parents) == {"subject", "context"}
        object.__setattr__(
            result,
            "_input_parents",
            tuple(
                (target, activation, port, parents["context" if port == "subject" else "subject"])
                for target, activation, port, _ in result._input_parents
            ),
        )
    elif case["case_id"] == "provenance/extra_input_producer":
        assert not removed and len(added) == 1 and added[0]["kind"] == "input_producer"
        target, activation, _, key = next(item for item in result._input_parents if item[2] == "subject")
        object.__setattr__(
            result, "_input_parents", (*result._input_parents, (target, activation, added[0]["port"], key))
        )
    elif added and added[0]["kind"] == "result_owner":
        assert not removed and len(added) == 1
        owner = added[0]
        assert owner["factory"] == "EXEC:I0"
        replacements = {
            "plan": PlanId.new(),
            "graph": GraphId.new(),
            "invocation": InvocationId.new(plan=result.record.plan),
        }
        changed = [field for field in replacements if owner[field] != baseline["declaration"]["execution"][field]]
        assert len(changed) == 1
        object.__setattr__(result.record, changed[0], replacements[changed[0]])
    else:
        assert len(removed) == 1 and len(added) <= 1
        original = removed[0]
        if original["kind"] == "assessment":
            assert len(added) == 1 and names.fact is not None
            changed = [field for field in original if original[field] != added[0][field]]
            assert len(changed) == 1
            field = changed[0]
            assert field in {
                "node",
                "outcome",
                "promise",
                "environment",
                "evidence_artifact",
                "authenticated_factory",
                "finding",
            }
            if field == "finding":
                object.__setattr__(names.fact.finding, "status", added[0][field])
            elif field == "authenticated_factory":
                assert added[0][field] == "CALLER"
                copied = object.__new__(type(names.fact))
                for member in fields(names.fact):
                    object.__setattr__(copied, member.name, getattr(names.fact, member.name))
                submissions = (AssessmentSubmission(fact=copied),)
            elif field == "evidence_artifact":
                assert added[0][field] not in names.names.values()
                foreign = ArtifactRef(
                    invocation=result.record.invocation, key=max(ref.key for ref, _ in result.artifacts) + 1, version=1
                )
                object.__setattr__(names.fact, "evidence_artifact", foreign)
            elif field == "environment":
                environment = names.fact.environment
                value = added[0][field]
                if value["absences"] != original[field]["absences"]:
                    object.__setattr__(
                        environment,
                        "absences",
                        frozenset(
                            AbsenceRef(invocation=result.record.invocation, query=int(key[1:]), scope_revision=revision)
                            for key, revision in value["absences"].items()
                        ),
                    )
                elif value["configurations"] != original[field]["configurations"]:
                    assert value["configurations"] == {}
                    object.__setattr__(environment, "configuration", None)
                else:
                    assert value["state"] == {}
                    object.__setattr__(environment, "state", StateRevisionView(revisions=frozenset()))
            else:
                value = NodeId.new(workflow=names.entry.template.workflow) if field == "node" else added[0][field]
                object.__setattr__(names.fact, field, value)
        elif original["kind"] == "terminal":
            assert len(added) == 1
            changed = [field for field in original if original[field] != added[0][field]]
            assert set(changed) <= {"structural", "reasons", "attempt"}
            (terminal,) = result.record.terminals
            for field in changed:
                value = frozenset(added[0][field]) if field == "reasons" else added[0][field]
                assert field != "attempt" or value is None
                object.__setattr__(terminal, field, value)
        elif original["kind"] == "artifact":
            ref = next(ref for ref, name in names.names.items() if name == original["ref"])
            if added:
                assert {key: value for key, value in original.items() if key != "invocation"} == {
                    key: value for key, value in added[0].items() if key != "invocation"
                }
                assert added[0]["invocation"] != original["invocation"]
                replacement = replace(ref, invocation=InvocationId.new(plan=result.record.plan))
                artifacts = tuple((replacement if key == ref else key, value) for key, value in result.artifacts)
            else:
                artifacts = tuple((key, value) for key, value in result.artifacts if key != ref)
            object.__setattr__(result, "artifacts", artifacts)
            object.__setattr__(result.record, "artifacts", frozenset(ref for ref, _ in artifacts))
        elif original["kind"] == "entry":
            assert len(added) == 1
            assert {key: value for key, value in original.items() if key != "node"} == {
                key: value for key, value in added[0].items() if key != "node"
            }
            object.__setattr__(names.entry, "template", NodeId.new(workflow=names.entry.template.workflow))
        elif original["kind"] == "final":
            assert len(added) == 1
            changed = [field for field in original if original[field] != added[0][field]]
            assert len(changed) == 1 and changed[0] in {"target", "outcome"}
            field = changed[0]
            value = DatumId.new(graph=result.record.graph) if field == "target" else added[0][field]
            object.__setattr__(result.final_outputs[0], field, value)
        elif original["kind"] == "provenance":
            provenance = next(fact for fact in result.provenance if fact.key.port == original["port"])
            if not added:
                object.__setattr__(
                    result, "provenance", tuple(fact for fact in result.provenance if fact is not provenance)
                )
            else:
                changed = [field for field in original if original[field] != added[0][field]]
                assert len(changed) == 1 and changed[0] in {"artifact", "target", "parents"}
                if changed[0] == "parents":
                    assert set(original["parents"]) == set(added[0]["parents"])
                    object.__setattr__(provenance, "parents", frozenset(reversed(tuple(provenance.parents))))
                elif changed[0] == "target":
                    object.__setattr__(
                        provenance, "key", replace(provenance.key, target=DatumId.new(graph=result.record.graph))
                    )
                else:
                    refs = {name: ref for ref, name in names.names.items()}
                    if added[0]["artifact"] == "MISSINGv0":
                        ref = ArtifactRef(
                            invocation=result.record.invocation,
                            key=max(ref.key for ref in refs.values()) + 1,
                            version=1,
                        )
                    else:
                        ref = refs[added[0]["artifact"]]
                    object.__setattr__(provenance, "artifact", ref)
        else:
            assert original["kind"] == "port"
            port = next(fact for fact in result.ports if fact.port == original["port"])
            if not added:
                object.__setattr__(result, "ports", tuple(fact for fact in result.ports if fact is not port))
            else:
                assert added[0]["kind"] == "port"
                changed = [field for field in original if original[field] != added[0][field]]
                assert len(changed) == 1 and changed[0] in {"node", "target", "activation", "role"}
                field = changed[0]
                if field == "role":
                    value = added[0][field]
                elif field == "activation":
                    value = replace(port.activation, occurrence=port.activation.occurrence + 100)
                elif field == "node":
                    value = NodeId.new(workflow=names.entry.template.workflow)
                else:
                    value = DatumId.new(graph=result.record.graph)
                object.__setattr__(port, field, value)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    return names.normalize(output)


@pytest.mark.parametrize(
    "case", [case for case in CORPUS if case["case_id"] in CORRUPTION_IDS], ids=lambda case: case["case_id"]
)
def test_reference_corruption_at_actual_retained_boundary(case: dict[str, Any]) -> None:
    for trace in [{"events": case["events"], "expected": case["expected"]}, *case["traces"]]:
        try:
            actual = _run_retained_corruption(case, trace["events"])
        except (EffectRejected, ContractViolation) as error:
            actual = {"status": "rejected", "code": error.code.value}
        assert _comparison_order(actual) == _comparison_order(trace["expected"])


PROPAGATION_IDS = frozenset(
    {
        "propagation/independent",
        "propagation/a_to_b",
        "propagation/b_to_a",
        "propagation/atomic_ab",
        "propagation/atomic_bc",
        "bounds/fixed_point_exact",
        "bounds/fixed_point_one_over",
    }
)


def _run_propagation(case: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    declaration = case["declaration"]
    baseline = CORPUS[0]["declaration"]
    assert declaration["productions"] == baseline["productions"]
    varying = {"targets", "dependencies", "atomic", "requirements", "root_outputs", "limits"}
    assert {key: value for key, value in declaration.items() if key not in varying} == {
        key: value for key, value in baseline.items() if key not in varying
    }
    labels = declaration["targets"]
    assert labels == ["A", "B", "C"]
    for field in ("requirements", "root_outputs"):
        assert len(declaration[field]) == len(labels)
        assert {item["target"] for item in declaration[field]} == set(labels)
        for item in declaration[field]:
            assert {key: value for key, value in item.items() if key != "target"} == {
                key: value for key, value in baseline[field][0].items() if key != "target"
            }
    assessments = {event["target"]: event for event in events if event["kind"] == "assessment"}
    if case["boundary"] == "admission":
        assert not events
    else:
        assert set(assessments) == set(labels)
    production = declaration["productions"][0]
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            target_labels=tuple(labels),
            target_findings=tuple(
                AssessmentFinding(status=assessments[label]["finding"] if assessments else "satisfied", code="observed")
                for label in labels
            ),
            dependencies=tuple(
                (labels.index(item["prerequisite"]), labels.index(item["dependent"]))
                for item in declaration["dependencies"]
            ),
            atomic_groups=tuple(
                tuple(labels.index(label) for label in group) for group in declaration["atomic"] if len(group) > 1
            ),
            coverage=frozenset(CoverageAtom(kind="field", name=name) for name in production["coverage"]),
            requirement_coverage=frozenset(
                CoverageAtom(kind="field", name=name) for name in declaration["requirements"][0]["coverage"]
            ),
            environment=True,
            expose_evidence=False,
        )
    )
    return _qualify_flat(case, events, execution, result)


@pytest.mark.parametrize(
    "case", [case for case in CORPUS if case["case_id"] in PROPAGATION_IDS], ids=lambda case: case["case_id"]
)
def test_reference_dependency_and_atomic_propagation(case: dict[str, Any]) -> None:
    for trace in [{"events": case["events"], "expected": case["expected"]}, *case["traces"]]:
        try:
            actual = _run_propagation(case, trace["events"])
        except (EffectRejected, ContractViolation) as error:
            actual = {"status": "rejected", "code": error.code.value}
        assert _comparison_order(actual) == _comparison_order(trace["expected"])


PARTIAL_PROMISE_IDS = frozenset(
    {
        "assessment/partial_promise_only",
        "assessment/complete_promise_only",
        "assessment/partial_and_complete_promises",
    }
)


def _run_partial_promises(case: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    declaration = case["declaration"]
    varying = {"productions", "output_dependencies", "requirements"}
    assert {key: value for key, value in declaration.items() if key not in varying} == {
        key: value for key, value in CORPUS[0]["declaration"].items() if key not in varying
    }
    assert declaration["output_dependencies"] == [
        *CORPUS[0]["declaration"]["output_dependencies"],
        {"node": "N", "outcome": "ok", "port": "evidence2", "inputs": ["context"], "identity_input": None},
    ]
    partial, complete = declaration["productions"]
    assert (partial["promise"], complete["promise"]) == ("P_PART", "P_FULL")
    assert (partial["evidence_port"], complete["evidence_port"]) == ("evidence", "evidence2")
    assert partial["coverage"] == ["K0"] and complete["coverage"] == ["K0", "K1"]
    invariant = {"promise", "evidence_port", "coverage"}
    baseline = CORPUS[0]["declaration"]["productions"][0]
    for production in (partial, complete):
        assert {key: value for key, value in production.items() if key not in invariant} == {
            key: value for key, value in baseline.items() if key not in invariant
        }
    assert declaration["targets"] == ["A"]
    assessments = [event for event in events if event["kind"] == "assessment"]
    assert len(assessments) == 2 and {event["finding"] for event in assessments} == {"satisfied"}
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            coverage=frozenset(CoverageAtom(kind="field", name=name) for name in partial["coverage"]),
            additional_coverage=frozenset(CoverageAtom(kind="field", name=name) for name in complete["coverage"]),
            requirement_coverage=frozenset(
                CoverageAtom(kind="field", name=name) for name in declaration["requirements"][0]["coverage"]
            ),
            environment=True,
            expose_evidence=False,
        )
    )
    return _qualify_flat(case, events, execution, result)


@pytest.mark.parametrize(
    "case", [case for case in CORPUS if case["case_id"] in PARTIAL_PROMISE_IDS], ids=lambda case: case["case_id"]
)
def test_reference_partial_and_complete_promises(case: dict[str, Any]) -> None:
    for trace in [{"events": case["events"], "expected": case["expected"]}, *case["traces"]]:
        actual = _run_partial_promises(case, trace["events"])
        assert _comparison_order(actual) == _comparison_order(trace["expected"])
