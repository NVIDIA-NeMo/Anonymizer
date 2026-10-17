# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare real execution and qualification with the independent R3 corpus.

The reference's ordinary ``v0`` names denote first-publication identities; SDK
ordinary artifacts use version 1. Materialized source versions retain their
literal version numbers. Normalization changes opaque names, never verdicts.
"""

from __future__ import annotations

import asyncio
from dataclasses import fields, replace
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk.capabilities import FrozenConfig, PreparationRejected
from anonymizer.engine.graph_sdk.evidence import (
    AssessmentSubmission,
    QualificationLimits,
    admit_qualification,
    evidence_revision_view,
    evidence_validity,
)
from anonymizer.engine.graph_sdk.executor import AdmittedExecutionPlan, AssessmentFinding, ExecutionResult
from anonymizer.engine.graph_sdk.qualification import QualificationResult, qualify
from anonymizer.engine.graph_sdk.records import CandidateRef
from anonymizer.graph import EffectRejected
from anonymizer.graph._values import ActivationKey, ArtifactRef, ContractViolation, DatumId
from anonymizer.graph.workflow import CoverageAtom, NodeId
from tests.graph_sdk.reference.corpora import load_cases
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context

CORPUS = load_cases("qualification")
BASELINE_IDS = frozenset(
    {
        "release/protection_success",
        "release/execution_only",
        "release/execution_only_empty",
        "validity/evidence_output_removed",
        "assessment/unsatisfied",
        "assessment/unknown",
        "assessment/missing",
        "assessment/duplicate",
        "commutation/independent_revisions",
        "validity/configuration_type_changed",
        "bounds/exact",
        "admission/production_one_over",
        "bounds/duplicate_artifacts_key",
        "bounds/duplicate_configurations_key",
        "revision/invented_artifact",
        "revision/invented_configuration",
        *(
            f"validity/{kind}_{state}"
            for kind in ("candidate", "evidence", "consumed")
            for state in ("current", "unknown")
        ),
        *(f"validity/{kind}_{state}" for kind in ("configuration",) for state in ("current", "stale", "unknown")),
        *(
            f"bounds/{kind}_{boundary}"
            for kind in (
                "ports",
                "consumed",
                "edges",
                "submissions",
                "revisions",
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
                consumed[port] = self.artifact(ref.artifact if isinstance(ref, CandidateRef) else ref)
                roles[port] = "candidate" if isinstance(ref, CandidateRef) else "artifact"
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
                    "environment": {"configurations": {"N": "c0"}},
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
                }
                for item in sorted(output.qualified, key=lambda item: self.target_name(item.target))
            ],
            "verified": verified,
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
    """Compare opaque identities after each model checks its own ordinal order.

    The independent corpus also models absence and state-revision environments
    and required decisions; the SDK retains only selected configurations and has
    no decision waits, so rows compare without those dimensions.
    """
    if "verified" not in value:
        return value

    def without_decisions(row: dict[str, Any]) -> dict[str, Any]:
        return {key: item for key, item in row.items() if key != "required_decisions"}

    return {
        **without_decisions(value),
        "targets": [without_decisions(row) for row in value["targets"]],
        "qualified": [without_decisions(row) for row in value["qualified"]],
        "record": {**value["record"], "targets": [without_decisions(row) for row in value["record"]["targets"]]},
        "verified": sorted(
            (
                {
                    **row,
                    "environment": {
                        key: item for key, item in row["environment"].items() if key not in {"absences", "state"}
                    },
                }
                for row in value["verified"]
            ),
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
    # The reference revision bound also counts state revisions, which the SDK
    # does not model; charge the product only for the entries it receives.
    state_entries = sum(event["kind"] == "revision" and event["collection"] == "state" for event in events)
    limits = replace(limits, max_revision_entries=limits.max_revision_entries - state_entries)
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
        configurations=tuple(
            (
                names.entry.template if event["key"] == "N" else NodeId.new(workflow=names.entry.template.workflow),
                configuration if event["value"] == "c0" else FrozenConfig(fields=()),
            )
            for event in revisions
            if event["collection"] == "configurations"
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


# Corruptions that are only reachable by writing into a frozen retained terminal (object.__setattr__);
# checked construction (OQ-2) makes these unrepresentable, so they no longer bind the retained boundary.
FROZEN_VALUE_TAMPER_IDS = frozenset(
    {
        "structural/non_bool_flag",
        "structural/success_with_reason",
        "joins/missing_candidate_artifact",
        "provenance/swapped_dependency_parents",
        "provenance/extra_input_producer",
        "assessment/root_missing_terminal_unsubmitted",
        "assessment/root_missing_terminal_submitted",
        "authentication/entry_node",
        "provenance/missing_producer",
        "provenance/duplicate_producer",
        "final_output/producer_activation",
        "provenance/missing_parent_artifact",
        "assessment/foreign_evidence",
        "authentication/evidence_port_node",
        "authentication/evidence_port_target",
        "authentication/missing_consumed_port_fact",
        "authentication/terminal_outcome",
        "final_output/producer_node",
        "final_output/wrong_outcome",
        "final_output/wrong_target",
        "joins/invocation_mismatch",
        "joins/missing_evidence_artifact",
        "joins/node_mismatch",
        "joins/outcome_mismatch",
        "joins/producer_artifact_mismatch",
        "joins/producer_target_mismatch",
        "joins/promise_mismatch",
        "membership/missing_root",
        "provenance/mismatched_source_artifact",
        "provenance/parent_order_invariant",
        "roles/consumed_unsupported_role",
        "roles/distinct_evidence_wrong_role",
        "structural/operation_cancelled_missing_attempt",
        "structural/operation_failed_missing_attempt",
        "structural/operation_lost_missing_attempt",
        "structural/operation_success_missing_attempt",
        "structural/structural_on_operation",
    }
)
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
        "authentication/terminal_outcome",
        "authentication/missing_configuration_environment",
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
    from tests.graph_sdk.test_qualification import _inputs

    baseline = CORPUS[0]
    assert case["declaration"] == baseline["declaration"]
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            coverage=frozenset(CoverageAtom(kind="field", name=name) for name in ("K0", "K1")),
            requirement_coverage=frozenset({CoverageAtom(kind="field", name="K0")}),
            expose_evidence=False,
            provenance_edge_limit=case["declaration"]["limits"]["max_provenance_edges"],
        )
    )
    names = _BaselineNames(execution, result)
    admitted, current, submissions = _inputs(execution, result)
    removed = [event for event in baseline["events"] if event not in events]
    added = [event for event in events if event not in baseline["events"]]
    if added and added[0]["kind"] == "result_owner":
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
        # The remaining retained corruptions change one caller-supplied assessment member.
        assert len(removed) == len(added) == 1 and removed[0]["kind"] == "assessment" and names.fact is not None
        (field,) = [field for field in removed[0] if removed[0][field] != added[0][field]]
        assert field in {"environment", "authenticated_factory", "finding"}
        if field == "finding":
            object.__setattr__(names.fact.finding, "status", added[0][field])
        elif field == "authenticated_factory":
            assert added[0][field] == "CALLER"
            copied = object.__new__(type(names.fact))
            for member in fields(names.fact):
                object.__setattr__(copied, member.name, getattr(names.fact, member.name))
            submissions = (AssessmentSubmission(fact=copied),)
        else:
            assert added[0][field]["configurations"] == {}
            object.__setattr__(names.fact.environment, "configuration", None)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    return names.normalize(output)


@pytest.mark.parametrize(
    "case",
    [case for case in CORPUS if case["case_id"] in CORRUPTION_IDS and case["case_id"] not in FROZEN_VALUE_TAMPER_IDS],
    ids=lambda case: case["case_id"],
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
            independent_context_sources=True,
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
