# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk.evidence import evidence_validity
from anonymizer.engine.graph_sdk.executor import OperationOutputKey, RootInputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk import test_qualification_subject_context as fixture
from tests.graph_sdk.qualification_fixtures import _inputs


def test_ordinary_shared_context_preserves_distinct_candidates() -> None:
    execution, result = asyncio.run(fixture._execute_separate_subject_context(target_labels=("A", "B")))
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 2
    contexts = [fact for fact in result.provenance if isinstance(fact.key, RootInputKey) and fact.key.port == "context"]
    assert len(contexts) == 2 and contexts[0].artifact == contexts[1].artifact and contexts[0].key != contexts[1].key
    finals = [
        fact for fact in result.provenance if isinstance(fact.key, OperationOutputKey) and fact.key.port == "result"
    ]
    assert len(finals) == 2 and finals[0].artifact != finals[1].artifact
    for final in finals:
        (final_output,) = tuple(item for item in result.final_outputs if item.producer == final.key)
        (context,) = tuple(fact for fact in contexts if fact.key.target == final_output.target)
        assert context.key in final.parents
        assert sum(fact.key in final.parents for fact in contexts) == 1
    assert len({item.reference.artifact for item in output.verified}) == 2
    assert all(status.qualification == "met" for status in output.record.statuses)


def test_new_final_cannot_use_old_subject_assessment(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline_execution, baseline_result = asyncio.run(fixture._execute_separate_subject_context())
    baseline_admitted, baseline_current, baseline_submissions = _inputs(baseline_execution, baseline_result)
    assert (
        len(
            qualify(
                admitted=baseline_admitted,
                result=baseline_result,
                current=baseline_current,
                submissions=baseline_submissions,
            ).qualified
        )
        == 1
    )
    original = fixture.OutputDependency

    def dependency(**kwargs: Any) -> Any:
        if kwargs["output"] == "result":
            kwargs["identity_input"] = None
        return original(**kwargs)

    monkeypatch.setattr(fixture, "OutputDependency", dependency)
    limits = fixture.ExecutionLimits

    def execution_limits(**kwargs: Any) -> Any:
        kwargs["max_runtime_artifacts"] += 1
        return limits(**kwargs)

    monkeypatch.setattr(fixture, "ExecutionLimits", execution_limits)
    requirement = fixture.ProtectionRequirement

    def protection(**kwargs: Any) -> Any:
        return requirement(**kwargs, candidate_port="result")

    monkeypatch.setattr(fixture, "ProtectionRequirement", protection)
    execution, result = asyncio.run(fixture._execute_separate_subject_context())
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    (target,) = output.targets
    (status,) = output.record.statuses
    (verified,) = output.verified
    assert target.candidate is not None and target.candidate.artifact != verified.subject.artifact
    assert target.candidate in current.candidates
    assert status.completion == "closed" and status.qualification == "unmet"
    assert status.artifact_available and not status.protection_available
    assert target.withholding == frozenset({"missing_assessment"})
    assert not target.verified
    assert evidence_validity(evidence=verified, current=current) == "stale"
    assert all(terminal.category == "success" for terminal in result.record.terminals)
    assert not output.qualified
    assert output.targets[0].withholding == frozenset({"missing_assessment"})


def test_distinct_evidence_cannot_replace_assessed_subject(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline_execution, baseline_result = asyncio.run(fixture._execute_separate_subject_context())
    baseline_admitted, baseline_current, baseline_submissions = _inputs(baseline_execution, baseline_result)
    assert (
        len(
            qualify(
                admitted=baseline_admitted,
                result=baseline_result,
                current=baseline_current,
                submissions=baseline_submissions,
            ).qualified
        )
        == 1
    )
    requirement = fixture.ProtectionRequirement

    def protection(**kwargs: Any) -> Any:
        return requirement(**kwargs, candidate_port="evidence")

    monkeypatch.setattr(fixture, "ProtectionRequirement", protection)
    admit = fixture.admit_static_workflow

    def workflow(**kwargs: Any) -> Any:
        interface = kwargs["interface"]
        kwargs["interface"] = replace(
            interface,
            outputs=tuple(port for port in interface.outputs if port.name == "evidence"),
            output_dependencies=tuple(dep for dep in interface.output_dependencies if dep.output == "evidence"),
            outcomes=tuple(replace(outcome, produced_ports=frozenset({"evidence"})) for outcome in interface.outcomes),
        )
        kwargs["output_bindings"] = tuple(
            binding for binding in kwargs["output_bindings"] if binding.destination.port == "evidence"
        )
        return admit(**kwargs)

    monkeypatch.setattr(fixture, "admit_static_workflow", workflow)
    execution, result = asyncio.run(fixture._execute_separate_subject_context())
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    (target,) = output.targets
    (status,) = output.record.statuses
    (verified,) = output.verified
    assert target.candidate is not None and target.candidate.artifact != verified.subject.artifact
    assert target.candidate in current.candidates
    assert status.completion == "closed" and status.qualification == "unmet"
    assert status.artifact_available and not status.protection_available
    assert target.withholding == frozenset({"missing_assessment"})
    assert not target.verified
    assert evidence_validity(evidence=verified, current=current) == "stale"
    assert all(terminal.category == "success" for terminal in result.record.terminals)
    assert target.candidate.artifact == verified.reference.artifact
    (final_output,) = result.final_outputs
    assert final_output.port == "evidence" and final_output.candidate == target.candidate
    assert not output.qualified
