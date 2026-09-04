# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from pytest import MonkeyPatch

import anonymizer.engine.execution.phase8_contract as phase8_contract
import anonymizer.engine.execution.phase8_service as phase8_service
from anonymizer.engine.constants import (
    COL_PHASE8_ATTEMPT_TOKEN,
    COL_PHASE8_INVOCATION_TOKEN,
    COL_PHASE8_OPERATION,
    COL_PHASE8_ROW_TOKEN,
    COL_PHASE8_TASK_TOKEN,
    COL_TARGET_WORK_ID,
)
from anonymizer.engine.execution.accounting_outcomes import _AccountingResult
from anonymizer.engine.execution.graph import (
    _AtomicGroup,
    _DatumId,
    _DatumPurpose,
    _ProtectionGraph,
    _RewriteGroup,
    _TextDatum,
)
from anonymizer.engine.execution.phase7_application import _AppliedDatum
from anonymizer.engine.execution.phase7_runtime import (
    _Phase7CleanupAttestation,
    _Phase7Execution,
    _Phase7Phase4Evidence,
)
from anonymizer.engine.execution.phase8_admission import (
    _compile_phase8_plan,
    _Phase8AdmissionCode,
    _Phase8Plan,
    _Phase8Rejected,
)
from anonymizer.engine.execution.phase8_cleanup import (
    _is_phase8_cleanup_receipt,
    _Phase8CleanupComponent,
    _Phase8CleanupPhase,
    _Phase8CleanupStatus,
)
from anonymizer.engine.execution.phase8_contract import (
    _canonical_digest,
    _compile_phase8_contract,
    _is_admitted_phase8_contract,
    _load_phase8_contract,
    _Phase8GroupedRewriteContract,
)
from anonymizer.engine.execution.phase8_ndd_backend import (
    _AnalysisResponse,
    _EvaluationResponse,
    _hydrate,
    _Phase8Correlation,
    _Phase8Operation,
)
from anonymizer.engine.execution.phase8_runtime import _GroupInconsistent, _GroupLost, _run_group_operation
from anonymizer.engine.execution.phase8_service import (
    _backend_group_operation,
    _Phase8AcceptedMention,
    _Phase8ContextProjection,
    _Phase8GroupedRewriteProtectionService,
    _Phase8GroupInput,
    _Phase8WireRegistry,
    _zero_route_admitted,
)
from anonymizer.engine.execution.phase8_validation import _evaluate_metrics
from anonymizer.engine.ndd.adapter import FailedRecord, WorkflowRunResult, _FailedRowEvidence
from tests.engine.execution.phase8_reference_model import Case, reduce


def _graph(groups: tuple[tuple[int, ...], ...]) -> _ProtectionGraph:
    ids = tuple(_DatumId(str(index)) for index in range(2))
    return _ProtectionGraph(
        tuple(_TextDatum(identifier, f"text-{index}", _DatumPurpose.TARGET) for index, identifier in enumerate(ids)),
        (),
        (),
        (),
        (_AtomicGroup(ids),),
        rewrite_groups=tuple(_RewriteGroup(tuple(ids[index] for index in group)) for group in groups),
    )


def test_phase8_contract_loader_admits_exact_frozen_contract() -> None:
    contract = _load_phase8_contract()
    assert _is_admitted_phase8_contract(contract)
    assert isinstance(contract, _Phase8GroupedRewriteContract)
    assert contract.digest == "597a410aee8cb8ca428e82737f385213ce9ce47eae216caea68ebc2f9907d227"
    assert dict(contract.limits)["max_repair_iterations"] == 3


def _digest_valid_envelope() -> dict[str, object]:
    path = Path("src/anonymizer/engine/execution/phase8_grouped_rewrite_contract.json")
    envelope = copy.deepcopy(json.loads(path.read_text()))
    assert isinstance(envelope["contract"], dict)
    envelope["digest"] = _canonical_digest(envelope["contract"])
    return envelope


def test_phase8_contract_loader_rejects_digest_valid_nested_shape_mutations(monkeypatch: MonkeyPatch) -> None:
    envelope = _digest_valid_envelope()
    contract = envelope["contract"]
    assert isinstance(contract, dict)
    contract["scope"]["unexpected"] = True
    envelope["digest"] = _canonical_digest(contract)
    monkeypatch.setattr(phase8_contract, "_DIGEST", envelope["digest"])

    assert not _is_admitted_phase8_contract(_compile_phase8_contract(envelope))


def test_phase8_contract_loader_rejects_digest_valid_type_and_limit_mutations(monkeypatch: MonkeyPatch) -> None:
    type_mutation = _digest_valid_envelope()
    type_contract = type_mutation["contract"]
    assert isinstance(type_contract, dict)
    type_contract["scheduling_and_limits"]["max_members_per_rewrite_group"] = "4"
    type_mutation["digest"] = _canonical_digest(type_contract)

    limit_mutation = _digest_valid_envelope()
    limit_contract = limit_mutation["contract"]
    assert isinstance(limit_contract, dict)
    del limit_contract["phase8_backend_capability"]["required_artifacts"]
    limit_mutation["digest"] = _canonical_digest(limit_contract)

    monkeypatch.setattr(phase8_contract, "_DIGEST", type_mutation["digest"])
    assert not _is_admitted_phase8_contract(_compile_phase8_contract(type_mutation))
    monkeypatch.setattr(phase8_contract, "_DIGEST", limit_mutation["digest"])
    assert not _is_admitted_phase8_contract(_compile_phase8_contract(limit_mutation))


def test_phase8_admission_requires_one_flat_exact_target_partition() -> None:
    accepted = _compile_phase8_plan(_graph(((1, 0),)), max_repairs=2)
    assert isinstance(accepted, _Phase8Plan)
    assert tuple(tuple(member.value for member in group.members) for group in accepted.groups) == (("1", "0"),)
    assert accepted.groups[0].operations.group_id is accepted.groups[0].id
    assert tuple(stage.name for stage in accepted.groups[0].operations.stages) == (
        "validate-baselines",
        "analyze",
        "rewrite",
        "evaluate-0",
        "repair-1",
        "evaluate-1",
        "repair-2",
        "evaluate-2",
    )
    rejected = _compile_phase8_plan(_graph(((0,),)))
    assert isinstance(rejected, _Phase8Rejected)
    assert rejected.code is _Phase8AdmissionCode.COVERAGE_GAP
    too_many_repairs = _compile_phase8_plan(_graph(((1, 0),)), max_repairs=4)
    assert isinstance(too_many_repairs, _Phase8Rejected)
    assert too_many_repairs.code is _Phase8AdmissionCode.LIMIT_EXCEEDED

    split = _compile_phase8_plan(_graph(((0,), (1,))), max_repairs=0)
    assert isinstance(split, _Phase8Plan)
    assert all(group.operations.group_id is group.id for group in split.groups)
    assert split.groups[0].id is not split.groups[1].id


def test_phase8_runtime_never_adopts_a_partial_group_repair() -> None:
    members = (object(), object())
    baselines = {members[0]: "one", members[1]: "two"}
    metric = _evaluate_metrics((), ((1, 0.0),), repair_any_high=False, repair_threshold=0.0, utility_floor=0.5)
    assert metric is not None
    outcome = _run_group_operation(
        members,
        baselines,
        analyze=lambda: (False, False),
        rewrite=lambda values: values,
        evaluate=lambda _values: metric,
        repair=lambda values, _round: {next(iter(values)): "only-one"},
        max_repairs=1,
    )
    assert outcome.state == "inconsistent"
    assert outcome.revisions is None


def test_phase8_zero_obligation_route_requires_all_guards() -> None:
    member = object()
    rewritten = False

    def rewrite(values: dict[object, str]) -> dict[object, str]:
        nonlocal rewritten
        rewritten = True
        return values

    metric = _evaluate_metrics((), (), repair_any_high=False, repair_threshold=0.0, utility_floor=0.0)
    assert metric is not None
    outcome = _run_group_operation(
        (member,),
        {member: "baseline"},
        analyze=lambda: (True, False),
        rewrite=rewrite,
        evaluate=lambda _values: metric,
        repair=lambda values, _round: values,
        max_repairs=0,
    )
    assert outcome.state == "succeeded"
    assert rewritten


def test_phase8_zero_route_rejects_a_mixed_applied_and_no_entity_group() -> None:
    first, second = object(), object()
    members = (first, second)
    baselines = {first: "plain", second: "original"}
    group_input = _Phase8GroupInput(
        originals={first: "plain", second: "original"},
        phase7_applied={first: False, second: True},
    )

    assert not _zero_route_admitted(members, baselines, group_input)


def test_phase8_zero_utility_only_scores_one_for_an_exact_baseline() -> None:
    non_baseline = _evaluate_metrics(
        (), (), repair_any_high=False, repair_threshold=0.0, utility_floor=0.5, exact_baseline=False
    )
    baseline = _evaluate_metrics(
        (), (), repair_any_high=False, repair_threshold=0.0, utility_floor=0.5, exact_baseline=True
    )
    assert non_baseline is not None
    assert baseline is not None
    assert non_baseline.utility_score == 0.0
    assert non_baseline.needs_repair
    assert baseline.utility_score == 1.0
    assert not baseline.needs_repair


def test_private_phase8_service_only_returns_a_complete_group_candidate() -> None:
    members = (object(), object())
    metric = _evaluate_metrics((), ((1, 1.0),), repair_any_high=False, repair_threshold=0.0, utility_floor=0.5)
    assert metric is not None
    result = _Phase8GroupedRewriteProtectionService().run_group(
        members,
        {members[0]: "one", members[1]: "two"},
        analyze=lambda: (False, False),
        rewrite=lambda values: values,
        evaluate=lambda _values: metric,
        repair=lambda values, _round: values,
        max_repairs=0,
    )
    assert result == ((members[0], "one"), (members[1], "two"))


def test_phase8_backend_uses_fresh_opaque_tokens_and_complete_operation_requests() -> None:
    """Wire correlations are private capabilities, not deterministic member indexes."""

    seen: list[dict[str, object]] = []

    class Backend:
        def run_operation(self, operation: _Phase8Operation, request: dict[str, object]):
            seen.append(request)
            members = request["members"]
            assert isinstance(members, list)
            tokens = [member["member_token"] for member in members]
            assert all(isinstance(token, str) and not token.startswith("m-") for token in tokens)
            assert set(request) == {
                "schema_version",
                "privacy_goal",
                "strict_entity_protection",
                "members",
                "context_bindings",
            }
            if operation is _Phase8Operation.ANALYZE:
                return _Result(
                    operation,
                    {
                        "analyzed_member_tokens": tokens,
                        "consumed_context_binding_tokens": [],
                        "privacy_obligations": [],
                        "utility_obligations": [],
                    },
                )
            raise AssertionError("zero route must not dispatch another operation")

    class _Result:
        def __init__(self, operation: _Phase8Operation, payload: dict[str, object]) -> None:
            self.operation = operation
            self.payload = payload
            self.failed = False

    members = (object(), object())
    baselines = {members[0]: "one", members[1]: "two"}
    input = _Phase8GroupInput(dict(baselines), {member: False for member in members})
    first = _backend_group_operation(Backend(), input)
    second = _backend_group_operation(Backend(), input)
    first_outcome = first(members, baselines)
    second_outcome = second(members, baselines)
    assert first_outcome.state == "succeeded" and first_outcome.revisions == baselines
    assert second_outcome.state == "succeeded" and second_outcome.revisions == baselines
    with pytest.raises(Exception, match="group_operation_reused"):
        first(members, baselines)
    assert len(seen) == 2
    assert first_outcome.ledger.is_closed and second_outcome.ledger.is_closed
    assert tuple(first_outcome.ledger.attempt_count(stage) for stage in first_outcome.ledger.plan.stages) == (
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    first_members, second_members = seen[0]["members"], seen[1]["members"]
    assert isinstance(first_members, list) and isinstance(second_members, list)
    assert [member["member_token"] for member in first_members] != [member["member_token"] for member in second_members]


def test_phase8_group_operation_cleanup_discards_candidate_evidence_and_token_authority() -> None:
    class Backend:
        def run_operation(self, operation: _Phase8Operation, request: dict[str, object]) -> object:
            members = request["members"]
            contexts = request["context_bindings"]
            assert isinstance(members, list)
            assert isinstance(contexts, list)
            return _Result(
                operation,
                {
                    "analyzed_member_tokens": [member["member_token"] for member in members],
                    "consumed_context_binding_tokens": [binding["binding_token"] for binding in contexts],
                    "privacy_obligations": [],
                    "utility_obligations": [],
                },
            )

    class _Result:
        def __init__(self, operation: _Phase8Operation, payload: dict[str, object]) -> None:
            self.operation, self.payload, self.failed = operation, payload, False

    member = object()
    privacy_goal = {"protect": "identity", "preserve": "meaning"}
    group_input = _Phase8GroupInput(
        {member: "baseline"},
        {member: False},
        context_projections=(_Phase8ContextProjection(member, object(), 0, "sensitive context"),),
        privacy_goal=privacy_goal,
    )
    registry = _Phase8WireRegistry()
    operation = _backend_group_operation(Backend(), group_input, registry)
    outcome = operation((member,), {member: "baseline"})
    assert outcome.revisions == {member: "baseline"}
    assert registry.issued

    identity = object()
    receipt = operation.discard_private_state(identity)

    assert _is_phase8_cleanup_receipt(
        receipt,
        identity=identity,
        phase=_Phase8CleanupPhase.PRE_REDUCTION,
        component=_Phase8CleanupComponent.OPERATION,
    )
    assert receipt is not None and receipt.status is _Phase8CleanupStatus.VERIFIED
    assert group_input.originals == {}
    assert group_input.phase7_applied == {}
    assert group_input.context_projections == ()
    assert group_input.privacy_goal is None
    assert privacy_goal == {}
    assert registry.issued == set()
    assert outcome.revisions == {}
    assert not outcome.ledger.is_closed
    assert operation.discard_private_state(identity) is None
    with pytest.raises(Exception, match="group_operation_reused"):
        operation((member,), {member: "baseline"})


def test_phase8_analysis_lowers_only_admitted_provenance_and_binds_privacy_authority() -> None:
    """Context and mentions inform analysis without becoming members or graph IDs."""

    member = object()
    seen: dict[str, object] = {}

    class Backend:
        def run_operation(self, operation: _Phase8Operation, request: dict[str, object]):
            seen.update(request)
            members = request["members"]
            assert isinstance(members, list)
            member_record = members[0]
            assert isinstance(member_record, dict)
            token = member_record["member_token"]
            contexts = request["context_bindings"]
            assert isinstance(contexts, list)
            mentions = member_record["accepted_mentions"]
            assert isinstance(mentions, list)
            assert request["schema_version"] == "phase8-group-workframe/v1"
            assert request["privacy_goal"] == {"protect": "hide", "preserve": "meaning"}
            assert request["strict_entity_protection"] is True
            assert "datum_id" not in member_record
            assert contexts == [
                {
                    "binding_token": contexts[0]["binding_token"],
                    "owner_member_token": token,
                    "ordinal": 0,
                    "text": "admitted context",
                }
            ]
            assert mentions == [
                {
                    "mention_token": mentions[0]["mention_token"],
                    "owner_member_token": token,
                    "start": 0,
                    "end": 5,
                    "text": "Alice",
                    "label": "name",
                    "source": "span_detector",
                }
            ]
            return _Response(
                operation,
                {
                    "analyzed_member_tokens": [token],
                    "consumed_context_binding_tokens": [contexts[0]["binding_token"]],
                    "privacy_obligations": [
                        {
                            "statement": "protect name",
                            "kind": "direct",
                            "sensitivity": "high",
                            "source_member_tokens": [token],
                            "source_mention_tokens": [mentions[0]["mention_token"]],
                        }
                    ],
                    "utility_obligations": [],
                },
            )

    class _Response:
        def __init__(self, operation: _Phase8Operation, payload: dict[str, object]) -> None:
            self.operation, self.payload, self.failed = operation, payload, False

    provenance = _Phase8GroupInput(
        originals={member: "Alice"},
        phase7_applied={member: True},
        accepted_mentions=(_Phase8AcceptedMention(member, object(), 0, 5, "Alice", "name", "span_detector"),),
        context_projections=(_Phase8ContextProjection(member, object(), 0, "admitted context"),),
        privacy_goal={"protect": "hide", "preserve": "meaning"},
        strict_entity_protection=True,
    )
    outcome = _backend_group_operation(Backend(), provenance)((member,), {member: "replacement"})
    assert isinstance(outcome.terminal, _GroupLost)


def test_phase8_retired_member_token_is_invocation_inconsistent_not_a_local_failure() -> None:
    """A stage may not accept a prior attempt's private correlation token."""

    member = object()
    retired: list[str] = []

    class Backend:
        def run_operation(self, operation: _Phase8Operation, request: dict[str, object]):
            members = request["members"]
            assert isinstance(members, list)
            token = members[0]["member_token"]
            if operation is _Phase8Operation.ANALYZE:
                retired.append(token)
                return _Response(
                    operation,
                    {
                        "analyzed_member_tokens": [token],
                        "consumed_context_binding_tokens": [],
                        "privacy_obligations": [
                            {
                                "statement": "p",
                                "kind": "latent",
                                "sensitivity": "high",
                                "source_member_tokens": [token],
                                "source_mention_tokens": [],
                            }
                        ],
                        "utility_obligations": [],
                    },
                )
            return _Response(
                operation,
                {"consumed_context_binding_tokens": [], "revisions": [{"member_token": retired[0], "text": "safe"}]},
            )

    class _Response:
        def __init__(self, operation: _Phase8Operation, payload: dict[str, object]) -> None:
            self.operation, self.payload, self.failed = operation, payload, False

    outcome = _backend_group_operation(Backend())((member,), {member: "baseline"})
    assert isinstance(outcome.terminal, _GroupInconsistent)
    assert outcome.terminal.invocation_global


def test_phase8_failed_record_is_a_local_failure_only_when_bound_to_the_active_work_token() -> None:
    """Adapter evidence can fail one group only when its private binding is exact."""
    token = "active-work-token"
    failure = FailedRecord("provider-record", "phase8", "dropped")
    result = WorkflowRunResult(pd.DataFrame(), [failure], (_FailedRowEvidence(token, failure),))

    hydrated = _hydrate(
        _Phase8Operation.ANALYZE,
        result,
        _Phase8Correlation("invocation", "task", "attempt", token),
        _AnalysisResponse,
        "analysis",
    )

    assert hydrated.failed
    assert hydrated.failure_kind == "local_failure"


def test_phase8_public_failed_record_id_without_private_binding_is_invocation_inconsistent() -> None:
    """A public record ID cannot become private complete-group identity."""
    token = "active-work-token"
    failure = FailedRecord(token, "phase8", "dropped")
    result = WorkflowRunResult(pd.DataFrame(), [failure])

    hydrated = _hydrate(
        _Phase8Operation.ANALYZE,
        result,
        _Phase8Correlation("invocation", "task", "attempt", token),
        _AnalysisResponse,
        "analysis",
    )

    assert hydrated.failed
    assert hydrated.failure_kind == "invocation_inconsistent"


def test_phase8_foreign_or_ambiguous_failed_record_evidence_is_invocation_inconsistent() -> None:
    """Unbound evidence cannot be attributed to the active complete group."""
    token = "active-work-token"
    failure = FailedRecord("provider-record", "phase8", "dropped")
    result = WorkflowRunResult(
        pd.DataFrame(),
        [failure],
        (_FailedRowEvidence(token, failure), _FailedRowEvidence("foreign-work-token", failure)),
    )

    hydrated = _hydrate(
        _Phase8Operation.ANALYZE,
        result,
        _Phase8Correlation("invocation", "task", "attempt", token),
        _AnalysisResponse,
        "analysis",
    )

    assert hydrated.failed
    assert hydrated.failure_kind == "invocation_inconsistent"


def test_phase8_invocation_inconsistency_embargoes_all_groups_before_phase4_release() -> None:
    """A foreign provider failure is an invocation fault, not a withholdable group fault."""
    first, second = _DatumId("first"), _DatumId("second")

    class Backend:
        def run_operation(self, operation: _Phase8Operation, _request: dict[str, object]):
            return _Result(operation, None, True, "invocation_inconsistent")

    class _Result:
        def __init__(self, operation: _Phase8Operation, payload: object, failed: bool, failure_kind: str) -> None:
            self.operation = operation
            self.payload = payload
            self.failed = failed
            self.failure_kind = failure_kind

    service = _Phase8GroupedRewriteProtectionService()
    execution = service.run_lifecycle(
        groups=((first,), (second,)),
        atomic_groups=((first,), (second,)),
        dependencies=(),
        phase7_released=((first, "one"), (second, "two")),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=(
            _backend_group_operation(Backend()),
            lambda members, baselines: tuple((member, baselines[member]) for member in members),
        ),
    )

    assert execution.released == ()
    assert execution.global_embargo
    assert execution.terminal_group_states == ("inconsistent", "blocked")


def test_phase8_each_stage_receives_disjoint_member_context_and_obligation_wires() -> None:
    """Stable obligations are rekeyed on every stage and cannot cross the wire."""
    seen: list[tuple[_Phase8Operation, set[str]]] = []

    class Backend:
        def run_operation(self, operation: _Phase8Operation, request: dict[str, object]):
            members = request["members"]
            contexts = request["context_bindings"]
            assert isinstance(members, list) and isinstance(contexts, list)
            wires = {member["member_token"] for member in members} | {binding["binding_token"] for binding in contexts}
            for field in ("privacy_obligations", "utility_obligations"):
                values = request.get(field, [])
                assert isinstance(values, list)
                wires.update(value["obligation_token"] for value in values)
            seen.append((operation, wires))
            member_tokens = [member["member_token"] for member in members]
            context_tokens = [binding["binding_token"] for binding in contexts]
            if operation is _Phase8Operation.ANALYZE:
                mention = members[0]["accepted_mentions"][0]
                return _Response(
                    operation,
                    {
                        "analyzed_member_tokens": member_tokens,
                        "consumed_context_binding_tokens": context_tokens,
                        "privacy_obligations": [
                            {
                                "statement": "p",
                                "kind": "direct",
                                "sensitivity": "high",
                                "source_member_tokens": member_tokens,
                                "source_mention_tokens": [mention["mention_token"]],
                            }
                        ],
                        "utility_obligations": [{"statement": "u", "importance": "important"}],
                    },
                )
            if operation in {_Phase8Operation.REWRITE, _Phase8Operation.REPAIR}:
                return _Response(
                    operation,
                    {
                        "consumed_context_binding_tokens": context_tokens,
                        "revisions": [{"member_token": token, "text": "safe"} for token in member_tokens],
                    },
                )
            privacy_obligations = request["privacy_obligations"]
            utility_obligations = request["utility_obligations"]
            assert isinstance(privacy_obligations, list) and isinstance(utility_obligations, list)
            assert isinstance(privacy_obligations[0], dict) and isinstance(utility_obligations[0], dict)
            return _Response(
                operation,
                {
                    "evaluated_member_tokens": member_tokens,
                    "consumed_context_binding_tokens": context_tokens,
                    "privacy_answers": [
                        {
                            "obligation_token": privacy_obligations[0]["obligation_token"],
                            "deducible": "no",
                            "confidence": 0.0,
                        }
                    ],
                    "utility_answers": [
                        {
                            "obligation_token": utility_obligations[0]["obligation_token"],
                            "preservation_score": 1.0,
                        }
                    ],
                },
            )

    class _Response:
        def __init__(self, operation: _Phase8Operation, payload: dict[str, object]) -> None:
            self.operation, self.payload, self.failed = operation, payload, False

    member = object()
    provenance = _Phase8GroupInput(
        {member: "Alice"},
        {member: True},
        (_Phase8AcceptedMention(member, object(), 0, 5, "Alice", "name", "span"),),
        (_Phase8ContextProjection(member, object(), 0, "context"),),
        {"protect": "hide", "preserve": "meaning"},
        True,
    )
    outcome = _backend_group_operation(Backend(), provenance)((member,), {member: "baseline"})
    assert outcome.state == "succeeded"
    assert outcome.revisions == {member: "safe"}
    assert [operation for operation, _ in seen] == [
        _Phase8Operation.ANALYZE,
        _Phase8Operation.REWRITE,
        _Phase8Operation.EVALUATE,
    ]
    assert all(left.isdisjoint(right) for _, left in seen for _, right in seen if left is not right)


def test_phase8_evaluation_attributes_permuted_answers_to_stable_obligations(
    monkeypatch: MonkeyPatch,
) -> None:
    """Answer order and provider defaults cannot rewrite accepted metadata."""
    observed: list[tuple[tuple[tuple[str, float, bool], ...], tuple[tuple[int, float], ...]]] = []
    evaluate_metrics = phase8_service._evaluate_metrics

    def capture_metrics(
        privacy: tuple[tuple[str, float, bool], ...],
        utility: tuple[tuple[int, float], ...],
        *,
        repair_any_high: bool,
        repair_threshold: float,
        utility_floor: float,
        exact_baseline: bool = False,
    ):
        observed.append((privacy, utility))
        return evaluate_metrics(
            privacy,
            utility,
            repair_any_high=repair_any_high,
            repair_threshold=repair_threshold,
            utility_floor=utility_floor,
            exact_baseline=exact_baseline,
        )

    monkeypatch.setattr(phase8_service, "_evaluate_metrics", capture_metrics)

    class Backend:
        evaluation = 0

        def run_operation(self, operation: _Phase8Operation, request: dict[str, object]):
            members = request["members"]
            assert isinstance(members, list)
            member_tokens = [member["member_token"] for member in members]
            contexts = request["context_bindings"]
            assert isinstance(contexts, list)
            context_tokens = [binding["binding_token"] for binding in contexts]
            if operation is _Phase8Operation.ANALYZE:
                return _Response(
                    operation,
                    {
                        "analyzed_member_tokens": member_tokens,
                        "consumed_context_binding_tokens": context_tokens,
                        "privacy_obligations": [
                            {
                                "statement": "low",
                                "kind": "latent",
                                "sensitivity": "low",
                                "source_member_tokens": member_tokens,
                                "source_mention_tokens": [],
                            },
                            {
                                "statement": "high",
                                "kind": "latent",
                                "sensitivity": "high",
                                "source_member_tokens": member_tokens,
                                "source_mention_tokens": [],
                            },
                        ],
                        "utility_obligations": [
                            {"statement": "critical", "importance": "critical"},
                            {"statement": "important", "importance": "important"},
                        ],
                    },
                )
            if operation in {_Phase8Operation.REWRITE, _Phase8Operation.REPAIR}:
                text = "repaired" if operation is _Phase8Operation.REPAIR else "safe"
                return _Response(
                    operation,
                    {
                        "consumed_context_binding_tokens": context_tokens,
                        "revisions": [{"member_token": token, "text": text} for token in member_tokens],
                    },
                )
            privacy = request["privacy_obligations"]
            utility = request["utility_obligations"]
            assert isinstance(privacy, list) and isinstance(utility, list)
            self.evaluation += 1
            if self.evaluation == 1:
                privacy_answers = [
                    {"obligation_token": privacy[1]["obligation_token"], "deducible": "no", "confidence": 0.0},
                    {"obligation_token": privacy[0]["obligation_token"], "deducible": "yes", "confidence": 0.5},
                ]
                utility_answers = [
                    {"obligation_token": utility[1]["obligation_token"], "preservation_score": 1.0},
                    {"obligation_token": utility[0]["obligation_token"], "preservation_score": 0.0},
                ]
            else:
                privacy_answers = [
                    {"obligation_token": item["obligation_token"], "deducible": "no", "confidence": 0.0}
                    for item in reversed(privacy)
                ]
                utility_answers = [
                    {"obligation_token": item["obligation_token"], "preservation_score": 1.0}
                    for item in reversed(utility)
                ]
            return _Response(
                operation,
                {
                    "evaluated_member_tokens": member_tokens,
                    "consumed_context_binding_tokens": context_tokens,
                    "privacy_answers": privacy_answers,
                    "utility_answers": utility_answers,
                },
            )

    class _Response:
        def __init__(self, operation: _Phase8Operation, payload: dict[str, object]) -> None:
            self.operation, self.payload, self.failed = operation, payload, False

    member = object()
    backend = Backend()
    result = _backend_group_operation(backend, _Phase8GroupInput({member: "original"}, {member: True}))(
        (member,), {member: "baseline"}
    )

    assert result.state == "succeeded"
    assert result.revisions == {member: "repaired"}
    assert observed[0] == (
        (("high", 0.0, False), ("low", 0.5, True)),
        ((1, 1.0), (2, 0.0)),
    )


def test_phase8_cross_group_analysis_member_token_embargoes_invocation() -> None:
    """A sibling-issued provenance capability is not a local schema failure."""
    first, second, third = (_DatumId(name) for name in ("first", "second", "third"))
    registry = phase8_service._Phase8WireRegistry()
    first_token: list[str] = []
    calls = 0

    class Backend:
        def run_operation(self, operation: _Phase8Operation, request: dict[str, object]):
            nonlocal calls
            assert operation is _Phase8Operation.ANALYZE
            calls += 1
            members = request["members"]
            assert isinstance(members, list)
            member_token = members[0]["member_token"]
            if calls == 1:
                first_token.append(member_token)
                obligations: list[dict[str, object]] = []
            else:
                obligations = [
                    {
                        "statement": "sibling provenance",
                        "kind": "latent",
                        "sensitivity": "high",
                        "source_member_tokens": [first_token[0]],
                        "source_mention_tokens": [],
                    }
                ]
            return _Response(
                operation,
                {
                    "analyzed_member_tokens": [member_token],
                    "consumed_context_binding_tokens": [],
                    "privacy_obligations": obligations,
                    "utility_obligations": [],
                },
            )

    class _Response:
        def __init__(self, operation: _Phase8Operation, payload: dict[str, object]) -> None:
            self.operation, self.payload, self.failed = operation, payload, False

    backend = Backend()
    operations = tuple(
        _backend_group_operation(backend, _Phase8GroupInput({member: value}, {member: False}), registry)
        for member, value in ((first, "one"), (second, "two"), (third, "three"))
    )
    execution = _Phase8GroupedRewriteProtectionService().run_lifecycle(
        groups=((first,), (second,), (third,)),
        atomic_groups=((first,), (second,), (third,)),
        dependencies=(),
        phase7_released=((first, "one"), (second, "two"), (third, "three")),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=operations,
    )

    assert execution.released == ()
    assert execution.global_embargo
    assert not execution.cleanup_verified
    assert execution.terminal_group_states == ("succeeded", "inconsistent", "blocked")
    assert calls == 2


def test_phase8_cross_group_context_token_embargoes_invocation() -> None:
    """A sibling-issued context capability cannot acknowledge the current group."""
    first, second, third = (_DatumId(name) for name in ("first", "second", "third"))
    registry = phase8_service._Phase8WireRegistry()
    first_token: list[str] = []
    calls = 0

    class Backend:
        def run_operation(self, operation: _Phase8Operation, request: dict[str, object]):
            nonlocal calls
            assert operation is _Phase8Operation.ANALYZE
            calls += 1
            members = request["members"]
            contexts = request["context_bindings"]
            assert isinstance(members, list) and isinstance(contexts, list)
            member_token = members[0]["member_token"]
            context_token = contexts[0]["binding_token"]
            if calls == 1:
                first_token.append(context_token)
            return _Response(
                operation,
                {
                    "analyzed_member_tokens": [member_token],
                    "consumed_context_binding_tokens": [context_token if calls == 1 else first_token[0]],
                    "privacy_obligations": [],
                    "utility_obligations": [],
                },
            )

    class _Response:
        def __init__(self, operation: _Phase8Operation, payload: dict[str, object]) -> None:
            self.operation, self.payload, self.failed = operation, payload, False

    backend = Backend()
    operations = tuple(
        _backend_group_operation(
            backend,
            _Phase8GroupInput(
                {member: value},
                {member: False},
                context_projections=(_Phase8ContextProjection(member, object(), 0, "context"),),
            ),
            registry,
        )
        for member, value in ((first, "one"), (second, "two"), (third, "three"))
    )
    execution = _Phase8GroupedRewriteProtectionService().run_lifecycle(
        groups=((first,), (second,), (third,)),
        atomic_groups=((first,), (second,), (third,)),
        dependencies=(),
        phase7_released=((first, "one"), (second, "two"), (third, "three")),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=operations,
    )

    assert execution.released == ()
    assert execution.global_embargo
    assert execution.terminal_group_states == ("succeeded", "inconsistent", "blocked")
    assert calls == 2


def test_phase8_cross_group_obligation_token_embargoes_invocation() -> None:
    """A sibling-issued obligation capability cannot answer a later group."""
    first, second, third = (_DatumId(name) for name in ("first", "second", "third"))
    registry = phase8_service._Phase8WireRegistry()
    first_token: list[str] = []
    analyzed: list[str] = []
    active_group = 0

    class Backend:
        def run_operation(self, operation: _Phase8Operation, request: dict[str, object]):
            nonlocal active_group
            members = request["members"]
            contexts = request["context_bindings"]
            assert isinstance(members, list) and isinstance(contexts, list)
            member_tokens = [member["member_token"] for member in members]
            if operation is _Phase8Operation.ANALYZE:
                active_group += 1
                analyzed.append(cast(str, members[0]["phase7_baseline"]))
                return _Response(
                    operation,
                    {
                        "analyzed_member_tokens": member_tokens,
                        "consumed_context_binding_tokens": [],
                        "privacy_obligations": [
                            {
                                "statement": "protect",
                                "kind": "latent",
                                "sensitivity": "high",
                                "source_member_tokens": member_tokens,
                                "source_mention_tokens": [],
                            }
                        ],
                        "utility_obligations": [{"statement": "meaning", "importance": "important"}],
                    },
                )
            if operation is _Phase8Operation.REWRITE:
                privacy = request["privacy_obligations"]
                assert isinstance(privacy, list)
                if active_group == 1:
                    first_token.append(privacy[0]["obligation_token"])
                return _Response(
                    operation,
                    {
                        "consumed_context_binding_tokens": [],
                        "revisions": [{"member_token": token, "text": "safe"} for token in member_tokens],
                    },
                )
            privacy = request["privacy_obligations"]
            utility = request["utility_obligations"]
            assert isinstance(privacy, list) and isinstance(utility, list)
            answer_token = privacy[0]["obligation_token"] if active_group == 1 else first_token[0]
            return _Response(
                operation,
                {
                    "evaluated_member_tokens": member_tokens,
                    "consumed_context_binding_tokens": [],
                    "privacy_answers": [{"obligation_token": answer_token, "deducible": "no", "confidence": 0.0}],
                    "utility_answers": [
                        {"obligation_token": utility[0]["obligation_token"], "preservation_score": 1.0}
                    ],
                },
            )

    class _Response:
        def __init__(self, operation: _Phase8Operation, payload: dict[str, object]) -> None:
            self.operation, self.payload, self.failed = operation, payload, False

    backend = Backend()
    operations = tuple(
        _backend_group_operation(backend, _Phase8GroupInput({member: value}, {member: True}), registry)
        for member, value in ((first, "one"), (second, "two"), (third, "three"))
    )
    execution = _Phase8GroupedRewriteProtectionService().run_lifecycle(
        groups=((first,), (second,), (third,)),
        atomic_groups=((first,), (second,), (third,)),
        dependencies=(),
        phase7_released=((first, "one"), (second, "two"), (third, "three")),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=operations,
    )

    assert execution.released == ()
    assert execution.global_embargo
    assert execution.terminal_group_states == ("succeeded", "inconsistent", "blocked")
    assert analyzed == ["one", "two"]


def test_phase8_hydration_rejects_tampered_outer_correlation_tuple() -> None:
    correlation = _Phase8Correlation("invocation", "task", "attempt", "row")
    result = WorkflowRunResult(
        pd.DataFrame(
            [
                {
                    COL_TARGET_WORK_ID: "row",
                    COL_PHASE8_INVOCATION_TOKEN: "foreign",
                    COL_PHASE8_TASK_TOKEN: "task",
                    COL_PHASE8_ATTEMPT_TOKEN: "attempt",
                    COL_PHASE8_ROW_TOKEN: "row",
                    "analysis": {},
                }
            ]
        ),
        [],
        (),
    )
    hydrated = _hydrate(_Phase8Operation.ANALYZE, result, correlation, _AnalysisResponse, "analysis")
    assert hydrated.failed and hydrated.failure_kind == "invocation_inconsistent"


@pytest.mark.parametrize("returned_operation", [None, _Phase8Operation.REPAIR.value])
def test_phase8_hydration_rejects_missing_or_tampered_operation(returned_operation: str | None) -> None:
    correlation = _Phase8Correlation("invocation", "task", "attempt", "row")
    result = WorkflowRunResult(
        pd.DataFrame(
            [
                {
                    COL_TARGET_WORK_ID: "row",
                    COL_PHASE8_INVOCATION_TOKEN: "invocation",
                    COL_PHASE8_TASK_TOKEN: "task",
                    COL_PHASE8_ATTEMPT_TOKEN: "attempt",
                    COL_PHASE8_ROW_TOKEN: "row",
                    COL_PHASE8_OPERATION: returned_operation,
                    "analysis": {
                        "analyzed_member_tokens": [],
                        "consumed_context_binding_tokens": [],
                        "privacy_obligations": [],
                        "utility_obligations": [],
                    },
                }
            ]
        ),
        [],
        (),
    )

    hydrated = _hydrate(_Phase8Operation.ANALYZE, result, correlation, _AnalysisResponse, "analysis")

    assert hydrated.failed
    assert hydrated.failure_kind == "invocation_inconsistent"


def test_phase8_hydration_accepts_contract_valid_string_obligation_tokens() -> None:
    correlation = _Phase8Correlation("invocation", "task", "attempt", "row")
    result = WorkflowRunResult(
        pd.DataFrame(
            [
                {
                    COL_TARGET_WORK_ID: "row",
                    COL_PHASE8_INVOCATION_TOKEN: "invocation",
                    COL_PHASE8_TASK_TOKEN: "task",
                    COL_PHASE8_ATTEMPT_TOKEN: "attempt",
                    COL_PHASE8_ROW_TOKEN: "row",
                    COL_PHASE8_OPERATION: _Phase8Operation.EVALUATE.value,
                    "evaluation": {
                        "evaluated_member_tokens": ["member"],
                        "consumed_context_binding_tokens": [],
                        "privacy_answers": [],
                        "utility_answers": [{"obligation_token": "obligation", "preservation_score": 1.0}],
                    },
                }
            ]
        ),
        [],
        (),
    )

    hydrated = _hydrate(_Phase8Operation.EVALUATE, result, correlation, _EvaluationResponse, "evaluation")

    assert not hydrated.failed
    assert hydrated.payload is not None


def test_phase8_zero_route_requires_phase7_identity_provenance() -> None:
    """An empty privacy analysis is not a baseline fallback without every identity guard."""

    calls: list[_Phase8Operation] = []

    class Backend:
        def run_operation(self, operation: _Phase8Operation, request: dict[str, object]):
            calls.append(operation)
            members = request["members"]
            assert isinstance(members, list)
            if operation is not _Phase8Operation.ANALYZE:
                raise RuntimeError
            return _Response(
                operation,
                {
                    "analyzed_member_tokens": [member["member_token"] for member in members],
                    "consumed_context_binding_tokens": [],
                    "privacy_obligations": [],
                    "utility_obligations": [{"statement": "keep meaning", "importance": "important"}],
                },
            )

    class _Response:
        def __init__(self, operation: _Phase8Operation, payload: dict[str, object]) -> None:
            self.operation = operation
            self.payload = payload
            self.failed = False

    member = object()
    baseline = {member: "baseline"}
    provenance = _Phase8GroupInput({member: "original"}, {member: False})
    outcome = _backend_group_operation(Backend(), provenance)((member,), baseline)
    assert isinstance(outcome.terminal, _GroupLost)
    assert calls == [_Phase8Operation.ANALYZE, _Phase8Operation.REWRITE]


def test_phase8_backend_rejects_missing_or_duplicate_obligation_answers() -> None:
    """Evaluation adoption requires an exact obligation-token bijection."""

    class Backend:
        def run_operation(self, operation: _Phase8Operation, request: dict[str, object]):
            members = request["members"]
            assert isinstance(members, list)
            tokens = [member["member_token"] for member in members]
            if operation is _Phase8Operation.ANALYZE:
                return _Response(
                    operation,
                    {
                        "analyzed_member_tokens": tokens,
                        "consumed_context_binding_tokens": [],
                        "privacy_obligations": [{"statement": "protect", "source_member_tokens": tokens}],
                        "utility_obligations": [],
                    },
                )
            if operation is _Phase8Operation.REWRITE:
                return _Response(
                    operation,
                    {
                        "consumed_context_binding_tokens": [],
                        "revisions": [{"member_token": token, "text": "safe"} for token in tokens],
                    },
                )
            if operation is _Phase8Operation.EVALUATE:
                return _Response(
                    operation,
                    {
                        "evaluated_member_tokens": tokens,
                        "consumed_context_binding_tokens": [],
                        "privacy_answers": [],
                        "utility_answers": [],
                    },
                )
            raise AssertionError

    class _Response:
        def __init__(self, operation: _Phase8Operation, payload: dict[str, object]) -> None:
            self.operation = operation
            self.payload = payload
            self.failed = False

    member = object()
    outcome = _backend_group_operation(Backend())((member,), {member: "baseline"})
    assert outcome.state == "failed"


def test_phase8_lifecycle_requires_a_released_phase7_baseline_and_withholds_the_atomic_group() -> None:
    """A failed complete group may not release its otherwise-successful sibling."""
    service = _Phase8GroupedRewriteProtectionService()
    first, second = _DatumId("first"), _DatumId("second")
    execution = service.run_lifecycle(
        groups=((first,), (second,)),
        atomic_groups=((first, second),),
        dependencies=(),
        phase7_released=((first, "one"), (second, "two")),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=(
            lambda members, baselines: service.run_group(
                members,
                baselines,
                analyze=lambda: (False, False),
                rewrite=lambda values: values,
                evaluate=lambda _values: _metric(),
                repair=lambda values, _round: values,
                max_repairs=0,
            ),
            lambda members, baselines: service.run_group(
                members,
                baselines,
                analyze=lambda: (False, False),
                rewrite=lambda _values: {},
                evaluate=lambda _values: _metric(),
                repair=lambda values, _round: values,
                max_repairs=0,
            ),
        ),
    )
    assert execution.released == ()
    assert execution.global_embargo is False
    assert execution.cleanup_verified
    assert execution.terminal_group_states == ("succeeded", "failed")


def test_phase8_lifecycle_embargoes_before_dispatch_when_phase7_handoff_is_not_clean() -> None:
    service = _Phase8GroupedRewriteProtectionService()
    datum = _DatumId("only")
    dispatched = False

    def operation(_members: tuple[object, ...], _baselines: dict[object, str]) -> tuple[tuple[object, str], ...] | None:
        nonlocal dispatched
        dispatched = True
        raise AssertionError("must not dispatch")

    execution = service.run_lifecycle(
        groups=((datum,),),
        atomic_groups=((datum,),),
        dependencies=(),
        phase7_released=((datum, "baseline"),),
        phase7_cleanup_verified=False,
        phase7_global_embargo=False,
        operations=(operation,),
    )
    assert not dispatched
    assert execution.global_embargo
    assert execution.released == ()


def test_phase8_lifecycle_consumes_only_the_phase7_released_baseline_handoff() -> None:
    service = _Phase8GroupedRewriteProtectionService()
    datum = _DatumId("only")
    cleanup = _Phase7CleanupAttestation("phase7-cleanup-attestation/v1", True, 0, 0, True, 0, False)
    phase7 = _Phase7Execution(
        (),
        cleanup,
        _Phase7Phase4Evidence((), cast(_AccountingResult[object], object()), cleanup, False),
        (_AppliedDatum(datum, "baseline", True),),
    )
    execution = service.run_from_phase7_execution(
        groups=((datum,),),
        atomic_groups=((datum,),),
        dependencies=(),
        phase7=phase7,
        operations=(
            lambda members, baselines: service.run_group(
                members,
                baselines,
                analyze=lambda: (True, True),
                rewrite=lambda values: values,
                evaluate=lambda _values: _metric(),
                repair=lambda values, _round: values,
                max_repairs=0,
            ),
        ),
    )
    assert execution.released == ((datum, "baseline"),)


def _metric():
    metric = _evaluate_metrics((), ((1, 1.0),), repair_any_high=False, repair_threshold=0.0, utility_floor=0.5)
    assert metric is not None
    return metric


def test_phase8_reference_model_is_pure_and_withholds_an_atomic_group() -> None:
    tree = ast.parse(Path("tests/engine/execution/phase8_reference_model.py").read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.name for alias in node.names)
    assert not {"pandas", "data_designer"}.intersection(names)
    assert reduce(Case((("a", "b"), ("c",)), (("a", "b"), ("c",)), (0,))) == ("c",)
