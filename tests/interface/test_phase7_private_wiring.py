# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import anonymizer.interface._protection as protection_module
from anonymizer.config.anonymizer_config import AnonymizerConfig, Rewrite
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.engine.execution.graph import (
    _AtomicGroup,
    _CoherenceScope,
    _ContextScope,
    _DatumId,
    _DatumPurpose,
    _ProtectionGraph,
    _TextDatum,
)
from anonymizer.engine.execution.phase6_plan import _Phase6Plan
from anonymizer.engine.execution.phase7_admission import _ScopeManifest
from anonymizer.engine.execution.phase7_contract import _Phase7StableSubstituteContract
from anonymizer.engine.execution.phase7_ndd_backend import _Phase7NddResult, _Phase7NddStatus
from anonymizer.engine.execution.phase7_runtime import _Phase7CleanupAttestation
from anonymizer.engine.execution.phase7_validation import _CandidateAssignment
from anonymizer.engine.execution.phase8_cleanup import (
    _issue_phase8_cleanup_receipt,
    _Phase8CleanupComponent,
    _Phase8CleanupPhase,
    _Phase8CleanupStatus,
)
from anonymizer.engine.execution.phase8_ndd_backend import _compile_phase8_capability, _Phase8Operation
from anonymizer.engine.execution.phase8_successor import _Phase8SuccessorHandoff
from anonymizer.engine.execution.protection_service import _Phase7SubstituteProtectionService
from anonymizer.interface._protection import (
    _BatchFailureCode,
    _Failed,
    _NoAcceptedDetections,
    _PlanRejected,
    _PlanUnsupported,
    _ProtectionApplied,
    _ProtectionBatchError,
    _ProtectionFlow,
    _ProtectionPlan,
    _ProtectionSucceeded,
)
from anonymizer.interface.anonymizer import Anonymizer
from tests.interface.test_phase6_public_compatibility import _base_dataframe, _synthetic_anonymizer, _write_input
from tests.interface.test_private_protection import _AnchoredPhase6Backend, _record
from tests.streaming.structured_trace_prototype import build_synthetic_anonymizer


@dataclass
class _CandidateBackend:
    value: str | None
    calls: int = 0
    closed: int = 0
    discarded: int = 0

    def propose_scope(
        self,
        manifest: object,
        handoffs: object,
        contract: object,
        dispatch: object,
    ) -> _Phase7NddResult:
        del handoffs, contract
        assert isinstance(manifest, _ScopeManifest)
        assert dispatch is not None
        assert isinstance(self.value, str)
        self.calls += 1
        return _Phase7NddResult(
            _Phase7NddStatus.CANDIDATE,
            tuple(_CandidateAssignment(slot.id, self.value) for slot in manifest.slots),
        )

    def close(self) -> None:
        self.closed += 1

    def discard_values(self) -> None:
        self.discarded += 1
        self.value = None

    def cleanup_attestation(self, cleanup_identity: object) -> object:
        return _Phase7CleanupAttestation(
            "phase7-cleanup-attestation/v1",
            True,
            0,
            0,
            True,
            0,
            False,
            cleanup_identity,
        )


@dataclass
class _GroupedRewriteBackend:
    """Deterministic Phase 8 boundary fake used only by the private-flow test."""

    calls: list[_Phase8Operation]
    evaluations: int = 0
    member_token_sets: list[tuple[str, ...]] | None = None
    accepted_mention_counts: list[int] | None = None
    context_binding_sets: list[tuple[dict[str, object], ...]] | None = None
    consume_context: bool = False
    retired: int = 0

    def phase8_capability(self, invocation: object) -> object:
        return _compile_phase8_capability(invocation)

    def run_operation(self, operation: _Phase8Operation, request: dict[str, object]) -> object:
        self.calls.append(operation)
        members = request["members"]
        assert isinstance(members, list)
        tokens = [member["member_token"] for member in members]
        if self.member_token_sets is None:
            self.member_token_sets = []
        self.member_token_sets.append(tuple(tokens))
        assert all(isinstance(token, str) for token in tokens)
        assert "context_bindings" in request
        context_bindings = request["context_bindings"]
        assert isinstance(context_bindings, list)
        if self.context_binding_sets is None:
            self.context_binding_sets = []
        self.context_binding_sets.append(tuple(context_bindings))
        context_tokens = [binding["binding_token"] for binding in context_bindings]
        assert all(isinstance(token, str) for token in context_tokens)
        consumed_context = context_tokens if self.consume_context else []
        if operation is _Phase8Operation.ANALYZE:
            assert all("accepted_mentions" in member for member in members)
            if self.accepted_mention_counts is None:
                self.accepted_mention_counts = []
            self.accepted_mention_counts.extend(
                len(member["accepted_mentions"])
                for member in members
                if isinstance(member.get("accepted_mentions"), list)
            )
        if operation is _Phase8Operation.ANALYZE:
            mentions = [
                mention["mention_token"]
                for member in members
                for mention in member["accepted_mentions"]
                if isinstance(mention, dict)
            ]
            return _Dispatch(
                operation,
                {
                    "analyzed_member_tokens": tokens,
                    "consumed_context_binding_tokens": consumed_context,
                    "privacy_obligations": [
                        {
                            "statement": "protect identifier",
                            "kind": "direct" if mentions else "latent",
                            "sensitivity": "high",
                            "source_member_tokens": tokens,
                            "source_mention_tokens": mentions,
                        }
                    ],
                    "utility_obligations": [{"statement": "preserve meaning", "importance": "important"}],
                },
            )
        if operation is _Phase8Operation.EVALUATE:
            privacy_obligations = request["privacy_obligations"]
            utility_obligations = request["utility_obligations"]
            assert isinstance(privacy_obligations, list) and isinstance(utility_obligations, list)
            privacy_obligation = privacy_obligations[0]
            utility_obligation = utility_obligations[0]
            assert isinstance(privacy_obligation, dict) and isinstance(utility_obligation, dict)
            self.evaluations += 1
            return _Dispatch(
                operation,
                {
                    "evaluated_member_tokens": tokens,
                    "consumed_context_binding_tokens": consumed_context,
                    "privacy_answers": [
                        {
                            "obligation_token": privacy_obligation["obligation_token"],
                            "deducible": "yes" if self.evaluations == 1 else "no",
                            "confidence": 1.0 if self.evaluations == 1 else 0.0,
                        }
                    ],
                    "utility_answers": [
                        {
                            "obligation_token": utility_obligation["obligation_token"],
                            "preservation_score": 1.0,
                        }
                    ],
                },
            )
        return _Dispatch(
            operation,
            {
                "consumed_context_binding_tokens": consumed_context,
                "revisions": [
                    {"member_token": token, "text": f"rewrite-{index}"} for index, token in enumerate(tokens)
                ],
            },
        )

    def retire_phase8(self, cleanup_identity: object) -> object:
        self.retired += 1
        return _issue_phase8_cleanup_receipt(
            _Phase8CleanupPhase.PRE_REDUCTION,
            _Phase8CleanupComponent.BACKEND,
            _Phase8CleanupStatus.VERIFIED,
            cleanup_identity,
        )


@dataclass
class _Dispatch:
    operation: _Phase8Operation
    payload: object
    failed: bool = False


def _private_substitute_flow(
    *,
    original: str,
    synthetic: str,
    label: str = "first_name",
) -> tuple[_ProtectionFlow, _CandidateBackend]:
    anonymizer = build_synthetic_anonymizer({original: label})
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Substitute(), emit_telemetry=False))
    assert isinstance(plan, _ProtectionPlan)
    backend = _CandidateBackend(synthetic)
    flow = protection_module._ProtectionFlow(
        anonymizer,
        plan,
        phase6_backend=_AnchoredPhase6Backend.from_entities({original: label}),
        phase7_backend=backend,
    )
    return flow, backend


def test_private_substitute_plan_selects_phase7_service_without_changing_other_profiles() -> None:
    anonymizer = build_synthetic_anonymizer({})

    substitute = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Substitute(), emit_telemetry=False))
    redact = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))

    assert isinstance(substitute, _ProtectionPlan)
    assert substitute.profile == "stable-substitute-v1"
    assert type(anonymizer._open_protection_flow(substitute)._runtime).__name__ == "_Phase7SubstituteProtectionService"
    assert isinstance(redact, _ProtectionPlan)
    assert redact.profile == "redact-release-v1"
    assert type(anonymizer._open_protection_flow(redact)._runtime).__name__ == "_Phase6RedactProtectionService"
    assert isinstance(anonymizer._compile_protection_plan(AnonymizerConfig(replace=Annotate())), _PlanRejected)
    assert isinstance(anonymizer._compile_protection_plan(AnonymizerConfig(replace=Hash())), _PlanUnsupported)
    assert isinstance(anonymizer._compile_protection_plan(AnonymizerConfig(rewrite=Rewrite())), _PlanRejected)
    rewrite = anonymizer._compile_protection_plan(AnonymizerConfig(rewrite=Rewrite(strict_entity_protection=True)))
    assert isinstance(rewrite, _ProtectionPlan)
    assert rewrite.profile == "grouped-rewrite-v1"
    assert isinstance(
        anonymizer._compile_protection_plan(AnonymizerConfig(replace=Substitute(instructions="legacy only"))),
        _PlanRejected,
    )


def test_private_grouped_rewrite_executes_phase7_then_all_phase8_operations() -> None:
    anonymizer = build_synthetic_anonymizer({"Alice": "first_name"})
    plan = anonymizer._compile_protection_plan(
        AnonymizerConfig(rewrite=Rewrite(strict_entity_protection=True), emit_telemetry=False)
    )
    assert isinstance(plan, _ProtectionPlan)
    phase7 = _CandidateBackend("Avery")
    phase8 = _GroupedRewriteBackend([])

    flow = _ProtectionFlow(
        anonymizer,
        plan,
        phase6_backend=_AnchoredPhase6Backend.from_entities({"Alice": "first_name"}),
        phase7_backend=phase7,
        phase8_backend=phase8,
    )
    result = flow.protect((_record("source-a", "Alice"),))

    assert phase8.calls
    assert isinstance(result.outcomes[0], _ProtectionSucceeded)
    assert result.outcomes[0].output == "rewrite-0"
    assert phase8.calls == [
        _Phase8Operation.ANALYZE,
        _Phase8Operation.REWRITE,
        _Phase8Operation.EVALUATE,
        _Phase8Operation.REPAIR,
        _Phase8Operation.EVALUATE,
    ]
    assert phase8.member_token_sets is not None
    assert len(set(phase8.member_token_sets)) == len(phase8.member_token_sets)
    assert phase8.accepted_mention_counts == [1]


def test_private_grouped_rewrite_with_no_entities_still_analyzes_the_admitted_group() -> None:
    anonymizer = build_synthetic_anonymizer({"Alice": "first_name"})
    plan = anonymizer._compile_protection_plan(
        AnonymizerConfig(rewrite=Rewrite(strict_entity_protection=True), emit_telemetry=False)
    )
    assert isinstance(plan, _ProtectionPlan)
    phase7 = _CandidateBackend("Avery")
    phase8 = _GroupedRewriteBackend([])
    flow = _ProtectionFlow(
        anonymizer,
        plan,
        phase6_backend=_AnchoredPhase6Backend.from_entities({"Alice": "first_name"}),
        phase7_backend=phase7,
        phase8_backend=phase8,
    )

    result = flow.protect((_record("source-a", "ordinary text"),))

    assert isinstance(result.outcomes[0], _ProtectionSucceeded)
    assert result.outcomes[0].output == "rewrite-0"
    assert phase7.calls == 0
    assert phase8.calls == [
        _Phase8Operation.ANALYZE,
        _Phase8Operation.REWRITE,
        _Phase8Operation.EVALUATE,
        _Phase8Operation.REPAIR,
        _Phase8Operation.EVALUATE,
    ]


@pytest.mark.parametrize("forged", [object(), (object(), object())])
def test_private_grouped_rewrite_rejects_forged_or_mismatched_predecessor_before_phase8_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    forged: object,
) -> None:
    """Only the Phase 7 owner may provide the exact sealed predecessor identity."""
    anonymizer = build_synthetic_anonymizer({"Alice": "first_name"})
    plan = anonymizer._compile_protection_plan(
        AnonymizerConfig(rewrite=Rewrite(strict_entity_protection=True), emit_telemetry=False)
    )
    assert isinstance(plan, _ProtectionPlan)
    phase8 = _GroupedRewriteBackend([])
    monkeypatch.setattr(_Phase7SubstituteProtectionService, "execute_successor", lambda *_args, **_kwargs: forged)
    flow = _ProtectionFlow(
        anonymizer,
        plan,
        phase6_backend=_AnchoredPhase6Backend.from_entities({"Alice": "first_name"}),
        phase7_backend=_CandidateBackend("Avery"),
        phase8_backend=phase8,
    )

    result = flow.protect((_record("source-a", "Alice"),))

    assert isinstance(result.outcomes[0], _Failed)
    assert phase8.calls == []


@pytest.mark.parametrize("replacement", ("reconstructed_execution", "mixed_execution"))
def test_private_grouped_rewrite_rejects_a_valid_but_unbound_phase7_predecessor_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    """A sealed handoff is not transferable to a lookalike or another run's execution."""
    anonymizer = build_synthetic_anonymizer({"Alice": "first_name"})
    plan = anonymizer._compile_protection_plan(
        AnonymizerConfig(rewrite=Rewrite(strict_entity_protection=True), emit_telemetry=False)
    )
    assert isinstance(plan, _ProtectionPlan)
    phase8 = _GroupedRewriteBackend([])
    captured: list[_Phase8SuccessorHandoff | None] = []
    original = _Phase7SubstituteProtectionService.execute_successor

    def capture(
        self: _Phase7SubstituteProtectionService,
        phase6: _Phase6Plan,
        *,
        contract: _Phase7StableSubstituteContract,
    ) -> _Phase8SuccessorHandoff | None:
        result = original(self, phase6, contract=contract)
        captured.append(result)
        return result

    monkeypatch.setattr(_Phase7SubstituteProtectionService, "execute_successor", capture)

    def new_flow() -> _ProtectionFlow:
        return _ProtectionFlow(
            anonymizer,
            plan,
            phase6_backend=_AnchoredPhase6Backend.from_entities({"Alice": "first_name"}),
            phase7_backend=_CandidateBackend("Avery"),
            phase8_backend=phase8,
        )

    assert isinstance(new_flow().protect((_record("source-a", "Alice"),)).outcomes[0], _ProtectionSucceeded)
    assert isinstance(new_flow().protect((_record("source-b", "Alice"),)).outcomes[0], _ProtectionSucceeded)
    first, second = captured
    assert isinstance(first, _Phase8SuccessorHandoff)
    assert isinstance(second, _Phase8SuccessorHandoff)
    phase8.calls.clear()
    forged = replace(
        first,
        phase7_execution=(
            replace(first.phase7_execution) if replacement == "reconstructed_execution" else second.phase7_execution
        ),
    )
    monkeypatch.setattr(_Phase7SubstituteProtectionService, "execute_successor", lambda *_args, **_kwargs: forged)

    result = new_flow().protect((_record("source-c", "Alice"),))

    assert isinstance(result.outcomes[0], _Failed)
    assert phase8.calls == []


def test_private_grouped_rewrite_projects_shared_context_per_owner_with_frozen_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared context is bound separately for each target and never becomes a member."""
    anonymizer = build_synthetic_anonymizer({})
    plan = anonymizer._compile_protection_plan(
        AnonymizerConfig(rewrite=Rewrite(strict_entity_protection=True), emit_telemetry=False)
    )
    assert isinstance(plan, _ProtectionPlan)
    phase8 = _GroupedRewriteBackend([], consume_context=True)

    def graph_with_shared_context(datums: tuple[_TextDatum, ...]) -> _ProtectionGraph:
        shared = _TextDatum(_DatumId("shared-context"), "shared context", _DatumPurpose.CONTEXT_ONLY)
        members = tuple(datum.id for datum in datums)
        return _ProtectionGraph(
            (*datums, shared),
            (),
            tuple(_ContextScope(datum.id, (shared.id,)) for datum in datums),
            tuple(_CoherenceScope((datum.id,)) for datum in datums),
            (_AtomicGroup(members),),
        )

    monkeypatch.setattr(protection_module, "_trivial_graph", graph_with_shared_context)
    flow = _ProtectionFlow(
        anonymizer,
        plan,
        phase6_backend=_AnchoredPhase6Backend.from_entities({}),
        phase7_backend=_CandidateBackend("Avery"),
        phase8_backend=phase8,
    )
    result = flow.protect((_record("source-a", "ordinary one"), _record("source-b", "ordinary two")))

    assert all(isinstance(outcome, _ProtectionSucceeded) for outcome in result.outcomes)
    assert phase8.context_binding_sets is not None
    bindings = phase8.context_binding_sets[0]
    assert len(bindings) == 2
    assert len({binding["binding_token"] for binding in bindings}) == 2
    assert {binding["ordinal"] for binding in bindings} == {0}
    assert {binding["text"] for binding in bindings} == {"shared context"}
    assert phase8.member_token_sets is not None
    assert {binding["owner_member_token"] for binding in bindings} == set(phase8.member_token_sets[0])


def test_private_substitute_releases_only_a_qualified_phase7_output() -> None:
    flow, backend = _private_substitute_flow(original="Alice", synthetic="Avery")

    result = flow.protect((_record("source-a", "Alice"),))

    outcome = result.outcomes[0]
    assert isinstance(outcome, _ProtectionSucceeded)
    assert isinstance(outcome.disposition, _ProtectionApplied)
    assert outcome.output == "Avery"
    assert outcome.receipt.profile == "stable-substitute-v1"
    assert backend.calls == 1
    assert backend.closed == 1
    assert backend.discarded == 1
    assert backend.value is None


def test_private_substitute_rejects_an_invalid_bundle_without_output() -> None:
    flow, backend = _private_substitute_flow(original="Alice", synthetic="Alice")

    result = flow.protect((_record("source-a", "Alice"),))

    outcome = result.outcomes[0]
    assert isinstance(outcome, _Failed)
    assert not hasattr(outcome, "output")
    assert backend.calls == 1
    assert backend.closed == 1
    assert backend.discarded == 1
    assert backend.value is None


def test_private_substitute_no_entity_scope_bypasses_phase7_adapter() -> None:
    flow, backend = _private_substitute_flow(original="absent", synthetic="Avery")

    result = flow.protect((_record("source-a", "ordinary text"),))

    outcome = result.outcomes[0]
    assert isinstance(outcome, _ProtectionSucceeded)
    assert isinstance(outcome.disposition, _NoAcceptedDetections)
    assert outcome.output == "ordinary text"
    assert backend.calls == 0
    assert backend.closed == 1
    assert backend.discarded == 1


def test_private_substitute_rejects_more_than_the_frozen_scope_limit_before_effects() -> None:
    flow, backend = _private_substitute_flow(original="Alice", synthetic="Avery")

    with pytest.raises(_ProtectionBatchError) as exc_info:
        flow.protect(
            (
                _record("source-a", "Alice"),
                _record("source-b", "Alice"),
                _record("source-c", "Alice"),
            )
        )

    assert exc_info.value.code is _BatchFailureCode.TOO_MANY_RECORDS
    assert backend.calls == 0
    assert backend.closed == 0
    assert backend.discarded == 0


def test_public_substitute_never_compiles_or_opens_a_private_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_data = _write_input(_base_dataframe(), tmp_path / "input.parquet", "parquet")
    anonymizer = _synthetic_anonymizer()

    def private_path_forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("public Substitute entered the private graph profile")

    monkeypatch.setattr(Anonymizer, "_compile_protection_plan", private_path_forbidden)
    monkeypatch.setattr(Anonymizer, "_open_protection_flow", private_path_forbidden)

    result = anonymizer.run(
        config=AnonymizerConfig(replace=Substitute(), emit_telemetry=False),
        data=input_data,
    )

    assert result.dataframe["text_replaced"].tolist() == ["Avery", "Blake", "Avery", "Casey"]
