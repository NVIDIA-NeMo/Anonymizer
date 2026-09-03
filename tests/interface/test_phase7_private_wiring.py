# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

import anonymizer.interface._protection as protection_module
from anonymizer.config.anonymizer_config import AnonymizerConfig, Rewrite
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.engine.execution.phase7_admission import _ScopeManifest
from anonymizer.engine.execution.phase7_ndd_backend import _Phase7NddResult, _Phase7NddStatus
from anonymizer.engine.execution.phase7_runtime import _Phase7CleanupAttestation
from anonymizer.engine.execution.phase7_validation import _CandidateAssignment
from anonymizer.engine.execution.phase8_ndd_backend import _Phase8Operation
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

    def run_operation(self, operation: _Phase8Operation, request: dict[str, object]) -> object:
        self.calls.append(operation)
        supplied_tokens = request["member_tokens"]
        assert isinstance(supplied_tokens, list) and all(isinstance(token, str) for token in supplied_tokens)
        tokens = supplied_tokens
        if operation is _Phase8Operation.ANALYZE:
            return _Dispatch(operation, {"analyzed_member_tokens": tokens, "privacy_obligations": [{"id": "p"}]})
        if operation is _Phase8Operation.EVALUATE:
            self.evaluations += 1
            return _Dispatch(
                operation,
                {
                    "evaluated_member_tokens": tokens,
                    "privacy_answers": [
                        {
                            "sensitivity": "high",
                            "confidence": 1.0 if self.evaluations == 1 else 0.0,
                            "leaked": self.evaluations == 1,
                        }
                    ],
                    "utility_answers": [{"weight": 1, "score": 1.0}],
                },
            )
        return _Dispatch(
            operation,
            {"revisions": [{"member_token": token, "text": f"rewrite-{index}"} for index, token in enumerate(tokens)]},
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
    rewrite = anonymizer._compile_protection_plan(AnonymizerConfig(rewrite=Rewrite()))
    assert isinstance(rewrite, _ProtectionPlan)
    assert rewrite.profile == "grouped-rewrite-v1"
    assert isinstance(
        anonymizer._compile_protection_plan(AnonymizerConfig(replace=Substitute(instructions="legacy only"))),
        _PlanRejected,
    )


def test_private_grouped_rewrite_executes_phase7_then_all_phase8_operations() -> None:
    anonymizer = build_synthetic_anonymizer({"Alice": "first_name"})
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(rewrite=Rewrite(), emit_telemetry=False))
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


def test_private_grouped_rewrite_with_no_entities_makes_no_phase7_or_phase8_calls() -> None:
    anonymizer = build_synthetic_anonymizer({"Alice": "first_name"})
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(rewrite=Rewrite(), emit_telemetry=False))
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
    assert result.outcomes[0].output == "ordinary text"
    assert phase7.calls == 0
    assert phase8.calls == []


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
