# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

from anonymizer.engine.execution import context_admission
from anonymizer.engine.execution.accounting_admission import _AccountingAdmissionCode
from anonymizer.engine.execution.accounting_plan import _AccountingLimits
from anonymizer.engine.execution.context_admission import (
    _compile_context_plan,
    _ContextAdmissionCode,
    _ContextPlan,
    _ContextRejected,
    _is_admitted_context_plan,
)
from anonymizer.engine.execution.context_contract import (
    _BackendArtifactClass,
    _capability_satisfies,
    _ContextBackendCapability,
    _ContextExecutionContract,
    _ContextLimits,
    _ContextOrdering,
    _ContextProfile,
    _ContextSchemaVersion,
    _RetentionPosture,
)
from anonymizer.engine.execution.graph import (
    _AtomicGroup,
    _CoherenceScope,
    _ContextScope,
    _DatumId,
    _DatumPurpose,
    _ProtectionGraph,
    _TextDatum,
)


def test_graph_declares_separate_target_and_context_only_purposes() -> None:
    assert {purpose.value for purpose in _DatumPurpose} == {"target", "context_only"}


def test_context_admission_has_a_dedicated_private_module() -> None:
    assert importlib.util.find_spec("anonymizer.engine.execution.context_admission") is not None


def test_context_contract_and_workframe_modules_are_separate() -> None:
    assert importlib.util.find_spec("anonymizer.engine.execution.context_contract") is not None
    assert importlib.util.find_spec("anonymizer.engine.execution.context_workframes") is not None


def test_context_capability_is_typed_bounded_and_retention_disabled() -> None:
    contract, capability = _contract_and_capability()

    assert _capability_satisfies(capability, contract)
    assert not _capability_satisfies(
        replace(capability, retention=_RetentionPosture.ENABLED),
        contract,
    )
    assert not _capability_satisfies(
        replace(
            capability,
            limits=replace(capability.limits, max_context_bytes_per_target=3),
        ),
        contract,
    )
    with pytest.raises(TypeError):
        contract.__reduce__()


@pytest.mark.parametrize(
    "invalid_limits",
    [
        _ContextLimits(True, 32, 4, 128),
        _ContextLimits(2, -1, 4, 128),
        _ContextLimits(2, 32, cast(int, 4.0), 128),
    ],
)
def test_context_capability_rejects_non_integer_or_negative_limits(
    invalid_limits: _ContextLimits,
) -> None:
    contract, capability = _contract_and_capability()

    assert not _capability_satisfies(replace(capability, limits=invalid_limits), contract)


def test_context_compiler_contract_is_available_before_behavior_tests() -> None:
    assert callable(getattr(context_admission, "_compile_context_plan", None))


def test_context_compiler_detaches_targets_context_and_ordered_bindings() -> None:
    graph = _context_graph()
    contract, capability = _contract_and_capability()

    result = _compile_context_plan(
        graph,
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=contract,
        capability=capability,
    )

    assert isinstance(result, _ContextPlan)
    assert _is_admitted_context_plan(result)
    assert tuple(datum.id.value for datum in result.accounting.datums) == ("target-a", "target-b")
    assert tuple(datum.id.value for datum in result.context_only_datums) == ("context-c",)
    assert tuple(
        tuple(binding.datum_id.value for binding in projection.bindings) for projection in result.projections
    ) == (("context-c", "target-b"), ("target-a",))
    assert tuple(tuple(binding.ordinal for binding in projection.bindings) for projection in result.projections) == (
        (0, 1),
        (0,),
    )
    assert result.projections[0].bindings[0].scope is result.projections[0].scope
    assert result.projections[0].scope is not result.projections[1].scope

    object.__setattr__(graph.datums[2], "text", "mutated-context")
    object.__setattr__(graph.datums[0], "text", "mutated-target")
    assert result.context_only_datums[0].text == "gamma"
    assert result.accounting.datums[0].text == "alpha"
    assert _is_admitted_context_plan(result)


def test_scope_declaration_order_does_not_change_compiled_projection_order() -> None:
    graph = _context_graph()
    contract, capability = _contract_and_capability()
    declared = _compile_context_plan(
        graph,
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=contract,
        capability=capability,
    )
    reversed_declarations = _compile_context_plan(
        replace(graph, context_scopes=tuple(reversed(graph.context_scopes))),
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=contract,
        capability=capability,
    )
    assert isinstance(declared, _ContextPlan)
    assert isinstance(reversed_declarations, _ContextPlan)

    def manifest(plan: _ContextPlan) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            (
                projection.target_datum_id.value,
                tuple(binding.datum_id.value for binding in projection.bindings),
            )
            for projection in plan.projections
        )

    assert manifest(declared) == manifest(reversed_declarations)


def test_nested_contract_tampering_invalidates_the_compiled_plan() -> None:
    contract, capability = _contract_and_capability()
    result = _compile_context_plan(
        _context_graph(),
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=contract,
        capability=capability,
    )
    assert isinstance(result, _ContextPlan)

    object.__setattr__(result.contract.limits, "max_context_bytes_per_target", 1_000_000)

    assert not _is_admitted_context_plan(result)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda graph: replace(graph, context_scopes=graph.context_scopes[:1]),
            _ContextAdmissionCode.MISSING_CONTEXT_SCOPE,
        ),
        (
            lambda graph: replace(graph, context_scopes=(*graph.context_scopes, graph.context_scopes[0])),
            _ContextAdmissionCode.DUPLICATE_CONTEXT_SCOPE,
        ),
        (
            lambda graph: replace(
                graph,
                context_scopes=(
                    replace(graph.context_scopes[0], context=(graph.datums[0].id,)),
                    graph.context_scopes[1],
                ),
            ),
            _ContextAdmissionCode.SELF_CONTEXT,
        ),
        (
            lambda graph: replace(
                graph,
                context_scopes=(
                    replace(
                        graph.context_scopes[0],
                        context=(graph.datums[2].id, graph.datums[2].id),
                    ),
                    graph.context_scopes[1],
                ),
            ),
            _ContextAdmissionCode.DUPLICATE_CONTEXT_MEMBER,
        ),
        (
            lambda graph: replace(
                graph,
                context_scopes=(
                    replace(graph.context_scopes[0], context=(graph.datums[1].id,)),
                    replace(graph.context_scopes[1], context=(graph.datums[0].id,)),
                ),
            ),
            _ContextAdmissionCode.ORPHAN_CONTEXT_DATUM,
        ),
    ],
)
def test_context_admission_rejects_incomplete_semantics_before_capability(
    mutation: Callable[[_ProtectionGraph], _ProtectionGraph],
    code: _ContextAdmissionCode,
) -> None:
    graph = mutation(_context_graph())
    contract, capability = _contract_and_capability()
    weakened = replace(capability, retention=_RetentionPosture.ENABLED)

    assert _compile_context_plan(
        graph,
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=contract,
        capability=weakened,
    ) == _ContextRejected(code)


def test_context_limits_use_utf8_bytes_and_reject_before_backend_compatibility() -> None:
    graph = _context_graph()
    graph = replace(
        graph,
        datums=(*graph.datums[:2], replace(graph.datums[2], text="éé")),
    )
    contract, capability = _contract_and_capability()
    exact_limits = replace(contract.limits, max_context_bytes_per_target=8)
    exact_contract = replace(contract, limits=exact_limits)
    exact_capability = replace(capability, limits=exact_limits)

    assert isinstance(
        _compile_context_plan(
            graph,
            accounting_limits=_ACCOUNTING_LIMITS,
            contract=exact_contract,
            capability=exact_capability,
        ),
        _ContextPlan,
    )
    too_small = replace(exact_contract, limits=replace(exact_limits, max_context_bytes_per_target=7))
    assert _compile_context_plan(
        graph,
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=too_small,
        capability=replace(exact_capability, retention=_RetentionPosture.ENABLED),
    ) == _ContextRejected(_ContextAdmissionCode.CONTEXT_BYTES_EXCEEDED)


def test_retention_enabled_capability_rejects_a_valid_projection() -> None:
    contract, capability = _contract_and_capability()

    assert _compile_context_plan(
        _context_graph(),
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=contract,
        capability=replace(capability, retention=_RetentionPosture.ENABLED),
    ) == _ContextRejected(_ContextAdmissionCode.BACKEND_INCOMPATIBLE)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda capability: replace(capability, profile=cast(_ContextProfile, "future")),
        lambda capability: replace(capability, schema_version=cast(_ContextSchemaVersion, "future")),
        lambda capability: replace(capability, ordering=cast(_ContextOrdering, "implicit")),
        lambda capability: replace(capability, artifact_classes=()),
        lambda capability: replace(capability, allow_target_as_context=False),
        lambda capability: replace(
            capability,
            limits=replace(capability.limits, max_total_context_references=2),
        ),
    ],
)
def test_each_capability_dimension_fails_closed(
    mutation: Callable[[_ContextBackendCapability], _ContextBackendCapability],
) -> None:
    contract, capability = _contract_and_capability()

    assert _compile_context_plan(
        _context_graph(),
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=contract,
        capability=mutation(capability),
    ) == _ContextRejected(_ContextAdmissionCode.BACKEND_INCOMPATIBLE)


@pytest.mark.parametrize(
    ("limits", "code"),
    [
        (_ContextLimits(1, 32, 4, 128), _ContextAdmissionCode.CONTEXT_MEMBERS_EXCEEDED),
        (_ContextLimits(2, 8, 4, 128), _ContextAdmissionCode.CONTEXT_BYTES_EXCEEDED),
        (_ContextLimits(2, 32, 2, 128), _ContextAdmissionCode.TOTAL_CONTEXT_REFERENCES_EXCEEDED),
        (_ContextLimits(2, 32, 4, 22), _ContextAdmissionCode.EXPANDED_FRAME_BYTES_EXCEEDED),
    ],
)
def test_each_context_limit_has_a_closed_rejection_code(
    limits: _ContextLimits,
    code: _ContextAdmissionCode,
) -> None:
    contract, capability = _contract_and_capability()

    assert _compile_context_plan(
        _context_graph(),
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=replace(contract, limits=limits),
        capability=replace(capability, limits=limits),
    ) == _ContextRejected(code)


def test_unknown_context_references_and_context_only_targets_are_distinct() -> None:
    graph = _context_graph()
    contract, capability = _contract_and_capability()
    unknown = _DatumId("unknown")

    unknown_target = replace(
        graph,
        context_scopes=(replace(graph.context_scopes[0], target=unknown), graph.context_scopes[1]),
    )
    unknown_member = replace(
        graph,
        context_scopes=(replace(graph.context_scopes[0], context=(unknown,)), graph.context_scopes[1]),
    )
    context_only_target = replace(
        graph,
        context_scopes=(*graph.context_scopes, _ContextScope(graph.datums[2].id)),
    )

    assert _compile_context_plan(
        unknown_target,
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=contract,
        capability=capability,
    ) == _ContextRejected(_ContextAdmissionCode.UNKNOWN_CONTEXT_TARGET)
    assert _compile_context_plan(
        unknown_member,
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=contract,
        capability=capability,
    ) == _ContextRejected(_ContextAdmissionCode.UNKNOWN_CONTEXT_DATUM)
    assert _compile_context_plan(
        context_only_target,
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=contract,
        capability=capability,
    ) == _ContextRejected(_ContextAdmissionCode.CONTEXT_ONLY_TARGET)


def test_target_as_context_requires_explicit_contract_and_capability_support() -> None:
    contract, capability = _contract_and_capability()

    assert _compile_context_plan(
        _context_graph(),
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=replace(contract, allow_target_as_context=False),
        capability=replace(capability, allow_target_as_context=False),
    ) == _ContextRejected(_ContextAdmissionCode.TARGET_CONTEXT_DISABLED)


def test_missing_capability_and_unsupported_artifact_contract_fail_closed() -> None:
    contract, capability = _contract_and_capability()

    assert _compile_context_plan(
        _context_graph(),
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=contract,
        capability=None,
    ) == _ContextRejected(_ContextAdmissionCode.BACKEND_INCOMPATIBLE)
    assert _compile_context_plan(
        _context_graph(),
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=replace(contract, required_artifacts=()),
        capability=capability,
    ) == _ContextRejected(_ContextAdmissionCode.UNSUPPORTED_CONTEXT_CONTRACT)


def test_context_only_datum_cannot_become_an_atomic_output() -> None:
    graph = _context_graph()
    contract, capability = _contract_and_capability()
    graph = replace(
        graph,
        atomic_groups=(*graph.atomic_groups, _AtomicGroup((graph.datums[2].id,))),
    )

    assert _compile_context_plan(
        graph,
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=contract,
        capability=capability,
    ) == _ContextRejected(_AccountingAdmissionCode.DANGLING_ATOMIC_MEMBER)


def test_malformed_contract_fails_closed_after_structural_validation() -> None:
    _contract, capability = _contract_and_capability()

    assert _compile_context_plan(
        _context_graph(),
        accounting_limits=_ACCOUNTING_LIMITS,
        contract=object(),
        capability=capability,
    ) == _ContextRejected(_ContextAdmissionCode.UNSUPPORTED_CONTEXT_CONTRACT)


def _contract_and_capability() -> tuple[_ContextExecutionContract, _ContextBackendCapability]:
    limits = _ContextLimits(
        max_context_members_per_target=2,
        max_context_bytes_per_target=32,
        max_total_context_references=4,
        max_expanded_frame_bytes=128,
    )
    contract = _ContextExecutionContract(
        profile=_ContextProfile.TARGET_CONTEXT_V1,
        schema_version=_ContextSchemaVersion.V1,
        limits=limits,
        allow_target_as_context=True,
        ordering=_ContextOrdering.DECLARED,
        required_artifacts=(_BackendArtifactClass.CONTEXT_REQUEST,),
    )
    capability = _ContextBackendCapability(
        profile=contract.profile,
        schema_version=contract.schema_version,
        limits=limits,
        allow_target_as_context=True,
        ordering=contract.ordering,
        artifact_classes=contract.required_artifacts,
        retention=_RetentionPosture.DISABLED,
    )
    return contract, capability


def _context_graph() -> _ProtectionGraph:
    target_a = _TextDatum(_DatumId("target-a"), "alpha", _DatumPurpose.TARGET)
    target_b = _TextDatum(_DatumId("target-b"), "beta", _DatumPurpose.TARGET)
    context = _TextDatum(_DatumId("context-c"), "gamma", _DatumPurpose.CONTEXT_ONLY)
    return _ProtectionGraph(
        datums=(target_a, target_b, context),
        links=(),
        context_scopes=(
            _ContextScope(target_a.id, (context.id, target_b.id)),
            _ContextScope(target_b.id, (target_a.id,)),
        ),
        coherence_scopes=(_CoherenceScope((target_a.id,)), _CoherenceScope((target_b.id,))),
        atomic_groups=(_AtomicGroup((target_a.id,)), _AtomicGroup((target_b.id,))),
    )


_ACCOUNTING_LIMITS = _AccountingLimits(
    max_datums=8,
    max_datum_bytes=64,
    max_graph_bytes=256,
)
