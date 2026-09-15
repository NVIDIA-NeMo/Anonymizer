# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import importlib
import inspect
import json
import pickle
import weakref
from dataclasses import FrozenInstanceError, fields, replace
from typing import Any, Never, cast

import pytest

from anonymizer.engine.execution.accounting_admission import (
    _AccountingAdmissionCode,
    _AccountingRejected,
    _compile_accounting_plan,
)
from anonymizer.engine.execution.accounting_ledger import _AccountingLedger
from anonymizer.engine.execution.accounting_plan import _AccountingLimits, _AccountingPlan
from anonymizer.engine.execution.graph import _DatumId, _RewriteGroup, _TextDatum, _trivial_graph
from anonymizer.engine.execution.phase8_cleanup import (
    _issue_phase8_cleanup_receipt,
    _Phase8CleanupComponent,
    _Phase8CleanupPhase,
    _Phase8CleanupStatus,
)


def _module() -> Any:
    return importlib.import_module("anonymizer.engine.execution.phase10_inspection")


class _InspectionIdentity:
    pass


def _provenance(module: Any, kind: Any) -> Any:
    return module._Phase10Provenance(
        module._Phase10InspectionSchemaVersion.V1,
        module._Phase10ContractVersion.V1,
        kind,
        module._Phase10SubjectKind.INVOCATION_SNAPSHOT,
        module._Phase10SemanticProfile.GROUPED_REWRITE_V1,
        module._Phase10ImplementationProfile.PANDAS_RUNTIME_V1,
        module._Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED,
        module._Phase10LifecycleState.TERMINAL,
    )


def _diagnostic(module: Any) -> Any:
    return module._Phase10Diagnostic(
        module._Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED,
        module._Phase10Stage.REWRITE,
        module._Phase10TerminalState.FAILED,
        module._Phase10ReasonCategory.BACKEND_FAILED,
        module._Phase10CountBucket.ONE,
        module._Phase10ReconciliationState.RECONCILED,
        module._Phase10CleanupState.NOT_ENTERED,
    )


def _encoder_views(module: Any) -> tuple[Any, Any, Any]:
    explain_provenance = module._Phase10Provenance(
        module._Phase10InspectionSchemaVersion.V1,
        module._Phase10ContractVersion.V1,
        module._Phase10ViewKind.EXPLAIN,
        module._Phase10SubjectKind.ADMITTED_PLAN,
        module._Phase10SemanticProfile.TARGET_CONTEXT_V1,
        module._Phase10ImplementationProfile.PANDAS_RUNTIME_V1,
        module._Phase10CaptureBoundary.ADMISSION_TERMINAL,
        module._Phase10LifecycleState.TERMINAL,
    )
    explain = module._Phase10ExplainView(
        explain_provenance,
        module._Phase10Route.NDD,
        (module._Phase10Capability.TERMINAL_ACCOUNTING,),
        module._phase10_declared_limits(),
        (),
        None,
    )
    inspect_view = module._Phase10InspectView(
        _provenance(module, module._Phase10ViewKind.INSPECT),
        module._Phase10Snapshot(
            (),
            (),
            module._Phase10ReconciliationState.RECONCILED,
            module._Phase10CleanupState.NOT_ENTERED,
            module._Phase10ReleaseState.WITHHELD,
        ),
    )
    diagnose = module._Phase10DiagnoseView(
        _provenance(module, module._Phase10ViewKind.DIAGNOSE),
        (_diagnostic(module),),
    )
    return explain, inspect_view, diagnose


def _accounting_plan(*texts: str, stages: tuple[str, ...] = ("protect",)) -> _AccountingPlan:
    graph = _trivial_graph(
        tuple(_TextDatum(_DatumId(f"private-datum-{index}"), text) for index, text in enumerate(texts))
    )
    result = _compile_accounting_plan(
        graph,
        limits=_AccountingLimits(
            max_datums=len(texts),
            max_datum_bytes=256,
            max_graph_bytes=512,
            max_stages=len(stages),
        ),
        stages=stages,
    )
    assert isinstance(result, _AccountingPlan)
    return result


def test_phase10_private_values_reject_unknown_closed_values() -> None:
    module = _module()

    for enum_type in (
        module._Phase10Operation,
        module._Phase10ViewKind,
        module._Phase10SubjectKind,
        module._Phase10CaptureBoundary,
        module._Phase10LifecycleState,
        module._Phase10ReasonCategory,
        module._Phase10CountBucket,
    ):
        with pytest.raises(ValueError):
            enum_type("unknown")
    with pytest.raises(TypeError):
        module._Phase10Diagnostic(
            "terminal_evidence_accepted",
            module._Phase10Stage.REWRITE,
            module._Phase10TerminalState.FAILED,
            module._Phase10ReasonCategory.BACKEND_FAILED,
            module._Phase10CountBucket.ONE,
            module._Phase10ReconciliationState.RECONCILED,
            module._Phase10CleanupState.NOT_ENTERED,
        )
    for value in (True, 1.0, -1, 65_537):
        with pytest.raises(TypeError):
            module._Phase10DeclaredLimit(module._Phase10LimitName.SUBJECTS_PER_CALL, value)


def test_phase10_grant_is_identity_bound_single_use_and_cause_free() -> None:
    module = _module()
    owner = _InspectionIdentity()
    subject = _InspectionIdentity()
    grant = module._issue_phase10_inspection_grant(owner, subject, module._Phase10Operation.INSPECT)

    assert type(grant) is module._Phase10InspectionGrant
    assert not module._consume_phase10_inspection_grant(
        grant,
        owner,
        subject,
        module._Phase10Operation.EXPLAIN,
    )
    assert module._consume_phase10_inspection_grant(
        grant,
        owner,
        subject,
        module._Phase10Operation.INSPECT,
    )
    assert not module._consume_phase10_inspection_grant(
        grant,
        owner,
        subject,
        module._Phase10Operation.INSPECT,
    )
    assert repr(grant).startswith("<private ")
    assert repr(owner) not in repr(grant)
    assert repr(subject) not in repr(grant)
    assert not hasattr(grant, "_state")
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(grant)
    with pytest.raises(FrozenInstanceError):
        grant.operation = module._Phase10Operation.EXPLAIN


def test_phase10_grant_rejects_forgery_wrong_identity_and_expiry() -> None:
    module = _module()
    owner = _InspectionIdentity()
    subject = _InspectionIdentity()
    grant = module._issue_phase10_inspection_grant(owner, subject, module._Phase10Operation.DIAGNOSE)

    assert not module._consume_phase10_inspection_grant(
        grant,
        object(),
        subject,
        module._Phase10Operation.DIAGNOSE,
    )
    module._revoke_phase10_inspection_grant(grant)
    assert not module._consume_phase10_inspection_grant(
        grant,
        owner,
        subject,
        module._Phase10Operation.DIAGNOSE,
    )
    forged = module._Phase10InspectionGrant(module._Phase10Operation.DIAGNOSE, object(), object())
    assert not module._consume_phase10_inspection_grant(
        forged,
        owner,
        subject,
        module._Phase10Operation.DIAGNOSE,
    )


def test_phase10_snapshot_view_provenance_and_diagnostic_are_immutable_and_masked() -> None:
    module = _module()
    stage = module._Phase10StageSummary(
        module._Phase10Stage.REWRITE,
        module._Phase10LifecycleState.TERMINAL,
        module._Phase10CountBucket.ONE,
    )
    terminal = module._Phase10TerminalSummary(
        module._Phase10Stage.REWRITE,
        module._Phase10TerminalState.FAILED,
        module._Phase10CountBucket.ONE,
    )
    snapshot = module._Phase10Snapshot(
        (stage,),
        (terminal,),
        module._Phase10ReconciliationState.RECONCILED,
        module._Phase10CleanupState.NOT_ENTERED,
        module._Phase10ReleaseState.WITHHELD,
    )
    view = module._Phase10InspectView(_provenance(module, module._Phase10ViewKind.INSPECT), snapshot)

    for value in (stage, terminal, snapshot, view.provenance, _diagnostic(module), view):
        assert repr(value).startswith("<private ")
        with pytest.raises(TypeError, match="not serializable"):
            pickle.dumps(value)
    with pytest.raises(FrozenInstanceError):
        snapshot.cleanup_state = module._Phase10CleanupState.VERIFIED


def test_phase10_encoder_emits_exact_canonical_json_and_preserves_semantic_array_order() -> None:
    module = _module()
    provenance = module._Phase10Provenance(
        module._Phase10InspectionSchemaVersion.V1,
        module._Phase10ContractVersion.V1,
        module._Phase10ViewKind.EXPLAIN,
        module._Phase10SubjectKind.ADMITTED_PLAN,
        module._Phase10SemanticProfile.GROUPED_REWRITE_V1,
        module._Phase10ImplementationProfile.PANDAS_RUNTIME_V1,
        module._Phase10CaptureBoundary.ADMISSION_TERMINAL,
        module._Phase10LifecycleState.TERMINAL,
    )
    view = module._Phase10ExplainView(
        provenance,
        module._Phase10Route.MIXED,
        (module._Phase10Capability.GROUPED_REWRITE, module._Phase10Capability.TERMINAL_ACCOUNTING),
        module._phase10_declared_limits(),
        (
            module._Phase10Aggregate(
                module._Phase10AggregateDimension.RELATIONSHIPS, module._Phase10CountBucket.TWO_TO_FOUR
            ),
        ),
        None,
    )

    result = module._encode_phase10_view(view)

    assert type(result) is module._Phase10CanonicalEncoding
    assert not result.value.endswith(b"\n")
    assert result.value == json.dumps(
        json.loads(result.value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = json.loads(result.value)
    assert [item["name"] for item in payload["declared_limits"]] == [
        "subjects_per_call",
        "views_per_call",
        "max_stage_summaries",
        "max_terminal_summary_rows",
        "max_diagnostic_entries",
        "max_reason_codes_per_diagnostic_entry",
        "max_provenance_fields",
        "max_top_level_fields",
        "max_json_nesting_depth",
        "max_allowlisted_string_utf8_bytes",
        "max_canonical_json_utf8_bytes",
        "max_builder_working_bytes",
    ]
    assert payload["required_capabilities"] == ["grouped_rewrite", "terminal_accounting"]
    assert payload["provenance"] == {
        "capture_boundary": "admission_terminal",
        "capture_lifecycle_state": "terminal",
        "implementation_profile_version": "pandas-runtime-v1",
        "inspection_contract_version": "anonymizer-phase10-bounded-inspection/v1",
        "inspection_schema_version": "phase10-bounded-inspection-view/v1",
        "semantic_profile_version": "anonymizer-phase8-grouped-rewrite/v1",
        "subject_kind": "admitted_plan",
        "view_kind": "explain",
    }
    assert repr(result).startswith("<private ")
    assert result.value.decode("utf-8") not in repr(result)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(result)


def test_phase10_encoder_rejects_unknown_values_and_encoding_overflow_without_partial_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    unknown = module._encode_phase10_view(object())
    assert type(unknown) is module._Phase10InspectionRejected
    assert unknown.code is module._Phase10RejectionCode.REDACTION_FAILED

    view = module._Phase10DiagnoseView(
        _provenance(module, module._Phase10ViewKind.DIAGNOSE),
        (_diagnostic(module),),
    )
    monkeypatch.setitem(
        module._PHASE10_LIMITS,
        module._Phase10LimitName.MAX_CANONICAL_JSON_UTF8_BYTES,
        1,
    )
    oversized = module._encode_phase10_view(view)

    assert type(oversized) is module._Phase10InspectionRejected
    assert oversized.code is module._Phase10RejectionCode.LIMIT_EXCEEDED
    assert "inspection_limit_exceeded" not in repr(oversized)


def test_phase10_payload_measurement_enforces_exact_and_one_over_boundaries() -> None:
    module = _module()

    exact_root = {f"k{index}": None for index in range(12)}
    exact_provenance = {"provenance": {f"k{index}": None for index in range(8)}}
    exact_depth = {"a": {"b": {"c": {"d": {}}}}}
    exact_list_depth = {"a": [[[[]]]]}
    exact_ascii = {"a" * 96: "b" * 96}
    exact_utf8 = {"value": "é" * 48}
    for payload, expected in (
        (exact_root, (12, 0, 1, 3)),
        (exact_provenance, (1, 8, 2, 10)),
        (exact_depth, (1, 0, 5, 1)),
        (exact_list_depth, (1, 0, 5, 1)),
        (exact_ascii, (1, 0, 1, 96)),
        (exact_utf8, (1, 0, 1, 96)),
    ):
        measured = module._measure_phase10_payload(payload)
        assert type(measured) is module._Phase10PayloadMeasurement
        assert (
            measured.top_level_fields,
            measured.provenance_fields,
            measured.json_nesting_depth,
            measured.longest_string_utf8_bytes,
        ) == expected

    over_root = {f"k{index}": None for index in range(13)}
    over_provenance = {"provenance": {f"k{index}": None for index in range(9)}}
    over_depth = {"a": {"b": {"c": {"d": {"e": {}}}}}}
    over_list_depth = {"a": [[[[[]]]]]}
    for payload in (
        over_root,
        over_provenance,
        over_depth,
        over_list_depth,
        {"value": "a" * 97},
        {"value": "é" * 49},
    ):
        rejected = module._measure_phase10_payload(payload)
        assert type(rejected) is module._Phase10InspectionRejected
        assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED


def test_phase10_payload_measurement_rejects_non_exact_builtins_without_traversal() -> None:
    module = _module()

    class DictSubclass(dict[str, object]):
        def items(self) -> Never:
            raise AssertionError("must not traverse a dictionary subclass")

    class ListSubclass(list[object]):
        def __iter__(self) -> Never:
            raise AssertionError("must not traverse a list subclass")

    class StringSubclass(str):
        def encode(self, *_args: object, **_kwargs: object) -> bytes:
            raise AssertionError("must not encode a string subclass")

    class IntSubclass(int):
        pass

    class RichValue:
        def __getattribute__(self, _name: str) -> object:
            raise AssertionError("must not inspect a rich object")

    payloads = (
        DictSubclass(),
        {"value": ListSubclass()},
        {"value": StringSubclass("unsafe")},
        {"value": IntSubclass(1)},
        {"value": 1.0},
        {"value": b"unsafe"},
        {"value": ()},
        {"value": RichValue()},
        {StringSubclass("key"): None},
        {"value": "\ud800"},
    )
    for payload in payloads:
        rejected = module._measure_phase10_payload(payload)
        assert type(rejected) is module._Phase10InspectionRejected
        assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED

    admitted = module._measure_phase10_payload({"values": [None, True, 1, "safe", [], {}]})
    assert type(admitted) is module._Phase10PayloadMeasurement


def test_phase10_encoder_applies_measurement_before_exact_schema_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    view = _encoder_views(module)[2]

    monkeypatch.setattr(module, "_view_payload", lambda _value: {"unknown": None})
    malformed = module._encode_phase10_view(view)
    assert type(malformed) is module._Phase10InspectionRejected
    assert malformed.code is module._Phase10RejectionCode.REDACTION_FAILED

    monkeypatch.setattr(module, "_view_payload", lambda _value: {f"unknown-{index}": None for index in range(13)})
    oversized = module._encode_phase10_view(view)
    assert type(oversized) is module._Phase10InspectionRejected
    assert oversized.code is module._Phase10RejectionCode.LIMIT_EXCEEDED


def test_phase10_payload_failure_precedence_stops_before_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    view = _encoder_views(module)[2]
    original_payload = module._view_payload
    encoder_calls = 0

    def throwing_encoder(*_args: object, **_kwargs: object) -> str:
        nonlocal encoder_calls
        encoder_calls += 1
        raise RuntimeError("PAYLOAD-PRECEDENCE-CANARY")

    monkeypatch.setattr(module.json, "dumps", throwing_encoder)
    monkeypatch.setattr(module, "_view_payload", lambda _value: {"value": object()})
    unsafe = module._encode_phase10_view(view)
    assert type(unsafe) is module._Phase10InspectionRejected
    assert unsafe.code is module._Phase10RejectionCode.REDACTION_FAILED
    assert not hasattr(unsafe, "value")
    assert encoder_calls == 0

    monkeypatch.setattr(module, "_view_payload", lambda _value: {f"field-{index}": None for index in range(13)})
    oversized = module._encode_phase10_view(view)
    assert type(oversized) is module._Phase10InspectionRejected
    assert oversized.code is module._Phase10RejectionCode.LIMIT_EXCEEDED
    assert not hasattr(oversized, "value")
    assert encoder_calls == 0

    monkeypatch.setattr(module, "_view_payload", original_payload)
    encoding_failed = module._encode_phase10_view(view)
    assert type(encoding_failed) is module._Phase10InspectionRejected
    assert encoding_failed.code is module._Phase10RejectionCode.ENCODING_FAILED
    assert not hasattr(encoding_failed, "value")
    assert "PAYLOAD-PRECEDENCE-CANARY" not in repr(encoding_failed)
    assert encoder_calls == 1


def test_phase10_every_encoder_variant_invokes_validator_and_enforces_actual_byte_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    views = _encoder_views(module)
    original_validator = module._validate_phase10_payload
    validated_kinds: list[str] = []

    def recording_validator(payload: object) -> object:
        if type(payload) is dict:
            view_kind = cast(dict[str, object], payload).get("view_kind")
            if type(view_kind) is str:
                validated_kinds.append(view_kind)
        return original_validator(payload)

    monkeypatch.setattr(module, "_validate_phase10_payload", recording_validator)
    limit_name = module._Phase10LimitName.MAX_CANONICAL_JSON_UTF8_BYTES
    original_limit = module._PHASE10_LIMITS[limit_name]
    for view in views:
        monkeypatch.setitem(module._PHASE10_LIMITS, limit_name, original_limit)
        baseline = module._encode_phase10_view(view)
        assert type(baseline) is module._Phase10CanonicalEncoding

        saved_bytes = baseline.value
        saved_limits = view.declared_limits if type(view) is module._Phase10ExplainView else None
        ceiling = len(saved_bytes)
        for _ in range(5):
            monkeypatch.setitem(module._PHASE10_LIMITS, limit_name, ceiling)
            exact_view = (
                replace(view, declared_limits=module._phase10_declared_limits())
                if type(view) is module._Phase10ExplainView
                else view
            )
            expected_bytes = json.dumps(
                module._view_payload(exact_view), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            if len(expected_bytes) == ceiling:
                break
            ceiling = len(expected_bytes)
        assert len(expected_bytes) == ceiling
        exact = module._encode_phase10_view(exact_view)
        assert type(exact) is module._Phase10CanonicalEncoding
        assert exact.value == expected_bytes

        monkeypatch.setitem(module._PHASE10_LIMITS, limit_name, ceiling - 1)
        over_view = (
            replace(view, declared_limits=module._phase10_declared_limits())
            if type(view) is module._Phase10ExplainView
            else view
        )
        over = module._encode_phase10_view(over_view)
        assert type(over) is module._Phase10InspectionRejected
        assert over.code is module._Phase10RejectionCode.LIMIT_EXCEEDED
        assert baseline.value == saved_bytes
        assert exact.value == expected_bytes
        if saved_limits is not None:
            assert view.declared_limits == saved_limits

    assert validated_kinds == [
        "explain",
        "explain",
        "explain",
        "inspect",
        "inspect",
        "inspect",
        "diagnose",
        "diagnose",
        "diagnose",
    ]


def test_phase10_operations_remain_fixed_arity_scalar_calls() -> None:
    module = _module()

    for name in ("_explain_phase10", "_inspect_phase10", "_diagnose_phase10"):
        operation = getattr(module, name)
        signature = inspect.signature(operation)
        assert tuple(signature.parameters) == ("owner", "subject", "grant")
        assert all(
            parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in signature.parameters.values()
        )
        with pytest.raises(TypeError):
            operation(object(), object(), object(), object())


def test_phase10_types_are_not_exposed_through_public_surfaces() -> None:
    import anonymizer
    import anonymizer.interface
    from anonymizer import Anonymizer

    public_names = set(dir(anonymizer)) | set(dir(anonymizer.interface)) | set(dir(Anonymizer))
    assert not any(
        name in public_names
        for name in (
            "explain",
            "inspect",
            "diagnose",
            "InspectionGrant",
            "InspectionView",
            "Phase10Snapshot",
        )
    )


def test_phase10_subject_domain_defers_publication_receipt_but_keeps_publication_outcomes() -> None:
    module = _module()

    assert tuple(item.value for item in module._Phase10SubjectKind) == (
        "admitted_plan",
        "admission_rejection_receipt",
        "invocation_snapshot",
        "terminal_receipt",
        "cleanup_receipt",
    )
    assert not hasattr(module._Phase10SubjectKind, "PUBLICATION_RECEIPT")
    assert module._Phase10Stage.PUBLICATION.value == "publication"
    assert module._Phase10ReasonCategory.PUBLICATION_FAILED.value == "publication_failed"
    assert module._Phase10CaptureBoundary.RELEASE_TERMINAL.value == "release_terminal"


def test_phase10_explain_projects_admitted_plan_and_rejection_without_content() -> None:
    module = _module()
    assert callable(getattr(module, "_explain_phase10", None)), "Tier 2 explain builder is missing"
    owner = _InspectionIdentity()
    plan = _accounting_plan("EXPLAIN-CONTENT-CANARY")
    grant = module._issue_phase10_inspection_grant(owner, plan, module._Phase10Operation.EXPLAIN)

    admitted = module._explain_phase10(owner, plan, grant)

    assert type(admitted) is module._Phase10ExplainView
    assert admitted.route is module._Phase10Route.NDD
    assert admitted.provenance.subject_kind is module._Phase10SubjectKind.ADMITTED_PLAN
    assert admitted.required_capabilities == (module._Phase10Capability.TERMINAL_ACCOUNTING,)
    encoded = module._encode_phase10_view(admitted)
    assert type(encoded) is module._Phase10CanonicalEncoding
    assert b"EXPLAIN-CONTENT-CANARY" not in encoded.value
    assert b"private-datum" not in encoded.value

    rejection = _AccountingRejected(_AccountingAdmissionCode.TOO_MANY_DATUMS)
    rejection_grant = module._issue_phase10_inspection_grant(
        owner,
        rejection,
        module._Phase10Operation.EXPLAIN,
    )
    rejected = module._explain_phase10(owner, rejection, rejection_grant)

    assert type(rejected) is module._Phase10ExplainView
    assert rejected.route is module._Phase10Route.REJECTED
    assert rejected.rejection_category is module._Phase10ReasonCategory.LIMIT_EXCEEDED
    assert rejected.provenance.subject_kind is module._Phase10SubjectKind.ADMISSION_REJECTION


def test_phase10_explain_projects_grouped_rewrite_plan_in_compiled_order() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_admission")
    graph = _trivial_graph((_TextDatum(_DatumId("phase8-private"), "PHASE8-PLAN-CANARY"),))
    graph = replace(graph, rewrite_groups=(_RewriteGroup((graph.datums[0].id,)),))
    plan = phase8._compile_phase8_plan(graph, max_repairs=2)
    assert phase8._is_admitted_phase8_plan(plan)
    owner = _InspectionIdentity()
    grant = module._issue_phase10_inspection_grant(owner, plan, module._Phase10Operation.EXPLAIN)

    explained = module._explain_phase10(owner, plan, grant)

    assert type(explained) is module._Phase10ExplainView
    assert explained.required_capabilities[-1] is module._Phase10Capability.GROUPED_REWRITE
    assert tuple(item.dimension for item in explained.relationship_buckets) == (
        module._Phase10AggregateDimension.DATUMS,
        module._Phase10AggregateDimension.GROUPS,
        module._Phase10AggregateDimension.OPERATIONS,
        module._Phase10AggregateDimension.REPAIRS,
    )
    encoded = module._encode_phase10_view(explained)
    assert type(encoded) is module._Phase10CanonicalEncoding
    assert b"PHASE8-PLAN-CANARY" not in encoded.value


def test_phase10_inspect_and_diagnose_use_one_owner_linearized_accounting_snapshot() -> None:
    module = _module()
    owner = _InspectionIdentity()
    ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("LEDGER-CONTENT-CANARY"))
    ledger.open()
    (task,) = ledger.ready_tasks()
    dispatch = ledger.dispatch(task)
    ledger.accept_failure(dispatch)
    capture: Any = ledger._phase10_snapshot()
    assert type(capture) is module._Phase10OwnerCapture
    assert repr(capture).startswith("<private ")
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(capture)

    inspect_grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.INSPECT)
    inspected = module._inspect_phase10(owner, capture, inspect_grant)

    assert type(inspected) is module._Phase10InspectView
    assert inspected.provenance.capture_boundary is module._Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED
    assert inspected.snapshot.stage_summaries == (
        module._Phase10StageSummary(
            module._Phase10Stage.TRANSFORM,
            module._Phase10LifecycleState.TERMINAL,
            module._Phase10CountBucket.ONE,
        ),
    )
    assert inspected.snapshot.terminal_summaries == (
        module._Phase10TerminalSummary(
            module._Phase10Stage.TRANSFORM,
            module._Phase10TerminalState.FAILED,
            module._Phase10CountBucket.ONE,
        ),
    )

    diagnose_grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.DIAGNOSE)
    diagnosed = module._diagnose_phase10(owner, capture, diagnose_grant)

    assert type(diagnosed) is module._Phase10DiagnoseView
    assert diagnosed.diagnostics == (
        module._Phase10Diagnostic(
            module._Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED,
            module._Phase10Stage.TRANSFORM,
            module._Phase10TerminalState.FAILED,
            module._Phase10ReasonCategory.BACKEND_FAILED,
            module._Phase10CountBucket.ONE,
            module._Phase10ReconciliationState.RECONCILED,
            module._Phase10CleanupState.NOT_ENTERED,
        ),
    )
    for view in (inspected, diagnosed):
        encoded = module._encode_phase10_view(view)
        assert type(encoded) is module._Phase10CanonicalEncoding
        assert b"LEDGER-CONTENT-CANARY" not in encoded.value
        assert b"private-datum" not in encoded.value


def test_phase10_denies_before_subject_snapshot_access() -> None:
    module = _module()
    owner = _InspectionIdentity()

    class DenialProbe:
        def __getattribute__(self, _name: str) -> object:
            raise AssertionError("subject accessed before grant validation")

    subject = DenialProbe()
    forged = module._Phase10InspectionGrant(module._Phase10Operation.INSPECT, object(), object())

    denied = module._inspect_phase10(owner, subject, forged)

    assert type(denied) is module._Phase10InspectionRejected
    assert denied.code is module._Phase10RejectionCode.DENIED


def test_phase10_cleanup_receipt_snapshot_is_bucketed_and_identity_free() -> None:
    module = _module()
    owner = _InspectionIdentity()
    identity = object()
    receipt = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.POST_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.UNCONFIRMED,
        identity,
        retained_candidate_cell_count=17,
    )
    capture: Any = receipt._phase10_snapshot()
    assert type(capture) is module._Phase10OwnerCapture
    grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.INSPECT)

    inspected = module._inspect_phase10(owner, capture, grant)

    assert type(inspected) is module._Phase10InspectView
    assert inspected.provenance.subject_kind is module._Phase10SubjectKind.CLEANUP_RECEIPT
    assert inspected.provenance.capture_boundary is module._Phase10CaptureBoundary.POST_REDUCTION_CLEANUP_TERMINAL
    assert inspected.snapshot.cleanup_state is module._Phase10CleanupState.UNCONFIRMED
    assert inspected.snapshot.stage_summaries == (
        module._Phase10StageSummary(
            module._Phase10Stage.CLEANUP,
            module._Phase10LifecycleState.CLEANUP_TERMINAL,
            module._Phase10CountBucket.ONE,
        ),
    )
    assert inspected.snapshot.terminal_summaries == (
        module._Phase10TerminalSummary(
            module._Phase10Stage.CLEANUP,
            module._Phase10TerminalState.INCONSISTENT,
            module._Phase10CountBucket.SEVENTEEN_TO_SIXTY_FOUR,
        ),
    )
    encoded = module._encode_phase10_view(inspected)
    assert type(encoded) is module._Phase10CanonicalEncoding
    assert str(id(identity)).encode() not in encoded.value


def test_phase10_phase8_operation_snapshot_collapses_rounds_and_preserves_reason_order() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(2, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    ledger.succeed(phase8._Phase8Stage.validate_baselines())
    ledger.fail(phase8._Phase8Stage.analyze(), phase8._Phase8Reason.BACKEND_FAILURE)
    owner = _InspectionIdentity()
    capture: Any = ledger._phase10_snapshot()
    assert type(capture) is module._Phase10OwnerCapture
    inspect_grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.INSPECT)

    inspected = module._inspect_phase10(owner, capture, inspect_grant)

    assert type(inspected) is module._Phase10InspectView
    assert tuple(item.stage for item in inspected.snapshot.stage_summaries) == (
        module._Phase10Stage.VALIDATE,
        module._Phase10Stage.ANALYZE,
        module._Phase10Stage.REWRITE,
        module._Phase10Stage.EVALUATE,
        module._Phase10Stage.REPAIR,
    )
    diagnose_grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.DIAGNOSE)
    diagnosed = module._diagnose_phase10(owner, capture, diagnose_grant)
    assert type(diagnosed) is module._Phase10DiagnoseView
    assert tuple(item.reason_category for item in diagnosed.diagnostics) == (
        module._Phase10ReasonCategory.BACKEND_FAILED,
        module._Phase10ReasonCategory.PREREQUISITE_BLOCKED,
        module._Phase10ReasonCategory.PREREQUISITE_BLOCKED,
        module._Phase10ReasonCategory.PREREQUISITE_BLOCKED,
    )


def test_phase10_reason_table_is_total_for_every_current_phase4_through_phase8_owner_reason() -> None:
    module = _module()

    assert module._unmapped_phase10_reason_values() == ()
    assert {reason_type.__name__ for reason_type in module._PHASE10_REASON_TABLE} == {
        "_AccountingAdmissionCode",
        "_ApplicationRejectionCode",
        "_BundleRejectionCode",
        "_CauseCode",
        "_ContextAdmissionCode",
        "_ContextBindingFault",
        "_MentionRejectionCode",
        "_PatchRejectionCode",
        "_Phase6PlanRejectionCode",
        "_Phase7AdmissionCode",
        "_Phase7ContractRejectionCode",
        "_Phase7NddReason",
        "_Phase8AdmissionCode",
        "_Phase8Reason",
        "_ResolutionRejectionCode",
        "_RolePolicyRejectionCode",
        "_UnsupportedRoleReason",
    }
    assert module._map_phase10_reason(object()) is module._Phase10ReasonCategory.UNEXPECTED_FAILURE


def test_phase10_phase8_cancellation_diagnostics_distinguish_pre_dispatch_from_trusted_stop() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(0, 3)
    assert plan is not None
    before = phase8._Phase8OperationLedger(plan)
    before.cancel(phase8._Phase8Stage.validate_baselines(), trusted_stop=True, dispatched=False)
    before_capture: Any = before._phase10_snapshot()
    assert tuple(item.reason_category for item in before_capture.diagnostics) == (
        module._Phase10ReasonCategory.CANCELLATION_BEFORE_DISPATCH,
        module._Phase10ReasonCategory.PREREQUISITE_BLOCKED,
        module._Phase10ReasonCategory.PREREQUISITE_BLOCKED,
        module._Phase10ReasonCategory.PREREQUISITE_BLOCKED,
    )

    after = phase8._Phase8OperationLedger(plan)
    after.succeed(phase8._Phase8Stage.validate_baselines())
    after.cancel(phase8._Phase8Stage.analyze(), trusted_stop=True, dispatched=True)
    after_capture: Any = after._phase10_snapshot()
    assert tuple(item.reason_category for item in after_capture.diagnostics) == (
        module._Phase10ReasonCategory.STOP_ACKNOWLEDGED,
        module._Phase10ReasonCategory.PREREQUISITE_BLOCKED,
        module._Phase10ReasonCategory.PREREQUISITE_BLOCKED,
    )


def test_phase10_phase8_successful_no_repair_route_has_no_failure_diagnostic() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(1, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    for stage in (
        phase8._Phase8Stage.validate_baselines(),
        phase8._Phase8Stage.analyze(),
        phase8._Phase8Stage.rewrite(),
        phase8._Phase8Stage.evaluate(0),
    ):
        ledger.succeed(stage)
    ledger.close_pass(phase8._Phase8Stage.evaluate(0))
    capture: Any = ledger._phase10_snapshot()
    assert capture.diagnostics == ()
    owner = _InspectionIdentity()
    grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.DIAGNOSE)

    rejected = module._diagnose_phase10(owner, capture, grant)

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.STATE_UNAVAILABLE


def test_phase10_owner_capture_rejects_incoherent_boundary_and_diagnostic_state() -> None:
    module = _module()
    snapshot = module._Phase10Snapshot(
        (
            module._Phase10StageSummary(
                module._Phase10Stage.REWRITE,
                module._Phase10LifecycleState.TERMINAL,
                module._Phase10CountBucket.ONE,
            ),
        ),
        (
            module._Phase10TerminalSummary(
                module._Phase10Stage.REWRITE,
                module._Phase10TerminalState.FAILED,
                module._Phase10CountBucket.ONE,
            ),
        ),
        module._Phase10ReconciliationState.RECONCILED,
        module._Phase10CleanupState.NOT_ENTERED,
        module._Phase10ReleaseState.WITHHELD,
    )
    diagnostic = module._Phase10Diagnostic(
        module._Phase10CaptureBoundary.POST_DISPATCH,
        module._Phase10Stage.REWRITE,
        module._Phase10TerminalState.FAILED,
        module._Phase10ReasonCategory.BACKEND_FAILED,
        module._Phase10CountBucket.ONE,
        module._Phase10ReconciliationState.RECONCILED,
        module._Phase10CleanupState.NOT_ENTERED,
    )

    with pytest.raises(TypeError, match="owner capture"):
        module._Phase10OwnerCapture(
            module._Phase10SubjectKind.INVOCATION_SNAPSHOT,
            module._Phase10SemanticProfile.GROUPED_REWRITE_V1,
            module._Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED,
            module._Phase10LifecycleState.TERMINAL,
            snapshot,
            (diagnostic,),
        )


def test_phase10_closed_graph_execution_snapshot_copies_no_candidate_or_plan_content() -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    plan = _accounting_plan("GRAPH-PLAN-CONTENT-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    (task,) = ledger.ready_tasks()
    ledger.mark_task_succeeded(task, "GRAPH-CANDIDATE-CONTENT-CANARY")
    result = ledger.finish()
    execution = graph_runtime._AccountingGraphExecution(plan, result, ())

    capture: Any = execution._phase10_snapshot()

    assert type(capture) is module._Phase10OwnerCapture
    assert capture.subject_kind is module._Phase10SubjectKind.TERMINAL_RECEIPT
    assert capture.snapshot.release_state is module._Phase10ReleaseState.RELEASED
    owner = _InspectionIdentity()
    grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.INSPECT)
    inspected = module._inspect_phase10(owner, capture, grant)
    encoded = module._encode_phase10_view(inspected)
    assert type(encoded) is module._Phase10CanonicalEncoding
    assert b"GRAPH-PLAN-CONTENT-CANARY" not in encoded.value
    assert b"GRAPH-CANDIDATE-CONTENT-CANARY" not in encoded.value


def test_phase10_phase8_lifecycle_snapshot_uses_only_terminal_and_cleanup_receipts() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    identity = object()
    pre = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.PRE_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        identity,
    )
    post = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.POST_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        identity,
        retained_candidate_cell_count=1,
    )
    lifecycle = phase8._Phase8LifecycleExecution(
        ((object(), "LIFECYCLE-CANDIDATE-CANARY"),),
        ("succeeded",),
        False,
        pre,
        post,
    )

    capture: Any = lifecycle._phase10_snapshot()

    assert type(capture) is module._Phase10OwnerCapture
    assert capture.snapshot.cleanup_state is module._Phase10CleanupState.VERIFIED
    assert capture.snapshot.release_state is module._Phase10ReleaseState.RELEASED
    owner = _InspectionIdentity()
    grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.INSPECT)
    inspected = module._inspect_phase10(owner, capture, grant)
    encoded = module._encode_phase10_view(inspected)
    assert type(encoded) is module._Phase10CanonicalEncoding
    assert b"LIFECYCLE-CANDIDATE-CANARY" not in encoded.value
    assert str(id(identity)).encode() not in encoded.value


def test_phase10_owner_snapshot_rejects_more_than_eight_distinct_stages_without_partial_view() -> None:
    module = _module()
    stages = ("admission", "detect", "augment", "validate", "finalize", "resolve", "classify", "transform", "verify")
    ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("LIMIT-CANARY", stages=stages))
    ledger.open()

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED


@pytest.mark.parametrize("closed", ["owner", "subject"])
def test_phase10_grant_expires_without_retaining_closed_identity(closed: str) -> None:
    module = _module()

    class Identity:
        pass

    owner = Identity()
    subject = Identity()
    owner_ref = weakref.ref(owner)
    subject_ref = weakref.ref(subject)
    grant = module._issue_phase10_inspection_grant(owner, subject, module._Phase10Operation.INSPECT)
    assert type(grant) is module._Phase10InspectionGrant
    nonce = grant._nonce
    try:
        if closed == "owner":
            del owner
        else:
            del subject
        gc.collect()

        assert (owner_ref() if closed == "owner" else subject_ref()) is None
        assert nonce not in module._GRANT_STATES
    finally:
        module._revoke_phase10_inspection_grant(grant)


def test_phase10_lifecycle_snapshot_rejects_throwing_cleanup_state_without_leaking() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")

    class ThrowingCleanup:
        @property
        def status(self) -> object:
            raise RuntimeError("THROWING-CLEANUP-PRIVACY-CANARY")

    execution = phase8._Phase8LifecycleExecution((), (), True, cast(Any, ThrowingCleanup()), None)
    try:
        rejected: Any = execution._phase10_snapshot()
    except Exception as error:  # pragma: no cover - red-phase privacy witness
        pytest.fail(f"owner snapshot leaked a malformed cleanup exception: {type(error).__name__}")

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED
    assert "THROWING-CLEANUP-PRIVACY-CANARY" not in repr(rejected)


def test_phase10_graph_snapshot_contains_throwing_nested_state_without_leaking() -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")

    class ThrowingAccounting:
        @property
        def tasks(self) -> object:
            raise RuntimeError("THROWING-GRAPH-PRIVACY-CANARY")

    execution = graph_runtime._AccountingGraphExecution(
        _accounting_plan("GRAPH-OWNER-CANARY"),
        cast(Any, ThrowingAccounting()),
        (),
    )
    try:
        rejected: Any = execution._phase10_snapshot()
    except Exception as error:  # pragma: no cover - red-phase privacy witness
        pytest.fail(f"graph snapshot leaked malformed nested state: {type(error).__name__}")

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED
    assert "THROWING-GRAPH-PRIVACY-CANARY" not in repr(rejected)


def test_phase10_partial_phase8_terminal_snapshot_has_a_coherent_capture_pair() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(1, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    ledger.succeed(phase8._Phase8Stage.validate_baselines())

    capture: Any = ledger._phase10_snapshot()

    assert type(capture) is module._Phase10OwnerCapture
    assert capture.capture_boundary is module._Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED
    assert capture.capture_lifecycle_state is module._Phase10LifecycleState.TERMINAL


def test_phase10_closed_phase8_lifecycle_preserves_known_terminal_reason() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    service = importlib.import_module("anonymizer.engine.execution.phase8_service")
    member = object()
    operation_plan = phase8._compile_group_operation_plan(0, 3)
    assert operation_plan is not None

    def fail_operation(_members: tuple[object, ...], _baselines: dict[object, str]) -> Any:
        return phase8._Phase8GroupOutcome(
            phase8._GroupFailed(phase8._Phase8Reason.BACKEND_FAILURE),
            None,
            0,
            phase8._Phase8OperationLedger(operation_plan),
        )

    execution = service._Phase8GroupedRewriteProtectionService().run_lifecycle(
        groups=((member,),),
        atomic_groups=((member,),),
        dependencies=(),
        phase7_released=((member, "KNOWN-REASON-CONTENT-CANARY"),),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=(fail_operation,),
    )

    capture: Any = execution._phase10_snapshot()

    assert type(capture) is module._Phase10OwnerCapture
    assert tuple(item.reason_category for item in capture.diagnostics) == (
        module._Phase10ReasonCategory.BACKEND_FAILED,
    )


def test_phase10_diagnostic_bytes_follow_closed_reason_order_not_task_order() -> None:
    module = _module()
    ledger_module = importlib.import_module("anonymizer.engine.execution.accounting_ledger")
    outcomes = importlib.import_module("anonymizer.engine.execution.accounting_outcomes")

    def encoded(codes: tuple[Any, Any]) -> bytes:
        ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("FIRST-CANARY", "SECOND-CANARY"))
        ledger.open()
        tasks = ledger.ready_tasks()
        assert len(tasks) == 2
        for task, code in zip(tasks, codes, strict=True):
            ledger._states[task] = outcomes._TaskFailed(task, ledger_module._causes(code))
        capture = ledger._phase10_snapshot()
        assert type(capture) is module._Phase10OwnerCapture
        owner = _InspectionIdentity()
        grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.DIAGNOSE)
        view = module._diagnose_phase10(owner, capture, grant)
        encoding = module._encode_phase10_view(view)
        assert type(encoding) is module._Phase10CanonicalEncoding
        return encoding.value

    verification = outcomes._CauseCode.VERIFICATION_FAILED
    backend = outcomes._CauseCode.KNOWN_FAILURE

    assert encoded((verification, backend)) == encoded((backend, verification))


def test_phase10_builder_budget_accepts_exact_limit_and_rejects_one_over() -> None:
    module = _module()
    budget = module._phase10_new_builder_budget()
    assert type(budget) is module._Phase10BuilderBudget

    for _ in range(module._MAX_STAGE_SUMMARIES + module._MAX_TERMINAL_SUMMARIES + module._MAX_DIAGNOSTICS):
        assert module._phase10_reserve_builder_row(budget)
    assert budget.used_bytes == module._MAX_BUILDER_WORKING_BYTES
    assert not module._phase10_reserve_builder_row(budget)
    assert budget.used_bytes == module._MAX_BUILDER_WORKING_BYTES


def test_phase10_owner_snapshot_reserves_builder_bytes_before_first_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "_BUILDER_BASE_BYTES", module._MAX_BUILDER_WORKING_BYTES)
    monkeypatch.setattr(module, "_BUILDER_ROW_BYTES", 1)
    ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("BUILDER-BUDGET-CANARY"))
    ledger.open()

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED


def test_phase10_grant_rejects_nonweakrefable_bindings_without_id_fallback() -> None:
    module = _module()
    owner = _InspectionIdentity()
    subject = _InspectionIdentity()

    bad_owner = module._issue_phase10_inspection_grant(object(), subject, module._Phase10Operation.INSPECT)
    bad_subject = module._issue_phase10_inspection_grant(owner, object(), module._Phase10Operation.INSPECT)

    for rejected in (bad_owner, bad_subject):
        assert type(rejected) is module._Phase10InspectionRejected
        assert rejected.code is module._Phase10RejectionCode.SUBJECT_INVALID


def test_phase10_supported_phase8_rejection_has_weak_identity_binding() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_admission")
    owner = _InspectionIdentity()
    subject = phase8._Phase8Rejected(phase8._Phase8AdmissionCode.INVALID_INPUT)
    subject_ref = weakref.ref(subject)

    grant = module._issue_phase10_inspection_grant(owner, subject, module._Phase10Operation.EXPLAIN)

    assert subject_ref() is subject
    assert type(grant) is module._Phase10InspectionGrant
    assert repr(subject).startswith("<private ")
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(subject)
    assert type(module._explain_phase10(owner, subject, grant)) is module._Phase10ExplainView


def test_phase10_phase8_private_sources_mask_repr_and_disable_pickle() -> None:
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(0, 3)
    assert plan is not None
    identity = object()
    receipt = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.PRE_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        identity,
    )
    values = (
        _AccountingLedger(_accounting_plan("PRIVATE-LEDGER-SERIALIZATION-CANARY")),
        phase8._Phase8OperationLedger(plan),
        phase8._Phase8LifecycleExecution(
            ((object(), "PRIVATE-SOURCE-SERIALIZATION-CANARY"),),
            ("succeeded",),
            False,
        ),
        receipt,
    )

    for value in values:
        representation = repr(value)
        assert representation.startswith("<private ")
        assert "PRIVATE-SOURCE-SERIALIZATION-CANARY" not in representation
        assert "0x" not in representation
        with pytest.raises(TypeError, match="not serializable"):
            pickle.dumps(value)


def test_phase10_snapshot_contains_throwing_lock_canaries() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")

    class ThrowingLock:
        def __enter__(self) -> object:
            raise RuntimeError("LOCK-PRIVACY-CANARY")

        def __exit__(self, *_args: object) -> None:
            return None

    accounting: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("LOCK-CANARY"))
    accounting._lock = cast(Any, ThrowingLock())
    operation_plan = phase8._compile_group_operation_plan(0, 3)
    assert operation_plan is not None
    operation = phase8._Phase8OperationLedger(operation_plan)
    operation._lock = cast(Any, ThrowingLock())

    for source in (accounting, operation):
        try:
            rejected: Any = source._phase10_snapshot()
        except Exception as error:  # pragma: no cover - red-phase privacy witness
            pytest.fail(f"snapshot leaked lock canary: {type(error).__name__}")
        assert type(rejected) is module._Phase10InspectionRejected
        assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED
        assert "LOCK-PRIVACY-CANARY" not in repr(rejected)


def test_phase10_phase8_operation_rejects_undispatched_terminal_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(1, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    evaluate_zero = phase8._Phase8Stage.evaluate(0)
    evaluate_one = phase8._Phase8Stage.evaluate(1)
    ledger._terminals[evaluate_zero] = phase8._StageFailed(evaluate_zero, phase8._Phase8Reason.ANALYSIS_STATE_MISSING)
    ledger._terminals[evaluate_one] = phase8._StageFailed(evaluate_one, phase8._Phase8Reason.BACKEND_FAILURE)

    capture: Any = ledger._phase10_snapshot()

    assert type(capture) is module._Phase10InspectionRejected
    assert capture.code is module._Phase10RejectionCode.REDACTION_FAILED


@pytest.mark.parametrize(
    ("operation_name", "operation"),
    (
        ("_explain_phase10", "EXPLAIN"),
        ("_inspect_phase10", "INSPECT"),
        ("_diagnose_phase10", "DIAGNOSE"),
    ),
)
def test_phase10_subject_validation_precedes_builder_budget(
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
    operation: str,
) -> None:
    module = _module()
    owner = _InspectionIdentity()
    subject = _InspectionIdentity()
    grant = module._issue_phase10_inspection_grant(owner, subject, getattr(module._Phase10Operation, operation))
    monkeypatch.setattr(module, "_BUILDER_BASE_BYTES", module._MAX_BUILDER_WORKING_BYTES)
    monkeypatch.setattr(module, "_BUILDER_ROW_BYTES", 1)

    result = getattr(module, operation_name)(owner, subject, grant)

    assert type(result) is module._Phase10InspectionRejected
    assert result.code is module._Phase10RejectionCode.SUBJECT_INVALID


def test_phase10_builder_budget_precedes_view_construction_for_valid_subject(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    owner = _InspectionIdentity()
    subject = _accounting_plan("EXPLAIN-BUDGET-ORDER-CANARY")
    grant = module._issue_phase10_inspection_grant(owner, subject, module._Phase10Operation.EXPLAIN)
    monkeypatch.setattr(module, "_BUILDER_BASE_BYTES", module._MAX_BUILDER_WORKING_BYTES)
    monkeypatch.setattr(module, "_BUILDER_ROW_BYTES", 1)

    def build_details(_subject: object) -> object:
        raise RuntimeError("VIEW-CONSTRUCTION-CANARY")

    monkeypatch.setattr(module, "_phase10_explain_details", build_details)

    result = module._explain_phase10(owner, subject, grant)

    assert type(result) is module._Phase10InspectionRejected
    assert result.code is module._Phase10RejectionCode.LIMIT_EXCEEDED


def test_phase10_graph_execution_private_input_rejects_pickle_canaries() -> None:
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    execution = graph_runtime._AccountingGraphExecution(
        cast(Any, "GRAPH-PRIVATE-INPUT-CANARY"),
        cast(Any, "ACCOUNTING-PRIVATE-INPUT-CANARY"),
        (),
    )

    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(execution)


def test_phase10_phase8_operation_snapshot_rejects_unknown_terminal_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(0, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    ledger._terminals[phase8._Phase8Stage.validate_baselines()] = cast(Any, object())

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_phase8_lifecycle_snapshot_rejects_success_reason_mismatch_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    lifecycle = phase8._Phase8LifecycleExecution(
        (),
        ("succeeded",),
        False,
        terminal_group_reasons=(phase8._Phase8Reason.BACKEND_FAILURE,),
    )

    rejected: Any = lifecycle._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_phase8_lifecycle_snapshot_rejects_non_boolean_embargo_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    lifecycle = phase8._Phase8LifecycleExecution((), ("blocked",), cast(Any, "embargoed"))

    rejected: Any = lifecycle._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_phase8_lifecycle_snapshot_rejects_release_without_verified_cleanup_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    lifecycle = phase8._Phase8LifecycleExecution((), ("succeeded",), False)

    rejected: Any = lifecycle._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_malformed_exact_grant_is_cause_free_denial() -> None:
    module = _module()
    malformed = object.__new__(module._Phase10InspectionGrant)

    assert not module._consume_phase10_inspection_grant(
        malformed,
        _InspectionIdentity(),
        _InspectionIdentity(),
        module._Phase10Operation.INSPECT,
    )


def test_phase10_malformed_authorized_subject_is_cause_free_redaction() -> None:
    module = _module()
    owner = _InspectionIdentity()
    malformed = object.__new__(_AccountingRejected)
    grant = module._issue_phase10_inspection_grant(owner, malformed, module._Phase10Operation.EXPLAIN)

    result = module._explain_phase10(owner, malformed, grant)

    assert type(result) is module._Phase10InspectionRejected
    assert result.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_malformed_view_encoding_is_cause_free_redaction() -> None:
    module = _module()
    malformed = object.__new__(module._Phase10InspectView)

    result = module._encode_phase10_view(malformed)

    assert type(result) is module._Phase10InspectionRejected
    assert result.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_cleanup_snapshot_rejects_unknown_receipt_subclass_mutation() -> None:
    module = _module()
    cleanup = importlib.import_module("anonymizer.engine.execution.phase8_cleanup")

    class PrivacyCanaryReceipt(cleanup._Phase8CleanupReceipt):
        pass

    original = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.PRE_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        object(),
    )
    subclass = PrivacyCanaryReceipt(*(getattr(original, item.name) for item in fields(original)))

    rejected: Any = subclass._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_lifecycle_snapshot_rejects_mismatched_cleanup_pair_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    pre = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.PRE_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        object(),
    )
    post = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.POST_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        object(),
    )
    lifecycle = phase8._Phase8LifecycleExecution((), ("failed",), True, pre, post)

    rejected: Any = lifecycle._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_lifecycle_snapshot_rejects_wrong_cleanup_slot_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    post = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.POST_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.UNCONFIRMED,
        object(),
    )
    lifecycle = phase8._Phase8LifecycleExecution((), ("failed",), True, post, None)

    rejected: Any = lifecycle._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def _phase8_lifecycle_with_reasons(reasons: tuple[Any, ...]) -> Any:
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    service = importlib.import_module("anonymizer.engine.execution.phase8_service")
    members = tuple(object() for _ in reasons)
    operation_plan = phase8._compile_group_operation_plan(0, 3)
    assert operation_plan is not None
    operations = tuple(
        (
            lambda reason: (
                lambda _members, _baselines: phase8._Phase8GroupOutcome(
                    phase8._GroupFailed(reason),
                    None,
                    0,
                    phase8._Phase8OperationLedger(operation_plan),
                )
            )
        )(reason)
        for reason in reasons
    )
    return service._Phase8GroupedRewriteProtectionService().run_lifecycle(
        groups=tuple((member,) for member in members),
        atomic_groups=tuple((member,) for member in members),
        dependencies=(),
        phase7_released=tuple((member, "REASON-LIMIT-CANARY") for member in members),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=operations,
    )


def test_phase10_lifecycle_diagnostic_accepts_exact_four_owning_reasons() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    execution = _phase8_lifecycle_with_reasons(
        (
            phase8._Phase8Reason.ANALYSIS_RECONCILIATION,
            phase8._Phase8Reason.CANDIDATE_RECONCILIATION,
            phase8._Phase8Reason.EVALUATION_RECONCILIATION,
            phase8._Phase8Reason.REPAIR_RECONCILIATION,
        )
    )

    capture: Any = execution._phase10_snapshot()

    assert type(capture) is module._Phase10OwnerCapture


def test_phase10_lifecycle_diagnostic_rejects_fifth_owning_reason() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    execution = _phase8_lifecycle_with_reasons(
        (
            phase8._Phase8Reason.ANALYSIS_RECONCILIATION,
            phase8._Phase8Reason.CANDIDATE_RECONCILIATION,
            phase8._Phase8Reason.EVALUATION_RECONCILIATION,
            phase8._Phase8Reason.REPAIR_RECONCILIATION,
            phase8._Phase8Reason.REWRITE_RECONCILIATION,
        )
    )

    rejected: Any = execution._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED


def test_phase10_post_dispatch_cancellation_reports_only_acknowledged_stop() -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    plan = _accounting_plan("POST-DISPATCH-CANCEL-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    (task,) = ledger.ready_tasks()
    dispatch = ledger.dispatch(task)
    ledger.request_cancellation()
    ledger.acknowledge_stop(dispatch)

    live_capture: Any = ledger._phase10_snapshot()
    result = ledger.finish()
    closed_capture: Any = graph_runtime._AccountingGraphExecution(plan, result, ())._phase10_snapshot()

    for capture in (live_capture, closed_capture):
        assert type(capture) is module._Phase10OwnerCapture
        assert tuple(item.reason_category for item in capture.diagnostics) == (
            module._Phase10ReasonCategory.STOP_ACKNOWLEDGED,
        )


def test_phase10_accounting_snapshot_rejects_unknown_task_state_mutation() -> None:
    module = _module()
    ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("UNKNOWN-STATE-CANARY"))
    ledger.open()
    (task,) = ledger._plan.tasks
    ledger._states[task] = cast(Any, object())

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_graph_snapshot_rejects_mismatched_task_outcome_mutation() -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    plan = _accounting_plan("UNKNOWN-OUTCOME-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    result = ledger.finish()
    malformed = replace(result, tasks=(cast(Any, object()),))

    rejected: Any = graph_runtime._AccountingGraphExecution(plan, malformed, ())._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_lifecycle_snapshot_rejects_cancelled_backend_reason_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    service = importlib.import_module("anonymizer.engine.execution.phase8_service")
    member = object()

    def cancel_with_backend_reason(_members: tuple[object, ...], _baselines: dict[object, str]) -> Any:
        plan = phase8._compile_group_operation_plan(0, 3)
        assert plan is not None
        return phase8._Phase8GroupOutcome(
            phase8._GroupCancelled(phase8._Phase8Reason.BACKEND_FAILURE),
            None,
            0,
            phase8._Phase8OperationLedger(plan),
        )

    execution = service._Phase8GroupedRewriteProtectionService().run_lifecycle(
        groups=((member,),),
        atomic_groups=((member,),),
        dependencies=(),
        phase7_released=((member, "CANCEL-REASON-CANARY"),),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=(cancel_with_backend_reason,),
    )

    rejected: Any = execution._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_operation_snapshot_rejects_missing_attempt_correlation_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(0, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    stage = phase8._Phase8Stage.validate_baselines()
    ledger._dispatched.add(stage)
    ledger._terminals[stage] = phase8._StageSucceeded(stage)

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_encoder_rejects_mutated_nested_stage_summary_without_reading_canary() -> None:
    module = _module()
    stage = module._Phase10StageSummary(
        module._Phase10Stage.REWRITE,
        module._Phase10LifecycleState.TERMINAL,
        module._Phase10CountBucket.ONE,
    )

    class ForeignStage:
        value = "NESTED-STAGE-CANARY"

    object.__setattr__(stage, "stage", ForeignStage())
    snapshot = module._Phase10Snapshot.__new__(module._Phase10Snapshot)
    object.__setattr__(snapshot, "stage_summaries", (stage,))
    object.__setattr__(snapshot, "terminal_summaries", ())
    object.__setattr__(snapshot, "reconciliation_state", module._Phase10ReconciliationState.RECONCILED)
    object.__setattr__(snapshot, "cleanup_state", module._Phase10CleanupState.NOT_ENTERED)
    object.__setattr__(snapshot, "release_state", module._Phase10ReleaseState.WITHHELD)
    view = module._Phase10InspectView.__new__(module._Phase10InspectView)
    object.__setattr__(view, "provenance", _provenance(module, module._Phase10ViewKind.INSPECT))
    object.__setattr__(view, "snapshot", snapshot)

    rejected = module._encode_phase10_view(view)

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED
    assert "NESTED-STAGE-CANARY" not in repr(rejected)


def test_phase10_grant_issue_and_weak_expiry_contain_throwing_lock_canary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    owner = _InspectionIdentity()
    subject = _InspectionIdentity()
    grant = module._issue_phase10_inspection_grant(owner, subject, module._Phase10Operation.INSPECT)

    class ThrowingLock:
        def __enter__(self) -> object:
            raise RuntimeError("GRANT-LOCK-CANARY")

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(module, "_GRANT_LOCK", ThrowingLock())
    rejected = module._issue_phase10_inspection_grant(subject, owner, module._Phase10Operation.INSPECT)
    del owner
    gc.collect()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED
    assert "GRANT-LOCK-CANARY" not in capsys.readouterr().err
    del grant


def test_phase10_encoder_contains_throwing_json_encoder_canary(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    view = module._Phase10DiagnoseView(
        _provenance(module, module._Phase10ViewKind.DIAGNOSE),
        (_diagnostic(module),),
    )

    def throw_encoder(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("ENCODER-CANARY")

    monkeypatch.setattr(module.json, "dumps", throw_encoder)
    rejected = module._encode_phase10_view(view)

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.ENCODING_FAILED
    assert "ENCODER-CANARY" not in repr(rejected)


def test_phase10_cleanup_snapshot_rejects_foreign_proof_mutation() -> None:
    module = _module()
    cleanup = importlib.import_module("anonymizer.engine.execution.phase8_cleanup")
    identity = object()
    receipt = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.PRE_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        identity,
    )
    object.__setattr__(receipt, "_proof", cast(Any, object()))

    rejected: Any = receipt._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert not cleanup._is_phase8_cleanup_receipt(
        receipt,
        identity=identity,
        phase=_Phase8CleanupPhase.PRE_REDUCTION,
        component=_Phase8CleanupComponent.RUNTIME,
    )


def test_phase10_phase8_attempt_receipt_masks_repr_and_rejects_pickle() -> None:
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    receipt = phase8._Phase8AttemptReceipt(
        object(),
        phase8._Phase8Stage.validate_baselines(),
        phase8._Phase8TerminalKind.FAILED,
    )

    assert repr(receipt) == "<private phase 8 attempt receipt>"
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(receipt)


def test_phase10_phase8_operation_rejects_failed_cancellation_reason_mutation() -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(0, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    stage = phase8._Phase8Stage.validate_baselines()
    assert ledger.dispatch(stage)
    ledger._terminals[stage] = phase8._StageFailed(stage, phase8._Phase8Reason.CANCELLATION)
    ledger._attempts[stage] = phase8._Phase8AttemptReceipt(
        plan.group_id,
        stage,
        phase8._Phase8TerminalKind.FAILED,
    )

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_live_accounting_rejects_released_state_before_close_mutation() -> None:
    module = _module()
    ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("OPEN-RELEASE-CANARY"))
    ledger.open()
    ledger._phase10_release_state = "released"

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_graph_snapshot_rejects_empty_groups_and_mismatched_invocation_mutations() -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    outcomes = importlib.import_module("anonymizer.engine.execution.accounting_outcomes")
    plan = _accounting_plan("GROUP-CORRELATION-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    result = ledger.finish()
    empty_groups = replace(result, groups=())
    mismatched_invocation = replace(result, invocation=outcomes._InvocationCompleted(()))

    for malformed in (empty_groups, mismatched_invocation):
        rejected: Any = graph_runtime._AccountingGraphExecution(plan, malformed, ())._phase10_snapshot()
        assert type(rejected) is module._Phase10InspectionRejected
        assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_group_release_failure_emits_publication_diagnostic() -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    plan = _accounting_plan("PUBLICATION-FAILURE-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    (task,) = ledger.ready_tasks()
    ledger.accept_success(ledger.dispatch(task), "protected")
    result = ledger.finish(group_release_predicate=lambda _outputs: False)

    for capture in (
        ledger._phase10_snapshot(),
        graph_runtime._AccountingGraphExecution(plan, result, ())._phase10_snapshot(),
    ):
        assert type(capture) is module._Phase10OwnerCapture
        capture = cast(Any, capture)
        assert module._Phase10ReasonCategory.PUBLICATION_FAILED in {
            item.reason_category for item in capture.diagnostics
        }


@pytest.mark.parametrize(
    ("mark", "category"),
    (
        ("mark_cleanup_failed", "CLEANUP_FAILED"),
        ("mark_cleanup_unconfirmed", "CLEANUP_UNCONFIRMED"),
    ),
)
def test_phase10_accounting_cleanup_failure_emits_diagnostic(mark: str, category: str) -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    plan = _accounting_plan(f"{category}-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    getattr(ledger, mark)()
    live_capture: Any = ledger._phase10_snapshot()
    result = ledger.finish()
    closed_capture: Any = graph_runtime._AccountingGraphExecution(plan, result, ())._phase10_snapshot()

    for capture in (live_capture, closed_capture):
        assert type(capture) is module._Phase10OwnerCapture
        assert getattr(module._Phase10ReasonCategory, category) in {
            item.reason_category for item in capture.diagnostics
        }
    assert live_capture.capture_boundary is module._Phase10CaptureBoundary.PRE_REDUCTION_CLEANUP_TERMINAL
    assert live_capture.capture_lifecycle_state is module._Phase10LifecycleState.CLEANUP_TERMINAL
    assert all(
        item.boundary is module._Phase10CaptureBoundary.PRE_REDUCTION_CLEANUP_TERMINAL
        for item in live_capture.diagnostics
    )


def test_phase10_untrusted_post_dispatch_cancellation_reports_only_execution_lost() -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    plan = _accounting_plan("UNTRUSTED-CANCEL-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    (task,) = ledger.ready_tasks()
    ledger.dispatch(task)
    ledger.request_cancellation()
    result = ledger.finish()

    for capture in (
        ledger._phase10_snapshot(),
        graph_runtime._AccountingGraphExecution(plan, result, ())._phase10_snapshot(),
    ):
        assert type(capture) is module._Phase10OwnerCapture
        capture = cast(Any, capture)
        assert tuple(item.reason_category for item in capture.diagnostics) == (
            module._Phase10ReasonCategory.EXECUTION_LOST,
        )


def test_phase10_graph_snapshot_rejects_failed_task_with_cancellation_mutation() -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    outcomes = importlib.import_module("anonymizer.engine.execution.accounting_outcomes")
    ledger_module = importlib.import_module("anonymizer.engine.execution.accounting_ledger")
    plan = _accounting_plan("TASK-SEMANTIC-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    result = ledger.finish()
    malformed = replace(
        result,
        tasks=(outcomes._TaskFailed(plan.tasks[0], ledger_module._causes(outcomes._CauseCode.CANCELLATION)),),
    )

    rejected: Any = graph_runtime._AccountingGraphExecution(plan, malformed, ())._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_phase8_preflight_failure_preserves_closed_reason() -> None:
    module = _module()
    service = importlib.import_module("anonymizer.engine.execution.phase8_service")
    execution = service._Phase8GroupedRewriteProtectionService().run_lifecycle(
        groups=(),
        atomic_groups=(),
        dependencies=(),
        phase7_released=(),
        phase7_cleanup_verified=True,
        phase7_global_embargo=False,
        operations=(),
    )

    capture: Any = execution._phase10_snapshot()

    assert type(capture) is module._Phase10OwnerCapture
    assert tuple(item.reason_category for item in capture.diagnostics) == (
        module._Phase10ReasonCategory.ADMISSION_REJECTED,
    )


@pytest.mark.parametrize("reason_count", (4, 5))
def test_phase10_accounting_diagnostic_enforces_distinct_owning_reason_limit(reason_count: int) -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    outcomes = importlib.import_module("anonymizer.engine.execution.accounting_outcomes")
    ledger_module = importlib.import_module("anonymizer.engine.execution.accounting_ledger")
    plan = _accounting_plan(*(f"REASON-{index}-CANARY" for index in range(reason_count)))
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    codes = (
        outcomes._CauseCode.MISSING,
        outcomes._CauseCode.DUPLICATE,
        outcomes._CauseCode.UNKNOWN,
        outcomes._CauseCode.FOREIGN,
        outcomes._CauseCode.STALE,
    )
    for task, code in zip(plan.tasks, codes, strict=False):
        ledger._states[task] = outcomes._TaskInconsistent(task, ledger_module._causes(code))
    live_capture: Any = ledger._phase10_snapshot()
    result = ledger.finish()
    closed_capture: Any = graph_runtime._AccountingGraphExecution(plan, result, ())._phase10_snapshot()

    expected = module._Phase10OwnerCapture if reason_count == 4 else module._Phase10InspectionRejected
    assert type(live_capture) is expected
    assert type(closed_capture) is expected
    if reason_count == 5:
        assert live_capture.code is module._Phase10RejectionCode.LIMIT_EXCEEDED
        assert closed_capture.code is module._Phase10RejectionCode.LIMIT_EXCEEDED


def test_phase10_encoder_rejects_mutated_nested_explain_rows() -> None:
    module = _module()
    provenance = module._Phase10Provenance(
        module._Phase10InspectionSchemaVersion.V1,
        module._Phase10ContractVersion.V1,
        module._Phase10ViewKind.EXPLAIN,
        module._Phase10SubjectKind.ADMITTED_PLAN,
        module._Phase10SemanticProfile.GROUPED_REWRITE_V1,
        module._Phase10ImplementationProfile.PANDAS_RUNTIME_V1,
        module._Phase10CaptureBoundary.ADMISSION_TERMINAL,
        module._Phase10LifecycleState.TERMINAL,
    )
    limit = module._Phase10DeclaredLimit(module._Phase10LimitName.SUBJECTS_PER_CALL, 1)
    aggregate = module._Phase10Aggregate(
        module._Phase10AggregateDimension.DATUMS,
        module._Phase10CountBucket.ONE,
    )
    view = module._Phase10ExplainView(
        provenance,
        module._Phase10Route.NDD,
        (),
        (limit,),
        (aggregate,),
        None,
    )
    object.__setattr__(limit, "value", cast(Any, "DECLARED-LIMIT-CANARY"))

    rejected = module._encode_phase10_view(view)

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED


def test_phase10_graph_correlation_never_invokes_candidate_equality_canary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    outcomes = importlib.import_module("anonymizer.engine.execution.accounting_outcomes")

    class CandidateCanary:
        calls = 0

        def __eq__(self, _other: object) -> bool:
            type(self).calls += 1
            print("CANDIDATE-EQUALITY-CANARY")
            raise RuntimeError("CANDIDATE-EQUALITY-CANARY")

    plan = _accounting_plan("CANDIDATE-CORRELATION-CANARY")
    ledger: _AccountingLedger[CandidateCanary] = _AccountingLedger(plan)
    ledger.open()
    (task,) = ledger.ready_tasks()
    ledger.accept_success(ledger.dispatch(task), CandidateCanary())
    result = ledger.finish()
    (group,) = result.groups
    assert type(group) is outcomes._GroupReleased
    cloned_group = outcomes._GroupReleased(
        group.group,
        tuple((datum_id, CandidateCanary()) for datum_id, _candidate in group.outputs),
    )
    malformed = replace(result, invocation=outcomes._InvocationCompleted((cloned_group,)))

    rejected: Any = graph_runtime._AccountingGraphExecution(plan, malformed, ())._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.REDACTION_FAILED
    assert CandidateCanary.calls == 0
    assert "CANDIDATE-EQUALITY-CANARY" not in capsys.readouterr().out


class _ComparisonCanary:
    calls = 0

    def __eq__(self, _other: object) -> bool:
        type(self).calls += 1
        print("MALFORMED-COMPARISON-CANARY")
        raise RuntimeError("MALFORMED-COMPARISON-CANARY")

    def __ne__(self, _other: object) -> bool:
        type(self).calls += 1
        print("MALFORMED-COMPARISON-CANARY")
        raise RuntimeError("MALFORMED-COMPARISON-CANARY")

    def __hash__(self) -> int:
        type(self).calls += 1
        print("MALFORMED-COMPARISON-CANARY")
        return 8675309

    def __bool__(self) -> bool:
        type(self).calls += 1
        print("MALFORMED-COMPARISON-CANARY")
        raise RuntimeError("MALFORMED-COMPARISON-CANARY")


def _assert_no_comparison_canary(capsys: pytest.CaptureFixture[str]) -> None:
    captured = capsys.readouterr()
    assert _ComparisonCanary.calls == 0
    assert "MALFORMED-COMPARISON-CANARY" not in captured.out
    assert "MALFORMED-COMPARISON-CANARY" not in captured.err


def test_phase10_accounting_task_correlation_never_invokes_malformed_equality_canary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    outcomes = importlib.import_module("anonymizer.engine.execution.accounting_outcomes")
    ledger_module = importlib.import_module("anonymizer.engine.execution.accounting_ledger")
    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    plan = _accounting_plan("TASK-COMPARISON-CANARY")
    ledger: _AccountingLedger[str] = _AccountingLedger(plan)
    ledger.open()
    result = ledger.finish()
    malformed = replace(
        result,
        tasks=(
            outcomes._TaskFailed(
                cast(Any, _ComparisonCanary()),
                ledger_module._causes(outcomes._CauseCode.KNOWN_FAILURE),
            ),
        ),
    )
    _ComparisonCanary.calls = 0
    capsys.readouterr()

    rejected: Any = graph_runtime._AccountingGraphExecution(plan, malformed, ())._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    _assert_no_comparison_canary(capsys)


def test_phase10_accounting_state_keys_never_invoke_malformed_hash_or_equality_canary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("STATE-KEY-CANARY"))
    ledger.open()
    malformed_key = _ComparisonCanary()
    ledger._states = cast(Any, {malformed_key: object()})
    _ComparisonCanary.calls = 0
    capsys.readouterr()

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    _assert_no_comparison_canary(capsys)


@pytest.mark.parametrize("field", ("terminal", "attempt"))
def test_phase10_phase8_correlation_never_invokes_malformed_equality_canary(
    field: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(0, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    stage = plan.stages[0]
    assert ledger.dispatch(stage)
    assert ledger.fail(stage, phase8._Phase8Reason.BACKEND_FAILURE)
    target = ledger._terminals[stage] if field == "terminal" else ledger._attempts[stage]
    object.__setattr__(target, "stage", cast(Any, _ComparisonCanary()))
    _ComparisonCanary.calls = 0
    capsys.readouterr()

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    _assert_no_comparison_canary(capsys)


def test_phase10_cleanup_proof_never_invokes_malformed_equality_canary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    identity = object()
    receipt = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.PRE_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        identity,
    )
    proof: Any = receipt._proof
    object.__setattr__(proof, "snapshot", (*proof.snapshot[:-1], _ComparisonCanary()))
    _ComparisonCanary.calls = 0
    capsys.readouterr()

    rejected: Any = receipt._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    _assert_no_comparison_canary(capsys)


@pytest.mark.parametrize("field", ("_phase10_release_state", "_opened", "_cleanup_failed"))
def test_phase10_accounting_scalars_never_invoke_malformed_comparison_canary(
    field: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    ledger: _AccountingLedger[str] = _AccountingLedger(_accounting_plan("SCALAR-CANARY"))
    ledger.open()
    setattr(ledger, field, _ComparisonCanary())
    _ComparisonCanary.calls = 0
    capsys.readouterr()

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    _assert_no_comparison_canary(capsys)


def test_phase10_phase8_retired_flag_never_invokes_malformed_truthiness_canary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    plan = phase8._compile_group_operation_plan(0, 3)
    assert plan is not None
    ledger = phase8._Phase8OperationLedger(plan)
    ledger._retired = cast(Any, _ComparisonCanary())
    _ComparisonCanary.calls = 0
    capsys.readouterr()

    rejected: Any = ledger._phase10_snapshot()

    assert type(rejected) is module._Phase10InspectionRejected
    _assert_no_comparison_canary(capsys)


@pytest.mark.parametrize(
    "variant",
    (
        "complete",
        "empty",
        "missing",
        "extra",
        "duplicate",
        "reordered",
        "wrong_value",
        "boolean",
        "unknown",
        "malformed",
    ),
)
def test_phase10_declared_table_payload_requires_exact_authoritative_sequence(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    module = _module()
    view = _encoder_views(module)[0]
    payload = module._view_payload(view)
    assert type(payload) is dict
    rows = payload["declared_limits"]
    if variant == "empty":
        rows.clear()
    elif variant == "missing":
        rows.pop()
    elif variant == "extra":
        rows.append(dict(rows[-1]))
    elif variant == "duplicate":
        rows[1] = dict(rows[0])
    elif variant == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif variant == "wrong_value":
        rows[0]["value"] = 2
    elif variant == "boolean":
        rows[0]["value"] = True
    elif variant == "unknown":
        rows[0]["name"] = "unknown"
    elif variant == "malformed":
        rows[0]["extra"] = None
    monkeypatch.setattr(module, "_view_payload", lambda _view: payload)

    result = module._encode_phase10_view(view)

    if variant == "complete":
        assert type(result) is module._Phase10CanonicalEncoding
    else:
        assert type(result) is module._Phase10InspectionRejected
        assert result.code is module._Phase10RejectionCode.REDACTION_FAILED
        assert not hasattr(result, "value")
        assert not hasattr(result, "__cause__")


@pytest.mark.parametrize("variant", ("missing", "reordered", "wrong_value"))
def test_phase10_declared_table_view_requires_exact_authoritative_sequence(variant: str) -> None:
    module = _module()
    view = _encoder_views(module)[0]
    rows = view.declared_limits
    if variant == "missing":
        rows = rows[:-1]
    elif variant == "reordered":
        rows = rows[::-1]
    else:
        rows = (replace(rows[0], value=2), *rows[1:])
    result = module._encode_phase10_view(replace(view, declared_limits=rows))
    assert type(result) is module._Phase10InspectionRejected
    assert result.code is module._Phase10RejectionCode.REDACTION_FAILED
    assert not hasattr(result, "value")


def test_phase10_declared_table_malformed_and_oversized_measures_first(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    view = _encoder_views(module)[0]
    payload = module._view_payload(view)
    payload["declared_limits"] = []
    payload["oversized"] = "x" * 97
    monkeypatch.setattr(module, "_view_payload", lambda _view: payload)
    result = module._encode_phase10_view(view)
    assert type(result) is module._Phase10InspectionRejected
    assert result.code is module._Phase10RejectionCode.LIMIT_EXCEEDED
    assert not hasattr(result, "value")


@pytest.mark.parametrize(
    "variant", ("list_subclass", "dict_subclass", "str_subclass", "int_subclass", "tuple", "boolean")
)
def test_phase10_declared_table_validator_requires_exact_builtins(variant: str) -> None:
    module = _module()

    class ListSubclass(list[Any]):
        pass

    class DictSubclass(dict[str, Any]):
        pass

    class StrSubclass(str):
        pass

    class IntSubclass(int):
        pass

    rows = module._view_payload(_encoder_views(module)[0])["declared_limits"]
    candidate: Any = rows
    if variant == "list_subclass":
        candidate = ListSubclass(rows)
    elif variant == "tuple":
        candidate = tuple(rows)
    elif variant == "dict_subclass":
        rows[0] = DictSubclass(rows[0])
    elif variant == "str_subclass":
        rows[0]["name"] = StrSubclass(rows[0]["name"])
    elif variant == "int_subclass":
        rows[0]["value"] = IntSubclass(rows[0]["value"])
    else:
        rows[0]["value"] = True
    assert module._valid_declared_limits_payload(candidate) is False


def _phase10_limit_test_owner(kind: str) -> Any:
    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    if kind in {"accounting", "graph"}:
        plan = _accounting_plan("OWNER-LIMIT-CANARY", stages=("detect", "protect"))
        ledger: _AccountingLedger[str] = _AccountingLedger(plan)
        ledger.open()
        for task in plan.tasks:
            ledger.mark_task_failed(task)
        if kind == "accounting":
            return ledger
        graph = importlib.import_module("anonymizer.engine.execution.graph_runtime")
        return graph._AccountingGraphExecution(plan, ledger.finish(), ())
    if kind == "operation":
        plan = phase8._compile_group_operation_plan(2, 3)
        assert plan is not None
        operations = phase8._Phase8OperationLedger(plan)
        assert operations.succeed(phase8._Phase8Stage.validate_baselines())
        assert operations.fail(phase8._Phase8Stage.analyze(), phase8._Phase8Reason.BACKEND_FAILURE)
        return operations
    return _phase8_lifecycle_with_reasons(
        (phase8._Phase8Reason.BACKEND_FAILURE, phase8._Phase8Reason.ANALYSIS_RECONCILIATION)
    )


@pytest.mark.parametrize(
    "kind,expected",
    (("accounting", (2, 2, 2)), ("graph", (3, 3, 2)), ("operation", (5, 5, 4)), ("lifecycle", (2, 2, 2))),
)
@pytest.mark.parametrize("category", ("stage", "terminal", "diagnostic"))
def test_phase10_owner_capture_uses_authoritative_row_limits(
    monkeypatch: pytest.MonkeyPatch, kind: str, expected: tuple[int, int, int], category: str
) -> None:
    module = _module()
    owner = _phase10_limit_test_owner(kind)
    baseline = owner._phase10_snapshot()
    assert type(baseline) is module._Phase10OwnerCapture
    assert (
        len(baseline.snapshot.stage_summaries),
        len(baseline.snapshot.terminal_summaries),
        len(baseline.diagnostics),
    ) == expected
    position = ("stage", "terminal", "diagnostic").index(category)
    name = (
        module._Phase10LimitName.MAX_STAGE_SUMMARIES,
        module._Phase10LimitName.MAX_TERMINAL_SUMMARY_ROWS,
        module._Phase10LimitName.MAX_DIAGNOSTIC_ENTRIES,
    )[position]
    original = module._PHASE10_LIMITS[name]
    monkeypatch.setitem(module._PHASE10_LIMITS, name, expected[position])
    exact = owner._phase10_snapshot()
    assert type(exact) is module._Phase10OwnerCapture
    assert exact.snapshot == baseline.snapshot
    assert exact.diagnostics == baseline.diagnostics
    monkeypatch.setitem(module._PHASE10_LIMITS, name, expected[position] - 1)
    rejected = owner._phase10_snapshot()
    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED
    assert not hasattr(rejected, "snapshot")
    assert not hasattr(rejected, "__cause__")
    monkeypatch.setitem(module._PHASE10_LIMITS, name, original)
    restored = owner._phase10_snapshot()
    assert restored.snapshot == baseline.snapshot
    assert restored.diagnostics == baseline.diagnostics


@pytest.mark.parametrize("kind", ("accounting", "graph", "lifecycle"))
@pytest.mark.parametrize("distinct", (True, False))
def test_phase10_owner_capture_uses_authoritative_reason_limit(
    monkeypatch: pytest.MonkeyPatch, kind: str, distinct: bool
) -> None:
    module = _module()
    if kind == "lifecycle":
        phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
        first = phase8._Phase8Reason.ANALYSIS_RECONCILIATION
        second = phase8._Phase8Reason.CANDIDATE_RECONCILIATION if distinct else first
        owner = _phase8_lifecycle_with_reasons((first, second))
    else:
        outcomes = importlib.import_module("anonymizer.engine.execution.accounting_outcomes")
        plan = _accounting_plan("FIRST-CANARY", "SECOND-CANARY")
        ledger: _AccountingLedger[str] = _AccountingLedger(plan)
        ledger.open()
        ledger.mark_task_inconsistent(plan.tasks[0], outcomes._CauseCode.MISSING)
        ledger.mark_task_inconsistent(
            plan.tasks[1], outcomes._CauseCode.DUPLICATE if distinct else outcomes._CauseCode.MISSING
        )
        owner = ledger
        if kind == "graph":
            graph = importlib.import_module("anonymizer.engine.execution.graph_runtime")
            owner = graph._AccountingGraphExecution(plan, ledger.finish(), ())
    baseline: Any = owner._phase10_snapshot()
    assert type(baseline) is module._Phase10OwnerCapture
    assert len(baseline.diagnostics) == 1
    assert baseline.diagnostics[0].reason_category is module._Phase10ReasonCategory.EVIDENCE_INCONSISTENT
    name = module._Phase10LimitName.MAX_REASON_CODES_PER_DIAGNOSTIC_ENTRY
    monkeypatch.setitem(module._PHASE10_LIMITS, name, 2 if distinct else 1)
    exact: Any = owner._phase10_snapshot()
    assert type(exact) is module._Phase10OwnerCapture
    assert exact.diagnostics == baseline.diagnostics
    monkeypatch.setitem(module._PHASE10_LIMITS, name, 1 if distinct else 0)
    rejected: Any = owner._phase10_snapshot()
    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED
    assert not hasattr(rejected, "snapshot")
    assert not hasattr(rejected, "__cause__")
