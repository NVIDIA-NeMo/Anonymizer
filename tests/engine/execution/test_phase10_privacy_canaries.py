# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import importlib
import json
import logging
import pickle
import weakref
from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, cast

import pytest

from anonymizer.engine.execution.accounting_admission import _compile_accounting_plan
from anonymizer.engine.execution.accounting_ledger import _AccountingLedger
from anonymizer.engine.execution.accounting_plan import _AccountingLimits, _AccountingPlan
from anonymizer.engine.execution.graph import _DatumId, _TextDatum, _trivial_graph
from anonymizer.engine.execution.phase8_cleanup import (
    _issue_phase8_cleanup_receipt,
    _Phase8CleanupComponent,
    _Phase8CleanupPhase,
    _Phase8CleanupStatus,
)


class _Identity:
    pass


class _AccessCanary:
    touched = False

    def __getattribute__(self, name: str) -> object:
        if name != "touched":
            type(self).touched = True
            raise AssertionError(f"privacy canary read through {name}")
        return object.__getattribute__(self, name)

    def __repr__(self) -> str:
        type(self).touched = True
        raise AssertionError("privacy canary repr")

    def __str__(self) -> str:
        type(self).touched = True
        raise AssertionError("privacy canary str")

    def __hash__(self) -> int:
        type(self).touched = True
        raise AssertionError("privacy canary hash")

    def __eq__(self, _other: object) -> bool:
        type(self).touched = True
        raise AssertionError("privacy canary equality")

    def __bool__(self) -> bool:
        type(self).touched = True
        raise AssertionError("privacy canary truthiness")

    def __reduce__(self) -> str | tuple[Any, ...]:
        type(self).touched = True
        raise AssertionError("privacy canary pickle")


def _module() -> Any:
    return importlib.import_module("anonymizer.engine.execution.phase10_inspection")


def _plan(text: str) -> _AccountingPlan:
    graph = _trivial_graph((_TextDatum(_DatumId("private-identity-canary"), text),))
    result = _compile_accounting_plan(
        graph,
        limits=_AccountingLimits(max_datums=1, max_datum_bytes=8_192, max_graph_bytes=16_384, max_stages=1),
    )
    assert isinstance(result, _AccountingPlan)
    return result


def _encoded_explain(text: str) -> bytes:
    module = _module()
    owner = _Identity()
    plan = _plan(text)
    grant = module._issue_phase10_inspection_grant(owner, plan, module._Phase10Operation.EXPLAIN)
    view = module._explain_phase10(owner, plan, grant)
    encoded = module._encode_phase10_view(view)
    assert type(encoded) is module._Phase10CanonicalEncoding
    return encoded.value


def _failed_accounting_capture(text: str) -> object:
    ledger: _AccountingLedger[str] = _AccountingLedger(_plan(text))
    ledger.open()
    (task,) = ledger.ready_tasks()
    dispatch = ledger.dispatch(task)
    assert dispatch is not None
    assert ledger.accept_failure(dispatch)
    return ledger._phase10_snapshot()


def _operation_views() -> tuple[tuple[str, object], ...]:
    module = _module()
    owner = _Identity()
    plan = _plan("OPERATION-VIEW-CONTENT-CANARY")
    explain_grant = module._issue_phase10_inspection_grant(owner, plan, module._Phase10Operation.EXPLAIN)
    explain = module._explain_phase10(owner, plan, explain_grant)

    capture = _failed_accounting_capture("OPERATION-CAPTURE-CONTENT-CANARY")
    inspect_grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.INSPECT)
    inspect_view = module._inspect_phase10(owner, capture, inspect_grant)
    diagnose_grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.DIAGNOSE)
    diagnose = module._diagnose_phase10(owner, capture, diagnose_grant)
    return (("explain", explain), ("inspect", inspect_view), ("diagnose", diagnose))


def _real_owner_registry(*, with_diagnostics: bool = False) -> tuple[tuple[str, object], ...]:
    accounting_ledger: _AccountingLedger[str] = _AccountingLedger(_plan("OWNER-LEDGER-CONTENT-CANARY"))
    accounting_ledger.open()
    (accounting_task,) = accounting_ledger.ready_tasks()
    accounting_dispatch = accounting_ledger.dispatch(accounting_task)
    assert accounting_dispatch is not None
    assert accounting_ledger.accept_failure(accounting_dispatch)

    graph_runtime = importlib.import_module("anonymizer.engine.execution.graph_runtime")
    graph_plan = _plan("OWNER-GRAPH-CONTENT-CANARY")
    graph_ledger: _AccountingLedger[str] = _AccountingLedger(graph_plan)
    graph_ledger.open()
    (graph_task,) = graph_ledger.ready_tasks()
    if with_diagnostics:
        graph_dispatch = graph_ledger.dispatch(graph_task)
        assert graph_dispatch is not None
        assert graph_ledger.accept_failure(graph_dispatch)
    else:
        graph_ledger.mark_task_succeeded(graph_task, "OWNER-GRAPH-CANDIDATE-CANARY")
    graph_execution = graph_runtime._AccountingGraphExecution(graph_plan, graph_ledger.finish(), ())

    phase8 = importlib.import_module("anonymizer.engine.execution.phase8_runtime")
    operation_plan = phase8._compile_group_operation_plan(0, 3)
    assert operation_plan is not None
    operation_ledger = phase8._Phase8OperationLedger(operation_plan)
    if with_diagnostics:
        for stage in operation_plan.stages:
            if stage == phase8._Phase8Stage.analyze():
                break
            assert operation_ledger.succeed(stage)
        assert operation_ledger.fail(phase8._Phase8Stage.analyze(), phase8._Phase8Reason.BACKEND_FAILURE)

    cleanup_identity = _Identity()
    pre_cleanup = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.PRE_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        cleanup_identity,
    )
    post_cleanup = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.POST_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.FAILED if with_diagnostics else _Phase8CleanupStatus.VERIFIED,
        cleanup_identity,
        retained_candidate_cell_count=0 if with_diagnostics else 1,
    )
    lifecycle = phase8._Phase8LifecycleExecution(
        () if with_diagnostics else ((_Identity(), "OWNER-LIFECYCLE-CONTENT-CANARY"),),
        ("failed",) if with_diagnostics else ("succeeded",),
        False,
        pre_cleanup,
        post_cleanup,
    )
    return (
        ("accounting-ledger", accounting_ledger),
        ("accounting-graph-execution", graph_execution),
        ("phase8-operation-ledger", operation_ledger),
        ("phase8-lifecycle", lifecycle),
        ("phase8-cleanup", post_cleanup),
    )


def _owner_fingerprint(owner: object) -> object:
    # Test-only traversal: freeze mutable state without retaining any owner value.
    # Production inspection must continue using its explicit, closed reducers.
    def freeze(value: object) -> object:
        if value is None or type(value) in {str, int, bool, bytes}:
            return value
        if isinstance(value, Enum):
            return (type(value), value.value)
        if isinstance(value, (dict, MappingProxyType)):
            return (id(value), tuple((freeze(key), freeze(item)) for key, item in value.items()))
        if isinstance(value, (tuple, list)):
            return (id(value), tuple(freeze(item) for item in value))
        if isinstance(value, (set, frozenset)):
            return (id(value), frozenset(freeze(item) for item in value))
        if is_dataclass(value) and not isinstance(value, type):
            return (id(value), tuple((field.name, freeze(getattr(value, field.name))) for field in fields(value)))
        return (type(value), id(value))

    if isinstance(owner, _AccountingLedger):
        return freeze(vars(owner))
    return freeze(owner)


@pytest.fixture(autouse=True)
def _silent_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> Iterator[list[str]]:
    telemetry = importlib.import_module("anonymizer.telemetry")
    calls: list[str] = []

    def trap(*_args: object, **_kwargs: object) -> None:
        calls.append("telemetry")
        # Record even attempted emission; never reach a network or queue boundary.
        return None

    caplog.set_level(logging.DEBUG, logger="anonymizer")
    for name in ("enqueue", "flush", "_flush_events", "_send_events_with_client"):
        monkeypatch.setattr(telemetry.TelemetryHandler, name, trap)
    monkeypatch.setattr(telemetry, "build_payload", trap)
    yield calls
    assert calls == []


@pytest.mark.parametrize(
    "forbidden",
    [
        "ORIGINAL-TARGET-and-context-substring",
        "ENTITY-label-offset-mention-cluster-role-obligation",
        "REPLACEMENT-baseline-revision-prompt-evaluation",
        "GRAPH-datum-row-group-scope-task-attempt-session-identity",
        "CALLER-and-source-identifier",
        "PRIVATE-correlation-lineage-token-map",
        "CREDENTIAL-endpoint-model-provider-payload-usage",
        "EXCEPTION-type-module-args-message-traceback-stack",
        "CONTENT-DERIVED-HASH-deadbeef",
        "CONTRACT-PLAN-PROMPT-PAYLOAD-DIGEST",
        "/private/path hostname ENV_SECRET 0xdeadbeef",
        "2099-12-31T23:59:59Z",
    ],
)
def test_every_forbidden_value_category_is_absent_from_constructed_views(forbidden: str) -> None:
    encoded = _encoded_explain(forbidden)

    assert forbidden.encode() not in encoded
    assert b"private-identity-canary" not in encoded


@pytest.mark.parametrize("operation_name", ["EXPLAIN", "INSPECT", "DIAGNOSE"])
@pytest.mark.parametrize(
    "grant_state", ["forged", "reused", "expired", "wrong_owner", "wrong_subject", "wrong_operation"]
)
def test_unauthorized_inspection_never_reads_hostile_subject_or_emits_output(
    operation_name: str,
    grant_state: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    operation = getattr(module._Phase10Operation, operation_name)
    inspection = getattr(module, f"_{operation_name.lower()}_phase10")
    owner = _Identity()
    canary = _AccessCanary()
    call_owner = owner
    call_subject: object = canary
    grant: object
    _AccessCanary.touched = False

    if grant_state == "forged":
        grant = object()
    elif grant_state == "wrong_subject":
        bound_subject = _Identity()
        grant = module._issue_phase10_inspection_grant(owner, bound_subject, operation)
    else:
        grant = module._issue_phase10_inspection_grant(owner, canary, operation)
        assert type(grant) is module._Phase10InspectionGrant
        if grant_state == "reused":
            assert module._consume_phase10_inspection_grant(grant, owner, canary, operation)
        elif grant_state == "expired":
            module._expire_phase10_grant_nonce(grant._nonce)
        elif grant_state == "wrong_owner":
            call_owner = _Identity()
        elif grant_state == "wrong_operation":
            other_operation = {
                module._Phase10Operation.EXPLAIN: module._Phase10Operation.INSPECT,
                module._Phase10Operation.INSPECT: module._Phase10Operation.DIAGNOSE,
                module._Phase10Operation.DIAGNOSE: module._Phase10Operation.EXPLAIN,
            }[operation]
            module._revoke_phase10_inspection_grant(grant)
            grant = module._issue_phase10_inspection_grant(owner, canary, other_operation)

    result = inspection(call_owner, call_subject, grant)
    module._revoke_phase10_inspection_grant(grant)

    assert type(result) is module._Phase10InspectionRejected
    assert result.code is module._Phase10RejectionCode.DENIED
    assert not _AccessCanary.touched
    assert caplog.records == []
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_malformed_nested_value_rejects_without_repr_comparison_hash_or_truthiness() -> None:
    module = _module()
    canary = _AccessCanary()
    _AccessCanary.touched = False
    provenance = module._Phase10Provenance(
        module._Phase10InspectionSchemaVersion.V1,
        module._Phase10ContractVersion.V1,
        module._Phase10ViewKind.INSPECT,
        module._Phase10SubjectKind.INVOCATION_SNAPSHOT,
        module._Phase10SemanticProfile.GROUPED_REWRITE_V1,
        module._Phase10ImplementationProfile.PANDAS_RUNTIME_V1,
        module._Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED,
        module._Phase10LifecycleState.TERMINAL,
    )
    malformed = object.__new__(module._Phase10Snapshot)
    object.__setattr__(malformed, "stage_summaries", (canary,))
    object.__setattr__(malformed, "terminal_summaries", ())
    object.__setattr__(malformed, "reconciliation_state", module._Phase10ReconciliationState.RECONCILED)
    object.__setattr__(malformed, "cleanup_state", module._Phase10CleanupState.NOT_ENTERED)
    object.__setattr__(malformed, "release_state", module._Phase10ReleaseState.NOT_ENTERED)
    view = object.__new__(module._Phase10InspectView)
    object.__setattr__(view, "provenance", provenance)
    object.__setattr__(view, "snapshot", malformed)

    result = module._encode_phase10_view(view)

    assert type(result) is module._Phase10InspectionRejected
    assert result.code is module._Phase10RejectionCode.REDACTION_FAILED
    assert not _AccessCanary.touched


def test_view_object_graph_contains_only_allowlisted_detached_values() -> None:
    module = _module()
    ledger: _AccountingLedger[str] = _AccountingLedger(_plan("DETACHED-CONTENT-CANARY"))
    ledger.open()
    capture = ledger._phase10_snapshot()
    assert type(capture) is module._Phase10OwnerCapture
    owner = _Identity()
    grant = module._issue_phase10_inspection_grant(owner, capture, module._Phase10Operation.INSPECT)
    view = module._inspect_phase10(owner, capture, grant)
    assert type(view) is module._Phase10InspectView

    scalars = tuple(_walk_private_value(view))

    assert all(value is None or type(value) in {bool, int, str, bytes} or isinstance(value, tuple) for value in scalars)
    assert "DETACHED-CONTENT-CANARY" not in repr(view)
    assert "private-identity-canary" not in repr(view)


def test_snapshot_is_observational_and_does_not_mutate_owning_ledger() -> None:
    ledger: _AccountingLedger[str] = _AccountingLedger(_plan("NONINTERFERENCE-CANARY"))
    ledger.open()
    before = (ledger._opened, ledger._closed, ledger._mutation_sealed, tuple(ledger._states.items()))

    ledger._phase10_snapshot()

    after = (ledger._opened, ledger._closed, ledger._mutation_sealed, tuple(ledger._states.items()))
    assert after == before


def test_every_real_owner_snapshot_is_observational_and_detached_from_its_owner() -> None:
    module = _module()
    for name, owner in _real_owner_registry():
        before = _owner_fingerprint(owner)
        capture = cast(Any, owner)._phase10_snapshot()
        after = _owner_fingerprint(owner)

        assert type(capture) is module._Phase10OwnerCapture, name
        assert after == before, name
        assert all(value is not owner for value in _walk_private_value(capture)), name


@pytest.mark.parametrize("failure_kind", ["exception", "builder_limit"])
def test_every_real_owner_snapshot_failure_is_observational_cause_free_and_silent(
    failure_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()

    def fail(_self: object) -> object:
        raise RuntimeError("OWNER-SNAPSHOT-EXCEPTION-DETAIL-CANARY")

    for name, owner in _real_owner_registry():
        before = _owner_fingerprint(owner)
        with monkeypatch.context() as scoped:
            if failure_kind == "exception":
                scoped.setattr(type(owner), "_phase10_snapshot_unchecked", fail)
            else:
                scoped.setitem(module._PHASE10_LIMITS, module._Phase10LimitName.MAX_BUILDER_WORKING_BYTES, 1)
            result = cast(Any, owner)._phase10_snapshot()
        assert type(result) is module._Phase10InspectionRejected, name
        expected_code = (
            module._Phase10RejectionCode.REDACTION_FAILED
            if failure_kind == "exception"
            else module._Phase10RejectionCode.LIMIT_EXCEEDED
        )
        assert result.code is expected_code, name
        assert "CANARY" not in repr(result), name
        assert _owner_fingerprint(owner) == before, name

    assert caplog.records == []
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_post_cleanup_snapshot_never_reads_or_revives_cleanup_identity() -> None:
    canary = _AccessCanary()
    _AccessCanary.touched = False
    receipt = _issue_phase8_cleanup_receipt(
        _Phase8CleanupPhase.POST_REDUCTION,
        _Phase8CleanupComponent.RUNTIME,
        _Phase8CleanupStatus.VERIFIED,
        canary,
    )

    receipt._phase10_snapshot()

    assert not _AccessCanary.touched


def _walk_private_value(value: object) -> list[object]:
    if isinstance(value, tuple):
        return [value, *(item for child in value for item in _walk_private_value(child))]
    try:
        value_fields = fields(cast(Any, value))
    except TypeError:
        if hasattr(value, "value") and type(getattr(value, "value")) is str:
            return [getattr(value, "value")]
        return [value]
    return [item for field in value_fields for item in _walk_private_value(getattr(value, field.name))]


def test_private_inputs_views_and_encodings_mask_repr_and_reject_pickle() -> None:
    module = _module()
    owner = _Identity()
    plan = _plan("PICKLE-CONTENT-CANARY")
    grant = module._issue_phase10_inspection_grant(owner, plan, module._Phase10Operation.EXPLAIN)
    view = module._explain_phase10(owner, plan, grant)
    encoding = module._encode_phase10_view(view)

    for value in (grant, view, encoding):
        assert "PICKLE-CONTENT-CANARY" not in repr(value)
        with pytest.raises(TypeError):
            pickle.dumps(value)


def test_consumed_grant_retains_no_owner_or_subject_reference() -> None:
    module = _module()
    owner = _Identity()
    subject = _Identity()
    owner_ref = weakref.ref(owner)
    subject_ref = weakref.ref(subject)
    grant = module._issue_phase10_inspection_grant(owner, subject, module._Phase10Operation.INSPECT)

    assert module._consume_phase10_inspection_grant(grant, owner, subject, module._Phase10Operation.INSPECT)
    del owner
    del subject
    gc.collect()

    assert owner_ref() is None
    assert subject_ref() is None


def test_encoding_failure_is_cause_free_and_returns_no_partial_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    owner = _Identity()
    plan = _plan("ENCODING-FAILURE-CANARY")
    grant = module._issue_phase10_inspection_grant(owner, plan, module._Phase10Operation.EXPLAIN)
    view = module._explain_phase10(owner, plan, grant)

    def fail(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("ENCODING-EXCEPTION-DETAIL-CANARY")

    monkeypatch.setattr(module.json, "dumps", fail)
    result = module._encode_phase10_view(view)

    assert type(result) is module._Phase10InspectionRejected
    assert result.code is module._Phase10RejectionCode.ENCODING_FAILED
    assert "CANARY" not in repr(result)


def test_oversize_encoding_returns_no_prefix_suffix_or_partial_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    owner = _Identity()
    plan = _plan("OVERSIZE-PARTIAL-CANARY")
    grant = module._issue_phase10_inspection_grant(owner, plan, module._Phase10Operation.EXPLAIN)
    monkeypatch.setitem(
        module._PHASE10_LIMITS,
        module._Phase10LimitName.MAX_CANONICAL_JSON_UTF8_BYTES,
        1,
    )

    view = module._explain_phase10(owner, plan, grant)
    result = module._encode_phase10_view(view)

    assert type(result) is module._Phase10InspectionRejected
    assert result.code is module._Phase10RejectionCode.LIMIT_EXCEEDED
    assert not hasattr(result, "value")


def test_encoded_json_contains_only_exact_allowlisted_keys_and_scalar_types() -> None:
    payload = json.loads(_encoded_explain("STRUCTURAL-ALLOWLIST-CANARY"))
    allowed = {
        "declared_limits",
        "inspection_schema_version",
        "provenance",
        "rejection_category",
        "relationship_buckets",
        "required_capabilities",
        "route",
        "subject_kind",
        "view_kind",
    }

    assert set(payload) == allowed
    _assert_json_types(payload)


def test_all_operation_payloads_use_only_exact_recursive_schema_keys_and_builtin_scalars() -> None:
    module = _module()
    allowed_keysets = {
        frozenset(
            {
                "inspection_schema_version",
                "view_kind",
                "subject_kind",
                "provenance",
                "route",
                "required_capabilities",
                "declared_limits",
                "relationship_buckets",
                "rejection_category",
            }
        ),
        frozenset(
            {
                "inspection_schema_version",
                "view_kind",
                "subject_kind",
                "provenance",
                "stage_summaries",
                "terminal_summaries",
                "reconciliation_state",
                "cleanup_state",
                "release_state",
            }
        ),
        frozenset({"inspection_schema_version", "view_kind", "subject_kind", "provenance", "diagnostics"}),
        frozenset(
            {
                "inspection_schema_version",
                "inspection_contract_version",
                "view_kind",
                "subject_kind",
                "semantic_profile_version",
                "implementation_profile_version",
                "capture_boundary",
                "capture_lifecycle_state",
            }
        ),
        frozenset({"name", "value"}),
        frozenset({"dimension", "count_bucket"}),
        frozenset({"stage", "lifecycle_state", "task_count_bucket"}),
        frozenset({"stage", "terminal_state", "impact_count_bucket"}),
        frozenset(
            {
                "boundary",
                "stage",
                "terminal_state",
                "reason_category",
                "impact_count_bucket",
                "reconciliation_state",
                "cleanup_state",
            }
        ),
    }

    for operation_name, view in _operation_views():
        payload = module._view_payload(view)
        assert type(payload) is dict
        assert type(module._validate_phase10_payload(payload)) is module._Phase10PayloadMeasurement
        _assert_exact_keysets(payload, allowed_keysets)
        _assert_json_types(payload)
        encoding = module._encode_phase10_view(view)
        assert type(encoding) is module._Phase10CanonicalEncoding, operation_name
        assert json.loads(encoding.value) == payload


def _assert_exact_keysets(value: object, allowed_keysets: set[frozenset[str]]) -> None:
    if type(value) is dict:
        payload = cast(dict[str, object], value)
        assert frozenset(payload) in allowed_keysets
        for item in payload.values():
            _assert_exact_keysets(item, allowed_keysets)
    elif type(value) is list:
        for item in cast(list[object], value):
            _assert_exact_keysets(item, allowed_keysets)


def test_payload_measurement_enforces_each_key_utf8_boundary_without_partial_output() -> None:
    module = _module()

    exact = module._measure_phase10_payload({"é" * 48: None})
    over = module._measure_phase10_payload({"é" * 49: None})

    assert type(exact) is module._Phase10PayloadMeasurement
    assert exact.longest_string_utf8_bytes == 96
    assert type(over) is module._Phase10InspectionRejected
    assert over.code is module._Phase10RejectionCode.LIMIT_EXCEEDED
    assert not hasattr(over, "value")


@pytest.mark.parametrize("operation_name", ["EXPLAIN", "INSPECT", "DIAGNOSE"])
def test_operation_exceptions_are_cause_free_and_emit_no_output(
    operation_name: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module()
    operation = getattr(module._Phase10Operation, operation_name)
    inspection = getattr(module, f"_{operation_name.lower()}_phase10")
    owner = _Identity()
    if operation_name == "EXPLAIN":
        subject = _plan("OPERATION-EXCEPTION-DETAIL-CANARY")
        seam = "_valid_explain_subject"
    else:
        subject = _failed_accounting_capture("OPERATION-EXCEPTION-DETAIL-CANARY")
        seam = "_valid_owner_capture" if operation_name == "INSPECT" else "_phase10_admission_diagnostic"
    grant = module._issue_phase10_inspection_grant(owner, subject, operation)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("OPERATION-EXCEPTION-DETAIL-CANARY")

    monkeypatch.setattr(module, seam, fail)
    result = inspection(owner, subject, grant)

    assert type(result) is module._Phase10InspectionRejected
    assert result.code is module._Phase10RejectionCode.REDACTION_FAILED
    assert "CANARY" not in repr(result)
    assert caplog.records == []
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("grant_state", ["active", "consumed", "revoked", "owner_expired", "subject_expired"])
def test_grant_registry_never_retains_owner_or_subject_strong_references(grant_state: str) -> None:
    module = _module()
    owner = _Identity()
    subject = _Identity()
    owner_ref = weakref.ref(owner)
    subject_ref = weakref.ref(subject)
    grant = module._issue_phase10_inspection_grant(owner, subject, module._Phase10Operation.INSPECT)
    assert type(grant) is module._Phase10InspectionGrant

    if grant_state == "consumed":
        assert module._consume_phase10_inspection_grant(grant, owner, subject, module._Phase10Operation.INSPECT)
    elif grant_state == "revoked":
        module._revoke_phase10_inspection_grant(grant)
    elif grant_state == "owner_expired":
        del owner
        gc.collect()
        assert owner_ref() is None
    elif grant_state == "subject_expired":
        del subject
        gc.collect()
        assert subject_ref() is None

    if grant_state != "owner_expired":
        del owner
    if grant_state != "subject_expired":
        del subject
    gc.collect()

    assert owner_ref() is None
    assert subject_ref() is None
    module._revoke_phase10_inspection_grant(grant)


def test_rejected_and_successful_calls_retain_no_owner_subject_or_grant_reference() -> None:
    module = _module()

    rejected_owner = _Identity()
    rejected_subject = _Identity()
    rejected_owner_ref = weakref.ref(rejected_owner)
    rejected_subject_ref = weakref.ref(rejected_subject)
    rejected_grant = module._issue_phase10_inspection_grant(
        rejected_owner,
        rejected_subject,
        module._Phase10Operation.INSPECT,
    )
    rejected = module._inspect_phase10(rejected_owner, rejected_subject, rejected_grant)
    assert type(rejected) is module._Phase10InspectionRejected
    del rejected_owner
    del rejected_subject
    del rejected_grant

    successful_owner = _Identity()
    successful_subject = _plan("SUCCESSFUL-RETENTION-CONTENT-CANARY")
    successful_owner_ref = weakref.ref(successful_owner)
    successful_subject_ref = weakref.ref(successful_subject)
    successful_grant = module._issue_phase10_inspection_grant(
        successful_owner,
        successful_subject,
        module._Phase10Operation.EXPLAIN,
    )
    view = module._explain_phase10(successful_owner, successful_subject, successful_grant)
    encoding = module._encode_phase10_view(view)
    assert type(view) is module._Phase10ExplainView
    assert type(encoding) is module._Phase10CanonicalEncoding
    del successful_owner
    del successful_subject
    del successful_grant
    gc.collect()

    assert rejected_owner_ref() is None
    assert rejected_subject_ref() is None
    assert successful_owner_ref() is None
    assert successful_subject_ref() is None
    assert "CANARY" not in repr(view)
    assert b"CANARY" not in encoding.value


def _assert_json_types(value: object) -> None:
    assert value is None or type(value) in {bool, int, str, list, dict}
    if type(value) is list:
        for item in value:
            _assert_json_types(item)
    elif type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            _assert_json_types(item)


_OWNER_NAMES = (
    "accounting-ledger",
    "accounting-graph-execution",
    "phase8-operation-ledger",
    "phase8-lifecycle",
    "phase8-cleanup",
)
_OWNER_OPERATIONS = (
    ("accounting-plan", "explain"),
    *((name, operation) for name in _OWNER_NAMES for operation in ("inspect", "diagnose")),
)
_PLAN_OPERATIONS = (
    *((f"{name}-plan", "explain") for name in ("accounting", "context", "phase6", "phase7", "phase8")),
    *(
        (f"{name}-rejection", operation)
        for name in ("accounting", "context", "phase6", "phase7", "phase8")
        for operation in ("explain", "diagnose")
    ),
)
_FAILURE_OPERATIONS = (*_PLAN_OPERATIONS, *_OWNER_OPERATIONS[1:])
_FAILURE_PATHS = ("success", "denied", "builder_limit", "redaction", "payload_limit", "encoding", "encoded_limit")


@pytest.mark.parametrize("owner_name,operation_name", _FAILURE_OPERATIONS)
@pytest.mark.parametrize("failure_path", _FAILURE_PATHS)
def test_failure_paths_leave_real_protection_state_unchanged_and_silent(
    owner_name: str,
    operation_name: str,
    failure_path: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    _silent_telemetry: list[str],
) -> None:
    module = _module()
    if owner_name not in _OWNER_NAMES:
        conformance = importlib.import_module("tests.engine.execution.test_phase10_reference_conformance")
        owner = next(
            subject for name, subject, _operations in conformance._real_plan_subject_registry() if name == owner_name
        )
        subject = owner
    else:
        owner = dict(_real_owner_registry(with_diagnostics=True))[owner_name]
        before_capture = _owner_fingerprint(owner)
        subject = cast(Any, owner)._phase10_snapshot()
        assert type(subject) is module._Phase10OwnerCapture
        assert _owner_fingerprint(owner) == before_capture
    authorization_owner = _Identity()
    operation = getattr(module._Phase10Operation, operation_name.upper())
    inspection = getattr(module, f"_{operation_name}_phase10")
    grant = module._issue_phase10_inspection_grant(authorization_owner, subject, operation)
    assert type(grant) is module._Phase10InspectionGrant
    before = _owner_fingerprint(owner)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("FAILURE-MATRIX-EXCEPTION-CANARY")

    with monkeypatch.context() as scoped:
        if failure_path == "denied":
            module._revoke_phase10_inspection_grant(grant)
        elif failure_path == "builder_limit":
            scoped.setitem(module._PHASE10_LIMITS, module._Phase10LimitName.MAX_BUILDER_WORKING_BYTES, 1)
        elif failure_path == "redaction":
            seam = {
                "explain": "_phase10_explain_details",
                "inspect": "_phase10_provenance",
                "diagnose": "_phase10_provenance",
            }[operation_name]
            scoped.setattr(module, seam, fail)
        elif failure_path == "payload_limit":
            scoped.setitem(module._PHASE10_LIMITS, module._Phase10LimitName.MAX_TOP_LEVEL_FIELDS, 1)
        elif failure_path == "encoded_limit":
            scoped.setitem(module._PHASE10_LIMITS, module._Phase10LimitName.MAX_CANONICAL_JSON_UTF8_BYTES, 1)
        result = inspection(authorization_owner, subject, grant)
        if failure_path in {"success", "payload_limit", "encoding", "encoded_limit"}:
            assert type(result) is getattr(module, f"_Phase10{operation_name.title()}View")
            view = result
            before_view = _owner_fingerprint(view)
            if failure_path == "encoding":
                scoped.setattr(module.json, "dumps", fail)
            result = module._encode_phase10_view(view)
            assert _owner_fingerprint(view) == before_view
        expected_code = {
            "denied": module._Phase10RejectionCode.DENIED,
            "builder_limit": module._Phase10RejectionCode.LIMIT_EXCEEDED,
            "redaction": module._Phase10RejectionCode.REDACTION_FAILED,
            "payload_limit": module._Phase10RejectionCode.LIMIT_EXCEEDED,
            "encoding": module._Phase10RejectionCode.ENCODING_FAILED,
            "encoded_limit": module._Phase10RejectionCode.LIMIT_EXCEEDED,
        }
        if failure_path == "success":
            assert type(result) is module._Phase10CanonicalEncoding
        else:
            assert type(result) is module._Phase10InspectionRejected
            assert result.code is expected_code[failure_path]
            assert not hasattr(result, "value")
            assert "CANARY" not in repr(result)
    assert _owner_fingerprint(owner) == before
    telemetry_calls = _silent_telemetry.copy()
    _silent_telemetry.clear()
    assert telemetry_calls == []
    assert caplog.records == []
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


# A numeric identity probe is used only for GC-tracked slots-only owner objects
# that cannot be weak-referenced. It never holds the observed object alive.
_LifetimeProbe = tuple[str, weakref.ReferenceType[object] | int, type[object]]


def _lifetime_probe(label: str, value: object) -> _LifetimeProbe:
    try:
        return (label, weakref.ref(value), type(value))
    except TypeError:
        assert gc.is_tracked(value)
        return (label, id(value), type(value))


def _assert_lifetimes_released(probes: list[_LifetimeProbe]) -> None:
    gc.collect()
    remaining = []
    for label, reference, value_type in probes:
        if isinstance(reference, weakref.ReferenceType):
            alive = reference() is not None
        else:
            alive = any(id(value) == reference and type(value) is value_type for value in gc.get_objects())
        if alive:
            remaining.append(label)
    assert remaining == []


def _retention_scenario(
    owner_name: str,
    operation_name: str,
    grant_state: str,
) -> tuple[list[_LifetimeProbe], object | None, object | None]:
    module = _module()
    authorization_owner = _Identity()
    if owner_name == "accounting-plan":
        owner: Any = _plan("RETENTION-MATRIX-CONTENT-CANARY")
        subject = owner
        content_state = owner.datums[0]
    else:
        owner = cast(Any, dict(_real_owner_registry(with_diagnostics=True))[owner_name])
        subject = owner._phase10_snapshot()
        assert type(subject) is module._Phase10OwnerCapture
        if owner_name == "accounting-ledger":
            content_state = owner._plan.datums[0]
        elif owner_name == "accounting-graph-execution":
            content_state = owner.plan.datums[0]
        elif owner_name == "phase8-lifecycle":
            content_state = owner.post_reduction_cleanup.identity
        elif owner_name == "phase8-cleanup":
            content_state = owner.identity
        else:
            content_state = owner.plan
    operation = getattr(module._Phase10Operation, operation_name.upper())
    grant = module._issue_phase10_inspection_grant(authorization_owner, subject, operation)
    assert type(grant) is module._Phase10InspectionGrant
    probes = [
        _lifetime_probe(label, value)
        for label, value in (
            ("owner", owner),
            ("authorization_owner", authorization_owner),
            ("subject", subject),
            ("content_state", content_state),
            ("grant", grant),
        )
    ]
    view = encoding = None
    if grant_state == "consumed":
        assert module._consume_phase10_inspection_grant(grant, authorization_owner, subject, operation)
    elif grant_state == "revoked":
        module._revoke_phase10_inspection_grant(grant)
    elif grant_state == "expired":
        module._expire_phase10_grant_nonce(grant._nonce)
    elif grant_state in {"rejected", "successful"}:
        if grant_state == "rejected":
            module._revoke_phase10_inspection_grant(grant)
        result = getattr(module, f"_{operation_name}_phase10")(authorization_owner, subject, grant)
        if grant_state == "rejected":
            assert type(result) is module._Phase10InspectionRejected
            assert result.code is module._Phase10RejectionCode.DENIED
            probes.append(_lifetime_probe("rejection", result))
        else:
            assert type(result) is getattr(module, f"_Phase10{operation_name.title()}View")
            view = result
            encoding = module._encode_phase10_view(view)
            assert type(encoding) is module._Phase10CanonicalEncoding
    # Return caller-owned views separately to prove input collection while they live.
    return probes, view, encoding


@pytest.mark.parametrize("owner_name,operation_name", _OWNER_OPERATIONS)
@pytest.mark.parametrize("grant_state", ["active", "consumed", "revoked", "expired", "rejected", "successful"])
def test_all_call_lifetimes_release_owner_subject_content_grant_and_view(
    owner_name: str,
    operation_name: str,
    grant_state: str,
) -> None:
    probes, view, encoding = _retention_scenario(owner_name, operation_name, grant_state)
    _assert_lifetimes_released(probes)
    if grant_state == "successful":
        assert view is not None and encoding is not None
        probes.extend((_lifetime_probe("view", view), _lifetime_probe("encoding", encoding)))
    del view, encoding
    _assert_lifetimes_released(probes)


@pytest.mark.parametrize("operation_name", ["EXPLAIN", "INSPECT", "DIAGNOSE"])
def test_active_registry_releases_grant_while_bindings_remain_alive(operation_name: str) -> None:
    module = _module()
    owner, subject = _Identity(), _Identity()
    grant = module._issue_phase10_inspection_grant(owner, subject, getattr(module._Phase10Operation, operation_name))
    assert type(grant) is module._Phase10InspectionGrant
    nonce = grant._nonce
    probe = _lifetime_probe("grant", grant)
    del grant
    _assert_lifetimes_released([probe])
    assert nonce not in module._GRANT_STATES
    assert owner is not subject


@pytest.mark.parametrize("operation", ("explain", "inspect", "diagnose"))
@pytest.mark.parametrize(
    "limit_name,position",
    (
        ("MAX_TOP_LEVEL_FIELDS", 0),
        ("MAX_PROVENANCE_FIELDS", 1),
        ("MAX_JSON_NESTING_DEPTH", 2),
        ("MAX_ALLOWLISTED_STRING_UTF8_BYTES", 3),
    ),
)
def test_full_encoder_enforces_lowered_measured_payload_ceiling(
    monkeypatch: pytest.MonkeyPatch, operation: str, limit_name: str, position: int
) -> None:
    module = _module()
    original_view = dict(_operation_views())[operation]
    original_fingerprint = _owner_fingerprint(original_view)
    payload = module._view_payload(original_view)
    measured = module._measure_phase10_payload(payload)
    expected = (5 if operation == "diagnose" else 9, 8, 3, 40)
    assert (
        measured.top_level_fields,
        measured.provenance_fields,
        measured.json_nesting_depth,
        measured.longest_string_utf8_bytes,
    ) == expected
    name = getattr(module._Phase10LimitName, limit_name)
    original_validator = module._validate_phase10_payload
    validated: list[str] = []

    def recording_validator(value: Any) -> object:
        validated.append(value["view_kind"])
        return original_validator(value)

    monkeypatch.setattr(module, "_validate_phase10_payload", recording_validator)
    monkeypatch.setitem(module._PHASE10_LIMITS, name, expected[position])
    exact_view = dict(_operation_views())[operation]
    exact_fingerprint = _owner_fingerprint(exact_view)
    exact = module._encode_phase10_view(exact_view)
    assert type(exact) is module._Phase10CanonicalEncoding
    monkeypatch.setitem(module._PHASE10_LIMITS, name, expected[position] - 1)
    over_view = dict(_operation_views())[operation]
    over_fingerprint = _owner_fingerprint(over_view)
    rejected = module._encode_phase10_view(over_view)
    assert type(rejected) is module._Phase10InspectionRejected
    assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED
    assert not hasattr(rejected, "value")
    assert validated == [operation, operation]
    assert _owner_fingerprint(original_view) == original_fingerprint
    assert _owner_fingerprint(exact_view) == exact_fingerprint
    assert _owner_fingerprint(over_view) == over_fingerprint
