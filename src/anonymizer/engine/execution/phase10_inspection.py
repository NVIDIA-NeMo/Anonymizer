# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private immutable values for Phase 10 bounded inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import TypeVar, cast, final

_MAX_STAGE_SUMMARIES = 8
_MAX_TERMINAL_SUMMARIES = 48
_MAX_DIAGNOSTICS = 64
_MAX_CANONICAL_JSON_BYTES = 16_384
_MAX_DECLARED_INTEGER = 65_536
_GRANT_SEAL = object()
_ENCODING_SEAL = object()
_GRANT_LOCK = Lock()


class _PrivatePhase10InspectionValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 10 inspection values are not serializable")


class _Phase10InspectionSchemaVersion(str, Enum):
    V1 = "phase10-bounded-inspection-view/v1"


class _Phase10ContractVersion(str, Enum):
    V1 = "anonymizer-phase10-bounded-inspection/v1"


class _Phase10Operation(str, Enum):
    EXPLAIN = "explain"
    INSPECT = "inspect"
    DIAGNOSE = "diagnose"


class _Phase10ViewKind(str, Enum):
    EXPLAIN = "explain"
    INSPECT = "inspect"
    DIAGNOSE = "diagnose"


class _Phase10SubjectKind(str, Enum):
    ADMITTED_PLAN = "admitted_plan"
    ADMISSION_REJECTION = "admission_rejection_receipt"
    INVOCATION_SNAPSHOT = "invocation_snapshot"
    TERMINAL_RECEIPT = "terminal_receipt"
    CLEANUP_RECEIPT = "cleanup_receipt"
    PUBLICATION_RECEIPT = "publication_receipt"


class _Phase10SemanticProfile(str, Enum):
    TARGET_CONTEXT_V1 = "target-context-v1"
    REDACT_V1 = "phase6-redact-graph/v1"
    SUBSTITUTE_V1 = "anonymizer-phase7-stable-substitute/v1"
    GROUPED_REWRITE_V1 = "anonymizer-phase8-grouped-rewrite/v1"


class _Phase10ImplementationProfile(str, Enum):
    PANDAS_RUNTIME_V1 = "pandas-runtime-v1"


class _Phase10CaptureBoundary(str, Enum):
    ADMISSION_TERMINAL = "admission_terminal"
    INVOCATION_OPENED = "invocation_opened"
    PRE_DISPATCH = "pre_dispatch"
    POST_DISPATCH = "post_dispatch"
    TERMINAL_EVIDENCE_ACCEPTED = "terminal_evidence_accepted"
    PRE_REDUCTION_CLEANUP_TERMINAL = "pre_reduction_cleanup_terminal"
    POST_REDUCTION_CLEANUP_TERMINAL = "post_reduction_cleanup_terminal"
    RELEASE_TERMINAL = "release_terminal"
    INVOCATION_CLOSED = "invocation_closed"


class _Phase10LifecycleState(str, Enum):
    REJECTED = "rejected"
    OPENED = "opened"
    PRE_DISPATCH = "pre_dispatch"
    POST_DISPATCH = "post_dispatch"
    TERMINAL = "terminal"
    CLEANUP_TERMINAL = "cleanup_terminal"
    RELEASE_TERMINAL = "release_terminal"
    CLOSED = "closed"


class _Phase10Stage(str, Enum):
    ADMISSION = "admission"
    DETECT = "detect"
    AUGMENT = "augment"
    VALIDATE = "validate"
    FINALIZE = "finalize"
    RESOLVE = "resolve"
    CLASSIFY = "classify"
    TRANSFORM = "transform"
    VERIFY = "verify"
    ANALYZE = "analyze"
    REWRITE = "rewrite"
    EVALUATE = "evaluate"
    REPAIR = "repair"
    RECONCILE = "reconcile"
    CLEANUP = "cleanup"
    RELEASE = "release"
    PUBLICATION = "publication"


class _Phase10TerminalState(str, Enum):
    INCONSISTENT = "inconsistent"
    LOST = "lost"
    CANCELLED = "cancelled"
    FAILED = "failed"
    BLOCKED = "blocked"
    WITHHELD = "withheld"
    REJECTED = "rejected"
    SUCCEEDED = "succeeded"


class _Phase10ReasonCategory(str, Enum):
    ADMISSION_REJECTED = "admission_rejected"
    CAPABILITY_MISMATCH = "capability_mismatch"
    LIMIT_EXCEEDED = "limit_exceeded"
    VERIFICATION_FAILED = "verification_failed"
    BACKEND_FAILED = "backend_failed"
    CANCELLATION_BEFORE_DISPATCH = "cancellation_before_dispatch"
    STOP_ACKNOWLEDGED = "stop_acknowledged"
    EXECUTION_LOST = "execution_lost"
    PREREQUISITE_BLOCKED = "prerequisite_blocked"
    EVIDENCE_INCONSISTENT = "evidence_inconsistent"
    CLEANUP_FAILED = "cleanup_failed"
    CLEANUP_UNCONFIRMED = "cleanup_unconfirmed"
    PUBLICATION_FAILED = "publication_failed"
    INSPECTION_DENIED = "inspection_denied"
    INSPECTION_SUBJECT_INVALID = "inspection_subject_invalid"
    INSPECTION_STATE_UNAVAILABLE = "inspection_state_unavailable"
    INSPECTION_LIMIT_EXCEEDED = "inspection_limit_exceeded"
    INSPECTION_REDACTION_FAILED = "inspection_redaction_failed"
    INSPECTION_ENCODING_FAILED = "inspection_encoding_failed"
    UNEXPECTED_FAILURE = "unexpected_failure"


class _Phase10RejectionCode(str, Enum):
    DENIED = "inspection_denied"
    SUBJECT_INVALID = "inspection_subject_invalid"
    STATE_UNAVAILABLE = "inspection_state_unavailable"
    LIMIT_EXCEEDED = "inspection_limit_exceeded"
    REDACTION_FAILED = "inspection_redaction_failed"
    ENCODING_FAILED = "inspection_encoding_failed"


class _Phase10CountBucket(str, Enum):
    ZERO = "0"
    ONE = "1"
    TWO_TO_FOUR = "2-4"
    FIVE_TO_SIXTEEN = "5-16"
    SEVENTEEN_TO_SIXTY_FOUR = "17-64"
    SIXTY_FIVE_PLUS = "65+"


class _Phase10ReconciliationState(str, Enum):
    NOT_ENTERED = "not_entered"
    PENDING = "pending"
    RECONCILED = "reconciled"
    FAILED = "failed"
    INCONSISTENT = "inconsistent"


class _Phase10CleanupState(str, Enum):
    NOT_ENTERED = "not_entered"
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    UNCONFIRMED = "unconfirmed"


class _Phase10ReleaseState(str, Enum):
    NOT_ENTERED = "not_entered"
    PENDING = "pending"
    RELEASED = "released"
    WITHHELD = "withheld"
    FAILED = "failed"
    UNCONFIRMED = "unconfirmed"


class _Phase10Route(str, Enum):
    LOCAL = "local"
    NDD = "ndd"
    MIXED = "mixed"
    REJECTED = "rejected"


class _Phase10Capability(str, Enum):
    TERMINAL_ACCOUNTING = "terminal_accounting"
    BOUNDED_CONTEXT = "bounded_context"
    ANCHORED_MENTIONS = "anchored_mentions"
    STABLE_SUBSTITUTE = "stable_substitute"
    GROUPED_REWRITE = "grouped_rewrite"


class _Phase10LimitName(str, Enum):
    SUBJECTS_PER_CALL = "subjects_per_call"
    VIEWS_PER_CALL = "views_per_call"
    MAX_STAGE_SUMMARIES = "max_stage_summaries"
    MAX_TERMINAL_SUMMARY_ROWS = "max_terminal_summary_rows"
    MAX_DIAGNOSTIC_ENTRIES = "max_diagnostic_entries"
    MAX_REASON_CODES_PER_DIAGNOSTIC_ENTRY = "max_reason_codes_per_diagnostic_entry"
    MAX_PROVENANCE_FIELDS = "max_provenance_fields"
    MAX_TOP_LEVEL_FIELDS = "max_top_level_fields"
    MAX_JSON_NESTING_DEPTH = "max_json_nesting_depth"
    MAX_ALLOWLISTED_STRING_UTF8_BYTES = "max_allowlisted_string_utf8_bytes"
    MAX_CANONICAL_JSON_UTF8_BYTES = "max_canonical_json_utf8_bytes"
    MAX_BUILDER_WORKING_BYTES = "max_builder_working_bytes"


class _Phase10AggregateDimension(str, Enum):
    DATUMS = "datums"
    RELATIONSHIPS = "relationships"
    STAGES = "stages"
    TASKS = "tasks"
    GROUPS = "groups"
    OPERATIONS = "operations"
    REPAIRS = "repairs"
    OUTCOMES = "outcomes"
    RETAINED_REFERENCES = "retained_references"


E = TypeVar("E", bound=Enum)


def _require_enum(value: object, expected: type[E]) -> None:
    if type(value) is not expected:
        raise TypeError("unknown private Phase 10 value")


@dataclass(slots=True, repr=False)
class _Phase10GrantState(_PrivatePhase10InspectionValue):
    owner: object | None
    subject: object | None
    active: bool = True


@final
@dataclass(frozen=True, slots=True, repr=False)
class _Phase10InspectionGrant(_PrivatePhase10InspectionValue):
    operation: _Phase10Operation
    _nonce: object = field(compare=False)
    _seal: object = field(compare=False)


@final
@dataclass(frozen=True, slots=True, repr=False)
class _Phase10InspectionRejected(_PrivatePhase10InspectionValue):
    code: _Phase10RejectionCode

    def __post_init__(self) -> None:
        _require_enum(self.code, _Phase10RejectionCode)


@final
@dataclass(frozen=True, slots=True, repr=False)
class _Phase10Provenance(_PrivatePhase10InspectionValue):
    inspection_schema_version: _Phase10InspectionSchemaVersion
    inspection_contract_version: _Phase10ContractVersion
    view_kind: _Phase10ViewKind
    subject_kind: _Phase10SubjectKind
    semantic_profile_version: _Phase10SemanticProfile
    implementation_profile_version: _Phase10ImplementationProfile
    capture_boundary: _Phase10CaptureBoundary
    capture_lifecycle_state: _Phase10LifecycleState

    def __post_init__(self) -> None:
        if not _valid_provenance(self):
            raise TypeError("unknown private Phase 10 provenance value")


@final
@dataclass(frozen=True, slots=True, repr=False)
class _Phase10StageSummary(_PrivatePhase10InspectionValue):
    stage: _Phase10Stage
    lifecycle_state: _Phase10LifecycleState
    task_count_bucket: _Phase10CountBucket

    def __post_init__(self) -> None:
        _require_enum(self.stage, _Phase10Stage)
        _require_enum(self.lifecycle_state, _Phase10LifecycleState)
        _require_enum(self.task_count_bucket, _Phase10CountBucket)


@final
@dataclass(frozen=True, slots=True, repr=False)
class _Phase10TerminalSummary(_PrivatePhase10InspectionValue):
    stage: _Phase10Stage
    terminal_state: _Phase10TerminalState
    impact_count_bucket: _Phase10CountBucket

    def __post_init__(self) -> None:
        _require_enum(self.stage, _Phase10Stage)
        _require_enum(self.terminal_state, _Phase10TerminalState)
        _require_enum(self.impact_count_bucket, _Phase10CountBucket)


@final
@dataclass(frozen=True, slots=True, repr=False)
class _Phase10Snapshot(_PrivatePhase10InspectionValue):
    stage_summaries: tuple[_Phase10StageSummary, ...]
    terminal_summaries: tuple[_Phase10TerminalSummary, ...]
    reconciliation_state: _Phase10ReconciliationState
    cleanup_state: _Phase10CleanupState
    release_state: _Phase10ReleaseState

    def __post_init__(self) -> None:
        if not _valid_snapshot(self):
            raise TypeError("invalid private Phase 10 snapshot")


@final
@dataclass(frozen=True, slots=True, repr=False)
class _Phase10Diagnostic(_PrivatePhase10InspectionValue):
    boundary: _Phase10CaptureBoundary
    stage: _Phase10Stage
    terminal_state: _Phase10TerminalState
    reason_category: _Phase10ReasonCategory
    impact_count_bucket: _Phase10CountBucket
    reconciliation_state: _Phase10ReconciliationState
    cleanup_state: _Phase10CleanupState

    def __post_init__(self) -> None:
        if not _valid_diagnostic(self):
            raise TypeError("unknown private Phase 10 diagnostic value")


@final
@dataclass(frozen=True, slots=True, repr=False)
class _Phase10DeclaredLimit(_PrivatePhase10InspectionValue):
    name: _Phase10LimitName
    value: int

    def __post_init__(self) -> None:
        _require_enum(self.name, _Phase10LimitName)
        if type(self.value) is not int or not 0 <= self.value <= _MAX_DECLARED_INTEGER:
            raise TypeError("invalid private Phase 10 declared limit")


@final
@dataclass(frozen=True, slots=True, repr=False)
class _Phase10Aggregate(_PrivatePhase10InspectionValue):
    dimension: _Phase10AggregateDimension
    count_bucket: _Phase10CountBucket

    def __post_init__(self) -> None:
        _require_enum(self.dimension, _Phase10AggregateDimension)
        _require_enum(self.count_bucket, _Phase10CountBucket)


@final
@dataclass(frozen=True, slots=True, repr=False)
class _Phase10ExplainView(_PrivatePhase10InspectionValue):
    provenance: _Phase10Provenance
    route: _Phase10Route
    required_capabilities: tuple[_Phase10Capability, ...]
    declared_limits: tuple[_Phase10DeclaredLimit, ...]
    relationship_buckets: tuple[_Phase10Aggregate, ...]
    rejection_category: _Phase10ReasonCategory | None

    def __post_init__(self) -> None:
        if not _valid_explain_view(self):
            raise TypeError("invalid private Phase 10 explain view")


@final
@dataclass(frozen=True, slots=True, repr=False)
class _Phase10InspectView(_PrivatePhase10InspectionValue):
    provenance: _Phase10Provenance
    snapshot: _Phase10Snapshot

    def __post_init__(self) -> None:
        if not _valid_inspect_view(self):
            raise TypeError("invalid private Phase 10 inspect view")


@final
@dataclass(frozen=True, slots=True, repr=False)
class _Phase10DiagnoseView(_PrivatePhase10InspectionValue):
    provenance: _Phase10Provenance
    diagnostics: tuple[_Phase10Diagnostic, ...]

    def __post_init__(self) -> None:
        if not _valid_diagnose_view(self):
            raise TypeError("invalid private Phase 10 diagnose view")


@final
@dataclass(frozen=True, slots=True, repr=False)
class _Phase10CanonicalEncoding(_PrivatePhase10InspectionValue):
    value: bytes
    _proof: object = field(compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.value) is not bytes
            or self._proof is not _ENCODING_SEAL
            or len(self.value) > _MAX_CANONICAL_JSON_BYTES
            or self.value.endswith(b"\n")
        ):
            raise TypeError("invalid private Phase 10 canonical encoding")


_GRANT_STATES: dict[object, _Phase10GrantState] = {}


def _issue_phase10_inspection_grant(
    owner: object,
    subject: object,
    operation: _Phase10Operation,
) -> _Phase10InspectionGrant | _Phase10InspectionRejected:
    if owner is None or subject is None or type(operation) is not _Phase10Operation:
        return _Phase10InspectionRejected(_Phase10RejectionCode.SUBJECT_INVALID)
    nonce = object()
    with _GRANT_LOCK:
        _GRANT_STATES[nonce] = _Phase10GrantState(owner, subject)
    return _Phase10InspectionGrant(operation, nonce, _GRANT_SEAL)


def _consume_phase10_inspection_grant(
    grant: object,
    owner: object,
    subject: object,
    operation: _Phase10Operation,
) -> bool:
    if type(grant) is not _Phase10InspectionGrant or grant._seal is not _GRANT_SEAL:
        return False
    with _GRANT_LOCK:
        state = _GRANT_STATES.get(grant._nonce)
        if not _grant_matches(grant, state, owner, subject, operation):
            return False
        del _GRANT_STATES[grant._nonce]
        _expire_grant_state(cast(_Phase10GrantState, state))
        return True


def _revoke_phase10_inspection_grant(grant: object) -> None:
    if type(grant) is _Phase10InspectionGrant and grant._seal is _GRANT_SEAL:
        with _GRANT_LOCK:
            state = _GRANT_STATES.pop(grant._nonce, None)
        if state is not None:
            _expire_grant_state(state)


def _grant_matches(
    grant: _Phase10InspectionGrant,
    state: _Phase10GrantState | None,
    owner: object,
    subject: object,
    operation: _Phase10Operation,
) -> bool:
    return (
        type(state) is _Phase10GrantState
        and type(operation) is _Phase10Operation
        and grant.operation is operation
        and state.active
        and state.owner is owner
        and state.subject is subject
    )


def _expire_grant_state(state: _Phase10GrantState) -> None:
    state.active = False
    state.owner = None
    state.subject = None


def _valid_provenance(value: object) -> bool:
    if type(value) is not _Phase10Provenance:
        return False
    fields = (
        (value.inspection_schema_version, _Phase10InspectionSchemaVersion),
        (value.inspection_contract_version, _Phase10ContractVersion),
        (value.view_kind, _Phase10ViewKind),
        (value.subject_kind, _Phase10SubjectKind),
        (value.semantic_profile_version, _Phase10SemanticProfile),
        (value.implementation_profile_version, _Phase10ImplementationProfile),
        (value.capture_boundary, _Phase10CaptureBoundary),
        (value.capture_lifecycle_state, _Phase10LifecycleState),
    )
    return all(type(field_value) is field_type for field_value, field_type in fields) and _valid_view_subject(
        value.view_kind,
        value.subject_kind,
    )


def _valid_view_subject(kind: _Phase10ViewKind, subject: _Phase10SubjectKind) -> bool:
    allowed = {
        _Phase10ViewKind.EXPLAIN: {_Phase10SubjectKind.ADMITTED_PLAN, _Phase10SubjectKind.ADMISSION_REJECTION},
        _Phase10ViewKind.INSPECT: {
            _Phase10SubjectKind.INVOCATION_SNAPSHOT,
            _Phase10SubjectKind.TERMINAL_RECEIPT,
            _Phase10SubjectKind.CLEANUP_RECEIPT,
            _Phase10SubjectKind.PUBLICATION_RECEIPT,
        },
        _Phase10ViewKind.DIAGNOSE: {
            _Phase10SubjectKind.ADMISSION_REJECTION,
            _Phase10SubjectKind.INVOCATION_SNAPSHOT,
            _Phase10SubjectKind.TERMINAL_RECEIPT,
            _Phase10SubjectKind.CLEANUP_RECEIPT,
            _Phase10SubjectKind.PUBLICATION_RECEIPT,
        },
    }
    return subject in allowed[kind]


def _valid_snapshot(value: object) -> bool:
    return (
        type(value) is _Phase10Snapshot
        and type(value.stage_summaries) is tuple
        and len(value.stage_summaries) <= _MAX_STAGE_SUMMARIES
        and all(type(item) is _Phase10StageSummary for item in value.stage_summaries)
        and type(value.terminal_summaries) is tuple
        and len(value.terminal_summaries) <= _MAX_TERMINAL_SUMMARIES
        and all(type(item) is _Phase10TerminalSummary for item in value.terminal_summaries)
        and type(value.reconciliation_state) is _Phase10ReconciliationState
        and type(value.cleanup_state) is _Phase10CleanupState
        and type(value.release_state) is _Phase10ReleaseState
    )


def _valid_diagnostic(value: object) -> bool:
    if type(value) is not _Phase10Diagnostic:
        return False
    fields = (
        (value.boundary, _Phase10CaptureBoundary),
        (value.stage, _Phase10Stage),
        (value.terminal_state, _Phase10TerminalState),
        (value.reason_category, _Phase10ReasonCategory),
        (value.impact_count_bucket, _Phase10CountBucket),
        (value.reconciliation_state, _Phase10ReconciliationState),
        (value.cleanup_state, _Phase10CleanupState),
    )
    return all(type(field_value) is field_type for field_value, field_type in fields)


def _unique_enum_fields(values: tuple[object, ...], field_name: str) -> bool:
    fields = tuple(getattr(value, field_name, None) for value in values)
    return None not in fields and len(fields) == len(set(fields))


def _valid_explain_view(value: object) -> bool:
    return (
        type(value) is _Phase10ExplainView
        and _valid_provenance(value.provenance)
        and value.provenance.view_kind is _Phase10ViewKind.EXPLAIN
        and type(value.route) is _Phase10Route
        and type(value.required_capabilities) is tuple
        and len(value.required_capabilities) <= _MAX_STAGE_SUMMARIES
        and all(type(item) is _Phase10Capability for item in value.required_capabilities)
        and len(value.required_capabilities) == len(set(value.required_capabilities))
        and type(value.declared_limits) is tuple
        and all(type(item) is _Phase10DeclaredLimit for item in value.declared_limits)
        and _unique_enum_fields(cast(tuple[object, ...], value.declared_limits), "name")
        and type(value.relationship_buckets) is tuple
        and all(type(item) is _Phase10Aggregate for item in value.relationship_buckets)
        and _unique_enum_fields(cast(tuple[object, ...], value.relationship_buckets), "dimension")
        and (value.rejection_category is None or type(value.rejection_category) is _Phase10ReasonCategory)
    )


def _valid_inspect_view(value: object) -> bool:
    return (
        type(value) is _Phase10InspectView
        and _valid_provenance(value.provenance)
        and value.provenance.view_kind is _Phase10ViewKind.INSPECT
        and _valid_snapshot(value.snapshot)
    )


def _valid_diagnose_view(value: object) -> bool:
    return (
        type(value) is _Phase10DiagnoseView
        and _valid_provenance(value.provenance)
        and value.provenance.view_kind is _Phase10ViewKind.DIAGNOSE
        and type(value.diagnostics) is tuple
        and len(value.diagnostics) <= _MAX_DIAGNOSTICS
        and all(_valid_diagnostic(item) for item in value.diagnostics)
    )


def _provenance_payload(value: _Phase10Provenance) -> dict[str, str]:
    return {
        "inspection_schema_version": value.inspection_schema_version.value,
        "inspection_contract_version": value.inspection_contract_version.value,
        "view_kind": value.view_kind.value,
        "subject_kind": value.subject_kind.value,
        "semantic_profile_version": value.semantic_profile_version.value,
        "implementation_profile_version": value.implementation_profile_version.value,
        "capture_boundary": value.capture_boundary.value,
        "capture_lifecycle_state": value.capture_lifecycle_state.value,
    }


def _diagnostic_payload(value: _Phase10Diagnostic) -> dict[str, str]:
    return {
        "boundary": value.boundary.value,
        "stage": value.stage.value,
        "terminal_state": value.terminal_state.value,
        "reason_category": value.reason_category.value,
        "impact_count_bucket": value.impact_count_bucket.value,
        "reconciliation_state": value.reconciliation_state.value,
        "cleanup_state": value.cleanup_state.value,
    }


def _explain_payload(value: _Phase10ExplainView) -> dict[str, object]:
    return {
        "inspection_schema_version": value.provenance.inspection_schema_version.value,
        "view_kind": value.provenance.view_kind.value,
        "subject_kind": value.provenance.subject_kind.value,
        "provenance": _provenance_payload(value.provenance),
        "route": value.route.value,
        "required_capabilities": [item.value for item in value.required_capabilities],
        "declared_limits": [{"name": item.name.value, "value": item.value} for item in value.declared_limits],
        "relationship_buckets": [
            {"dimension": item.dimension.value, "count_bucket": item.count_bucket.value}
            for item in value.relationship_buckets
        ],
        "rejection_category": value.rejection_category.value if value.rejection_category is not None else None,
    }


def _inspect_payload(value: _Phase10InspectView) -> dict[str, object]:
    snapshot = value.snapshot
    return {
        "inspection_schema_version": value.provenance.inspection_schema_version.value,
        "view_kind": value.provenance.view_kind.value,
        "subject_kind": value.provenance.subject_kind.value,
        "provenance": _provenance_payload(value.provenance),
        "stage_summaries": [
            {
                "stage": item.stage.value,
                "lifecycle_state": item.lifecycle_state.value,
                "task_count_bucket": item.task_count_bucket.value,
            }
            for item in snapshot.stage_summaries
        ],
        "terminal_summaries": [
            {
                "stage": item.stage.value,
                "terminal_state": item.terminal_state.value,
                "impact_count_bucket": item.impact_count_bucket.value,
            }
            for item in snapshot.terminal_summaries
        ],
        "reconciliation_state": snapshot.reconciliation_state.value,
        "cleanup_state": snapshot.cleanup_state.value,
        "release_state": snapshot.release_state.value,
    }


def _diagnose_payload(value: _Phase10DiagnoseView) -> dict[str, object]:
    return {
        "inspection_schema_version": value.provenance.inspection_schema_version.value,
        "view_kind": value.provenance.view_kind.value,
        "subject_kind": value.provenance.subject_kind.value,
        "provenance": _provenance_payload(value.provenance),
        "diagnostics": [_diagnostic_payload(item) for item in value.diagnostics],
    }


def _encode_phase10_view(
    value: object,
) -> _Phase10CanonicalEncoding | _Phase10InspectionRejected:
    payload = _view_payload(value)
    if payload is None:
        return _Phase10InspectionRejected(_Phase10RejectionCode.REDACTION_FAILED)
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError):
        return _Phase10InspectionRejected(_Phase10RejectionCode.ENCODING_FAILED)
    if len(encoded) > _MAX_CANONICAL_JSON_BYTES:
        return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
    return _Phase10CanonicalEncoding(encoded, _ENCODING_SEAL)


def _view_payload(value: object) -> dict[str, object] | None:
    if _valid_explain_view(value):
        return _explain_payload(cast(_Phase10ExplainView, value))
    if _valid_inspect_view(value):
        return _inspect_payload(cast(_Phase10InspectView, value))
    if _valid_diagnose_view(value):
        return _diagnose_payload(cast(_Phase10DiagnoseView, value))
    return None
