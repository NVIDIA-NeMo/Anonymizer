# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private immutable values for Phase 10 bounded inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Protocol, TypeAlias, TypeVar, cast, final
from weakref import ReferenceType, finalize, ref

from anonymizer.engine.execution.accounting_admission import (
    _AccountingAdmissionCode,
    _AccountingRejected,
)
from anonymizer.engine.execution.accounting_outcomes import _CauseCode
from anonymizer.engine.execution.accounting_plan import _AccountingPlan, _is_admitted_accounting_plan
from anonymizer.engine.execution.context_admission import (
    _ContextAdmissionCode,
    _ContextPlan,
    _ContextRejected,
    _is_admitted_context_plan,
)
from anonymizer.engine.execution.context_workframes import _ContextBindingFault
from anonymizer.engine.execution.mention_admission import _MentionRejectionCode
from anonymizer.engine.execution.mention_resolution import _ResolutionRejectionCode
from anonymizer.engine.execution.phase6_plan import (
    _is_admitted_phase6_plan,
    _Phase6Plan,
    _Phase6PlanRejectionCode,
    _Phase6ProfileVersion,
    _Phase6Rejected,
)
from anonymizer.engine.execution.phase7_admission import (
    _is_admitted_phase7_plan,
    _Phase7AdmissionCode,
    _Phase7Plan,
    _Phase7Rejected,
)
from anonymizer.engine.execution.phase7_application import _ApplicationRejectionCode
from anonymizer.engine.execution.phase7_contract import _Phase7ContractRejectionCode
from anonymizer.engine.execution.phase7_ndd_backend import _Phase7NddReason
from anonymizer.engine.execution.phase7_validation import _BundleRejectionCode
from anonymizer.engine.execution.phase8_admission import (
    _is_admitted_phase8_plan,
    _Phase8AdmissionCode,
    _Phase8Plan,
    _Phase8Rejected,
)
from anonymizer.engine.execution.phase8_runtime import _Phase8Reason
from anonymizer.engine.execution.redact_patches import _PatchRejectionCode
from anonymizer.engine.execution.role_policy import _RolePolicyRejectionCode, _UnsupportedRoleReason

_MAX_STAGE_SUMMARIES = 8
_MAX_TERMINAL_SUMMARIES = 48
_MAX_DIAGNOSTICS = 64
_MAX_REASON_CODES_PER_DIAGNOSTIC = 4
_MAX_CANONICAL_JSON_BYTES = 16_384
_MAX_BUILDER_WORKING_BYTES = 65_536
_MAX_DECLARED_INTEGER = 65_536
_BUILDER_BASE_BYTES = 4_096
_BUILDER_ROW_BYTES = 512
_MAX_EXPLAIN_BUILDER_ROWS = 8 + 12 + 9
_MAX_INSPECT_BUILDER_ROWS = _MAX_STAGE_SUMMARIES + _MAX_TERMINAL_SUMMARIES
_MAX_DIAGNOSE_BUILDER_ROWS = _MAX_DIAGNOSTICS
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


_PHASE10_REASON_TABLE: dict[type[Enum], dict[Enum, _Phase10ReasonCategory]] = {
    _AccountingAdmissionCode: {
        _AccountingAdmissionCode.MALFORMED_GRAPH: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _AccountingAdmissionCode.TOO_MANY_DATUMS: _Phase10ReasonCategory.LIMIT_EXCEEDED,
        _AccountingAdmissionCode.DATUM_TOO_LARGE: _Phase10ReasonCategory.LIMIT_EXCEEDED,
        _AccountingAdmissionCode.GRAPH_TOO_LARGE: _Phase10ReasonCategory.LIMIT_EXCEEDED,
        _AccountingAdmissionCode.DUPLICATE_DATUM_ID: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _AccountingAdmissionCode.TOO_MANY_DEPENDENCIES: _Phase10ReasonCategory.LIMIT_EXCEEDED,
        _AccountingAdmissionCode.TOO_MANY_ATOMIC_GROUPS: _Phase10ReasonCategory.LIMIT_EXCEEDED,
        _AccountingAdmissionCode.MALFORMED_DEPENDENCY: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _AccountingAdmissionCode.DANGLING_DEPENDENCY: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _AccountingAdmissionCode.SELF_DEPENDENCY: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _AccountingAdmissionCode.DUPLICATE_DEPENDENCY: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _AccountingAdmissionCode.DEPENDENCY_CYCLE: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _AccountingAdmissionCode.EMPTY_ATOMIC_GROUP: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _AccountingAdmissionCode.DANGLING_ATOMIC_MEMBER: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _AccountingAdmissionCode.DUPLICATE_ATOMIC_MEMBER: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _AccountingAdmissionCode.DUPLICATE_ATOMIC_GROUP: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _AccountingAdmissionCode.ATOMIC_COVERAGE_GAP: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _AccountingAdmissionCode.ATOMIC_GROUP_OVERLAP: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _AccountingAdmissionCode.UNSUPPORTED_ATOMIC_NESTING: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
        _AccountingAdmissionCode.UNSUPPORTED_RELATIONSHIPS: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
        _AccountingAdmissionCode.UNSUPPORTED_CONTEXT: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
        _AccountingAdmissionCode.UNSUPPORTED_COHERENCE: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
        _AccountingAdmissionCode.UNSUPPORTED_TASK_CARDINALITY: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
    },
    _ContextAdmissionCode: {
        _ContextAdmissionCode.MALFORMED_GRAPH: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _ContextAdmissionCode.INVALID_DATUM_PURPOSE: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _ContextAdmissionCode.MISSING_CONTEXT_SCOPE: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _ContextAdmissionCode.DUPLICATE_CONTEXT_SCOPE: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _ContextAdmissionCode.UNKNOWN_CONTEXT_TARGET: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _ContextAdmissionCode.UNKNOWN_CONTEXT_DATUM: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _ContextAdmissionCode.CONTEXT_ONLY_TARGET: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _ContextAdmissionCode.ORPHAN_CONTEXT_DATUM: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _ContextAdmissionCode.SELF_CONTEXT: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _ContextAdmissionCode.DUPLICATE_CONTEXT_MEMBER: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _ContextAdmissionCode.TARGET_CONTEXT_DISABLED: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
        _ContextAdmissionCode.CONTEXT_MEMBERS_EXCEEDED: _Phase10ReasonCategory.LIMIT_EXCEEDED,
        _ContextAdmissionCode.CONTEXT_BYTES_EXCEEDED: _Phase10ReasonCategory.LIMIT_EXCEEDED,
        _ContextAdmissionCode.TOTAL_CONTEXT_REFERENCES_EXCEEDED: _Phase10ReasonCategory.LIMIT_EXCEEDED,
        _ContextAdmissionCode.EXPANDED_FRAME_BYTES_EXCEEDED: _Phase10ReasonCategory.LIMIT_EXCEEDED,
        _ContextAdmissionCode.UNSUPPORTED_CONTEXT_CONTRACT: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
        _ContextAdmissionCode.BACKEND_INCOMPATIBLE: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
    },
    _CauseCode: {
        _CauseCode.KNOWN_FAILURE: _Phase10ReasonCategory.BACKEND_FAILED,
        _CauseCode.VERIFICATION_FAILED: _Phase10ReasonCategory.VERIFICATION_FAILED,
        _CauseCode.RELEASE_PREDICATE_FAILED: _Phase10ReasonCategory.PUBLICATION_FAILED,
        _CauseCode.CANCELLATION: _Phase10ReasonCategory.CANCELLATION_BEFORE_DISPATCH,
        _CauseCode.STOP_ACKNOWLEDGED: _Phase10ReasonCategory.STOP_ACKNOWLEDGED,
        _CauseCode.TRANSPORT_LOST: _Phase10ReasonCategory.EXECUTION_LOST,
        _CauseCode.MISSING: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _CauseCode.DUPLICATE: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _CauseCode.UNKNOWN: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _CauseCode.FOREIGN: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _CauseCode.STALE: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _CauseCode.SWAPPED: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _CauseCode.CONTRADICTORY: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _CauseCode.PLAN_MISMATCH: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _CauseCode.PREREQUISITE: _Phase10ReasonCategory.PREREQUISITE_BLOCKED,
        _CauseCode.RESULT_CONSTRUCTION_FAILED: _Phase10ReasonCategory.UNEXPECTED_FAILURE,
        _CauseCode.CLEANUP_FAILED: _Phase10ReasonCategory.CLEANUP_FAILED,
        _CauseCode.CLEANUP_UNCONFIRMED: _Phase10ReasonCategory.CLEANUP_UNCONFIRMED,
    },
    _ContextBindingFault: {
        _ContextBindingFault.MISSING: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _ContextBindingFault.DUPLICATE: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _ContextBindingFault.CONTRADICTORY: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
    },
    _MentionRejectionCode: {
        _MentionRejectionCode.UNKNOWN_TARGET: _Phase10ReasonCategory.VERIFICATION_FAILED,
        _MentionRejectionCode.INVALID_OFFSET: _Phase10ReasonCategory.VERIFICATION_FAILED,
        _MentionRejectionCode.SOURCE_SLICE_MISMATCH: _Phase10ReasonCategory.VERIFICATION_FAILED,
        _MentionRejectionCode.UNSUPPORTED_PROVENANCE: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
        _MentionRejectionCode.MISSING_DECISION: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _MentionRejectionCode.DUPLICATE_DECISION: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _MentionRejectionCode.OVERLAP: _Phase10ReasonCategory.VERIFICATION_FAILED,
        _MentionRejectionCode.FOREIGN_TOKEN: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _MentionRejectionCode.STALE_TOKEN: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _MentionRejectionCode.CONTRADICTORY_CANDIDATE: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
    },
    _ResolutionRejectionCode: {
        _ResolutionRejectionCode.FOREIGN_TOKEN: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _ResolutionRejectionCode.STALE_TOKEN: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _ResolutionRejectionCode.INVALID_EVIDENCE: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _ResolutionRejectionCode.EVIDENCE_CONTRADICTION: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
    },
    _UnsupportedRoleReason: {
        _UnsupportedRoleReason.UNSUPPORTED_ROLE: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
    },
    _RolePolicyRejectionCode: {
        _RolePolicyRejectionCode.UNSUPPORTED_ROLE: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
    },
    _Phase6PlanRejectionCode: {
        _Phase6PlanRejectionCode.INVALID_PROFILE: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
    },
    _PatchRejectionCode: {
        _PatchRejectionCode.FOREIGN_TOKEN: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _PatchRejectionCode.STALE_TOKEN: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _PatchRejectionCode.INVALID_PATCH: _Phase10ReasonCategory.VERIFICATION_FAILED,
        _PatchRejectionCode.RELEASE_PREDICATE_FAILED: _Phase10ReasonCategory.PUBLICATION_FAILED,
    },
    _Phase7ContractRejectionCode: {
        _Phase7ContractRejectionCode.INVALID_CONTRACT: _Phase10ReasonCategory.ADMISSION_REJECTED,
    },
    _Phase7AdmissionCode: {
        _Phase7AdmissionCode.INVALID_INPUT: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase7AdmissionCode.LIMIT_EXCEEDED: _Phase10ReasonCategory.LIMIT_EXCEEDED,
        _Phase7AdmissionCode.EMPTY_SCOPE: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase7AdmissionCode.DUPLICATE_SCOPE: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase7AdmissionCode.DUPLICATE_SCOPE_MEMBER: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase7AdmissionCode.UNKNOWN_SCOPE_DATUM: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase7AdmissionCode.SCOPE_COVERAGE_GAP: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase7AdmissionCode.SCOPE_OVERLAP: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase7AdmissionCode.UNSUPPORTED_SCOPE_NESTING: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
        _Phase7AdmissionCode.PHASE6_HANDOFF_MISMATCH: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _Phase7AdmissionCode.UNSUPPORTED_SELECTOR: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
        _Phase7AdmissionCode.SELECTOR_MISSING: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase7AdmissionCode.SELECTOR_AMBIGUOUS: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _Phase7AdmissionCode.UNSUPPORTED_RELATION: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
        _Phase7AdmissionCode.CROSS_SCOPE_RELATION: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase7AdmissionCode.RELATION_ROLE_MISMATCH: _Phase10ReasonCategory.VERIFICATION_FAILED,
    },
    _ApplicationRejectionCode: {
        _ApplicationRejectionCode.INVALID_APPLICATION: _Phase10ReasonCategory.VERIFICATION_FAILED,
    },
    _BundleRejectionCode: {
        _BundleRejectionCode.INVALID_INPUT: _Phase10ReasonCategory.VERIFICATION_FAILED,
        _BundleRejectionCode.DUPLICATE_SLOT: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _BundleRejectionCode.FOREIGN_SLOT: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _BundleRejectionCode.PARTIAL_BUNDLE: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _BundleRejectionCode.CANDIDATE_MATCHES_ORIGINAL: _Phase10ReasonCategory.VERIFICATION_FAILED,
        _BundleRejectionCode.LIMIT_EXCEEDED: _Phase10ReasonCategory.LIMIT_EXCEEDED,
        _BundleRejectionCode.UNSUPPORTED_ROLE: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
        _BundleRejectionCode.CANONICAL_COLLISION: _Phase10ReasonCategory.VERIFICATION_FAILED,
        _BundleRejectionCode.UNSUPPORTED_CONSTRAINT: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
        _BundleRejectionCode.RELATION_FAILED: _Phase10ReasonCategory.VERIFICATION_FAILED,
    },
    _Phase7NddReason: {
        _Phase7NddReason.BACKEND_FAILED: _Phase10ReasonCategory.BACKEND_FAILED,
        _Phase7NddReason.EVIDENCE_UNATTRIBUTABLE: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _Phase7NddReason.LIMIT_EXCEEDED: _Phase10ReasonCategory.LIMIT_EXCEEDED,
        _Phase7NddReason.CONTRACT_INVALID: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase7NddReason.PHASE6_HANDOFF_MISMATCH: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
    },
    _Phase8AdmissionCode: {
        _Phase8AdmissionCode.INVALID_INPUT: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase8AdmissionCode.EMPTY_GROUP: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase8AdmissionCode.COVERAGE_GAP: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase8AdmissionCode.DUPLICATE_GROUP: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase8AdmissionCode.DUPLICATE_MEMBER: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase8AdmissionCode.OVERLAP: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase8AdmissionCode.UNKNOWN_MEMBER: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase8AdmissionCode.CROSS_ATOMIC: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase8AdmissionCode.LIMIT_EXCEEDED: _Phase10ReasonCategory.LIMIT_EXCEEDED,
    },
    _Phase8Reason: {
        _Phase8Reason.ANALYSIS_INVALID: _Phase10ReasonCategory.VERIFICATION_FAILED,
        _Phase8Reason.ANALYSIS_RECONCILIATION: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _Phase8Reason.ANALYSIS_STATE_MISSING: _Phase10ReasonCategory.UNEXPECTED_FAILURE,
        _Phase8Reason.BACKEND_FAILURE: _Phase10ReasonCategory.BACKEND_FAILED,
        _Phase8Reason.BACKEND_UNAVAILABLE: _Phase10ReasonCategory.CAPABILITY_MISMATCH,
        _Phase8Reason.CANCELLATION: _Phase10ReasonCategory.STOP_ACKNOWLEDGED,
        _Phase8Reason.CANDIDATE_RECONCILIATION: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _Phase8Reason.EVALUATION_INVALID: _Phase10ReasonCategory.VERIFICATION_FAILED,
        _Phase8Reason.EVALUATION_RECONCILIATION: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _Phase8Reason.GROUP_OPERATION_REUSED: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _Phase8Reason.INCOMPLETE_GROUP: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _Phase8Reason.INVALID_EVALUATION: _Phase10ReasonCategory.VERIFICATION_FAILED,
        _Phase8Reason.INVALID_GROUP_INPUT: _Phase10ReasonCategory.ADMISSION_REJECTED,
        _Phase8Reason.INVALID_REPAIR_BOUND: _Phase10ReasonCategory.LIMIT_EXCEEDED,
        _Phase8Reason.INVOCATION_INCONSISTENT: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _Phase8Reason.MISSING_BASELINE: _Phase10ReasonCategory.PREREQUISITE_BLOCKED,
        _Phase8Reason.NO_REPAIR_NEEDED: _Phase10ReasonCategory.PREREQUISITE_BLOCKED,
        _Phase8Reason.OPERATION_CORRELATION_MISMATCH: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _Phase8Reason.PREREQUISITE: _Phase10ReasonCategory.PREREQUISITE_BLOCKED,
        _Phase8Reason.REPAIR_EXHAUSTED: _Phase10ReasonCategory.VERIFICATION_FAILED,
        _Phase8Reason.REPAIR_MEMBERS: _Phase10ReasonCategory.VERIFICATION_FAILED,
        _Phase8Reason.REPAIR_RECONCILIATION: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _Phase8Reason.RETIRED_CORRELATION_TOKEN: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _Phase8Reason.REVISION_INVALID: _Phase10ReasonCategory.VERIFICATION_FAILED,
        _Phase8Reason.REVISION_LIMIT: _Phase10ReasonCategory.LIMIT_EXCEEDED,
        _Phase8Reason.REWRITE_MEMBERS: _Phase10ReasonCategory.VERIFICATION_FAILED,
        _Phase8Reason.REWRITE_RECONCILIATION: _Phase10ReasonCategory.EVIDENCE_INCONSISTENT,
        _Phase8Reason.ROUTE_NOT_SELECTED: _Phase10ReasonCategory.PREREQUISITE_BLOCKED,
        _Phase8Reason.TRANSPORT_LOST: _Phase10ReasonCategory.EXECUTION_LOST,
        _Phase8Reason.UNATTRIBUTABLE_PROVIDER_FAILURE: _Phase10ReasonCategory.BACKEND_FAILED,
    },
}


def _map_phase10_reason(value: object) -> _Phase10ReasonCategory:
    mapping = _PHASE10_REASON_TABLE.get(type(value))
    return (
        mapping.get(cast(Enum, value), _Phase10ReasonCategory.UNEXPECTED_FAILURE)
        if mapping
        else _Phase10ReasonCategory.UNEXPECTED_FAILURE
    )


def _unmapped_phase10_reason_values() -> tuple[Enum, ...]:
    return tuple(
        value for reason_type, mapping in _PHASE10_REASON_TABLE.items() for value in reason_type if value not in mapping
    )


E = TypeVar("E", bound=Enum)


class _Phase10AccountingPlanLike(Protocol):
    @property
    def datums(self) -> tuple[object, ...]: ...

    @property
    def dependencies(self) -> tuple[object, ...]: ...

    @property
    def task_predecessors(self) -> tuple[object, ...]: ...

    @property
    def stages(self) -> tuple[object, ...]: ...

    @property
    def tasks(self) -> tuple[object, ...]: ...

    @property
    def atomic_groups(self) -> tuple[object, ...]: ...


def _require_enum(value: object, expected: type[E]) -> None:
    if type(value) is not expected:
        raise TypeError("unknown private Phase 10 value")


@dataclass(slots=True, repr=False)
class _Phase10GrantState(_PrivatePhase10InspectionValue):
    owner_binding: _Phase10IdentityBinding | None
    subject_binding: _Phase10IdentityBinding | None
    active: bool = True


@dataclass(frozen=True, slots=True, repr=False)
class _Phase10IdentityBinding(_PrivatePhase10InspectionValue):
    weak_reference: ReferenceType[object] = field(compare=False)

    def matches(self, value: object) -> bool:
        return self.weak_reference() is value


@dataclass(slots=True, repr=False)
class _Phase10BuilderBudget(_PrivatePhase10InspectionValue):
    used_bytes: int = 0

    def __post_init__(self) -> None:
        if type(self.used_bytes) is not int or not 0 <= self.used_bytes <= _MAX_BUILDER_WORKING_BYTES:
            raise TypeError("invalid private Phase 10 builder budget")

    def reserve(self, size: object) -> bool:
        if type(size) is not int or size < 0 or size > _MAX_BUILDER_WORKING_BYTES - self.used_bytes:
            return False
        self.used_bytes += size
        return True


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
class _Phase10OwnerCapture(_PrivatePhase10InspectionValue):
    """Detached owner-issued input for one bounded inspect or diagnose call."""

    subject_kind: _Phase10SubjectKind
    semantic_profile_version: _Phase10SemanticProfile
    capture_boundary: _Phase10CaptureBoundary
    capture_lifecycle_state: _Phase10LifecycleState
    snapshot: _Phase10Snapshot
    diagnostics: tuple[_Phase10Diagnostic, ...] = ()

    def __post_init__(self) -> None:
        if not _valid_owner_capture(self):
            raise TypeError("invalid private Phase 10 owner capture")


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


_Phase10ExplainDetails: TypeAlias = tuple[
    _Phase10SubjectKind,
    _Phase10SemanticProfile,
    _Phase10Route,
    tuple[_Phase10Capability, ...],
    tuple[_Phase10Aggregate, ...],
    _Phase10ReasonCategory | None,
]


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
    try:
        if owner is None or subject is None or type(operation) is not _Phase10Operation:
            return _Phase10InspectionRejected(_Phase10RejectionCode.SUBJECT_INVALID)
        nonce = object()
        owner_binding = _phase10_identity_binding(owner, nonce)
        subject_binding = _phase10_identity_binding(subject, nonce)
        if owner_binding is None or subject_binding is None:
            return _Phase10InspectionRejected(_Phase10RejectionCode.SUBJECT_INVALID)
        grant = _Phase10InspectionGrant(operation, nonce, _GRANT_SEAL)
        finalize(grant, _expire_phase10_grant_nonce, nonce)
        with _GRANT_LOCK:
            _GRANT_STATES[nonce] = _Phase10GrantState(owner_binding, subject_binding)
        return grant
    except Exception:
        return _Phase10InspectionRejected(_Phase10RejectionCode.REDACTION_FAILED)


def _consume_phase10_inspection_grant(
    grant: object,
    owner: object,
    subject: object,
    operation: _Phase10Operation,
) -> bool:
    try:
        if type(grant) is not _Phase10InspectionGrant or grant._seal is not _GRANT_SEAL:
            return False
        with _GRANT_LOCK:
            state = _GRANT_STATES.get(grant._nonce)
            if not _grant_matches(grant, state, owner, subject, operation):
                return False
            del _GRANT_STATES[grant._nonce]
            _expire_grant_state(cast(_Phase10GrantState, state))
            return True
    except Exception:
        return False


def _revoke_phase10_inspection_grant(grant: object) -> None:
    try:
        if type(grant) is _Phase10InspectionGrant and grant._seal is _GRANT_SEAL:
            with _GRANT_LOCK:
                state = _GRANT_STATES.pop(grant._nonce, None)
            if state is not None:
                _expire_grant_state(state)
    except Exception:
        return


def _phase10_identity_binding(value: object, nonce: object) -> _Phase10IdentityBinding | None:
    try:
        weak_reference = ref(value, lambda _reference: _expire_phase10_grant_nonce(nonce))
    except TypeError:
        return None
    return _Phase10IdentityBinding(weak_reference)


def _expire_phase10_grant_nonce(nonce: object) -> None:
    try:
        with _GRANT_LOCK:
            state = _GRANT_STATES.pop(nonce, None)
        if state is not None:
            _expire_grant_state(state)
    except Exception:
        return


def _explain_phase10(
    owner: object,
    subject: object,
    grant: object,
) -> _Phase10ExplainView | _Phase10InspectionRejected:
    if not _consume_phase10_inspection_grant(grant, owner, subject, _Phase10Operation.EXPLAIN):
        return _Phase10InspectionRejected(_Phase10RejectionCode.DENIED)
    try:
        if _phase10_builder_budget(_MAX_EXPLAIN_BUILDER_ROWS) is None:
            return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
        details = _phase10_explain_details(subject)
        if isinstance(details, _Phase10InspectionRejected):
            return details
        subject_kind, profile, route, capabilities, aggregates, rejection = details
        return _Phase10ExplainView(
            _phase10_provenance(
                _Phase10ViewKind.EXPLAIN,
                subject_kind,
                profile,
                _Phase10CaptureBoundary.ADMISSION_TERMINAL,
                _Phase10LifecycleState.REJECTED if rejection is not None else _Phase10LifecycleState.TERMINAL,
            ),
            route,
            capabilities,
            _phase10_declared_limits(),
            aggregates,
            rejection,
        )
    except Exception:
        return _Phase10InspectionRejected(_Phase10RejectionCode.REDACTION_FAILED)


def _inspect_phase10(
    owner: object,
    subject: object,
    grant: object,
) -> _Phase10InspectView | _Phase10InspectionRejected:
    if not _consume_phase10_inspection_grant(grant, owner, subject, _Phase10Operation.INSPECT):
        return _Phase10InspectionRejected(_Phase10RejectionCode.DENIED)
    try:
        if _phase10_builder_budget(_MAX_INSPECT_BUILDER_ROWS) is None:
            return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
        if not _valid_owner_capture(subject):
            return _Phase10InspectionRejected(_Phase10RejectionCode.SUBJECT_INVALID)
        capture = cast(_Phase10OwnerCapture, subject)
        return _Phase10InspectView(
            _phase10_provenance(
                _Phase10ViewKind.INSPECT,
                capture.subject_kind,
                capture.semantic_profile_version,
                capture.capture_boundary,
                capture.capture_lifecycle_state,
            ),
            capture.snapshot,
        )
    except Exception:
        return _Phase10InspectionRejected(_Phase10RejectionCode.REDACTION_FAILED)


def _diagnose_phase10(
    owner: object,
    subject: object,
    grant: object,
) -> _Phase10DiagnoseView | _Phase10InspectionRejected:
    if not _consume_phase10_inspection_grant(grant, owner, subject, _Phase10Operation.DIAGNOSE):
        return _Phase10InspectionRejected(_Phase10RejectionCode.DENIED)
    try:
        if _phase10_builder_budget(_MAX_DIAGNOSE_BUILDER_ROWS) is None:
            return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
        admission = _phase10_admission_diagnostic(subject)
        if admission is not None:
            profile, diagnostic = admission
            return _Phase10DiagnoseView(
                _phase10_provenance(
                    _Phase10ViewKind.DIAGNOSE,
                    _Phase10SubjectKind.ADMISSION_REJECTION,
                    profile,
                    _Phase10CaptureBoundary.ADMISSION_TERMINAL,
                    _Phase10LifecycleState.REJECTED,
                ),
                (diagnostic,),
            )
        if _valid_owner_capture(subject):
            capture = cast(_Phase10OwnerCapture, subject)
            if not capture.diagnostics:
                return _Phase10InspectionRejected(_Phase10RejectionCode.STATE_UNAVAILABLE)
            return _Phase10DiagnoseView(
                _phase10_provenance(
                    _Phase10ViewKind.DIAGNOSE,
                    capture.subject_kind,
                    capture.semantic_profile_version,
                    capture.capture_boundary,
                    capture.capture_lifecycle_state,
                ),
                capture.diagnostics,
            )
        return _Phase10InspectionRejected(_Phase10RejectionCode.SUBJECT_INVALID)
    except Exception:
        return _Phase10InspectionRejected(_Phase10RejectionCode.REDACTION_FAILED)


def _phase10_provenance(
    view_kind: _Phase10ViewKind,
    subject_kind: _Phase10SubjectKind,
    profile: _Phase10SemanticProfile,
    boundary: _Phase10CaptureBoundary,
    lifecycle: _Phase10LifecycleState,
) -> _Phase10Provenance:
    return _Phase10Provenance(
        _Phase10InspectionSchemaVersion.V1,
        _Phase10ContractVersion.V1,
        view_kind,
        subject_kind,
        profile,
        _Phase10ImplementationProfile.PANDAS_RUNTIME_V1,
        boundary,
        lifecycle,
    )


def _phase10_declared_limits() -> tuple[_Phase10DeclaredLimit, ...]:
    values = (
        (_Phase10LimitName.SUBJECTS_PER_CALL, 1),
        (_Phase10LimitName.VIEWS_PER_CALL, 1),
        (_Phase10LimitName.MAX_STAGE_SUMMARIES, _MAX_STAGE_SUMMARIES),
        (_Phase10LimitName.MAX_TERMINAL_SUMMARY_ROWS, _MAX_TERMINAL_SUMMARIES),
        (_Phase10LimitName.MAX_DIAGNOSTIC_ENTRIES, _MAX_DIAGNOSTICS),
        (_Phase10LimitName.MAX_REASON_CODES_PER_DIAGNOSTIC_ENTRY, _MAX_REASON_CODES_PER_DIAGNOSTIC),
        (_Phase10LimitName.MAX_PROVENANCE_FIELDS, 8),
        (_Phase10LimitName.MAX_TOP_LEVEL_FIELDS, 12),
        (_Phase10LimitName.MAX_JSON_NESTING_DEPTH, 5),
        (_Phase10LimitName.MAX_ALLOWLISTED_STRING_UTF8_BYTES, 96),
        (_Phase10LimitName.MAX_CANONICAL_JSON_UTF8_BYTES, _MAX_CANONICAL_JSON_BYTES),
        (_Phase10LimitName.MAX_BUILDER_WORKING_BYTES, _MAX_BUILDER_WORKING_BYTES),
    )
    return tuple(_Phase10DeclaredLimit(name, value) for name, value in values)


def _phase10_count_bucket(value: object) -> _Phase10CountBucket | None:
    if type(value) is not int or value < 0:
        return None
    if value == 0:
        return _Phase10CountBucket.ZERO
    if value == 1:
        return _Phase10CountBucket.ONE
    if value <= 4:
        return _Phase10CountBucket.TWO_TO_FOUR
    if value <= 16:
        return _Phase10CountBucket.FIVE_TO_SIXTEEN
    if value <= 64:
        return _Phase10CountBucket.SEVENTEEN_TO_SIXTY_FOUR
    return _Phase10CountBucket.SIXTY_FIVE_PLUS


def _phase10_stage(value: object) -> _Phase10Stage | None:
    if type(value) is not str:
        return None
    direct = {stage.value: stage for stage in _Phase10Stage}
    aliases = {
        "protect": _Phase10Stage.TRANSFORM,
        "phase7-plan": _Phase10Stage.RESOLVE,
        "phase7-apply": _Phase10Stage.TRANSFORM,
        "phase8-group": _Phase10Stage.REWRITE,
        "phase8-qualification": _Phase10Stage.VERIFY,
        "phase8-compatibility": _Phase10Stage.VERIFY,
        "validate-baselines": _Phase10Stage.VALIDATE,
    }
    if value.removeprefix("evaluate-") in {"0", "1", "2", "3"} and value.startswith("evaluate-"):
        return _Phase10Stage.EVALUATE
    if value.removeprefix("repair-") in {"1", "2", "3"} and value.startswith("repair-"):
        return _Phase10Stage.REPAIR
    return direct.get(value, aliases.get(value))


def _phase10_builder_budget(rows: object) -> _Phase10BuilderBudget | None:
    if type(rows) is not int or rows < 0:
        return None
    budget = _phase10_new_builder_budget()
    if budget is None:
        return None
    for _ in range(rows):
        if not _phase10_reserve_builder_row(budget):
            return None
    return budget


def _phase10_new_builder_budget() -> _Phase10BuilderBudget | None:
    budget = _Phase10BuilderBudget()
    if not budget.reserve(_BUILDER_BASE_BYTES):
        return None
    return budget


def _phase10_reserve_builder_row(budget: object) -> bool:
    return type(budget) is _Phase10BuilderBudget and budget.reserve(_BUILDER_ROW_BYTES)


def _phase10_explain_details(
    subject: object,
) -> _Phase10ExplainDetails | _Phase10InspectionRejected:
    rejection = _phase10_admission_reason(subject)
    if rejection is not None:
        profile, category = rejection
        return (
            _Phase10SubjectKind.ADMISSION_REJECTION,
            profile,
            _Phase10Route.REJECTED,
            (),
            (),
            category,
        )
    if type(subject) is _AccountingPlan and _is_admitted_accounting_plan(subject):
        return _phase10_accounting_explain_details(
            subject,
            _Phase10SemanticProfile.TARGET_CONTEXT_V1,
            _Phase10Route.NDD,
            (_Phase10Capability.TERMINAL_ACCOUNTING,),
        )
    if type(subject) is _ContextPlan and _is_admitted_context_plan(subject):
        relationship_count = sum(len(projection.bindings) for projection in subject.projections)
        return _phase10_accounting_explain_details(
            subject.accounting,
            _Phase10SemanticProfile.TARGET_CONTEXT_V1,
            _Phase10Route.NDD,
            (_Phase10Capability.TERMINAL_ACCOUNTING, _Phase10Capability.BOUNDED_CONTEXT),
            relationship_count,
        )
    if type(subject) is _Phase6Plan and _is_admitted_phase6_plan(subject):
        return _phase10_phase6_explain_details(subject)
    if type(subject) is _Phase7Plan and _is_admitted_phase7_plan(subject):
        relationship_count = sum(len(manifest.relations) for manifest in subject.manifests)
        return _phase10_accounting_explain_details(
            subject.accounting,
            _Phase10SemanticProfile.SUBSTITUTE_V1,
            _Phase10Route.MIXED,
            (
                _Phase10Capability.TERMINAL_ACCOUNTING,
                _Phase10Capability.BOUNDED_CONTEXT,
                _Phase10Capability.ANCHORED_MENTIONS,
                _Phase10Capability.STABLE_SUBSTITUTE,
            ),
            relationship_count,
        )
    if type(subject) is _Phase8Plan and _is_admitted_phase8_plan(subject):
        return _phase10_phase8_explain_details(subject)
    return _Phase10InspectionRejected(_Phase10RejectionCode.SUBJECT_INVALID)


def _phase10_accounting_explain_details(
    plan: _Phase10AccountingPlanLike,
    profile: _Phase10SemanticProfile,
    route: _Phase10Route,
    capabilities: tuple[_Phase10Capability, ...],
    extra_relationships: int = 0,
) -> _Phase10ExplainDetails:
    return (
        _Phase10SubjectKind.ADMITTED_PLAN,
        profile,
        route,
        capabilities,
        _phase10_accounting_aggregates(plan, extra_relationships),
        None,
    )


def _phase10_phase6_explain_details(plan: _Phase6Plan) -> _Phase10ExplainDetails:
    profile = (
        _Phase10SemanticProfile.REDACT_V1
        if plan.profile_version is _Phase6ProfileVersion.REDACT_V1
        else _Phase10SemanticProfile.SUBSTITUTE_V1
    )
    relationships = sum(len(projection.bindings) for projection in plan.context.projections)
    return _phase10_accounting_explain_details(
        plan.accounting,
        profile,
        _Phase10Route.MIXED,
        (
            _Phase10Capability.TERMINAL_ACCOUNTING,
            _Phase10Capability.BOUNDED_CONTEXT,
            _Phase10Capability.ANCHORED_MENTIONS,
        ),
        relationships,
    )


def _phase10_phase8_explain_details(
    plan: _Phase8Plan,
) -> _Phase10ExplainDetails | _Phase10InspectionRejected:
    aggregates = _phase10_aggregates(
        (
            (_Phase10AggregateDimension.DATUMS, sum(len(group.members) for group in plan.groups)),
            (_Phase10AggregateDimension.GROUPS, len(plan.groups)),
            (_Phase10AggregateDimension.OPERATIONS, sum(len(group.operations.stages) for group in plan.groups)),
            (_Phase10AggregateDimension.REPAIRS, sum(group.operations.max_repairs for group in plan.groups)),
        )
    )
    if aggregates is None:
        return _Phase10InspectionRejected(_Phase10RejectionCode.REDACTION_FAILED)
    return (
        _Phase10SubjectKind.ADMITTED_PLAN,
        _Phase10SemanticProfile.GROUPED_REWRITE_V1,
        _Phase10Route.NDD,
        (
            _Phase10Capability.TERMINAL_ACCOUNTING,
            _Phase10Capability.BOUNDED_CONTEXT,
            _Phase10Capability.ANCHORED_MENTIONS,
            _Phase10Capability.STABLE_SUBSTITUTE,
            _Phase10Capability.GROUPED_REWRITE,
        ),
        aggregates,
        None,
    )


def _phase10_accounting_aggregates(
    plan: _Phase10AccountingPlanLike,
    extra_relationships: int = 0,
) -> tuple[_Phase10Aggregate, ...]:
    counts = (
        (_Phase10AggregateDimension.DATUMS, len(plan.datums)),
        (
            _Phase10AggregateDimension.RELATIONSHIPS,
            len(plan.dependencies) + len(plan.task_predecessors) + extra_relationships,
        ),
        (_Phase10AggregateDimension.STAGES, len(plan.stages)),
        (_Phase10AggregateDimension.TASKS, len(plan.tasks)),
        (_Phase10AggregateDimension.GROUPS, len(plan.atomic_groups)),
    )
    aggregates = _phase10_aggregates(counts)
    if aggregates is None:
        return ()
    return aggregates


def _phase10_aggregates(
    counts: tuple[tuple[_Phase10AggregateDimension, int], ...],
) -> tuple[_Phase10Aggregate, ...] | None:
    aggregates: list[_Phase10Aggregate] = []
    for dimension, count in counts:
        bucket = _phase10_count_bucket(count)
        if type(dimension) is not _Phase10AggregateDimension or bucket is None:
            return None
        aggregates.append(_Phase10Aggregate(dimension, bucket))
    return tuple(aggregates)


def _phase10_admission_reason(
    subject: object,
) -> tuple[_Phase10SemanticProfile, _Phase10ReasonCategory] | None:
    if type(subject) is _AccountingRejected:
        profile = _Phase10SemanticProfile.TARGET_CONTEXT_V1
        code = subject.code
    elif type(subject) is _ContextRejected:
        profile = _Phase10SemanticProfile.TARGET_CONTEXT_V1
        code = subject.code
    elif type(subject) is _Phase6Rejected:
        profile = _Phase10SemanticProfile.REDACT_V1
        code = subject.code
    elif type(subject) is _Phase7Rejected:
        profile = _Phase10SemanticProfile.SUBSTITUTE_V1
        code = subject.code
    elif type(subject) is _Phase8Rejected:
        profile = _Phase10SemanticProfile.GROUPED_REWRITE_V1
        code = subject.code
    else:
        return None
    return profile, _map_phase10_reason(code)


def _phase10_admission_diagnostic(
    subject: object,
) -> tuple[_Phase10SemanticProfile, _Phase10Diagnostic] | None:
    rejection = _phase10_admission_reason(subject)
    if rejection is None:
        return None
    profile, category = rejection
    return profile, _Phase10Diagnostic(
        _Phase10CaptureBoundary.ADMISSION_TERMINAL,
        _Phase10Stage.ADMISSION,
        _Phase10TerminalState.REJECTED,
        category,
        _Phase10CountBucket.ONE,
        _Phase10ReconciliationState.NOT_ENTERED,
        _Phase10CleanupState.NOT_ENTERED,
    )


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
        and state.owner_binding is not None
        and state.owner_binding.matches(owner)
        and state.subject_binding is not None
        and state.subject_binding.matches(subject)
    )


def _expire_grant_state(state: _Phase10GrantState) -> None:
    state.active = False
    state.owner_binding = None
    state.subject_binding = None


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
        },
        _Phase10ViewKind.DIAGNOSE: {
            _Phase10SubjectKind.ADMISSION_REJECTION,
            _Phase10SubjectKind.INVOCATION_SNAPSHOT,
            _Phase10SubjectKind.TERMINAL_RECEIPT,
            _Phase10SubjectKind.CLEANUP_RECEIPT,
        },
    }
    return subject in allowed[kind]


def _valid_stage_summary(value: object) -> bool:
    return (
        type(value) is _Phase10StageSummary
        and type(value.stage) is _Phase10Stage
        and type(value.lifecycle_state) is _Phase10LifecycleState
        and type(value.task_count_bucket) is _Phase10CountBucket
    )


def _valid_terminal_summary(value: object) -> bool:
    return (
        type(value) is _Phase10TerminalSummary
        and type(value.stage) is _Phase10Stage
        and type(value.terminal_state) is _Phase10TerminalState
        and type(value.impact_count_bucket) is _Phase10CountBucket
    )


def _valid_snapshot(value: object) -> bool:
    return (
        type(value) is _Phase10Snapshot
        and type(value.stage_summaries) is tuple
        and len(value.stage_summaries) <= _MAX_STAGE_SUMMARIES
        and all(_valid_stage_summary(item) for item in value.stage_summaries)
        and type(value.terminal_summaries) is tuple
        and len(value.terminal_summaries) <= _MAX_TERMINAL_SUMMARIES
        and all(_valid_terminal_summary(item) for item in value.terminal_summaries)
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


def _valid_owner_capture(value: object) -> bool:
    if not (
        type(value) is _Phase10OwnerCapture
        and type(value.subject_kind) is _Phase10SubjectKind
        and value.subject_kind
        in {
            _Phase10SubjectKind.INVOCATION_SNAPSHOT,
            _Phase10SubjectKind.TERMINAL_RECEIPT,
            _Phase10SubjectKind.CLEANUP_RECEIPT,
        }
        and type(value.semantic_profile_version) is _Phase10SemanticProfile
        and type(value.capture_boundary) is _Phase10CaptureBoundary
        and type(value.capture_lifecycle_state) is _Phase10LifecycleState
        and _valid_snapshot(value.snapshot)
        and type(value.diagnostics) is tuple
        and len(value.diagnostics) <= _MAX_DIAGNOSTICS
        and all(_valid_diagnostic(item) for item in value.diagnostics)
    ):
        return False
    boundary_lifecycle = {
        _Phase10CaptureBoundary.INVOCATION_OPENED: _Phase10LifecycleState.OPENED,
        _Phase10CaptureBoundary.PRE_DISPATCH: _Phase10LifecycleState.PRE_DISPATCH,
        _Phase10CaptureBoundary.POST_DISPATCH: _Phase10LifecycleState.POST_DISPATCH,
        _Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED: _Phase10LifecycleState.TERMINAL,
        _Phase10CaptureBoundary.PRE_REDUCTION_CLEANUP_TERMINAL: _Phase10LifecycleState.CLEANUP_TERMINAL,
        _Phase10CaptureBoundary.POST_REDUCTION_CLEANUP_TERMINAL: _Phase10LifecycleState.CLEANUP_TERMINAL,
        _Phase10CaptureBoundary.RELEASE_TERMINAL: _Phase10LifecycleState.RELEASE_TERMINAL,
        _Phase10CaptureBoundary.INVOCATION_CLOSED: _Phase10LifecycleState.CLOSED,
    }
    allowed_boundaries = {
        _Phase10SubjectKind.INVOCATION_SNAPSHOT: set(boundary_lifecycle),
        _Phase10SubjectKind.TERMINAL_RECEIPT: {
            _Phase10CaptureBoundary.TERMINAL_EVIDENCE_ACCEPTED,
            _Phase10CaptureBoundary.RELEASE_TERMINAL,
            _Phase10CaptureBoundary.INVOCATION_CLOSED,
        },
        _Phase10SubjectKind.CLEANUP_RECEIPT: {
            _Phase10CaptureBoundary.PRE_REDUCTION_CLEANUP_TERMINAL,
            _Phase10CaptureBoundary.POST_REDUCTION_CLEANUP_TERMINAL,
        },
    }
    stages = tuple(item.stage for item in value.snapshot.stage_summaries)
    terminals = tuple((item.stage, item.terminal_state) for item in value.snapshot.terminal_summaries)
    diagnostic_keys = tuple(
        (item.boundary, item.stage, item.terminal_state, item.reason_category) for item in value.diagnostics
    )
    return (
        boundary_lifecycle.get(value.capture_boundary) is value.capture_lifecycle_state
        and value.capture_boundary in allowed_boundaries[value.subject_kind]
        and len(stages) == len(set(stages))
        and len(terminals) == len(set(terminals))
        and len(diagnostic_keys) == len(set(diagnostic_keys))
        and all(
            item.boundary is value.capture_boundary
            and item.reconciliation_state is value.snapshot.reconciliation_state
            and item.cleanup_state is value.snapshot.cleanup_state
            and item.stage in stages
            and (item.stage, item.terminal_state) in terminals
            for item in value.diagnostics
        )
    )


def _unique_enum_fields(values: tuple[object, ...], field_name: str) -> bool:
    fields = tuple(getattr(value, field_name, None) for value in values)
    return None not in fields and len(fields) == len(set(fields))


def _valid_declared_limit(value: object) -> bool:
    return (
        type(value) is _Phase10DeclaredLimit
        and type(value.name) is _Phase10LimitName
        and type(value.value) is int
        and 0 <= value.value <= _MAX_DECLARED_INTEGER
    )


def _valid_aggregate(value: object) -> bool:
    return (
        type(value) is _Phase10Aggregate
        and type(value.dimension) is _Phase10AggregateDimension
        and type(value.count_bucket) is _Phase10CountBucket
    )


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
        and all(_valid_declared_limit(item) for item in value.declared_limits)
        and _unique_enum_fields(cast(tuple[object, ...], value.declared_limits), "name")
        and type(value.relationship_buckets) is tuple
        and all(_valid_aggregate(item) for item in value.relationship_buckets)
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
    try:
        payload = _view_payload(value)
    except Exception:
        return _Phase10InspectionRejected(_Phase10RejectionCode.REDACTION_FAILED)
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
    except Exception:
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
