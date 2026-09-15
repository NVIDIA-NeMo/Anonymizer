# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure symbolic oracle for the private Phase 10 bounded-inspection contract.

The model deliberately imports no production, graph/runtime, dataframe, provider,
adapter, or measurement code. It reduces closed symbolic observations only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import TypedDict, cast

REFERENCE_MODEL_VERSION = "phase10-bounded-inspection-reference-model/v1"
GENERATOR_VERSION = "phase10-bounded-inspection-finite-envelope/v1"
CORPUS_SCHEMA_VERSION = "phase10-bounded-inspection-reference-corpus/v1"
MANIFEST_SCHEMA_VERSION = "phase10-bounded-inspection-reference-manifest/v1"

OPERATIONS = ("explain", "inspect", "diagnose")
SUBJECTS = (
    "admitted_plan",
    "admission_rejection_receipt",
    "invocation_snapshot",
    "terminal_receipt",
    "cleanup_receipt",
)
CAPTURE_BOUNDARIES = (
    "admission_terminal",
    "invocation_opened",
    "pre_dispatch",
    "post_dispatch",
    "terminal_evidence_accepted",
    "pre_reduction_cleanup_terminal",
    "post_reduction_cleanup_terminal",
    "release_terminal",
    "invocation_closed",
)
LIFECYCLE_STATES = (
    "rejected",
    "opened",
    "pre_dispatch",
    "post_dispatch",
    "terminal",
    "cleanup_terminal",
    "release_terminal",
    "closed",
)
STAGES = (
    "admission",
    "detect",
    "augment",
    "validate",
    "finalize",
    "resolve",
    "classify",
    "transform",
    "verify",
    "analyze",
    "rewrite",
    "evaluate",
    "repair",
    "reconcile",
    "cleanup",
    "release",
    "publication",
)
TERMINALS = ("inconsistent", "lost", "cancelled", "failed", "blocked", "withheld", "rejected", "succeeded")
REASONS = (
    "admission_rejected",
    "capability_mismatch",
    "limit_exceeded",
    "verification_failed",
    "backend_failed",
    "cancellation_before_dispatch",
    "stop_acknowledged",
    "execution_lost",
    "prerequisite_blocked",
    "evidence_inconsistent",
    "cleanup_failed",
    "cleanup_unconfirmed",
    "publication_failed",
    "inspection_denied",
    "inspection_subject_invalid",
    "inspection_state_unavailable",
    "inspection_limit_exceeded",
    "inspection_redaction_failed",
    "inspection_encoding_failed",
    "unexpected_failure",
)
COUNT_BUCKETS = ("0", "1", "2-4", "5-16", "17-64", "65+")
BYTE_BUCKETS = ("0", "1-256", "257-4096", "4097-65536", "65537+")
SEMANTIC_PROFILES = (
    "target-context-v1",
    "phase6-redact-graph/v1",
    "anonymizer-phase7-stable-substitute/v1",
    "anonymizer-phase8-grouped-rewrite/v1",
)
ROUTES = ("local", "ndd", "mixed", "rejected")
CAPABILITIES = (
    "terminal_accounting",
    "bounded_context",
    "anchored_mentions",
    "stable_substitute",
    "grouped_rewrite",
)
AGGREGATE_DIMENSIONS = (
    "datums",
    "relationships",
    "stages",
    "tasks",
    "groups",
    "operations",
    "repairs",
    "outcomes",
    "retained_references",
)
RECONCILIATION_STATES = ("not_entered", "pending", "reconciled", "failed", "inconsistent")
CLEANUP_STATES = ("not_entered", "pending", "verified", "failed", "unconfirmed")
RELEASE_STATES = ("not_entered", "pending", "released", "withheld", "failed", "unconfirmed")
PROVENANCE_FIELDS = (
    "inspection_schema_version",
    "inspection_contract_version",
    "view_kind",
    "subject_kind",
    "semantic_profile_version",
    "implementation_profile_version",
    "capture_boundary",
    "capture_lifecycle_state",
)
LIMITS = (
    ("subjects_per_call", 1),
    ("views_per_call", 1),
    ("max_stage_summaries", 8),
    ("max_terminal_summary_rows", 48),
    ("max_diagnostic_entries", 64),
    ("max_reason_codes_per_diagnostic_entry", 4),
    ("max_provenance_fields", 8),
    ("max_top_level_fields", 12),
    ("max_json_nesting_depth", 5),
    ("max_allowlisted_string_utf8_bytes", 96),
    ("max_canonical_json_utf8_bytes", 16_384),
    ("max_builder_working_bytes", 65_536),
)
MUTATION_CLASSES = (
    "grant-validation-before-subject-access",
    "grant-forgery-reuse-expiry-operation-subject",
    "grant-authority-overclaim",
    "public-surface-exposure",
    "forbidden-rich-traversal",
    "retained-private-reference",
    "forbidden-value-leakage",
    "partial-or-truncated-output",
    "forbidden-sort-key",
    "permutation-byte-drift",
    "inspection-side-effect",
    "terminal-evidence-rewrite",
    "post-cleanup-revival",
    "private-serialization-or-auto-persistence",
    "noncanonical-json",
    "unmapped-owner-reason",
    "unexpected-failure-detail-leak",
    "p9-or-public-compatibility-drift",
    "ndd-boundary-bypass",
    "wheel-contract-omission",
)
REFERENCE_MUTATION_INSTANCES = (
    "authorization-bypass",
    "unknown-operation-acceptance",
    "operation-subject-union",
    "limit-bypass",
    "redaction-bypass",
    "synchronous-publication-rejected",
    "capture-coherence-bypass",
    "empty-diagnosis-publishes",
    "malformed-row-acceptance",
    "count-bucket-upper-edge",
    "byte-bucket-upper-edge",
    "terminal-precedence-inversion",
    "reason-order-inversion",
    "noncompact-corpus-json",
    "partial-rejection-payload",
    "noninterference-claim-dropped",
    "exact-limit-off-by-one",
    "payload-root-limit-bypass",
    "payload-provenance-limit-bypass",
    "payload-depth-limit-bypass",
    "payload-list-depth-limit-bypass",
    "payload-string-limit-bypass",
    "payload-key-string-limit-bypass",
    "canonical-byte-limit-bypass",
    "derived-root-measurement-corruption",
    "fixed-arity-request-counter-introduction",
    "closed-domain-admission-bypass",
    "measurement-schema-precedence-inversion",
    "admission-boundary-domain-bypass",
    "admission-lifecycle-domain-bypass",
    "admission-rejection-domain-bypass",
)
PRODUCTION_MUTATION_INSTANCES = (
    *MUTATION_CLASSES[:-1],
    "encoder-validator-invocation-bypass",
    "payload-root-limit-bypass",
    "payload-provenance-limit-bypass",
    "payload-dict-depth-limit-bypass",
    "payload-list-depth-limit-bypass",
    "payload-key-string-limit-bypass",
    "payload-value-string-limit-bypass",
    "payload-root-exact-built-in-bypass",
    "payload-schema-admission-bypass",
    "declared-limit-order-drift",
    "explain-subject-validation-order-inversion",
    "inspect-subject-validation-order-inversion",
    "diagnose-subject-validation-order-inversion",
    "canonical-byte-exact-limit-off-by-one",
    "fixed-arity-extra-parameter",
    "grant-exact-type-bypass",
    "grant-seal-bypass",
    "grant-state-type-bypass",
    "grant-active-state-bypass",
    "grant-operation-binding-bypass",
    "grant-subject-binding-bypass",
    "grant-consumption-invalidation-bypass",
    "grant-revocation-invalidation-bypass",
    "grant-expiry-invalidation-bypass",
    "fixed-arity-inspect-extra-parameter",
    "fixed-arity-diagnose-extra-parameter",
    "grant-missing-state-admission",
    "grant-operation-type-bypass",
    "grant-missing-owner-binding-admission",
    "grant-missing-subject-binding-admission",
    "phase10-retained-owner",
    "phase10-retained-subject",
    "phase10-retained-grant",
    "phase10-retained-content-state",
    "phase10-telemetry-emission",
    "owner-accounting-stage-table-freeze",
    "owner-accounting-terminal-table-freeze",
    "owner-accounting-diagnostic-table-freeze",
    "owner-accounting-reason-table-freeze",
    "owner-graph-stage-table-freeze",
    "owner-graph-terminal-table-freeze",
    "owner-graph-diagnostic-table-freeze",
    "owner-graph-reason-table-freeze",
    "owner-operation-stage-table-freeze",
    "owner-operation-terminal-table-freeze",
    "owner-operation-diagnostic-table-freeze",
    "owner-lifecycle-stage-table-freeze",
    "owner-lifecycle-terminal-table-freeze",
    "owner-lifecycle-diagnostic-table-freeze",
    "owner-lifecycle-reason-table-freeze",
    "declared-table-length-bypass",
    "declared-table-order-bypass",
    "declared-table-value-bypass",
    "declared-table-integer-bypass",
    "declared-table-boolean-bypass",
    "declared-table-name-type-bypass",
    "declared-table-list-type-bypass",
    "declared-table-entry-type-bypass",
    "declared-table-entry-keys-bypass",
    "full-encoder-root-limit-bypass",
    "full-encoder-provenance-limit-bypass",
    "full-encoder-depth-limit-bypass",
    "full-encoder-string-limit-bypass",
    MUTATION_CLASSES[-1],
)

_LIMITS = dict(LIMITS)
_BOUNDARY_LIFECYCLE = {
    "invocation_opened": "opened",
    "pre_dispatch": "pre_dispatch",
    "post_dispatch": "post_dispatch",
    "terminal_evidence_accepted": "terminal",
    "pre_reduction_cleanup_terminal": "cleanup_terminal",
    "post_reduction_cleanup_terminal": "cleanup_terminal",
    "release_terminal": "release_terminal",
    "invocation_closed": "closed",
}
_OPERATION_SUBJECTS = {
    "explain": {"admitted_plan", "admission_rejection_receipt"},
    "inspect": {"invocation_snapshot", "terminal_receipt", "cleanup_receipt"},
    "diagnose": {
        "admission_rejection_receipt",
        "invocation_snapshot",
        "terminal_receipt",
        "cleanup_receipt",
    },
}
_SUBJECT_BOUNDARIES = {
    "invocation_snapshot": set(_BOUNDARY_LIFECYCLE),
    "terminal_receipt": {"terminal_evidence_accepted", "release_terminal", "invocation_closed"},
    "cleanup_receipt": {"pre_reduction_cleanup_terminal", "post_reduction_cleanup_terminal"},
}


class _SubjectCapture(TypedDict):
    capture_boundary: str
    lifecycle_state: str


@dataclass(frozen=True, slots=True)
class ReferenceDiagnostic:
    boundary: str
    stage: str
    terminal_state: str
    reason_category: str
    impact_count_bucket: str = "1"
    reconciliation_state: str = "reconciled"
    cleanup_state: str = "not_entered"


@dataclass(frozen=True, slots=True)
class ReferenceCase:
    name: str
    operation: str = "inspect"
    grant_state: str = "valid"
    subject_kind: str = "invocation_snapshot"
    capture_boundary: str = "terminal_evidence_accepted"
    lifecycle_state: str = "terminal"
    stages: tuple[str, ...] = ("rewrite",)
    stage_lifecycle_states: tuple[str, ...] = ()
    stage_task_buckets: tuple[str, ...] = ()
    terminals: tuple[tuple[str, str, str], ...] = (("rewrite", "failed", "1"),)
    diagnostics: tuple[ReferenceDiagnostic, ...] = ()
    reconciliation_state: str = "reconciled"
    cleanup_state: str = "not_entered"
    release_state: str = "not_entered"
    semantic_profile_version: str = "anonymizer-phase8-grouped-rewrite/v1"
    route: str = "mixed"
    capabilities: tuple[str, ...] = ("grouped_rewrite",)
    aggregates: tuple[tuple[str, str], ...] = (("datums", "1"),)
    rejection_category: str | None = None
    builder_working_bytes: int = 4_096
    reasons_per_diagnostic: int = 1
    bucket_value: int | None = None
    byte_value: int | None = None
    payload_witness: str | None = None
    canonical_limit_delta: int | None = None
    unsafe_value: bool = False


@dataclass(frozen=True, slots=True)
class ReferencePayloadMeasurement:
    top_level_fields: int
    provenance_fields: int
    json_nesting_depth: int
    longest_string_utf8_bytes: int


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    decision: str
    rejection_code: str | None
    subject_accessed: bool
    view_kind: str | None
    count_bucket: str | None
    byte_bucket: str | None
    canonical_json: str | None
    payload_measurement: ReferencePayloadMeasurement | None = None
    canonical_json_utf8_bytes: int | None = None
    protection_unchanged: bool = True


def reduce_reference(case: ReferenceCase) -> ReferenceResult:
    """Reduce one bounded symbolic inspection observation."""
    count = count_bucket(case.bucket_value) if case.bucket_value is not None else None
    bytes_ = byte_bucket(case.byte_value) if case.byte_value is not None else None
    if case.grant_state != "valid":
        return _rejected("inspection_denied", False, count, bytes_)
    if case.operation not in OPERATIONS or case.subject_kind not in SUBJECTS:
        return _rejected("inspection_subject_invalid", True, count, bytes_)
    if case.subject_kind not in _OPERATION_SUBJECTS[case.operation]:
        return _rejected("inspection_subject_invalid", True, count, bytes_)
    if case.subject_kind in _SUBJECT_BOUNDARIES and not _valid_capture(case):
        return _rejected("inspection_subject_invalid", True, count, bytes_)
    if case.operation == "diagnose" and case.subject_kind != "admission_rejection_receipt" and not case.diagnostics:
        return _rejected("inspection_state_unavailable", True, count, bytes_)
    if not _within_construction_limits(case):
        return _rejected("inspection_limit_exceeded", True, count, bytes_)
    if not _valid_value_domains(case):
        return _rejected("inspection_redaction_failed", True, count, bytes_)
    if not _valid_rows(case):
        return _rejected("inspection_redaction_failed", True, count, bytes_)
    payload = (
        structural_payload_witness(case.payload_witness) if case.payload_witness is not None else _payload(case, count)
    )
    measured = measure_reference_payload(payload)
    if isinstance(measured, str):
        return _rejected(measured, True, count, bytes_)
    if case.unsafe_value:
        return _rejected("inspection_redaction_failed", True, count, bytes_, measured)
    if case.payload_witness is not None:
        return _rejected("inspection_redaction_failed", True, count, bytes_, measured)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    encoded_bytes = len(encoded.encode("utf-8"))
    canonical_limit = _LIMITS["max_canonical_json_utf8_bytes"]
    if case.canonical_limit_delta is not None:
        canonical_limit = encoded_bytes + case.canonical_limit_delta
    if encoded_bytes > canonical_limit:
        return _rejected(
            "inspection_limit_exceeded",
            True,
            count,
            bytes_,
            measured,
            encoded_bytes,
        )
    return ReferenceResult(
        "view",
        None,
        True,
        case.operation,
        count,
        bytes_,
        encoded,
        measured,
        encoded_bytes,
    )


def count_bucket(value: object) -> str | None:
    if type(value) is not int or value < 0:
        return None
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 4:
        return "2-4"
    if value <= 16:
        return "5-16"
    if value <= 64:
        return "17-64"
    return "65+"


def byte_bucket(value: object) -> str | None:
    if type(value) is not int or value < 0:
        return None
    if value == 0:
        return "0"
    if value <= 256:
        return "1-256"
    if value <= 4_096:
        return "257-4096"
    if value <= 65_536:
        return "4097-65536"
    return "65537+"


def _rejected(
    code: str,
    accessed: bool,
    count: str | None,
    bytes_: str | None,
    measurement: ReferencePayloadMeasurement | None = None,
    canonical_json_utf8_bytes: int | None = None,
) -> ReferenceResult:
    return ReferenceResult(
        "rejected",
        code,
        accessed,
        None,
        count,
        bytes_,
        None,
        measurement,
        canonical_json_utf8_bytes,
    )


def _within_construction_limits(case: ReferenceCase) -> bool:
    observed = {
        "max_stage_summaries": len(case.stages),
        "max_terminal_summary_rows": len(case.terminals),
        "max_diagnostic_entries": len(case.diagnostics),
        "max_reason_codes_per_diagnostic_entry": case.reasons_per_diagnostic,
        "max_builder_working_bytes": case.builder_working_bytes,
    }
    return all(type(value) is int and 0 <= value <= _LIMITS[name] for name, value in observed.items())


def measure_reference_payload(payload: object) -> ReferencePayloadMeasurement | str:
    """Measure one exact-built-in detached payload without production imports."""
    if type(payload) is not dict:
        return "inspection_redaction_failed"
    root = cast(dict[object, object], payload)
    if any(type(key) is not str for key in root):
        return "inspection_redaction_failed"
    if len(root) > _LIMITS["max_top_level_fields"]:
        return "inspection_limit_exceeded"
    provenance_fields = 0
    if "provenance" in root:
        provenance = root["provenance"]
        if type(provenance) is not dict:
            return "inspection_redaction_failed"
        provenance_fields = len(provenance)
        if provenance_fields > _LIMITS["max_provenance_fields"]:
            return "inspection_limit_exceeded"
    measured = _measure_payload_node(payload, 1)
    if isinstance(measured, str):
        return measured
    nesting_depth, longest_string = measured
    return ReferencePayloadMeasurement(len(root), provenance_fields, nesting_depth, longest_string)


def _measure_payload_node(value: object, container_depth: int) -> tuple[int, int] | str:
    if type(value) is dict:
        if container_depth > _LIMITS["max_json_nesting_depth"]:
            return "inspection_limit_exceeded"
        maximum_depth = container_depth
        longest_string = 0
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                return "inspection_redaction_failed"
            try:
                key_bytes = len(key.encode("utf-8"))
            except UnicodeEncodeError:
                return "inspection_redaction_failed"
            if key_bytes > _LIMITS["max_allowlisted_string_utf8_bytes"]:
                return "inspection_limit_exceeded"
            child_depth = container_depth + 1 if type(item) in {dict, list} else container_depth
            child = _measure_payload_node(item, child_depth)
            if isinstance(child, str):
                return child
            measured_depth, measured_string = child
            maximum_depth = max(maximum_depth, measured_depth)
            longest_string = max(longest_string, key_bytes, measured_string)
        return maximum_depth, longest_string
    if type(value) is list:
        if container_depth > _LIMITS["max_json_nesting_depth"]:
            return "inspection_limit_exceeded"
        maximum_depth = container_depth
        longest_string = 0
        for item in cast(list[object], value):
            child_depth = container_depth + 1 if type(item) in {dict, list} else container_depth
            child = _measure_payload_node(item, child_depth)
            if isinstance(child, str):
                return child
            measured_depth, measured_string = child
            maximum_depth = max(maximum_depth, measured_depth)
            longest_string = max(longest_string, measured_string)
        return maximum_depth, longest_string
    if type(value) is str:
        try:
            string_bytes = len(value.encode("utf-8"))
        except UnicodeEncodeError:
            return "inspection_redaction_failed"
        if string_bytes > _LIMITS["max_allowlisted_string_utf8_bytes"]:
            return "inspection_limit_exceeded"
        return 0, string_bytes
    if value is None or type(value) in {bool, int}:
        return 0, 0
    return "inspection_redaction_failed"


def structural_payload_witness(witness: str) -> dict[str, object]:
    if witness == "root-exact":
        return {f"k{index}": None for index in range(12)}
    if witness == "root-over":
        return {f"k{index}": None for index in range(13)}
    if witness == "provenance-exact":
        return {"provenance": {f"k{index}": None for index in range(8)}}
    if witness == "provenance-over":
        return {"provenance": {f"k{index}": None for index in range(9)}}
    if witness == "depth-exact":
        return {"a": {"b": {"c": {"d": {}}}}}
    if witness == "depth-over":
        return {"a": {"b": {"c": {"d": {"e": {}}}}}}
    if witness == "list-depth-exact":
        return {"a": [[[[]]]]}
    if witness == "list-depth-over":
        return {"a": [[[[[]]]]]}
    if witness == "string-exact":
        return {"value": "é" * 48}
    if witness == "string-over":
        return {"value": "é" * 48 + "x"}
    if witness == "key-string-exact":
        return {"é" * 48: None}
    if witness == "key-string-over":
        return {"é" * 48 + "x": None}
    raise ValueError("unknown structural payload witness")


def _valid_capture(case: ReferenceCase) -> bool:
    return (
        _BOUNDARY_LIFECYCLE.get(case.capture_boundary) == case.lifecycle_state
        and case.capture_boundary in _SUBJECT_BOUNDARIES[case.subject_kind]
    )


def _valid_value_domains(case: ReferenceCase) -> bool:
    aggregate_dimensions = tuple(dimension for dimension, _bucket in case.aggregates)
    return (
        case.semantic_profile_version in SEMANTIC_PROFILES
        and case.capture_boundary in CAPTURE_BOUNDARIES
        and case.lifecycle_state in LIFECYCLE_STATES
        and (case.rejection_category is None or case.rejection_category in REASONS)
        and case.route in ROUTES
        and all(capability in CAPABILITIES for capability in case.capabilities)
        and len(case.capabilities) == len(set(case.capabilities))
        and all(dimension in AGGREGATE_DIMENSIONS and bucket in COUNT_BUCKETS for dimension, bucket in case.aggregates)
        and len(aggregate_dimensions) == len(set(aggregate_dimensions))
        and case.reconciliation_state in RECONCILIATION_STATES
        and case.cleanup_state in CLEANUP_STATES
        and case.release_state in RELEASE_STATES
    )


def _valid_rows(case: ReferenceCase) -> bool:
    if any(stage not in STAGES for stage in case.stages) or len(case.stages) != len(set(case.stages)):
        return False
    if case.stage_lifecycle_states and (
        len(case.stage_lifecycle_states) != len(case.stages)
        or any(state not in LIFECYCLE_STATES for state in case.stage_lifecycle_states)
    ):
        return False
    if case.stage_task_buckets and (
        len(case.stage_task_buckets) != len(case.stages)
        or any(bucket not in COUNT_BUCKETS for bucket in case.stage_task_buckets)
    ):
        return False
    if any(
        stage not in case.stages or terminal not in TERMINALS or bucket not in COUNT_BUCKETS
        for stage, terminal, bucket in case.terminals
    ):
        return False
    if len(case.terminals) != len({(stage, terminal) for stage, terminal, _bucket in case.terminals}):
        return False
    keys: set[tuple[str, str, str, str]] = set()
    for item in case.diagnostics:
        key = (item.boundary, item.stage, item.terminal_state, item.reason_category)
        if (
            key in keys
            or item.boundary != case.capture_boundary
            or item.stage not in case.stages
            or (item.stage, item.terminal_state) not in {(stage, terminal) for stage, terminal, _ in case.terminals}
            or item.reason_category not in REASONS
            or item.impact_count_bucket not in COUNT_BUCKETS
            or item.reconciliation_state != case.reconciliation_state
            or item.cleanup_state != case.cleanup_state
        ):
            return False
        keys.add(key)
    return True


def _provenance(case: ReferenceCase) -> dict[str, str]:
    return {
        "inspection_schema_version": "phase10-bounded-inspection-view/v1",
        "inspection_contract_version": "anonymizer-phase10-bounded-inspection/v1",
        "view_kind": case.operation,
        "subject_kind": case.subject_kind,
        "semantic_profile_version": case.semantic_profile_version,
        "implementation_profile_version": "pandas-runtime-v1",
        "capture_boundary": case.capture_boundary,
        "capture_lifecycle_state": case.lifecycle_state,
    }


def _payload(case: ReferenceCase, bucket: str | None) -> dict[str, object]:
    common: dict[str, object] = {
        "inspection_schema_version": "phase10-bounded-inspection-view/v1",
        "view_kind": case.operation,
        "subject_kind": case.subject_kind,
        "provenance": _provenance(case),
    }
    if case.operation == "explain":
        common.update(
            {
                "route": case.route,
                "required_capabilities": list(case.capabilities),
                "declared_limits": [{"name": name, "value": value} for name, value in LIMITS],
                "relationship_buckets": [
                    {"dimension": dimension, "count_bucket": bucket if index == 0 and bucket else value}
                    for index, (dimension, value) in enumerate(case.aggregates)
                ],
                "rejection_category": case.rejection_category,
            }
        )
    elif case.operation == "inspect":
        stage_positions = {stage: index for index, stage in enumerate(case.stages)}
        terminals = sorted(
            case.terminals,
            key=lambda item: (stage_positions[item[0]], TERMINALS.index(item[1])),
        )
        common.update(
            {
                "stage_summaries": [
                    {
                        "stage": stage,
                        "lifecycle_state": (
                            case.stage_lifecycle_states[index] if case.stage_lifecycle_states else case.lifecycle_state
                        ),
                        "task_count_bucket": case.stage_task_buckets[index] if case.stage_task_buckets else "1",
                    }
                    for index, stage in enumerate(case.stages)
                ],
                "terminal_summaries": [
                    {"stage": stage, "terminal_state": terminal, "impact_count_bucket": impact}
                    for stage, terminal, impact in terminals
                ],
                "reconciliation_state": case.reconciliation_state,
                "cleanup_state": case.cleanup_state,
                "release_state": case.release_state,
            }
        )
    else:
        diagnostics = case.diagnostics
        if case.subject_kind == "admission_rejection_receipt" and not diagnostics:
            diagnostics = (
                ReferenceDiagnostic(
                    "admission_terminal",
                    "admission",
                    "rejected",
                    case.rejection_category or "admission_rejected",
                    reconciliation_state="not_entered",
                ),
            )
        stage_positions = {stage: index for index, stage in enumerate(case.stages)}
        diagnostics = tuple(
            sorted(
                diagnostics,
                key=lambda item: (
                    CAPTURE_BOUNDARIES.index(item.boundary),
                    stage_positions.get(item.stage, len(stage_positions)),
                    TERMINALS.index(item.terminal_state),
                    REASONS.index(item.reason_category),
                    COUNT_BUCKETS.index(item.impact_count_bucket),
                ),
            )
        )
        common["diagnostics"] = [asdict(item) for item in diagnostics]
    return common


def finite_reference_cases() -> tuple[ReferenceCase, ...]:
    cases: list[ReferenceCase] = []
    for operation in OPERATIONS:
        for subject in SUBJECTS:
            cases.append(_operation_subject_case(operation, subject))
    for boundary in tuple(_BOUNDARY_LIFECYCLE):
        for lifecycle in LIFECYCLE_STATES:
            cases.append(
                _diagnosed_case(
                    f"capture-{boundary}-{lifecycle}",
                    capture_boundary=boundary,
                    lifecycle_state=lifecycle,
                )
            )
    for subject, boundaries in _SUBJECT_BOUNDARIES.items():
        for boundary, lifecycle in _BOUNDARY_LIFECYCLE.items():
            cases.append(
                _diagnosed_case(
                    f"subject-boundary-{subject}-{boundary}",
                    subject_kind=subject,
                    capture_boundary=boundary,
                    lifecycle_state=lifecycle,
                )
            )
    for grant in ("forged", "reused", "expired", "wrong_operation", "wrong_subject"):
        cases.append(ReferenceCase(f"authorization-{grant}", grant_state=grant))
    for value in (-1, 0, 1, 2, 4, 5, 16, 17, 64, 65):
        cases.append(_explained_case(f"count-bucket-{value}", bucket_value=value))
    for value in (-1, 0, 1, 256, 257, 4_096, 4_097, 65_536, 65_537):
        cases.append(_explained_case(f"byte-bucket-{value}", byte_value=value))
    for name, maximum in LIMITS:
        if name in {"subjects_per_call", "views_per_call"}:
            continue
        cases.append(_limit_case(name, maximum, False))
        cases.append(_limit_case(name, maximum + 1, True))
    cases.extend(
        (
            _diagnosed_case(
                "limit-max_json_nesting_depth-list-exact",
                payload_witness="list-depth-exact",
            ),
            _diagnosed_case(
                "limit-max_json_nesting_depth-list-over",
                payload_witness="list-depth-over",
            ),
            _diagnosed_case(
                "limit-max_allowlisted_string_utf8_bytes-key-exact",
                payload_witness="key-string-exact",
            ),
            _diagnosed_case(
                "limit-max_allowlisted_string_utf8_bytes-key-over",
                payload_witness="key-string-over",
            ),
        )
    )
    for reason in REASONS:
        cases.append(_diagnosed_case(f"reason-{reason}", diagnostics=(_diagnostic(reason=reason),)))
    for terminal in TERMINALS:
        cases.append(
            _diagnosed_case(
                f"terminal-{terminal}",
                terminals=(("rewrite", terminal, "1"),),
                diagnostics=(_diagnostic(terminal=terminal),),
            )
        )
    cases.extend(
        (
            ReferenceCase("unknown-operation", operation="unknown"),
            ReferenceCase("unknown-subject", subject_kind="unknown"),
            ReferenceCase("unknown-stage", stages=("unknown",), terminals=()),
            ReferenceCase("duplicate-stage", stages=("rewrite", "rewrite"), terminals=()),
            ReferenceCase("unknown-terminal", terminals=(("rewrite", "unknown", "1"),)),
            ReferenceCase("unknown-semantic-profile", semantic_profile_version="unknown"),
            _explained_case("unknown-admission-capture-boundary", capture_boundary="unknown"),
            _explained_case("unknown-admission-lifecycle", lifecycle_state="unknown"),
            _explained_case(
                "unknown-explain-rejection-category",
                subject_kind="admission_rejection_receipt",
                lifecycle_state="rejected",
                rejection_category="unknown",
            ),
            replace(_admission_diagnosis("unknown-diagnose-rejection-category"), rejection_category="unknown"),
            _explained_case("unknown-route", route="unknown"),
            _explained_case("unknown-capability", capabilities=("unknown",)),
            _explained_case("unknown-aggregate-dimension", aggregates=(("unknown", "1"),)),
            _explained_case("unknown-aggregate-bucket", aggregates=(("datums", "unknown"),)),
            ReferenceCase("unknown-reconciliation", reconciliation_state="unknown"),
            ReferenceCase("unknown-cleanup", cleanup_state="unknown"),
            ReferenceCase("unknown-release", release_state="unknown"),
            ReferenceCase(
                "measurement-before-unsafe-schema",
                payload_witness="root-over",
                unsafe_value=True,
            ),
            ReferenceCase(
                "duplicate-terminal",
                terminals=(("rewrite", "failed", "1"), ("rewrite", "failed", "2-4")),
            ),
            _diagnosed_case(
                "diagnostic-boundary-mismatch",
                diagnostics=(_diagnostic(boundary="post_dispatch"),),
            ),
            _diagnosed_case(
                "diagnostic-unknown-reason",
                diagnostics=(_diagnostic(reason="unknown"),),
            ),
            ReferenceCase(
                "terminal-order",
                terminals=(("rewrite", "succeeded", "1"), ("rewrite", "failed", "1")),
            ),
            _diagnosed_case(
                "diagnostic-order",
                diagnostics=(
                    _diagnostic(reason="verification_failed"),
                    _diagnostic(reason="backend_failed"),
                ),
            ),
            ReferenceCase("redaction-unknown-value", unsafe_value=True),
            ReferenceCase("synchronous-return-publication"),
            _diagnosed_case("cleanup-failed", cleanup_state="failed", diagnostics=(_diagnostic(cleanup="failed"),)),
            _diagnosed_case(
                "publication-failed",
                capture_boundary="release_terminal",
                lifecycle_state="release_terminal",
                release_state="failed",
                diagnostics=(
                    _diagnostic(
                        boundary="release_terminal", stage="publication", terminal="failed", reason="publication_failed"
                    ),
                ),
                stages=("publication",),
                terminals=(("publication", "failed", "1"),),
            ),
            _diagnosed_case("diagnose-no-reasons", diagnostics=()),
        )
    )
    names = tuple(case.name for case in cases)
    if len(names) != len(set(names)):
        raise AssertionError("duplicate Phase 10 reference case")
    return tuple(cases)


def _operation_subject_case(operation: str, subject: str) -> ReferenceCase:
    name = f"operation-subject-{operation}-{subject}"
    if operation == "explain":
        return _explained_case(
            name,
            subject_kind=subject,
            lifecycle_state="rejected" if subject == "admission_rejection_receipt" else "terminal",
            rejection_category="admission_rejected" if subject == "admission_rejection_receipt" else None,
        )
    if operation == "diagnose":
        if subject == "admission_rejection_receipt":
            return _admission_diagnosis(name)
        return _diagnosed_case(
            name,
            subject_kind=subject,
            diagnostics=(_diagnostic(**_diagnostic_capture(subject)),),
            **_subject_capture(subject),
        )
    return ReferenceCase(name, operation=operation, subject_kind=subject, **_subject_capture(subject))


def _subject_capture(subject: str) -> _SubjectCapture:
    if subject == "cleanup_receipt":
        return {"capture_boundary": "pre_reduction_cleanup_terminal", "lifecycle_state": "cleanup_terminal"}
    return {"capture_boundary": "terminal_evidence_accepted", "lifecycle_state": "terminal"}


def _diagnostic_capture(subject: str) -> dict[str, str]:
    capture = _subject_capture(subject)
    return {"boundary": capture["capture_boundary"]}


def _explained_case(name: str, **changes: object) -> ReferenceCase:
    base = ReferenceCase(
        name,
        operation="explain",
        subject_kind="admitted_plan",
        capture_boundary="admission_terminal",
        lifecycle_state="terminal",
        stages=(),
        terminals=(),
    )
    return replace(base, **changes)


def _admission_diagnosis(name: str) -> ReferenceCase:
    return ReferenceCase(
        name,
        operation="diagnose",
        subject_kind="admission_rejection_receipt",
        capture_boundary="admission_terminal",
        lifecycle_state="rejected",
        stages=(),
        terminals=(),
        rejection_category="admission_rejected",
    )


def _diagnosed_case(name: str, **changes: object) -> ReferenceCase:
    if "diagnostics" not in changes:
        boundary = changes.get("capture_boundary", "terminal_evidence_accepted")
        cleanup = changes.get("cleanup_state", "not_entered")
        if type(boundary) is not str or type(cleanup) is not str:
            raise TypeError("invalid directed diagnosis")
        changes["diagnostics"] = (_diagnostic(boundary=boundary, cleanup=cleanup),)
    base = ReferenceCase(name, operation="diagnose", diagnostics=(_diagnostic(),))
    return replace(base, **changes)


def _diagnostic(
    *,
    boundary: str = "terminal_evidence_accepted",
    stage: str = "rewrite",
    terminal: str = "failed",
    reason: str = "backend_failed",
    cleanup: str = "not_entered",
) -> ReferenceDiagnostic:
    return ReferenceDiagnostic(boundary, stage, terminal, reason, cleanup_state=cleanup)


def _limit_case(name: str, observed: int, over: bool) -> ReferenceCase:
    changes: dict[str, object] = {}
    base = _diagnosed_case(f"limit-{name}-{'over' if over else 'exact'}")
    if name == "max_stage_summaries":
        base = ReferenceCase(f"limit-{name}-{'over' if over else 'exact'}")
        changes["stages"] = tuple(STAGES[8:16]) if not over else (*STAGES[8:16], "stage-over")
        changes["terminals"] = ()
    elif name == "max_terminal_summary_rows":
        base = ReferenceCase(f"limit-{name}-{'over' if over else 'exact'}")
        stages = STAGES[:6]
        changes["stages"] = stages
        rows = tuple((stage, terminal, "1") for stage in stages for terminal in TERMINALS)
        changes["terminals"] = rows if not over else (*rows, ("stage-over", "failed", "1"))
    elif name == "max_diagnostic_entries":
        stages = STAGES[:8]
        terminals = tuple((stage, terminal, "1") for stage in stages for terminal in TERMINALS[:6])
        first = tuple(
            ReferenceDiagnostic("terminal_evidence_accepted", stage, terminal, REASONS[index % len(REASONS)])
            for index, (stage, terminal, _impact) in enumerate(terminals)
        )
        second = tuple(
            replace(item, reason_category=REASONS[(index + 1) % len(REASONS)]) for index, item in enumerate(first[:16])
        )
        diagnostics = (*first, *second)
        changes.update(stages=stages, terminals=terminals, diagnostics=diagnostics)
        if over:
            changes["diagnostics"] = (*diagnostics, replace(diagnostics[0], reason_category=REASONS[2]))
    elif name == "max_reason_codes_per_diagnostic_entry":
        changes["reasons_per_diagnostic"] = observed
    elif name == "max_provenance_fields":
        changes["payload_witness"] = "provenance-over" if over else "provenance-exact"
    elif name == "max_top_level_fields":
        changes["payload_witness"] = "root-over" if over else "root-exact"
    elif name == "max_json_nesting_depth":
        changes["payload_witness"] = "depth-over" if over else "depth-exact"
    elif name == "max_allowlisted_string_utf8_bytes":
        changes["payload_witness"] = "string-over" if over else "string-exact"
    elif name == "max_canonical_json_utf8_bytes":
        changes["canonical_limit_delta"] = -1 if over else 0
    elif name == "max_builder_working_bytes":
        changes["builder_working_bytes"] = observed
    return replace(base, **changes)


def case_by_name(name: str) -> ReferenceCase:
    return next(case for case in finite_reference_cases() if case.name == name)


def canonical_corpus_bytes() -> bytes:
    records = [
        {"case": asdict(case), "result": asdict(reduce_reference(case))}
        for case in sorted(finite_reference_cases(), key=lambda item: item.name)
    ]
    return json.dumps(
        {
            "cases": records,
            "generator_version": GENERATOR_VERSION,
            "reference_model_version": REFERENCE_MODEL_VERSION,
            "schema_version": CORPUS_SCHEMA_VERSION,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def reference_manifest() -> dict[str, object]:
    cases = finite_reference_cases()
    results = tuple(reduce_reference(case) for case in cases)
    return {
        "actual_event_count": sum(_event_count(case) for case in cases),
        "byte_buckets": list(BYTE_BUCKETS),
        "canonical_serialization": "UTF8_compact_sorted_key_JSON_complete_corpus_no_trailing_newline",
        "canonical_trace_count": len(cases),
        "capture_matrix_case_count": sum(case.name.startswith("capture-") for case in cases),
        "case_count": len(cases),
        "corpus_sha256": hashlib.sha256(canonical_corpus_bytes()).hexdigest(),
        "count_buckets": list(COUNT_BUCKETS),
        "decision_counts": {
            decision: sum(result.decision == decision for result in results) for decision in ("view", "rejected")
        },
        "directed_case_count": sum(
            not case.name.startswith(("capture-", "subject-boundary-", "operation-subject-")) for case in cases
        ),
        "generator_version": GENERATOR_VERSION,
        "fixed_arity_evidence": [
            "private_operation_signatures_accept_owner_subject_grant",
            "private_operations_return_one_scalar_view_or_rejection",
        ],
        "limit_case_count": sum(case.name.startswith("limit-") for case in cases),
        "limits": {name: value for name, value in LIMITS},
        "max_canonical_view_utf8_bytes": max(
            (result.canonical_json_utf8_bytes or 0 for result in results),
            default=0,
        ),
        "max_payload_json_nesting_depth": max(
            (result.payload_measurement.json_nesting_depth for result in results if result.payload_measurement),
            default=0,
        ),
        "max_payload_provenance_fields": max(
            (result.payload_measurement.provenance_fields for result in results if result.payload_measurement),
            default=0,
        ),
        "max_payload_string_utf8_bytes": max(
            (result.payload_measurement.longest_string_utf8_bytes for result in results if result.payload_measurement),
            default=0,
        ),
        "max_payload_top_level_fields": max(
            (result.payload_measurement.top_level_fields for result in results if result.payload_measurement),
            default=0,
        ),
        "mutation_class_count": len(MUTATION_CLASSES),
        "mutation_classes": list(MUTATION_CLASSES),
        "operation_subject_case_count": sum(case.name.startswith("operation-subject-") for case in cases),
        "payload_witness_case_count": sum(case.payload_witness is not None for case in cases),
        "production_mutation_instance_count": len(PRODUCTION_MUTATION_INSTANCES),
        "production_mutation_instances": list(PRODUCTION_MUTATION_INSTANCES),
        "reference_model_version": REFERENCE_MODEL_VERSION,
        "reference_mutation_instance_count": len(REFERENCE_MUTATION_INSTANCES),
        "reference_mutation_instances": list(REFERENCE_MUTATION_INSTANCES),
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "subject_boundary_case_count": sum(case.name.startswith("subject-boundary-") for case in cases),
    }


def _event_count(case: ReferenceCase) -> int:
    return 1 + len(case.stages) + len(case.terminals) + len(case.diagnostics)
