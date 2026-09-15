# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from xml.etree import ElementTree as ET

import pytest

from tests.engine.execution.phase10_reference_model import MUTATION_CLASSES, PRODUCTION_MUTATION_INSTANCES

_Mutation = tuple[str, str, tuple[tuple[str, str], ...], str]
_MUTATIONS: tuple[_Mutation, ...] = (
    (
        "grant-validation-before-subject-access",
        "engine/execution/phase10_inspection.py",
        (
            (
                """    if not _consume_phase10_inspection_grant(grant, owner, subject, _Phase10Operation.INSPECT):
        return _Phase10InspectionRejected(_Phase10RejectionCode.DENIED)
    try:
""",
                """    if not _valid_owner_capture(subject):
        return _Phase10InspectionRejected(_Phase10RejectionCode.SUBJECT_INVALID)
    if not _consume_phase10_inspection_grant(grant, owner, subject, _Phase10Operation.INSPECT):
        return _Phase10InspectionRejected(_Phase10RejectionCode.DENIED)
    try:
""",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_denies_before_subject_snapshot_access",
    ),
    (
        "grant-forgery-reuse-expiry-operation-subject",
        "engine/execution/phase10_inspection.py",
        (
            (
                "and state.owner_binding.matches(owner)",
                "and True",
            ),
        ),
        "tests/engine/execution/test_phase10_production_mutations.py::test_grant_validation_rejects_each_invalid_binding_and_state[owner]",
    ),
    (
        "grant-authority-overclaim",
        "engine/execution/phase10_bounded_inspection_contract.json",
        (
            (
                '"global_enable_flag_or_environment_variable": false',
                '"global_enable_flag_or_environment_variable": true',
            ),
        ),
        "tests/engine/execution/test_phase10_contract.py::test_phase10_contract_resource_has_exact_approved_raw_and_member_digests",
    ),
    (
        "public-surface-exposure",
        "__init__.py",
        (
            (
                "DEFAULT_ENTITY_LABELS: tuple[str, ...] = tuple(_DEFAULT_ENTITY_LABELS)",
                "DEFAULT_ENTITY_LABELS: tuple[str, ...] = tuple(_DEFAULT_ENTITY_LABELS)\nInspectionView = object()",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_types_are_not_exposed_through_public_surfaces",
    ),
    (
        "forbidden-rich-traversal",
        "engine/execution/phase10_inspection.py",
        (
            (
                """    if _valid_explain_view(value):
        return _explain_payload(cast(_Phase10ExplainView, value))
""",
                """    if type(value) is _Phase10ExplainView:
        return vars(value)
""",
            ),
        ),
        "tests/engine/execution/test_phase10_reference_conformance.py::test_every_admitted_reference_payload_matches_production_canonical_serializer[operation-subject-explain-admitted_plan]",
    ),
    (
        "retained-private-reference",
        "engine/execution/phase10_inspection.py",
        (
            (
                "    return _Phase10CanonicalEncoding(encoded, _ENCODING_SEAL)",
                '    globals()["_retained_phase10_view"] = value\n    return _Phase10CanonicalEncoding(encoded, _ENCODING_SEAL)',
            ),
        ),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_all_call_lifetimes_release_owner_subject_content_grant_and_view[successful-accounting-plan-explain]",
    ),
    (
        "forbidden-value-leakage",
        "engine/execution/phase10_inspection.py",
        (('"route": value.route.value,', '"route": value.route.value,\n        "_private_identity": id(value),'),),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_encoded_json_contains_only_exact_allowlisted_keys_and_scalar_types",
    ),
    (
        "partial-or-truncated-output",
        "engine/execution/phase10_inspection.py",
        (
            (
                """    if len(encoded) > _phase10_limit(_Phase10LimitName.MAX_CANONICAL_JSON_UTF8_BYTES):
        return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
    return _Phase10CanonicalEncoding(encoded, _ENCODING_SEAL)
""",
                """    if len(encoded) > _phase10_limit(_Phase10LimitName.MAX_CANONICAL_JSON_UTF8_BYTES):
        return _Phase10CanonicalEncoding(
            encoded[:_phase10_limit(_Phase10LimitName.MAX_CANONICAL_JSON_UTF8_BYTES)],
            _ENCODING_SEAL,
        )
    return _Phase10CanonicalEncoding(encoded, _ENCODING_SEAL)
""",
            ),
        ),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_oversize_encoding_returns_no_prefix_suffix_or_partial_payload",
    ),
    (
        "forbidden-sort-key",
        "engine/execution/phase10_inspection.py",
        (("sort_keys=True,", "sort_keys=False,"),),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_encoder_emits_exact_canonical_json_and_preserves_semantic_array_order",
    ),
    (
        "permutation-byte-drift",
        "engine/execution/phase10_inspection.py",
        (
            (
                '"diagnostics": [_diagnostic_payload(item) for item in value.diagnostics],',
                '"diagnostics": [_diagnostic_payload(item) for item in reversed(value.diagnostics)],',
            ),
        ),
        "tests/engine/execution/test_phase10_reference_conformance.py::test_every_admitted_reference_payload_matches_production_canonical_serializer[diagnostic-order]",
    ),
    (
        "inspection-side-effect",
        "engine/execution/accounting_ledger.py",
        (
            (
                "return self._phase10_build_capture()",
                "self._closed = True\n        return self._phase10_build_capture()",
            ),
        ),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_snapshot_is_observational_and_does_not_mutate_owning_ledger",
    ),
    (
        "terminal-evidence-rewrite",
        "engine/execution/phase8_runtime.py",
        (
            (
                """    def succeed(self, stage: _Phase8Stage) -> bool:
        if stage not in self.plan.stages or stage in self._terminals:
            return False
""",
                """    def succeed(self, stage: _Phase8Stage) -> bool:
        if stage not in self.plan.stages:
            return False
""",
            ),
            (
                """    def _close(self, stage: _Phase8Stage, terminal: _Phase8Terminal) -> bool:
        if stage in self._terminals:
            return False
""",
                """    def _close(self, stage: _Phase8Stage, terminal: _Phase8Terminal) -> bool:
        if False and stage in self._terminals:
            return False
""",
            ),
        ),
        "tests/engine/execution/test_phase8_operation_ledger.py::test_terminal_is_absorbing_and_failure_blocks_descendants",
    ),
    (
        "post-cleanup-revival",
        "engine/execution/phase8_cleanup.py",
        (
            (
                '"""Issue a detached bounded view of one sealed cleanup terminal."""',
                '"""Issue a detached bounded view of one sealed cleanup terminal."""\n        repr(self.identity)',
            ),
        ),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_post_cleanup_snapshot_never_reads_or_revives_cleanup_identity",
    ),
    (
        "private-serialization-or-auto-persistence",
        "engine/execution/phase10_inspection.py",
        (
            (
                'raise TypeError("private Phase 10 inspection values are not serializable")',
                "return (dict, ())",
            ),
        ),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_private_inputs_views_and_encodings_mask_repr_and_reject_pickle",
    ),
    (
        "noncanonical-json",
        "engine/execution/phase10_inspection.py",
        (('separators=(",", ":"),', 'separators=(", ", ": "),'),),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_encoder_emits_exact_canonical_json_and_preserves_semantic_array_order",
    ),
    (
        "unmapped-owner-reason",
        "engine/execution/phase10_inspection.py",
        (
            (
                "_CauseCode.KNOWN_FAILURE: _Phase10ReasonCategory.BACKEND_FAILED,",
                "_CauseCode.KNOWN_FAILURE: _Phase10ReasonCategory.UNEXPECTED_FAILURE,",
            ),
        ),
        "tests/engine/execution/test_phase10_reference_conformance.py::test_known_owner_reason_mapping_matches_the_oracle",
    ),
    (
        "unexpected-failure-detail-leak",
        "engine/execution/phase10_inspection.py",
        (
            (
                """    except Exception:
        return _Phase10InspectionRejected(_Phase10RejectionCode.ENCODING_FAILED)
""",
                """    except Exception as error:
        return _Phase10CanonicalEncoding(str(error).encode("utf-8"), _ENCODING_SEAL)
""",
            ),
        ),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_encoding_failure_is_cause_free_and_returns_no_partial_bytes",
    ),
    (
        "p9-or-public-compatibility-drift",
        "interface/result_compatibility_contract.json",
        (('"version": "result-compatibility-v1"', '"version": "result-compatibility-v2"'),),
        "tests/engine/execution/test_phase10_production_mutations.py::test_imported_package_p9_contract_witness",
    ),
    (
        "ndd-boundary-bypass",
        "engine/execution/phase8_ndd_backend.py",
        (("self._adapter.run_workflow(", "self._adapter.preview("),),
        "tests/engine/execution/test_phase10_production_mutations.py::test_imported_package_ndd_boundary_witness",
    ),
)

_ENFORCEMENT_MUTATIONS: tuple[_Mutation, ...] = (
    (
        "encoder-validator-invocation-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                """    validation = _validate_phase10_payload(payload)
    if type(validation) is _Phase10InspectionRejected:
        return validation
""",
                """    validation = None
    if type(validation) is _Phase10InspectionRejected:
        return validation
""",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_every_encoder_variant_invokes_validator_and_enforces_actual_byte_boundary",
    ),
    (
        "payload-root-limit-bypass",
        "engine/execution/phase10_inspection.py",
        (("if top_level_fields > _phase10_limit(_Phase10LimitName.MAX_TOP_LEVEL_FIELDS):", "if False:"),),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_payload_measurement_enforces_exact_and_one_over_boundaries",
    ),
    (
        "payload-provenance-limit-bypass",
        "engine/execution/phase10_inspection.py",
        (("if provenance_fields > _phase10_limit(_Phase10LimitName.MAX_PROVENANCE_FIELDS):", "if False:"),),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_payload_measurement_enforces_exact_and_one_over_boundaries",
    ),
    (
        "payload-dict-depth-limit-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                """    if type(value) is dict:
        if container_depth > _phase10_limit(_Phase10LimitName.MAX_JSON_NESTING_DEPTH):
""",
                """    if type(value) is dict:
        if False:
""",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_payload_measurement_enforces_exact_and_one_over_boundaries",
    ),
    (
        "payload-list-depth-limit-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                """    if type(value) is list:
        if container_depth > _phase10_limit(_Phase10LimitName.MAX_JSON_NESTING_DEPTH):
""",
                """    if type(value) is list:
        if False:
""",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_payload_measurement_enforces_exact_and_one_over_boundaries",
    ),
    (
        "payload-key-string-limit-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                "if key_bytes > _phase10_limit(_Phase10LimitName.MAX_ALLOWLISTED_STRING_UTF8_BYTES):",
                "if False:",
            ),
        ),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_payload_measurement_enforces_each_key_utf8_boundary_without_partial_output",
    ),
    (
        "payload-value-string-limit-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                "if string_bytes > _phase10_limit(_Phase10LimitName.MAX_ALLOWLISTED_STRING_UTF8_BYTES):",
                "if False:",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_payload_measurement_enforces_exact_and_one_over_boundaries",
    ),
    (
        "payload-root-exact-built-in-bypass",
        "engine/execution/phase10_inspection.py",
        (("if type(payload) is not dict:", "if isinstance(payload, dict):"),),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_payload_measurement_rejects_non_exact_builtins_without_traversal",
    ),
    (
        "payload-schema-admission-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                "if not (_valid_explain_payload(payload) or _valid_inspect_payload(payload) or _valid_diagnose_payload(payload)):",
                "if False:",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_encoder_applies_measurement_before_exact_schema_admission",
    ),
    (
        "declared-limit-order-drift",
        "engine/execution/phase10_inspection.py",
        (
            (
                "return tuple(_Phase10DeclaredLimit(name, value) for name, value in _PHASE10_LIMITS.items())",
                "return tuple(_Phase10DeclaredLimit(name, value) for name, value in reversed(_PHASE10_LIMITS.items()))",
            ),
        ),
        "tests/engine/execution/test_phase10_reference_conformance.py::test_named_real_owner_registry_matches_reference_decisions_provenance_rows_limits_and_bytes",
    ),
    (
        "explain-subject-validation-order-inversion",
        "engine/execution/phase10_inspection.py",
        (
            (
                """        if not _valid_explain_subject(subject):
            return _Phase10InspectionRejected(_Phase10RejectionCode.SUBJECT_INVALID)
        if _phase10_builder_budget(_MAX_EXPLAIN_BUILDER_ROWS) is None:
            return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
""",
                """        if _phase10_builder_budget(_MAX_EXPLAIN_BUILDER_ROWS) is None:
            return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
        if not _valid_explain_subject(subject):
            return _Phase10InspectionRejected(_Phase10RejectionCode.SUBJECT_INVALID)
""",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_subject_validation_precedes_builder_budget[_explain_phase10-EXPLAIN]",
    ),
    (
        "inspect-subject-validation-order-inversion",
        "engine/execution/phase10_inspection.py",
        (
            (
                """        if not _valid_owner_capture(subject):
            return _Phase10InspectionRejected(_Phase10RejectionCode.SUBJECT_INVALID)
        if _phase10_builder_budget(_MAX_INSPECT_BUILDER_ROWS) is None:
            return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
""",
                """        if _phase10_builder_budget(_MAX_INSPECT_BUILDER_ROWS) is None:
            return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
        if not _valid_owner_capture(subject):
            return _Phase10InspectionRejected(_Phase10RejectionCode.SUBJECT_INVALID)
""",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_subject_validation_precedes_builder_budget[_inspect_phase10-INSPECT]",
    ),
    (
        "diagnose-subject-validation-order-inversion",
        "engine/execution/phase10_inspection.py",
        (
            (
                """        if not _valid_owner_capture(subject):
            return _Phase10InspectionRejected(_Phase10RejectionCode.SUBJECT_INVALID)
        capture = cast(_Phase10OwnerCapture, subject)
        if not capture.diagnostics:
""",
                """        if _phase10_builder_budget(_MAX_DIAGNOSE_BUILDER_ROWS) is None:
            return _Phase10InspectionRejected(_Phase10RejectionCode.LIMIT_EXCEEDED)
        if not _valid_owner_capture(subject):
            return _Phase10InspectionRejected(_Phase10RejectionCode.SUBJECT_INVALID)
        capture = cast(_Phase10OwnerCapture, subject)
        if not capture.diagnostics:
""",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_subject_validation_precedes_builder_budget[_diagnose_phase10-DIAGNOSE]",
    ),
    (
        "canonical-byte-exact-limit-off-by-one",
        "engine/execution/phase10_inspection.py",
        (
            (
                "if len(encoded) > _phase10_limit(_Phase10LimitName.MAX_CANONICAL_JSON_UTF8_BYTES):",
                "if len(encoded) >= _phase10_limit(_Phase10LimitName.MAX_CANONICAL_JSON_UTF8_BYTES):",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_every_encoder_variant_invokes_validator_and_enforces_actual_byte_boundary",
    ),
    (
        "fixed-arity-extra-parameter",
        "engine/execution/phase10_inspection.py",
        (
            (
                """def _explain_phase10(
    owner: object,
    subject: object,
    grant: object,
) -> _Phase10ExplainView | _Phase10InspectionRejected:
""",
                """def _explain_phase10(
    owner: object,
    subject: object,
    grant: object,
    extra_subject: object | None = None,
) -> _Phase10ExplainView | _Phase10InspectionRejected:
""",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_operations_remain_fixed_arity_scalar_calls",
    ),
)

_GRANT_AND_ARITY_MUTATIONS: tuple[_Mutation, ...] = (
    (
        "grant-exact-type-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                "if type(grant) is not _Phase10InspectionGrant or grant._seal is not _GRANT_SEAL:",
                "if grant._seal is not _GRANT_SEAL:",
            ),
        ),
        "tests/engine/execution/test_phase10_production_mutations.py::test_grant_validation_rejects_each_invalid_binding_and_state[type]",
    ),
    (
        "grant-seal-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                "if type(grant) is not _Phase10InspectionGrant or grant._seal is not _GRANT_SEAL:",
                "if type(grant) is not _Phase10InspectionGrant:",
            ),
        ),
        "tests/engine/execution/test_phase10_production_mutations.py::test_grant_validation_rejects_each_invalid_binding_and_state[seal]",
    ),
    (
        "grant-state-type-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                "type(state) is _Phase10GrantState\n        and type(operation)",
                "state is not None\n        and type(operation)",
            ),
        ),
        "tests/engine/execution/test_phase10_production_mutations.py::test_grant_validation_rejects_each_invalid_binding_and_state[state_type]",
    ),
    (
        "grant-active-state-bypass",
        "engine/execution/phase10_inspection.py",
        (("and state.active\n", "and True\n"),),
        "tests/engine/execution/test_phase10_production_mutations.py::test_grant_validation_rejects_each_invalid_binding_and_state[inactive]",
    ),
    (
        "grant-operation-binding-bypass",
        "engine/execution/phase10_inspection.py",
        (("and grant.operation is operation", "and True"),),
        "tests/engine/execution/test_phase10_production_mutations.py::test_grant_validation_rejects_each_invalid_binding_and_state[operation]",
    ),
    (
        "grant-subject-binding-bypass",
        "engine/execution/phase10_inspection.py",
        (("and state.subject_binding.matches(subject)", "and True"),),
        "tests/engine/execution/test_phase10_production_mutations.py::test_grant_validation_rejects_each_invalid_binding_and_state[subject]",
    ),
    (
        "grant-consumption-invalidation-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                "            del _GRANT_STATES[grant._nonce]\n            _expire_grant_state(cast(_Phase10GrantState, state))",
                "            pass",
            ),
        ),
        "tests/engine/execution/test_phase10_production_mutations.py::test_grant_validation_rejects_each_invalid_binding_and_state[reused]",
    ),
    (
        "grant-revocation-invalidation-bypass",
        "engine/execution/phase10_inspection.py",
        (("state = _GRANT_STATES.pop(grant._nonce, None)", "state = None"),),
        "tests/engine/execution/test_phase10_production_mutations.py::test_grant_validation_rejects_each_invalid_binding_and_state[revoked]",
    ),
    (
        "grant-expiry-invalidation-bypass",
        "engine/execution/phase10_inspection.py",
        (("state = _GRANT_STATES.pop(nonce, None)", "state = None"),),
        "tests/engine/execution/test_phase10_production_mutations.py::test_grant_validation_rejects_each_invalid_binding_and_state[expired]",
    ),
    (
        "fixed-arity-inspect-extra-parameter",
        "engine/execution/phase10_inspection.py",
        (
            (
                "def _inspect_phase10(\n    owner: object,\n    subject: object,\n    grant: object,\n)",
                "def _inspect_phase10(\n    owner: object,\n    subject: object,\n    grant: object,\n    extra_subject: object | None = None,\n)",
            ),
        ),
        "tests/engine/execution/test_phase10_production_mutations.py::test_each_operation_has_exact_scalar_signature[inspect]",
    ),
    (
        "fixed-arity-diagnose-extra-parameter",
        "engine/execution/phase10_inspection.py",
        (
            (
                "def _diagnose_phase10(\n    owner: object,\n    subject: object,\n    grant: object,\n)",
                "def _diagnose_phase10(\n    owner: object,\n    subject: object,\n    grant: object,\n    extra_subject: object | None = None,\n)",
            ),
        ),
        "tests/engine/execution/test_phase10_production_mutations.py::test_each_operation_has_exact_scalar_signature[diagnose]",
    ),
)

_GRANT_STATE_MUTATIONS: tuple[_Mutation, ...] = (
    (
        "grant-missing-state-admission",
        "engine/execution/phase10_inspection.py",
        (
            (
                "            state = _GRANT_STATES.get(grant._nonce)\n",
                "            state = _GRANT_STATES.get(grant._nonce)\n            if state is None:\n                return True\n",
            ),
        ),
        "tests/engine/execution/test_phase10_production_mutations.py::test_grant_validation_rejects_each_invalid_binding_and_state[missing_state]",
    ),
    (
        "grant-operation-type-bypass",
        "engine/execution/phase10_inspection.py",
        (("and type(operation) is _Phase10Operation", "and True"),),
        "tests/engine/execution/test_phase10_production_mutations.py::test_grant_validation_rejects_each_invalid_binding_and_state[operation_type]",
    ),
    (
        "grant-missing-owner-binding-admission",
        "engine/execution/phase10_inspection.py",
        (
            (
                "and state.owner_binding is not None\n        and state.owner_binding.matches(owner)",
                "and (state.owner_binding is None or state.owner_binding.matches(owner))",
            ),
        ),
        "tests/engine/execution/test_phase10_production_mutations.py::test_grant_validation_rejects_each_invalid_binding_and_state[missing_owner_binding]",
    ),
    (
        "grant-missing-subject-binding-admission",
        "engine/execution/phase10_inspection.py",
        (
            (
                "and state.subject_binding is not None\n        and state.subject_binding.matches(subject)",
                "and (state.subject_binding is None or state.subject_binding.matches(subject))",
            ),
        ),
        "tests/engine/execution/test_phase10_production_mutations.py::test_grant_validation_rejects_each_invalid_binding_and_state[missing_subject_binding]",
    ),
)

_PRIVACY_MUTATIONS: tuple[_Mutation, ...] = (
    (
        "phase10-retained-owner",
        "engine/execution/phase10_inspection.py",
        (("        return grant\n", '        globals()["_retained_phase10_owner"] = owner\n        return grant\n'),),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_all_call_lifetimes_release_owner_subject_content_grant_and_view[active-accounting-plan-explain]",
    ),
    (
        "phase10-retained-subject",
        "engine/execution/phase10_inspection.py",
        (
            (
                "        return grant\n",
                '        globals()["_retained_phase10_subject"] = subject\n        return grant\n',
            ),
        ),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_all_call_lifetimes_release_owner_subject_content_grant_and_view[active-accounting-plan-explain]",
    ),
    (
        "phase10-retained-grant",
        "engine/execution/phase10_inspection.py",
        (("        return grant\n", '        globals()["_retained_phase10_grant"] = grant\n        return grant\n'),),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_all_call_lifetimes_release_owner_subject_content_grant_and_view[active-accounting-plan-explain]",
    ),
    (
        "phase10-retained-content-state",
        "engine/execution/phase10_inspection.py",
        (
            (
                "    if type(subject) is _AccountingPlan and _is_admitted_accounting_plan(subject):\n",
                '    if type(subject) is _AccountingPlan and _is_admitted_accounting_plan(subject):\n        globals()["_retained_phase10_content"] = subject.datums[0]\n',
            ),
        ),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_all_call_lifetimes_release_owner_subject_content_grant_and_view[successful-accounting-plan-explain]",
    ),
    (
        "phase10-telemetry-emission",
        "engine/execution/phase10_inspection.py",
        (
            (
                "        payload = _view_payload(value)",
                "        from anonymizer.telemetry import TelemetryHandler\n        TelemetryHandler.enqueue(None, None)\n        payload = _view_payload(value)",
            ),
        ),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_failure_paths_leave_real_protection_state_unchanged_and_silent[success-accounting-plan-explain]",
    ),
)

_TABLE_MUTATIONS: tuple[_Mutation, ...] = (
    (
        "owner-accounting-stage-table-freeze",
        "engine/execution/accounting_ledger.py",
        (
            (
                "            _Phase10TerminalSummary,\n        )\n\n",
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n"
                "        _current_limit = _phase10_limit\n"
                "\n"
                "        def _phase10_limit(name):\n"
                "            return 8 if name is _Phase10LimitName.MAX_STAGE_SUMMARIES else _current_limit(name)\n"
                "\n",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_owner_capture_uses_authoritative_row_limits[stage-accounting-expected0]",
    ),
    (
        "owner-accounting-terminal-table-freeze",
        "engine/execution/accounting_ledger.py",
        (
            (
                "            _Phase10TerminalSummary,\n        )\n\n",
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n"
                "        _current_limit = _phase10_limit\n"
                "\n"
                "        def _phase10_limit(name):\n"
                "            return 48 if name is _Phase10LimitName.MAX_TERMINAL_SUMMARY_ROWS else _current_limit(name)\n"
                "\n",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_owner_capture_uses_authoritative_row_limits[terminal-accounting-expected0]",
    ),
    (
        "owner-accounting-diagnostic-table-freeze",
        "engine/execution/accounting_ledger.py",
        (
            (
                "            _Phase10TerminalSummary,\n        )\n\n",
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n"
                "        _current_limit = _phase10_limit\n"
                "\n"
                "        def _phase10_limit(name):\n"
                "            return 64 if name is _Phase10LimitName.MAX_DIAGNOSTIC_ENTRIES else _current_limit(name)\n"
                "\n",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_owner_capture_uses_authoritative_row_limits[diagnostic-accounting-expected0]",
    ),
    (
        "owner-accounting-reason-table-freeze",
        "engine/execution/accounting_ledger.py",
        (
            (
                "            _Phase10TerminalSummary,\n        )\n\n",
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n"
                "        _current_limit = _phase10_limit\n"
                "\n"
                "        def _phase10_limit(name):\n"
                "            return 4 if name is _Phase10LimitName.MAX_REASON_CODES_PER_DIAGNOSTIC_ENTRY else "
                "_current_limit(name)\n"
                "\n",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_owner_capture_uses_authoritative_reason_limit[True-accounting]",
    ),
    (
        "owner-graph-stage-table-freeze",
        "engine/execution/graph_runtime.py",
        (
            (
                "            _Phase10TerminalSummary,\n        )\n\n",
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n"
                "        _current_limit = _phase10_limit\n"
                "\n"
                "        def _phase10_limit(name):\n"
                "            return 8 if name is _Phase10LimitName.MAX_STAGE_SUMMARIES else _current_limit(name)\n"
                "\n",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_owner_capture_uses_authoritative_row_limits[stage-graph-expected1]",
    ),
    (
        "owner-graph-terminal-table-freeze",
        "engine/execution/graph_runtime.py",
        (
            (
                "            _Phase10TerminalSummary,\n        )\n\n",
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n"
                "        _current_limit = _phase10_limit\n"
                "\n"
                "        def _phase10_limit(name):\n"
                "            return 48 if name is _Phase10LimitName.MAX_TERMINAL_SUMMARY_ROWS else _current_limit(name)\n"
                "\n",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_owner_capture_uses_authoritative_row_limits[terminal-graph-expected1]",
    ),
    (
        "owner-graph-diagnostic-table-freeze",
        "engine/execution/graph_runtime.py",
        (
            (
                "            _Phase10TerminalSummary,\n        )\n\n",
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n"
                "        _current_limit = _phase10_limit\n"
                "\n"
                "        def _phase10_limit(name):\n"
                "            return 64 if name is _Phase10LimitName.MAX_DIAGNOSTIC_ENTRIES else _current_limit(name)\n"
                "\n",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_owner_capture_uses_authoritative_row_limits[diagnostic-graph-expected1]",
    ),
    (
        "owner-graph-reason-table-freeze",
        "engine/execution/graph_runtime.py",
        (
            (
                "            _Phase10TerminalSummary,\n        )\n\n",
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n"
                "        _current_limit = _phase10_limit\n"
                "\n"
                "        def _phase10_limit(name):\n"
                "            return 4 if name is _Phase10LimitName.MAX_REASON_CODES_PER_DIAGNOSTIC_ENTRY else "
                "_current_limit(name)\n"
                "\n",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_owner_capture_uses_authoritative_reason_limit[True-graph]",
    ),
    (
        "owner-operation-stage-table-freeze",
        "engine/execution/phase8_runtime.py",
        (
            (
                "            _Phase10Snapshot,\n"
                "            _Phase10StageSummary,\n"
                "            _Phase10SubjectKind,\n"
                "            _Phase10TerminalState,\n"
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n",
                "            _Phase10Snapshot,\n"
                "            _Phase10StageSummary,\n"
                "            _Phase10SubjectKind,\n"
                "            _Phase10TerminalState,\n"
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n"
                "        _current_limit = _phase10_limit\n"
                "\n"
                "        def _phase10_limit(name):\n"
                "            return 8 if name is _Phase10LimitName.MAX_STAGE_SUMMARIES else _current_limit(name)\n"
                "\n",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_owner_capture_uses_authoritative_row_limits[stage-operation-expected2]",
    ),
    (
        "owner-operation-terminal-table-freeze",
        "engine/execution/phase8_runtime.py",
        (
            (
                "            _Phase10Snapshot,\n"
                "            _Phase10StageSummary,\n"
                "            _Phase10SubjectKind,\n"
                "            _Phase10TerminalState,\n"
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n",
                "            _Phase10Snapshot,\n"
                "            _Phase10StageSummary,\n"
                "            _Phase10SubjectKind,\n"
                "            _Phase10TerminalState,\n"
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n"
                "        _current_limit = _phase10_limit\n"
                "\n"
                "        def _phase10_limit(name):\n"
                "            return 48 if name is _Phase10LimitName.MAX_TERMINAL_SUMMARY_ROWS else _current_limit(name)\n"
                "\n",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_owner_capture_uses_authoritative_row_limits[terminal-operation-expected2]",
    ),
    (
        "owner-operation-diagnostic-table-freeze",
        "engine/execution/phase8_runtime.py",
        (
            (
                "            _Phase10Snapshot,\n"
                "            _Phase10StageSummary,\n"
                "            _Phase10SubjectKind,\n"
                "            _Phase10TerminalState,\n"
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n",
                "            _Phase10Snapshot,\n"
                "            _Phase10StageSummary,\n"
                "            _Phase10SubjectKind,\n"
                "            _Phase10TerminalState,\n"
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n"
                "        _current_limit = _phase10_limit\n"
                "\n"
                "        def _phase10_limit(name):\n"
                "            return 64 if name is _Phase10LimitName.MAX_DIAGNOSTIC_ENTRIES else _current_limit(name)\n"
                "\n",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_owner_capture_uses_authoritative_row_limits[diagnostic-operation-expected2]",
    ),
    (
        "owner-lifecycle-stage-table-freeze",
        "engine/execution/phase8_runtime.py",
        (
            (
                "            _Phase10Stage,\n"
                "            _Phase10StageSummary,\n"
                "            _Phase10SubjectKind,\n"
                "            _Phase10TerminalState,\n"
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n",
                "            _Phase10Stage,\n"
                "            _Phase10StageSummary,\n"
                "            _Phase10SubjectKind,\n"
                "            _Phase10TerminalState,\n"
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n"
                "        _current_limit = _phase10_limit\n"
                "\n"
                "        def _phase10_limit(name):\n"
                "            return 8 if name is _Phase10LimitName.MAX_STAGE_SUMMARIES else _current_limit(name)\n"
                "\n",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_owner_capture_uses_authoritative_row_limits[stage-lifecycle-expected3]",
    ),
    (
        "owner-lifecycle-terminal-table-freeze",
        "engine/execution/phase8_runtime.py",
        (
            (
                "            _Phase10Stage,\n"
                "            _Phase10StageSummary,\n"
                "            _Phase10SubjectKind,\n"
                "            _Phase10TerminalState,\n"
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n",
                "            _Phase10Stage,\n"
                "            _Phase10StageSummary,\n"
                "            _Phase10SubjectKind,\n"
                "            _Phase10TerminalState,\n"
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n"
                "        _current_limit = _phase10_limit\n"
                "\n"
                "        def _phase10_limit(name):\n"
                "            return 48 if name is _Phase10LimitName.MAX_TERMINAL_SUMMARY_ROWS else _current_limit(name)\n"
                "\n",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_owner_capture_uses_authoritative_row_limits[terminal-lifecycle-expected3]",
    ),
    (
        "owner-lifecycle-diagnostic-table-freeze",
        "engine/execution/phase8_runtime.py",
        (
            (
                "            _Phase10Stage,\n"
                "            _Phase10StageSummary,\n"
                "            _Phase10SubjectKind,\n"
                "            _Phase10TerminalState,\n"
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n",
                "            _Phase10Stage,\n"
                "            _Phase10StageSummary,\n"
                "            _Phase10SubjectKind,\n"
                "            _Phase10TerminalState,\n"
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n"
                "        _current_limit = _phase10_limit\n"
                "\n"
                "        def _phase10_limit(name):\n"
                "            return 64 if name is _Phase10LimitName.MAX_DIAGNOSTIC_ENTRIES else _current_limit(name)\n"
                "\n",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_owner_capture_uses_authoritative_row_limits[diagnostic-lifecycle-expected3]",
    ),
    (
        "owner-lifecycle-reason-table-freeze",
        "engine/execution/phase8_runtime.py",
        (
            (
                "            _Phase10Stage,\n"
                "            _Phase10StageSummary,\n"
                "            _Phase10SubjectKind,\n"
                "            _Phase10TerminalState,\n"
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n",
                "            _Phase10Stage,\n"
                "            _Phase10StageSummary,\n"
                "            _Phase10SubjectKind,\n"
                "            _Phase10TerminalState,\n"
                "            _Phase10TerminalSummary,\n"
                "        )\n"
                "\n"
                "        _current_limit = _phase10_limit\n"
                "\n"
                "        def _phase10_limit(name):\n"
                "            return 4 if name is _Phase10LimitName.MAX_REASON_CODES_PER_DIAGNOSTIC_ENTRY else "
                "_current_limit(name)\n"
                "\n",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_owner_capture_uses_authoritative_reason_limit[True-lifecycle]",
    ),
    (
        "declared-table-length-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                "    if len(items) != len(_PHASE10_LIMITS):\n"
                "        return False\n"
                "    for item, (expected_name, expected_limit) in zip(items, _PHASE10_LIMITS.items(), strict=True):",
                "    for item, (expected_name, expected_limit) in zip(items, _PHASE10_LIMITS.items(), strict=False):",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_declared_table_payload_requires_exact_authoritative_sequence[empty]",
    ),
    (
        "declared-table-order-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                "        if type(name) is not str or name != expected_name.value or type(limit) is not int or limit != "
                "expected_limit:",
                "        if type(name) is not str or name not in {key.value for key in _PHASE10_LIMITS} or type(limit) is not "
                "int or limit != expected_limit:",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_declared_table_payload_requires_exact_authoritative_sequence[reordered]",
    ),
    (
        "declared-table-value-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                "        if type(name) is not str or name != expected_name.value or type(limit) is not int or limit != "
                "expected_limit:",
                "        if type(name) is not str or name != expected_name.value or type(limit) is not int:",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_declared_table_payload_requires_exact_authoritative_sequence[wrong_value]",
    ),
    (
        "declared-table-integer-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                "        if type(name) is not str or name != expected_name.value or type(limit) is not int or limit != "
                "expected_limit:",
                "        if type(name) is not str or name != expected_name.value or limit != expected_limit:",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_declared_table_validator_requires_exact_builtins[int_subclass]",
    ),
    (
        "declared-table-boolean-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                "        if type(name) is not str or name != expected_name.value or type(limit) is not int or limit != "
                "expected_limit:",
                "        if type(name) is not str or name != expected_name.value or limit != expected_limit:",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_declared_table_validator_requires_exact_builtins[boolean]",
    ),
    (
        "declared-table-name-type-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                "        if type(name) is not str or name != expected_name.value or type(limit) is not int or limit != "
                "expected_limit:",
                "        if name != expected_name.value or type(limit) is not int or limit != expected_limit:",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_declared_table_validator_requires_exact_builtins[str_subclass]",
    ),
    (
        "declared-table-list-type-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                "def _valid_declared_limits_payload(value: object) -> bool:\n    if type(value) is not list:",
                "def _valid_declared_limits_payload(value: object) -> bool:\n    if not isinstance(value, list):",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_declared_table_validator_requires_exact_builtins[list_subclass]",
    ),
    (
        "declared-table-entry-type-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                '        if not _payload_has_exact_keys(item, frozenset({"name", "value"})):\n'
                "            return False\n"
                "        record = cast(dict[str, object], item)",
                '        if not isinstance(item, dict) or set(item) != {"name", "value"}:\n'
                "            return False\n"
                "        record = cast(dict[str, object], item)",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_declared_table_validator_requires_exact_builtins[dict_subclass]",
    ),
    (
        "declared-table-entry-keys-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                '        if not _payload_has_exact_keys(item, frozenset({"name", "value"})):\n'
                "            return False\n"
                "        record = cast(dict[str, object], item)",
                '        if type(item) is not dict or not {"name", "value"}.issubset(item):\n'
                "            return False\n"
                "        record = cast(dict[str, object], item)",
            ),
        ),
        "tests/engine/execution/test_phase10_bounded_inspection.py::test_phase10_declared_table_payload_requires_exact_authoritative_sequence[malformed]",
    ),
)


_TABLE_MUTATIONS += (
    (
        "full-encoder-root-limit-bypass",
        "engine/execution/phase10_inspection.py",
        (("if top_level_fields > _phase10_limit(_Phase10LimitName.MAX_TOP_LEVEL_FIELDS):", "if False:"),),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_full_encoder_enforces_lowered_measured_payload_ceiling[MAX_TOP_LEVEL_FIELDS-0-diagnose]",
    ),
    (
        "full-encoder-provenance-limit-bypass",
        "engine/execution/phase10_inspection.py",
        (("if provenance_fields > _phase10_limit(_Phase10LimitName.MAX_PROVENANCE_FIELDS):", "if False:"),),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_full_encoder_enforces_lowered_measured_payload_ceiling[MAX_PROVENANCE_FIELDS-1-explain]",
    ),
    (
        "full-encoder-depth-limit-bypass",
        "engine/execution/phase10_inspection.py",
        (
            (
                "    if type(value) is dict:\n"
                "        if container_depth > _phase10_limit(_Phase10LimitName.MAX_JSON_NESTING_DEPTH):\n",
                "    if type(value) is dict:\n        if False:\n",
            ),
        ),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_full_encoder_enforces_lowered_measured_payload_ceiling[MAX_JSON_NESTING_DEPTH-2-inspect]",
    ),
    (
        "full-encoder-string-limit-bypass",
        "engine/execution/phase10_inspection.py",
        (("if string_bytes > _phase10_limit(_Phase10LimitName.MAX_ALLOWLISTED_STRING_UTF8_BYTES):", "if False:"),),
        "tests/engine/execution/test_phase10_privacy_canaries.py::test_full_encoder_enforces_lowered_measured_payload_ceiling[MAX_ALLOWLISTED_STRING_UTF8_BYTES-3-explain]",
    ),
)


_DESIGNATED_ASSERTIONS: dict[str, tuple[str, str]] = {
    "grant-validation-before-subject-access": (
        "test_phase10_denies_before_subject_snapshot_access",
        "assert denied.code is module._Phase10RejectionCode.DENIED",
    ),
    "grant-forgery-reuse-expiry-operation-subject": (
        "test_grant_validation_rejects_each_invalid_binding_and_state",
        "assert accepted is False",
    ),
    "grant-authority-overclaim": (
        "test_phase10_contract_resource_has_exact_approved_raw_and_member_digests",
        "assert hashlib.sha256(resource).hexdigest() == _CONTRACT_RAW_DIGEST",
    ),
    "public-surface-exposure": (
        "test_phase10_types_are_not_exposed_through_public_surfaces",
        """assert not any(
        name in public_names
        for name in (
            "explain",
            "inspect",
            "diagnose",
            "InspectionGrant",
            "InspectionView",
            "Phase10Snapshot",
        )
    )""",
    ),
    "forbidden-rich-traversal": (
        "_production_encoding",
        "assert type(encoding) is module._Phase10CanonicalEncoding",
    ),
    "retained-private-reference": (
        "_assert_lifetimes_released",
        "assert remaining == []",
    ),
    "forbidden-value-leakage": (
        "_encoded_explain",
        "assert type(encoded) is module._Phase10CanonicalEncoding",
    ),
    "partial-or-truncated-output": (
        "test_oversize_encoding_returns_no_prefix_suffix_or_partial_payload",
        "assert type(result) is module._Phase10InspectionRejected",
    ),
    "forbidden-sort-key": (
        "test_phase10_encoder_emits_exact_canonical_json_and_preserves_semantic_array_order",
        """assert result.value == json.dumps(
        json.loads(result.value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")""",
    ),
    "permutation-byte-drift": (
        "test_every_admitted_reference_payload_matches_production_canonical_serializer",
        "assert _production_encoding(case) == expected.canonical_json",
    ),
    "inspection-side-effect": (
        "test_snapshot_is_observational_and_does_not_mutate_owning_ledger",
        "assert after == before",
    ),
    "terminal-evidence-rewrite": (
        "test_terminal_is_absorbing_and_failure_blocks_descendants",
        "assert isinstance(ledger.terminal(phase8_runtime._Phase8Stage.analyze()), phase8_runtime._StageFailed)",
    ),
    "post-cleanup-revival": (
        "test_post_cleanup_snapshot_never_reads_or_revives_cleanup_identity",
        "assert not _AccessCanary.touched",
    ),
    "private-serialization-or-auto-persistence": (
        "test_private_inputs_views_and_encodings_mask_repr_and_reject_pickle",
        "with pytest.raises(TypeError):\n            pickle.dumps(value)",
    ),
    "noncanonical-json": (
        "test_phase10_encoder_emits_exact_canonical_json_and_preserves_semantic_array_order",
        """assert result.value == json.dumps(
        json.loads(result.value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")""",
    ),
    "unmapped-owner-reason": (
        "test_known_owner_reason_mapping_matches_the_oracle",
        """assert (
        module._map_phase10_reason(outcomes._CauseCode.KNOWN_FAILURE).value
        == case_by_name("reason-backend_failed").diagnostics[0].reason_category
    )""",
    ),
    "unexpected-failure-detail-leak": (
        "test_encoding_failure_is_cause_free_and_returns_no_partial_bytes",
        "assert type(result) is module._Phase10InspectionRejected",
    ),
    "p9-or-public-compatibility-drift": (
        "test_imported_package_p9_contract_witness",
        'assert envelope["digest"] == hashlib.sha256(canonical).hexdigest()',
    ),
    "ndd-boundary-bypass": (
        "test_imported_package_ndd_boundary_witness",
        'assert calls.count("run_workflow") == 1',
    ),
    "encoder-validator-invocation-bypass": (
        "test_phase10_every_encoder_variant_invokes_validator_and_enforces_actual_byte_boundary",
        """assert validated_kinds == [
        "explain",
        "explain",
        "explain",
        "inspect",
        "inspect",
        "inspect",
        "diagnose",
        "diagnose",
        "diagnose",
    ]""",
    ),
    "payload-root-limit-bypass": (
        "test_phase10_payload_measurement_enforces_exact_and_one_over_boundaries",
        "assert type(rejected) is module._Phase10InspectionRejected",
    ),
    "payload-provenance-limit-bypass": (
        "test_phase10_payload_measurement_enforces_exact_and_one_over_boundaries",
        "assert type(rejected) is module._Phase10InspectionRejected",
    ),
    "payload-dict-depth-limit-bypass": (
        "test_phase10_payload_measurement_enforces_exact_and_one_over_boundaries",
        "assert type(rejected) is module._Phase10InspectionRejected",
    ),
    "payload-list-depth-limit-bypass": (
        "test_phase10_payload_measurement_enforces_exact_and_one_over_boundaries",
        "assert type(rejected) is module._Phase10InspectionRejected",
    ),
    "payload-key-string-limit-bypass": (
        "test_payload_measurement_enforces_each_key_utf8_boundary_without_partial_output",
        "assert type(over) is module._Phase10InspectionRejected",
    ),
    "payload-value-string-limit-bypass": (
        "test_phase10_payload_measurement_enforces_exact_and_one_over_boundaries",
        "assert type(rejected) is module._Phase10InspectionRejected",
    ),
    "payload-root-exact-built-in-bypass": (
        "test_phase10_payload_measurement_rejects_non_exact_builtins_without_traversal",
        "assert type(admitted) is module._Phase10PayloadMeasurement",
    ),
    "payload-schema-admission-bypass": (
        "test_phase10_encoder_applies_measurement_before_exact_schema_admission",
        "assert type(malformed) is module._Phase10InspectionRejected",
    ),
    "declared-limit-order-drift": (
        "_assert_owner_view_matches_reference",
        "assert type(encoded) is module._Phase10CanonicalEncoding",
    ),
    "explain-subject-validation-order-inversion": (
        "test_phase10_subject_validation_precedes_builder_budget",
        "assert result.code is module._Phase10RejectionCode.SUBJECT_INVALID",
    ),
    "inspect-subject-validation-order-inversion": (
        "test_phase10_subject_validation_precedes_builder_budget",
        "assert result.code is module._Phase10RejectionCode.SUBJECT_INVALID",
    ),
    "diagnose-subject-validation-order-inversion": (
        "test_phase10_subject_validation_precedes_builder_budget",
        "assert result.code is module._Phase10RejectionCode.SUBJECT_INVALID",
    ),
    "canonical-byte-exact-limit-off-by-one": (
        "test_phase10_every_encoder_variant_invokes_validator_and_enforces_actual_byte_boundary",
        "assert type(exact) is module._Phase10CanonicalEncoding",
    ),
    "fixed-arity-extra-parameter": (
        "test_phase10_operations_remain_fixed_arity_scalar_calls",
        'assert tuple(signature.parameters) == ("owner", "subject", "grant")',
    ),
    "grant-exact-type-bypass": (
        "test_grant_validation_rejects_each_invalid_binding_and_state",
        "assert accepted is False",
    ),
    "grant-seal-bypass": ("test_grant_validation_rejects_each_invalid_binding_and_state", "assert accepted is False"),
    "grant-state-type-bypass": (
        "test_grant_validation_rejects_each_invalid_binding_and_state",
        "assert accepted is False",
    ),
    "grant-active-state-bypass": (
        "test_grant_validation_rejects_each_invalid_binding_and_state",
        "assert accepted is False",
    ),
    "grant-operation-binding-bypass": (
        "test_grant_validation_rejects_each_invalid_binding_and_state",
        "assert accepted is False",
    ),
    "grant-subject-binding-bypass": (
        "test_grant_validation_rejects_each_invalid_binding_and_state",
        "assert accepted is False",
    ),
    "grant-consumption-invalidation-bypass": (
        "test_grant_validation_rejects_each_invalid_binding_and_state",
        "assert accepted is False",
    ),
    "grant-revocation-invalidation-bypass": (
        "test_grant_validation_rejects_each_invalid_binding_and_state",
        "assert accepted is False",
    ),
    "grant-expiry-invalidation-bypass": (
        "test_grant_validation_rejects_each_invalid_binding_and_state",
        "assert accepted is False",
    ),
    "fixed-arity-inspect-extra-parameter": (
        "test_each_operation_has_exact_scalar_signature",
        'assert tuple(signature.parameters) == ("owner", "subject", "grant")',
    ),
    "fixed-arity-diagnose-extra-parameter": (
        "test_each_operation_has_exact_scalar_signature",
        'assert tuple(signature.parameters) == ("owner", "subject", "grant")',
    ),
    "phase10-retained-owner": ("_assert_lifetimes_released", "assert remaining == []"),
    "phase10-retained-subject": ("_assert_lifetimes_released", "assert remaining == []"),
    "phase10-retained-grant": ("_assert_lifetimes_released", "assert remaining == []"),
    "phase10-retained-content-state": ("_assert_lifetimes_released", "assert remaining == []"),
    "phase10-telemetry-emission": (
        "test_failure_paths_leave_real_protection_state_unchanged_and_silent",
        "assert telemetry_calls == []",
    ),
    "grant-missing-state-admission": (
        "test_grant_validation_rejects_each_invalid_binding_and_state",
        "assert accepted is False",
    ),
    "grant-operation-type-bypass": (
        "test_grant_validation_rejects_each_invalid_binding_and_state",
        "assert accepted is False",
    ),
    "grant-missing-owner-binding-admission": (
        "test_grant_validation_rejects_each_invalid_binding_and_state",
        "assert accepted is False",
    ),
    "grant-missing-subject-binding-admission": (
        "test_grant_validation_rejects_each_invalid_binding_and_state",
        "assert accepted is False",
    ),
}

_DESIGNATED_ASSERTIONS.update(
    {
        "owner-accounting-stage-table-freeze": (
            "test_phase10_owner_capture_uses_authoritative_row_limits",
            "assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED",
        ),
        "owner-accounting-terminal-table-freeze": (
            "test_phase10_owner_capture_uses_authoritative_row_limits",
            "assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED",
        ),
        "owner-accounting-diagnostic-table-freeze": (
            "test_phase10_owner_capture_uses_authoritative_row_limits",
            "assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED",
        ),
        "owner-accounting-reason-table-freeze": (
            "test_phase10_owner_capture_uses_authoritative_reason_limit",
            "assert type(rejected) is module._Phase10InspectionRejected",
        ),
        "owner-graph-stage-table-freeze": (
            "test_phase10_owner_capture_uses_authoritative_row_limits",
            "assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED",
        ),
        "owner-graph-terminal-table-freeze": (
            "test_phase10_owner_capture_uses_authoritative_row_limits",
            "assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED",
        ),
        "owner-graph-diagnostic-table-freeze": (
            "test_phase10_owner_capture_uses_authoritative_row_limits",
            "assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED",
        ),
        "owner-graph-reason-table-freeze": (
            "test_phase10_owner_capture_uses_authoritative_reason_limit",
            "assert type(rejected) is module._Phase10InspectionRejected",
        ),
        "owner-operation-stage-table-freeze": (
            "test_phase10_owner_capture_uses_authoritative_row_limits",
            "assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED",
        ),
        "owner-operation-terminal-table-freeze": (
            "test_phase10_owner_capture_uses_authoritative_row_limits",
            "assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED",
        ),
        "owner-operation-diagnostic-table-freeze": (
            "test_phase10_owner_capture_uses_authoritative_row_limits",
            "assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED",
        ),
        "owner-lifecycle-stage-table-freeze": (
            "test_phase10_owner_capture_uses_authoritative_row_limits",
            "assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED",
        ),
        "owner-lifecycle-terminal-table-freeze": (
            "test_phase10_owner_capture_uses_authoritative_row_limits",
            "assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED",
        ),
        "owner-lifecycle-diagnostic-table-freeze": (
            "test_phase10_owner_capture_uses_authoritative_row_limits",
            "assert rejected.code is module._Phase10RejectionCode.LIMIT_EXCEEDED",
        ),
        "owner-lifecycle-reason-table-freeze": (
            "test_phase10_owner_capture_uses_authoritative_reason_limit",
            "assert type(rejected) is module._Phase10InspectionRejected",
        ),
        "declared-table-length-bypass": (
            "test_phase10_declared_table_payload_requires_exact_authoritative_sequence",
            "assert type(result) is module._Phase10InspectionRejected",
        ),
        "declared-table-order-bypass": (
            "test_phase10_declared_table_payload_requires_exact_authoritative_sequence",
            "assert type(result) is module._Phase10InspectionRejected",
        ),
        "declared-table-value-bypass": (
            "test_phase10_declared_table_payload_requires_exact_authoritative_sequence",
            "assert type(result) is module._Phase10InspectionRejected",
        ),
        "declared-table-integer-bypass": (
            "test_phase10_declared_table_validator_requires_exact_builtins",
            "assert module._valid_declared_limits_payload(candidate) is False",
        ),
        "declared-table-boolean-bypass": (
            "test_phase10_declared_table_validator_requires_exact_builtins",
            "assert module._valid_declared_limits_payload(candidate) is False",
        ),
        "declared-table-name-type-bypass": (
            "test_phase10_declared_table_validator_requires_exact_builtins",
            "assert module._valid_declared_limits_payload(candidate) is False",
        ),
        "declared-table-list-type-bypass": (
            "test_phase10_declared_table_validator_requires_exact_builtins",
            "assert module._valid_declared_limits_payload(candidate) is False",
        ),
        "declared-table-entry-type-bypass": (
            "test_phase10_declared_table_validator_requires_exact_builtins",
            "assert module._valid_declared_limits_payload(candidate) is False",
        ),
        "declared-table-entry-keys-bypass": (
            "test_phase10_declared_table_payload_requires_exact_authoritative_sequence",
            "assert type(result) is module._Phase10InspectionRejected",
        ),
    }
)

_DESIGNATED_ASSERTIONS.update(
    {
        "full-encoder-depth-limit-bypass": (
            "test_full_encoder_enforces_lowered_measured_payload_ceiling",
            "assert type(rejected) is module._Phase10InspectionRejected",
        ),
        "full-encoder-provenance-limit-bypass": (
            "test_full_encoder_enforces_lowered_measured_payload_ceiling",
            "assert type(rejected) is module._Phase10InspectionRejected",
        ),
        "full-encoder-root-limit-bypass": (
            "test_full_encoder_enforces_lowered_measured_payload_ceiling",
            "assert type(rejected) is module._Phase10InspectionRejected",
        ),
        "full-encoder-string-limit-bypass": (
            "test_full_encoder_enforces_lowered_measured_payload_ceiling",
            "assert type(rejected) is module._Phase10InspectionRejected",
        ),
    }
)


def _instrument_designated_assertion(source: str, scope: str, assertion: str, marker: str) -> str:
    tree = ast.parse(source)
    matches: list[ast.Assert | ast.With] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != scope:
            continue
        matches.extend(
            candidate
            for candidate in ast.walk(node)
            if isinstance(candidate, (ast.Assert, ast.With)) and ast.get_source_segment(source, candidate) == assertion
        )
    assert len(matches) == 1, f"designated mutation assertion drifted for {scope}"
    target = matches[0]

    class MarkerInjector(ast.NodeTransformer):
        def visit_Assert(self, node: ast.Assert) -> ast.Assert:
            if node is target:
                assert node.msg is None
                node.msg = ast.Constant(marker)
            return node

        def visit_With(self, node: ast.With) -> ast.With | ast.Assert:
            if node is not target:
                self.generic_visit(node)
                return node
            assert len(node.items) == len(node.body) == 1
            context = node.items[0].context_expr
            assert isinstance(context, ast.Call) and context.args
            body = node.body[0]
            assert isinstance(body, ast.Expr)
            return ast.Assert(
                test=ast.Call(
                    func=ast.Name(id="_phase10_designated_raises"),
                    args=[
                        context.args[0],
                        ast.Lambda(
                            args=ast.arguments(
                                posonlyargs=[],
                                args=[],
                                kwonlyargs=[],
                                kw_defaults=[],
                                defaults=[],
                            ),
                            body=body.value,
                        ),
                    ],
                    keywords=[],
                ),
                msg=ast.Constant(marker),
            )

    instrumented = MarkerInjector().visit(tree)
    if isinstance(target, ast.With):
        helper = ast.parse(
            """def _phase10_designated_raises(exception_type, callback):
    try:
        callback()
    except exception_type:
        return True
    return False
"""
        ).body[0]
        insertion = next(
            (
                index
                for index, node in enumerate(instrumented.body)
                if not isinstance(node, (ast.Import, ast.ImportFrom))
            ),
            len(instrumented.body),
        )
        instrumented.body.insert(insertion, helper)
    ast.fix_missing_locations(instrumented)
    return f"{ast.unparse(instrumented)}\n"


def _instrumented_witness(
    repository: Path,
    witness: str,
    destination: Path,
    designated: tuple[str, str],
    marker: str,
) -> str:
    relative_path, node = witness.split("::", maxsplit=1)
    source = (repository / relative_path).read_text(encoding="utf-8")
    destination.write_text(_instrument_designated_assertion(source, *designated, marker), encoding="utf-8")
    return f"{destination}::{node}"


def _witness_environment(repository: Path, source: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("COV_CORE_"):
            environment.pop(key)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source), str(repository), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    return environment


def _run_witness(
    repository: Path,
    source: Path,
    witness: str,
    junit: Path,
    environment_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(repository / witness),
        "-q",
        f"--junitxml={junit}",
    ]
    return subprocess.run(
        command,
        cwd=repository,
        env={**_witness_environment(repository, source), **(environment_overrides or {})},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _junit_case(junit: Path, witness: str) -> ET.Element:
    root = ET.parse(junit).getroot()
    cases = root.findall(".//testcase")
    assert len(cases) == 1
    case = cases[0]
    assert case.attrib["name"] == witness.rsplit("::", maxsplit=1)[1]
    assert not root.findall(".//error")
    assert not root.findall(".//skipped")
    return case


@pytest.mark.parametrize(
    "mutation",
    (
        *_MUTATIONS,
        *_ENFORCEMENT_MUTATIONS,
        *_GRANT_AND_ARITY_MUTATIONS,
        *_GRANT_STATE_MUTATIONS,
        *_PRIVACY_MUTATIONS,
        *_TABLE_MUTATIONS,
    ),
    ids=lambda mutation: mutation[0],
)
def test_each_phase10_production_seam_mutant_is_killed(mutation: _Mutation, tmp_path: Path) -> None:
    name, relative_path, replacements, witness = mutation
    repository = Path(__file__).parents[3]
    mutant_root = tmp_path / "mutant-repository"
    mutant_source = mutant_root / "src"
    shutil.copytree(repository / "src" / "anonymizer", mutant_source / "anonymizer")
    baseline_junit = tmp_path / "baseline.xml"
    baseline = _run_witness(repository, mutant_source, witness, baseline_junit)
    baseline_case = _junit_case(baseline_junit, witness)
    assert baseline.returncode == 0, f"baseline witness failed for {name}\n{baseline.stdout}\n{baseline.stderr}"
    assert baseline_case.find("failure") is None

    target = mutant_source / "anonymizer" / relative_path
    source = target.read_text(encoding="utf-8")
    for original, replacement in replacements:
        assert source.count(original) == 1, f"production mutation seam drifted for {name}"
        source = source.replace(original, replacement)
    target.write_text(source, encoding="utf-8")
    marker = f"PHASE10_WITNESS:{name}"
    instrumented_witness = _instrumented_witness(
        repository,
        witness,
        tmp_path / "test_designated_witness.py",
        _DESIGNATED_ASSERTIONS[name],
        marker,
    )
    mutant_junit = tmp_path / "mutant.xml"
    completed = _run_witness(repository, mutant_source, instrumented_witness, mutant_junit)
    assert mutant_junit.exists(), (
        f"mutant witness did not produce JUnit for {name}\n{completed.stdout}\n{completed.stderr}"
    )
    mutant_case = _junit_case(mutant_junit, instrumented_witness)
    failure = mutant_case.find("failure")

    assert completed.returncode == 1, (
        f"required Phase 10 production mutation survived or did not reach its assertion: {name}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert failure is not None
    failure_text = "".join(failure.itertext())
    assert marker in failure_text
    assert "AssertionError" in failure_text or "Failed:" in failure_text


def test_wheel_contract_omission_mutation_is_killed(tmp_path: Path) -> None:
    repository = Path(__file__).parents[3]
    member = "anonymizer/engine/execution/phase10_bounded_inspection_contract.json"

    baseline_output = tmp_path / "baseline-dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(baseline_output)],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    (baseline_wheel,) = tuple(baseline_output.glob("*.whl"))

    mutant_root = tmp_path / "source-mutant"
    shutil.copytree(repository / "src", mutant_root / "src")
    for filename in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(repository / filename, mutant_root / filename)
    (mutant_root / "src" / member).unlink()
    mutant_output = tmp_path / "mutant-dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(mutant_output)],
        cwd=mutant_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    (mutant_wheel,) = tuple(mutant_output.glob("*.whl"))

    witness_file = tmp_path / "test_wheel_acceptance.py"
    marker = "PHASE10_WITNESS:wheel-contract-omission"
    witness_file.write_text(
        "import os\nimport zipfile\n"
        "def test_wheel_acceptance():\n"
        "    with zipfile.ZipFile(os.environ['PHASE10_WITNESS_WHEEL']) as archive:\n"
        f"        assert {member!r} in archive.namelist(), {marker!r}\n",
        encoding="utf-8",
    )
    witness = f"{witness_file}::test_wheel_acceptance"
    for label, wheel in (("baseline", baseline_wheel), ("mutant", mutant_wheel)):
        junit = tmp_path / f"wheel-{label}.xml"
        completed = _run_witness(
            repository,
            repository / "src",
            witness,
            junit,
            {"PHASE10_WITNESS_WHEEL": str(wheel)},
        )
        case = _junit_case(junit, witness)
        failure = case.find("failure")
        if label == "baseline":
            assert completed.returncode == 0, completed.stdout + completed.stderr
            assert failure is None
        else:
            assert completed.returncode == 1, completed.stdout + completed.stderr
            assert failure is not None
            assert failure.attrib["message"].startswith(f"AssertionError: {marker}")


def test_only_designated_mutation_assertion_emits_witness_marker(tmp_path: Path) -> None:
    marker = "PHASE10_WITNESS:designated"
    source = "def test_witness():\n    assert False, 'unrelated'\n    assert False\n"
    witness = tmp_path / "test_witness.py"
    witness.write_text(
        _instrument_designated_assertion(source, "test_witness", "assert False", marker),
        encoding="utf-8",
    )
    junit = tmp_path / "unrelated.xml"

    repository = Path(__file__).parents[3]
    completed = _run_witness(repository, repository / "src", f"{witness}::test_witness", junit)
    failure = _junit_case(junit, f"{witness}::test_witness").find("failure")

    assert completed.returncode == 1
    assert failure is not None
    assert marker not in "".join(failure.itertext())


def test_designated_mutation_assertion_emits_witness_marker(tmp_path: Path) -> None:
    marker = "PHASE10_WITNESS:designated"
    source = "def test_witness():\n    assert True\n    assert False\n"
    witness = tmp_path / "test_witness.py"
    witness.write_text(
        _instrument_designated_assertion(source, "test_witness", "assert False", marker),
        encoding="utf-8",
    )
    junit = tmp_path / "designated.xml"

    repository = Path(__file__).parents[3]
    completed = _run_witness(repository, repository / "src", f"{witness}::test_witness", junit)
    failure = _junit_case(junit, f"{witness}::test_witness").find("failure")

    assert completed.returncode == 1
    assert failure is not None
    assert marker in "".join(failure.itertext())


def test_imported_package_p9_contract_witness() -> None:
    interface = importlib.import_module("anonymizer.interface")
    resource = Path(interface.__file__).parent / "result_compatibility_contract.json"
    envelope = json.loads(resource.read_text(encoding="utf-8"))
    canonical = json.dumps(envelope["contract"], sort_keys=True, separators=(",", ":")).encode("utf-8")

    assert envelope["digest"] == hashlib.sha256(canonical).hexdigest()
    assert envelope["digest"] == "c91a410289c3549f608cc0b088da3ce9db56ac10aeabe430a8254b637ef4b12d"


def test_imported_package_ndd_boundary_witness() -> None:
    backend = importlib.import_module("anonymizer.engine.execution.phase8_ndd_backend")
    tree = ast.parse(Path(backend.__file__).read_text(encoding="utf-8"))
    calls = tuple(
        node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )

    assert calls.count("run_workflow") == 1
    assert "create" not in calls
    assert "preview" not in calls


def test_production_mutation_inventory_is_complete_unique_and_contract_bound() -> None:
    names = tuple(name for name, _path, _replacements, _witness in _MUTATIONS) + ("wheel-contract-omission",)
    instances = tuple(
        name
        for name, _path, _replacements, _witness in (
            *_MUTATIONS,
            *_ENFORCEMENT_MUTATIONS,
            *_GRANT_AND_ARITY_MUTATIONS,
            *_GRANT_STATE_MUTATIONS,
            *_PRIVACY_MUTATIONS,
            *_TABLE_MUTATIONS,
        )
    ) + ("wheel-contract-omission",)

    assert names == MUTATION_CLASSES
    assert len(names) == len(set(names)) == 20
    assert instances == PRODUCTION_MUTATION_INSTANCES
    assert len(instances) == len(set(instances))


@pytest.mark.parametrize(
    "invalidity",
    [
        "type",
        "seal",
        "state_type",
        "missing_state",
        "operation_type",
        "missing_owner_binding",
        "missing_subject_binding",
        "inactive",
        "operation",
        "owner",
        "subject",
        "reused",
        "revoked",
        "expired",
    ],
)
def test_grant_validation_rejects_each_invalid_binding_and_state(invalidity: str) -> None:
    module = importlib.import_module("anonymizer.engine.execution.phase10_inspection")

    class Identity:
        pass

    owner = Identity()
    subject = Identity()
    _bound_identities = (owner, subject)
    operation = module._Phase10Operation.INSPECT
    grant = module._issue_phase10_inspection_grant(owner, subject, operation)
    assert type(grant) is module._Phase10InspectionGrant
    original_grant = grant
    nonce = grant._nonce
    if invalidity == "type":
        grant = SimpleNamespace(operation=operation, _nonce=nonce, _seal=grant._seal)
    elif invalidity == "seal":
        grant = module._Phase10InspectionGrant(operation, nonce, object())
    elif invalidity == "state_type":
        state = module._GRANT_STATES[nonce]
        cast(dict[object, object], module._GRANT_STATES)[nonce] = SimpleNamespace(
            active=True,
            owner_binding=state.owner_binding,
            subject_binding=state.subject_binding,
        )
    elif invalidity == "missing_state":
        module._GRANT_STATES.pop(nonce)
    elif invalidity == "operation_type":
        operation = cast(Any, object())
        grant = module._Phase10InspectionGrant(operation, nonce, original_grant._seal)
    elif invalidity == "missing_owner_binding":
        module._GRANT_STATES[nonce].owner_binding = None
    elif invalidity == "missing_subject_binding":
        module._GRANT_STATES[nonce].subject_binding = None
    elif invalidity == "inactive":
        module._GRANT_STATES[nonce].active = False
    elif invalidity == "operation":
        operation = module._Phase10Operation.DIAGNOSE
    elif invalidity == "owner":
        owner = Identity()
    elif invalidity == "subject":
        subject = Identity()
    elif invalidity == "reused":
        assert module._consume_phase10_inspection_grant(grant, owner, subject, operation)
    elif invalidity == "revoked":
        module._revoke_phase10_inspection_grant(grant)
    elif invalidity == "expired":
        module._expire_phase10_grant_nonce(nonce)
    try:
        accepted = module._consume_phase10_inspection_grant(grant, owner, subject, operation)
        assert accepted is False
    finally:
        module._revoke_phase10_inspection_grant(original_grant)
        module._GRANT_STATES.pop(nonce, None)


@pytest.mark.parametrize("operation_name", ["explain", "inspect", "diagnose"])
def test_each_operation_has_exact_scalar_signature(operation_name: str) -> None:
    module = importlib.import_module("anonymizer.engine.execution.phase10_inspection")
    operation = getattr(module, f"_{operation_name}_phase10")
    signature = inspect.signature(operation)
    assert tuple(signature.parameters) == ("owner", "subject", "grant")
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in signature.parameters.values())
    with pytest.raises(TypeError):
        operation(object(), object(), object(), object())
