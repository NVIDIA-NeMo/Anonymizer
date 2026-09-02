# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, cast

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine import constants as execution_constants
from anonymizer.engine.execution.accounting_evidence import (
    _AttemptId,
    _Dispatch,
    _InvocationId,
    _RowToken,
)
from anonymizer.engine.execution.accounting_plan import _ScopeTaskSubject, _StageId, _TaskKey
from anonymizer.engine.execution.graph import _CoherenceScope
from anonymizer.engine.execution.mention_resolution import _ClusterId
from anonymizer.engine.execution.phase7_admission import _Phase7Plan, _ScopeManifest
from anonymizer.engine.execution.phase7_contract import _load_phase7_contract, _Phase7StableSubstituteContract
from anonymizer.engine.execution.phase7_validation import _BundleRejected, _validate_scope_bundle, _ValidatedBundle
from anonymizer.engine.ndd.adapter import (
    RECORD_ID_COLUMN,
    FailedRecord,
    NddAdapter,
    WorkflowRunResult,
    _FailedRowEvidence,
)
from tests.engine.execution.test_phase7_admission import (
    _compile_phase7,
    _ids,
    _person_relation_fixture,
    _Proposal,
    _qualified_phase6,
)
from tests.engine.execution.test_phase7_validation import _compiled_scope
from tests.streaming.structured_trace_prototype import build_synthetic_anonymizer

_Response = Callable[[pd.DataFrame, list[Any]], WorkflowRunResult]


@dataclass
class _ScriptedAdapter:
    response: _Response
    calls: list[pd.DataFrame] = field(default_factory=list)
    columns: list[list[Any]] = field(default_factory=list)
    workflow_names: list[str] = field(default_factory=list)
    private_depth: int = 0

    @contextmanager
    def private_execution(self) -> Iterator[None]:
        self.private_depth += 1
        try:
            yield
        finally:
            self.private_depth -= 1

    def run_workflow(
        self,
        dataframe: pd.DataFrame,
        *,
        columns: list[Any],
        workflow_name: str,
        **_: Any,
    ) -> WorkflowRunResult:
        assert self.private_depth == 1
        self.calls.append(dataframe.copy())
        self.columns.append(columns)
        self.workflow_names.append(workflow_name)
        return self.response(dataframe.copy(), columns)


def _backend_module() -> ModuleType:
    module_name = "anonymizer.engine.execution.phase7_ndd_backend"
    assert importlib.util.find_spec(module_name) is not None, "the private Phase 7 NDD backend module is missing"
    return importlib.import_module(module_name)


def _column(name: str) -> str:
    value = getattr(execution_constants, name, None)
    assert isinstance(value, str), f"the private Phase 7 {name} column is missing"
    return value


def _invocation() -> Any:
    anonymizer = build_synthetic_anonymizer({})
    return anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False)).invocation


def _dispatch(
    *,
    invocation: str = "invocation-current",
    attempt: str = "attempt-current",
    row: str = "row-current",
) -> _Dispatch:
    return _Dispatch(
        _InvocationId(invocation),
        _TaskKey(_StageId("phase7-plan"), _ScopeTaskSubject()),
        _AttemptId(attempt),
        _RowToken(row),
    )


def _identity_factory(*values: str) -> Callable[[], str]:
    iterator = iter(values)
    return iterator.__next__


def _single_name_scope() -> tuple[_ScopeManifest, tuple[object, ...]]:
    manifest, handoffs = _compiled_scope(
        ("Alice",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"),)},
    )
    return manifest, cast(tuple[object, ...], handoffs)


def _empty_scope() -> tuple[_ScopeManifest, tuple[object, ...]]:
    plan, _backend, execution = _qualified_phase6(("Nothing sensitive",), (("target-0",),), {})
    compiled = _compile_phase7(plan, execution, plan.coherence_scopes)
    assert isinstance(compiled, _Phase7Plan)
    assert len(compiled.manifests) == 1
    assert compiled.manifests[0].slots == ()
    return compiled.manifests[0], cast(tuple[object, ...], execution.handoffs)


def _related_scope() -> tuple[_ScopeManifest, tuple[object, ...]]:
    plan, _backend, execution, raw_cluster = _person_relation_fixture()
    cluster = cast(_ClusterId, raw_cluster)
    module = importlib.import_module("anonymizer.engine.execution.phase7_admission")
    selector = module._ClusterRoleSelector
    relation = module._RelationDeclaration(
        "email_from_name/v1",
        (
            selector("cluster_role/v1", cluster, "person_given_name"),
            selector("cluster_role/v1", cluster, "person_family_name"),
        ),
        selector("cluster_role/v1", cluster, "email_address"),
    )
    compiled = _compile_phase7(plan, execution, (_CoherenceScope(_ids(plan)),), (relation,))
    assert isinstance(compiled, _Phase7Plan)
    return compiled.manifests[0], cast(tuple[object, ...], execution.handoffs)


def _same_role_scope() -> tuple[_ScopeManifest, tuple[object, ...]]:
    return _compiled_scope(
        ("Alice", "Bob"),
        (("target-0",), ("target-1",)),
        {
            "target-0": (_Proposal("Alice", "first_name", "person-a"),),
            "target-1": (_Proposal("Bob", "first_name", "person-b"),),
        },
        combined_scope=True,
    )


def _request(frame: pd.DataFrame) -> dict[str, Any]:
    value = frame.iloc[0][_column("COL_PHASE7_CANDIDATE_REQUEST")]
    assert isinstance(value, str)
    parsed = json.loads(value)
    assert isinstance(parsed, dict)
    return parsed


def _success_response(values_by_role: dict[str, str]) -> _Response:
    def respond(frame: pd.DataFrame, _columns: list[Any]) -> WorkflowRunResult:
        request = _request(frame)
        assignments = [
            {"slot_token": slot["slot_token"], "value": values_by_role[slot["role"]]}
            for slot in reversed(request["slots"])
        ]
        output = frame.copy()
        output[_column("COL_PHASE7_CANDIDATE_BUNDLE")] = [{"assignments": assignments}]
        output["_anonymizer_record_id"] = ["backend-randomized-id"]
        return WorkflowRunResult(output, [])

    return respond


def _backend(adapter: _ScriptedAdapter, *identities: str) -> Any:
    backend_type = getattr(_backend_module(), "_Phase7NddBackend")
    return backend_type(
        cast(NddAdapter, adapter),
        _invocation(),
        identity_factory=_identity_factory(*identities),
    )


def _propose(
    backend: Any,
    manifest: object,
    handoffs: object,
    dispatch: object,
) -> object:
    return backend.propose_scope(manifest, handoffs, _load_phase7_contract(), dispatch)


def _status(result: object) -> str:
    status = getattr(result, "status", None)
    value = getattr(status, "value", None)
    assert isinstance(value, str)
    return value


def _reason(result: object) -> str | None:
    reason = getattr(result, "reason", None)
    if reason is None:
        return None
    value = getattr(reason, "value", None)
    assert isinstance(value, str)
    return value


def _assignment_pairs(result: object) -> tuple[tuple[object, str], ...]:
    assignments = getattr(result, "assignments", None)
    assert isinstance(assignments, tuple)
    return tuple((assignment.token, assignment.value) for assignment in assignments)


def test_phase7_zero_slot_scope_is_no_work_without_an_attempt_or_adapter_call() -> None:
    manifest, handoffs = _empty_scope()
    adapter = _ScriptedAdapter(lambda _frame, _columns: pytest.fail("empty scope called the adapter"))
    backend = _backend(adapter)

    result = _propose(backend, manifest, handoffs, None)

    assert _status(result) == "no_work"
    assert _assignment_pairs(result) == ()
    assert adapter.calls == []


def test_phase7_nonempty_scope_uses_one_private_single_row_adapter_call() -> None:
    manifest, handoffs = _single_name_scope()
    adapter = _ScriptedAdapter(_success_response({"person_given_name": "Mira"}))
    backend = _backend(adapter, "task-token", "slot-token")

    result = _propose(backend, manifest, handoffs, _dispatch())

    assert _status(result) == "candidate"
    assert len(adapter.calls) == 1
    assert len(adapter.calls[0]) == 1
    assert adapter.workflow_names == ["phase7-candidate-planning"]
    assert adapter.private_depth == 0


def test_phase7_two_nonempty_scopes_each_cross_the_adapter_once() -> None:
    plan, _phase6_backend, execution = _qualified_phase6(
        ("Alice", "Bob"),
        (("target-0",), ("target-1",)),
        {
            "target-0": (_Proposal("Alice", "first_name", "person-a"),),
            "target-1": (_Proposal("Bob", "first_name", "person-b"),),
        },
    )
    compiled = _compile_phase7(plan, execution, plan.coherence_scopes)
    assert isinstance(compiled, _Phase7Plan)
    adapter = _ScriptedAdapter(_success_response({"person_given_name": "Mira"}))
    backend = _backend(adapter, "task-a", "slot-a", "task-b", "slot-b")

    results = tuple(
        _propose(
            backend,
            manifest,
            execution.handoffs,
            _dispatch(attempt=f"attempt-{index}", row=f"row-{index}"),
        )
        for index, manifest in enumerate(compiled.manifests)
    )

    assert tuple(_status(result) for result in results) == ("candidate", "candidate")
    assert len(adapter.calls) == 2
    assert all(len(frame) == 1 for frame in adapter.calls)


def test_phase7_workframe_contains_only_opaque_correlations_and_governed_request_material() -> None:
    manifest, handoffs = _single_name_scope()
    adapter = _ScriptedAdapter(_success_response({"person_given_name": "Mira"}))
    backend = _backend(adapter, "task-token", "slot-token")

    _propose(backend, manifest, handoffs, _dispatch())

    frame = adapter.calls[0]
    assert tuple(frame.columns) == (
        execution_constants.COL_TARGET_WORK_ID,
        _column("COL_PHASE7_INVOCATION_ID"),
        execution_constants.COL_TASK_ID,
        execution_constants.COL_ATTEMPT_ID,
        _column("COL_PHASE7_CANDIDATE_REQUEST"),
        RECORD_ID_COLUMN,
    )
    assert frame.iloc[0][execution_constants.COL_TARGET_WORK_ID] == "row-current"
    assert frame.iloc[0][_column("COL_PHASE7_INVOCATION_ID")] == "invocation-current"
    assert frame.iloc[0][execution_constants.COL_TASK_ID] == "task-token"
    assert frame.iloc[0][execution_constants.COL_ATTEMPT_ID] == "attempt-current"
    request = _request(frame)
    assert set(request) == {"schema_version", "slots", "required_distinct_pairs", "relations"}
    assert request["schema_version"] == "phase7-workframe/v1"
    assert request["slots"] == [
        {
            "slot_token": "slot-token",
            "role": "person_given_name",
            "format": "unicode_person_name/v1",
            "mask": "none/v1",
            "source_values": ["Alice"],
        }
    ]
    serialized = json.dumps(request, sort_keys=True)
    assert "target-0" not in serialized
    assert "source_id" not in serialized
    assert "mention" not in serialized
    assert "cluster" not in serialized
    assert frame.iloc[0][RECORD_ID_COLUMN] == "row-current"


def test_phase7_adapter_tracking_identity_is_opaque_and_not_content_derived() -> None:
    manifest, handoffs = _single_name_scope()
    adapter = _ScriptedAdapter(_success_response({"person_given_name": "Mira"}))
    backend = _backend(adapter, "task-token", "slot-token")

    _propose(backend, manifest, handoffs, _dispatch())

    frame = adapter.calls[0]
    real_adapter = object.__new__(NddAdapter)
    attached = real_adapter._attach_record_ids(frame)
    assert attached[RECORD_ID_COLUMN].tolist() == ["row-current"]
    assert "Alice" not in attached[RECORD_ID_COLUMN].iloc[0]


def test_phase7_ndd_declaration_uses_the_replacement_model_and_governed_request_column() -> None:
    manifest, handoffs = _single_name_scope()
    adapter = _ScriptedAdapter(_success_response({"person_given_name": "Mira"}))
    backend = _backend(adapter, "task-token", "slot-token")

    _propose(backend, manifest, handoffs, _dispatch())

    assert len(adapter.columns[0]) == 1
    column = adapter.columns[0][0]
    assert column.name == _column("COL_PHASE7_CANDIDATE_BUNDLE")
    assert _column("COL_PHASE7_CANDIDATE_REQUEST") in str(column.prompt)
    assert column.model_alias == _invocation().selected_models.replace.replacement_generator


def test_phase7_hydrates_permuted_assignments_by_ephemeral_slot_token() -> None:
    manifest, handoffs = _related_scope()
    values = {
        "person_given_name": "Mira",
        "person_family_name": "Stone",
        "email_address": "mira.stone@example.com",
    }
    adapter = _ScriptedAdapter(_success_response(values))
    backend = _backend(adapter, "task-token", "slot-a", "slot-b", "slot-c")

    result = _propose(backend, manifest, handoffs, _dispatch())

    assert _status(result) == "candidate"
    assert _assignment_pairs(result) == tuple((slot.id, values[slot.role]) for slot in manifest.slots)


def test_phase7_same_role_slots_hydrate_by_token_instead_of_role_or_order() -> None:
    manifest, handoffs = _same_role_scope()

    def respond(frame: pd.DataFrame, _columns: list[Any]) -> WorkflowRunResult:
        slots = _request(frame)["slots"]
        output = frame.copy()
        output[_column("COL_PHASE7_CANDIDATE_BUNDLE")] = [
            {
                "assignments": [
                    {"slot_token": slots[1]["slot_token"], "value": "Nora"},
                    {"slot_token": slots[0]["slot_token"], "value": "Mira"},
                ]
            }
        ]
        return WorkflowRunResult(output, [])

    result = _propose(
        _backend(_ScriptedAdapter(respond), "task-token", "slot-a", "slot-b"),
        manifest,
        handoffs,
        _dispatch(),
    )

    assert _assignment_pairs(result) == ((manifest.slots[0].id, "Mira"), (manifest.slots[1].id, "Nora"))


def test_phase7_backend_ids_and_output_column_order_are_not_identity() -> None:
    manifest, handoffs = _single_name_scope()

    def respond(frame: pd.DataFrame, _columns: list[Any]) -> WorkflowRunResult:
        request = _request(frame)
        output = pd.DataFrame(
            {
                "_anonymizer_record_id": ["untrusted-random-backend-id"],
                _column("COL_PHASE7_CANDIDATE_BUNDLE"): [
                    {"assignments": [{"slot_token": request["slots"][0]["slot_token"], "value": "Mira"}]}
                ],
                execution_constants.COL_ATTEMPT_ID: ["attempt-current"],
                execution_constants.COL_TASK_ID: ["task-token"],
                _column("COL_PHASE7_INVOCATION_ID"): ["invocation-current"],
                execution_constants.COL_TARGET_WORK_ID: ["row-current"],
            }
        )
        return WorkflowRunResult(output, [])

    result = _propose(_backend(_ScriptedAdapter(respond), "task-token", "slot-token"), manifest, handoffs, _dispatch())

    assert _status(result) == "candidate"
    assert _assignment_pairs(result) == ((manifest.slots[0].id, "Mira"),)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "foreign"])
def test_phase7_rejects_nonexact_slot_token_hydration_as_invocation_inconsistent(fault: str) -> None:
    manifest, handoffs = _single_name_scope()

    def respond(frame: pd.DataFrame, _columns: list[Any]) -> WorkflowRunResult:
        token = _request(frame)["slots"][0]["slot_token"]
        assignments: list[dict[str, str]] = []
        if fault != "missing":
            assignments.append({"slot_token": "foreign-slot" if fault == "foreign" else token, "value": "Mira"})
        if fault == "duplicate":
            assignments.append({"slot_token": token, "value": "Other"})
        output = frame.copy()
        output[_column("COL_PHASE7_CANDIDATE_BUNDLE")] = [{"assignments": assignments}]
        return WorkflowRunResult(output, [])

    result = _propose(_backend(_ScriptedAdapter(respond), "task-token", "slot-token"), manifest, handoffs, _dispatch())

    assert _status(result) == "invocation_inconsistent"
    assert _reason(result) == "evidence_unattributable"
    assert _assignment_pairs(result) == ()


@pytest.mark.parametrize(
    ("column_name", "value"),
    [
        ("COL_PHASE7_INVOCATION_ID", "invocation-foreign"),
        ("COL_TASK_ID", "task-foreign"),
        ("COL_ATTEMPT_ID", "attempt-stale"),
        ("COL_TARGET_WORK_ID", "row-foreign"),
    ],
)
def test_phase7_rejects_foreign_or_stale_row_correlations(column_name: str, value: str) -> None:
    manifest, handoffs = _single_name_scope()

    def respond(frame: pd.DataFrame, _columns: list[Any]) -> WorkflowRunResult:
        token = _request(frame)["slots"][0]["slot_token"]
        output = frame.copy()
        output[_column(column_name)] = [value]
        output[_column("COL_PHASE7_CANDIDATE_BUNDLE")] = [{"assignments": [{"slot_token": token, "value": "Mira"}]}]
        return WorkflowRunResult(output, [])

    result = _propose(_backend(_ScriptedAdapter(respond), "task-token", "slot-token"), manifest, handoffs, _dispatch())

    assert _status(result) == "invocation_inconsistent"
    assert _assignment_pairs(result) == ()


@pytest.mark.parametrize("column_name", ["COL_PHASE7_INVOCATION_ID", "COL_ATTEMPT_ID", "COL_TARGET_WORK_ID"])
def test_phase7_missing_scalar_correlations_fail_closed(column_name: str) -> None:
    manifest, handoffs = _single_name_scope()

    def respond(frame: pd.DataFrame, _columns: list[Any]) -> WorkflowRunResult:
        token = _request(frame)["slots"][0]["slot_token"]
        output = frame.copy()
        output[_column(column_name)] = [pd.NA]
        output[_column("COL_PHASE7_CANDIDATE_BUNDLE")] = [{"assignments": [{"slot_token": token, "value": "Mira"}]}]
        return WorkflowRunResult(output, [])

    result = _propose(_backend(_ScriptedAdapter(respond), "task-token", "slot-token"), manifest, handoffs, _dispatch())

    assert _status(result) == "invocation_inconsistent"
    assert _reason(result) == "evidence_unattributable"


def test_phase7_rejects_duplicate_success_rows_even_when_payloads_match() -> None:
    manifest, handoffs = _single_name_scope()

    def respond(frame: pd.DataFrame, _columns: list[Any]) -> WorkflowRunResult:
        token = _request(frame)["slots"][0]["slot_token"]
        output = pd.concat([frame, frame], ignore_index=True)
        output[_column("COL_PHASE7_CANDIDATE_BUNDLE")] = [
            {"assignments": [{"slot_token": token, "value": "Mira"}]},
            {"assignments": [{"slot_token": token, "value": "Mira"}]},
        ]
        return WorkflowRunResult(output, [])

    result = _propose(_backend(_ScriptedAdapter(respond), "task-token", "slot-token"), manifest, handoffs, _dispatch())

    assert _status(result) == "invocation_inconsistent"


def test_phase7_trusted_call_token_attributes_exactly_one_complete_scope_task_failure() -> None:
    manifest, handoffs = _single_name_scope()
    failure = FailedRecord("untrusted-backend-record", "phase7-candidate-planning", "dropped")

    def respond(frame: pd.DataFrame, _columns: list[Any]) -> WorkflowRunResult:
        return WorkflowRunResult(frame.iloc[0:0].copy(), [failure], (_FailedRowEvidence("row-current", failure),))

    result = _propose(_backend(_ScriptedAdapter(respond), "task-token", "slot-token"), manifest, handoffs, _dispatch())

    assert _status(result) == "task_failed"
    assert _reason(result) == "backend_failed"
    assert _assignment_pairs(result) == ()


def test_phase7_rejects_a_non_failed_record_even_with_matching_private_evidence() -> None:
    manifest, handoffs = _single_name_scope()
    malformed = object()

    def respond(frame: pd.DataFrame, _columns: list[Any]) -> WorkflowRunResult:
        return WorkflowRunResult(
            frame.iloc[0:0].copy(),
            [cast(FailedRecord, malformed)],
            (_FailedRowEvidence("row-current", cast(FailedRecord, malformed)),),
        )

    result = _propose(_backend(_ScriptedAdapter(respond), "task-token", "slot-token"), manifest, handoffs, _dispatch())

    assert _status(result) == "invocation_inconsistent"
    assert _reason(result) == "evidence_unattributable"


def test_phase7_rejects_missing_failed_evidence_correlation_without_raising() -> None:
    manifest, handoffs = _single_name_scope()
    failure = FailedRecord("untrusted-backend-record", "phase7-candidate-planning", "dropped")

    def respond(frame: pd.DataFrame, _columns: list[Any]) -> WorkflowRunResult:
        return WorkflowRunResult(
            frame.iloc[0:0].copy(),
            [failure],
            (_FailedRowEvidence(cast(str, pd.NA), failure),),
        )

    result = _propose(_backend(_ScriptedAdapter(respond), "task-token", "slot-token"), manifest, handoffs, _dispatch())

    assert _status(result) == "invocation_inconsistent"
    assert _reason(result) == "evidence_unattributable"


@pytest.mark.parametrize(
    "fault",
    ["unattributed", "foreign", "stale", "duplicate", "duplicate_records", "success_plus_failure"],
)
def test_phase7_ambiguous_failed_record_evidence_causes_global_inconsistency(fault: str) -> None:
    manifest, handoffs = _single_name_scope()
    failure = FailedRecord("untrusted-backend-record", "phase7-candidate-planning", "dropped")

    def respond(frame: pd.DataFrame, _columns: list[Any]) -> WorkflowRunResult:
        failed = [failure]
        evidence: tuple[_FailedRowEvidence, ...] = ()
        output = frame.iloc[0:0].copy()
        if fault == "foreign":
            evidence = (_FailedRowEvidence("row-foreign", failure),)
        elif fault == "stale":
            evidence = (_FailedRowEvidence("row-from-stale-attempt", failure),)
        elif fault == "duplicate":
            evidence = (_FailedRowEvidence("row-current", failure), _FailedRowEvidence("row-current", failure))
        elif fault == "duplicate_records":
            failed = [failure, failure]
            evidence = (_FailedRowEvidence("row-current", failure),)
        elif fault == "success_plus_failure":
            token = _request(frame)["slots"][0]["slot_token"]
            output = frame.copy()
            output[_column("COL_PHASE7_CANDIDATE_BUNDLE")] = [{"assignments": [{"slot_token": token, "value": "Mira"}]}]
            evidence = (_FailedRowEvidence("row-current", failure),)
        return WorkflowRunResult(output, failed, evidence)

    result = _propose(_backend(_ScriptedAdapter(respond), "task-token", "slot-token"), manifest, handoffs, _dispatch())

    assert _status(result) == "invocation_inconsistent"
    assert _reason(result) == "evidence_unattributable"
    assert _assignment_pairs(result) == ()


def test_phase7_failed_row_evidence_without_a_failure_is_inconsistent() -> None:
    manifest, handoffs = _single_name_scope()
    failure = FailedRecord("untrusted-backend-record", "phase7-candidate-planning", "dropped")

    def respond(frame: pd.DataFrame, _columns: list[Any]) -> WorkflowRunResult:
        return WorkflowRunResult(frame, [], (_FailedRowEvidence("row-current", failure),))

    result = _propose(_backend(_ScriptedAdapter(respond), "task-token", "slot-token"), manifest, handoffs, _dispatch())

    assert _status(result) == "invocation_inconsistent"


def test_phase7_adapter_exception_is_sanitized_to_the_current_task_failure() -> None:
    manifest, handoffs = _single_name_scope()

    def fail(_frame: pd.DataFrame, _columns: list[Any]) -> WorkflowRunResult:
        raise RuntimeError("SECRET-CANDIDATE-and-original")

    adapter = _ScriptedAdapter(fail)
    result = _propose(_backend(adapter, "task-token", "slot-token"), manifest, handoffs, _dispatch())

    assert _status(result) == "task_failed"
    assert _reason(result) == "backend_failed"
    assert "SECRET" not in repr(result)
    assert len(adapter.calls) == 1


def test_phase7_rejects_an_over_limit_workframe_before_adapter_execution() -> None:
    manifest, handoffs = _single_name_scope()
    adapter = _ScriptedAdapter(lambda _frame, _columns: pytest.fail("over-limit workframe reached adapter"))
    backend = _backend(adapter, "x" * 20_000, "slot-token")

    result = _propose(backend, manifest, handoffs, _dispatch())

    assert _status(result) == "invocation_inconsistent"
    assert _reason(result) == "limit_exceeded"
    assert adapter.calls == []


def test_phase7_workframe_byte_ceiling_accepts_exactly_limit_and_rejects_one_over() -> None:
    contract = _load_phase7_contract()
    assert isinstance(contract, _Phase7StableSubstituteContract)
    limit = dict(contract.byte_limits)["max_workframe_bytes_per_scope"]
    row = {
        execution_constants.COL_TARGET_WORK_ID: "row-current",
        _column("COL_PHASE7_INVOCATION_ID"): "invocation-current",
        execution_constants.COL_TASK_ID: "task-token",
        execution_constants.COL_ATTEMPT_ID: "attempt-current",
        _column("COL_PHASE7_CANDIDATE_REQUEST"): "",
        RECORD_ID_COLUMN: "row-current",
    }
    fixed_bytes = len(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    exact_request = "x" * (limit - fixed_bytes)
    dataframe = _backend_module()._candidate_dataframe(
        _dispatch(),
        "task-token",
        exact_request,
        max_bytes=limit,
    )
    rejected = _backend_module()._candidate_dataframe(
        _dispatch(),
        "task-token",
        exact_request + "x",
        max_bytes=limit,
    )

    assert isinstance(dataframe, pd.DataFrame)
    assert rejected is None


@pytest.mark.parametrize(
    "identities",
    [
        ("", "slot-token"),
        ("row-current", "slot-token"),
        ("task-token", "task-token"),
        ("task-token", cast(str, None)),
    ],
)
def test_phase7_invalid_or_colliding_generated_identity_fails_before_adapter(
    identities: tuple[str, str],
) -> None:
    manifest, handoffs = _single_name_scope()
    adapter = _ScriptedAdapter(lambda _frame, _columns: pytest.fail("invalid identity reached adapter"))

    result = _propose(_backend(adapter, *identities), manifest, handoffs, _dispatch())

    assert _status(result) == "invocation_inconsistent"
    assert _reason(result) == "limit_exceeded"
    assert adapter.calls == []


def test_phase7_structural_output_parse_failure_is_content_free_and_inconsistent() -> None:
    manifest, handoffs = _single_name_scope()

    def respond(frame: pd.DataFrame, _columns: list[Any]) -> WorkflowRunResult:
        output = frame.copy()
        output[_column("COL_PHASE7_CANDIDATE_BUNDLE")] = [
            {"assignments": [{"slot_token": "slot-token", "value": b"SECRET-CANDIDATE"}]}
        ]
        return WorkflowRunResult(output, [])

    result = _propose(_backend(_ScriptedAdapter(respond), "task-token", "slot-token"), manifest, handoffs, _dispatch())

    assert _status(result) == "invocation_inconsistent"
    assert _reason(result) == "evidence_unattributable"
    assert "SECRET" not in repr(result)


def test_phase7_hydration_is_proposal_only_and_p5_remains_validation_authority() -> None:
    manifest, handoffs = _single_name_scope()
    adapter = _ScriptedAdapter(_success_response({"person_given_name": "Alice"}))
    result = _propose(_backend(adapter, "task-token", "slot-token"), manifest, handoffs, _dispatch())

    assert _status(result) == "candidate"
    p5_result = _validate_scope_bundle(
        manifest,
        handoffs,
        getattr(result, "assignments"),
        _load_phase7_contract(),
    )
    assert isinstance(p5_result, _BundleRejected)
    assert p5_result.code.value == "candidate_matches_original"


def test_phase7_hydrated_valid_candidate_can_cross_the_unchanged_p5_boundary() -> None:
    manifest, handoffs = _single_name_scope()
    adapter = _ScriptedAdapter(_success_response({"person_given_name": "Mira"}))
    result = _propose(_backend(adapter, "task-token", "slot-token"), manifest, handoffs, _dispatch())

    p5_result = _validate_scope_bundle(
        manifest,
        handoffs,
        getattr(result, "assignments"),
        _load_phase7_contract(),
    )

    assert isinstance(p5_result, _ValidatedBundle)


def test_phase7_backend_has_no_direct_datadesigner_execution_path() -> None:
    source = inspect.getsource(_backend_module())

    assert "DataDesigner.create" not in source
    assert "DataDesigner.preview" not in source
    assert ".create(" not in source
    assert ".preview(" not in source
