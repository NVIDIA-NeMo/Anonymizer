# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import logging
import pickle
from collections.abc import Iterator
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import (
    COL_ATTEMPT_ID,
    COL_DETECTED_ENTITIES,
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_PHASE7_CANDIDATE_REQUEST,
    COL_PHASE7_INVOCATION_ID,
    COL_REPLACEMENT_APPLICATION,
    COL_REPLACEMENT_MAP,
    COL_TAGGED_TEXT,
    COL_TARGET_WORK_ID,
    COL_TASK_ID,
    COL_TEXT,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.execution.phase7_runtime import _Phase7CleanupAttestation
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.engine.replace.strategies import apply_replacement_map
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from anonymizer.interface._protection import _Failed, _ProtectionFlow, _ProtectionPlan, _ProtectionSucceeded
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.measurement.collector import MeasurementCollector
from anonymizer.measurement.session import measurement_session
from tests.engine.execution.test_phase7_admission import _Proposal
from tests.engine.execution.test_phase7_ndd_backend import (
    _backend,
    _dispatch,
    _propose,
    _request,
    _ScriptedAdapter,
    _success_response,
)
from tests.engine.execution.test_phase7_validation import _compiled_scope
from tests.interface.test_phase6_public_compatibility import _write_input
from tests.interface.test_phase7_private_wiring import _CandidateBackend, _private_substitute_flow
from tests.interface.test_private_protection import _AnchoredPhase6Backend, _record
from tests.streaming.structured_trace_prototype import build_synthetic_anonymizer

_ORIGINAL = "P8-ORIGINAL-2f106d9b91b64e5988c4e4e4264cb7f1@example.test"
_SYNTHETIC = "P8-SYNTHETIC-a98765c2c93945d4a4d86a260267451b@example.test"
_PROMPT = "P8-PROMPT-51ed1eb892ca481483281142ccb3e109"
_SOURCE_ID = "P8-SOURCE-ID-7ba6ec61e2aa43c289f28612f895b607"
_DIGEST_MATERIAL = "P8-DIGEST-MATERIAL-2cbabdb08e10433e97cba79b2a5e23fe"
_CONTENT_DIGESTS = tuple(
    hashlib.sha256(value.encode("utf-8")).hexdigest()
    for value in (_ORIGINAL, _SYNTHETIC, _PROMPT, _SOURCE_ID, _DIGEST_MATERIAL)
)


def _leaf_paths(value: object, path: str = "root") -> Iterator[tuple[str, object]]:
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _leaf_paths(getattr(value, field.name), f"{path}.{field.name}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaf_paths(item, f"{path}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _leaf_paths(item, f"{path}[{index}]")
        return
    yield path, value


def _frame_leaf_paths(frame: pd.DataFrame, name: str) -> Iterator[tuple[str, object]]:
    for row_position in range(len(frame)):
        for column in frame.columns:
            yield from _leaf_paths(frame.iloc[row_position][column], f"{name}[{row_position}].{column}")


def _assert_forbidden_content_absent(value: object) -> None:
    rendered = repr(value)
    for forbidden in (_ORIGINAL, _SYNTHETIC, _PROMPT, _DIGEST_MATERIAL, *_CONTENT_DIGESTS):
        assert forbidden not in rendered


def _public_substitute_anonymizer() -> Anonymizer:
    detection = Mock(spec=EntityDetectionWorkflow)

    def detect(dataframe: pd.DataFrame, **_kwargs: object) -> EntityDetectionResult:
        output = dataframe.copy()
        entity = {
            "id": "public-entity",
            "value": _ORIGINAL,
            "label": "email",
            "start_position": 0,
            "end_position": len(_ORIGINAL),
            "score": 1.0,
            "source": "privacy-test",
        }
        output[COL_DETECTED_ENTITIES] = [{"entities": [entity]}]
        output[COL_FINAL_ENTITIES] = [{"entities": [entity]}]
        output[COL_ENTITIES_BY_VALUE] = [
            {"entities_by_value": [{"value": _ORIGINAL, "labels": ["email"]}]}
        ]
        output[COL_TAGGED_TEXT] = [f"<email>{_ORIGINAL}</email>"]
        return EntityDetectionResult(output, [])

    detection.run.side_effect = detect
    replacement = Mock(spec=ReplacementWorkflow)

    def replace(dataframe: pd.DataFrame, **_kwargs: object) -> ReplacementResult:
        output = dataframe.copy()
        output[COL_REPLACEMENT_MAP] = [
            {
                "replacements": [
                    {"original": _ORIGINAL, "label": "email", "synthetic": _SYNTHETIC}
                ]
            }
        ]
        return ReplacementResult(apply_replacement_map(output), [])

    replacement.run.side_effect = replace
    return Anonymizer(
        data_designer=Mock(),
        detection_workflow=detection,
        replace_runner=replacement,
        rewrite_runner=Mock(spec=RewriteWorkflow),
    )


def test_private_release_and_serialization_have_an_exact_content_allowlist(
    caplog: pytest.LogCaptureFixture,
) -> None:
    flow, backend = _private_substitute_flow(original=_ORIGINAL, synthetic=_SYNTHETIC, label="email")
    collector = MeasurementCollector(run_id="phase7-p8-private-release")

    with caplog.at_level(logging.DEBUG), measurement_session(collector):
        result = flow.protect((_record(_SOURCE_ID, _ORIGINAL),))

    outcome = result.outcomes[0]
    assert isinstance(outcome, _ProtectionSucceeded)
    leaves = list(_leaf_paths(result))
    assert {path for path, value in leaves if value == _SYNTHETIC} == {"root.outcomes[0].output"}
    assert {path for path, value in leaves if value == _SOURCE_ID} == {"root.outcomes[0].ref.value"}
    assert not {path for path, value in leaves if value == _ORIGINAL}
    assert set(field.name for field in fields(outcome.receipt)) == {
        "contract_version",
        "profile",
        "implementation_version",
        "terminal_accounting_verified",
        "accepted_detections_verified",
        "plan_digest",
        "attempt_id",
    }
    assert backend.value is None

    restored = pickle.loads(pickle.dumps(result))
    restored_leaves = list(_leaf_paths(restored))
    assert {path for path, value in restored_leaves if value == _SYNTHETIC} == {"root.outcomes[0].output"}
    assert {path for path, value in restored_leaves if value == _SOURCE_ID} == {"root.outcomes[0].ref.value"}
    diagnostic_surfaces = (
        caplog.text,
        collector.records,
        outcome.receipt,
        repr(result),
    )
    for surface in diagnostic_surfaces:
        _assert_forbidden_content_absent(surface)
        assert _SOURCE_ID not in repr(surface)
    allowed_metric_fields = {
        "boundary",
        "byte_count_bucket",
        "cleanup",
        "context_count_bucket",
        "duration_sec",
        "event",
        "implementation_profile",
        "observation_schema",
        "outcome",
        "reason",
        "reconciliation",
        "record_type",
        "route",
        "run_id",
        "run_tags",
        "schema_version",
        "semantic_profile",
        "target_count_bucket",
        "timestamp_unix_sec",
    }
    assert collector.records
    assert all(set(record) <= allowed_metric_fields for record in collector.records)


@dataclass
class _CleanupFailingBackend(_CandidateBackend):
    def discard_values(self) -> None:
        super().discard_values()
        try:
            raise ValueError(_ORIGINAL)
        except ValueError as cause:
            raise RuntimeError(_PROMPT) from cause


def test_cleanup_error_and_withheld_result_expose_no_candidate_or_exception_chain(
    caplog: pytest.LogCaptureFixture,
) -> None:
    anonymizer = build_synthetic_anonymizer({_ORIGINAL: "email"})
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Substitute(), emit_telemetry=False))
    assert isinstance(plan, _ProtectionPlan)
    backend = _CleanupFailingBackend(_SYNTHETIC)
    flow = _ProtectionFlow(
        anonymizer,
        plan,
        phase6_backend=_AnchoredPhase6Backend.from_entities({_ORIGINAL: "email"}),
        phase7_backend=backend,
    )
    collector = MeasurementCollector(run_id="phase7-p8-cleanup-failure")

    with caplog.at_level(logging.DEBUG), measurement_session(collector):
        result = flow.protect((_record(_SOURCE_ID, _ORIGINAL),))

    outcome = result.outcomes[0]
    assert isinstance(outcome, _Failed)
    assert set(field.name for field in fields(outcome)) == {"ref", "failure"}
    assert not hasattr(outcome, "output")
    assert not hasattr(outcome, "receipt")
    assert backend.value is None
    leaves = list(_leaf_paths(result))
    assert {path for path, value in leaves if value == _SOURCE_ID} == {"root.outcomes[0].ref.value"}
    assert not any(isinstance(value, BaseException) for _path, value in leaves)
    for surface in (caplog.text, collector.records, outcome.failure, repr(result), pickle.dumps(result)):
        _assert_forbidden_content_absent(surface)


def test_active_workframe_planner_state_and_cleanup_follow_structural_allowlists() -> None:
    manifest, handoffs = _compiled_scope(
        (_ORIGINAL,),
        (("target-0",),),
        {"target-0": (_Proposal(_ORIGINAL, "email", "email-cluster"),)},
    )
    adapter = _ScriptedAdapter(_success_response({"email_address": _SYNTHETIC}))
    backend = _backend(
        adapter,
        "task-correlation-33d423175b2a48c1a21fba6f202231fb",
        "slot-correlation-6a041a838dc8460a804349ff59e58fcb",
    )

    result = _propose(
        backend,
        manifest,
        handoffs,
        _dispatch(
            invocation="invocation-correlation-bf56bebc0644465ea109b1daec9633ff",
            attempt="attempt-correlation-9a3bf13e16034c0984dcc71a5d971476",
            row="row-correlation-85ca65005ce24fdb98eddf264e352bc3",
        ),
    )

    frame = adapter.calls[0]
    assert set(frame.columns) == {
        COL_TARGET_WORK_ID,
        COL_PHASE7_INVOCATION_ID,
        COL_TASK_ID,
        COL_ATTEMPT_ID,
        COL_PHASE7_CANDIDATE_REQUEST,
        RECORD_ID_COLUMN,
    }
    request = _request(frame)
    assert set(request) == {"schema_version", "slots", "required_distinct_pairs", "relations"}
    assert len(request["slots"]) == 1
    assert set(request["slots"][0]) == {"slot_token", "role", "format", "mask", "source_values"}
    request_leaves = list(_leaf_paths(request, "request"))
    assert {path for path, value in request_leaves if value == _ORIGINAL} == {
        "request['slots'][0]['source_values'][0]"
    }
    for forbidden in (_SYNTHETIC, _PROMPT, _SOURCE_ID, _DIGEST_MATERIAL, *_CONTENT_DIGESTS):
        assert forbidden not in repr(request)

    active = backend._planner.current(manifest.id)
    active_leaves = list(_leaf_paths(active, "planner"))
    assert {path for path, value in active_leaves if value == _SYNTHETIC} == {
        "planner.value.assignments[0].value"
    }
    assert not any(value == _ORIGINAL for _path, value in active_leaves)
    result_leaves = list(_leaf_paths(result, "candidate"))
    assert {path for path, value in result_leaves if value == _SYNTHETIC} == {
        "candidate.assignments[0].value"
    }
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(result)

    backend.close()
    backend.discard_values()
    cleanup_identity = object()
    attestation = backend.cleanup_attestation(cleanup_identity)

    assert isinstance(attestation, _Phase7CleanupAttestation)
    assert attestation.verified
    assert attestation.cleanup_identity is cleanup_identity
    assert backend._planner.cleanup_observation() == (0, 0, False)
    retired = backend._planner.current(manifest.id)
    assert retired is not None
    assert retired.value is None


def test_public_result_trace_and_serialization_exclude_every_private_identity_and_digest(
    tmp_path: Path,
) -> None:
    source = pd.DataFrame({"record_id": [_SOURCE_ID], "text": [_ORIGINAL]})
    data = _write_input(source, tmp_path / "input.parquet", "parquet")
    anonymizer = _public_substitute_anonymizer()
    private_plan = anonymizer._compile_protection_plan(
        AnonymizerConfig(replace=Substitute(), emit_telemetry=False)
    )
    assert isinstance(private_plan, _ProtectionPlan)
    assert private_plan.phase7_contract is not None

    result = anonymizer.run(
        config=AnonymizerConfig(replace=Substitute(instructions=_PROMPT), emit_telemetry=False),
        data=data,
    )

    assert {field.name for field in fields(result)} == {
        "dataframe",
        "trace_dataframe",
        "resolved_text_column",
        "failed_records",
        "replace_method",
        "rewrite_config",
        "entity_labels",
        "data_summary",
        "_display_cycle_index",
    }
    assert list(result.dataframe.columns) == ["text", COL_FINAL_ENTITIES, "text_with_spans", "text_replaced"]
    assert list(result.trace_dataframe.columns) == [
        "record_id",
        "text",
        COL_DETECTED_ENTITIES,
        COL_FINAL_ENTITIES,
        COL_ENTITIES_BY_VALUE,
        "text_with_spans",
        COL_REPLACEMENT_MAP,
        "text_replaced",
        COL_REPLACEMENT_APPLICATION,
    ]
    leaves = [
        *list(_frame_leaf_paths(result.dataframe, "dataframe")),
        *list(_frame_leaf_paths(result.trace_dataframe, "trace")),
    ]
    synthetic_paths = {path for path, value in leaves if value == _SYNTHETIC}
    assert synthetic_paths == {
        "dataframe[0].text_replaced",
        "trace[0]._replacement_map['replacements'][0]['synthetic']",
        "trace[0].text_replaced",
    }
    source_paths = {path for path, value in leaves if value == _SOURCE_ID}
    assert source_paths == {"trace[0].record_id"}
    allowed_original_columns = {
        "text",
        COL_DETECTED_ENTITIES,
        COL_FINAL_ENTITIES,
        COL_ENTITIES_BY_VALUE,
        "text_with_spans",
        COL_REPLACEMENT_MAP,
    }
    assert all(path.rsplit(".", 1)[-1].split("[", 1)[0] in allowed_original_columns for path, value in leaves if value == _ORIGINAL)

    serialized = pickle.loads(pickle.dumps(result))
    assert list(serialized.dataframe.columns) == list(result.dataframe.columns)
    assert list(serialized.trace_dataframe.columns) == list(result.trace_dataframe.columns)
    rendered = (result.dataframe.to_json() or "") + (result.trace_dataframe.to_json() or "") + repr(serialized)
    for forbidden in (
        _PROMPT,
        _DIGEST_MATERIAL,
        private_plan.digest,
        private_plan.phase7_contract.digest,
        *_CONTENT_DIGESTS,
    ):
        assert forbidden not in rendered
    for column in (*result.dataframe.columns, *result.trace_dataframe.columns):
        assert not any(token in column for token in ("phase7", "scope", "slot", "task", "attempt", "bundle", "digest"))
