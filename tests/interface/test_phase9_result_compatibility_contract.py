# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import json
import pickle
from collections.abc import Callable
from dataclasses import MISSING, fields
from importlib.resources import files
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Detect, Rewrite
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_REWRITTEN_TEXT, COL_TEXT
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.replace.replace_runner import ReplacementResult
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult
from anonymizer.interface import _result_compatibility as compatibility
from anonymizer.interface.results import AnonymizerResult, PreviewResult
from tests.interface.test_anonymizer_interface import _make_anonymizer

_CONTRACT_DIGEST = "c91a410289c3549f608cc0b088da3ce9db56ac10aeabe430a8254b637ef4b12d"
_P9_PICKLE_FIXTURE = Path(__file__).parent / "reference_models" / "phase9_p9_pickle_fixture.json"


def _field_names(value: Any) -> list[str]:
    return [field.name for field in fields(value)]


def test_public_result_type_locations_and_field_order_are_unchanged() -> None:
    assert AnonymizerResult.__module__ == "anonymizer.interface.results"
    assert PreviewResult.__module__ == "anonymizer.interface.results"
    assert FailedRecord.__module__ == "anonymizer.engine.ndd.adapter"
    assert _field_names(AnonymizerResult) == [
        "dataframe",
        "trace_dataframe",
        "resolved_text_column",
        "failed_records",
        "replace_method",
        "rewrite_config",
        "entity_labels",
        "data_summary",
        "_display_cycle_index",
    ]
    assert _field_names(PreviewResult) == [
        "dataframe",
        "trace_dataframe",
        "resolved_text_column",
        "failed_records",
        "preview_num_records",
        "replace_method",
        "rewrite_config",
        "entity_labels",
        "data_summary",
        "_display_cycle_index",
    ]
    assert _field_names(FailedRecord) == ["record_id", "step", "reason"]


def test_public_result_constructor_parameters_and_defaults_are_unchanged() -> None:
    anonymizer_parameters = inspect.signature(AnonymizerResult).parameters
    preview_parameters = inspect.signature(PreviewResult).parameters

    assert list(anonymizer_parameters) == [
        "dataframe",
        "trace_dataframe",
        "resolved_text_column",
        "failed_records",
        "replace_method",
        "rewrite_config",
        "entity_labels",
        "data_summary",
    ]
    assert list(preview_parameters) == [
        "dataframe",
        "trace_dataframe",
        "resolved_text_column",
        "failed_records",
        "preview_num_records",
        "replace_method",
        "rewrite_config",
        "entity_labels",
        "data_summary",
    ]
    for parameters, required in (
        (anonymizer_parameters, 4),
        (preview_parameters, 5),
    ):
        values = list(parameters.values())
        assert all(parameter.default is inspect.Parameter.empty for parameter in values[:required])
        assert all(parameter.default is None for parameter in values[required:])

    cycle_field = next(field for field in fields(AnonymizerResult) if field.name == "_display_cycle_index")
    assert cycle_field.init is False
    assert cycle_field.repr is False
    assert cycle_field.default == 0
    assert cycle_field.default_factory is MISSING


def _contract_envelope() -> dict[str, object]:
    resource = files("anonymizer.interface").joinpath("result_compatibility_contract.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def test_bundled_contract_has_the_exact_approved_digest_and_shape() -> None:
    envelope = _contract_envelope()
    encoded_contract = json.dumps(
        envelope["contract"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert set(envelope) == {"schema_version", "digest_algorithm", "digest", "contract"}
    assert envelope["schema_version"] == "anonymizer-phase9-result-compatibility-owner-contract-envelope/v1"
    assert envelope["digest"] == hashlib.sha256(encoded_contract).hexdigest() == _CONTRACT_DIGEST
    assert compatibility._is_admitted_result_compatibility_contract(
        compatibility._compile_result_compatibility_contract(envelope)
    )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda value: value.update(extra=True), id="unknown-envelope-key"),
        pytest.param(lambda value: value.update(digest="0" * 64), id="digest"),
        pytest.param(lambda value: value.update(schema_version="wrong"), id="schema"),
        pytest.param(lambda value: value["contract"].update(version="wrong"), id="version"),
        pytest.param(lambda value: value["contract"].update(sdk_phase="9"), id="json-type"),
    ],
)
def test_contract_loader_rejects_mutated_envelopes(mutate: Callable[[dict[str, object]], None]) -> None:
    envelope = copy.deepcopy(_contract_envelope())
    mutate(envelope)

    rejected = compatibility._compile_result_compatibility_contract(envelope)

    assert not compatibility._is_admitted_result_compatibility_contract(rejected)
    assert repr(rejected) == "<private resultcompatibilitycontractrejected>"
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(rejected)


def test_contract_loader_rejects_an_unavailable_resource() -> None:
    with patch.object(compatibility, "_read_contract_envelope", side_effect=OSError):
        assert compatibility._frozen_contract() is None
        rejected = compatibility._load_result_compatibility_contract()

    assert not compatibility._is_admitted_result_compatibility_contract(rejected)


@pytest.mark.parametrize("malformed", [None, [], {}, {"contract": []}])
def test_frozen_contract_snapshot_rejects_malformed_resources_without_raising(malformed: object) -> None:
    with patch.object(compatibility, "_read_contract_envelope", return_value=malformed):
        assert compatibility._frozen_contract() is None


def test_run_and_preview_factories_preserve_metadata_and_container_identity() -> None:
    labels = ["email", "first_name"]
    config = AnonymizerConfig(replace=Redact(), detect=Detect(entity_labels=labels))
    failures = [FailedRecord(record_id="opaque", step="detection", reason="unavailable")]
    source = pd.DataFrame(
        {
            "__nemo_anonymizer_text_input__": ["Alice"],
            "__nemo_anonymizer_text_output__": ["[REDACTED_FIRST_NAME]"],
            "tagged_text": ["<first_name>Alice</first_name>"],
            "final_entities": [{"entities": []}],
        }
    )

    result = compatibility._materialize_run_result(
        source,
        config=config,
        resolved_text_column="bio",
        failed_records=failures,
        data_summary="a private source summary",
    )
    preview = compatibility._materialize_preview_result(result, config=config, preview_num_records=10)

    assert result.failed_records is failures
    assert result.replace_method is config.replace
    assert result.rewrite_config is None
    assert result.entity_labels is config.detect.entity_labels
    assert result.data_summary == "a private source summary"
    assert preview.dataframe is result.dataframe
    assert preview.trace_dataframe is result.trace_dataframe
    assert preview.failed_records is failures
    assert preview.preview_num_records == 10
    assert preview.replace_method is config.replace
    assert preview.entity_labels is config.detect.entity_labels
    assert preview.data_summary is result.data_summary

    restored_result = pickle.loads(pickle.dumps(result))
    restored_preview = pickle.loads(pickle.dumps(preview))
    assert type(restored_result) is AnonymizerResult
    assert type(restored_preview) is PreviewResult
    assert restored_result.__class__.__module__ == "anonymizer.interface.results"
    assert restored_preview.__class__.__module__ == "anonymizer.interface.results"
    assert restored_result.failed_records[0].__class__.__module__ == "anonymizer.engine.ndd.adapter"
    assert list(vars(restored_result)) == list(vars(result))
    assert list(vars(restored_preview)) == list(vars(preview))


def test_p10_loads_frozen_p9_result_pickles_with_exact_public_state() -> None:
    fixture = json.loads(_P9_PICKLE_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["source_commit"] == "614e1f4104e107a673864eb7a2e12de5e49607f0"
    assert fixture["pickle_protocol"] == 4

    restored_result = pickle.loads(base64.b64decode(fixture["anonymizer_result_base64"]))
    restored_preview = pickle.loads(base64.b64decode(fixture["preview_result_base64"]))
    index = pd.Index([3, 1], dtype="Int64", name="source-row")
    expected_public = pd.DataFrame(
        {
            "bio": pd.Series(["Alice", None], index=index, dtype="string"),
            "bio_replaced": pd.Series(["[REDACTED]", None], index=index, dtype="string"),
        },
        index=index,
    )
    expected_public.attrs.update({"dataset": {"version": "p9"}})
    expected_trace = expected_public.copy()
    expected_trace["private_trace"] = pd.Series([1, None], index=index, dtype="Int64")

    assert type(restored_result) is AnonymizerResult
    assert type(restored_preview) is PreviewResult
    for restored in (restored_result, restored_preview):
        pd.testing.assert_frame_equal(restored.dataframe, expected_public, check_exact=True)
        pd.testing.assert_frame_equal(restored.trace_dataframe, expected_trace, check_exact=True)
        assert restored.dataframe.attrs == {"dataset": {"version": "p9"}}
        assert restored.trace_dataframe.attrs == {"dataset": {"version": "p9"}}
        assert restored.resolved_text_column == "bio"
        assert restored.failed_records == [FailedRecord(record_id="opaque", step="detection", reason="unavailable")]
        assert type(restored.replace_method) is Redact
        assert restored.rewrite_config is None
        assert restored.entity_labels == ["first_name"]
        assert restored.data_summary == "P9 fixture"
        assert restored._display_cycle_index == 1
    assert restored_preview.preview_num_records == 10


def test_preview_factory_preserves_an_over_request_for_an_empty_result() -> None:
    config = AnonymizerConfig(replace=Redact())
    frame = pd.DataFrame(
        {
            "__nemo_anonymizer_text_input__": pd.Series([], dtype="string"),
            "__nemo_anonymizer_text_output__": pd.Series([], dtype="string"),
        }
    )
    result = compatibility._materialize_run_result(
        frame,
        config=config,
        resolved_text_column="text",
        failed_records=[],
        data_summary=None,
    )

    preview = compatibility._materialize_preview_result(result, config=config, preview_num_records=10)

    assert preview.preview_num_records == 10
    assert preview.dataframe.empty
    assert list(preview.dataframe.columns) == ["text", "text_replaced"]
    assert [str(dtype) for dtype in preview.dataframe.dtypes] == ["string", "string"]


def test_rewrite_run_and_preview_use_the_exact_privacy_goal_reference() -> None:
    config = AnonymizerConfig(rewrite=Rewrite())
    assert config.rewrite is not None
    source = pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_REWRITTEN_TEXT: ["A person"],
        }
    )

    result = compatibility._materialize_run_result(
        source,
        config=config,
        resolved_text_column="bio",
        failed_records=[],
        data_summary=None,
    )
    preview = compatibility._materialize_preview_result(
        result,
        config=config,
        preview_num_records=10,
    )

    assert result.rewrite_config is config.rewrite.privacy_goal
    assert preview.rewrite_config is config.rewrite.privacy_goal
    assert result.replace_method is None
    assert preview.replace_method is None


def test_rich_pandas_observables_survive_run_preview_and_evaluation_factories() -> None:
    nested = {"entities": [{"value": "Alice"}]}
    index = pd.MultiIndex.from_tuples(
        [("a", 2), ("a", 1), ("a", 2)],
        names=["group", "position"],
    )
    source = pd.DataFrame(
        {
            COL_TEXT: pd.Series(["Alice", None, "Bob"], index=index, dtype="string"),
            COL_REWRITTEN_TEXT: pd.Series(["A person", None, "B person"], index=index, dtype="string"),
            "utility_score": pd.Series([0.9, None, 0.8], index=index, dtype="Float64"),
            "detection_valid": pd.Series([True, None, False], index=index, dtype="boolean"),
            "final_entities": pd.Series([nested, {"entities": []}, nested], index=index, dtype="object"),
            "ignored": pd.Series([1, None, 3], index=index, dtype="Int64"),
        },
        index=index,
    )
    source.columns = pd.Index(source.columns, dtype="string", name="pipeline-column")
    source.attrs.update({"dataset": {"kind": "factory-characterization"}})
    config = AnonymizerConfig(rewrite=Rewrite())

    result = compatibility._materialize_run_result(
        source,
        config=config,
        resolved_text_column="bio",
        failed_records=[],
        data_summary="summary",
    )
    preview = compatibility._materialize_preview_result(
        result,
        config=config,
        preview_num_records=10,
    )
    evaluated = compatibility._materialize_evaluation_result(
        source,
        resolved_text_column="bio",
        failed_records=[],
        compute_detection_validity=True,
        rewrite_config=config.rewrite.privacy_goal if config.rewrite is not None else None,
        entity_labels=config.detect.entity_labels,
        data_summary="summary",
    )
    expected_trace = source.rename(columns={COL_TEXT: "bio", COL_REWRITTEN_TEXT: "bio_rewritten"})
    expected_run = expected_trace[["bio", "bio_rewritten", "utility_score"]].copy()
    expected_evaluation = expected_trace[["bio", "bio_rewritten", "utility_score", "detection_valid"]].copy()

    pd.testing.assert_frame_equal(result.trace_dataframe, expected_trace, check_exact=True)
    pd.testing.assert_frame_equal(result.dataframe, expected_run, check_exact=True)
    assert preview.dataframe is result.dataframe
    assert preview.trace_dataframe is result.trace_dataframe
    pd.testing.assert_frame_equal(evaluated.trace_dataframe, expected_trace, check_exact=True)
    pd.testing.assert_frame_equal(evaluated.dataframe, expected_evaluation, check_exact=True)
    assert result.dataframe.attrs == source.attrs
    assert result.dataframe.attrs is not source.attrs
    assert result.dataframe.attrs["dataset"] is not source.attrs["dataset"]
    assert result.trace_dataframe.iloc[0]["final_entities"] is nested


def test_preview_factory_failure_occurs_after_completed_telemetry(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice"]}).to_csv(source, index=False)
    data = AnonymizerInput(source=str(source))
    config = AnonymizerConfig(replace=Redact())
    anonymizer_instance, _, _, _ = _make_anonymizer()

    with (
        patch.object(anonymizer_instance, "_maybe_emit_telemetry") as emit_telemetry,
        patch(
            "anonymizer.interface.anonymizer._materialize_preview_result",
            side_effect=RuntimeError("preview factory failed"),
        ),
        pytest.raises(RuntimeError, match="preview factory failed"),
    ):
        anonymizer_instance.preview(config=config, data=data, num_records=10)

    assert emit_telemetry.call_count == 1
    assert emit_telemetry.call_args.kwargs["status"].value == "completed"
    assert type(emit_telemetry.call_args.kwargs["result"]) is AnonymizerResult


def test_evaluation_factory_preserves_failure_and_rewrite_metadata_identity() -> None:
    config = AnonymizerConfig(rewrite=Rewrite())
    rewrite = config.rewrite
    assert rewrite is not None
    failures = [
        FailedRecord(record_id="opaque-a", step="rewrite-judge", reason="unavailable"),
        FailedRecord(record_id="opaque-a", step="entity-coverage", reason="unavailable"),
    ]
    judged = pd.DataFrame(
        {
            "__nemo_anonymizer_text_input__": ["Alice"],
            "_rewritten_text": ["A person"],
            "utility_score": pd.Series([0.8], dtype="Float64"),
        }
    )

    result = compatibility._materialize_evaluation_result(
        judged,
        resolved_text_column="bio",
        failed_records=failures,
        compute_detection_validity=False,
        rewrite_config=rewrite.privacy_goal,
        entity_labels=config.detect.entity_labels,
        data_summary="summary",
    )

    assert result.failed_records is failures
    assert result.replace_method is None
    assert result.rewrite_config is rewrite.privacy_goal
    assert result.entity_labels is config.detect.entity_labels
    assert str(result.dataframe["utility_score"].dtype) == "Float64"


def test_materializers_reject_private_graph_outcomes_without_exposing_them() -> None:
    from anonymizer.engine.execution.graph import _DatumId
    from anonymizer.engine.execution.protection_service import (
        _GraphProtectionFailed,
        _GraphProtectionResult,
        _GraphProtectionSucceeded,
    )

    graph_results = (
        _GraphProtectionResult((_GraphProtectionSucceeded(_DatumId("secret-id"), "protected text", True),)),
        _GraphProtectionResult((_GraphProtectionFailed(_DatumId("secret-id"), "stage", "scope"),)),
    )

    for graph_result in graph_results:
        with pytest.raises(TypeError, match="pandas DataFrame") as exc_info:
            compatibility._rename_output_columns(cast(pd.DataFrame, graph_result), resolved_text_column="bio")
        assert "secret-id" not in str(exc_info.value)
        assert "protected text" not in str(exc_info.value)


def test_rewrite_evaluation_replaces_prior_failures_and_preserves_order_and_identity() -> None:
    anonymizer, _, _, rewrite_runner = _make_anonymizer()
    rewrite = Rewrite()
    assert rewrite.privacy_goal is not None
    prior = FailedRecord(record_id="prior", step="run", reason="prior")
    rewrite_failure = FailedRecord(record_id="duplicate", step="rewrite-judge", reason="judge")
    coverage_failure = FailedRecord(record_id="duplicate", step="entity-coverage", reason="coverage")
    judged = pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_REWRITTEN_TEXT: ["A person"],
            "judge_evaluation": [None],
            "entity_coverage": [None],
        }
    )
    rewrite_runner.evaluate.return_value = RewriteResult(dataframe=judged, failed_records=[rewrite_failure])
    output = AnonymizerResult(
        dataframe=pd.DataFrame(),
        trace_dataframe=judged,
        resolved_text_column="text",
        failed_records=[prior],
        replace_method=Redact(),
        rewrite_config=rewrite.privacy_goal,
    )

    with patch("anonymizer.interface.anonymizer.EntityCoverageWorkflow") as coverage_workflow:
        coverage_workflow.return_value.run_non_critical.return_value = (judged, [coverage_failure, coverage_failure])
        evaluated = anonymizer.evaluate(output)
        reevaluated = anonymizer.evaluate(evaluated)

    assert evaluated.failed_records == [rewrite_failure, coverage_failure, coverage_failure]
    assert evaluated.failed_records is not rewrite_runner.evaluate.return_value.failed_records
    assert evaluated.failed_records[0] is rewrite_failure
    assert evaluated.failed_records[1] is coverage_failure
    assert evaluated.failed_records[2] is coverage_failure
    assert prior not in evaluated.failed_records
    assert evaluated.rewrite_config is rewrite.privacy_goal
    assert evaluated.replace_method is None
    assert type(reevaluated) is AnonymizerResult
    assert reevaluated.failed_records == [rewrite_failure, coverage_failure, coverage_failure]
    assert reevaluated.failed_records is not evaluated.failed_records
    assert reevaluated.rewrite_config is rewrite.privacy_goal


def test_replace_evaluation_reuses_current_failure_list_and_drops_prior_failures() -> None:
    anonymizer, _, replace_runner, _ = _make_anonymizer()
    prior = FailedRecord(record_id="prior", step="run", reason="prior")
    current = FailedRecord(record_id="duplicate", step="entity-coverage", reason="judge")
    failures = [current, current]
    judged = pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_REPLACED_TEXT: ["[REDACTED_FIRST_NAME]"],
            COL_FINAL_ENTITIES: [{"entities": []}],
            "entity_coverage": [None],
        }
    )
    replace_runner.evaluate.return_value = ReplacementResult(dataframe=judged, failed_records=failures)
    replace = Redact()
    output = AnonymizerResult(
        dataframe=pd.DataFrame(),
        trace_dataframe=judged,
        resolved_text_column="text",
        failed_records=[prior],
        replace_method=replace,
    )

    evaluated = anonymizer.evaluate(output)

    assert evaluated.failed_records is failures
    assert evaluated.failed_records == [current, current]
    assert prior not in evaluated.failed_records
    assert evaluated.replace_method is replace
    assert evaluated.rewrite_config is None
