# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import pickle
from dataclasses import fields
from itertools import combinations, product
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_ENTITY_COVERAGE, COL_REPLACED_TEXT, COL_REWRITTEN_TEXT, COL_TEXT
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.replace.replace_runner import ReplacementResult
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult
from anonymizer.interface import _result_compatibility as compatibility
from anonymizer.interface._result_compatibility import (
    _build_user_dataframe,
    _rename_output_columns,
    _unrename_output_columns,
)
from anonymizer.interface.cli.main import app
from anonymizer.interface.results import AnonymizerResult, PreviewResult
from anonymizer.telemetry import TaskEnum, TaskStatusEnum
from tests.interface.test_anonymizer_interface import _make_anonymizer

_REFERENCE_DIRECTORY = Path(__file__).parent / "reference_models"
_REFERENCE_PATH = _REFERENCE_DIRECTORY / "phase9_result_compatibility_v1.py"
_MANIFEST_PATH = _REFERENCE_DIRECTORY / "phase9_result_compatibility_v1_manifest.json"
_MANIFEST_DIGEST = "478b7ef5d052146ac8642faec3fc22168ffc33b3dc388929fad98d84da99b6ae"
_PAIRWISE_DIGEST = "591fd8117cb0dbd6335762ee2cd0824a164d9604c0d7021d1bc995983ea7265c"


def _load_reference() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase9_result_compatibility_reference_v1", _REFERENCE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib defensive guard
        raise RuntimeError("could not load Phase 9 reference model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_manifest() -> dict[str, object]:
    return cast(dict[str, object], json.loads(_MANIFEST_PATH.read_text(encoding="utf-8")))


def _case_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [[f"value-{position}" for position in range(len(columns))]],
        columns=pd.Index(columns, dtype="object"),
    )


def test_reference_manifest_is_frozen_and_self_consistent() -> None:
    reference = _load_reference()
    manifest = _load_manifest()
    directed_cases = cast(list[dict[str, object]], manifest["cases"])
    pairwise_cases = cast(list[dict[str, object]], reference.generate_pairwise_core())
    encoded_pairwise = json.dumps(
        pairwise_cases,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    encoded_corpus = json.dumps(
        {"directed_cases": directed_cases, "pairwise_cases": pairwise_cases},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    assert manifest["schema_version"] == "anonymizer-phase9-result-compatibility-reference-manifest/v1"
    assert manifest["reference_version"] == reference.REFERENCE_VERSION
    assert manifest["generator_version"] == reference.GENERATOR_VERSION
    assert manifest["directed_case_count"] == len(directed_cases) == 22
    assert manifest["pairwise_case_count"] == len(pairwise_cases) == 11
    assert manifest["case_count"] == len(directed_cases) + len(pairwise_cases) == 33
    assert manifest["pairwise_dimensions"] == {
        name: list(values) for name, values in reference.PAIRWISE_DIMENSIONS.items()
    }
    assert manifest["pairwise_digest"] == hashlib.sha256(encoded_pairwise).hexdigest() == _PAIRWISE_DIGEST
    assert manifest["digest"] == hashlib.sha256(encoded_corpus).hexdigest() == _MANIFEST_DIGEST
    assert all(reference.reduce_reference(case) == case["expected"] for case in directed_cases)

    dimension_names = tuple(reference.PAIRWISE_DIMENSIONS)
    for left, right in combinations(dimension_names, 2):
        for left_value, right_value in product(
            reference.PAIRWISE_DIMENSIONS[left],
            reference.PAIRWISE_DIMENSIONS[right],
        ):
            assert any(case[left] == left_value and case[right] == right_value for case in pairwise_cases), (
                left,
                left_value,
                right,
                right_value,
            )


def test_reference_model_has_no_production_or_dataframe_dependency() -> None:
    tree = ast.parse(_REFERENCE_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])

    assert imported_roots == {"__future__", "collections", "itertools", "typing"}


def test_directed_projection_cases_match_the_p9_helpers() -> None:
    reference = _load_reference()
    cases = cast(list[dict[str, object]], _load_manifest()["cases"])

    for case in cases:
        if case["operation"] != "project":
            continue
        columns = cast(list[str], case["columns"])
        source = _case_frame(columns)
        expected = reference.reduce_reference(case)
        renamed = _rename_output_columns(source, resolved_text_column=cast(str, case["resolved_text_column"]))
        public = _build_user_dataframe(
            renamed,
            resolved_text_column=cast(str, case["resolved_text_column"]),
            compute_detection_validity=cast(bool, case["compute_detection_validity"]),
        )

        assert list(renamed.columns) == expected["trace_columns"], case["name"]
        assert list(public.columns) == expected["public_columns"], case["name"]
        assert (renamed is source) is expected["forward_returns_same_object"], case["name"]
        assert public is not renamed, case["name"]


def test_directed_reverse_rename_cases_match_the_p9_helpers() -> None:
    reference = _load_reference()
    cases = cast(list[dict[str, object]], _load_manifest()["cases"])

    for case in cases:
        if case["operation"] != "unrename":
            continue
        source = _case_frame(cast(list[str], case["columns"]))
        expected = reference.reduce_reference(case)
        result = _unrename_output_columns(source, resolved_text_column=cast(str, case["resolved_text_column"]))

        assert list(result.columns) == expected["columns"], case["name"]
        assert (result is source) is expected["returns_same_object"], case["name"]


def _pairwise_index(shape: str) -> pd.Index:
    if shape == "range-unnamed":
        return pd.RangeIndex(3)
    if shape == "string-duplicate-named":
        return pd.Index(["b", "a", "b"], dtype="string", name="source-row")
    if shape == "multi-duplicate-named":
        return pd.MultiIndex.from_tuples(
            [("a", 2), ("a", 1), ("a", 2)],
            names=["group", "position"],
        )
    raise AssertionError(f"unknown index shape: {shape}")


def _pairwise_values(dtype: str) -> list[object]:
    if dtype == "string":
        return ["protected-a", None, "protected-c"]
    if dtype == "Float64":
        return [1.5, None, 3.5]
    if dtype == "boolean":
        return [True, None, False]
    raise AssertionError(f"unknown value dtype: {dtype}")


def test_generated_pairwise_pandas_core_matches_production_materialization() -> None:
    reference = _load_reference()

    for case in cast(list[dict[str, object]], reference.generate_pairwise_core()):
        expected = cast(dict[str, object], reference.reduce_reference(case))
        index = _pairwise_index(cast(str, case["index_shape"]))
        dtype = cast(str, case["value_dtype"])
        nested = {"entities": [{"value": "Alice"}]}
        text = pd.Series(["Alice", None, "Bob"], index=index, dtype="string")
        values = pd.Series(_pairwise_values(dtype), index=index, dtype=dtype)
        mode = cast(str, case["mode"])
        if mode == "replace":
            source = pd.DataFrame(
                {
                    COL_TEXT: text,
                    COL_REPLACED_TEXT: values,
                    "final_entities": pd.Series(
                        [nested, {"entities": []}, nested],
                        index=index,
                        dtype="object",
                    ),
                },
                index=index,
            )
            config = AnonymizerConfig(replace=Redact())
        else:
            source = pd.DataFrame(
                {
                    COL_TEXT: text,
                    COL_REWRITTEN_TEXT: values,
                    "utility_score": pd.Series([0.9, None, 0.8], index=index, dtype="Float64"),
                    "missed_entities": pd.Series(
                        [nested, {"entities": []}, nested],
                        index=index,
                        dtype="object",
                    ),
                },
                index=index,
            )
            config = AnonymizerConfig(rewrite=Rewrite())
        source.columns = pd.Index(source.columns, dtype="string", name="pipeline-column")
        if case["attrs_shape"] == "nested":
            source.attrs.update({"dataset": {"kind": "pairwise"}})
        resolved = cast(str, expected["resolved_text_column"])
        result = compatibility._materialize_run_result(
            source,
            config=config,
            resolved_text_column=resolved,
            failed_records=[],
            data_summary=None,
        )
        expected_trace = source.copy()
        expected_trace.columns = pd.Index(
            cast(list[str], expected["trace_columns"]),
            dtype=cast(str, expected["column_index_dtype"]),
            name=cast(str, expected["column_index_name"]),
        )
        expected_public = expected_trace[cast(list[str], expected["public_columns"])].copy()

        assert_frame_equal(result.trace_dataframe, expected_trace, check_exact=True, check_flags=True)
        assert_frame_equal(result.dataframe, expected_public, check_exact=True, check_flags=True)
        assert type(result.dataframe.index) is type(index), case["name"]
        assert result.dataframe.index.names == index.names, case["name"]
        assert list(result.dataframe.index) == list(index), case["name"]
        assert str(result.trace_dataframe[cast(str, expected["value_column"])].dtype) == dtype
        assert result.dataframe.attrs == cast(dict[str, object], expected["attrs"])
        assert result.dataframe.attrs is not result.trace_dataframe.attrs
        assert result.trace_dataframe.attrs is not source.attrs
        if case["attrs_shape"] == "nested":
            assert result.dataframe.attrs["dataset"] is not result.trace_dataframe.attrs["dataset"]
            assert result.trace_dataframe.attrs["dataset"] is not source.attrs["dataset"]
        assert result.dataframe.iloc[0][cast(str, expected["nested_column"])] is nested
        if case["metadata_token"] == "entity-labels":
            assert result.entity_labels is config.detect.entity_labels
        elif mode == "replace":
            assert result.replace_method is config.replace
        else:
            assert config.rewrite is not None
            assert result.rewrite_config is config.rewrite.privacy_goal


def _failure_records(names: list[str], *, step: str) -> list[FailedRecord]:
    return [FailedRecord(record_id=name, step=step, reason="unavailable") for name in names]


def _observe_preview(case: dict[str, object]) -> dict[str, object]:
    config = AnonymizerConfig(rewrite=Rewrite())
    failures = _failure_records(["opaque"], step="rewrite")
    public = pd.DataFrame({"text": ["a", "b"]})
    trace = pd.DataFrame({COL_TEXT: ["a", "b"]})
    run = AnonymizerResult(
        dataframe=public,
        trace_dataframe=trace,
        resolved_text_column="text",
        failed_records=failures,
        rewrite_config=config.rewrite.privacy_goal if config.rewrite is not None else None,
        data_summary="summary",
    )
    result = compatibility._materialize_preview_result(
        run,
        config=config,
        preview_num_records=cast(int, case["requested"]),
    )
    return {
        "result_type": type(result).__name__,
        "preview_num_records": result.preview_num_records,
        "dataframe_binding": "run.dataframe" if result.dataframe is run.dataframe else "copy",
        "trace_binding": "run.trace_dataframe" if result.trace_dataframe is run.trace_dataframe else "copy",
        "failures_binding": "run.failed_records" if result.failed_records is run.failed_records else "copy",
        "strategy_binding": (
            "config" if config.rewrite is not None and result.rewrite_config is config.rewrite.privacy_goal else "copy"
        ),
        "data_summary_binding": "run.data_summary" if result.data_summary is run.data_summary else "copy",
    }


def _observe_evaluation_failures(case: dict[str, object]) -> dict[str, object]:
    anonymizer, _, replace_runner, rewrite_runner = _make_anonymizer()
    primary = _failure_records(cast(list[str], case["primary_failures"]), step="primary")
    coverage = _failure_records(cast(list[str], case["coverage_failures"]), step="coverage")
    prior = _failure_records(cast(list[str], case["prior_failures"]), step="prior")
    is_rewrite = cast(bool, case["rewrite"])
    if is_rewrite:
        rewrite = Rewrite()
        judged = pd.DataFrame(
            {
                COL_TEXT: ["Alice"],
                COL_REWRITTEN_TEXT: ["A person"],
                COL_ENTITY_COVERAGE: [None],
            }
        )
        rewrite_runner.evaluate.return_value = RewriteResult(dataframe=judged, failed_records=primary)
        output = AnonymizerResult(
            dataframe=pd.DataFrame(),
            trace_dataframe=judged,
            resolved_text_column="text",
            failed_records=prior,
            rewrite_config=rewrite.privacy_goal,
        )
        with patch("anonymizer.interface.anonymizer.EntityCoverageWorkflow") as coverage_workflow:
            coverage_workflow.return_value.run_non_critical.return_value = (judged, coverage)
            result = anonymizer.evaluate(output)
        primary_binding = result.failed_records is primary
    else:
        judged = pd.DataFrame(
            {
                COL_TEXT: ["Alice"],
                COL_REPLACED_TEXT: ["[REDACTED_FIRST_NAME]"],
                COL_ENTITY_COVERAGE: [None],
            }
        )
        replace_runner.evaluate.return_value = ReplacementResult(dataframe=judged, failed_records=primary)
        output = AnonymizerResult(
            dataframe=pd.DataFrame(),
            trace_dataframe=judged,
            resolved_text_column="text",
            failed_records=prior,
            replace_method=Redact(),
        )
        result = anonymizer.evaluate(output)
        primary_binding = result.failed_records is primary
    return {
        "result_type": type(result).__name__,
        "failures": [record.record_id for record in result.failed_records],
        "failure_container": "primary" if primary_binding else "new",
        "prior_failures_retained": any(record in result.failed_records for record in prior),
    }


def _observe_pickle(case: dict[str, object]) -> dict[str, object]:
    result_type = cast(str, case["result_type"])
    dataframe = pd.DataFrame({"text": ["Alice"]})
    if result_type == "AnonymizerResult":
        value: object = AnonymizerResult(dataframe, dataframe.copy(), "text", [])
    elif result_type == "PreviewResult":
        value = PreviewResult(dataframe, dataframe.copy(), "text", [], 10)
    else:
        value = FailedRecord(record_id="opaque", step="test", reason="unavailable")
    restored = pickle.loads(pickle.dumps(value))
    return {
        "module": type(restored).__module__,
        "fields": [field.name for field in fields(restored)],
    }


def _observe_cli(
    case: dict[str, object],
    *,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> dict[str, object]:
    source = tmp_path / "reference-cli-input.csv"
    pd.DataFrame({"text": ["Alice"]}).to_csv(source, index=False)
    public = pd.DataFrame({"text": ["Alice"], "text_replaced": ["[REDACTED]"]})
    trace = pd.DataFrame({"private-marker": ["must-not-serialize"]})
    failure = FailedRecord(record_id="private-record-id", step="test", reason="private-reason")
    command = cast(str, case["command"])
    mock_anonymizer = MagicMock()
    if command == "run":
        output = tmp_path / "reference-cli-output.csv"
        mock_anonymizer.run.return_value = AnonymizerResult(public, trace, "text", [failure])
        with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
            with pytest.raises(SystemExit) as exc_info:
                app(
                    [
                        "run",
                        "--source",
                        str(source),
                        "--replace",
                        "redact",
                        "--output",
                        str(output),
                    ]
                )
        printed = capsys.readouterr()
        serialized = pd.read_csv(output)
        return {
            "serialized_frame": "result.dataframe" if serialized.equals(public) else "other",
            "index": any(str(column).startswith("Unnamed:") for column in serialized.columns),
            "degraded_success_exit": exc_info.value.code,
            "failure_details_printed": "private-record-id" in printed.out or "private-reason" in printed.out,
        }
    preview = PreviewResult(public, trace, "text", [failure], 10)
    mock_anonymizer.preview.return_value = preview
    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit):
            app(["preview", "--source", str(source), "--replace", "redact"])
    printed = capsys.readouterr()
    return {
        "render": (
            "result.dataframe.to_string(max_colwidth=80)"
            if printed.out.rstrip("\n") == public.to_string(max_colwidth=80)
            else "other"
        ),
        "failure_details_printed": "private-record-id" in printed.out or "private-reason" in printed.out,
    }


def _observe_telemetry(case: dict[str, object], *, tmp_path: Path) -> dict[str, object]:
    anonymizer, *_ = _make_anonymizer()
    failures = _failure_records(cast(list[str], case["failures"]), step="unknown")
    input_count = cast(int, case["input_count"])
    result = AnonymizerResult(pd.DataFrame(), pd.DataFrame(), "text", failures)
    source = tmp_path / "reference-telemetry-input.csv"
    pd.DataFrame({"text": ["text"] * input_count}).to_csv(source, index=False)
    event = anonymizer._build_telemetry_event(
        task=TaskEnum.BATCH,
        status=TaskStatusEnum.COMPLETED,
        config=AnonymizerConfig(replace=Redact()),
        data=AnonymizerInput(source=str(source)),
        input_df=pd.DataFrame({COL_TEXT: ["text"] * input_count}),
        result=result,
        duration_sec=0.0,
    )
    return {
        "status": event.task_status.value,
        "failure_count": event.num_failure_records,
        "success_count": event.num_success_records,
        "deduplicate_failures": event.num_failure_records != len(failures),
    }


def _observe_exception(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "reference-exception-input.csv"
    pd.DataFrame({"text": ["Alice"]}).to_csv(source, index=False)
    anonymizer, detection_workflow, _, _ = _make_anonymizer()
    detection_workflow.run.side_effect = RuntimeError("private provider cause")
    caught: BaseException | None = None
    try:
        anonymizer.run(
            config=AnonymizerConfig(replace=Redact()),
            data=AnonymizerInput(source=str(source)),
        )
    except BaseException as exc:  # noqa: BLE001 - the observation records the exact public failure
        caught = exc
    assert caught is not None
    return {
        "type": type(caught).__name__,
        "message": str(caught),
        "cause": caught.__cause__,
        "partial_result": hasattr(caught, "result"),
    }


def _observe_graph(case: dict[str, object]) -> dict[str, object]:
    from anonymizer.engine.execution.graph import _DatumId
    from anonymizer.engine.execution.protection_service import (
        _GraphProtectionFailed,
        _GraphProtectionResult,
        _GraphProtectionSucceeded,
    )

    if case["shape"] == "released":
        private: object = _GraphProtectionResult(
            (_GraphProtectionSucceeded(_DatumId("private-id"), "private text", True),)
        )
    else:
        private = _GraphProtectionResult((_GraphProtectionFailed(_DatumId("private-id"), "stage", "scope"),))
    try:
        _rename_output_columns(cast(pd.DataFrame, private), resolved_text_column="text")
    except TypeError:
        return {"admission": "rejected", "public_projection": None}
    return {"admission": "admitted", "public_projection": "private"}


def _observe_production_case(
    case: dict[str, object],
    *,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> dict[str, object]:
    operation = cast(str, case["operation"])
    if operation == "preview":
        return _observe_preview(case)
    if operation == "evaluate_failures":
        return _observe_evaluation_failures(case)
    if operation == "pickle":
        return _observe_pickle(case)
    if operation == "cli":
        return _observe_cli(case, tmp_path=tmp_path, capsys=capsys)
    if operation == "telemetry":
        return _observe_telemetry(case, tmp_path=tmp_path)
    if operation == "exception":
        return _observe_exception(tmp_path)
    if operation == "graph":
        return _observe_graph(case)
    raise AssertionError(f"no in-repository production observation for {operation}")


def test_directed_consumer_cases_match_production(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference = _load_reference()
    cases = cast(list[dict[str, object]], _load_manifest()["cases"])

    for case in cases:
        if case["operation"] in {"project", "unrename", "platform"}:
            continue
        expected = reference.reduce_reference(case)
        actual = _observe_production_case(case, tmp_path=tmp_path, capsys=capsys)
        assert actual == expected, case["name"]


def test_platform_case_is_explicitly_an_external_pinned_source_claim() -> None:
    reference = _load_reference()
    cases = cast(list[dict[str, object]], _load_manifest()["cases"])
    platform_case = next(case for case in cases if case["operation"] == "platform")

    assert reference.reduce_reference(platform_case)["claim_status"] == "pinned-source-only"
    assert "Platform" not in inspect.getsource(compatibility._materialize_run_result)


def test_projection_preserves_rich_pandas_observables() -> None:
    nested = {"entities": [{"value": "Alice"}]}
    index = pd.Index([11, 4, 11], dtype="Int64", name="source-row")
    source = pd.DataFrame(
        {
            "__nemo_anonymizer_text_input__": pd.Series(["Alice", None, "Bob"], index=index, dtype="string"),
            "__nemo_anonymizer_text_output__": pd.Series(["Avery", None, "Blake"], index=index, dtype="string"),
            "final_entities": pd.Series([nested, {"entities": []}, nested], index=index, dtype="object"),
            "ignored": pd.Series([1, None, 3], index=index, dtype="Int64"),
        },
        index=index,
    )
    source.columns = pd.Index(source.columns, dtype="string", name="pipeline-column")
    source.attrs.update({"dataset": {"source": "characterization"}})

    trace = _rename_output_columns(source, resolved_text_column="bio")
    public = _build_user_dataframe(trace, resolved_text_column="bio")
    expected = trace[["bio", "bio_replaced", "final_entities"]].copy()

    assert_frame_equal(public, expected, check_exact=True, check_flags=True)
    assert type(public.index) is type(expected.index)
    assert public.index.name == "source-row"
    assert type(public.columns) is type(expected.columns)
    assert public.columns.name == "pipeline-column"
    assert [str(dtype) for dtype in public.dtypes] == ["string", "string", "object"]
    assert public.attrs == trace.attrs == source.attrs
    assert public.attrs is not trace.attrs
    assert public.attrs["dataset"] is not trace.attrs["dataset"]
    assert public is not trace
    assert public.at[11, "final_entities"].iloc[0] is trace.at[11, "final_entities"].iloc[0]


@pytest.mark.parametrize(
    "index",
    [
        pytest.param(pd.RangeIndex(4, name="range-row"), id="range"),
        pytest.param(pd.Index(["b", "a", "b", "c"], dtype="string", name="string-row"), id="string-duplicate"),
        pytest.param(pd.Index([2.0, None, 1.0, 2.0], name="nullable-row"), id="null-nonmonotonic"),
        pytest.param(
            pd.MultiIndex.from_tuples(
                [("a", 2), ("a", 1), ("a", 2), ("b", 1)],
                names=["group", "position"],
            ),
            id="multi-index",
        ),
    ],
)
def test_projection_preserves_supported_index_shapes(index: pd.Index) -> None:
    source = pd.DataFrame(
        {
            "__nemo_anonymizer_text_input__": ["a", "b", "c", "d"],
            "__nemo_anonymizer_text_output__": ["A", "B", "C", "D"],
            "final_entities": [{"entities": []} for _ in range(4)],
        },
        index=index,
    )

    trace = _rename_output_columns(source, resolved_text_column="text")
    public = _build_user_dataframe(trace, resolved_text_column="text")

    assert_frame_equal(public, trace[["text", "text_replaced", "final_entities"]].copy(), check_exact=True)
    assert type(public.index) is type(index)
    assert public.index.equals(index)
    assert public.index.names == index.names


def test_projection_preserves_extension_dtypes_and_empty_schema() -> None:
    source = pd.DataFrame(
        {
            "__nemo_anonymizer_text_input__": pd.Series([], dtype="string"),
            "__nemo_anonymizer_text_output__": pd.Series(
                pd.Categorical([], categories=["redacted", "substituted"], ordered=True)
            ),
            "tagged_text": pd.Series([], dtype="object"),
            "final_entities": pd.Series([], dtype="object"),
            "entity_coverage": pd.Series([], dtype="Float64"),
            "missed_entities": pd.Series([], dtype="object"),
            "type_fidelity_valid": pd.Series([], dtype="boolean"),
            "type_fidelity_invalid_replacements": pd.Series([], dtype="object"),
            "relational_consistency_valid": pd.Series([], dtype="bool"),
            "relational_consistency_invalid_relations": pd.Series([], dtype="object"),
            "attribute_fidelity_valid": pd.Series([], dtype="boolean"),
            "attribute_fidelity_invalid_entities": pd.Series([], dtype="object"),
            "detection_valid": pd.Series([], dtype="datetime64[ns, UTC]"),
            "detection_invalid_entities": pd.Series([], dtype="object"),
            "ignored": pd.Series([], dtype="Int64"),
        }
    )

    trace = _rename_output_columns(source, resolved_text_column="text")
    public = _build_user_dataframe(
        trace,
        resolved_text_column="text",
        compute_detection_validity=True,
    )
    expected_columns = [column for column in trace.columns if column != "ignored"]
    expected = trace[expected_columns].copy()

    assert_frame_equal(public, expected, check_exact=True)
    assert public.empty
    assert list(public.columns) == expected_columns
    assert [str(dtype) for dtype in public.dtypes] == [str(dtype) for dtype in expected.dtypes]
