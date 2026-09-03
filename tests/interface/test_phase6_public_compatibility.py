# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, ReplaceMethod, Substitute
from anonymizer.engine.constants import (
    COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES,
    COL_ATTRIBUTE_FIDELITY_VALID,
    COL_DETECTED_ENTITIES,
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_COVERAGE,
    COL_FINAL_ENTITIES,
    COL_MISSED_ENTITIES,
    COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS,
    COL_RELATIONAL_CONSISTENCY_VALID,
    COL_REPLACEMENT_APPLICATION,
    COL_REPLACEMENT_MAP,
    COL_TAGGED_TEXT,
    COL_TARGET_WORK_ID,
    COL_TEXT,
    COL_TYPE_FIDELITY_INVALID_REPLACEMENTS,
    COL_TYPE_FIDELITY_VALID,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.execution.phase6_runtime import _CandidateProposal
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow, _get_replacement_mapping_prompt
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.engine.replace.strategies import (
    ReplacementEntry,
    apply_local_replace_strategy,
    apply_replacement_map,
    apply_replacements_to_spans,
)
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from anonymizer.engine.schemas import EntitiesSchema
from anonymizer.interface import _protection as protection_module
from anonymizer.interface._protection import _Failed
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.cli.main import app
from tests.interface.test_private_protection import _AnchoredPhase6Backend, _record
from tests.streaming.structured_trace_prototype import build_synthetic_anonymizer

_SYNTHETIC_VALUES = {
    "Alice": "Avery",
    "Bob": "Blake",
    "Carol": "Casey",
}


def _entity_payload(value: str) -> dict[str, list[dict[str, object]]]:
    return {
        "entities": [
            {
                "id": f"entity-{value}",
                "value": value,
                "label": "first_name",
                "start_position": 0,
                "end_position": len(value),
                "score": 1.0,
                "source": "synthetic-test",
            }
        ]
    }


def _detect(dataframe: pd.DataFrame, **_kwargs: object) -> EntityDetectionResult:
    detected = dataframe.copy()
    values = detected[COL_TEXT].astype(str).tolist()
    payloads = [_entity_payload(value) for value in values]
    detected[COL_DETECTED_ENTITIES] = payloads
    detected[COL_FINAL_ENTITIES] = payloads
    detected[COL_ENTITIES_BY_VALUE] = [
        {"entities_by_value": [{"value": value, "labels": ["first_name"]}]} for value in values
    ]
    detected[COL_TAGGED_TEXT] = [f"<first_name>{value}</first_name>" for value in values]
    return EntityDetectionResult(dataframe=detected, failed_records=[])


def _detect_no_entities(dataframe: pd.DataFrame, **_kwargs: object) -> EntityDetectionResult:
    detected = dataframe.copy()
    detected[COL_DETECTED_ENTITIES] = [{"entities": []} for _ in range(len(detected))]
    detected[COL_FINAL_ENTITIES] = [{"entities": []} for _ in range(len(detected))]
    detected[COL_ENTITIES_BY_VALUE] = [{"entities_by_value": []} for _ in range(len(detected))]
    detected[COL_TAGGED_TEXT] = detected[COL_TEXT].astype(str)
    return EntityDetectionResult(dataframe=detected, failed_records=[])


def _detect_malformed_anchor(dataframe: pd.DataFrame, **_kwargs: object) -> EntityDetectionResult:
    detected = dataframe.copy()
    malformed = {
        "entities": [
            {
                "id": "malformed-alice",
                "value": "Alice",
                "label": "first_name",
                "start_position": 0,
                "end_position": 4,
                "score": 1.0,
                "source": "synthetic-test",
            }
        ]
    }
    detected[COL_DETECTED_ENTITIES] = [malformed for _ in range(len(detected))]
    detected[COL_FINAL_ENTITIES] = [malformed for _ in range(len(detected))]
    detected[COL_ENTITIES_BY_VALUE] = [
        {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]} for _ in range(len(detected))
    ]
    detected[COL_TAGGED_TEXT] = detected[COL_TEXT].astype(str)
    return EntityDetectionResult(dataframe=detected, failed_records=[])


def _replacement_map(value: str) -> dict[str, list[dict[str, str]]]:
    return {
        "replacements": [
            {
                "original": value,
                "label": "first_name",
                "synthetic": _SYNTHETIC_VALUES[value],
            }
        ]
    }


def _replace(
    dataframe: pd.DataFrame,
    *,
    replace_method: ReplaceMethod,
    **_kwargs: object,
) -> ReplacementResult:
    if isinstance(replace_method, Substitute):
        mapped = dataframe.copy()
        mapped[COL_REPLACEMENT_MAP] = [_replacement_map(str(value)) for value in mapped[COL_TEXT]]
        replaced = apply_replacement_map(mapped)
    else:
        replaced = apply_local_replace_strategy(dataframe, strategy=replace_method)
    return ReplacementResult(dataframe=replaced, failed_records=[])


def _evaluate(
    dataframe: pd.DataFrame,
    *,
    replace_method: ReplaceMethod,
    **_kwargs: object,
) -> ReplacementResult:
    evaluated = dataframe.copy()
    evaluated[COL_ENTITY_COVERAGE] = [1.0] * len(evaluated)
    evaluated[COL_MISSED_ENTITIES] = [[] for _ in range(len(evaluated))]
    if isinstance(replace_method, Substitute):
        evaluated[COL_TYPE_FIDELITY_VALID] = [True] * len(evaluated)
        evaluated[COL_TYPE_FIDELITY_INVALID_REPLACEMENTS] = [[] for _ in range(len(evaluated))]
        evaluated[COL_RELATIONAL_CONSISTENCY_VALID] = [True] * len(evaluated)
        evaluated[COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS] = [[] for _ in range(len(evaluated))]
        evaluated[COL_ATTRIBUTE_FIDELITY_VALID] = [True] * len(evaluated)
        evaluated[COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES] = [[] for _ in range(len(evaluated))]
    return ReplacementResult(dataframe=evaluated, failed_records=[])


def _synthetic_anonymizer(
    *,
    detection_failures: tuple[FailedRecord, ...] = (),
    replacement_failures: tuple[FailedRecord, ...] = (),
    detect: Callable[..., EntityDetectionResult] = _detect,
) -> Anonymizer:
    detection = Mock(spec=EntityDetectionWorkflow)
    detection.run.side_effect = lambda dataframe, **kwargs: EntityDetectionResult(
        dataframe=detect(dataframe, **kwargs).dataframe,
        failed_records=list(detection_failures),
    )
    replacement = Mock(spec=ReplacementWorkflow)
    replacement.run.side_effect = lambda dataframe, **kwargs: ReplacementResult(
        dataframe=_replace(dataframe, **kwargs).dataframe,
        failed_records=list(replacement_failures),
    )
    replacement.evaluate.side_effect = _evaluate
    return Anonymizer(
        data_designer=Mock(),
        detection_workflow=detection,
        replace_runner=replacement,
        rewrite_runner=Mock(spec=RewriteWorkflow),
    )


def _base_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["r-alice-1", "r-bob", "r-alice-2", "r-carol"],
            "text": ["Alice", "Bob", "Alice", "Carol"],
        },
        index=pd.Index([11, 4, 11, 2]),
    )


_FRAME_VARIANTS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "duplicate_non_monotonic_index": lambda frame: frame.copy(),
    "filtered": lambda frame: frame.iloc[[3, 1]],
    "reordered": lambda frame: frame.iloc[[2, 0, 3, 1]],
    "concatenated": lambda frame: pd.concat([frame.iloc[2:], frame.iloc[:2]]),
    "reset_index": lambda frame: frame.iloc[[1, 3, 0]].reset_index(drop=True),
}


def _write_input(frame: pd.DataFrame, path: Path, file_format: str) -> AnonymizerInput:
    if file_format == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=True)
    return AnonymizerInput(source=str(path), id_column="record_id")


_EXPECTED_REPLACEMENTS = {
    "redact": {
        "Alice": "[REDACTED_FIRST_NAME]",
        "Bob": "[REDACTED_FIRST_NAME]",
        "Carol": "[REDACTED_FIRST_NAME]",
    },
    "annotate": {
        "Alice": "<Alice, first_name>",
        "Bob": "<Bob, first_name>",
        "Carol": "<Carol, first_name>",
    },
    "hash": {
        "Alice": "<HASH_FIRST_NAME_3bc51062973c>",
        "Bob": "<HASH_FIRST_NAME_cd9fb1e148cc>",
        "Carol": "<HASH_FIRST_NAME_b2dd7d8a7056>",
    },
    "substitute": {
        "Alice": "Avery",
        "Bob": "Blake",
        "Carol": "Casey",
    },
}


@pytest.mark.parametrize(
    ("strategy_name", "replace_method"),
    [
        pytest.param("redact", Redact(), id="redact"),
        pytest.param("annotate", Annotate(), id="annotate"),
        pytest.param("hash", Hash(), id="hash"),
        pytest.param("substitute", Substitute(), id="substitute"),
    ],
)
@pytest.mark.parametrize("variant_name", list(_FRAME_VARIANTS))
@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_public_run_preserves_supported_dataframe_shapes_and_strategy_outputs(
    tmp_path: Path,
    strategy_name: str,
    replace_method: ReplaceMethod,
    variant_name: str,
    file_format: str,
) -> None:
    source_frame = _FRAME_VARIANTS[variant_name](_base_dataframe())
    input_data = _write_input(source_frame, tmp_path / f"input.{file_format}", file_format)

    result = _synthetic_anonymizer().run(
        config=AnonymizerConfig(replace=replace_method, emit_telemetry=False),
        data=input_data,
    )

    expected_text = source_frame["text"].tolist()
    assert result.dataframe["text"].tolist() == expected_text
    assert result.dataframe["text_replaced"].tolist() == [
        _EXPECTED_REPLACEMENTS[strategy_name][value] for value in expected_text
    ]
    assert result.dataframe["text_with_spans"].tolist() == [
        f"<first_name>{value}</first_name>" for value in expected_text
    ]
    assert result.trace_dataframe["record_id"].tolist() == source_frame["record_id"].tolist()
    assert set(result.dataframe.columns) == {"text", "text_replaced", "text_with_spans", COL_FINAL_ENTITIES}
    assert COL_DETECTED_ENTITIES in result.trace_dataframe.columns
    assert COL_REPLACEMENT_MAP in result.trace_dataframe.columns
    assert COL_TARGET_WORK_ID not in result.trace_dataframe.columns
    assert result.failed_records == []
    expected_index = list(range(len(source_frame))) if file_format == "csv" else source_frame.index.tolist()
    assert result.dataframe.index.tolist() == expected_index
    assert result.trace_dataframe.index.tolist() == expected_index


@pytest.mark.parametrize(
    ("strategy_name", "replace_method"),
    [
        pytest.param("redact", Redact(), id="redact"),
        pytest.param("annotate", Annotate(), id="annotate"),
        pytest.param("hash", Hash(), id="hash"),
        pytest.param("substitute", Substitute(), id="substitute"),
    ],
)
def test_public_preview_and_evaluate_keep_order_columns_and_metrics(
    tmp_path: Path,
    strategy_name: str,
    replace_method: ReplaceMethod,
) -> None:
    source_frame = _FRAME_VARIANTS["reordered"](_base_dataframe())
    input_data = _write_input(source_frame, tmp_path / "input.parquet", "parquet")
    anonymizer = _synthetic_anonymizer()

    preview = anonymizer.preview(
        config=AnonymizerConfig(replace=replace_method, emit_telemetry=False),
        data=input_data,
        num_records=2,
    )
    evaluated = anonymizer.evaluate(preview)

    expected_text = source_frame["text"].iloc[:2].tolist()
    assert preview.preview_num_records == 2
    assert preview.dataframe["text"].tolist() == expected_text
    assert preview.dataframe["text_replaced"].tolist() == [
        _EXPECTED_REPLACEMENTS[strategy_name][value] for value in expected_text
    ]
    assert preview.dataframe.index.tolist() == source_frame.index[:2].tolist()
    assert evaluated.dataframe["text"].tolist() == expected_text
    assert evaluated.dataframe[COL_ENTITY_COVERAGE].tolist() == [1.0, 1.0]
    assert evaluated.dataframe[COL_MISSED_ENTITIES].tolist() == [[], []]
    assert evaluated.trace_dataframe.index.tolist() == source_frame.index[:2].tolist()
    assert evaluated.failed_records == []
    if isinstance(replace_method, Substitute):
        assert evaluated.dataframe[COL_TYPE_FIDELITY_VALID].tolist() == [True, True]
        assert evaluated.dataframe[COL_RELATIONAL_CONSISTENCY_VALID].tolist() == [True, True]
        assert evaluated.dataframe[COL_ATTRIBUTE_FIDELITY_VALID].tolist() == [True, True]
    else:
        assert COL_TYPE_FIDELITY_VALID not in evaluated.dataframe.columns
        assert COL_RELATIONAL_CONSISTENCY_VALID not in evaluated.dataframe.columns
        assert COL_ATTRIBUTE_FIDELITY_VALID not in evaluated.dataframe.columns


@pytest.mark.parametrize(
    ("replace_method", "expected"),
    [
        pytest.param(Redact(), "[REDACTED_FIRST_NAME]", id="redact"),
        pytest.param(Substitute(), "Blake", id="substitute"),
    ],
)
def test_public_display_renders_the_selected_transformed_record(
    tmp_path: Path,
    replace_method: ReplaceMethod,
    expected: str,
) -> None:
    source_frame = _FRAME_VARIANTS["filtered"](_base_dataframe())
    input_data = _write_input(source_frame, tmp_path / "input.parquet", "parquet")
    result = _synthetic_anonymizer().run(
        config=AnonymizerConfig(replace=replace_method, emit_telemetry=False),
        data=input_data,
    )

    display = Mock()
    ipython_display = Mock(HTML=lambda html: html, display=display)
    with patch.dict("sys.modules", {"IPython": Mock(), "IPython.display": ipython_display}):
        result.display_record(index=1)

    rendered = display.call_args.args[0]
    assert "Bob" in rendered
    assert expected in rendered
    assert "Carol" not in rendered
    assert result._display_cycle_index == 0


@pytest.mark.parametrize(
    "replace_method",
    [
        pytest.param(Redact(), id="redact"),
        pytest.param(Annotate(), id="annotate"),
        pytest.param(Hash(), id="hash"),
        pytest.param(Substitute(), id="substitute"),
    ],
)
def test_public_validate_config_accepts_each_supported_replacement_strategy(replace_method: ReplaceMethod) -> None:
    config = AnonymizerConfig(replace=replace_method, emit_telemetry=False)

    _synthetic_anonymizer().validate_config(config)

    assert type(config.replace).__name__ == type(replace_method).__name__


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        pytest.param("redact", ["[REDACTED_FIRST_NAME]"] * 4, id="redact"),
        pytest.param("substitute", ["Avery", "Casey", "Avery", "Blake"], id="substitute"),
    ],
)
def test_cli_run_preserves_transformed_row_order_and_literal_output(
    tmp_path: Path,
    strategy: str,
    expected: list[str],
) -> None:
    source_frame = _FRAME_VARIANTS["concatenated"](_base_dataframe())
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    source_frame.to_csv(source, index=False)

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=_synthetic_anonymizer()):
        with pytest.raises(SystemExit) as exc_info:
            app(
                [
                    "run",
                    "--source",
                    str(source),
                    "--replace",
                    strategy,
                    "--no-emit-telemetry",
                    "--output",
                    str(output),
                ]
            )

    assert exc_info.value.code == 0
    written = pd.read_csv(output)
    assert written["text"].tolist() == ["Alice", "Carol", "Alice", "Bob"]
    assert written["text_replaced"].tolist() == expected
    assert list(written.columns) == ["text", COL_FINAL_ENTITIES, "text_with_spans", "text_replaced"]


def test_public_substitute_bypasses_replacement_provider_for_no_entity_rows(tmp_path: Path) -> None:
    input_data = _write_input(_base_dataframe(), tmp_path / "input.parquet", "parquet")
    adapter = Mock(spec=NddAdapter)
    detection = Mock(spec=EntityDetectionWorkflow)
    detection.run.side_effect = _detect_no_entities
    replacement = ReplacementWorkflow(llm_workflow=LlmReplaceWorkflow(adapter=adapter))
    anonymizer = Anonymizer(
        data_designer=Mock(),
        detection_workflow=detection,
        replace_runner=replacement,
        rewrite_runner=Mock(spec=RewriteWorkflow),
    )

    result = anonymizer.run(
        config=AnonymizerConfig(replace=Substitute(), emit_telemetry=False),
        data=input_data,
    )

    assert result.dataframe["text_replaced"].tolist() == ["Alice", "Bob", "Alice", "Carol"]
    assert result.failed_records == []
    adapter.run_workflow.assert_not_called()


def test_public_substitute_keeps_custom_instructions_in_the_legacy_prompt() -> None:
    instruction = "P8-PROMPT-7f3bd124a9c84d7bb9f05a0b6fb420a1"

    prompt = _get_replacement_mapping_prompt(
        entities_column="_legacy_entities",
        instructions=instruction,
    )

    assert f"Additional instructions: {instruction}" in prompt


def test_public_substitute_keeps_legacy_value_only_fallback_and_non_cascading_application() -> None:
    entities = EntitiesSchema.from_raw(
        {
            "entities": [
                {
                    "id": "entity-alice",
                    "value": "Alice",
                    "label": "first_name",
                    "start_position": 0,
                    "end_position": 5,
                    "score": 1.0,
                    "source": "compatibility-test",
                }
            ]
        }
    )

    output, application = apply_replacements_to_spans(
        "Alice Avery",
        entities,
        [ReplacementEntry("Alice", "legacy_person", "Avery")],
        allow_value_fallback=True,
    )

    assert output == "Avery Avery"
    assert application.to_metrics() == {
        "targeted_span_count": 1,
        "applied_span_count": 1,
        "skipped_span_count": 0,
        "skipped_span_label_counts": {},
    }


def test_malformed_anchor_is_accepted_by_public_legacy_and_fails_closed_in_private_phase6(tmp_path: Path) -> None:
    source = pd.DataFrame({"record_id": ["r-alice"], "text": ["Alice"]})
    input_data = _write_input(source, tmp_path / "input.csv", "csv")

    public = _synthetic_anonymizer(detect=_detect_malformed_anchor).run(
        config=AnonymizerConfig(replace=Substitute(), emit_telemetry=False),
        data=input_data,
    )

    private_anonymizer = build_synthetic_anonymizer({"Alice": "first_name"})
    plan = private_anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))
    backend = _AnchoredPhase6Backend({"Alice": (_CandidateProposal(0, 4, "Alice", "first_name"),)})
    private = protection_module._ProtectionFlow(private_anonymizer, plan, phase6_backend=backend).protect(
        (_record("r-alice", "Alice"),)
    )

    assert public.dataframe["text_replaced"].tolist() == ["Alice"]
    assert public.trace_dataframe[COL_REPLACEMENT_APPLICATION].tolist() == [
        {
            "targeted_span_count": 1,
            "applied_span_count": 0,
            "skipped_span_count": 1,
            "skipped_span_label_counts": {"first_name": 1},
        }
    ]
    assert public.failed_records == []
    assert isinstance(private.outcomes[0], _Failed)
    assert not hasattr(private.outcomes[0], "output")


def test_public_run_keeps_failed_record_shape_and_stage_order(tmp_path: Path) -> None:
    detection_failure = FailedRecord(record_id="r-bob", step="entity-detection", reason="detector-timeout")
    replacement_failure = FailedRecord(record_id="r-carol", step="replace-map-generation", reason="invalid-map")
    source_frame = _FRAME_VARIANTS["concatenated"](_base_dataframe())
    input_data = _write_input(source_frame, tmp_path / "input.parquet", "parquet")

    result = _synthetic_anonymizer(
        detection_failures=(detection_failure,),
        replacement_failures=(replacement_failure,),
    ).run(
        config=AnonymizerConfig(replace=Redact(), emit_telemetry=False),
        data=input_data,
    )

    assert result.failed_records == [detection_failure, replacement_failure]
    assert result.failed_records[0] is detection_failure
    assert result.failed_records[1] is replacement_failure
    assert [vars(failure) for failure in result.failed_records] == [
        {"record_id": "r-bob", "step": "entity-detection", "reason": "detector-timeout"},
        {"record_id": "r-carol", "step": "replace-map-generation", "reason": "invalid-map"},
    ]
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
    assert result.trace_dataframe.index.tolist() == [11, 2, 11, 4]
    assert [(failure.record_id, failure.step, failure.reason) for failure in result.failed_records] == [
        ("r-bob", "entity-detection", "detector-timeout"),
        ("r-carol", "replace-map-generation", "invalid-map"),
    ]
