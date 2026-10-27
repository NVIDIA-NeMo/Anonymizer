# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_DISPOSITION_COVERAGE,
    COL_DISPOSITION_EXPLICIT_ENTITIES,
    COL_DISPOSITION_LATENT_ENTITIES,
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_CLASSIFICATION,
    COL_LATENT_ENTITIES,
    COL_PRIVACY_QA,
    COL_RAW_SENSITIVITY_DISPOSITION,
    COL_REWRITE_DISPOSITION_BLOCK,
    COL_SENSITIVITY_DISPOSITION,
    COL_TAGGED_TEXT,
    _jinja,
)
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.rewrite.parsers import normalize_payload
from anonymizer.engine.rewrite.qa_generation import _generate_privacy_qa_column
from anonymizer.engine.rewrite.rewrite_generation import _format_rewrite_disposition_block
from anonymizer.engine.rewrite.sensitivity_disposition import (
    SensitivityDispositionWorkflow,
    _get_sensitivity_disposition_prompt,
    _number_disposition_entities,
    _parse_numbered_lines,
    _verify_disposition_coverage,
    build_numbered_entities,
    verify_disposition_coverage,
)
from anonymizer.engine.schemas import SensitivityDispositionSchema
from anonymizer.engine.schemas.rewrite import RawSensitivityDispositionSchema, StrictRawSensitivityDispositionSchema

_STUB_PRIVACY_GOAL = PrivacyGoal(
    protect="Protect direct identifiers and quasi-identifier combinations from re-identification.",
    preserve="General utility and semantic meaning of the original text.",
)

_STUB_LATENT = {
    "latent_entities": [
        {
            "category": "latent_identifier",
            "label": "employer",
            "value": "a large cloud provider",
            "confidence": "high",
            "evidence": ["worked on the S3 team"],
            "rationale": "The team name strongly implies a specific cloud employer.",
        }
    ]
}


def test_columns_numbers_entities_then_runs_llm_with_disposition_analyzer_alias(
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    cols = SensitivityDispositionWorkflow().columns(
        selected_models=stub_rewrite_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
    )
    assert [c.name for c in cols] == [
        COL_DISPOSITION_EXPLICIT_ENTITIES,
        COL_RAW_SENSITIVITY_DISPOSITION,
        COL_SENSITIVITY_DISPOSITION,
    ]
    assert isinstance(cols[0], CustomColumnConfig)
    assert isinstance(cols[1], LLMStructuredColumnConfig)
    assert cols[1].model_alias == stub_rewrite_model_selection.disposition_analyzer
    assert isinstance(cols[2], CustomColumnConfig)


def test_columns_output_schema_follows_strict_flag(stub_rewrite_model_selection: RewriteModelSelection) -> None:
    default_llm = SensitivityDispositionWorkflow().columns(
        selected_models=stub_rewrite_model_selection, privacy_goal=_STUB_PRIVACY_GOAL
    )[1]
    strict_llm = SensitivityDispositionWorkflow().columns(
        selected_models=stub_rewrite_model_selection, privacy_goal=_STUB_PRIVACY_GOAL, strict_entity_protection=True
    )[1]
    assert isinstance(default_llm, LLMStructuredColumnConfig)
    assert isinstance(strict_llm, LLMStructuredColumnConfig)
    # NDD stores output_format as a JSON schema, so compare against each model's schema.
    assert default_llm.output_format == RawSensitivityDispositionSchema.model_json_schema()
    assert strict_llm.output_format == StrictRawSensitivityDispositionSchema.model_json_schema()
    assert default_llm.output_format != strict_llm.output_format


def test_privacy_goal_interpolated_into_prompt() -> None:
    prompt = _get_sensitivity_disposition_prompt(_STUB_PRIVACY_GOAL)
    assert "Protect direct identifiers and quasi-identifier combinations" in prompt
    assert "General utility and semantic meaning" in prompt


def test_prompt_references_required_columns() -> None:
    prompt = _get_sensitivity_disposition_prompt(_STUB_PRIVACY_GOAL)
    assert _jinja(COL_TAGGED_TEXT) in prompt
    assert _jinja(COL_DISPOSITION_EXPLICIT_ENTITIES) in prompt
    assert _jinja(COL_DISPOSITION_LATENT_ENTITIES) in prompt
    assert _jinja(COL_ENTITY_CLASSIFICATION) in prompt
    # The raw (un-numbered) entity columns must not be fed to the prompt directly.
    assert _jinja(COL_ENTITIES_BY_VALUE) not in prompt
    assert _jinja(COL_LATENT_ENTITIES) not in prompt


def test_prompt_has_no_domain_context_or_tag_notation() -> None:
    prompt = _get_sensitivity_disposition_prompt(_STUB_PRIVACY_GOAL)
    assert "<domain_context>" not in prompt
    assert "Dataset description:" not in prompt
    assert "_domain" not in prompt
    assert "_tag_notation" not in prompt


def test_prompt_has_no_unfilled_placeholders() -> None:
    for strict in (False, True):
        prompt = _get_sensitivity_disposition_prompt(_STUB_PRIVACY_GOAL, strict_entity_protection=strict)
        assert "<<" not in prompt


def test_strict_block_only_present_in_strict_mode() -> None:
    assert "<strict_entity_protection>" not in _get_sensitivity_disposition_prompt(_STUB_PRIVACY_GOAL)
    strict = _get_sensitivity_disposition_prompt(_STUB_PRIVACY_GOAL, strict_entity_protection=True)
    assert "<strict_entity_protection>" in strict
    assert "Override the low eligibility rules: every supplied entity must be protected." in strict


def test_prompt_does_not_mention_combined_risk_level() -> None:
    assert "protection_reason" not in _get_sensitivity_disposition_prompt(_STUB_PRIVACY_GOAL)
    assert "combined_risk" not in _get_sensitivity_disposition_prompt(_STUB_PRIVACY_GOAL)


def test_build_numbered_entities_shares_id_space_explicit_then_latent() -> None:
    entities = {
        "entities_by_value": [
            {"value": "Alice", "labels": ["first_name"]},
            {"value": "Portland", "labels": ["city"]},
        ]
    }
    explicit, latent = build_numbered_entities(entities, _STUB_LATENT)
    explicit_rows = [json.loads(line) for line in explicit.splitlines()]
    latent_rows = [json.loads(line) for line in latent.splitlines()]
    assert [(r["id"], r["label"], r["value"]) for r in explicit_rows] == [
        (1, "first_name", "Alice"),
        (2, "city", "Portland"),
    ]
    assert [(r["id"], r["label"]) for r in latent_rows] == [(3, "employer")]
    assert latent_rows[0]["evidence"] == ["worked on the S3 team"]


def test_build_numbered_entities_one_id_per_value_label_pair() -> None:
    entities = {"entities_by_value": [{"value": "Paris", "labels": ["city", "first_name"]}]}
    explicit, _ = build_numbered_entities(entities, None)
    assert [(json.loads(line)["id"], json.loads(line)["label"]) for line in explicit.splitlines()] == [
        (1, "city"),
        (2, "first_name"),
    ]


def test_build_numbered_entities_preserves_non_ascii_values() -> None:
    explicit, _ = build_numbered_entities({"entities_by_value": [{"value": "José", "labels": ["first_name"]}]}, None)
    assert "José" in explicit


@pytest.mark.parametrize(
    "separator",
    ["\u2028", "\u2029", "\u0085", "\n"],
    ids=["line-separator", "paragraph-separator", "next-line", "newline"],
)
def test_numbered_entities_round_trip_embedded_separators(separator: str) -> None:
    value = f"José{separator}García"
    evidence = f"worked on{separator}the S3 team"
    latent_entity = {**_STUB_LATENT["latent_entities"][0], "evidence": [evidence]}
    explicit, latent = build_numbered_entities(
        {"entities_by_value": [{"value": value, "labels": ["first_name"]}, {"value": "Portland", "labels": ["city"]}]},
        {"latent_entities": [latent_entity]},
    )

    explicit_rows = _parse_numbered_lines(explicit)
    latent_rows = _parse_numbered_lines(latent)
    assert [entry["id"] for entry in explicit_rows + latent_rows] == [1, 2, 3]
    assert explicit_rows[0]["value"] == value
    assert explicit_rows[1]["value"] == "Portland"
    assert latent_rows[0]["evidence"] == [evidence]

    row = {
        COL_RAW_SENSITIVITY_DISPOSITION: {
            "sensitivity_disposition": [
                {"id": i, "sensitivity": "medium", "protection_method_suggestion": "generalize"} for i in range(1, 4)
            ]
        },
        COL_DISPOSITION_EXPLICIT_ENTITIES: explicit,
        COL_DISPOSITION_LATENT_ENTITIES: latent,
        COL_ENTITY_CLASSIFICATION: {"first_name": "direct_identifier", "city": "quasi_identifier"},
    }
    result = _verify_disposition_coverage(row)
    assert result[COL_DISPOSITION_COVERAGE] is True
    assert result[COL_SENSITIVITY_DISPOSITION]["sensitivity_disposition"][0]["entity_value"] == value


def test_build_numbered_entities_marks_empty_lists() -> None:
    assert build_numbered_entities({"entities_by_value": []}, {"latent_entities": []}) == ("(none)", "(none)")
    assert build_numbered_entities(None, None) == ("(none)", "(none)")


def test_number_disposition_entities_column_writes_both_outputs() -> None:
    row = {
        COL_ENTITIES_BY_VALUE: {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]},
        COL_LATENT_ENTITIES: _STUB_LATENT,
    }
    result = _number_disposition_entities(row)
    assert json.loads(result[COL_DISPOSITION_EXPLICIT_ENTITIES])["id"] == 1
    assert json.loads(result[COL_DISPOSITION_LATENT_ENTITIES])["id"] == 2


# ---------------------------------------------------------------------------
# Coverage verification
# ---------------------------------------------------------------------------

_CLASSIFICATIONS = {"first_name": "direct_identifier", "city": "quasi_identifier"}

_TWO_ENTITIES = {
    "entities_by_value": [
        {"value": "Alice", "labels": ["first_name"]},
        {"value": "Portland", "labels": ["city"]},
    ]
}


def _disposition(*entries: tuple[int, str, str]) -> SensitivityDispositionSchema:
    return SensitivityDispositionSchema.model_validate(
        {
            "sensitivity_disposition": [
                {
                    "id": entry_id,
                    "source": "tagged",
                    "category": _CLASSIFICATIONS[label],
                    "sensitivity": "high" if label == "first_name" else "low",
                    "low_sensitivity_reason": None if label == "first_name" else "No meaningful contextual linkage.",
                    "entity_label": label,
                    "entity_value": value,
                    "protection_method_suggestion": "replace" if label == "first_name" else "leave_as_is",
                }
                for entry_id, label, value in entries
            ],
        }
    )


def _compact(disposition: SensitivityDispositionSchema | dict[str, Any]) -> dict[str, Any]:
    payload = (
        disposition.model_dump(mode="json") if isinstance(disposition, SensitivityDispositionSchema) else disposition
    )
    fields = {"id", "sensitivity", "protection_method_suggestion", "low_sensitivity_reason"}
    return {
        "sensitivity_disposition": [
            {key: value for key, value in entry.items() if key in fields}
            for entry in payload["sensitivity_disposition"]
        ]
    }


def test_verify_disposition_coverage_accepts_exact_coverage() -> None:
    explicit, latent = build_numbered_entities(_TWO_ENTITIES, None)
    verify_disposition_coverage(
        _disposition((1, "first_name", "Alice"), (2, "city", "Portland")), explicit, latent, _CLASSIFICATIONS
    )


def test_verify_disposition_coverage_rejects_truncated_disposition() -> None:
    explicit, latent = build_numbered_entities(_TWO_ENTITIES, None)
    with pytest.raises(ValueError, match="1 entries but 2 entities were supplied"):
        verify_disposition_coverage(_disposition((1, "first_name", "Alice")), explicit, latent, _CLASSIFICATIONS)


def test_verify_disposition_coverage_counts_latent_entities() -> None:
    explicit, latent = build_numbered_entities(_TWO_ENTITIES, _STUB_LATENT)
    with pytest.raises(ValueError, match="2 entries but 3 entities were supplied"):
        verify_disposition_coverage(
            _disposition((1, "first_name", "Alice"), (2, "city", "Portland")), explicit, latent, _CLASSIFICATIONS
        )


def test_verify_disposition_coverage_rejects_extra_entries() -> None:
    explicit, latent = build_numbered_entities({"entities_by_value": []}, None)
    with pytest.raises(ValueError, match="1 entries but 0 entities were supplied"):
        verify_disposition_coverage(_disposition((1, "city", "Portland")), explicit, latent, _CLASSIFICATIONS)


@pytest.mark.parametrize("source", ["tagged", "latent"])
def test_verify_disposition_coverage_restores_identity_on_mismatch(
    caplog: pytest.LogCaptureFixture, source: str
) -> None:
    supplied = json.dumps({"id": 1, "label": "occupation", "value": "teacher"})
    disposition = _disposition((1, "city", "Portland"))
    original = disposition.sensitivity_disposition[0].model_dump()
    with caplog.at_level("WARNING", logger="anonymizer.rewrite.sensitivity_disposition"):
        verify_disposition_coverage(
            disposition,
            supplied if source == "tagged" else "(none)",
            supplied if source == "latent" else "(none)",
            {"occupation": "quasi_identifier"},
        )
    entry = disposition.sensitivity_disposition[0]
    assert entry.entity_label == "occupation"
    assert entry.entity_value == "teacher"
    assert entry.sensitivity == original["sensitivity"]
    assert entry.protection_method_suggestion == original["protection_method_suggestion"]
    assert "does not match supplied entity" in caplog.text
    assert "restoring supplied identity" in caplog.text


def test_verify_disposition_column_marks_row_verified() -> None:
    explicit, latent = build_numbered_entities(_TWO_ENTITIES, None)
    row = {
        COL_RAW_SENSITIVITY_DISPOSITION: _compact(_disposition((1, "first_name", "Alice"), (2, "city", "Portland"))),
        COL_DISPOSITION_EXPLICIT_ENTITIES: explicit,
        COL_DISPOSITION_LATENT_ENTITIES: latent,
        COL_ENTITY_CLASSIFICATION: _CLASSIFICATIONS,
    }
    assert _verify_disposition_coverage(row)[COL_DISPOSITION_COVERAGE] is True


def test_compact_response_schema_has_only_decision_fields() -> None:
    schema = RawSensitivityDispositionSchema.model_json_schema()
    fields = schema["$defs"]["RawEntityDispositionSchema"]["properties"]
    assert set(fields) == {"id", "sensitivity", "protection_method_suggestion", "low_sensitivity_reason"}
    prompt = _get_sensitivity_disposition_prompt(_STUB_PRIVACY_GOAL)
    output = prompt.split("<output>")[1]
    assert "Do not repeat source, category" in output
    assert "- entity_value" not in output


def test_compact_disposition_reconstructs_identity_and_normalizes_policy() -> None:
    entities = {"entities_by_value": [{"value": "José\u202fGarcía", "labels": ["first_name", "city"]}]}
    explicit, latent = build_numbered_entities(entities, _STUB_LATENT)
    raw = {
        "sensitivity_disposition": [
            {"id": 1, "sensitivity": "medium", "protection_method_suggestion": "generalize"},
            {
                "id": 2,
                "sensitivity": "low",
                "protection_method_suggestion": "leave_as_is",
                "low_sensitivity_reason": "No additional linkage in context.",
            },
            {"id": 3, "sensitivity": "medium", "protection_method_suggestion": "remove"},
        ]
    }
    row = {
        COL_RAW_SENSITIVITY_DISPOSITION: json.dumps(raw),
        COL_DISPOSITION_EXPLICIT_ENTITIES: explicit,
        COL_DISPOSITION_LATENT_ENTITIES: latent,
        COL_ENTITY_CLASSIFICATION: _CLASSIFICATIONS,
    }

    result = _verify_disposition_coverage(row)
    first, second, third = result[COL_SENSITIVITY_DISPOSITION]["sensitivity_disposition"]

    assert (first["id"], first["source"], first["category"], first["entity_label"], first["entity_value"]) == (
        1,
        "tagged",
        "direct_identifier",
        "first_name",
        "José\u202fGarcía",
    )
    assert first["sensitivity"] == "high"
    assert first["protection_method_suggestion"] == "generalize"
    assert second["entity_label"] == "city"
    assert second["entity_value"] == first["entity_value"]
    assert second["sensitivity"] == "low"
    assert second["low_sensitivity_reason"] == "No additional linkage in context."
    assert (third["source"], third["category"], third["entity_label"], third["entity_value"]) == (
        "latent",
        "latent_identifier",
        "employer",
        "a large cloud provider",
    )
    assert third["protection_method_suggestion"] == "suppress_inference"
    assert result[COL_DISPOSITION_COVERAGE] is True
    assert result[COL_RAW_SENSITIVITY_DISPOSITION] == json.dumps(raw)


@pytest.mark.parametrize("count", [1, 3])
def test_compact_disposition_rejects_incomplete_or_extra_coverage(count: int) -> None:
    explicit, latent = build_numbered_entities(_TWO_ENTITIES, None)
    row = {
        COL_RAW_SENSITIVITY_DISPOSITION: {
            "sensitivity_disposition": [
                {"id": i, "sensitivity": "medium", "protection_method_suggestion": "generalize"}
                for i in range(1, count + 1)
            ]
        },
        COL_DISPOSITION_EXPLICIT_ENTITIES: explicit,
        COL_DISPOSITION_LATENT_ENTITIES: latent,
        COL_ENTITY_CLASSIFICATION: _CLASSIFICATIONS,
    }
    with pytest.raises(ValueError, match=f"{count} entries but 2 entities were supplied"):
        _verify_disposition_coverage(row)
    assert COL_DISPOSITION_COVERAGE not in row


@pytest.mark.parametrize("ids", [[1, 1], [2, 1], [1, 3]])
def test_compact_response_rejects_duplicate_reordered_or_missing_ids(ids: list[int]) -> None:
    with pytest.raises(ValueError, match="IDs must be sequential"):
        RawSensitivityDispositionSchema.model_validate(
            {
                "sensitivity_disposition": [
                    {"id": i, "sensitivity": "medium", "protection_method_suggestion": "generalize"} for i in ids
                ]
            }
        )


def test_compact_disposition_rejects_missing_category() -> None:
    explicit, latent = build_numbered_entities(_TWO_ENTITIES, None)
    row = {
        COL_RAW_SENSITIVITY_DISPOSITION: _compact(_disposition((1, "first_name", "Alice"), (2, "city", "Portland"))),
        COL_DISPOSITION_EXPLICIT_ENTITIES: explicit,
        COL_DISPOSITION_LATENT_ENTITIES: latent,
        COL_ENTITY_CLASSIFICATION: {},
    }
    with pytest.raises(ValueError, match="missing or invalid supplied category"):
        _verify_disposition_coverage(row)


def test_compact_response_checks_low_sensitivity_reason_before_reconstruction() -> None:
    with pytest.raises(ValueError, match="low sensitivity requires a non-empty"):
        RawSensitivityDispositionSchema.model_validate(
            {
                "sensitivity_disposition": [
                    {"id": 1, "sensitivity": "low", "protection_method_suggestion": "leave_as_is"}
                ]
            }
        )


@pytest.mark.parametrize(("sensitivity", "method"), [("low", "leave_as_is"), ("medium", "leave_as_is")])
def test_strict_compact_response_disallows_unprotected_decisions(sensitivity: str, method: str) -> None:
    with pytest.raises(ValueError):
        StrictRawSensitivityDispositionSchema.model_validate(
            {"sensitivity_disposition": [{"id": 1, "sensitivity": sensitivity, "protection_method_suggestion": method}]}
        )


@pytest.mark.parametrize(
    ("supplied_value", "echoed_value"),
    [
        ("Southern\u202fPublishing", "Southern Publishing"),
        ("Southern\u00a0Publishing", "Southern Publishing"),
        ("40‑year‑old", "40-year-old"),
        ("master’s degree", "master's degree"),
        ("Café", "Cafe\u0301"),
    ],
)
def test_verify_disposition_logs_formatting_corrections_only_at_debug(
    supplied_value: str, echoed_value: str, caplog: pytest.LogCaptureFixture
) -> None:
    disposition = _disposition((1, "city", echoed_value))
    supplied = json.dumps({"id": 1, "label": "city", "value": supplied_value})
    with caplog.at_level("DEBUG", logger="anonymizer.rewrite.sensitivity_disposition"):
        verify_disposition_coverage(disposition, supplied, "(none)", _CLASSIFICATIONS)
    assert disposition.sensitivity_disposition[0].entity_value == supplied_value
    assert {record.levelname for record in caplog.records} == {"DEBUG"}
    assert "restoring supplied identity" in caplog.text


def test_verify_disposition_warns_once_for_substantive_identity_correction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    disposition = _disposition((1, "city", "Portland"))
    supplied = json.dumps({"id": 1, "label": "city", "value": "Seattle"})
    verify_disposition_coverage(disposition, supplied, "(none)", _CLASSIFICATIONS)
    assert disposition.sensitivity_disposition[0].entity_value == "Seattle"
    assert {record.levelname for record in caplog.records} == {"WARNING"}
    assert "does not match supplied entity" in caplog.text
    assert "source/category/sensitivity/method" not in caplog.text


def test_verify_disposition_warns_for_policy_change_alongside_formatting_correction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    disposition = _disposition((1, "city", "Southern Publishing"))
    supplied = json.dumps({"id": 1, "label": "city", "value": "Southern\u202fPublishing"})
    with caplog.at_level("DEBUG", logger="anonymizer.rewrite.sensitivity_disposition"):
        verify_disposition_coverage(disposition, supplied, "(none)", {"city": "direct_identifier"})
    entry = disposition.sensitivity_disposition[0]
    assert entry.entity_value == "Southern\u202fPublishing"
    assert entry.sensitivity == "high"
    assert entry.protection_method_suggestion == "replace"
    assert {record.levelname for record in caplog.records} == {"WARNING", "DEBUG"}


@pytest.mark.parametrize(("value", "method"), [("O.K.", "generalize"), ("Kotu", "replace")])
def test_verify_disposition_normalizes_surname_and_preserves_method(value: str, method: str) -> None:
    explicit = json.dumps({"id": 1, "label": "last_name", "value": value})
    payload = _disposition((1, "city", value)).model_dump(mode="json")
    payload["sensitivity_disposition"][0].update(
        entity_label="last_name", sensitivity="medium", protection_method_suggestion=method
    )
    row = {
        COL_RAW_SENSITIVITY_DISPOSITION: _compact(payload),
        COL_DISPOSITION_EXPLICIT_ENTITIES: explicit,
        COL_DISPOSITION_LATENT_ENTITIES: "(none)",
        COL_ENTITY_CLASSIFICATION: {"last_name": "direct_identifier"},
    }
    result = _verify_disposition_coverage(row)
    entry = result[COL_SENSITIVITY_DISPOSITION]["sensitivity_disposition"][0]
    assert entry["category"] == "direct_identifier"
    assert entry["sensitivity"] == "high"
    assert entry["protection_method_suggestion"] == method
    assert result[COL_DISPOSITION_COVERAGE] is True


def test_verify_disposition_upgrades_retained_direct_identifier(caplog: pytest.LogCaptureFixture) -> None:
    disposition = _disposition((1, "city", "Alice"))
    explicit = json.dumps({"id": 1, "label": "city", "value": "Alice"})
    verify_disposition_coverage(disposition, explicit, "(none)", {"city": "direct_identifier"})
    entry = disposition.sensitivity_disposition[0]
    assert entry.sensitivity == "high"
    assert entry.protection_method_suggestion == "replace"
    assert "protection_reason" not in entry.model_dump()
    assert "Normalizing disposition entity 1" in caplog.text


def test_verify_disposition_rejects_missing_classification() -> None:
    explicit, latent = build_numbered_entities(_TWO_ENTITIES, None)
    with pytest.raises(ValueError, match="missing or invalid supplied category"):
        verify_disposition_coverage(
            _disposition((1, "first_name", "Alice"), (2, "city", "Portland")), explicit, latent, {}
        )


def test_verify_disposition_normalizes_latent_source_and_category() -> None:
    disposition = _disposition((1, "city", "Portland"))
    latent = json.dumps({"id": 1, "label": "city", "value": "Portland"})
    verify_disposition_coverage(disposition, "(none)", latent, {})
    entry = disposition.sensitivity_disposition[0]
    assert entry.source == "latent"
    assert entry.category == "latent_identifier"
    assert entry.sensitivity == "low"
    assert entry.protection_method_suggestion == "leave_as_is"


@pytest.mark.parametrize("sensitivity", ["medium", "high"])
@pytest.mark.parametrize("method", ["replace", "generalize", "remove", "suppress_inference"])
@pytest.mark.parametrize("reported_source", ["tagged", "latent"])
def test_verify_disposition_normalizes_protected_latent_method(
    sensitivity: str, method: str, reported_source: str, caplog: pytest.LogCaptureFixture
) -> None:
    payload = _disposition((1, "first_name", "Alice")).model_dump(mode="json")
    payload["sensitivity_disposition"][0].update(
        source=reported_source, sensitivity=sensitivity, protection_method_suggestion=method
    )
    disposition = SensitivityDispositionSchema.model_validate(payload)
    latent = json.dumps({"id": 1, "label": "first_name", "value": "Alice"})

    verify_disposition_coverage(disposition, "(none)", latent, {})

    entry = disposition.sensitivity_disposition[0]
    assert entry.source == "latent"
    assert entry.category == "latent_identifier"
    assert entry.sensitivity == sensitivity
    assert entry.protection_method_suggestion == "suppress_inference"
    if method != "suppress_inference":
        assert "Normalizing disposition entity 1" in caplog.text


@custom_column_generator(required_columns=[COL_DISPOSITION_EXPLICIT_ENTITIES])
def _generate_compact_disposition(row: dict[str, Any]) -> dict[str, Any]:
    row[COL_RAW_SENSITIVITY_DISPOSITION] = {
        "sensitivity_disposition": [
            {
                "id": 1,
                "sensitivity": "medium",
                "protection_method_suggestion": "generalize",
            }
        ]
    }
    return row


def test_scheduler_propagates_normalized_disposition(
    tmp_path: Path,
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    columns = SensitivityDispositionWorkflow().columns(
        selected_models=stub_rewrite_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
    )
    # Replace only the remote LLM call; execute normalization and consumers in the real scheduler.
    columns[1] = CustomColumnConfig(
        name=COL_RAW_SENSITIVITY_DISPOSITION,
        generator_function=_generate_compact_disposition,
    )
    columns.extend(
        [
            CustomColumnConfig(name=COL_PRIVACY_QA, generator_function=_generate_privacy_qa_column),
            CustomColumnConfig(
                name=COL_REWRITE_DISPOSITION_BLOCK, generator_function=_format_rewrite_disposition_block
            ),
        ]
    )
    adapter = NddAdapter(DataDesigner(artifact_path=tmp_path / "artifacts", auto_configure_logging=False))
    result = adapter.run_workflow(
        pd.DataFrame(
            {
                COL_ENTITIES_BY_VALUE: [
                    {"entities_by_value": [{"value": value, "labels": ["last_name"]}]} for value in ["O.K.", "Kotu"]
                ],
                COL_LATENT_ENTITIES: [{"latent_entities": []}] * 2,
                COL_ENTITY_CLASSIFICATION: [{"last_name": "direct_identifier"}] * 2,
            }
        ),
        model_configs=[],
        columns=columns,
        workflow_name="disposition-normalization",
        preview_num_records=2,
    )
    assert result.failed_records == []
    assert len(result.dataframe) == 2
    for _, row in result.dataframe.iterrows():
        raw = normalize_payload(row[COL_RAW_SENSITIVITY_DISPOSITION])["sensitivity_disposition"][0]
        normalized = normalize_payload(row[COL_SENSITIVITY_DISPOSITION])["sensitivity_disposition"][0]
        qa = normalize_payload(row[COL_PRIVACY_QA])["items"][0]
        rewrite = normalize_payload(row[COL_REWRITE_DISPOSITION_BLOCK])[0]
        assert "category" not in raw
        assert "entity_label" not in raw
        assert "entity_value" not in raw
        assert normalized["entity_value"] in {"O.K.", "Kotu"}
        assert raw["sensitivity"] == "medium"
        assert normalized["category"] == qa["category"] == "direct_identifier"
        assert normalized["sensitivity"] == qa["sensitivity"] == rewrite["sensitivity"] == "high"
        assert normalized["protection_method_suggestion"] == rewrite["protection_method_suggestion"] == "generalize"
        assert row[COL_DISPOSITION_COVERAGE]
