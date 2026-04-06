# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig

from anonymizer.config.models import RewriteModelSelection
from anonymizer.engine.constants import (
    COL_DOMAIN,
    COL_DOMAIN_SUPPLEMENT,
    COL_MEANING_UNITS,
    COL_MEANING_UNITS_SERIALIZED,
    COL_PRIVACY_QA,
    COL_QUALITY_QA,
    COL_SENSITIVITY_DISPOSITION,
    COL_SENSITIVITY_DISPOSITION_BLOCK,
    COL_TEXT,
    _jinja,
)
from anonymizer.engine.rewrite.qa_generation import (
    _DOMAIN_KEY,
    QAGenerationWorkflow,
    _format_disposition_block,
    _generate_privacy_qa_column,
    _get_meaning_unit_extraction_prompt,
    _get_quality_qa_prompt,
    _serialize_meaning_units,
    generate_privacy_qa_from_disposition,
)
from anonymizer.engine.schemas import (
    EntityCategory,
    EntityDispositionSchema,
    EntitySource,
    MeaningUnitAspect,
    MeaningUnitSchema,
    MeaningUnitsSchema,
    PrivacyQAPairsSchema,
    ProtectionMethod,
    SensitivityDispositionSchema,
    SensitivityLevel,
)

_STUB_DISPOSITION = SensitivityDispositionSchema(
    sensitivity_disposition=[
        EntityDispositionSchema(
            id=1,
            source=EntitySource.tagged,
            category=EntityCategory.direct_identifier,
            sensitivity=SensitivityLevel.high,
            entity_label="first_name",
            entity_value="Alice",
            needs_protection=True,
            protection_reason="Full name directly identifies the individual.",
            protection_method_suggestion=ProtectionMethod.replace,
            combined_risk_level="high",
        ),
        EntityDispositionSchema(
            id=2,
            source=EntitySource.tagged,
            category=EntityCategory.quasi_identifier,
            sensitivity=SensitivityLevel.low,
            entity_label="city",
            entity_value="Portland",
            needs_protection=False,
            protection_reason="City alone does not create meaningful re-identification risk here.",
            protection_method_suggestion=ProtectionMethod.leave_as_is,
            combined_risk_level="low",
        ),
    ]
)

_STUB_MEANING_UNITS = MeaningUnitsSchema(
    units=[
        MeaningUnitSchema(id=1, aspect=MeaningUnitAspect.ROLE, unit="An individual works as a software engineer."),
        MeaningUnitSchema(id=2, aspect=MeaningUnitAspect.ENVIRONMENT, unit="The individual works remotely."),
    ]
)


def test_columns_returns_exactly_five_in_order(
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    cols = QAGenerationWorkflow().columns(selected_models=stub_rewrite_model_selection)
    assert len(cols) == 5
    assert isinstance(cols[0], CustomColumnConfig)
    assert isinstance(cols[1], LLMStructuredColumnConfig)
    assert isinstance(cols[2], CustomColumnConfig)
    assert isinstance(cols[3], LLMStructuredColumnConfig)
    assert isinstance(cols[4], CustomColumnConfig)
    assert cols[0].name == COL_SENSITIVITY_DISPOSITION_BLOCK
    assert cols[1].name == COL_MEANING_UNITS
    assert cols[2].name == COL_MEANING_UNITS_SERIALIZED
    assert cols[3].name == COL_QUALITY_QA
    assert cols[4].name == COL_PRIVACY_QA


def test_meaning_extractor_alias_used(
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    cols = QAGenerationWorkflow().columns(selected_models=stub_rewrite_model_selection)
    assert cols[1].model_alias == stub_rewrite_model_selection.meaning_extractor


def test_qa_generator_alias_used(
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    cols = QAGenerationWorkflow().columns(selected_models=stub_rewrite_model_selection)
    assert cols[3].model_alias == stub_rewrite_model_selection.qa_generator


def test_format_disposition_block_produces_valid_json() -> None:
    row = {COL_SENSITIVITY_DISPOSITION: _STUB_DISPOSITION}
    result = _format_disposition_block(row)
    block = json.loads(result[COL_SENSITIVITY_DISPOSITION_BLOCK])
    assert len(block) == 2
    assert block[0]["entity_value"] == "Alice"
    assert block[0]["does_need_protection"] is True
    assert block[0]["protection_method_suggestion"] == "replace"
    assert block[1]["entity_value"] == "Portland"
    assert block[1]["does_need_protection"] is False


def test_format_disposition_block_accepts_dict_payload() -> None:
    row = {COL_SENSITIVITY_DISPOSITION: _STUB_DISPOSITION.model_dump(mode="python")}
    result = _format_disposition_block(row)
    block = json.loads(result[COL_SENSITIVITY_DISPOSITION_BLOCK])
    assert len(block) == 2
    assert block[0]["entity_value"] == "Alice"


def test_serialize_meaning_units_produces_valid_json() -> None:
    row = {COL_MEANING_UNITS: _STUB_MEANING_UNITS}
    result = _serialize_meaning_units(row)
    serialized = json.loads(result[COL_MEANING_UNITS_SERIALIZED])
    assert len(serialized) == 2
    assert serialized[0]["id"] == 1
    assert serialized[0]["aspect"] == "role"
    assert "software engineer" in serialized[0]["unit"]


def test_serialize_meaning_units_accepts_dict_payload() -> None:
    row = {COL_MEANING_UNITS: _STUB_MEANING_UNITS.model_dump(mode="python")}
    result = _serialize_meaning_units(row)
    serialized = json.loads(result[COL_MEANING_UNITS_SERIALIZED])
    assert len(serialized) == 2
    assert serialized[1]["id"] == 2


def test_generate_privacy_qa_column_only_protected_entities() -> None:
    row = {COL_SENSITIVITY_DISPOSITION: _STUB_DISPOSITION}
    result = _generate_privacy_qa_column(row)
    qa = PrivacyQAPairsSchema.model_validate(result[COL_PRIVACY_QA])
    # Only Alice needs protection; Portland does not
    assert len(qa.items) == 1
    assert "Alice" in qa.items[0].question
    assert qa.items[0].entity_label == "first_name"
    assert qa.items[0].sensitivity == SensitivityLevel.high


def test_generate_privacy_qa_column_accepts_dict_payload() -> None:
    row = {COL_SENSITIVITY_DISPOSITION: _STUB_DISPOSITION.model_dump(mode="python")}
    result = _generate_privacy_qa_column(row)
    qa = PrivacyQAPairsSchema.model_validate(result[COL_PRIVACY_QA])
    assert len(qa.items) == 1
    assert qa.items[0].entity_value == "Alice"


def test_generate_privacy_qa_column_no_protected_entities() -> None:
    disposition = SensitivityDispositionSchema(
        sensitivity_disposition=[
            EntityDispositionSchema(
                id=1,
                source=EntitySource.tagged,
                category=EntityCategory.quasi_identifier,
                sensitivity=SensitivityLevel.low,
                entity_label="city",
                entity_value="Portland",
                needs_protection=False,
                protection_reason="City alone does not create meaningful re-identification risk.",
                protection_method_suggestion=ProtectionMethod.leave_as_is,
                combined_risk_level="low",
            )
        ]
    )
    row = {COL_SENSITIVITY_DISPOSITION: disposition}
    result = _generate_privacy_qa_column(row)
    qa = PrivacyQAPairsSchema.model_validate(result[COL_PRIVACY_QA])
    assert len(qa.items) == 0


def test_generate_privacy_qa_from_disposition_only_protected_entities() -> None:
    qa = generate_privacy_qa_from_disposition(_STUB_DISPOSITION)
    assert len(qa.items) == 1
    assert "Alice" in qa.items[0].question
    assert qa.items[0].entity_label == "first_name"
    assert qa.items[0].sensitivity == SensitivityLevel.high


def test_generate_privacy_qa_from_disposition_empty_when_nothing_to_protect() -> None:
    disposition = SensitivityDispositionSchema(
        sensitivity_disposition=[
            EntityDispositionSchema(
                id=1,
                source=EntitySource.tagged,
                category=EntityCategory.quasi_identifier,
                sensitivity=SensitivityLevel.low,
                entity_label="city",
                entity_value="Portland",
                needs_protection=False,
                protection_reason="City alone does not create meaningful re-identification risk.",
                protection_method_suggestion=ProtectionMethod.leave_as_is,
                combined_risk_level="low",
            )
        ]
    )
    assert generate_privacy_qa_from_disposition(disposition).items == []


def test_generate_privacy_qa_from_disposition_ids_are_sequential() -> None:
    disposition = SensitivityDispositionSchema(
        sensitivity_disposition=[
            EntityDispositionSchema(
                id=1,
                source=EntitySource.tagged,
                category=EntityCategory.direct_identifier,
                sensitivity=SensitivityLevel.high,
                entity_label="first_name",
                entity_value="Alice",
                needs_protection=True,
                protection_reason="Direct identifier.",
                protection_method_suggestion=ProtectionMethod.replace,
                combined_risk_level="high",
            ),
            EntityDispositionSchema(
                id=2,
                source=EntitySource.tagged,
                category=EntityCategory.direct_identifier,
                sensitivity=SensitivityLevel.high,
                entity_label="last_name",
                entity_value="Smith",
                needs_protection=True,
                protection_reason="Direct identifier.",
                protection_method_suggestion=ProtectionMethod.replace,
                combined_risk_level="high",
            ),
        ]
    )
    qa = generate_privacy_qa_from_disposition(disposition)
    assert [item.id for item in qa.items] == [1, 2]


def test_meaning_unit_prompt_references_required_columns() -> None:
    prompt = _get_meaning_unit_extraction_prompt()
    assert _jinja(COL_SENSITIVITY_DISPOSITION_BLOCK) in prompt
    assert _jinja(COL_DOMAIN, key=_DOMAIN_KEY) in prompt
    assert _jinja(COL_DOMAIN_SUPPLEMENT) in prompt
    assert _jinja(COL_TEXT) in prompt


def test_meaning_unit_prompt_preserves_gitlab_protection_branches() -> None:
    prompt = _get_meaning_unit_extraction_prompt()
    assert "does_need_protection = True" in prompt
    assert 'protection_method_suggestion is "replace" OR "remove"' in prompt
    assert 'protection_method_suggestion is "generalize" OR "suppress_inference"' in prompt
    assert "does_need_protection = False" in prompt


def test_meaning_unit_prompt_keeps_xml_style_blocks() -> None:
    prompt = _get_meaning_unit_extraction_prompt()
    assert "<entity_protection_rules>" in prompt
    assert "<importance_criteria>" in prompt
    assert "<segmentation_rules>" in prompt
    assert "<domain_context>" in prompt
    assert "<input>" in prompt


def test_quality_qa_prompt_references_meaning_units_serialized() -> None:
    prompt = _get_quality_qa_prompt()
    assert _jinja(COL_MEANING_UNITS_SERIALIZED) in prompt
