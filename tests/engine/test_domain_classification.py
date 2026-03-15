# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig

from anonymizer.config.models import RewriteModelSelection
from anonymizer.engine.constants import COL_DOMAIN, COL_DOMAIN_SUPPLEMENT, COL_TEXT, _jinja
from anonymizer.engine.rewrite.domain_classification import (
    DOMAIN_SUPPLEMENT_MAP,
    DomainClassificationWorkflow,
    _enrich_domain,
    _get_domain_classification_prompt,
)
from anonymizer.engine.schemas import Domain, DomainClassificationSchema


def test_columns_returns_exactly_two_in_order(
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    cols = DomainClassificationWorkflow().columns(selected_models=stub_rewrite_model_selection)
    assert len(cols) == 2
    assert isinstance(cols[0], LLMStructuredColumnConfig)
    assert isinstance(cols[1], CustomColumnConfig)
    assert cols[0].name == COL_DOMAIN
    assert cols[1].name == COL_DOMAIN_SUPPLEMENT
    assert cols[0].model_alias == stub_rewrite_model_selection.domain_classifier


def test_enrich_domain_populates_supplement_for_known_domain() -> None:
    domain_obj = DomainClassificationSchema(domain=Domain.BIOGRAPHY, domain_confidence=0.9)
    result = _enrich_domain({COL_DOMAIN: domain_obj})
    assert result[COL_DOMAIN_SUPPLEMENT] == DOMAIN_SUPPLEMENT_MAP[Domain.BIOGRAPHY]


def test_domain_classification_prompt_includes_text_jinja_placeholder() -> None:
    prompt = _get_domain_classification_prompt()
    assert _jinja(COL_TEXT) in prompt


def test_domain_classification_prompt_includes_data_summary() -> None:
    prompt = _get_domain_classification_prompt(data_summary="Clinical notes from primary care visits")
    assert "Clinical notes from primary care visits" in prompt
    assert "<data_context>" in prompt
    assert "Dataset description:" in prompt


def test_domain_classification_prompt_omits_data_context_when_none() -> None:
    prompt = _get_domain_classification_prompt(data_summary=None)
    assert "<data_context>" not in prompt


def test_domain_supplement_map_covers_all_domains() -> None:
    for domain in Domain:
        assert domain in DOMAIN_SUPPLEMENT_MAP, f"Missing supplement for domain: {domain}"
