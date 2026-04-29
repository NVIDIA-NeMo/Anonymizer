# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig

from anonymizer.config.models import RewriteModelSelection
from anonymizer.engine.constants import (
    COL_DOMAIN,
    COL_DOMAIN_SUPPLEMENT,
    COL_DOMAIN_SUPPLEMENT_PRIVACY,
    COL_TEXT,
    _jinja,
)
from anonymizer.engine.rewrite.domain_classification import (
    _DOMAIN_BY_ENUM,
    DOMAIN_METADATA,
    DomainClassificationWorkflow,
    DomainMetadata,
    _enrich_domain,
    _enrich_domain_privacy,
    _get_domain_classification_prompt,
)
from anonymizer.engine.schemas import Domain, DomainClassificationSchema


def test_columns_returns_exactly_three_in_order(
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    cols = DomainClassificationWorkflow().columns(selected_models=stub_rewrite_model_selection)
    assert len(cols) == 3
    assert isinstance(cols[0], LLMStructuredColumnConfig)
    assert isinstance(cols[1], CustomColumnConfig)
    assert isinstance(cols[2], CustomColumnConfig)
    assert cols[0].name == COL_DOMAIN
    assert cols[1].name == COL_DOMAIN_SUPPLEMENT
    assert cols[2].name == COL_DOMAIN_SUPPLEMENT_PRIVACY
    assert cols[0].model_alias == stub_rewrite_model_selection.domain_classifier


def test_enrich_domain_populates_supplement_for_known_domain() -> None:
    result = _enrich_domain({COL_DOMAIN: {"domain": Domain.BIOGRAPHY, "domain_confidence": 0.9}})
    assert result[COL_DOMAIN_SUPPLEMENT] == _DOMAIN_BY_ENUM[Domain.BIOGRAPHY].rewrite_supplement


def test_enrich_domain_accepts_schema_instance() -> None:
    domain_obj = DomainClassificationSchema(domain=Domain.BIOGRAPHY, domain_confidence=0.9)
    result = _enrich_domain({COL_DOMAIN: domain_obj})
    assert result[COL_DOMAIN_SUPPLEMENT] == _DOMAIN_BY_ENUM[Domain.BIOGRAPHY].rewrite_supplement


def test_enrich_domain_requires_domain_column() -> None:
    try:
        _enrich_domain({})
    except KeyError as exc:
        assert exc.args == (COL_DOMAIN,)
    else:
        raise AssertionError("Expected KeyError when COL_DOMAIN is missing")


def test_enrich_domain_privacy_populates_supplement_for_known_domain() -> None:
    result = _enrich_domain_privacy({COL_DOMAIN: {"domain": Domain.BIOGRAPHY, "domain_confidence": 0.9}})
    assert result[COL_DOMAIN_SUPPLEMENT_PRIVACY] == _DOMAIN_BY_ENUM[Domain.BIOGRAPHY].privacy_supplement


def test_enrich_domain_privacy_accepts_schema_instance() -> None:
    domain_obj = DomainClassificationSchema(domain=Domain.BIOGRAPHY, domain_confidence=0.9)
    result = _enrich_domain_privacy({COL_DOMAIN: domain_obj})
    assert result[COL_DOMAIN_SUPPLEMENT_PRIVACY] == _DOMAIN_BY_ENUM[Domain.BIOGRAPHY].privacy_supplement


def test_enrich_domain_privacy_requires_domain_column() -> None:
    try:
        _enrich_domain_privacy({})
    except KeyError as exc:
        assert exc.args == (COL_DOMAIN,)
    else:
        raise AssertionError("Expected KeyError when COL_DOMAIN is missing")


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


def test_domain_metadata_covers_every_domain_exactly_once() -> None:
    """Backstop for the module-level _validate_domain_coverage guard: every Domain has
    exactly one DomainMetadata entry, so rewrite/privacy/classification
    lookups can never drift from the enum."""
    assert {m.domain for m in DOMAIN_METADATA} == set(Domain)
    assert len(DOMAIN_METADATA) == len(set(Domain))
    for meta in DOMAIN_METADATA:
        assert isinstance(meta, DomainMetadata)
        assert meta.classification_description
        assert meta.rewrite_supplement
        assert meta.privacy_supplement
