# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from anonymizer.config.anonymizer_config import (
    AnonymizerConfig,
    AnonymizerInput,
    Detect,
    Rewrite,
    infer_input_source_suffix,
)
from anonymizer.config.regex import BuiltinRegex, RegexCandidate, RegexRule
from anonymizer.config.replace_strategies import (
    Annotate,
    Hash,
    Redact,
)


def test_hash_is_deterministic() -> None:
    strategy = Hash()
    value = strategy.replace(text="alice@example.com", label="email")
    assert value == strategy.replace(text="alice@example.com", label="email")


def test_rewrite_defaults_privacy_goal() -> None:
    config = AnonymizerConfig(rewrite=Rewrite())
    assert config.rewrite is not None
    assert config.rewrite.privacy_goal is not None


def test_data_summary_on_input(tmp_path: Path) -> None:
    source_path = tmp_path / "data.csv"
    source_path.write_text("text\nsample\n")
    inp = AnonymizerInput(
        source=str(source_path),
        data_summary="Medical clinic visit notes from outpatient encounters.",
    )
    assert inp.data_summary is not None


def test_input_source_accepts_http_url_without_local_path_check() -> None:
    inp = AnonymizerInput(source="https://example.com/data.csv")
    assert inp.source == "https://example.com/data.csv"


def test_input_source_accepts_http_url_with_fragment() -> None:
    inp = AnonymizerInput(source="https://example.com/data.csv#preview")
    assert inp.source == "https://example.com/data.csv#preview"


def test_infer_input_source_suffix_ignores_url_fragment() -> None:
    assert infer_input_source_suffix("https://example.com/data.csv#preview") == ".csv"


def test_input_source_rejects_unsupported_url_scheme() -> None:
    with pytest.raises(ValidationError, match="Unsupported input URL scheme"):
        AnonymizerInput(source="ftp://example.com/data.csv")


def test_replace_and_rewrite_together_raises() -> None:
    with pytest.raises(ValueError, match="Cannot use both replace and rewrite"):
        AnonymizerConfig(replace=Redact(), rewrite=Rewrite())


def test_neither_replace_nor_rewrite_raises() -> None:
    with pytest.raises(ValueError, match="Exactly one of replace or rewrite"):
        AnonymizerConfig()


def test_annotate_accepts_custom_template() -> None:
    strategy = Annotate(format_template="[{label}]::{text}")
    assert strategy.replace(text="Alice", label="name") == "[name]::Alice"


def test_redact_defaults_to_label_aware_output() -> None:
    strategy = Redact()
    assert strategy.replace(text="Alice", label="first_name") == "[REDACTED_FIRST_NAME]"


def test_redact_allows_constant_template() -> None:
    strategy = Redact(format_template="****")
    assert strategy.replace(text="Alice", label="first_name") == "****"


def test_entity_labels_defaults_to_none() -> None:
    config = AnonymizerConfig(replace=Redact())
    assert config.detect.entity_labels is None


def test_entity_labels_accepts_list() -> None:
    config = AnonymizerConfig(detect={"entity_labels": ["FIRST_NAME", "email"]}, replace=Redact())
    assert config.detect.entity_labels is not None
    assert set(config.detect.entity_labels) == {"first_name", "email"}


def test_entity_labels_strips_whitespace() -> None:
    config = AnonymizerConfig(detect={"entity_labels": ["  first_name ", "email"]}, replace=Redact())
    assert config.detect.entity_labels is not None
    assert "first_name" in config.detect.entity_labels
    assert "email" in config.detect.entity_labels


def test_entity_labels_deduplicates() -> None:
    config = AnonymizerConfig(detect={"entity_labels": ["email", "email"]}, replace=Redact())
    assert config.detect.entity_labels == ["email"]


def test_entity_labels_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        AnonymizerConfig(detect={"entity_labels": []}, replace=Redact())


def test_entity_labels_whitespace_only_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        AnonymizerConfig(detect={"entity_labels": ["  ", ""]}, replace=Redact())


def test_both_modes_set_exits() -> None:
    """Setting both replace and rewrite on AnonymizerConfig violates the model_validator."""
    with pytest.raises(ValidationError):
        AnonymizerConfig(replace=Redact(), rewrite=Rewrite())


def test_detect_chunked_validation_defaults() -> None:
    config = AnonymizerConfig(replace=Redact())
    assert config.detect.validation_max_entities_per_call == 100
    assert config.detect.validation_excerpt_window_chars == 500


def test_detect_chunked_validation_accepts_overrides() -> None:
    config = AnonymizerConfig(
        detect={
            "validation_max_entities_per_call": 25,
            "validation_excerpt_window_chars": 1000,
        },
        replace=Redact(),
    )
    assert config.detect.validation_max_entities_per_call == 25
    assert config.detect.validation_excerpt_window_chars == 1000


def test_detect_validation_max_entities_per_call_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        AnonymizerConfig(detect={"validation_max_entities_per_call": 0}, replace=Redact())


def test_detect_validation_excerpt_window_chars_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        AnonymizerConfig(detect={"validation_excerpt_window_chars": 0}, replace=Redact())


def test_detect_enables_builtin_regexes_by_default() -> None:
    assert Detect().builtin_regexes is True


def test_regex_rule_defaults_to_llm_validation() -> None:
    rule = RegexRule(label="CASE_ID", pattern=r"CASE-[0-9]{8}")

    assert rule.label == "case_id"
    assert rule.validate_with_llm is True


def test_regex_rule_accepts_direct_validator_callable() -> None:
    def validate(candidate: RegexCandidate) -> bool:
        return candidate.value != "CASE-00000000"

    rule = RegexRule(label="case_id", pattern=r"CASE-[0-9]{8}", validator=validate)

    assert rule.validator is validate


def test_regex_rule_rejects_invalid_and_empty_matching_patterns() -> None:
    with pytest.raises(ValidationError, match="Invalid regex pattern"):
        RegexRule(label="case_id", pattern="[")
    with pytest.raises(ValidationError, match="must not match an empty string"):
        RegexRule(label="case_id", pattern=".*")


def test_detect_rejects_custom_rule_missing_from_explicit_labels() -> None:
    with pytest.raises(ValidationError, match="missing from explicit entity_labels"):
        Detect(
            entity_labels=["email"],
            regex_rules=[RegexRule(label="case_id", pattern=r"CASE-[0-9]{8}")],
        )


def test_detect_accepts_builtin_regex_llm_override() -> None:
    detect = Detect(regex_rules=[BuiltinRegex(label="credit_debit_card", validate_with_llm=False)])

    rule = detect.regex_rules[0]
    assert isinstance(rule, BuiltinRegex)
    assert rule.enabled is True
    assert rule.validate_with_llm is False


def test_detect_accepts_builtin_regex_enabled_override() -> None:
    detect = Detect(regex_rules=[BuiltinRegex(label="email", enabled=False)])

    rule = detect.regex_rules[0]
    assert isinstance(rule, BuiltinRegex)
    assert rule.enabled is False
    assert rule.validate_with_llm is True


def test_detect_rejects_duplicate_builtin_regex_entries() -> None:
    with pytest.raises(ValidationError, match="duplicate built-in labels"):
        Detect(
            regex_rules=[
                BuiltinRegex(label="email", enabled=False),
                BuiltinRegex(label="email", validate_with_llm=False),
            ]
        )


def test_detect_parses_builtin_and_custom_rules_from_serialized_config() -> None:
    original = Detect(
        regex_rules=[
            BuiltinRegex(label="email", enabled=False),
            RegexRule(label="support_case", pattern=r"CASE-\d+"),
        ]
    )

    restored = Detect.model_validate_json(original.model_dump_json())

    assert isinstance(restored.regex_rules[0], BuiltinRegex)
    assert isinstance(restored.regex_rules[1], RegexRule)
    assert restored == original


def test_builtin_regex_rejects_unknown_label() -> None:
    with pytest.raises(ValidationError, match="Unsupported built-in regex label"):
        BuiltinRegex(label="support_case")
