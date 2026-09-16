# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from anonymizer.config.anonymizer_config import (
    AnonymizerConfig,
    AnonymizerInput,
    Rewrite,
    infer_input_source_suffix,
)
from anonymizer.config.replace_strategies import (
    Annotate,
    Hash,
    Redact,
)
from anonymizer.engine.constants import DEFAULT_ENTITY_LABELS


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


# ── excluded_entity_labels ────────────────────────────────────────────────────


def test_excluded_entity_labels_defaults_to_none() -> None:
    config = AnonymizerConfig(replace=Redact())
    assert config.detect.excluded_entity_labels is None


def test_excluded_entity_labels_accepts_list() -> None:
    config = AnonymizerConfig(detect={"excluded_entity_labels": ["EMAIL", "city"]}, replace=Redact())
    assert config.detect.excluded_entity_labels is not None
    assert set(config.detect.excluded_entity_labels) == {"email", "city"}


def test_excluded_entity_labels_strips_whitespace_and_lowercases() -> None:
    config = AnonymizerConfig(detect={"excluded_entity_labels": ["  FIRST_NAME ", "Email"]}, replace=Redact())
    assert config.detect.excluded_entity_labels is not None
    assert "first_name" in config.detect.excluded_entity_labels
    assert "email" in config.detect.excluded_entity_labels


def test_excluded_entity_labels_deduplicates(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        config = AnonymizerConfig(detect={"excluded_entity_labels": ["email", "email"]}, replace=Redact())
    assert config.detect.excluded_entity_labels == ["email"]
    assert "duplicates" in caplog.text


def test_excluded_entity_labels_empty_list_raises() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        AnonymizerConfig(detect={"excluded_entity_labels": []}, replace=Redact())


def test_excluded_entity_labels_whitespace_only_raises() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        AnonymizerConfig(detect={"excluded_entity_labels": ["  ", ""]}, replace=Redact())


def test_excluded_entity_labels_overlap_with_entity_labels_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        AnonymizerConfig(
            detect={"entity_labels": ["email", "city"], "excluded_entity_labels": ["email"]},
            replace=Redact(),
        )
    assert "email" in caplog.text
    assert "will never be detected" in caplog.text


def test_excluded_entity_labels_no_overlap_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        AnonymizerConfig(
            detect={"entity_labels": ["email", "city"], "excluded_entity_labels": ["first_name"]},
            replace=Redact(),
        )
    assert "will never be detected" not in caplog.text


def test_excluded_entity_labels_overlap_warning_only_fires_when_allowlist_explicit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No warning when entity_labels=None (defaults) even if exclusions are set."""
    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        AnonymizerConfig(
            detect={"excluded_entity_labels": ["email"]},
            replace=Redact(),
        )
    assert "will never be detected" not in caplog.text


def test_excluded_entity_labels_covering_all_defaults_raises() -> None:
    """entity_labels=None falls back to DEFAULT_ENTITY_LABELS; excluding all of it must also raise."""
    with pytest.raises(ValidationError, match="entirely overlaps DEFAULT_ENTITY_LABELS"):
        AnonymizerConfig(
            detect={"excluded_entity_labels": list(DEFAULT_ENTITY_LABELS)},
            replace=Redact(),
        )


def test_excluded_entity_labels_partial_default_coverage_does_not_raise() -> None:
    """Excluding some — but not all — default labels is the documented common case."""
    config = AnonymizerConfig(
        detect={"excluded_entity_labels": ["occupation", "gender"]},
        replace=Redact(),
    )
    assert config.detect.excluded_entity_labels == ["gender", "occupation"]


def test_excluded_entity_labels_fully_overlapping_entity_labels_raises() -> None:
    with pytest.raises(ValidationError, match="entirely overlaps"):
        AnonymizerConfig(
            detect={"entity_labels": ["email", "city"], "excluded_entity_labels": ["email", "city"]},
            replace=Redact(),
        )


def test_excluded_entity_labels_superset_of_entity_labels_raises() -> None:
    """excluded_entity_labels covering entity_labels plus extra labels still empties the set."""
    with pytest.raises(ValidationError, match="entirely overlaps"):
        AnonymizerConfig(
            detect={
                "entity_labels": ["email", "city"],
                "excluded_entity_labels": ["email", "city", "bank_account"],
            },
            replace=Redact(),
        )


def test_entity_labels_superset_of_excluded_entity_labels_only_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """entity_labels covering excluded_entity_labels plus extra labels still detects something."""
    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        config = AnonymizerConfig(
            detect={
                "entity_labels": ["email", "city", "bank_account"],
                "excluded_entity_labels": ["email", "city"],
            },
            replace=Redact(),
        )
    assert config.detect.entity_labels == ["bank_account", "city", "email"]
    assert "will never be detected" in caplog.text


# ── entity_label_examples ─────────────────────────────────────────────────────


def test_entity_label_examples_defaults_to_none() -> None:
    config = AnonymizerConfig(detect={}, replace=Redact())
    assert config.detect.entity_label_examples is None


def test_entity_label_examples_normalizes_keys_and_values() -> None:
    config = AnonymizerConfig(
        detect={
            "entity_labels": ["api_key", "user_name"],
            "entity_label_examples": {
                " API_KEY ": ["sk-ant-api03-abc123", " OPENAI_API_KEY=sk-proj-abc "],
                "User_Name": ["jsmith", "alice.chen"],
            },
        },
        replace=Redact(),
    )
    assert config.detect.entity_label_examples == {
        "api_key": ["sk-ant-api03-abc123", "OPENAI_API_KEY=sk-proj-abc"],
        "user_name": ["jsmith", "alice.chen"],
    }


def test_entity_label_examples_empty_dict_raises() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        AnonymizerConfig(detect={"entity_label_examples": {}}, replace=Redact())


def test_entity_label_examples_empty_example_list_raises() -> None:
    with pytest.raises(ValidationError, match="at least one non-empty example"):
        AnonymizerConfig(
            detect={"entity_labels": ["api_key"], "entity_label_examples": {"api_key": []}},
            replace=Redact(),
        )


def test_entity_label_examples_whitespace_only_examples_raises() -> None:
    with pytest.raises(ValidationError, match="at least one non-empty example"):
        AnonymizerConfig(
            detect={"entity_labels": ["api_key"], "entity_label_examples": {"api_key": ["  ", ""]}},
            replace=Redact(),
        )


def test_entity_label_examples_key_not_in_explicit_entity_labels_raises() -> None:
    with pytest.raises(ValidationError, match="not present in entity_labels"):
        AnonymizerConfig(
            detect={
                "entity_labels": ["api_key"],
                "entity_label_examples": {"user_name": ["alice"]},
            },
            replace=Redact(),
        )


def test_entity_label_examples_subset_of_explicit_entity_labels_is_valid() -> None:
    """Not every entity_labels entry needs an example — only example keys must be a subset."""
    config = AnonymizerConfig(
        detect={
            "entity_labels": ["api_key", "user_name", "email"],
            "entity_label_examples": {"api_key": ["sk-proj-abc"], "user_name": ["alice"]},
        },
        replace=Redact(),
    )
    assert config.detect.entity_label_examples is not None
    assert set(config.detect.entity_label_examples) == {"api_key", "user_name"}


def test_entity_label_examples_new_label_under_default_entity_labels_warns_not_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A label outside DEFAULT_ENTITY_LABELS is accepted with a warning, not a hard error —

    the config is still valid, but the example has no effect on detection until entity_labels
    is set explicitly to include that label.
    """
    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        config = AnonymizerConfig(
            detect={"entity_label_examples": {"vendor_token": ["vt_live_abc123"]}},
            replace=Redact(),
        )
    assert config.detect.entity_label_examples == {"vendor_token": ["vt_live_abc123"]}
    assert "not in the default detection set" in caplog.text
    assert "vendor_token" in caplog.text


def test_entity_label_examples_label_in_default_entity_labels_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert "email" in DEFAULT_ENTITY_LABELS
    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        AnonymizerConfig(
            detect={"entity_label_examples": {"email": ["alice@example.com"]}},
            replace=Redact(),
        )
    assert "not in the default detection set" not in caplog.text
