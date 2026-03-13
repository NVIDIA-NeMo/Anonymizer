# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Rewrite
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
    assert set(config.detect.entity_labels) == {"first_name", "email"}


def test_entity_labels_strips_whitespace() -> None:
    config = AnonymizerConfig(detect={"entity_labels": ["  first_name ", "email"]}, replace=Redact())
    assert "first_name" in config.detect.entity_labels
    assert "email" in config.detect.entity_labels


def test_entity_labels_deduplicates() -> None:
    config = AnonymizerConfig(detect={"entity_labels": ["email", "email"]}, replace=Redact())
    assert config.detect.entity_labels == ["email"]
