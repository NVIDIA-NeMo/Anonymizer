# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest
import tyro

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute

# ---------------------------------------------------------------------------
# Shared mirror dataclasses used to test arg parsing independently of the CLI
# wiring in main.py.  These mirror what the run/preview subcommand functions
# accept, so the same tyro parsing logic is exercised.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _RunArgs:
    config: AnonymizerConfig
    data: AnonymizerInput
    output: str | None = None


@dataclasses.dataclass
class _PreviewArgs:
    config: AnonymizerConfig
    data: AnonymizerInput
    num_records: int = 10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    """A minimal CSV with a 'text' column that passes AnonymizerInput validation."""
    path = tmp_path / "data.csv"
    pd.DataFrame({"text": ["Alice works at Acme Corp"]}).to_csv(path, index=False)
    return path


@pytest.fixture
def body_csv_file(tmp_path: Path) -> Path:
    """A minimal CSV with a 'body' column for text-column override tests."""
    path = tmp_path / "data_body.csv"
    pd.DataFrame({"body": ["Hello world"]}).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_redact_defaults(csv_file: Path) -> None:
    """Redact with all defaults produces a valid AnonymizerConfig."""
    result = tyro.cli(
        _RunArgs,
        args=[
            "--data.source",
            str(csv_file),
            "config.replace:redact",
        ],
    )
    assert isinstance(result.config.replace, Redact)
    assert result.config.replace.format_template == "[REDACTED_{label}]"
    assert result.config.replace.normalize_label is True
    assert result.config.rewrite is None
    assert result.data.source == str(csv_file)


def test_parse_hash_custom_params(csv_file: Path) -> None:
    """Hash with algorithm=md5 and digest_length=8."""
    result = tyro.cli(
        _RunArgs,
        args=[
            "--data.source",
            str(csv_file),
            "config.replace:hash",
            "--config.replace.algorithm",
            "md5",
            "--config.replace.digest-length",
            "8",
        ],
    )
    assert isinstance(result.config.replace, Hash)
    assert result.config.replace.algorithm == "md5"
    assert result.config.replace.digest_length == 8


def test_parse_annotate_template(csv_file: Path) -> None:
    """Annotate with a custom format_template."""
    result = tyro.cli(
        _RunArgs,
        args=[
            "--data.source",
            str(csv_file),
            "config.replace:annotate",
            "--config.replace.format-template",
            "[{label}: {text}]",
        ],
    )
    assert isinstance(result.config.replace, Annotate)
    assert result.config.replace.format_template == "[{label}: {text}]"


def test_parse_substitute(csv_file: Path) -> None:
    """Substitute strategy is parsed without extra flags."""
    result = tyro.cli(
        _RunArgs,
        args=[
            "--data.source",
            str(csv_file),
            "config.replace:substitute",
        ],
    )
    assert isinstance(result.config.replace, Substitute)


def test_parse_text_column_override(body_csv_file: Path) -> None:
    """--data.text-column overrides the default 'text' column name."""
    result = tyro.cli(
        _RunArgs,
        args=[
            "--data.source",
            str(body_csv_file),
            "--data.text-column",
            "body",
            "config.replace:redact",
        ],
    )
    assert result.data.text_column == "body"


def test_parse_entity_labels(csv_file: Path) -> None:
    """list[str] | None entity_labels are parsed from space-separated tokens."""
    result = tyro.cli(
        _RunArgs,
        args=[
            "--data.source",
            str(csv_file),
            "--config.detect.entity-labels",
            "person",
            "org",
            "config.replace:redact",
        ],
    )
    labels = result.config.detect.entity_labels
    assert labels is not None
    assert "person" in labels
    assert "org" in labels


def test_parse_threshold(csv_file: Path) -> None:
    """gliner_threshold=0.7 is parsed as a float."""
    result = tyro.cli(
        _RunArgs,
        args=[
            "--data.source",
            str(csv_file),
            "--config.detect.gliner-threshold",
            "0.7",
            "config.replace:redact",
        ],
    )
    assert abs(result.config.detect.gliner_threshold - 0.7) < 1e-9


def test_parse_preview_num_records(csv_file: Path) -> None:
    """--num-records is parsed as an integer for the preview command."""
    result = tyro.cli(
        _PreviewArgs,
        args=[
            "--data.source",
            str(csv_file),
            "--num-records",
            "5",
            "config.replace:redact",
        ],
    )
    assert result.num_records == 5


def test_parse_data_summary(csv_file: Path) -> None:
    """--data.data-summary sets the data_summary field on AnonymizerInput."""
    result = tyro.cli(
        _RunArgs,
        args=[
            "--data.source",
            str(csv_file),
            "--data.data-summary",
            "customer support tickets",
            "config.replace:redact",
        ],
    )
    assert result.data.data_summary == "customer support tickets"
