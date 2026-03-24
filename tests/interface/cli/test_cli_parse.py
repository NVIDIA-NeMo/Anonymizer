# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.interface.cli.main import (
    _build_anonymizer_config,
    _build_anonymizer_input,
)

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
    config = _build_anonymizer_config(replace="redact")
    data = _build_anonymizer_input(source=str(csv_file))
    assert isinstance(config.replace, Redact)
    assert config.replace.format_template == "[REDACTED_{label}]"
    assert config.replace.normalize_label is True
    assert config.rewrite is None
    assert data.source == str(csv_file)


def test_parse_hash_custom_params() -> None:
    """Hash with algorithm=md5 and digest_length=8."""
    config = _build_anonymizer_config(replace="hash", algorithm="md5", digest_length=8)
    assert isinstance(config.replace, Hash)
    assert config.replace.algorithm == "md5"
    assert config.replace.digest_length == 8


def test_parse_annotate_template() -> None:
    """Annotate with a custom format_template."""
    config = _build_anonymizer_config(
        replace="annotate", format_template="[{label}: {text}]"
    )
    assert isinstance(config.replace, Annotate)
    assert config.replace.format_template == "[{label}: {text}]"


def test_parse_substitute() -> None:
    """Substitute strategy is parsed without extra flags."""
    config = _build_anonymizer_config(replace="substitute")
    assert isinstance(config.replace, Substitute)


def test_parse_text_column_override(body_csv_file: Path) -> None:
    """--text-column overrides the default 'text' column name."""
    data = _build_anonymizer_input(source=str(body_csv_file), text_column="body")
    assert data.text_column == "body"


def test_parse_entity_labels() -> None:
    """entity_labels list is forwarded to the Detect config."""
    config = _build_anonymizer_config(replace="redact", entity_labels=["person", "org"])
    labels = config.detect.entity_labels
    assert labels is not None
    assert "person" in labels
    assert "org" in labels


def test_parse_threshold() -> None:
    """gliner_threshold=0.7 is stored as a float."""
    config = _build_anonymizer_config(replace="redact", gliner_threshold=0.7)
    assert abs(config.detect.gliner_threshold - 0.7) < 1e-9


def test_parse_preview_num_records(csv_file: Path) -> None:
    """--num-records is passed as an integer to the preview command."""
    import unittest.mock as mock

    from anonymizer.interface.cli.main import app

    with mock.patch("anonymizer.interface.cli.main.Anonymizer") as mock_cls:
        mock_result = mock.MagicMock()
        mock_result.dataframe = []
        mock_result.failed_records = []
        mock_cls.return_value.preview.return_value = mock_result
        with pytest.raises(SystemExit) as exc:
            app(
                [
                    "preview",
                    "--source",
                    str(csv_file),
                    "--replace",
                    "redact",
                    "--num-records",
                    "5",
                ]
            )
        assert exc.value.code == 0

    assert mock_cls.return_value.preview.call_args.kwargs["num_records"] == 5


def test_parse_data_summary(csv_file: Path) -> None:
    """--data-summary sets the data_summary field on AnonymizerInput."""
    data = _build_anonymizer_input(
        source=str(csv_file), data_summary="customer support tickets"
    )
    assert data.data_summary == "customer support tickets"
