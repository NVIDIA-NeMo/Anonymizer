# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest.mock as mock
from pathlib import Path

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Detect, Rewrite
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.config.rewrite import DEFAULT_PRESERVE_TEXT, DEFAULT_PROTECT_TEXT, RiskTolerance
from anonymizer.interface.cli.main import CliOpts, _build_replace_strategy, _build_rewrite_config, app
from anonymizer.interface.errors import InvalidConfigError

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
# Replace strategy builder tests
# ---------------------------------------------------------------------------


def test_parse_redact_defaults() -> None:
    """Redact with all defaults produces a valid strategy and config."""
    strategy = _build_replace_strategy(CliOpts(replace="redact"))
    assert isinstance(strategy, Redact)
    assert strategy.format_template == "[REDACTED_{label}]"
    assert strategy.normalize_label is True
    config = AnonymizerConfig(replace=strategy)
    assert config.rewrite is None


def test_parse_hash_custom_params() -> None:
    """Hash with algorithm=md5 and digest_length=8."""
    strategy = _build_replace_strategy(CliOpts(replace="hash", algorithm="md5", digest_length=8))
    assert isinstance(strategy, Hash)
    assert strategy.algorithm == "md5"
    assert strategy.digest_length == 8


def test_parse_annotate_template() -> None:
    """Annotate with a custom format_template."""
    strategy = _build_replace_strategy(CliOpts(replace="annotate", format_template="[{label}: {text}]"))
    assert isinstance(strategy, Annotate)
    assert strategy.format_template == "[{label}: {text}]"


def test_parse_substitute() -> None:
    """Substitute strategy is parsed without extra flags."""
    strategy = _build_replace_strategy(CliOpts(replace="substitute"))
    assert isinstance(strategy, Substitute)


# ---------------------------------------------------------------------------
# Pydantic model integration tests (cyclopts passes these directly)
# ---------------------------------------------------------------------------


def test_parse_text_column_override(body_csv_file: Path) -> None:
    """AnonymizerInput accepts text_column override."""
    data = AnonymizerInput(source=str(body_csv_file), text_column="body")
    assert data.text_column == "body"


def test_parse_entity_labels() -> None:
    """Detect model forwards entity_labels."""
    detect = Detect(entity_labels=["person", "org"])
    assert detect.entity_labels is not None
    assert "person" in detect.entity_labels
    assert "org" in detect.entity_labels


def test_parse_threshold() -> None:
    """gliner_threshold=0.7 is stored as a float."""
    detect = Detect(gliner_threshold=0.7)
    assert abs(detect.gliner_threshold - 0.7) < 1e-9


def test_parse_preview_num_records(csv_file: Path) -> None:
    """--num-records is passed as an integer to the preview command."""
    with mock.patch("anonymizer.interface.cli.main.Anonymizer") as mock_cls:
        mock_result = mock.MagicMock()
        mock_result.dataframe = pd.DataFrame()
        mock_result.failed_records = []
        mock_cls.return_value.preview.return_value = mock_result
        with pytest.raises(SystemExit) as exc:
            app(["preview", "--source", str(csv_file), "--replace", "redact", "--num-records", "5"])
        assert exc.value.code == 0

    assert mock_cls.return_value.preview.call_args.kwargs["num_records"] == 5


def test_parse_data_summary(csv_file: Path) -> None:
    """AnonymizerInput accepts data_summary."""
    data = AnonymizerInput(source=str(csv_file), data_summary="customer support tickets")
    assert data.data_summary == "customer support tickets"


# ---------------------------------------------------------------------------
# Rewrite config builder tests
# ---------------------------------------------------------------------------


def test_build_rewrite_defaults() -> None:
    """Rewrite with all defaults produces valid config with auto-populated privacy goal."""
    opts = CliOpts(rewrite=True)
    rewrite = _build_rewrite_config(opts)
    assert isinstance(rewrite, Rewrite)
    assert rewrite.privacy_goal is not None
    assert rewrite.privacy_goal.protect == DEFAULT_PROTECT_TEXT
    assert rewrite.privacy_goal.preserve == DEFAULT_PRESERVE_TEXT
    config = AnonymizerConfig(rewrite=rewrite)
    assert config.replace is None


def test_build_rewrite_custom_goals() -> None:
    """Custom protect/preserve are forwarded to PrivacyGoal."""
    opts = CliOpts(
        rewrite=True,
        protect="SSNs, addresses, and medical record numbers",
        preserve="Clinical meaning and treatment context",
    )
    rewrite = _build_rewrite_config(opts)
    assert rewrite.privacy_goal is not None
    assert "SSNs" in rewrite.privacy_goal.protect
    assert "Clinical" in rewrite.privacy_goal.preserve


def test_build_rewrite_partial_goal_uses_defaults() -> None:
    """Supplying only --protect fills --preserve with the default."""
    opts = CliOpts(rewrite=True, protect="Direct identifiers and quasi-identifiers")
    rewrite = _build_rewrite_config(opts)
    assert rewrite.privacy_goal is not None
    assert rewrite.privacy_goal.preserve == DEFAULT_PRESERVE_TEXT


def test_build_rewrite_risk_tolerance() -> None:
    """--risk-tolerance strict is forwarded to EvaluationCriteria."""
    opts = CliOpts(rewrite=True, risk_tolerance="strict")
    rewrite = _build_rewrite_config(opts)
    assert rewrite.evaluation.risk_tolerance == RiskTolerance.strict


def test_build_rewrite_max_repair_iterations() -> None:
    """--max-repair-iterations is forwarded."""
    opts = CliOpts(rewrite=True, max_repair_iterations=5)
    rewrite = _build_rewrite_config(opts)
    assert rewrite.evaluation.max_repair_iterations == 5


def test_build_rewrite_instructions() -> None:
    """--instructions is forwarded to Rewrite config."""
    opts = CliOpts(rewrite=True, instructions="Keep dates approximate")
    rewrite = _build_rewrite_config(opts)
    assert rewrite.instructions == "Keep dates approximate"


# ---------------------------------------------------------------------------
# Rewrite CLI integration tests (via app)
# ---------------------------------------------------------------------------


def test_preview_rewrite_via_app(csv_file: Path) -> None:
    """preview --rewrite routes through the rewrite config path."""
    with mock.patch("anonymizer.interface.cli.main.Anonymizer") as mock_cls:
        mock_result = mock.MagicMock()
        mock_result.dataframe = pd.DataFrame()
        mock_result.failed_records = []
        mock_cls.return_value.preview.return_value = mock_result
        with pytest.raises(SystemExit) as exc:
            app(["preview", "--source", str(csv_file), "--rewrite", "--num-records", "3"])
        assert exc.value.code == 0

    call_kwargs = mock_cls.return_value.preview.call_args.kwargs
    assert call_kwargs["config"].rewrite is not None
    assert call_kwargs["config"].replace is None
    assert call_kwargs["num_records"] == 3


def test_no_mode_exits_nonzero(csv_file: Path) -> None:
    """Omitting both --replace and --rewrite causes an error exit."""
    with mock.patch("anonymizer.interface.cli.main.Anonymizer") as mock_cls:
        mock_cls.return_value.run.return_value = mock.MagicMock()
        with pytest.raises(SystemExit) as exc:
            app(["run", "--source", str(csv_file)])
        assert exc.value.code != 0


def test_both_modes_raises() -> None:
    """Specifying both --replace and --rewrite raises at construction."""
    with pytest.raises(InvalidConfigError, match="Cannot use both"):
        CliOpts(replace="redact", rewrite=True)


def test_replace_mode_warns_on_rewrite_flags(caplog: pytest.LogCaptureFixture) -> None:
    """Rewrite-only flags in replace mode produce a warning."""
    with caplog.at_level("WARNING", logger="anonymizer.cli"):
        CliOpts(replace="redact", protect="SSNs, addresses, and medical records")
    assert "--protect" in caplog.text


def test_rewrite_mode_warns_on_replace_flags(caplog: pytest.LogCaptureFixture) -> None:
    """Replace-only flags in rewrite mode produce a warning."""
    with caplog.at_level("WARNING", logger="anonymizer.cli"):
        CliOpts(rewrite=True, format_template="[{label}]")
    assert "--format-template" in caplog.text
