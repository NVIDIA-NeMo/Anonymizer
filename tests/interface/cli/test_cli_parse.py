# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generated CLI: one subcommand per mode, flags named after the config fields."""

from __future__ import annotations

import unittest.mock as mock
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, Detect
from anonymizer.config.regex import BuiltinRegex, RegexRule
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.config.rewrite import DEFAULT_PRESERVE_TEXT, PrivacyGoal, RiskTolerance
from anonymizer.interface.cli.main import MODES, app


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "data.csv"
    pd.DataFrame({"text": ["Alice works at Acme Corp"]}).to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def _hash_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANONYMIZER_HASH_KEY", "s3cr3t")


def _validate(args: list[str]) -> tuple[int, AnonymizerConfig | None, Any]:
    """Run ``anonymizer validate`` with a mocked Anonymizer; return the exit code, the config and the mock."""
    with mock.patch("anonymizer.interface.cli.main.Anonymizer") as mock_cls:
        with pytest.raises(SystemExit) as exc:
            app(["validate", *args])
    call = mock_cls.return_value.validate_config.call_args
    code = exc.value.code if isinstance(exc.value.code, int) else 1
    return code, (call.args[0] if call else None), mock_cls


def test_modes_are_the_replace_strategies_and_rewrite() -> None:
    assert set(MODES) == {"redact", "hash", "annotate", "substitute", "rewrite"}


@pytest.mark.parametrize(
    ("mode", "cls"), [("redact", Redact), ("hash", Hash), ("annotate", Annotate), ("substitute", Substitute)]
)
def test_each_replace_subcommand_builds_its_strategy_with_model_defaults(
    csv_file: Path, mode: str, cls: type[Any]
) -> None:
    code, config, _ = _validate([mode, "--source", str(csv_file)])
    assert code == 0 and config is not None
    assert isinstance(config.replace, cls)
    assert config.rewrite is None
    assert config.detect == Detect()


def test_hash_reads_its_key_from_the_environment(csv_file: Path) -> None:
    code, config, _ = _validate(["hash", "--source", str(csv_file), "--digest-length", "8"])
    assert code == 0 and config is not None and isinstance(config.replace, Hash)
    assert config.replace.key is not None and config.replace.key.get_secret_value() == "s3cr3t"
    assert config.replace.digest_length == 8


def test_hash_without_a_key_names_the_variable(
    csv_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANONYMIZER_HASH_KEY")
    code, config, _ = _validate(["hash", "--source", str(csv_file)])
    assert code == 1 and config is None
    assert "ANONYMIZER_HASH_KEY" in capsys.readouterr().err


def test_strategy_flags_reach_the_strategy(csv_file: Path) -> None:
    code, config, _ = _validate(
        ["substitute", "--source", str(csv_file), "--instructions", "Keep it short", "--consistency", "run"]
    )
    assert code == 0 and config is not None and isinstance(config.replace, Substitute)
    assert (config.replace.instructions, config.replace.consistency) == ("Keep it short", "run")
    code, config, _ = _validate(["annotate", "--source", str(csv_file), "--format-template", "[{label}: {text}]"])
    assert config is not None and isinstance(config.replace, Annotate)
    assert config.replace.format_template == "[{label}: {text}]"
    code, config, _ = _validate(["redact", "--source", str(csv_file), "--no-normalize-label"])
    assert config is not None and isinstance(config.replace, Redact)
    assert config.replace.normalize_label is False


def test_rewrite_flags_reach_rewrite(csv_file: Path) -> None:
    code, config, _ = _validate(
        [
            "rewrite", "--source", str(csv_file),
            "--protect", "SSNs, addresses and medical record numbers",
            "--risk-tolerance", "minimal", "--max-repair-iterations", "5",
            "--use-combined-graph", "--strict-entity-protection",
        ]
    )  # fmt: skip
    assert code == 0 and config is not None and config.rewrite is not None
    assert config.replace is None
    assert config.rewrite.privacy_goal == PrivacyGoal(
        protect="SSNs, addresses and medical record numbers", preserve=DEFAULT_PRESERVE_TEXT
    )
    assert config.rewrite.risk_tolerance == RiskTolerance.minimal
    assert config.rewrite.max_repair_iterations == 5
    assert config.rewrite.use_combined_graph and config.rewrite.strict_entity_protection


def test_rewrite_defaults(csv_file: Path) -> None:
    code, config, _ = _validate(["rewrite", "--source", str(csv_file)])
    assert code == 0 and config is not None
    assert config.rewrite == AnonymizerConfig(rewrite={}).rewrite


def test_detect_flags_reach_detect(csv_file: Path) -> None:
    code, config, _ = _validate(
        [
            "redact", "--source", str(csv_file),
            "--gliner-threshold", "0.7",
            "--entity-labels", "first_name", "--entity-labels", "city",
            "--entity-label-examples", '{"first_name": ["Alice", "Bob"]}',
            "--regex-rules", '{"label": "city", "pattern": "C-[0-9]+"}',
            "--regex-rules", '{"label": "email", "enabled": false}',
            "--regex-limits.max-matches-per-rule", "5",
        ]
    )  # fmt: skip
    assert code == 0 and config is not None
    detect = config.detect
    assert detect.gliner_threshold == pytest.approx(0.7)
    assert sorted(detect.entity_labels or []) == ["city", "first_name"]
    assert detect.entity_label_examples == {"first_name": ["Alice", "Bob"]}
    custom, builtin = detect.regex_rules
    assert isinstance(custom, RegexRule) and isinstance(builtin, BuiltinRegex) and not builtin.enabled
    assert detect.regex_limits.max_matches_per_rule == 5
    # Only the given flags are set, so explicit-versus-default checks downstream still hold.
    assert "builtin_regexes" not in detect.model_fields_set


def test_run_options_reach_the_anonymizer(csv_file: Path) -> None:
    code, config, mock_cls = _validate(
        ["redact", "--source", str(csv_file), "--artifact-path", "/tmp/a", "--no-emit-telemetry"]
    )
    assert code == 0 and config is not None
    assert config.emit_telemetry is False
    assert mock_cls.call_args.kwargs["artifact_path"] == "/tmp/a"


def test_preview_passes_num_records(csv_file: Path) -> None:
    with mock.patch("anonymizer.interface.cli.main.Anonymizer") as mock_cls:
        mock_cls.return_value.preview.return_value.dataframe = pd.DataFrame()
        with pytest.raises(SystemExit) as exc:
            app(["preview", "rewrite", "--source", str(csv_file), "--num-records", "3"])
    assert exc.value.code == 0
    kwargs = mock_cls.return_value.preview.call_args.kwargs
    assert kwargs["num_records"] == 3
    assert kwargs["config"].rewrite is not None


@pytest.mark.parametrize(
    ("mode", "flag"),
    [
        ("redact", ["--risk-tolerance", "high"]),
        ("redact", ["--protect", "SSNs and street addresses"]),
        ("redact", ["--digest-length", "8"]),
        ("redact", ["--instructions", "Keep it short"]),
        ("hash", ["--normalize-label"]),
        ("hash", ["--key", "on-the-command-line"]),
        ("substitute", ["--format-template", "[{label}]"]),
        ("rewrite", ["--format-template", "[{label}]"]),
        ("rewrite", ["--instructions", "Keep dates approximate"]),
        ("rewrite", ["--algorithm", "sha256"]),
        ("redact", ["--kind", "hash"]),
    ],
)
def test_a_flag_of_another_mode_is_an_unknown_option(
    csv_file: Path, capsys: pytest.CaptureFixture[str], mode: str, flag: list[str]
) -> None:
    code, config, _ = _validate([mode, "--source", str(csv_file), *flag])
    assert code != 0 and config is None
    out = capsys.readouterr()
    assert "Unknown option" in out.out + out.err and flag[0] in out.out + out.err


def test_no_mode_is_an_error(csv_file: Path) -> None:
    code, config, _ = _validate(["--source", str(csv_file)])
    assert code != 0 and config is None


def test_no_empty_list_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        app(["run", "redact", "--help"])
    assert "--empty-" not in capsys.readouterr().out
