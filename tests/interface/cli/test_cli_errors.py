# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import anonymizer.graph as graph
from anonymizer.interface.cli.main import _cli_error_handler, app
from anonymizer.interface.errors import AnonymizerIOError, InvalidConfigError


def test_invalid_source_exits(tmp_path: Path) -> None:
    """A non-existent source path causes SystemExit due to Pydantic validation."""
    with pytest.raises(SystemExit) as exc_info:
        app(
            [
                "run",
                "redact",
                "--source",
                str(tmp_path / "nonexistent.csv"),
            ]
        )
    assert exc_info.value.code != 0


def test_missing_required_args_exits() -> None:
    """Omitting the required --source flag causes SystemExit."""
    with pytest.raises(SystemExit):
        app(["run", "redact"])


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    f = tmp_path / "data.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(f, index=False)
    return f


@pytest.mark.parametrize(
    "subcommand,method",
    [
        ("run", "run"),
        ("preview", "preview"),
        ("validate", "validate_config"),
    ],
)
@pytest.mark.parametrize(
    "exc",
    [
        InvalidConfigError("bad config"),
        AnonymizerIOError("io error"),
        ValueError("bad value"),
    ],
)
def test_error_handler_exits_nonzero(csv_file: Path, exc: Exception, subcommand: str, method: str) -> None:
    """_cli_error_handler converts each wrapped error type to SystemExit(1) for every subcommand."""
    mock_anonymizer = MagicMock()
    getattr(mock_anonymizer, method).side_effect = exc

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit) as exc_info:
            app([subcommand, "redact", "--source", str(csv_file)])

    assert exc_info.value.code != 0


def test_bad_output_format_exits_before_pipeline(csv_file: Path) -> None:
    """An unsupported output extension is rejected before the pipeline runs."""
    with patch("anonymizer.interface.cli.main.Anonymizer") as mock_cls:
        with pytest.raises(SystemExit) as exc_info:
            app(["run", "redact", "--source", str(csv_file), "--output", "/tmp/out.xyz"])
        assert exc_info.value.code != 0
    mock_cls.return_value.run.assert_not_called()


def test_output_same_as_source_exits(csv_file: Path) -> None:
    """Output path identical to source is rejected before the pipeline runs."""
    with patch("anonymizer.interface.cli.main.Anonymizer") as mock_cls:
        with pytest.raises(SystemExit) as exc_info:
            app(["run", "redact", "--source", str(csv_file), "--output", str(csv_file)])
        assert exc_info.value.code != 0
    mock_cls.return_value.run.assert_not_called()


def test_threshold_out_of_range_exits(tmp_path: Path) -> None:
    """gliner_threshold=2.0 violates the le=1.0 constraint → SystemExit."""
    csv_file = tmp_path / "data.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(csv_file, index=False)

    with pytest.raises(SystemExit) as exc_info:
        app(
            [
                "run",
                "redact",
                "--source",
                str(csv_file),
                "--gliner-threshold",
                "2.0",
            ]
        )
    assert exc_info.value.code != 0


def test_error_handler_lets_graph_rejection_propagate() -> None:
    def fail(exc: Exception) -> None:
        raise exc

    with pytest.raises(graph.GraphRejection):
        _cli_error_handler(fail)(graph.GraphRejection("over_limit", "max_targets"))
    with pytest.raises(SystemExit) as exit_info:
        _cli_error_handler(fail)(ValueError("plain"))
    assert exit_info.value.code == 1


def test_a_hash_key_error_does_not_echo_the_key(
    csv_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The entry point prints parse errors without input values: a rejected key is not echoed."""
    from anonymizer.interface.cli.main import main

    rejected_key = "*" * 17  # a masked placeholder, which Hash refuses
    monkeypatch.setenv("ANONYMIZER_HASH_KEY", rejected_key)
    monkeypatch.setattr("sys.argv", ["anonymizer", "validate", "hash", "--source", str(csv_file)])
    with patch("anonymizer.interface.cli.main.Anonymizer"), pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    out = capsys.readouterr()
    assert "masked placeholder" in out.err
    assert rejected_key not in out.out + out.err
    assert "input_value" not in out.out + out.err


def test_an_unknown_option_exits_through_the_entry_point(
    csv_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from anonymizer.interface.cli.main import main

    monkeypatch.setattr(
        "sys.argv", ["anonymizer", "validate", "redact", "--source", str(csv_file), "--digest-length", "8"]
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    assert "--digest-length" in capsys.readouterr().err
