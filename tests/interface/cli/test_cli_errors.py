# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import anonymizer.graph as graph
from anonymizer.engine.io.reader import read_input
from anonymizer.interface.cli.app import app
from anonymizer.interface.cli.errors import _cli_error_handler
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

    with patch("anonymizer.interface.cli.commands.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit) as exc_info:
            app([subcommand, "redact", "--source", str(csv_file)])

    assert exc_info.value.code != 0


def test_bad_output_format_exits_before_pipeline(csv_file: Path) -> None:
    """An unsupported output extension is rejected before the pipeline runs."""
    with patch("anonymizer.interface.cli.commands.Anonymizer") as mock_cls:
        with pytest.raises(SystemExit) as exc_info:
            app(["run", "redact", "--source", str(csv_file), "--output", "/tmp/out.xyz"])
        assert exc_info.value.code != 0
    mock_cls.return_value.run.assert_not_called()


def test_output_same_as_source_exits(csv_file: Path) -> None:
    """Output path identical to source is rejected before the pipeline runs."""
    with patch("anonymizer.interface.cli.commands.Anonymizer") as mock_cls:
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
        _cli_error_handler("redact", fail)(graph.GraphRejection("over_limit", "max_targets"))
    with pytest.raises(SystemExit) as exit_info:
        _cli_error_handler("redact", fail)(ValueError("plain"))
    assert exit_info.value.code == 1


def test_a_hash_key_error_does_not_echo_the_key(
    csv_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The entry point prints parse errors without input values: a rejected key is not echoed."""
    from anonymizer.interface.cli.main import main

    rejected_key = "*" * 17  # a masked placeholder, which Hash refuses
    monkeypatch.setenv("ANONYMIZER_HASH_KEY", rejected_key)
    monkeypatch.setattr("sys.argv", ["anonymizer", "validate", "hash", "--source", str(csv_file)])
    with patch("anonymizer.interface.cli.commands.Anonymizer"), pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
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
    assert exc_info.value.code == 2
    assert "--digest-length" in capsys.readouterr().err


@pytest.mark.parametrize("ids", [["a", "  "], ["a", "a"]])
def test_an_unusable_id_column_is_one_error_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], ids: list[str]
) -> None:
    """InvalidInputError from the reader is reported like other input errors: one line, no traceback."""
    from anonymizer.interface.cli.main import main

    source = tmp_path / "data.csv"
    pd.DataFrame({"doc": ids, "text": ["Alice", "Bob"]}).to_csv(source, index=False)
    argv = ["anonymizer", "run", "redact", "--source", str(source), "--id-column", "doc"]
    monkeypatch.setattr("sys.argv", argv)
    with patch("anonymizer.interface.cli.commands.Anonymizer") as mock_cls:
        mock_cls.return_value.run.side_effect = lambda *, config, data: read_input(data)
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    # Log lines may precede it; the error itself is one line.
    assert [line for line in err.splitlines() if not line.startswith("[")] == [err.splitlines()[-1]]
    assert err.splitlines()[-1].startswith("Error: Input id column 'doc'")
    assert "Traceback" not in err


def _error_lines(err: str) -> list[str]:
    """stderr without log lines (which start with a timestamp in brackets)."""
    return [line for line in err.splitlines() if line and not line.startswith("[")]


@pytest.mark.parametrize(
    ("argv", "code", "expected"),
    [
        (["validate", "redact", "--digest-length", "8"], 2, "Error: Unknown option: --digest-length"),
        (["validate", "rewrite", "--protect", "short"], 2, "Error: --protect: String should have at least 10"),
        (["validate", "redact", "--entity-label-examples", "nope"], 2, "Error: --entity-label-examples: "),
        (["validate", "redact", "--regex-limits.max-matches-per-rule", "0"], 2, "Error: --regex-limits.max-matches"),
    ],
)
def test_parse_errors_are_one_line_naming_the_flag_for_every_caller(
    csv_file: Path, capsys: pytest.CaptureFixture[str], argv: list[str], code: int, expected: str
) -> None:
    """Calling ``app`` directly (as tests and embedders do) prints the same one-line errors as the entry point."""
    with patch("anonymizer.interface.cli.commands.Anonymizer"), pytest.raises(SystemExit) as exc_info:
        app([*argv, "--source", str(csv_file)])
    assert exc_info.value.code == code
    out = capsys.readouterr()
    lines = _error_lines(out.err)
    assert len(lines) == 1 and lines[0].startswith(expected), out.err
    assert "input_value" not in out.err and "╭" not in out.err + out.out


def test_a_missing_hash_key_is_one_line_naming_the_variable(
    csv_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANONYMIZER_HASH_KEY", raising=False)
    with patch("anonymizer.interface.cli.commands.Anonymizer"), pytest.raises(SystemExit) as exc_info:
        app(["validate", "hash", "--source", str(csv_file)])
    assert exc_info.value.code == 2
    lines = _error_lines(capsys.readouterr().err)
    assert lines == [lines[0]] and lines[0].startswith("Error: ANONYMIZER_HASH_KEY is not set")
    assert "errors.pydantic.dev" not in lines[0] and "validation error" not in lines[0]
