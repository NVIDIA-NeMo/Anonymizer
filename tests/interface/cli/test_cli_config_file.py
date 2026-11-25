# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``--config FILE``: an AnonymizerConfig plus an optional input section, with the given flags laid over it."""

from __future__ import annotations

import json
import unittest.mock as mock
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from anonymizer.config.replace_strategies import Hash
from anonymizer.config.rewrite import DEFAULT_PRESERVE_TEXT, RiskTolerance
from anonymizer.interface.cli.app import app


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "data.csv"
    pd.DataFrame({"text": ["Alice works at Acme Corp"], "body": ["Bob"]}).to_csv(path, index=False)
    return path


def _write(tmp_path: Path, data: dict[str, Any], suffix: str = ".yaml") -> Path:
    path = tmp_path / f"config{suffix}"
    path.write_text(json.dumps(data) if suffix == ".json" else yaml.safe_dump(data), encoding="utf-8")
    return path


def _call(verb: str, args: list[str]) -> tuple[int, Any]:
    """Run ``anonymizer <verb> ...`` with a mocked Anonymizer; return the exit code and the mock."""
    with mock.patch("anonymizer.interface.cli.commands.Anonymizer") as mock_cls:
        mock_cls.return_value.preview.return_value.dataframe = pd.DataFrame()
        with pytest.raises(SystemExit) as exc:
            app([verb, *args])
    return (exc.value.code if isinstance(exc.value.code, int) else 1), mock_cls


def _validated(args: list[str]) -> tuple[int, Any]:
    code, mock_cls = _call("validate", args)
    call = mock_cls.return_value.validate_config.call_args
    return code, (call.args[0] if call else None)


def _error(capsys: pytest.CaptureFixture[str]) -> str:
    lines = [line for line in capsys.readouterr().err.splitlines() if line and not line.startswith("[")]
    assert len(lines) == 1, lines
    return lines[0]


_REWRITE = {
    "detect": {"gliner_threshold": 0.5, "regex_limits": {"max_matches_per_rule": 7}},
    "rewrite": {"privacy_goal": {"protect": "Names, addresses and phone numbers"}, "risk_tolerance": "high"},
}


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_a_file_alone_builds_the_config(csv_file: Path, tmp_path: Path, suffix: str) -> None:
    config_path = _write(tmp_path, {**_REWRITE, "input": {"source": str(csv_file)}}, suffix)
    code, config = _validated(["rewrite", "--config", str(config_path)])
    assert code == 0 and config is not None and config.rewrite is not None
    assert config.detect.gliner_threshold == pytest.approx(0.5)
    assert config.detect.regex_limits.max_matches_per_rule == 7
    assert config.rewrite.privacy_goal.protect == "Names, addresses and phone numbers"
    assert config.rewrite.risk_tolerance == RiskTolerance.high


def test_given_flags_replace_file_values_field_by_field(csv_file: Path, tmp_path: Path) -> None:
    config_path = _write(tmp_path, _REWRITE)
    code, config = _validated(
        [
            "rewrite", "--config", str(config_path), "--source", str(csv_file),
            "--gliner-threshold", "0.7", "--preserve", "Clinical meaning and context",
            "--regex-limits.min-timeout-seconds", "0.2",
        ]
    )  # fmt: skip
    assert code == 0 and config is not None and config.rewrite is not None
    assert config.detect.gliner_threshold == pytest.approx(0.7)
    assert config.detect.regex_limits.max_matches_per_rule == 7  # from the file
    assert config.detect.regex_limits.min_timeout_seconds == pytest.approx(0.2)
    assert config.rewrite.privacy_goal.protect == "Names, addresses and phone numbers"  # from the file
    assert config.rewrite.privacy_goal.preserve == "Clinical meaning and context"


def test_a_flag_left_at_its_default_does_not_replace_the_file(csv_file: Path, tmp_path: Path) -> None:
    """--risk-tolerance defaults to low and --gliner-threshold to 0.3; neither was given, so the file's values hold."""
    config_path = _write(tmp_path, _REWRITE)
    code, config = _validated(
        ["rewrite", "--config", str(config_path), "--source", str(csv_file), "--max-repair-iterations", "1"]
    )
    assert code == 0 and config is not None and config.rewrite is not None
    assert config.rewrite.risk_tolerance == RiskTolerance.high
    assert config.detect.gliner_threshold == pytest.approx(0.5)
    assert config.rewrite.privacy_goal.preserve == DEFAULT_PRESERVE_TEXT
    assert config.rewrite.max_repair_iterations == 1


@pytest.mark.parametrize(
    ("file_value", "flag", "expected"),
    [
        (False, [], False),
        (True, ["--no-emit-telemetry"], False),
        (None, ["--no-emit-telemetry"], False),
        (False, ["--emit-telemetry"], False),
        (True, [], True),
        (None, [], True),
    ],
)
def test_a_telemetry_opt_out_from_either_side_wins(
    csv_file: Path, tmp_path: Path, file_value: bool | None, flag: list[str], expected: bool
) -> None:
    data: dict[str, Any] = {"replace": {"kind": "redact"}}
    if file_value is not None:
        data["emit_telemetry"] = file_value
    config_path = _write(tmp_path, data)
    code, config = _validated(["redact", "--config", str(config_path), "--source", str(csv_file), *flag])
    assert code == 0 and config is not None
    assert config.emit_telemetry is expected


@pytest.mark.parametrize(
    ("mode", "data", "message"),
    [
        ("redact", _REWRITE, "the file sets rewrite, but this command is redact"),
        ("rewrite", {"replace": {"kind": "redact"}}, "the file sets replace, but this command is rewrite"),
        ("redact", {"replace": {"kind": "annotate"}}, "replace.kind is annotate, but this command is redact"),
    ],
)
def test_a_mode_mismatch_is_refused(
    csv_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], mode: str, data: dict[str, Any], message: str
) -> None:
    config_path = _write(tmp_path, data)
    code, config = _validated([mode, "--config", str(config_path), "--source", str(csv_file)])
    assert code == 2 and config is None
    assert _error(capsys) == f"Error: {config_path}: {message}; use " + (
        "the annotate subcommand" if "kind" in message else "the matching subcommand"
    )


@pytest.mark.parametrize(
    ("data", "where"),
    [
        ({"detect": {"gliner_threshold": 5}}, "detect.gliner_threshold"),
        ({"rewrite": {"privacy_goal": {"protect": "short"}}}, "rewrite.privacy_goal.protect"),
        ({"input": {"text_column": ""}}, "input.text_column"),
    ],
)
def test_a_bad_value_in_the_file_names_the_file_and_field(
    csv_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], data: dict[str, Any], where: str
) -> None:
    config_path = _write(tmp_path, data)
    code, config = _validated(["rewrite", "--config", str(config_path), "--source", str(csv_file)])
    assert code == 2 and config is None
    line = _error(capsys)
    assert line.startswith(f"Error: {config_path}: {where}: ")
    assert "input_value" not in line


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("detect: [unclosed", "not valid YAML or JSON at line"),
        ("- a list", "expected a mapping"),
        ("detects: {}", "unknown section(s) detects"),
    ],
)
def test_a_malformed_file_is_one_line(
    csv_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], text: str, message: str
) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(text, encoding="utf-8")
    code, _ = _validated(["redact", "--config", str(config_path), "--source", str(csv_file)])
    assert code == 2
    line = _error(capsys)
    assert line.startswith(f"Error: {config_path}: ") and message in line


def test_a_missing_file_is_one_line(tmp_path: Path, csv_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = _validated(["redact", "--config", str(tmp_path / "nope.yaml"), "--source", str(csv_file)])
    assert code == 2
    assert _error(capsys).startswith(f"Error: {tmp_path / 'nope.yaml'}: cannot read the config file")


def test_a_hash_key_in_the_file_is_refused_without_echoing_it(
    csv_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANONYMIZER_HASH_KEY", "from-the-environment")
    secret = "file-held-secret-value"
    config_path = _write(tmp_path, {"replace": {"kind": "hash", "key": secret}})
    code, config = _validated(["hash", "--config", str(config_path), "--source", str(csv_file)])
    assert code == 2 and config is None
    out = capsys.readouterr()
    lines = [line for line in out.err.splitlines() if line and not line.startswith("[")]
    assert lines == [
        f"Error: {config_path}: replace.key is not read from a file; set ANONYMIZER_HASH_KEY instead and remove the key from the file"
    ]
    assert secret not in out.out + out.err


def test_a_hash_file_takes_the_key_from_the_environment(
    csv_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANONYMIZER_HASH_KEY", "from-the-environment")
    config_path = _write(tmp_path, {"replace": {"kind": "hash", "digest_length": 8}})
    code, config = _validated(["hash", "--config", str(config_path), "--source", str(csv_file)])
    assert code == 0 and config is not None and isinstance(config.replace, Hash)
    assert config.replace.digest_length == 8
    assert config.replace.key is not None and config.replace.key.get_secret_value() == "from-the-environment"


def test_the_input_section_with_and_without_flags(csv_file: Path, tmp_path: Path) -> None:
    config_path = _write(tmp_path, {"input": {"source": str(csv_file), "text_column": "body", "data_summary": "notes"}})
    code, mock_cls = _call("preview", ["redact", "--config", str(config_path)])
    assert code == 0
    data = mock_cls.return_value.preview.call_args.kwargs["data"]
    assert (str(data.source), data.text_column, data.data_summary) == (str(csv_file), "body", "notes")

    code, mock_cls = _call("preview", ["redact", "--config", str(config_path), "--text-column", "text"])
    assert code == 0
    data = mock_cls.return_value.preview.call_args.kwargs["data"]
    assert (data.text_column, data.data_summary) == ("text", "notes")


def test_without_a_source_anywhere_the_error_names_both_places(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write(tmp_path, {"replace": {"kind": "redact"}})
    code, _ = _validated(["redact", "--config", str(config_path)])
    assert code == 2
    assert _error(capsys) == f"Error: --source is required (or input.source in {config_path})"
    code, _ = _validated(["redact"])
    assert code == 2
    assert _error(capsys) == "Error: --source is required"


@pytest.mark.parametrize(
    ("mode", "data", "where"),
    [
        ("redact", {"detect": {"glinr_threshold": 0.5}}, "detect.glinr_threshold"),
        ("redact", {"detect": {"regex_limits": {"max_matches": 5}}}, "detect.regex_limits.max_matches"),
        ("redact", {"replace": {"formt": "[{label}]"}}, "replace.formt"),
        ("redact", {"replace": {"digest_length": 8}}, "replace.digest_length"),
        ("rewrite", {"rewrite": {"risk_tolerence": "high"}}, "rewrite.risk_tolerence"),
        ("rewrite", {"rewrite": {"privacy_goal": {"protects": "x"}}}, "rewrite.privacy_goal.protects"),
        ("redact", {"input": {"bogus": 1}}, "input.bogus"),
    ],
)
def test_an_unknown_key_in_the_file_is_refused(
    csv_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], mode: str, data: dict[str, Any], where: str
) -> None:
    config_path = _write(tmp_path, data)
    code, config = _validated([mode, "--config", str(config_path), "--source", str(csv_file)])
    assert code == 2 and config is None
    assert _error(capsys) == f"Error: {config_path}: {where}: unknown field"


@pytest.mark.parametrize("spelling", ["Key", "KEY"])
def test_a_case_variant_of_the_key_is_the_key(
    csv_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], spelling: str
) -> None:
    monkeypatch.setenv("ANONYMIZER_HASH_KEY", "from-the-environment")
    secret = "file-held-secret-value"
    config_path = _write(tmp_path, {"replace": {"kind": "hash", spelling: secret}})
    code, config = _validated(["hash", "--config", str(config_path), "--source", str(csv_file)])
    assert code == 2 and config is None
    out = capsys.readouterr()
    assert "replace.key is not read from a file; set ANONYMIZER_HASH_KEY" in out.err
    assert secret not in out.out + out.err


def test_an_unknown_kind_is_not_echoed(csv_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = _write(tmp_path, {"replace": {"kind": "sk-SECRET123"}})
    code, config = _validated(["redact", "--config", str(config_path), "--source", str(csv_file)])
    assert code == 2 and config is None
    line = _error(capsys)
    assert line == f"Error: {config_path}: replace.kind is not one of annotate, redact, hash, substitute"
    assert "sk-SECRET123" not in line


def test_a_file_that_is_not_utf8_is_one_line(
    csv_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "latin1.yaml"
    config_path.write_bytes("detect:\n  gliner_threshold: 0.5 # caf\xe9\n".encode("latin-1"))
    code, config = _validated(["redact", "--config", str(config_path), "--source", str(csv_file)])
    assert code == 2 and config is None
    assert _error(capsys) == f"Error: {config_path}: cannot read the config file (not UTF-8 text)"


def test_a_mapping_given_by_a_flag_replaces_the_files_whole_value(csv_file: Path, tmp_path: Path) -> None:
    config_path = _write(tmp_path, {"detect": {"entity_label_examples": {"first_name": ["Ada"], "city": ["Paris"]}}})
    code, config = _validated(
        [
            "redact",
            "--config",
            str(config_path),
            "--source",
            str(csv_file),
            "--entity-label-examples",
            '{"first_name": ["Bob"]}',
        ]
    )
    assert code == 0 and config is not None
    assert config.detect.entity_label_examples == {"first_name": ["Bob"]}


def test_a_list_given_by_a_flag_replaces_the_files_list(csv_file: Path, tmp_path: Path) -> None:
    config_path = _write(tmp_path, {"detect": {"entity_labels": ["first_name", "city"]}})
    code, config = _validated(
        ["redact", "--config", str(config_path), "--source", str(csv_file), "--entity-labels", "email"]
    )
    assert code == 0 and config is not None
    assert config.detect.entity_labels == ["email"]


def test_a_flag_equal_to_its_default_still_replaces_the_file(csv_file: Path, tmp_path: Path) -> None:
    config_path = _write(tmp_path, {"detect": {"gliner_threshold": 0.5}})
    code, config = _validated(
        ["redact", "--config", str(config_path), "--source", str(csv_file), "--gliner-threshold", "0.3"]
    )
    assert code == 0 and config is not None
    assert config.detect.gliner_threshold == pytest.approx(0.3)


def test_a_field_filled_by_a_validator_does_not_replace_the_file(csv_file: Path, tmp_path: Path) -> None:
    """Rewrite fills privacy_goal with PrivacyGoal() after any rewrite flag, which marks the field as set. Its own
    fields are unset, so the file's goal survives."""
    config_path = _write(tmp_path, _REWRITE)
    code, config = _validated(
        ["rewrite", "--config", str(config_path), "--source", str(csv_file), "--risk-tolerance", "low"]
    )
    assert code == 0 and config is not None and config.rewrite is not None
    assert config.rewrite.privacy_goal.protect == "Names, addresses and phone numbers"
    assert config.rewrite.risk_tolerance == RiskTolerance.low


def test_help_shows_the_input_defaults_and_puts_config_with_input(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        app(["run", "redact", "--help"])
    out = capsys.readouterr().out
    input_panel = out.split("╭─ Input and output")[1].split("╰")[0]
    assert "[default: text]" in input_panel and "--config" in input_panel


def test_flag_examples_for_labels_the_file_declares(csv_file: Path, tmp_path: Path) -> None:
    """Greptile's case: the flags alone hold examples for a label only the file declares."""
    config_path = _write(tmp_path, {"detect": {"entity_labels": ["ticket_id"]}})
    code, config = _validated(
        ["redact", "--config", str(config_path), "--source", str(csv_file),
         "--entity-label-examples", '{"ticket_id": ["TKT-123"]}']
    )  # fmt: skip
    assert code == 0 and config is not None
    assert config.detect.entity_labels == ["ticket_id"]
    assert config.detect.entity_label_examples == {"ticket_id": ["TKT-123"]}


def test_flag_exclusions_that_leave_only_the_files_custom_rule(csv_file: Path, tmp_path: Path) -> None:
    """The flags alone exclude every default label; the file's custom rule keeps the merged set non-empty."""
    from anonymizer import DEFAULT_ENTITY_LABELS

    config_path = _write(tmp_path, {"detect": {"regex_rules": [{"label": "ticket_id", "pattern": "TKT-[0-9]+"}]}})
    excluded = [arg for label in DEFAULT_ENTITY_LABELS for arg in ("--excluded-entity-labels", label)]
    code, config = _validated(["redact", "--config", str(config_path), "--source", str(csv_file), *excluded])
    assert code == 0 and config is not None
    assert [rule.label for rule in config.detect.regex_rules] == ["ticket_id"]


def test_flag_rules_that_conflict_only_while_builtins_are_on(csv_file: Path, tmp_path: Path) -> None:
    """Two email rules disagree on detect_additional_matches; the file turns the built-in one off."""
    config_path = _write(tmp_path, {"detect": {"builtin_regexes": False}})
    code, config = _validated(
        ["redact", "--config", str(config_path), "--source", str(csv_file),
         "--regex-rules", '{"label": "email", "detect_additional_matches": false}',
         "--regex-rules", '{"label": "email", "pattern": "[a-z]+@[a-z]+[.]com"}']
    )  # fmt: skip
    assert code == 0 and config is not None
    assert config.detect.builtin_regexes is False


def test_a_merged_cross_field_error_is_still_refused(
    csv_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write(tmp_path, {"detect": {"entity_labels": ["first_name"]}})
    code, config = _validated(
        ["redact", "--config", str(config_path), "--source", str(csv_file),
         "--entity-label-examples", '{"ticket_id": ["TKT-123"]}']
    )  # fmt: skip
    assert code == 2 and config is None
    assert _error(capsys).startswith(
        "Error: detect: entity_label_examples contains labels outside the active label set"
    )


def _sdk_dump(mode: str) -> Any:
    from anonymizer.config.anonymizer_config import AnonymizerConfig, Rewrite
    from anonymizer.config.replace_strategies import Annotate, Redact, Substitute

    strategies = {"redact": Redact(), "annotate": Annotate(), "substitute": Substitute(), "hash": Hash(key="sdk-key")}
    if mode == "rewrite":
        return AnonymizerConfig(rewrite=Rewrite(risk_tolerance=RiskTolerance.high))
    return AnonymizerConfig(replace=strategies[mode], detect={"gliner_threshold": 0.4})


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
@pytest.mark.parametrize("mode", ["redact", "annotate", "substitute", "hash", "rewrite"])
def test_an_sdk_dump_round_trips_through_config(
    csv_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, suffix: str
) -> None:
    """AnonymizerConfig -> model_dump -> file -> --config. The dump writes the unused mode as null and, for hash,
    the key masked; neither is refused."""
    monkeypatch.setenv("ANONYMIZER_HASH_KEY", "from-the-environment")
    sdk = _sdk_dump(mode)
    path = tmp_path / f"dump{suffix}"
    path.write_text(sdk.model_dump_json() if suffix == ".json" else yaml.safe_dump(sdk.model_dump(mode="json")))
    code, config = _validated([mode, "--config", str(path), "--source", str(csv_file)])
    assert code == 0 and config is not None
    assert config.detect == sdk.detect
    if mode == "rewrite":
        assert config.rewrite == sdk.rewrite and config.replace is None
    else:
        assert config.rewrite is None and type(config.replace) is type(sdk.replace)
        assert config.replace.model_dump(exclude={"key"}) == sdk.replace.model_dump(exclude={"key"})
    if mode == "hash":
        assert isinstance(config.replace, Hash) and config.replace.key is not None
        assert config.replace.key.get_secret_value() == "from-the-environment"


def test_a_keyless_hash_dump_takes_the_key_from_the_environment(
    csv_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A result keeps its Hash without the key (key: null); that dump is accepted too."""
    from anonymizer.config.anonymizer_config import AnonymizerConfig

    monkeypatch.setenv("ANONYMIZER_HASH_KEY", "from-the-environment")
    keyless = AnonymizerConfig(replace=Hash(key="sdk-key")).model_copy(
        update={"replace": Hash(key="sdk-key").without_key()}
    )
    path = _write(tmp_path, keyless.model_dump(mode="json"))
    code, config = _validated(["hash", "--config", str(path), "--source", str(csv_file)])
    assert code == 0 and config is not None and isinstance(config.replace, Hash)
    assert config.replace.key is not None and config.replace.key.get_secret_value() == "from-the-environment"


def test_a_non_null_other_mode_is_still_refused(
    csv_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, {"replace": {"kind": "redact"}, "rewrite": {}})
    code, config = _validated(["redact", "--config", str(path), "--source", str(csv_file)])
    assert code == 2 and config is None
    assert (
        _error(capsys)
        == f"Error: {path}: the file sets rewrite, but this command is redact; use the matching subcommand"
    )
