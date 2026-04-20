# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from anonymizer.engine.ndd.model_loader import parse_model_configs
from anonymizer.interface.anonymizer import _resolve_model_providers
from anonymizer.interface.cli.main import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "data.csv"
    pd.DataFrame({"text": ["Alice works at Acme Corp"]}).to_csv(path, index=False)
    return path


@pytest.fixture
def model_configs_yaml(tmp_path: Path) -> Path:
    """Minimal valid model_configs YAML file."""
    path = tmp_path / "models.yaml"
    path.write_text(
        textwrap.dedent("""\
            model_configs:
              - alias: my-model
                model: nvidia/some-model
                provider: nvidia
        """)
    )
    return path


@pytest.fixture
def providers_yaml(tmp_path: Path) -> Path:
    """Minimal valid providers YAML file."""
    path = tmp_path / "providers.yaml"
    path.write_text(
        textwrap.dedent("""\
            providers:
              - name: nvidia
                endpoint: https://example.com/v1
                provider_type: openai
        """)
    )
    return path


# ---------------------------------------------------------------------------
# parse_model_configs: file-path detection
# ---------------------------------------------------------------------------


def test_model_configs_yaml_path_loads_file(model_configs_yaml: Path) -> None:
    """A string ending in .yaml that exists on disk is loaded as a file."""
    result = parse_model_configs(str(model_configs_yaml))
    aliases = {mc.alias for mc in result.model_configs}
    assert "my-model" in aliases


def test_model_configs_yml_extension_loads_file(tmp_path: Path) -> None:
    """A string ending in .yml (not .yaml) is also treated as a file path."""
    path = tmp_path / "models.yml"
    path.write_text(
        textwrap.dedent("""\
            model_configs:
              - alias: yml-model
                model: nvidia/yml-model
                provider: nvidia
        """)
    )
    result = parse_model_configs(str(path))
    aliases = {mc.alias for mc in result.model_configs}
    assert "yml-model" in aliases


def test_model_configs_missing_yaml_file_raises_file_not_found() -> None:
    """A .yaml path that does not exist raises FileNotFoundError with a clear message."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        parse_model_configs("does_not_exist/models_super.yaml")


def test_model_configs_missing_yaml_message_names_the_file() -> None:
    """The FileNotFoundError message includes the bad file path so users can self-diagnose."""
    bad_path = "configs/models_typo.yaml"
    with pytest.raises(FileNotFoundError, match="models_typo.yaml"):
        parse_model_configs(bad_path)


def test_model_configs_inline_yaml_still_works() -> None:
    """Multi-line inline YAML (no file extension) still parses correctly — no regression."""
    inline = textwrap.dedent("""\
        model_configs:
          - alias: inline-model
            model: nvidia/inline
            provider: nvidia
    """)
    result = parse_model_configs(inline)
    aliases = {mc.alias for mc in result.model_configs}
    assert "inline-model" in aliases


def test_model_configs_directory_path_raises_file_not_found(tmp_path: Path) -> None:
    """A directory named *.yaml is rejected with FileNotFoundError, not a cryptic open() error."""
    yaml_dir = tmp_path / "models.yaml"
    yaml_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        parse_model_configs(str(yaml_dir))


def test_model_providers_directory_path_raises_file_not_found(tmp_path: Path) -> None:
    """A directory named *.yaml for providers is rejected with FileNotFoundError."""
    yaml_dir = tmp_path / "providers.yaml"
    yaml_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Providers config file not found"):
        _resolve_model_providers(str(yaml_dir))


def test_model_configs_tilde_path_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """~/models.yaml expands to the user's home directory (not treated as a literal ~ path)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "models.yaml"
    path.write_text(
        textwrap.dedent("""\
            model_configs:
              - alias: tilde-model
                model: nvidia/tilde
                provider: nvidia
        """)
    )
    result = parse_model_configs("~/models.yaml")
    assert "tilde-model" in {mc.alias for mc in result.model_configs}


def test_model_providers_tilde_path_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """~/providers.yaml expands to the user's home directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "providers.yaml"
    path.write_text(
        textwrap.dedent("""\
            providers:
              - name: tilde-provider
                endpoint: https://example.com/v1
                provider_type: openai
        """)
    )
    result = _resolve_model_providers("~/providers.yaml")
    assert result is not None
    assert "tilde-provider" in {p.name for p in result}


def test_model_configs_non_yaml_extension_string_parsed_as_yaml() -> None:
    """A string without a .yaml/.yml extension is treated as inline YAML, not a file path."""
    with pytest.raises(ValueError, match="Expected YAML mapping"):
        parse_model_configs("not_a_file.json")


def test_model_configs_none_uses_defaults() -> None:
    """None falls back to bundled defaults without error."""
    result = parse_model_configs(None)
    assert len(result.model_configs) > 0


def test_model_configs_path_object_still_works(model_configs_yaml: Path) -> None:
    """Passing a Path object (programmatic use) is unchanged — no regression."""
    result = parse_model_configs(model_configs_yaml)
    aliases = {mc.alias for mc in result.model_configs}
    assert "my-model" in aliases


# ---------------------------------------------------------------------------
# _resolve_model_providers: file-path detection
# ---------------------------------------------------------------------------


def test_model_providers_yaml_path_loads_file(providers_yaml: Path) -> None:
    """A string ending in .yaml that exists on disk is loaded as a file."""
    result = _resolve_model_providers(str(providers_yaml))
    assert result is not None
    names = {p.name for p in result}
    assert "nvidia" in names


def test_model_providers_missing_yaml_file_raises_file_not_found() -> None:
    """A .yaml path that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Providers config file not found"):
        _resolve_model_providers("does_not_exist/providers.yaml")


def test_model_providers_missing_yaml_message_names_the_file() -> None:
    """The FileNotFoundError message includes the bad file path."""
    bad_path = "configs/providers_typo.yaml"
    with pytest.raises(FileNotFoundError, match="providers_typo.yaml"):
        _resolve_model_providers(bad_path)


def test_model_providers_none_returns_none() -> None:
    """None returns None — no providers registered."""
    assert _resolve_model_providers(None) is None


def test_model_providers_list_passthrough() -> None:
    """A pre-built list of ModelProvider objects is returned as-is."""
    from data_designer.config.models import ModelProvider

    providers = [ModelProvider(name="my-provider", endpoint="https://example.com/v1")]
    result = _resolve_model_providers(providers)
    assert result is providers


# ---------------------------------------------------------------------------
# CLI integration: _cli_error_handler catches FileNotFoundError
# ---------------------------------------------------------------------------


def test_cli_missing_model_configs_exits_cleanly(csv_file: Path, capsys: pytest.CaptureFixture) -> None:
    """A missing --model-configs file exits with code 1 and prints a useful message — no traceback."""
    with pytest.raises(SystemExit) as exc_info:
        app(
            [
                "run",
                "--source",
                str(csv_file),
                "--replace",
                "redact",
                "--model-configs",
                "nonexistent/models.yaml",
            ]
        )
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "models.yaml" in captured.err
    assert "Traceback" not in captured.err


def test_cli_missing_model_providers_exits_cleanly(csv_file: Path, capsys: pytest.CaptureFixture) -> None:
    """A missing --model-providers file exits with code 1 and prints a useful message."""
    with patch("anonymizer.interface.cli.main.Anonymizer") as mock_cls:
        mock_cls.side_effect = OSError("Providers config file not found: nonexistent/providers.yaml")
        with pytest.raises(SystemExit) as exc_info:
            app(
                [
                    "run",
                    "--source",
                    str(csv_file),
                    "--replace",
                    "redact",
                    "--model-providers",
                    "nonexistent/providers.yaml",
                ]
            )
    assert exc_info.value.code != 0


def test_cli_oserror_exits_cleanly(csv_file: Path, capsys: pytest.CaptureFixture) -> None:
    """Any OSError (e.g. PermissionError) is caught cleanly — no traceback."""
    with patch("anonymizer.interface.cli.main.Anonymizer") as mock_cls:
        mock_cls.side_effect = PermissionError("Permission denied: /secure/models.yaml")
        with pytest.raises(SystemExit) as exc_info:
            app(
                [
                    "run",
                    "--source",
                    str(csv_file),
                    "--replace",
                    "redact",
                    "--model-configs",
                    "/secure/models.yaml",
                ]
            )
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err


def test_cli_model_configs_file_path_accepted(csv_file: Path, model_configs_yaml: Path) -> None:
    """--model-configs accepts a YAML file path; Anonymizer is constructed with the loaded configs."""
    mock_anonymizer = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = pd.DataFrame()
    mock_result.failed_records = []
    mock_anonymizer.run.return_value = mock_result

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer) as mock_cls:
        with pytest.raises(SystemExit) as exc_info:
            app(
                [
                    "run",
                    "--source",
                    str(csv_file),
                    "--replace",
                    "redact",
                    "--model-configs",
                    str(model_configs_yaml),
                ]
            )
    assert exc_info.value.code == 0
    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["model_configs"] == str(model_configs_yaml)


def test_cli_model_providers_file_path_accepted(csv_file: Path, providers_yaml: Path) -> None:
    """--model-providers accepts a YAML file path."""
    mock_anonymizer = MagicMock()
    mock_result = MagicMock()
    mock_result.dataframe = pd.DataFrame()
    mock_result.failed_records = []
    mock_anonymizer.run.return_value = mock_result

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer) as mock_cls:
        with pytest.raises(SystemExit) as exc_info:
            app(
                [
                    "run",
                    "--source",
                    str(csv_file),
                    "--replace",
                    "redact",
                    "--model-providers",
                    str(providers_yaml),
                ]
            )
    assert exc_info.value.code == 0
    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["model_providers"] == str(providers_yaml)


def test_cli_old_confusing_error_message_is_gone(csv_file: Path, capsys: pytest.CaptureFixture) -> None:
    """The old 'Expected YAML mapping, got str' message no longer appears for a missing .yaml file."""
    with pytest.raises(SystemExit):
        app(
            [
                "run",
                "--source",
                str(csv_file),
                "--replace",
                "redact",
                "--model-configs",
                "configs/models_super.yaml",
            ]
        )
    captured = capsys.readouterr()
    assert "Expected YAML mapping" not in captured.err
    assert "got str" not in captured.err
