# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the output template in ``skills/anonymizer/SKILL.md`` against a stub model endpoint.

The template is read from SKILL.md at test time, its placeholders are filled, and it runs as a script in a temporary
working directory with no network and no API key. Every file the template reports saving must exist and load with the
reader it names, so a save step that fails or a reader that cannot read the file fails here.

Set ``ANONYMIZER_SKILL_MD`` to run the same checks against another copy of SKILL.md (for example, an older revision).
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml
from data_designer.config.models import ModelProvider

import anonymizer
from anonymizer.engine.ndd.model_loader import (
    DEFAULT_CONFIG_DIR,
    parse_model_configs,
    parse_model_providers,
    validate_model_alias_references,
    validate_model_configs_reference_providers,
)
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.errors import InvalidConfigError
from tests.skills.stub_endpoint import StubEndpoint

SKILL_MD = Path(
    os.environ.get("ANONYMIZER_SKILL_MD") or Path(__file__).resolve().parents[2] / "skills/anonymizer/SKILL.md"
)
TEMPLATE_MARKER = "def main()"
LLM_PROVIDER = "stub-llm"
DETECTOR_PROVIDER = "local-gliner2"
INPUT_ROWS = [
    "Alice Smith lives in Paris and works at Acme Corp.",
    "Bob Jones sent the invoice to Carol Lee in Paris.",
    "Carol Lee, an engineer at Acme Corp, met Alice Smith.",
]
PLACEHOLDER_VALUES = {
    "text_column": "text",
    "data_summary": "Short synthetic sentences about people and workplaces",
    "protect": "Names of people, their employer, and the city where they live",
    "preserve": "The social relationships and the sequence of events in the text",
    "instructions": "keep names plausible ",
}
READER_ITEM = re.compile(r"([\w.\-]+\.\w+)\s*\((?:[^)]*?)(pd\.read_\w+)\)")


def _skill_python_blocks(markdown: str) -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", markdown, re.S | re.M)


def _load_template() -> str:
    blocks = [block for block in _skill_python_blocks(SKILL_MD.read_text(encoding="utf-8")) if TEMPLATE_MARKER in block]
    assert len(blocks) == 1, (
        f"expected one python block containing {TEMPLATE_MARKER!r} in {SKILL_MD}, found {len(blocks)}"
    )
    return blocks[0]


def _choose_strategy(source: str, strategy: str) -> str:
    """Keep the template's Rewrite config, or uncomment its Replace config and drop the Rewrite one."""
    if strategy == "rewrite":
        return source
    lines = source.splitlines()
    replace_at = next(i for i, line in enumerate(lines) if line.strip().startswith("# Replace mode ("))
    rewrite_at = next(i for i, line in enumerate(lines) if line.strip().startswith("# Rewrite mode ("))
    return_at = next(i for i, line in enumerate(lines) if line.strip() == "return data, config")
    uncommented = [
        re.sub(r"^(\s*)# ?", r"\1", line, count=1) if line.strip() else line
        for line in lines[replace_at + 1 : rewrite_at]
    ]
    return "\n".join([*lines[: replace_at + 1], *uncommented, *lines[return_at:]])


def _fill_placeholders(source: str, input_path: Path) -> str:
    values = {**PLACEHOLDER_VALUES, "source": str(input_path)}

    def fill(match: re.Match[str]) -> str:
        assert match.group(1) in values, f"no test value for template placeholder {match.group(0)!r}"
        return f"{match.group(1)}={values[match.group(1)]!r}"

    filled = re.sub(r'\b(\w+)="TODO[^"\n]*"', fill, source)
    assert '"TODO' not in filled, "template still holds an unfilled placeholder"
    return filled


def _stub_model_providers(endpoint: StubEndpoint) -> list[ModelProvider]:
    """A custom provider list that keeps the detector's provider, as the skill's provider setup text requires."""
    return [
        ModelProvider(name=LLM_PROVIDER, endpoint=endpoint.url, provider_type="openai", api_key="stub-key"),
        ModelProvider(name=DETECTOR_PROVIDER, endpoint=endpoint.url, provider_type="openai", api_key="EMPTY"),
    ]


def _stub_model_configs() -> str:
    """The bundled model pool with every LLM alias moved to the stub provider; the detector keeps its provider."""
    bundled = yaml.safe_load((DEFAULT_CONFIG_DIR / "models.yaml").read_text(encoding="utf-8"))
    for config in bundled["model_configs"]:
        if config["provider"] != DETECTOR_PROVIDER:
            config["provider"] = LLM_PROVIDER
    return yaml.safe_dump(bundled)


@pytest.fixture
def endpoint() -> Iterator[StubEndpoint]:
    stub = StubEndpoint()
    stub.start()
    try:
        yield stub
    finally:
        stub.stop()


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    pd.DataFrame({"text": INPUT_ROWS}).to_csv(tmp_path / "input.csv", index=False)
    return tmp_path


@pytest.fixture
def run_template(
    workdir: Path, endpoint: StubEndpoint, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Callable[..., str]:
    """Return a runner that executes the filled template with the given CLI args and returns its stdout."""
    real_anonymizer = anonymizer.Anonymizer

    def stub_anonymizer(**kwargs: Any) -> Anonymizer:
        # The template builds a plain ``Anonymizer()``; give it the custom provider setup the stub needs.
        return real_anonymizer(
            model_providers=_stub_model_providers(endpoint),
            model_configs=_stub_model_configs(),
            artifact_path=workdir / "artifacts",
            **kwargs,
        )

    monkeypatch.setattr(anonymizer, "Anonymizer", stub_anonymizer)

    def run(strategy: str, *args: str) -> str:
        source = _fill_placeholders(_choose_strategy(_load_template(), strategy), workdir / "input.csv")
        namespace: dict[str, Any] = {
            "__name__": "__main__",
            "__file__": str(workdir / "anonymize_template.py"),
        }
        monkeypatch.setattr(sys, "argv", ["anonymize_template.py", *args])
        capsys.readouterr()
        exec(compile(source, "anonymize_template.py", "exec"), namespace)  # noqa: S102
        return capsys.readouterr().out

    return run


def _assert_reported_saves_load(output: str, workdir: Path, expected_rows: int) -> dict[str, pd.DataFrame]:
    """Every ``file (pd.read_x)`` the template reports saving exists and loads with that reader."""
    saved_lines = [line for line in output.splitlines() if line.startswith("Saved:")]
    assert saved_lines, f"template reported no saved files:\n{output}"
    frames: dict[str, pd.DataFrame] = {}
    for line in saved_lines:
        items = READER_ITEM.findall(line)
        assert items, f"no 'file (pd.read_*)' pair in: {line}"
        for name, reader in items:
            path = workdir / name
            assert path.is_file(), f"{name} reported as saved but missing"
            frames[name] = getattr(pd, reader.removeprefix("pd."))(path)
            assert len(frames[name]) == expected_rows, f"{name} has {len(frames[name])} rows"
    return frames


@pytest.mark.parametrize("strategy", ["rewrite", "replace"])
def test_template_preview_saves_files_that_load(strategy: str, run_template: Callable[..., str], workdir: Path) -> None:
    output = run_template(strategy, "--num-records", "3")

    assert "Previewed 3 rows." in output
    frames = _assert_reported_saves_load(output, workdir, expected_rows=3)
    assert "preview.parquet" in frames
    assert "preview_trace.pkl" in frames
    assert set(frames["preview.parquet"].columns) <= set(frames["preview_trace.pkl"].columns)
    if strategy == "rewrite":
        assert "leakage_mass:" in output and "utility_score:" in output


@pytest.mark.parametrize("strategy", ["rewrite", "replace"])
def test_template_full_run_writes_output_parquet(
    strategy: str, run_template: Callable[..., str], workdir: Path
) -> None:
    output = run_template(strategy, "--full")

    assert "Wrote 3 rows to output.parquet" in output
    path = workdir / "output.parquet"
    assert pq.read_table(path).num_rows == 3
    if strategy == "replace":
        # Known gap: the Replace output frame carries a struct-typed ``final_entities`` column that pandas
        # writes but cannot read back. The test fails once pandas reads the file, so the marker gets removed.
        with pytest.raises(ValueError, match="is not recognized"):
            pd.read_parquet(path)
        pytest.xfail("pd.read_parquet cannot read back the Replace output frame")
    saved = pd.read_parquet(path)
    assert len(saved) == 3
    assert "text_rewritten" in saved.columns


def test_template_run_resolves_every_alias_through_custom_providers(
    run_template: Callable[..., str], endpoint: StubEndpoint
) -> None:
    run_template("rewrite", "--full")

    pool = {config["model"] for config in yaml.safe_load(_stub_model_configs())["model_configs"]}
    selected = parse_model_configs(_stub_model_configs()).selected_models
    pool_by_alias = {config.alias: config.model for config in parse_model_configs(_stub_model_configs()).model_configs}
    assert endpoint.models_seen <= pool
    assert pool_by_alias[selected.detection.entity_detector] in endpoint.models_seen
    # These roles share a model; their response schemas prove both requests reached the stub (#352).
    assert frozenset({"rewritten_text"}) in endpoint.response_shapes_seen
    assert frozenset({"decisions"}) in endpoint.response_shapes_seen


def test_custom_provider_setup_follows_the_skill_bullet(endpoint: StubEndpoint) -> None:
    """The setup the skill describes is two objects, and a custom provider list must keep the detector's provider."""
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "`model_providers`" in text and "`model_configs`" in text
    assert f"keep `{DETECTOR_PROVIDER}`" in text

    providers = parse_model_providers(_stub_model_providers(endpoint))
    parsed = parse_model_configs(_stub_model_configs())
    validate_model_configs_reference_providers(parsed.model_configs, providers)
    validate_model_alias_references(
        parsed.model_configs, parsed.selected_models, check_rewrite=True, check_substitute=True
    )

    detector_only_llm_provider = [provider for provider in providers if provider.name == LLM_PROVIDER]
    with pytest.raises(InvalidConfigError, match=DETECTOR_PROVIDER):
        Anonymizer(model_providers=detector_only_llm_provider, model_configs=_stub_model_configs())
