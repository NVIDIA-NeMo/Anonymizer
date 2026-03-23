# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from typing import Annotated, Literal

import cyclopts
from pydantic import ValidationError

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Detect
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.cli._output import format_summary, write_result
from anonymizer.logging import LoggingConfig, configure_logging

app = cyclopts.App(help="NeMo Anonymizer CLI")

ReplaceChoice = Literal["redact", "hash", "annotate", "substitute"]

_STRATEGY_CLS = {
    "redact": Redact,
    "hash": Hash,
    "annotate": Annotate,
    "substitute": Substitute,
}


def _build_replace_strategy(
    name: ReplaceChoice,
    format_template: str | None = None,
    normalize_label: bool = True,
    algorithm: Literal["sha256", "sha1", "md5"] = "sha256",
    digest_length: int = 12,
    instructions: str | None = None,
) -> Redact | Hash | Annotate | Substitute:
    """Build a replace strategy instance from CLI args.

    Only passes non-default kwargs so Pydantic field defaults are preserved.
    """
    cls = _STRATEGY_CLS[name]
    kw: dict = {}
    if format_template is not None and "format_template" in cls.model_fields:
        kw["format_template"] = format_template
    if cls is Redact:
        kw["normalize_label"] = normalize_label
    if cls is Hash:
        kw["algorithm"] = algorithm
        kw["digest_length"] = digest_length
    if cls is Substitute and instructions is not None:
        kw["instructions"] = instructions
    return cls(**kw)


def _configure_logging(*, verbose: bool, debug: bool) -> None:
    if debug:
        configure_logging(LoggingConfig.debug())
    elif verbose:
        configure_logging(LoggingConfig.verbose())
    else:
        configure_logging(LoggingConfig.default())


@app.command
def run(
    *,
    data: Annotated[AnonymizerInput, cyclopts.Parameter(name="*")],
    replace: ReplaceChoice,
    output: str | None = None,
    detect: Annotated[Detect, cyclopts.Parameter(name="*")] = Detect(),
    format_template: str | None = None,
    normalize_label: bool = True,
    algorithm: Annotated[Literal["sha256", "sha1", "md5"], cyclopts.Parameter(help="Hash algorithm.")] = "sha256",
    digest_length: Annotated[int, cyclopts.Parameter(help="Hash digest length (6-64).")] = 12,
    instructions: Annotated[str | None, cyclopts.Parameter(help="Extra instructions for substitute.")] = None,
    model_configs: str | None = None,
    model_providers: str | None = None,
    artifact_path: str | None = None,
    verbose: bool = False,
    debug: bool = False,
) -> None:
    """Run the full anonymization pipeline (detection + replacement)."""
    _configure_logging(verbose=verbose, debug=debug)
    try:
        strategy = _build_replace_strategy(
            replace, format_template, normalize_label, algorithm, digest_length, instructions
        )
        config = AnonymizerConfig(replace=strategy, detect=detect)
    except (ValidationError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    anonymizer = Anonymizer(model_configs=model_configs, model_providers=model_providers, artifact_path=artifact_path)
    result = anonymizer.run(config=config, data=data)
    print(format_summary(result))
    if output:
        written = write_result(result, output)
        print(f"Output written to: {written}")


@app.command
def preview(
    *,
    data: Annotated[AnonymizerInput, cyclopts.Parameter(name="*")],
    replace: ReplaceChoice,
    num_records: int = 10,
    detect: Annotated[Detect, cyclopts.Parameter(name="*")] = Detect(),
    format_template: str | None = None,
    normalize_label: bool = True,
    algorithm: Annotated[Literal["sha256", "sha1", "md5"], cyclopts.Parameter(help="Hash algorithm.")] = "sha256",
    digest_length: Annotated[int, cyclopts.Parameter(help="Hash digest length (6-64).")] = 12,
    instructions: Annotated[str | None, cyclopts.Parameter(help="Extra instructions for substitute.")] = None,
    model_configs: str | None = None,
    model_providers: str | None = None,
    artifact_path: str | None = None,
    verbose: bool = False,
    debug: bool = False,
) -> None:
    """Run the pipeline on a subset of records for quick inspection."""
    _configure_logging(verbose=verbose, debug=debug)
    try:
        strategy = _build_replace_strategy(
            replace, format_template, normalize_label, algorithm, digest_length, instructions
        )
        config = AnonymizerConfig(replace=strategy, detect=detect)
    except (ValidationError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    anonymizer = Anonymizer(model_configs=model_configs, model_providers=model_providers, artifact_path=artifact_path)
    result = anonymizer.preview(config=config, data=data, num_records=num_records)
    print(format_summary(result))


@app.command
def validate(
    *,
    data: Annotated[AnonymizerInput, cyclopts.Parameter(name="*")],
    replace: ReplaceChoice,
    detect: Annotated[Detect, cyclopts.Parameter(name="*")] = Detect(),
    format_template: str | None = None,
    normalize_label: bool = True,
    algorithm: Annotated[Literal["sha256", "sha1", "md5"], cyclopts.Parameter(help="Hash algorithm.")] = "sha256",
    digest_length: Annotated[int, cyclopts.Parameter(help="Hash digest length (6-64).")] = 12,
    instructions: Annotated[str | None, cyclopts.Parameter(help="Extra instructions for substitute.")] = None,
    model_configs: str | None = None,
    model_providers: str | None = None,
    artifact_path: str | None = None,
    verbose: bool = False,
    debug: bool = False,
) -> None:
    """Validate that the active config is compatible with model selections."""
    _configure_logging(verbose=verbose, debug=debug)
    try:
        strategy = _build_replace_strategy(
            replace, format_template, normalize_label, algorithm, digest_length, instructions
        )
        config = AnonymizerConfig(replace=strategy, detect=detect)
        AnonymizerInput.model_validate(data.model_dump())
    except (ValidationError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    anonymizer = Anonymizer(model_configs=model_configs, model_providers=model_providers, artifact_path=artifact_path)
    anonymizer.validate_config(config)
    print("Config is valid.")


def main() -> None:
    """Entry point for the anonymizer CLI."""
    app()
