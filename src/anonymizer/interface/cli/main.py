# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Annotated, Literal

import cyclopts
import functools
from pydantic import ValidationError

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Detect
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.errors import InvalidConfigError, AnonymizerIOError
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


@dataclass
class CliOpts:
    """Parameters shared across all CLI subcommands."""

    replace: ReplaceChoice
    detect: Annotated[Detect, cyclopts.Parameter(name="*")] = field(default_factory=Detect)
    format_template: str | None = None
    normalize_label: bool = True
    algorithm: Annotated[Literal["sha256", "sha1", "md5"], cyclopts.Parameter(help="Hash algorithm.")] = "sha256"
    digest_length: Annotated[int, cyclopts.Parameter(help="Hash digest length (6-64).")] = 12
    instructions: Annotated[str | None, cyclopts.Parameter(help="Extra instructions for substitute.")] = None
    model_configs: str | None = None
    model_providers: str | None = None
    artifact_path: str | None = None
    verbose: bool = False
    debug: bool = False


def _cli_error_handler(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValidationError, ValueError, InvalidConfigError, AnonymizerIOError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1)

    return wrapper


def _build_replace_strategy(opts: CliOpts) -> Redact | Hash | Annotate | Substitute:
    """Build a replace strategy instance from CLI args.

    Only passes non-default kwargs so Pydantic field defaults are preserved.
    """
    cls = _STRATEGY_CLS[opts.replace]
    kw: dict = {}
    if opts.format_template is not None and "format_template" in cls.model_fields:
        kw["format_template"] = opts.format_template
    if cls is Redact:
        kw["normalize_label"] = opts.normalize_label
    if cls is Hash:
        kw["algorithm"] = opts.algorithm
        kw["digest_length"] = opts.digest_length
    if cls is Substitute and opts.instructions is not None:
        kw["instructions"] = opts.instructions
    return cls(**kw)


def _build_config_and_anonymizer(opts: CliOpts) -> tuple[AnonymizerConfig, Anonymizer]:
    """Build the shared AnonymizerConfig and Anonymizer from CLI args."""
    strategy = _build_replace_strategy(opts)
    config = AnonymizerConfig(replace=strategy, detect=opts.detect)
    anonymizer = Anonymizer(
        model_configs=opts.model_configs,
        model_providers=opts.model_providers,
        artifact_path=opts.artifact_path,
    )
    return config, anonymizer


def _configure_logging(opts: CliOpts) -> None:
    if opts.debug:
        configure_logging(LoggingConfig.debug())
    elif opts.verbose:
        configure_logging(LoggingConfig.verbose())
    else:
        configure_logging(LoggingConfig.default())


@app.command
@_cli_error_handler
def run(
    *,
    data: Annotated[AnonymizerInput, cyclopts.Parameter(name="*")],
    opts: Annotated[CliOpts, cyclopts.Parameter(name="*")],
    output: str | None = None,
) -> None:
    """Run the full anonymization pipeline (detection + replacement)."""
    _configure_logging(opts)
    config, anonymizer = _build_config_and_anonymizer(opts)
    result = anonymizer.run(config=config, data=data)
    print(format_summary(result))
    if output:
        written = write_result(result, output)
        print(f"Output written to: {written}")


@app.command
@_cli_error_handler
def preview(
    *,
    data: Annotated[AnonymizerInput, cyclopts.Parameter(name="*")],
    opts: Annotated[CliOpts, cyclopts.Parameter(name="*")],
    num_records: int = 10,
) -> None:
    """Run the pipeline on a subset of records for quick inspection."""
    _configure_logging(opts)
    config, anonymizer = _build_config_and_anonymizer(opts)
    result = anonymizer.preview(config=config, data=data, num_records=num_records)
    print(format_summary(result))


@app.command
@_cli_error_handler
def validate(
    *,
    data: Annotated[AnonymizerInput, cyclopts.Parameter(name="*")],
    opts: Annotated[CliOpts, cyclopts.Parameter(name="*")],
) -> None:
    """Validate that the active config is compatible with model selections."""
    _configure_logging(opts)
    config, anonymizer = _build_config_and_anonymizer(opts)
    anonymizer.validate_config(config)
    print("Config is valid.")


def main() -> None:
    """Entry point for the anonymizer CLI."""
    app()
