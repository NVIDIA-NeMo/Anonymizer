# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import sys

import tyro
from pydantic import ValidationError

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.cli._output import format_summary, write_result
from anonymizer.logging import LoggingConfig, configure_logging


@dataclasses.dataclass
class GlobalOpts:
    """Global options shared across all subcommands."""

    model_configs: str | None = None
    """Path or YAML string for model pool configuration. None uses bundled defaults."""

    model_providers: str | None = None
    """Path or YAML string for provider definitions (endpoint, API key)."""

    artifact_path: str | None = None
    """Directory for intermediate artifacts. Defaults to .anonymizer-artifacts."""

    verbose: bool = False
    """Enable verbose logging (data_designer logs at INFO level)."""

    debug: bool = False
    """Enable debug logging (anonymizer logs at DEBUG level)."""


app = tyro.extras.SubcommandApp()


@app.command
def run(
    config: AnonymizerConfig,
    data: AnonymizerInput,
    output: str | None = None,
    global_opts: GlobalOpts = GlobalOpts(),
) -> None:
    """Run the full anonymization pipeline (detection + replacement).

    The output is written to --output if provided; otherwise only the
    summary is printed to stdout.
    """
    _configure_logging(global_opts)
    anonymizer = Anonymizer(
        model_configs=global_opts.model_configs,
        model_providers=global_opts.model_providers,
        artifact_path=global_opts.artifact_path,
    )
    result = anonymizer.run(config=config, data=data)
    print(format_summary(result))
    if output:
        written = write_result(result, output)
        print(f"Output written to: {written}")


@app.command
def preview(
    config: AnonymizerConfig,
    data: AnonymizerInput,
    num_records: int = 10,
    global_opts: GlobalOpts = GlobalOpts(),
) -> None:
    """Run the pipeline on a subset of records for quick inspection."""
    _configure_logging(global_opts)
    anonymizer = Anonymizer(
        model_configs=global_opts.model_configs,
        model_providers=global_opts.model_providers,
        artifact_path=global_opts.artifact_path,
    )
    result = anonymizer.preview(config=config, data=data, num_records=num_records)
    print(format_summary(result))


@app.command
def validate(
    config: AnonymizerConfig,
    data: AnonymizerInput,
    global_opts: GlobalOpts = GlobalOpts(),
) -> None:
    """Validate that the active config is compatible with model selections."""
    _configure_logging(global_opts)
    anonymizer = Anonymizer(
        model_configs=global_opts.model_configs,
        model_providers=global_opts.model_providers,
        artifact_path=global_opts.artifact_path,
    )
    anonymizer.validate_config(config)
    print("Config is valid.")


def _configure_logging(opts: GlobalOpts) -> None:
    if opts.debug:
        configure_logging(LoggingConfig.debug())
    elif opts.verbose:
        configure_logging(LoggingConfig.verbose())
    else:
        configure_logging(LoggingConfig.default())


def main() -> None:
    """Entry point for the anonymizer CLI."""
    try:
        app.cli()
    except (ValidationError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
