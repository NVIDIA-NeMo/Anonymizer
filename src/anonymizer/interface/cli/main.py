# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from typing import Annotated

import cyclopts
from pydantic import ValidationError

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Detect
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, ReplaceMethod, Substitute
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.cli._output import format_summary, write_result
from anonymizer.logging import LoggingConfig, configure_logging

app = cyclopts.App(help="NeMo Anonymizer CLI")

_REPLACE_CHOICES = ("redact", "hash", "annotate", "substitute")


def _build_replace_strategy(
    replace: str,
    format_template: str | None,
    normalize_label: bool,
    algorithm: str,
    digest_length: int,
    instructions: str | None,
) -> ReplaceMethod:
    match replace.lower():
        case "redact":
            kw = {} if format_template is None else {"format_template": format_template}
            return Redact(normalize_label=normalize_label, **kw)
        case "hash":
            kw = {} if format_template is None else {"format_template": format_template}
            return Hash(algorithm=algorithm, digest_length=digest_length, **kw)
        case "annotate":
            kw = {} if format_template is None else {"format_template": format_template}
            return Annotate(**kw)
        case "substitute":
            return Substitute(instructions=instructions)
        case _:
            raise ValueError(
                f"Unknown replace strategy: {replace!r}. "
                f"Choose from: {', '.join(_REPLACE_CHOICES)}"
            )


def _build_anonymizer_config(
    *,
    replace: str,
    entity_labels: list[str] | None = None,
    gliner_threshold: float = 0.3,
    format_template: str | None = None,
    normalize_label: bool = True,
    algorithm: str = "sha256",
    digest_length: int = 12,
    instructions: str | None = None,
) -> AnonymizerConfig:
    replace_strategy = _build_replace_strategy(
        replace=replace,
        format_template=format_template,
        normalize_label=normalize_label,
        algorithm=algorithm,
        digest_length=digest_length,
        instructions=instructions,
    )
    detect = Detect(entity_labels=entity_labels, gliner_threshold=gliner_threshold)
    return AnonymizerConfig(replace=replace_strategy, detect=detect)


def _build_anonymizer_input(
    *,
    source: str,
    text_column: str = "text",
    id_column: str | None = None,
    data_summary: str | None = None,
) -> AnonymizerInput:
    return AnonymizerInput(
        source=source,
        text_column=text_column,
        id_column=id_column,
        data_summary=data_summary,
    )


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
    source: Annotated[str, cyclopts.Parameter(help="Path to input CSV or Parquet file.")],
    replace: Annotated[str, cyclopts.Parameter(help="Replace strategy: redact | hash | annotate | substitute.")],
    output: Annotated[str | None, cyclopts.Parameter(help="Output file path.")] = None,
    text_column: Annotated[str, cyclopts.Parameter(help="Column containing text to anonymize.")] = "text",
    id_column: Annotated[str | None, cyclopts.Parameter(help="Column to use as record identifier.")] = None,
    data_summary: Annotated[str | None, cyclopts.Parameter(help="Short description of the data.")] = None,
    entity_labels: Annotated[list[str] | None, cyclopts.Parameter(help="Entity labels to detect.")] = None,
    gliner_threshold: Annotated[float, cyclopts.Parameter(help="GLiNER detection threshold (0.0–1.0).")] = 0.3,
    format_template: Annotated[
        str | None, cyclopts.Parameter(help="Replacement template for redact/annotate/hash strategies.")
    ] = None,
    normalize_label: Annotated[bool, cyclopts.Parameter(help="Normalize label in redact template.")] = True,
    algorithm: Annotated[str, cyclopts.Parameter(help="Hash algorithm: sha256 | sha1 | md5.")] = "sha256",
    digest_length: Annotated[int, cyclopts.Parameter(help="Hash digest length in hex characters (6–64).")] = 12,
    instructions: Annotated[str | None, cyclopts.Parameter(help="Extra instructions for substitute strategy.")] = None,
    model_configs: Annotated[str | None, cyclopts.Parameter(help="Model pool config path or YAML string.")] = None,
    model_providers: Annotated[str | None, cyclopts.Parameter(help="Provider definitions path or YAML string.")] = None,
    artifact_path: Annotated[str | None, cyclopts.Parameter(help="Directory for intermediate artifacts.")] = None,
    verbose: Annotated[bool, cyclopts.Parameter(help="Enable verbose logging.")] = False,
    debug: Annotated[bool, cyclopts.Parameter(help="Enable debug logging.")] = False,
) -> None:
    """Run the full anonymization pipeline (detection + replacement).

    Output is written to --output if provided; otherwise only the summary is printed.
    """
    _configure_logging(verbose=verbose, debug=debug)
    try:
        config = _build_anonymizer_config(
            replace=replace,
            entity_labels=entity_labels,
            gliner_threshold=gliner_threshold,
            format_template=format_template,
            normalize_label=normalize_label,
            algorithm=algorithm,
            digest_length=digest_length,
            instructions=instructions,
        )
        data = _build_anonymizer_input(
            source=source,
            text_column=text_column,
            id_column=id_column,
            data_summary=data_summary,
        )
    except (ValidationError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    anonymizer = Anonymizer(
        model_configs=model_configs,
        model_providers=model_providers,
        artifact_path=artifact_path,
    )
    result = anonymizer.run(config=config, data=data)
    print(format_summary(result))
    if output:
        written = write_result(result, output)
        print(f"Output written to: {written}")


@app.command
def preview(
    *,
    source: Annotated[str, cyclopts.Parameter(help="Path to input CSV or Parquet file.")],
    replace: Annotated[str, cyclopts.Parameter(help="Replace strategy: redact | hash | annotate | substitute.")],
    num_records: Annotated[int, cyclopts.Parameter(help="Number of records to preview.")] = 10,
    text_column: Annotated[str, cyclopts.Parameter(help="Column containing text to anonymize.")] = "text",
    id_column: Annotated[str | None, cyclopts.Parameter(help="Column to use as record identifier.")] = None,
    data_summary: Annotated[str | None, cyclopts.Parameter(help="Short description of the data.")] = None,
    entity_labels: Annotated[list[str] | None, cyclopts.Parameter(help="Entity labels to detect.")] = None,
    gliner_threshold: Annotated[float, cyclopts.Parameter(help="GLiNER detection threshold (0.0–1.0).")] = 0.3,
    format_template: Annotated[
        str | None, cyclopts.Parameter(help="Replacement template for redact/annotate/hash strategies.")
    ] = None,
    normalize_label: Annotated[bool, cyclopts.Parameter(help="Normalize label in redact template.")] = True,
    algorithm: Annotated[str, cyclopts.Parameter(help="Hash algorithm: sha256 | sha1 | md5.")] = "sha256",
    digest_length: Annotated[int, cyclopts.Parameter(help="Hash digest length in hex characters (6–64).")] = 12,
    instructions: Annotated[str | None, cyclopts.Parameter(help="Extra instructions for substitute strategy.")] = None,
    model_configs: Annotated[str | None, cyclopts.Parameter(help="Model pool config path or YAML string.")] = None,
    model_providers: Annotated[str | None, cyclopts.Parameter(help="Provider definitions path or YAML string.")] = None,
    artifact_path: Annotated[str | None, cyclopts.Parameter(help="Directory for intermediate artifacts.")] = None,
    verbose: Annotated[bool, cyclopts.Parameter(help="Enable verbose logging.")] = False,
    debug: Annotated[bool, cyclopts.Parameter(help="Enable debug logging.")] = False,
) -> None:
    """Run the pipeline on a subset of records for quick inspection."""
    _configure_logging(verbose=verbose, debug=debug)
    try:
        config = _build_anonymizer_config(
            replace=replace,
            entity_labels=entity_labels,
            gliner_threshold=gliner_threshold,
            format_template=format_template,
            normalize_label=normalize_label,
            algorithm=algorithm,
            digest_length=digest_length,
            instructions=instructions,
        )
        data = _build_anonymizer_input(
            source=source,
            text_column=text_column,
            id_column=id_column,
            data_summary=data_summary,
        )
    except (ValidationError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    anonymizer = Anonymizer(
        model_configs=model_configs,
        model_providers=model_providers,
        artifact_path=artifact_path,
    )
    result = anonymizer.preview(config=config, data=data, num_records=num_records)
    print(format_summary(result))


@app.command
def validate(
    *,
    source: Annotated[str, cyclopts.Parameter(help="Path to input CSV or Parquet file.")],
    replace: Annotated[str, cyclopts.Parameter(help="Replace strategy: redact | hash | annotate | substitute.")],
    text_column: Annotated[str, cyclopts.Parameter(help="Column containing text to anonymize.")] = "text",
    id_column: Annotated[str | None, cyclopts.Parameter(help="Column to use as record identifier.")] = None,
    data_summary: Annotated[str | None, cyclopts.Parameter(help="Short description of the data.")] = None,
    entity_labels: Annotated[list[str] | None, cyclopts.Parameter(help="Entity labels to detect.")] = None,
    gliner_threshold: Annotated[float, cyclopts.Parameter(help="GLiNER detection threshold (0.0–1.0).")] = 0.3,
    format_template: Annotated[
        str | None, cyclopts.Parameter(help="Replacement template for redact/annotate/hash strategies.")
    ] = None,
    normalize_label: Annotated[bool, cyclopts.Parameter(help="Normalize label in redact template.")] = True,
    algorithm: Annotated[str, cyclopts.Parameter(help="Hash algorithm: sha256 | sha1 | md5.")] = "sha256",
    digest_length: Annotated[int, cyclopts.Parameter(help="Hash digest length in hex characters (6–64).")] = 12,
    instructions: Annotated[str | None, cyclopts.Parameter(help="Extra instructions for substitute strategy.")] = None,
    model_configs: Annotated[str | None, cyclopts.Parameter(help="Model pool config path or YAML string.")] = None,
    model_providers: Annotated[str | None, cyclopts.Parameter(help="Provider definitions path or YAML string.")] = None,
    artifact_path: Annotated[str | None, cyclopts.Parameter(help="Directory for intermediate artifacts.")] = None,
    verbose: Annotated[bool, cyclopts.Parameter(help="Enable verbose logging.")] = False,
    debug: Annotated[bool, cyclopts.Parameter(help="Enable debug logging.")] = False,
) -> None:
    """Validate that the active config is compatible with model selections."""
    _configure_logging(verbose=verbose, debug=debug)
    try:
        config = _build_anonymizer_config(
            replace=replace,
            entity_labels=entity_labels,
            gliner_threshold=gliner_threshold,
            format_template=format_template,
            normalize_label=normalize_label,
            algorithm=algorithm,
            digest_length=digest_length,
            instructions=instructions,
        )
        _build_anonymizer_input(
            source=source,
            text_column=text_column,
            id_column=id_column,
            data_summary=data_summary,
        )
    except (ValidationError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    anonymizer = Anonymizer(
        model_configs=model_configs,
        model_providers=model_providers,
        artifact_path=artifact_path,
    )
    anonymizer.validate_config(config)
    print("Config is valid.")


def main() -> None:
    """Entry point for the anonymizer CLI."""
    app()
