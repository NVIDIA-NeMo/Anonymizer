# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, ClassVar, Literal

logger = logging.getLogger("anonymizer.cli")

import cyclopts
from pydantic import ValidationError

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Detect, Rewrite
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.config.rewrite import (
    DEFAULT_PRESERVE_TEXT,
    DEFAULT_PROTECT_TEXT,
    PrivacyGoal,
    RiskTolerance,
)
from anonymizer.engine.io.constants import SUPPORTED_IO_FORMATS
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.cli._output import write_result
from anonymizer.interface.errors import AnonymizerIOError, InvalidConfigError
from anonymizer.logging import LoggingConfig, configure_logging

app = cyclopts.App(help="NeMo Anonymizer CLI")

ReplaceChoice = Literal["redact", "hash", "annotate", "substitute"]
RiskToleranceChoice = Literal["minimal", "low", "moderate", "high"]

_STRATEGY_CLS = {
    "substitute": Substitute,
    "redact": Redact,
    "hash": Hash,
    "annotate": Annotate,
}


@dataclass
class CliOpts:
    """CLI options for anonymizer commands.

    Exactly one of ``--replace`` or ``--rewrite`` must be provided.

    Replace-specific flags (``format_template``, ``normalize_label``,
    ``algorithm``, ``digest_length``) only apply when ``--replace`` is set.

    Rewrite-specific flags (``protect``, ``preserve``, ``risk_tolerance``,
    ``max_repair_iterations``) only apply when ``--rewrite`` is set.
    """

    # -- mode selection (exactly one required) --
    replace: ReplaceChoice | None = None
    rewrite: bool = False

    # -- shared --
    detect: Annotated[Detect, cyclopts.Parameter(name="*")] = field(default_factory=Detect)
    model_configs: Annotated[
        str | None,
        cyclopts.Parameter(
            help=(
                "Path to a YAML file (.yaml/.yml) or inline YAML string defining the model pool "
                "and optional selected_models overrides. Omit to use bundled defaults."
            )
        ),
    ] = None
    model_providers: Annotated[
        str | None,
        cyclopts.Parameter(
            help=(
                "Path to a YAML file (.yaml/.yml) or inline YAML string defining custom provider "
                "endpoints. Each entry maps a provider name to its API base URL and optional key."
            )
        ),
    ] = None
    artifact_path: Annotated[
        str | None,
        cyclopts.Parameter(help="Directory for intermediate pipeline artifacts. Defaults to .anonymizer-artifacts."),
    ] = None
    verbose: bool = False
    debug: bool = False

    # -- replace-specific --
    format_template: Annotated[
        str | None,
        cyclopts.Parameter(
            help=(
                "Output template for the replacement token. "
                "Redact supports {label}; Annotate requires {text} and {label}; "
                "Hash requires {digest} and optionally {label}. Not used by substitute."
            )
        ),
    ] = None
    normalize_label: bool = True
    algorithm: Annotated[Literal["sha256", "sha1", "md5"], cyclopts.Parameter(help="Hash algorithm.")] = "sha256"
    digest_length: Annotated[int, cyclopts.Parameter(help="Hash digest length (6-64).")] = 12

    # -- rewrite-specific --
    protect: Annotated[str | None, cyclopts.Parameter(help="What to protect (privacy goal).")] = None
    preserve: Annotated[str | None, cyclopts.Parameter(help="What to preserve (utility goal).")] = None
    risk_tolerance: Annotated[RiskToleranceChoice, cyclopts.Parameter(help="Risk tolerance preset.")] = "low"
    max_repair_iterations: Annotated[int, cyclopts.Parameter(help="Max evaluate-repair iterations.")] = 3

    # -- shared between substitute (replace) and rewrite --
    instructions: Annotated[str | None, cyclopts.Parameter(help="Extra instructions for the LLM.")] = None

    _REPLACE_ONLY_FLAGS: ClassVar[tuple[str, ...]] = ("format_template",)
    _REWRITE_ONLY_FLAGS: ClassVar[tuple[str, ...]] = ("protect", "preserve")

    def __post_init__(self) -> None:
        if self.replace and self.rewrite:
            raise InvalidConfigError("Cannot use both --replace and --rewrite. Choose one mode.")
        self._warn_cross_mode_flags()

    def _warn_cross_mode_flags(self) -> None:
        if self.rewrite:
            stray = [f for f in self._REPLACE_ONLY_FLAGS if getattr(self, f) is not None]
            if stray:
                logger.warning(
                    "Ignoring replace-only flag(s) in rewrite mode: %s",
                    ", ".join(f"--{f.replace('_', '-')}" for f in stray),
                )
        elif self.replace:
            stray = [f for f in self._REWRITE_ONLY_FLAGS if getattr(self, f) not in (None, False)]
            if stray:
                logger.warning(
                    "Ignoring rewrite-only flag(s) in replace mode: %s",
                    ", ".join(f"--{f.replace('_', '-')}" for f in stray),
                )


def _cli_error_handler(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValidationError, ValueError, InvalidConfigError, AnonymizerIOError, FileNotFoundError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1)

    return wrapper


def _build_replace_strategy(opts: CliOpts) -> Substitute | Redact | Hash | Annotate:
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


def _build_rewrite_config(opts: CliOpts) -> Rewrite:
    """Build a Rewrite config from CLI args."""
    privacy_goal = None
    if opts.protect or opts.preserve:
        privacy_goal = PrivacyGoal(
            protect=opts.protect or DEFAULT_PROTECT_TEXT,
            preserve=opts.preserve or DEFAULT_PRESERVE_TEXT,
        )
    return Rewrite(
        privacy_goal=privacy_goal,
        instructions=opts.instructions,
        risk_tolerance=RiskTolerance(opts.risk_tolerance),
        max_repair_iterations=opts.max_repair_iterations,
    )


def _build_config_and_anonymizer(opts: CliOpts) -> tuple[AnonymizerConfig, Anonymizer]:
    """Build the shared AnonymizerConfig and Anonymizer from CLI args."""
    if opts.rewrite:
        strategy = _build_rewrite_config(opts)
        config = AnonymizerConfig(rewrite=strategy, detect=opts.detect)
    elif opts.replace:
        strategy = _build_replace_strategy(opts)
        config = AnonymizerConfig(replace=strategy, detect=opts.detect)
    else:
        raise InvalidConfigError("Specify --replace <strategy> or --rewrite.")

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
    opts: Annotated[CliOpts, cyclopts.Parameter(name="*")] = CliOpts(),
    output: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Output file path (.csv or .parquet). Defaults to source stem + _anonymized or _rewritten."
        ),
    ] = None,
) -> None:
    """Run the full anonymization pipeline (detection + replacement or rewrite)."""
    if output is None:
        source = Path(data.source)
        suffix = "_rewritten" if opts.rewrite else "_anonymized"
        output = str(source.parent / f"{source.stem}{suffix}{source.suffix}")
    output_path = Path(output).resolve()
    if output_path.suffix.lower() not in SUPPORTED_IO_FORMATS:
        raise InvalidConfigError(
            f"Unsupported output format: {output_path.suffix!r}. Use one of {SUPPORTED_IO_FORMATS}"
        )
    if output_path == Path(data.source).resolve():
        raise InvalidConfigError(f"Output path must differ from source: {output_path}")
    _configure_logging(opts)
    config, anonymizer = _build_config_and_anonymizer(opts)
    result = anonymizer.run(config=config, data=data)
    written = write_result(result, output)
    print(f"Output written to: {written}")


@app.command
@_cli_error_handler
def preview(
    *,
    data: Annotated[AnonymizerInput, cyclopts.Parameter(name="*")],
    opts: Annotated[CliOpts, cyclopts.Parameter(name="*")] = CliOpts(),
    num_records: int = 10,
) -> None:
    """Run the pipeline on a subset of records for quick inspection."""
    _configure_logging(opts)
    config, anonymizer = _build_config_and_anonymizer(opts)
    result = anonymizer.preview(config=config, data=data, num_records=num_records)
    print(result.dataframe.to_string(max_colwidth=80))


@app.command
@_cli_error_handler
def validate(
    *,
    data: Annotated[AnonymizerInput, cyclopts.Parameter(name="*")],
    opts: Annotated[CliOpts, cyclopts.Parameter(name="*")] = CliOpts(),
) -> None:
    """Validate that the active config is compatible with model selections."""
    _configure_logging(opts)
    config, anonymizer = _build_config_and_anonymizer(opts)
    anonymizer.validate_config(config)
    print("Config is valid.")


def main() -> None:
    """Entry point for the anonymizer CLI."""
    app()
