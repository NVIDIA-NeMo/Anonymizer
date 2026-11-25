# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``anonymizer`` CLI, generated from the config models.

Each of ``run``, ``preview`` and ``validate`` has one subcommand per mode: the replace strategies in
``ReplaceMethod`` and ``rewrite``. A subcommand's flags are the fields of ``AnonymizerInput``, ``Detect`` and that
mode's model, under the fields' own names and with their descriptions as help, plus ``RunOptions``. A flag of
another mode is an unknown option. The models hold every default; this module holds none.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, get_args

# When invoked via the CLI entry point, telemetry deployment type defaults to "cli".
# Anonymizer's SDK path sets "sdk" via os.environ.setdefault in Anonymizer.__init__,
# but the CLI is loaded first so its setdefault wins for CLI-driven runs.
os.environ.setdefault("NEMO_DEPLOYMENT_TYPE", "cli")

logger = logging.getLogger("anonymizer.cli")

import cyclopts
from pydantic import BaseModel, ValidationError

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Detect, Rewrite
from anonymizer.config.replace_strategies import ReplaceMethod
from anonymizer.engine.graph_core.kernel import GraphRejection
from anonymizer.engine.io.constants import SUPPORTED_IO_FORMATS
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.cli._output import write_result
from anonymizer.interface.errors import AnonymizerIOError, InvalidConfigError, InvalidInputError
from anonymizer.logging import LoggingConfig, configure_logging

# One subcommand per mode, from the members of ReplaceMethod (by their ``kind``) plus rewrite.
MODES: dict[str, type[BaseModel]] = {
    cls.model_fields["kind"].default: cls for cls in get_args(get_args(ReplaceMethod)[0])
} | {"rewrite": Rewrite}


@dataclass(frozen=True)
class RunOptions:
    """Options that are not config: how the ``Anonymizer`` is built, telemetry and logging."""

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
    # AnonymizerConfig.emit_telemetry; the other AnonymizerConfig fields are the subcommand and its flags.
    emit_telemetry: Annotated[
        bool,
        cyclopts.Parameter(
            help=(
                "Whether to emit anonymous Anonymizer telemetry events. "
                "Use --no-emit-telemetry to opt out for this invocation. "
                "Also overridable globally via NEMO_TELEMETRY_ENABLED=false."
            )
        ),
    ] = AnonymizerConfig.model_fields["emit_telemetry"].default
    verbose: Annotated[bool, cyclopts.Parameter(help="Verbose logging.", negative="")] = False
    debug: Annotated[bool, cyclopts.Parameter(help="Debug logging.", negative="")] = False


# ``--empty-<list>`` flags would only restate a list's empty default, so they are turned off.
app = cyclopts.App(
    help="NeMo Anonymizer CLI",
    default_parameter=cyclopts.Parameter(negative_iterable=()),
    # Field descriptions are plain text; markup rendering would drop tokens such as <LABEL>.
    help_format="plaintext",
)


def _cli_error_handler(fn: Callable[..., None]) -> Callable[..., None]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            return fn(*args, **kwargs)
        except GraphRejection:
            raise  # a graph wiring fault keeps its traceback
        except (ValidationError, ValueError, InvalidConfigError, InvalidInputError, AnonymizerIOError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1)

    return wrapper


def _build(
    mode: str, strategy: BaseModel | None, detect: Detect | None, options: RunOptions
) -> tuple[AnonymizerConfig, Anonymizer]:
    """Build the config and the Anonymizer. A model the user gave no flag for is built from its defaults."""
    if strategy is None:
        strategy = MODES[mode]()
    detect = detect if detect is not None else Detect()
    if isinstance(strategy, Rewrite):
        config = AnonymizerConfig(rewrite=strategy, detect=detect, emit_telemetry=options.emit_telemetry)
    else:
        config = AnonymizerConfig.model_validate(
            {"replace": strategy, "detect": detect, "emit_telemetry": options.emit_telemetry}
        )
    if options.debug:
        configure_logging(LoggingConfig.debug())
    elif options.verbose:
        configure_logging(LoggingConfig.verbose())
    else:
        configure_logging(LoggingConfig.default())
    anonymizer = Anonymizer(
        model_configs=options.model_configs,
        model_providers=options.model_providers,
        artifact_path=options.artifact_path,
    )
    return config, anonymizer


def _output_path(data: AnonymizerInput, mode: str, output: str | None) -> str:
    if output is None:
        source = Path(data.source)
        suffix = "_rewritten" if mode == "rewrite" else "_anonymized"
        output = str(source.parent / f"{source.stem}{suffix}{source.suffix}")
    output_path = Path(output).resolve()
    if output_path.suffix.lower() not in SUPPORTED_IO_FORMATS:
        raise InvalidConfigError(
            f"Unsupported output format: {output_path.suffix!r}. Use one of {SUPPORTED_IO_FORMATS}"
        )
    if output_path == Path(data.source).resolve():
        raise InvalidConfigError(f"Output path must differ from source: {output_path}")
    return output


_PREVIEW_NUM_RECORDS: int = inspect.signature(Anonymizer.preview).parameters["num_records"].default


# Help panels in reading order: what to read and write, the mode's own settings, detection, then run options.
INPUT_GROUP = cyclopts.Group("Input and output", sort_key=0)
DETECT_GROUP = cyclopts.Group(
    "Detect",
    sort_key=2,
    help=(
        "List flags take one value each; repeat the flag for more (--entity-labels first_name --entity-labels "
        "city). --entity-label-examples takes a JSON object. --regex-rules takes one JSON rule or a JSON list of "
        "rules per flag; a rule's validator is a registered validator name, since Python callables cannot be "
        "given on the command line."
    ),
)
RUN_GROUP = cyclopts.Group("Run", sort_key=3)


def _mode_group(mode: str, model: type[BaseModel]) -> cyclopts.Group:
    """The mode's help panel. A field read only from the environment has no flag, so the panel names it."""
    notes = [
        f"The {name} is read from {' or '.join(meta.env_var)}; it has no flag."
        for name, info in model.model_fields.items()
        for meta in info.metadata
        if isinstance(meta, cyclopts.Parameter) and meta.parse is False and meta.env_var
    ]
    return cyclopts.Group(mode.title(), sort_key=1, help=" ".join(notes))


def _command(verb: str, mode: str) -> Callable[..., None]:
    """The ``<verb> <mode>`` command. Its annotations are set here, since the mode's model is only known now."""
    if verb == "run":

        def command(*, data, strategy=None, detect=None, options=RunOptions(), output=None) -> None:
            path = _output_path(data, mode, output)
            config, anonymizer = _build(mode, strategy, detect, options)
            result = anonymizer.run(config=config, data=data)
            print(f"Output written to: {write_result(result, path)}")

        extra: dict[str, Any] = {
            "output": Annotated[
                str | None,
                cyclopts.Parameter(
                    group=INPUT_GROUP,
                    help=(
                        "Output file (.csv or .parquet) holding the input columns plus the anonymized text and the "
                        "result columns. Defaults to the source name with _anonymized (replace) or _rewritten (rewrite)."
                    ),
                ),
            ]
        }
        summary = "Run the full pipeline"
    elif verb == "preview":

        def command(
            *, data, strategy=None, detect=None, options=RunOptions(), num_records=_PREVIEW_NUM_RECORDS
        ) -> None:
            config, anonymizer = _build(mode, strategy, detect, options)
            result = anonymizer.preview(config=config, data=data, num_records=num_records)
            print(result.dataframe.to_string(max_colwidth=80))

        extra = {
            "num_records": Annotated[int, cyclopts.Parameter(group=INPUT_GROUP, help="Number of records to preview.")]
        }
        summary = "Run the pipeline on the first records for quick inspection"
    else:

        def command(*, data, strategy=None, detect=None, options=RunOptions()) -> None:
            config, anonymizer = _build(mode, strategy, detect, options)
            anonymizer.validate_config(config)
            print("Config is valid.")

        extra = {}
        summary = "Validate that the config is compatible with the model selections"

    command.__annotations__ = {
        "data": Annotated[AnonymizerInput, cyclopts.Parameter(name="*", group=INPUT_GROUP)],
        # The mode's model is a runtime value here, which a type checker cannot read as a type.
        "strategy": Annotated[MODES[mode] | None, cyclopts.Parameter(name="*", group=_mode_group(mode, MODES[mode]))],  # ty: ignore[invalid-type-form]
        "detect": Annotated[Detect | None, cyclopts.Parameter(name="*", group=DETECT_GROUP)],
        "options": Annotated[RunOptions, cyclopts.Parameter(name="*", group=RUN_GROUP)],
        **extra,
        "return": None,
    }
    label = "rewrite" if mode == "rewrite" else f"replace with {mode}"
    command.__doc__ = f"{summary} ({label})."
    command.__name__ = mode
    return _cli_error_handler(command)


for _verb, _help in (
    ("run", "Run the full anonymization pipeline (detection + replacement or rewrite)."),
    ("preview", "Run the pipeline on a subset of records for quick inspection."),
    ("validate", "Validate that the active config is compatible with model selections."),
):
    _sub = cyclopts.App(name=_verb, help=_help)
    for _mode in MODES:
        _sub.command(_command(_verb, _mode), name=_mode)
    app.command(_sub)


def _describe(error: ValidationError) -> str:
    """A validation error without the input values, which can hold the hash key."""
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or 'value'}: {item['msg']}"
        for item in error.errors(include_input=False, include_url=False)
    )


def main() -> None:
    """Entry point for the anonymizer CLI.

    cyclopts validates the config models through a ``TypeAdapter``, whose errors print each input value and ignore
    the models' ``hide_input_in_errors``. Parse errors are printed here instead, without the inputs.
    """
    try:
        app(sys.argv[1:], exit_on_error=False, print_error=False)
    except cyclopts.CycloptsError as exc:
        cause = exc.__cause__
        message = _describe(cause) if isinstance(cause, ValidationError) else str(exc)
        print(f"Error: {message}", file=sys.stderr)
        raise SystemExit(1) from None
