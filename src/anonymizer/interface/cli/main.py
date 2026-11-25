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
from rich.console import Console
from rich.text import Text

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


# Exit codes. A usage or config error (an unknown flag, a value a model refuses, a bad config) exits 2, as cyclopts
# does for parse errors; an input or I/O error (a bad input file or id column, an unreadable path) exits 1. Any other
# error is a runtime fault and keeps its traceback.
EXIT_USAGE = 2
EXIT_INPUT = 1


def _env_only(model: type[BaseModel]) -> dict[str, tuple[str, ...]]:
    """The fields of ``model`` read only from the environment, with their variables."""
    return {
        name: tuple(meta.env_var)
        for name, info in model.model_fields.items()
        for meta in info.metadata
        if isinstance(meta, cyclopts.Parameter) and meta.parse is False and meta.env_var
    }


def _flag_for(argument: Any, loc: tuple[Any, ...]) -> str | None:
    """The flag for a pydantic error location, from the argument cyclopts was converting: the child whose field
    path is the longest prefix of ``loc``. A field with no flag (the hash key) is named by its variable."""
    best: tuple[int, str] | None = None
    stack = [argument]
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        keys = tuple(node.keys)
        if keys and tuple(str(part) for part in loc[: len(keys)]) == keys and node.name.startswith("--"):
            name = node.name if node.parse else " or ".join(node.parameter.env_var or ()) or node.name
            if best is None or len(keys) > best[0]:
                best = (len(keys), name)
    return best[1] if best else None


def _flag_for_path(mode: str, loc: tuple[Any, ...]) -> str | None:
    """The flag for an error raised after parsing, from its location in ``AnonymizerConfig`` or the mode's model."""
    parts = [str(part) for part in loc]
    model: type[BaseModel] = MODES[mode]
    if parts and parts[0] in ("replace", "rewrite"):
        parts = parts[1:]
        if parts and parts[0] == mode:  # the discriminator tag of a replace strategy
            parts = parts[1:]
    elif parts and parts[0] == "detect":
        model, parts = Detect, parts[1:]
    prefix = ""
    for part in parts:
        info = model.model_fields.get(part)
        if info is None:
            return None
        if part in _env_only(model):
            return " or ".join(_env_only(model)[part])
        nested = _nested_model(info.annotation)
        if nested is None:
            return f"--{prefix}{part.replace('_', '-')}"
        if not any(isinstance(meta, cyclopts.Parameter) and meta.name == ("*",) for meta in info.metadata):
            prefix += f"{part.replace('_', '-')}."
        model = nested
    return None


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    models = [arg for arg in get_args(annotation) if isinstance(arg, type) and issubclass(arg, BaseModel)]
    return models[0] if len(models) == 1 else None


def _describe(error: ValidationError, name_for: Callable[[tuple[Any, ...]], str | None]) -> str:
    """A validation error by flag, without the input values, which can hold the hash key."""
    parts = []
    for item in error.errors(include_input=False, include_url=False):
        where = name_for(item["loc"]) or ".".join(str(part) for part in item["loc"]) or "value"
        parts.append(f"{where}: {item['msg'].removeprefix('Value error, ')}")
    return "; ".join(parts)


def _format_error(exc: cyclopts.CycloptsError) -> Text:
    """cyclopts' parse errors, one line each. cyclopts validates the models through a ``TypeAdapter``, whose errors
    print each input value and ignore ``hide_input_in_errors``, so validation errors are described here without
    them. ``Text`` keeps rich from reading ``[REDACTED_{label}]`` as markup."""
    cause = exc.__cause__
    if isinstance(cause, ValidationError):
        argument = exc.argument
        message = _describe(cause, lambda loc: _flag_for(argument, loc) if argument is not None else None)
    else:
        message = str(exc)
    return Text(f"Error: {message}")


# ``--empty-<list>`` flags would only restate a list's empty default, so they are turned off.
app = cyclopts.App(
    help="NeMo Anonymizer CLI",
    default_parameter=cyclopts.Parameter(negative_iterable=()),
    # Field descriptions are plain text; markup rendering would drop tokens such as <LABEL>.
    help_format="plaintext",
    error_formatter=_format_error,
    # Errors are plain, unwrapped lines, like the ones the commands print.
    error_console=Console(stderr=True, soft_wrap=True, highlight=False),
)


def _cli_error_handler(mode: str, fn: Callable[..., None]) -> Callable[..., None]:
    """Errors raised after parsing: one line on stderr, no input values, and the exit codes above."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            return fn(*args, **kwargs)
        except GraphRejection:
            raise  # a graph wiring fault keeps its traceback
        except ValidationError as exc:
            print(f"Error: {_describe(exc, lambda loc: _flag_for_path(mode, loc))}", file=sys.stderr)
            raise SystemExit(EXIT_USAGE)
        except InvalidConfigError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(EXIT_USAGE)
        except (InvalidInputError, AnonymizerIOError, OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(EXIT_INPUT)

    return wrapper


def _default_strategy(mode: str) -> BaseModel:
    """The mode's model from its defaults, for a command given no flag of it. A field read only from the
    environment that is unset is named by its variable."""
    model = MODES[mode]
    missing = [env[0] for env in _env_only(model).values() if not any(os.environ.get(var) for var in env)]
    if missing:
        raise InvalidConfigError(
            f"{' and '.join(missing)} is not set. run {mode} reads its secret key from it; there is no flag."
        )
    return model()


def _build(
    mode: str, strategy: BaseModel | None, detect: Detect | None, options: RunOptions
) -> tuple[AnonymizerConfig, Anonymizer]:
    """Build the config and the Anonymizer. A model the user gave no flag for is built from its defaults."""
    if strategy is None:
        strategy = _default_strategy(mode)
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
    return _cli_error_handler(mode, command)


for _verb, _help in (
    ("run", "Run the full anonymization pipeline (detection + replacement or rewrite)."),
    ("preview", "Run the pipeline on a subset of records for quick inspection."),
    ("validate", "Validate that the active config is compatible with model selections."),
):
    _sub = cyclopts.App(name=_verb, help=_help)
    for _mode in MODES:
        _sub.command(_command(_verb, _mode), name=_mode)
    app.command(_sub)


def main() -> None:
    """Entry point for the anonymizer CLI. Errors are printed by ``_format_error`` and ``_cli_error_handler``."""
    app(sys.argv[1:])
