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
import yaml
from pydantic import BaseModel, Field, ValidationError, create_model
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


# The input flags: ``AnonymizerInput``'s fields, each optional, since a ``--config`` file's ``input`` section can
# supply them. Generated from the model, with its descriptions; ``_input`` validates the result as an
# ``AnonymizerInput``.
_INPUT_FIELDS: dict[str, Any] = {
    name: (info.annotation | None, Field(default=None, description=info.description))
    for name, info in AnonymizerInput.model_fields.items()
}
_InputFlags: type[BaseModel] = create_model("_InputFlags", **_INPUT_FIELDS)


@dataclass(frozen=True)
class RunOptions:
    """Options that are not config: how the ``Anonymizer`` is built, telemetry and logging."""

    config: Annotated[
        Path | None,
        cyclopts.Parameter(
            help=(
                "YAML or JSON file holding an AnonymizerConfig (detect, and replace or rewrite, as "
                "AnonymizerConfig.model_validate reads it; emit_telemetry) plus an optional input section "
                "(source, text_column, id_column, data_summary). Flags given on the command line replace the "
                "file's values. The file must match the subcommand's mode and must not hold the hash key."
            )
        ),
    ] = None
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

_PREVIEW_NUM_RECORDS: int = inspect.signature(Anonymizer.preview).parameters["num_records"].default


# -- error messages: one line, named by flag, without input values ------------------------------------------------


def _env_only(model: type[BaseModel]) -> dict[str, tuple[str, ...]]:
    """The fields of ``model`` read only from the environment, with their variables."""
    return {
        name: tuple(meta.env_var)
        for name, info in model.model_fields.items()
        for meta in info.metadata
        if isinstance(meta, cyclopts.Parameter) and meta.parse is False and meta.env_var
    }


def _flag_for(argument: Any, loc: tuple[Any, ...]) -> str | None:
    """The flag for a pydantic error location under ``argument``: the child whose field path is the longest prefix
    of ``loc``. A field with no flag (the hash key) is named by its variable."""
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


def _error_root(mode: str, loc: tuple[Any, ...]) -> tuple[str, tuple[Any, ...]]:
    """The command parameter a post-parse error location falls under, and the location inside it: ``AnonymizerConfig``
    wraps the mode's model in ``replace`` (with its discriminator tag) or ``rewrite``, and ``Detect`` in ``detect``."""
    if loc and loc[0] in ("replace", "rewrite"):
        rest = loc[1:]
        return "strategy", rest[1:] if rest and rest[0] == mode else rest
    if loc and loc[0] == "detect":
        return "detect", loc[1:]
    return "strategy", loc


@functools.cache
def _command_arguments(mode: str) -> dict[str, Any]:
    """The ``<mode>`` command's top-level arguments by parameter name. Every verb shares the config parameters, so
    one verb's collection serves all three. Built once per mode, on the first post-parse error."""
    collection = app["validate"][mode].assemble_argument_collection()
    return {argument.field_info.name: argument for argument in collection if not argument.keys}


def _flag_after_parse(mode: str, loc: tuple[Any, ...]) -> str | None:
    """The flag for an error raised after parsing, such as from ``AnonymizerConfig``."""
    parameter, rest = _error_root(mode, loc)
    return _flag_for(_command_arguments(mode)[parameter], rest)


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


def _cli_error_handler(mode: str, fn: Callable[..., None]) -> Callable[..., None]:
    """Errors raised after parsing: one line on stderr, no input values, and the exit codes above."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            return fn(*args, **kwargs)
        except GraphRejection:
            raise  # a graph wiring fault keeps its traceback
        except ValidationError as exc:
            print(f"Error: {_describe(exc, lambda loc: _flag_after_parse(mode, loc))}", file=sys.stderr)
            raise SystemExit(EXIT_USAGE)
        except InvalidConfigError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(EXIT_USAGE)
        except (InvalidInputError, AnonymizerIOError, OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(EXIT_INPUT)

    return wrapper


# -- building the config and the Anonymizer ----------------------------------------------------------------------


def _given(model: BaseModel | None) -> dict[str, Any]:
    """The fields the user set on the command line (or through the environment), nested models included. cyclopts
    builds each model from only the tokens it was given, so ``model_fields_set`` holds exactly those fields and a
    flag's default never stands in for a value from the config file."""
    if model is None:
        return {}
    return {
        name: _given(value) if isinstance(value, BaseModel) else value
        for name in model.model_fields_set
        for value in (getattr(model, name),)
    }


def _merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    """``base`` with ``top`` laid over it, mapping by mapping; any other value in ``top`` replaces the file's."""
    merged = dict(base)
    for key, value in top.items():
        merged[key] = (
            _merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
        )
    return merged


_CONFIG_SECTIONS = ("input", "detect", "replace", "rewrite", "emit_telemetry")


def _read_config_file(path: Path, mode: str) -> dict[str, Any]:
    """The ``--config`` file: an ``AnonymizerConfig`` in YAML or JSON (JSON is YAML) plus an optional ``input``
    section. The file may set only this subcommand's mode, and never the hash key."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InvalidConfigError(f"{path}: cannot read the config file ({exc.strerror or exc})") from None
    except yaml.YAMLError as exc:
        # The problem and its position only: yaml's own message quotes the offending line.
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
        raise InvalidConfigError(
            f"{path}: not valid YAML or JSON{where}: {getattr(exc, 'problem', None) or 'parse error'}"
        ) from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise InvalidConfigError(f"{path}: expected a mapping with {', '.join(_CONFIG_SECTIONS)}")
    unknown = sorted(str(key) for key in data if key not in _CONFIG_SECTIONS)
    if unknown:
        raise InvalidConfigError(
            f"{path}: unknown section(s) {', '.join(unknown)}; expected {', '.join(_CONFIG_SECTIONS)}"
        )
    section, other = ("rewrite", "replace") if mode == "rewrite" else ("replace", "rewrite")
    if data.get(other) is not None:
        raise InvalidConfigError(
            f"{path}: the file sets {other}, but this command is {mode}; use the matching subcommand"
        )
    body = data.get(section)
    if body is not None and not isinstance(body, dict):
        raise InvalidConfigError(f"{path}: {section}: expected a mapping")
    if section == "replace" and body:
        kind = str(body.get("kind", mode)).lower()
        if kind != mode:
            raise InvalidConfigError(f"{path}: replace.kind is {kind}, but this command is {mode}; use run {kind}")
        if "key" in body:
            env = " or ".join(next(iter(_env_only(MODES[mode]).values()), ("ANONYMIZER_HASH_KEY",)))
            raise InvalidConfigError(
                f"{path}: replace.key is not read from a file; set {env} instead and remove the key from the file"
            )
    for name in ("input", "detect"):
        if data.get(name) is not None and not isinstance(data[name], dict):
            raise InvalidConfigError(f"{path}: {name}: expected a mapping")
    return data


def _missing_env(mode: str) -> None:
    """A field read only from the environment that is unset is named by its variable."""
    missing = [env[0] for env in _env_only(MODES[mode]).values() if not any(os.environ.get(var) for var in env)]
    if missing:
        raise InvalidConfigError(
            f"{' and '.join(missing)} is not set. run {mode} reads its secret key from it; there is no flag."
        )


def _given_at(given: dict[str, Any], loc: tuple[Any, ...]) -> bool:
    """Whether the user gave a flag for the field at ``loc``."""
    value: Any = given
    for part in loc:
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


def _input(mode: str, data: BaseModel | None, file: dict[str, Any], path: Path | None) -> AnonymizerInput:
    """The input: the file's ``input`` section with the input flags over it."""
    given = _given(data)
    merged = _merge(file.get("input") or {}, given)
    if "source" not in merged:
        raise InvalidConfigError("--source is required" + (f" (or input.source in {path})" if path else ""))

    def name_for(loc: tuple[Any, ...]) -> str | None:
        if path is not None and not _given_at(given, loc):
            return f"{path}: input.{'.'.join(map(str, loc))}"
        return _flag_for(_command_arguments(mode)["data"], loc)

    try:
        return AnonymizerInput.model_validate(merged)
    except ValidationError as exc:
        raise InvalidConfigError(_describe(exc, name_for)) from None


def _config(
    mode: str, data: BaseModel | None, strategy: BaseModel | None, detect: Detect | None, options: RunOptions
) -> tuple[AnonymizerConfig, AnonymizerInput]:
    """The config and the input: the ``--config`` file as the base, the flags the user gave laid over it field by
    field, validated once. A model the user gave no flag for comes from the file, else from its defaults. Telemetry
    is off if either the file or ``--no-emit-telemetry`` turns it off."""
    path = options.config
    file = _read_config_file(path, mode) if path is not None else {}
    section = "rewrite" if mode == "rewrite" else "replace"
    if strategy is None:
        _missing_env(mode)
        strategy = MODES[mode]()
    flags = {"strategy": _given(strategy), "detect": _given(detect)}
    merged = _merge(
        {key: value for key, value in file.items() if key != "input"},
        {section: flags["strategy"], "detect": flags["detect"]},
    )
    if section == "replace":
        merged["replace"] = {**merged["replace"], "kind": mode}

    def name_for(loc: tuple[Any, ...]) -> str | None:
        parameter, rest = _error_root(mode, loc)
        if path is not None and not _given_at(flags[parameter], rest):
            return f"{path}: {'.'.join(map(str, loc))}"
        return _flag_for(_command_arguments(mode)[parameter], rest)

    try:
        config = AnonymizerConfig.model_validate(merged)
    except ValidationError as exc:
        raise InvalidConfigError(_describe(exc, name_for)) from None
    if not options.emit_telemetry and config.emit_telemetry:
        config = config.model_copy(update={"emit_telemetry": False})
    return config, _input(mode, data, file, path)


def _configure_logging(options: RunOptions) -> None:
    if options.debug:
        configure_logging(LoggingConfig.debug())
    elif options.verbose:
        configure_logging(LoggingConfig.verbose())
    else:
        configure_logging(LoggingConfig.default())


def _anonymizer(options: RunOptions) -> Anonymizer:
    return Anonymizer(
        model_configs=options.model_configs,
        model_providers=options.model_providers,
        artifact_path=options.artifact_path,
    )


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


# -- the verbs -----------------------------------------------------------------------------------------------------


def _run(mode: str, *, data=None, strategy=None, detect=None, options=RunOptions(), output=None) -> None:
    config, data = _config(mode, data, strategy, detect, options)
    path = _output_path(data, mode, output)
    _configure_logging(options)
    result = _anonymizer(options).run(config=config, data=data)
    print(f"Output written to: {write_result(result, path)}")


def _preview(
    mode: str, *, data=None, strategy=None, detect=None, options=RunOptions(), num_records=_PREVIEW_NUM_RECORDS
) -> None:
    config, data = _config(mode, data, strategy, detect, options)
    _configure_logging(options)
    result = _anonymizer(options).preview(config=config, data=data, num_records=num_records)
    print(result.dataframe.to_string(max_colwidth=80))


def _validate(mode: str, *, data=None, strategy=None, detect=None, options=RunOptions()) -> None:
    config, _ = _config(mode, data, strategy, detect, options)
    _configure_logging(options)
    _anonymizer(options).validate_config(config)
    print("Config is valid.")


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


@dataclass(frozen=True)
class Verb:
    """One of ``run``, ``preview`` and ``validate``: its body, the parameters it adds, and its help."""

    body: Callable[..., None]
    extra: dict[str, Any]
    summary: str
    help: str


VERBS: dict[str, Verb] = {
    "run": Verb(
        _run,
        {
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
        },
        "Run the full pipeline",
        "Run the full anonymization pipeline (detection + replacement or rewrite).",
    ),
    "preview": Verb(
        _preview,
        {"num_records": Annotated[int, cyclopts.Parameter(group=INPUT_GROUP, help="Number of records to preview.")]},
        "Run the pipeline on the first records for quick inspection",
        "Run the pipeline on a subset of records for quick inspection.",
    ),
    "validate": Verb(
        _validate,
        {},
        "Validate that the config is compatible with the model selections",
        "Validate that the active config is compatible with model selections.",
    ),
}


def _mode_group(mode: str) -> cyclopts.Group:
    """The mode's help panel. A field read only from the environment has no flag, so the panel names it."""
    notes = [
        f"The {name} is read from {' or '.join(env)}; it has no flag." for name, env in _env_only(MODES[mode]).items()
    ]
    return cyclopts.Group(mode.title(), sort_key=1, help=" ".join(notes))


def _annotations(mode: str, extra: dict[str, Any]) -> dict[str, Any]:
    """The command's parameters: the input, the mode's model, ``Detect`` and ``RunOptions``, each flattened into
    flags, plus the verb's own. Set as ``__annotations__``, since the mode's model is only known at runtime."""
    return {
        # Like the mode's model, a model built at runtime, which a type checker cannot read as a type.
        "data": Annotated[_InputFlags | None, cyclopts.Parameter(name="*", group=INPUT_GROUP)],  # ty: ignore[invalid-type-form]
        # The mode's model is a runtime value here, which a type checker cannot read as a type.
        "strategy": Annotated[MODES[mode] | None, cyclopts.Parameter(name="*", group=_mode_group(mode))],  # ty: ignore[invalid-type-form]
        "detect": Annotated[Detect | None, cyclopts.Parameter(name="*", group=DETECT_GROUP)],
        "options": Annotated[RunOptions, cyclopts.Parameter(name="*", group=RUN_GROUP)],
        **extra,
        "return": None,
    }


def _command(verb: Verb, mode: str) -> Callable[..., None]:
    """The ``<verb> <mode>`` command: the verb's body with the mode bound, its keyword parameters and defaults, and
    the annotations above."""

    def command(**kwargs: Any) -> None:
        verb.body(mode, **kwargs)

    signature = inspect.signature(verb.body)
    command.__signature__ = signature.replace(parameters=list(signature.parameters.values())[1:])  # ty: ignore[unresolved-attribute]
    command.__annotations__ = _annotations(mode, verb.extra)
    command.__name__ = mode
    return _cli_error_handler(mode, command)


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

for _name, _verb in VERBS.items():
    _sub = cyclopts.App(name=_name, help=_verb.help)
    for _mode in MODES:
        _label = "rewrite" if _mode == "rewrite" else f"replace with {_mode}"
        _sub.command(_command(_verb, _mode), name=_mode, help=f"{_verb.summary} ({_label}).")
    app.command(_sub)


def main() -> None:
    """Entry point for the anonymizer CLI. Errors are printed by ``_format_error`` and ``_cli_error_handler``."""
    app(sys.argv[1:])
