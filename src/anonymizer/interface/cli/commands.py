# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``run``, ``preview`` and ``validate`` commands: their bodies, the Anonymizer and logging setup, and the
output path."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import cyclopts

from anonymizer.config.anonymizer_config import AnonymizerInput
from anonymizer.engine.io.constants import SUPPORTED_IO_FORMATS
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.interface.cli._output import write_result
from anonymizer.interface.cli.config_file import _config
from anonymizer.interface.cli.errors import Arguments, _cli_error_handler
from anonymizer.interface.cli.params import (
    DETECT_GROUP,
    INPUT_GROUP,
    MODES,
    RUN_GROUP,
    RunOptions,
    _DetectFlags,
    _InputFlags,
    env_only,
)
from anonymizer.interface.errors import InvalidConfigError
from anonymizer.logging import LoggingConfig, configure_logging

_PREVIEW_NUM_RECORDS: int = inspect.signature(Anonymizer.preview).parameters["num_records"].default


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


def _run(
    mode: str, arguments: Arguments, *, data=None, strategy=None, detect=None, options=RunOptions(), output=None
) -> None:
    config, data = _config(arguments, mode, data, strategy, detect, options)
    path = _output_path(data, mode, output)
    _configure_logging(options)
    result = _anonymizer(options).run(config=config, data=data)
    print(f"Output written to: {write_result(result, path)}")


def _preview(
    mode: str,
    arguments: Arguments,
    *,
    data=None,
    strategy=None,
    detect=None,
    options=RunOptions(),
    num_records=_PREVIEW_NUM_RECORDS,
) -> None:
    config, data = _config(arguments, mode, data, strategy, detect, options)
    _configure_logging(options)
    result = _anonymizer(options).preview(config=config, data=data, num_records=num_records)
    print(result.dataframe.to_string(max_colwidth=80))


def _validate(mode: str, arguments: Arguments, *, data=None, strategy=None, detect=None, options=RunOptions()) -> None:
    config, _ = _config(arguments, mode, data, strategy, detect, options)
    _configure_logging(options)
    _anonymizer(options).validate_config(config)
    print("Config is valid.")


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
        f"The {name} is read from {' or '.join(env)}; it has no flag." for name, env in env_only(MODES[mode]).items()
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
        "detect": Annotated[_DetectFlags | None, cyclopts.Parameter(name="*", group=DETECT_GROUP)],
        "options": Annotated[RunOptions, cyclopts.Parameter(name="*", group=RUN_GROUP)],
        **extra,
        "return": None,
    }


def _command(verb: Verb, mode: str, arguments: Arguments) -> Callable[..., None]:
    """The ``<verb> <mode>`` command: the verb's body with the mode bound, its keyword parameters and defaults, and
    the annotations above."""

    def command(**kwargs: Any) -> None:
        verb.body(mode, arguments, **kwargs)

    signature = inspect.signature(verb.body)
    command.__signature__ = signature.replace(parameters=list(signature.parameters.values())[2:])  # ty: ignore[unresolved-attribute]
    command.__annotations__ = _annotations(mode, verb.extra)
    command.__name__ = mode
    return _cli_error_handler(mode, command, arguments)
