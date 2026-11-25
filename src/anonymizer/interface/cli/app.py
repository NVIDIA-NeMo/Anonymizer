# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cyclopts ``App``: its error formatter, and one ``<verb> <mode>`` command per verb and mode."""

from __future__ import annotations

import functools
from typing import Any

import cyclopts
from rich.console import Console

from anonymizer.interface.cli.commands import VERBS, _command
from anonymizer.interface.cli.errors import _format_error
from anonymizer.interface.cli.params import MODES


@functools.cache
def _command_arguments(mode: str) -> dict[str, Any]:
    """The ``<mode>`` command's top-level arguments by parameter name. Every verb shares the config parameters, so
    one verb's collection serves all three. Built once per mode, on the first post-parse error; passed to the
    commands as a callable, so ``errors`` need not import the app."""
    collection = app["validate"][mode].assemble_argument_collection()
    return {argument.field_info.name: argument for argument in collection if not argument.keys}


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
        _sub.command(
            _command(_verb, _mode, functools.partial(_command_arguments, _mode)),
            name=_mode,
            help=f"{_verb.summary} ({_label}).",
        )
    app.command(_sub)
