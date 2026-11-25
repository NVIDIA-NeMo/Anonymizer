# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI error messages: one line, named by the flag the user typed, without input values, with the exit codes in
``params``."""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cyclopts
from pydantic import ValidationError
from rich.text import Text

from anonymizer.engine.graph_core.kernel import GraphRejection
from anonymizer.interface.cli.params import EXIT_INPUT, EXIT_USAGE
from anonymizer.interface.errors import AnonymizerIOError, InvalidConfigError, InvalidInputError

# A command's top-level cyclopts arguments by parameter name (``data``, ``strategy``, ``detect``, ``options``), as
# ``app`` assembles them. Passed in by the caller, so this module does not import the app.
Arguments = Callable[[], dict[str, Any]]


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
    if loc and loc[0] == "input":
        return "data", loc[1:]
    return "strategy", loc


def _error_root(mode: str, loc: tuple[Any, ...]) -> tuple[str, tuple[Any, ...]]:
    """The command parameter a post-parse error location falls under, and the location inside it: ``AnonymizerConfig``
    wraps the mode's model in ``replace`` (with its discriminator tag) or ``rewrite``, and ``Detect`` in ``detect``."""
    if loc and loc[0] in ("replace", "rewrite"):
        rest = loc[1:]
        return "strategy", rest[1:] if rest and rest[0] == mode else rest
    if loc and loc[0] == "detect":
        return "detect", loc[1:]
    if loc and loc[0] == "input":
        return "data", loc[1:]
    return "strategy", loc


def _given_at(given: dict[str, Any], loc: tuple[Any, ...]) -> bool:
    """Whether the user gave a flag for the field at ``loc``."""
    value: Any = given
    for part in loc:
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


def _flag_after_parse(
    arguments: Arguments,
    loc: tuple[Any, ...],
    mode: str,
    given: dict[str, dict[str, Any]] | None = None,
    path: Path | None = None,
) -> str | None:
    """The name for an error raised after parsing, such as from ``AnonymizerConfig``: its flag, or, for a value
    that came from the ``--config`` file at ``path`` rather than from a flag in ``given``, the file and field path."""
    parameter, rest = _error_root(mode, loc)
    if path is not None and given is not None and not _given_at(given.get(parameter, {}), rest):
        return f"{path}: {'.'.join(map(str, loc))}"
    return _flag_for(arguments()[parameter], rest)


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


def _cli_error_handler(mode: str, fn: Callable[..., None], arguments: Arguments | None = None) -> Callable[..., None]:
    """Errors raised after parsing: one line on stderr, no input values, and the exit codes above."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            return fn(*args, **kwargs)
        except GraphRejection:
            raise  # a graph wiring fault keeps its traceback
        except ValidationError as exc:
            name_for = (
                (lambda loc: _flag_after_parse(arguments, loc, mode)) if arguments is not None else (lambda loc: None)
            )
            print(f"Error: {_describe(exc, name_for)}", file=sys.stderr)
            raise SystemExit(EXIT_USAGE)
        except InvalidConfigError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(EXIT_USAGE)
        except (InvalidInputError, AnonymizerIOError, OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(EXIT_INPUT)

    return wrapper
