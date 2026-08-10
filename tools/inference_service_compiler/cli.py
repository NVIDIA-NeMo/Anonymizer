# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line transport for the inference service compiler."""

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar

import cyclopts
from pydantic import BaseModel, ValidationError

from inference_service_compiler.compiler import CompilationError, compile_intent, load_plan
from inference_service_compiler.models import InferenceIntent, LaunchReceipt, SecretEnvironmentVariable
from inference_service_compiler.runtime import (
    RuntimeEffectError,
    cancel_run,
    default_cache_root,
    discover_cached_models,
    inspect_run,
    launch_plan,
    probe_endpoint,
)

app = cyclopts.App(help="Compile and manage local inference services from typed intent.")

P = ParamSpec("P")
R = TypeVar("R")


def command_errors(function: Callable[P, R]) -> Callable[P, R]:
    """Render transport and compiler errors with the standard bad-input exit code."""

    @functools.wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return function(*args, **kwargs)
        except (CompilationError, OSError, RuntimeEffectError, ValidationError, ValueError) as exc:
            sys.stderr.write(f"error: {exc}\n")
            raise SystemExit(125) from exc

    return wrapped


@app.command(name="compile")
@command_errors
def compile_plan(
    *,
    intent: Path,
    source_revision: str,
    output: Path | None = None,
) -> None:
    """Compile a v1 intent JSON document without performing runtime effects."""
    parsed = InferenceIntent.model_validate_json(intent.read_text(encoding="utf-8"))
    plan = compile_intent(parsed, source_revision=source_revision)
    write_json(plan, output)


@app.command
@command_errors
def launch(
    *,
    plan: Path,
    output: Path | None = None,
    log_directory: Path = Path(".inference-service-runs"),
) -> None:
    """Launch a compiled plan and write its reconnectable handle receipt."""
    parsed = load_plan(plan.read_text(encoding="utf-8"))
    required_secrets = {
        variable.source_environment_variable
        for variable in parsed.command.environment
        if isinstance(variable, SecretEnvironmentVariable)
    }
    secret_values = {name: os.environ[name] for name in required_secrets if name in os.environ}
    write_json(
        launch_plan(parsed, secret_values=secret_values, log_directory=log_directory),
        output,
    )


@app.command
@command_errors
def probe(*, plan: Path, output: Path | None = None) -> None:
    """Probe the endpoint declared by a managed plan and write capability evidence."""
    parsed = load_plan(plan.read_text(encoding="utf-8"))
    source = parsed.readiness.bearer_token_environment_variable
    secret_values = {source: os.environ[source]} if source is not None and source in os.environ else {}
    write_json(probe_endpoint(parsed, secret_values=secret_values), output)


@app.command(name="inspect")
@command_errors
def inspect_command(*, receipt: Path, output: Path | None = None) -> None:
    """Inspect the reconnectable identity in a launch receipt."""
    launch_receipt = LaunchReceipt.model_validate_json(receipt.read_text(encoding="utf-8"))
    write_json(inspect_run(launch_receipt), output)


@app.command
@command_errors
def cancel(*, receipt: Path, output: Path | None = None) -> None:
    """Cancel and clean up the managed identity in a launch receipt."""
    launch_receipt = LaunchReceipt.model_validate_json(receipt.read_text(encoding="utf-8"))
    write_json(cancel_run(launch_receipt), output)


@app.command
@command_errors
def models(*, cache_root: Path | None = None, output: Path | None = None) -> None:
    """List existing Hugging Face cache snapshots without downloading models."""
    write_json(discover_cached_models(cache_root or default_cache_root()), output)


def write_json(value: BaseModel, output: Path | None) -> None:
    """Write one versioned transport value to a file or standard output."""
    rendered = value.model_dump_json(indent=2) + "\n"
    if output is None:
        sys.stdout.write(rendered)
    else:
        output.write_text(rendered, encoding="utf-8")
