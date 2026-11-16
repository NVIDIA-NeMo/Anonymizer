# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line transport for the inference service compiler."""

from __future__ import annotations

import functools
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from tomllib import TOMLDecodeError
from typing import ParamSpec, TypeVar

import cyclopts
from pydantic import BaseModel, ValidationError

from inference_service_compiler.compiler import compile_spec, load_plan
from inference_service_compiler.errors import InferenceServiceError, RuntimeEffectError
from inference_service_compiler.models import ConnectionInfo, LaunchReceipt, RunPlan
from inference_service_compiler.profiles import load_profile
from inference_service_compiler.runtime import (
    launch_plan,
    probe_endpoint,
    status_run,
    stop_run,
    validate_plan,
)

app = cyclopts.App(help="Compile and manage local inference services from TOML profiles.")

P = ParamSpec("P")
R = TypeVar("R")


def command_errors(function: Callable[P, R]) -> Callable[P, R]:
    """Render transport and compiler errors with the standard bad-input exit code."""

    @functools.wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return function(*args, **kwargs)
        except (InferenceServiceError, OSError, TOMLDecodeError, UnicodeDecodeError, ValidationError) as exc:
            detail = exc.diagnostic.model_dump_json() if isinstance(exc, RuntimeEffectError) else str(exc)
            sys.stderr.write(f"error: {detail}\n")
            raise SystemExit(125) from exc

    return wrapped


@app.command(name="compile")
@command_errors
def compile_plan(
    *,
    profile: Path,
    output: Path | None = None,
) -> None:
    """Compile a v3 TOML profile without performing runtime effects."""
    write_json(compile_spec(load_profile(profile)), output)


@app.command
@command_errors
def validate(*, plan: Path) -> None:
    """Validate the plan in its serving interpreter without starting a server."""
    parsed = _read_plan(plan)
    validate_plan(parsed)


@app.command
@command_errors
def connection(*, plan: Path, output: Path | None = None) -> None:
    """Emit non-secret client connection inputs from a compiled plan."""
    parsed = _read_plan(plan)
    write_json(
        ConnectionInfo(url=parsed.base_url, model=parsed.served_model_name, api_key_env=parsed.spec.api_key_env),
        output,
    )


@app.command
@command_errors
def launch(
    *,
    plan: Path,
    output: Path | None = None,
    log_directory: Path = Path(".inference-service-runs"),
) -> None:
    """Launch a compiled plan and record its observed handle and readiness."""
    _validate_output_destination(output)
    parsed = _read_plan(plan)
    receipt = launch_plan(parsed, secret_values=_secret_values(parsed), log_directory=log_directory)
    try:
        write_json(receipt, output)
    except BaseException as exc:
        try:
            stop_run(receipt)
        except BaseException as cleanup_exc:
            exc.add_note(f"managed-process cleanup failed after receipt write failure: {cleanup_exc}")
        raise


@app.command
@command_errors
def probe(*, plan: Path) -> None:
    """Check that the plan's endpoint serves its model and answers a backend task request."""
    parsed = _read_plan(plan)
    probe_endpoint(parsed, secret_values=_secret_values(parsed))


@app.command(name="status")
@command_errors
def status_command(*, receipt: Path, output: Path | None = None) -> None:
    """Record the current state observed for a launch receipt handle."""
    launch_receipt = _read_receipt(receipt)
    write_json(status_run(launch_receipt), output)


@app.command
@command_errors
def stop(*, receipt: Path, output: Path | None = None) -> None:
    """Stop and clean up the process group recorded by a launch receipt."""
    launch_receipt = _read_receipt(receipt)
    write_json(stop_run(launch_receipt), output)


def _read_plan(path: Path) -> RunPlan:
    return load_plan(path.read_text(encoding="utf-8"))


def _read_receipt(path: Path) -> LaunchReceipt:
    return LaunchReceipt.model_validate_json(path.read_text(encoding="utf-8"))


def _secret_values(plan: RunPlan) -> dict[str, str]:
    source = plan.spec.api_key_env
    return {source: os.environ[source]} if source is not None and source in os.environ else {}


def write_json(value: BaseModel, output: Path | None) -> None:
    """Write one versioned transport value to a file or standard output."""
    rendered = value.model_dump_json(indent=2) + "\n"
    if output is None:
        sys.stdout.write(rendered)
    else:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output.parent,
                prefix=f".{output.name}.",
                delete=False,
            ) as temporary_file:
                temporary_file.write(rendered)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
                temporary_path = Path(temporary_file.name)
            temporary_path.replace(output)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _validate_output_destination(output: Path | None) -> None:
    """Verify that a receipt destination can create files before launching."""
    if output is None:
        return
    if output.is_dir():
        raise IsADirectoryError(output)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=True):
        pass
