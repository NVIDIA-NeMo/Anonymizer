# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit local-process effects for frozen, checksummed run plans."""

from __future__ import annotations

import os
import secrets
import signal
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Never, cast

import httpx

from inference_service_compiler.backends import traits
from inference_service_compiler.compiler import InferenceServiceError, verify_plan
from inference_service_compiler.launch_ownership import (
    LAUNCH_OWNERSHIP_HEADER,
    LAUNCH_OWNERSHIP_PATH,
    LAUNCH_OWNERSHIP_PROOF_FIELD,
    LAUNCH_TOKEN_ENVIRONMENT_VARIABLE,
    launch_token_proof,
)
from inference_service_compiler.models import (
    LaunchReceipt,
    LocalProcessHandle,
    RunPlan,
    RuntimeDiagnostic,
    StatusReceipt,
    StopReceipt,
)


class RuntimeEffectError(InferenceServiceError, RuntimeError):
    """A launch or probe failed with a serializable partial-effects record."""

    def __init__(self, diagnostic: RuntimeDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


@dataclass(frozen=True, slots=True)
class StopOutcome:
    """Immutable result of terminating one managed process group."""

    outcome: Literal["terminated", "forced"]
    cleanup_complete: bool


def validate_plan(plan: RunPlan) -> None:
    """Validate a verified plan in its configured interpreter without starting the service."""
    verify_plan(plan)
    result = subprocess.run(
        traits(plan.spec).validate_argv,
        env=_child_environment(),
        check=False,
    )
    if result.returncode:
        raise RuntimeEffectError(
            RuntimeDiagnostic(
                code="validation-failed", message=f"serving interpreter validation failed (exit {result.returncode})"
            )
        )


def probe_endpoint(
    plan: RunPlan,
    *,
    secret_values: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> None:
    """Require the endpoint to list the served model and answer one backend task request."""
    verify_plan(plan)
    headers = _probe_headers(plan, secret_values or {})
    try:
        with httpx.Client(headers=headers, timeout=10, transport=transport) as client:
            models_response = client.get(f"{plan.base_url}/models")
            if models_response.status_code != 200:
                raise ValueError(f"readiness probe returned status {models_response.status_code}, expected 200")
            models = _parse_models(models_response.json())
            if plan.served_model_name not in models:
                raise ValueError(f"endpoint serves {list(models)}, not {plan.served_model_name!r}")
            traits(plan.spec).probe(client, plan.base_url)
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeEffectError(RuntimeDiagnostic(code="probe-failed", message=f"probe failed: {exc}")) from exc


def launch_plan(
    plan: RunPlan,
    *,
    secret_values: Mapping[str, str],
    log_directory: Path,
) -> LaunchReceipt:
    """Launch a verified plan and record its observed handle once it is ready."""
    verify_plan(plan)
    backend = traits(plan.spec)
    api_key = _resolve_api_key(plan, secret_values)
    environment = _child_environment() | dict(backend.child_environment)
    if api_key is not None:
        environment[backend.api_key_child_env] = api_key
    launch_token = secrets.token_urlsafe(32)
    environment[LAUNCH_TOKEN_ENVIRONMENT_VARIABLE] = launch_token
    log_directory.mkdir(parents=True, exist_ok=True)
    handle = _launch_process(plan, environment, log_directory)
    try:
        wait_for_readiness(plan, secret_values=secret_values, handle=handle, launch_token=launch_token)
    except BaseException as exc:
        _raise_readiness_failure(exc, handle, plan.spec.shutdown_timeout_seconds)
    return LaunchReceipt(plan=plan, launched_at=_now(), handle=handle)


def _child_environment() -> dict[str, str]:
    """Keep the serving interpreter's imports isolated from the controller's environment."""
    environment = os.environ.copy()
    for variable in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(variable, None)
    return environment


def _attempt_failed_launch_cleanup(handle: LocalProcessHandle, timeout_seconds: float) -> tuple[bool, str | None]:
    try:
        return _cleanup_handle(handle, timeout_seconds), None
    except Exception as exc:
        return False, str(exc)


def _raise_readiness_failure(
    exc: BaseException,
    handle: LocalProcessHandle,
    timeout_seconds: float,
) -> Never:
    cleanup_complete, cleanup_error = _attempt_failed_launch_cleanup(handle, timeout_seconds)
    cleanup_suffix = f"; cleanup failed: {cleanup_error}" if cleanup_error is not None else ""
    if isinstance(exc, RuntimeEffectError):
        raise RuntimeEffectError(
            exc.diagnostic.model_copy(
                update={
                    "message": f"{exc.diagnostic.message}{cleanup_suffix}",
                    "known_effects": (handle.label,),
                    "cleanup_complete": cleanup_complete,
                }
            )
        ) from exc
    if isinstance(exc, Exception):
        raise RuntimeEffectError(
            RuntimeDiagnostic(
                code="unexpected-readiness-failure",
                message=f"readiness failed unexpectedly: {exc}{cleanup_suffix}",
                known_effects=(handle.label,),
                cleanup_complete=cleanup_complete,
            )
        ) from exc
    if cleanup_error is not None:
        exc.add_note(f"managed-process cleanup failed: {cleanup_error}")
    raise exc


def status_run(launch: LaunchReceipt) -> StatusReceipt:
    """Observe a recorded handle without changing process state."""
    state = "running" if is_process_group_active(launch.handle) else "stopped"
    return StatusReceipt(
        plan_digest=launch.plan.plan_digest,
        observed_at=_now(),
        handle=launch.handle,
        state=state,
    )


def stop_run(launch: LaunchReceipt) -> StopReceipt:
    """Stop the exact process group recorded by a launch receipt."""
    handle = launch.handle
    if not is_process_group_active(handle):
        return StopReceipt(
            plan_digest=launch.plan.plan_digest,
            stopped_at=_now(),
            handle=handle,
            outcome="already-stopped",
            cleanup_complete=True,
        )
    stop = _stop_running_handle(handle, launch.plan.spec.shutdown_timeout_seconds)
    return StopReceipt(
        plan_digest=launch.plan.plan_digest,
        stopped_at=_now(),
        handle=handle,
        outcome=stop.outcome,
        cleanup_complete=stop.cleanup_complete,
    )


def is_handle_running(handle: LocalProcessHandle) -> bool:
    """Check the recorded process identity without following a reused PID or group."""
    current_marker = read_process_start_marker(handle.pid)
    if current_marker is None:
        return False
    if current_marker != handle.start_marker:
        raise _process_identity_mismatch(
            handle,
            f"PID {handle.pid} now has start marker {current_marker!r}, expected {handle.start_marker!r}",
        )
    if read_process_state(handle.pid) == "Z":
        return False
    try:
        current_process_group_id = os.getpgid(handle.pid)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise RuntimeEffectError(
            RuntimeDiagnostic(
                code="process-identity-unavailable",
                message=f"cannot verify the process group for PID {handle.pid}: {exc}",
            )
        ) from exc
    if current_process_group_id != handle.process_group_id:
        raise _process_identity_mismatch(
            handle,
            f"PID {handle.pid} now belongs to process group {current_process_group_id}, "
            f"expected {handle.process_group_id}",
        )
    try:
        os.kill(handle.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_process_group_active(handle: LocalProcessHandle) -> bool:
    """Report whether the leader or any worker in the recorded process group remains.

    The leader's start marker and group are verified first, so a reused PID still
    raises. Once the leader is gone, a surviving group is still the recorded one:
    the kernel does not reissue a PID while a group with that ID exists.
    """
    return is_handle_running(handle) or _is_process_group_running(handle.process_group_id)


def _process_identity_mismatch(handle: LocalProcessHandle, detail: str) -> RuntimeEffectError:
    return RuntimeEffectError(
        RuntimeDiagnostic(
            code="process-identity-mismatch",
            message=f"refusing to act on recorded process {handle.label}: {detail}",
        )
    )


def _is_process_group_running(process_group_id: int) -> bool:
    """Return whether any process remains in the recorded process group."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return not _group_members_are_all_zombies(process_group_id)


def _group_members_are_all_zombies(process_group_id: int) -> bool:
    """Return whether /proc shows members of the group and every one is a zombie.

    ``killpg(pgid, 0)`` succeeds while a zombie remains, but a zombie holds no GPU
    memory and cannot be signaled away. When /proc yields no member, the
    ``killpg`` answer stands.
    """
    states: list[str] = []
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return False
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            payload = (entry / "stat").read_text(encoding="utf-8")
        except OSError:
            continue
        parsed = _parse_process_group_stat(payload)
        if parsed is not None and parsed[1] == process_group_id:
            states.append(parsed[0])
    return bool(states) and all(state in {"Z", "X"} for state in states)


def _parse_process_group_stat(payload: str) -> tuple[str, int] | None:
    _prefix, separator, suffix = payload.rpartition(")")
    fields = suffix.strip().split() if separator else []
    if len(fields) < 3 or not fields[2].lstrip("-").isdigit():
        return None
    return fields[0], int(fields[2])


def _cleanup_handle(handle: LocalProcessHandle, timeout_seconds: float) -> bool:
    if not is_process_group_active(handle):
        return True
    return _stop_running_handle(handle, timeout_seconds).cleanup_complete


def _stop_running_handle(
    handle: LocalProcessHandle,
    timeout_seconds: float,
) -> StopOutcome:
    try:
        os.killpg(handle.process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return StopOutcome(outcome="terminated", cleanup_complete=True)
    deadline = time.monotonic() + timeout_seconds
    running = True
    while time.monotonic() < deadline:
        running = _is_process_group_running(handle.process_group_id)
        if not running:
            break
        time.sleep(0.1)
    outcome: Literal["terminated", "forced"] = "terminated"
    if running:
        try:
            os.killpg(handle.process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            return StopOutcome(outcome="forced", cleanup_complete=True)
        running = _is_process_group_running(handle.process_group_id)
        outcome = "forced"
    return StopOutcome(outcome=outcome, cleanup_complete=not running)


def wait_for_readiness(
    plan: RunPlan,
    *,
    secret_values: Mapping[str, str] | None = None,
    handle: LocalProcessHandle | None = None,
    launch_token: str | None = None,
) -> None:
    """Poll launch ownership and the backend probe until both pass or the startup timeout ends."""
    deadline = time.monotonic() + plan.spec.startup_timeout_seconds
    last_error: RuntimeEffectError | None = None
    while time.monotonic() < deadline:
        if handle is not None and not is_handle_running(handle):
            raise RuntimeEffectError(
                RuntimeDiagnostic(
                    code="launch-exited",
                    message=f"managed service exited before readiness; see {handle.stderr_path}",
                )
            )
        try:
            if launch_token is not None:
                _probe_launch_ownership(plan, launch_token, secret_values)
            probe_endpoint(plan, secret_values=secret_values)
            return
        except RuntimeEffectError as exc:
            last_error = exc
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))
    message = str(last_error) if last_error is not None else "readiness probe timed out"
    raise RuntimeEffectError(RuntimeDiagnostic(code="readiness-timeout", message=message))


def _probe_launch_ownership(
    plan: RunPlan,
    launch_token: str,
    secret_values: Mapping[str, str] | None = None,
) -> None:
    """Require a proof that the responding server inherited this launch's token."""
    host, port = plan.endpoint
    headers = _probe_headers(plan, secret_values or {})
    headers[LAUNCH_OWNERSHIP_HEADER] = launch_token
    try:
        response = httpx.get(f"http://{host}:{port}{LAUNCH_OWNERSHIP_PATH}", headers=headers, timeout=10)
        if response.status_code != 200:
            raise ValueError(f"launch ownership probe returned status {response.status_code}")
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise TypeError("launch ownership response must be an object")
        observed_proof = payload.get(LAUNCH_OWNERSHIP_PROOF_FIELD)
        expected_proof = launch_token_proof(launch_token)
        if not isinstance(observed_proof, str) or not secrets.compare_digest(observed_proof, expected_proof):
            raise ValueError("launch ownership proof did not match")
    except (httpx.HTTPError, TypeError, ValueError) as exc:
        raise RuntimeEffectError(
            RuntimeDiagnostic(code="launch-ownership-not-observed", message=f"launch ownership probe failed: {exc}")
        ) from exc


def read_process_start_marker(pid: int) -> str | None:
    """Read Linux process start ticks to disambiguate PID reuse when available."""
    stat = _read_process_stat(pid)
    return stat[1] if stat is not None else None


def read_process_state(pid: int) -> str | None:
    """Read the Linux process state so zombies count as stopped."""
    stat = _read_process_stat(pid)
    return stat[0] if stat is not None else None


def _read_process_stat(pid: int) -> tuple[str, str] | None:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        payload = stat_path.read_text(encoding="utf-8")
    except OSError:
        return None
    return _parse_process_stat(payload)


def _parse_process_stat(payload: str) -> tuple[str, str] | None:
    _prefix, separator, suffix = payload.rpartition(")")
    fields = suffix.strip().split() if separator else []
    return (fields[0], fields[19]) if len(fields) > 19 else None


def _parse_models(payload: object) -> tuple[str, ...]:
    if not isinstance(payload, Mapping):
        raise ValueError("models response must contain a data list")
    data = cast(Mapping[str, object], payload).get("data")
    if not isinstance(data, list):
        raise ValueError("models response must contain a data list")
    models: list[str] = []
    for item in cast(list[object], data):
        if isinstance(item, Mapping):
            record = cast(Mapping[str, object], item)
            if "id" in record:
                models.append(str(record["id"]))
    return tuple(models)


def _resolve_api_key(plan: RunPlan, secret_values: Mapping[str, str]) -> str | None:
    source = plan.spec.api_key_env
    if source is None:
        return None
    if source not in secret_values:
        raise RuntimeEffectError(
            RuntimeDiagnostic(code="missing-secret", message=f"secret environment variable {source!r} is not resolved")
        )
    return secret_values[source]


def _probe_headers(plan: RunPlan, secret_values: Mapping[str, str]) -> dict[str, str]:
    api_key = _resolve_api_key(plan, secret_values)
    return {} if api_key is None else {"Authorization": f"Bearer {api_key}"}


def _launch_process(
    plan: RunPlan,
    environment: dict[str, str],
    log_directory: Path,
) -> LocalProcessHandle:
    stdout_path = log_directory / f"{plan.plan_digest}.stdout.log"
    stderr_path = log_directory / f"{plan.plan_digest}.stderr.log"
    with stdout_path.open("ab") as stdout_file, stderr_path.open("ab") as stderr_file:
        process = subprocess.Popen(
            plan.argv,
            env=environment,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )
    marker = read_process_start_marker(process.pid)
    if marker is None:
        _terminate_unmarked_launch(process)
        raise RuntimeEffectError(
            RuntimeDiagnostic(
                code="missing-process-start-marker",
                message="cannot record a managed process without a Linux start marker",
                known_effects=(str(process.pid),),
                cleanup_complete=process.poll() is not None,
            )
        )
    return LocalProcessHandle(
        pid=process.pid,
        start_marker=marker,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def _terminate_unmarked_launch(process: subprocess.Popen[bytes]) -> None:
    """Make a bounded cleanup attempt when a child cannot be recorded."""
    for signal_to_send in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signal_to_send)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            continue
        return


def _now() -> datetime:
    return datetime.now(UTC)
