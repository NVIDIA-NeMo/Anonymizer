# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit local-process effects for frozen, checksummed run plans."""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Never, assert_never, cast

import httpx

from inference_service_compiler.compiler import verify_plan
from inference_service_compiler.lifecycle import (
    LAUNCH_OWNERSHIP_HEADER,
    LAUNCH_OWNERSHIP_PATH,
    LAUNCH_OWNERSHIP_PROOF_FIELD,
    LAUNCH_TOKEN_ENVIRONMENT_VARIABLE,
    launch_token_proof,
)
from inference_service_compiler.models import (
    Capability,
    CapabilityProbeReceipt,
    EntityDetection,
    Generation,
    LaunchReceipt,
    LocalProcessHandle,
    RunPlan,
    RuntimeDiagnostic,
    StatusReceipt,
    StopReceipt,
)


class RuntimeEffectError(RuntimeError):
    """A launch or probe failed with a serializable partial-effects record."""

    def __init__(self, diagnostic: RuntimeDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


@dataclass(frozen=True, slots=True)
class StopOutcome:
    """Immutable result of terminating one managed process group."""

    outcome: Literal["terminated", "forced"]
    cleanup_complete: bool


def probe_endpoint(
    plan: RunPlan,
    *,
    client: httpx.Client | None = None,
    secret_values: Mapping[str, str] | None = None,
) -> CapabilityProbeReceipt:
    """Probe one managed endpoint and record only capabilities observed at runtime."""
    verify_plan(plan)
    headers = _probe_headers(plan, secret_values or {})
    try:
        with nullcontext(client) if client is not None else httpx.Client(timeout=10) as active_client:
            models_response = active_client.get(f"{plan.endpoint.url}{plan.readiness.path}", headers=headers)
            if models_response.status_code != plan.readiness.expected_status:
                raise ValueError(
                    f"readiness probe returned status {models_response.status_code}, "
                    f"expected {plan.readiness.expected_status}"
                )
            models = _parse_models(models_response.json())
            observed = _probe_task(plan, active_client, headers)
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeEffectError(
            RuntimeDiagnostic(code="probe-failed", message=f"capability probe failed: {exc}")
        ) from exc
    return CapabilityProbeReceipt(
        plan_digest=plan.plan_digest,
        endpoint=plan.endpoint,
        observed_at=_now(),
        models=models,
        observed_capabilities=observed,
        passed=plan.served_model_name in models and set(plan.required_capabilities).issubset(observed),
    )


def launch_plan(
    plan: RunPlan,
    *,
    secret_values: dict[str, str],
    log_directory: Path,
) -> LaunchReceipt:
    """Launch a checksummed plan and record its observed handle and readiness."""
    verify_plan(plan)
    try:
        command_environment = plan.command.resolve_environment(secret_values)
    except ValueError as exc:
        raise RuntimeEffectError(RuntimeDiagnostic(code="missing-secret", message=str(exc))) from exc
    argv = plan.command.render_argv()
    environment = os.environ.copy()
    environment.update(command_environment)
    launch_token = secrets.token_urlsafe(32)
    environment[LAUNCH_TOKEN_ENVIRONMENT_VARIABLE] = launch_token
    log_directory.mkdir(parents=True, exist_ok=True)
    handle = _launch_process(plan, argv, environment, log_directory)
    probe = _probe_or_cleanup(plan, handle, secret_values, launch_token)
    return LaunchReceipt(
        plan_digest=plan.plan_digest,
        launched_at=_now(),
        shutdown_timeout_seconds=plan.spec.local.shutdown_timeout_seconds,
        handle=handle,
        probe=probe,
    )


def _probe_or_cleanup(
    plan: RunPlan,
    handle: LocalProcessHandle,
    secret_values: Mapping[str, str],
    launch_token: str,
) -> CapabilityProbeReceipt:
    try:
        probe = wait_for_readiness(
            plan,
            secret_values=secret_values,
            handle=handle,
            launch_token=launch_token,
        )
    except BaseException as exc:
        _raise_readiness_failure(exc, handle, plan.spec.local.shutdown_timeout_seconds)
    if not probe.passed:
        cleanup_complete, cleanup_error = _attempt_failed_launch_cleanup(
            handle,
            plan.spec.local.shutdown_timeout_seconds,
        )
        message = "endpoint became ready but did not satisfy required capabilities"
        if cleanup_error is not None:
            message = f"{message}; cleanup failed: {cleanup_error}"
        raise RuntimeEffectError(
            RuntimeDiagnostic(
                code="capability-mismatch",
                message=message,
                known_effects=(handle.external_id,),
                cleanup_complete=cleanup_complete,
            )
        )
    return probe


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
                    "known_effects": (handle.external_id,),
                    "cleanup_complete": cleanup_complete,
                }
            )
        ) from exc
    if isinstance(exc, Exception):
        raise RuntimeEffectError(
            RuntimeDiagnostic(
                code="unexpected-readiness-failure",
                message=f"readiness failed unexpectedly: {exc}{cleanup_suffix}",
                known_effects=(handle.external_id,),
                cleanup_complete=cleanup_complete,
            )
        ) from exc
    if cleanup_error is not None:
        exc.add_note(f"managed-process cleanup failed: {cleanup_error}")
    raise exc


def status_run(launch: LaunchReceipt) -> StatusReceipt:
    """Observe a recorded handle without changing process state."""
    state = "running" if is_handle_running(launch.handle) else "stopped"
    return StatusReceipt(
        plan_digest=launch.plan_digest,
        observed_at=_now(),
        handle=launch.handle,
        state=state,
    )


def stop_run(launch: LaunchReceipt) -> StopReceipt:
    """Stop the exact process group recorded by a launch receipt."""
    handle = launch.handle
    if not is_handle_running(handle):
        return StopReceipt(
            plan_digest=launch.plan_digest,
            stopped_at=_now(),
            handle=handle,
            outcome="already-stopped",
            cleanup_complete=True,
        )
    stop = _stop_running_handle(handle, launch.shutdown_timeout_seconds)
    return StopReceipt(
        plan_digest=launch.plan_digest,
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


def _process_identity_mismatch(handle: LocalProcessHandle, detail: str) -> RuntimeEffectError:
    return RuntimeEffectError(
        RuntimeDiagnostic(
            code="process-identity-mismatch",
            message=f"refusing to act on recorded process {handle.external_id}: {detail}",
        )
    )


def _cleanup_handle(handle: LocalProcessHandle, timeout_seconds: float) -> bool:
    if not is_handle_running(handle):
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
        running = is_handle_running(handle)
        if not running:
            break
        time.sleep(0.1)
    outcome: Literal["terminated", "forced"] = "terminated"
    if running:
        try:
            os.killpg(handle.process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            return StopOutcome(outcome="forced", cleanup_complete=True)
        running = is_handle_running(handle)
        outcome = "forced"
    return StopOutcome(outcome=outcome, cleanup_complete=not running)


def wait_for_readiness(
    plan: RunPlan,
    *,
    secret_values: Mapping[str, str] | None = None,
    handle: LocalProcessHandle | None = None,
    launch_token: str | None = None,
) -> CapabilityProbeReceipt:
    """Poll the declared readiness contract until it passes or times out."""
    deadline = time.monotonic() + plan.readiness.timeout_seconds
    last_error: RuntimeEffectError | None = None
    while time.monotonic() < deadline:
        if handle is not None and not is_handle_running(handle):
            log_hint = f"; status {handle.stderr_path}"
            raise RuntimeEffectError(
                RuntimeDiagnostic(
                    code="launch-exited",
                    message=f"managed service exited before readiness{log_hint}",
                )
            )
        try:
            if launch_token is not None:
                _probe_launch_ownership(plan, launch_token)
            receipt = probe_endpoint(plan, secret_values=secret_values)
            if receipt.passed:
                return receipt
            last_error = RuntimeEffectError(
                RuntimeDiagnostic(code="capability-mismatch", message="required capabilities were not observed")
            )
        except RuntimeEffectError as exc:
            last_error = exc
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))
    message = str(last_error) if last_error is not None else "readiness probe timed out"
    raise RuntimeEffectError(RuntimeDiagnostic(code="readiness-timeout", message=message))


def _probe_launch_ownership(plan: RunPlan, launch_token: str) -> None:
    """Require a proof that the responding server inherited this launch's token."""
    url = f"{plan.endpoint.scheme}://{plan.endpoint.host}:{plan.endpoint.port}{LAUNCH_OWNERSHIP_PATH}"
    try:
        response = httpx.get(
            url,
            headers={LAUNCH_OWNERSHIP_HEADER: launch_token},
            timeout=10,
        )
        if response.status_code != 200:
            raise ValueError(f"launch ownership probe returned status {response.status_code}")
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise TypeError("launch ownership response must be an object")
        observed_proof = payload.get(LAUNCH_OWNERSHIP_PROOF_FIELD)
        expected_proof = launch_token_proof(launch_token)
        if not isinstance(observed_proof, str) or not secrets.compare_digest(observed_proof, expected_proof):
            raise ValueError("launch ownership proof did not match")
    except (httpx.HTTPError, TypeError, ValueError, json.JSONDecodeError) as exc:
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


def _probe_task(plan: RunPlan, client: httpx.Client, headers: Mapping[str, str]) -> tuple[Capability, ...]:
    match plan.spec.task:
        case Generation():
            response = client.post(
                f"{plan.endpoint.url}/chat/completions",
                json={
                    "model": plan.served_model_name,
                    "messages": [{"role": "user", "content": "Reply with the word ready."}],
                    "max_tokens": 128,
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                        "reasoning_effort": "low",
                    },
                },
                headers=headers,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("chat completion content must be a string")
            return ("chat-completions",)
        case EntityDetection():
            response = client.post(
                f"{plan.endpoint.url}/chat/completions",
                json={
                    "model": plan.served_model_name,
                    "messages": [{"role": "user", "content": "Ada Lovelace"}],
                    "labels": ["person"],
                    "threshold": 0.1,
                },
                headers=headers,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            payload = json.loads(content)
            entities = payload["entities"]
            if not isinstance(entities, list):
                raise ValueError("detector entities must be a list")
            observed: list[Capability] = ["dynamic-labels"]
            if entities and all(isinstance(entity, dict) and {"start", "end"} <= entity.keys() for entity in entities):
                observed.append("offsets")
            if entities and all(isinstance(entity, dict) and "score" in entity for entity in entities):
                observed.append("scores")
            return tuple(observed)
        case _:
            assert_never(plan.spec.task)


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


def _probe_headers(plan: RunPlan, secret_values: Mapping[str, str]) -> dict[str, str]:
    source = plan.readiness.bearer_token_environment_variable
    if source is None:
        return {}
    if source not in secret_values:
        raise RuntimeEffectError(
            RuntimeDiagnostic(
                code="missing-secret",
                message=f"secret environment variable {source!r} is not resolved",
            )
        )
    return {"Authorization": f"Bearer {secret_values[source]}"}


def _launch_process(
    plan: RunPlan,
    argv: tuple[str, ...],
    environment: dict[str, str],
    log_directory: Path,
) -> LocalProcessHandle:
    stdout_path = log_directory / f"{plan.plan_digest}.stdout.log"
    stderr_path = log_directory / f"{plan.plan_digest}.stderr.log"
    with stdout_path.open("ab") as stdout_file, stderr_path.open("ab") as stderr_file:
        process = subprocess.Popen(
            argv,
            cwd=plan.command.working_directory,
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
        external_id=f"{process.pid}:{marker}",
        pid=process.pid,
        process_group_id=process.pid,
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


def _now() -> str:
    return datetime.now(UTC).isoformat()
