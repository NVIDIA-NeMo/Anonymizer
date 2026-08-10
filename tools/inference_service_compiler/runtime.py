# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit local-process and Docker effects for immutable run plans."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import httpx

from inference_service_compiler.compiler import verify_plan
from inference_service_compiler.models import (
    CachedModel,
    CachedModels,
    CancellationReceipt,
    Capability,
    CapabilityProbeReceipt,
    DockerHandle,
    DockerRuntime,
    EntityDetection,
    Generation,
    LaunchReceipt,
    LocalProcessHandle,
    LocalProcessRuntime,
    RunPlan,
    RuntimeDiagnostic,
    SecretEnvironmentVariable,
    StatusReceipt,
)


class RuntimeEffectError(RuntimeError):
    """A launch or probe failed with a serializable partial-effects record."""

    def __init__(self, diagnostic: RuntimeDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


def discover_cached_models(cache_root: Path) -> CachedModels:
    """List existing Hugging Face snapshots without creating or downloading files."""
    discovered: list[CachedModel] = []
    if cache_root.exists():
        for model_directory in sorted(cache_root.glob("models--*")):
            repository = model_directory.name.removeprefix("models--").replace("--", "/")
            for snapshot in sorted((model_directory / "snapshots").glob("*")):
                if snapshot.is_dir():
                    discovered.append(
                        CachedModel(
                            repository=repository,
                            revision=snapshot.name,
                            snapshot_path=str(snapshot),
                        )
                    )
    return CachedModels(cache_root=str(cache_root), models=tuple(discovered))


def default_cache_root() -> Path:
    """Resolve the Hugging Face cache location without creating it."""
    if hub_cache := os.getenv("HF_HUB_CACHE"):
        return Path(hub_cache)
    if hf_home := os.getenv("HF_HOME"):
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def probe_endpoint(
    plan: RunPlan,
    *,
    client: httpx.Client | None = None,
    secret_values: Mapping[str, str] | None = None,
) -> CapabilityProbeReceipt:
    """Probe one managed endpoint and record only capabilities observed at runtime."""
    verify_plan(plan)
    headers = _probe_headers(plan, secret_values or {})
    owns_client = client is None
    active_client = client or httpx.Client(timeout=10)
    try:
        models_response = active_client.get(plan.readiness.url, headers=headers)
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
    finally:
        if owns_client:
            active_client.close()
    return CapabilityProbeReceipt(
        plan_digest=plan.plan_digest,
        endpoint=plan.endpoint,
        observed_at=_now(),
        models=models,
        observed_capabilities=observed,
        passed=plan.expected_model in models and set(plan.required_capabilities).issubset(observed),
    )


def launch_plan(
    plan: RunPlan,
    *,
    secret_values: dict[str, str],
    log_directory: Path,
) -> LaunchReceipt:
    """Launch a verified plan and return reconnectable identity plus readiness evidence."""
    verify_plan(plan)
    resolved_secrets = _resolve_secrets(plan, secret_values)
    argv = plan.command.render_argv()
    environment = os.environ.copy()
    environment.update(plan.command.render_environment(resolve_secrets=resolved_secrets))
    log_directory.mkdir(parents=True, exist_ok=True)
    handle = _launch_handle(plan, argv, environment, log_directory)
    probe = _probe_or_cleanup(plan, handle, resolved_secrets)
    return LaunchReceipt(
        plan_digest=plan.plan_digest,
        launched_at=_now(),
        shutdown_timeout_seconds=plan.intent.lifecycle.shutdown_timeout_seconds,
        handle=handle,
        probe=probe,
    )


def _launch_handle(
    plan: RunPlan,
    argv: tuple[str, ...],
    environment: dict[str, str],
    log_directory: Path,
) -> LocalProcessHandle | DockerHandle:
    match plan.runtime:
        case LocalProcessRuntime():
            return _launch_process(plan, argv, environment, log_directory)
        case DockerRuntime():
            return _launch_docker(argv, environment)


def _probe_or_cleanup(
    plan: RunPlan,
    handle: LocalProcessHandle | DockerHandle,
    secret_values: Mapping[str, str],
) -> CapabilityProbeReceipt:
    try:
        probe = wait_for_readiness(plan, secret_values=secret_values, handle=handle)
    except RuntimeEffectError as exc:
        cleanup_complete = _cleanup_handle(handle, plan.intent.lifecycle.shutdown_timeout_seconds)
        raise RuntimeEffectError(
            exc.diagnostic.model_copy(
                update={
                    "known_effects": (handle.external_id,),
                    "cleanup_complete": cleanup_complete,
                }
            )
        ) from exc
    if not probe.passed:
        cleanup_complete = _cleanup_handle(handle, plan.intent.lifecycle.shutdown_timeout_seconds)
        raise RuntimeEffectError(
            RuntimeDiagnostic(
                code="capability-mismatch",
                message="endpoint became ready but did not satisfy required capabilities",
                known_effects=(handle.external_id,),
                cleanup_complete=cleanup_complete,
            )
        )
    return probe


def inspect_run(launch: LaunchReceipt) -> StatusReceipt:
    """Inspect a reconnectable handle without changing its state."""
    state = "running" if is_handle_running(launch.handle) else "stopped"
    return StatusReceipt(
        plan_digest=launch.plan_digest,
        observed_at=_now(),
        handle=launch.handle,
        state=state,
    )


def cancel_run(launch: LaunchReceipt) -> CancellationReceipt:
    """Stop the exact process group or container recorded by a launch receipt."""
    handle = launch.handle
    if not is_handle_running(handle):
        return CancellationReceipt(
            plan_digest=launch.plan_digest,
            canceled_at=_now(),
            handle=handle,
            outcome="already-stopped",
            cleanup_complete=True,
        )
    outcome, cleanup_complete = _stop_running_handle(handle, launch.shutdown_timeout_seconds)
    return CancellationReceipt(
        plan_digest=launch.plan_digest,
        canceled_at=_now(),
        handle=handle,
        outcome=outcome,
        cleanup_complete=cleanup_complete,
    )


def is_handle_running(handle: LocalProcessHandle | DockerHandle) -> bool:
    """Check the external identity while guarding against Linux PID reuse."""
    match handle:
        case LocalProcessHandle():
            current_marker = read_process_start_marker(handle.pid)
            if handle.start_marker is not None and current_marker != handle.start_marker:
                return False
            if read_process_state(handle.pid) == "Z":
                return False
            try:
                os.kill(handle.pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            return True
        case DockerHandle():
            completed = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", handle.container_id],
                capture_output=True,
                text=True,
                check=False,
            )
            return completed.returncode == 0 and completed.stdout.strip().lower() == "true"


def _cleanup_handle(handle: LocalProcessHandle | DockerHandle, timeout_seconds: float) -> bool:
    if not is_handle_running(handle):
        return True
    _outcome, cleanup_complete = _stop_running_handle(handle, timeout_seconds)
    return cleanup_complete


def _stop_running_handle(
    handle: LocalProcessHandle | DockerHandle,
    timeout_seconds: float,
) -> tuple[Literal["terminated", "forced"], bool]:
    match handle:
        case LocalProcessHandle():
            try:
                os.killpg(handle.process_group_id, signal.SIGTERM)
            except ProcessLookupError:
                return "terminated", True
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
                    return "forced", True
                running = is_handle_running(handle)
                outcome = "forced"
            return outcome, not running
        case DockerHandle():
            completed = subprocess.run(
                ["docker", "stop", "--time", str(int(timeout_seconds)), handle.container_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeEffectError(
                    RuntimeDiagnostic(
                        code="docker-cancel-failed",
                        message=completed.stderr.strip() or "docker stop failed",
                        known_effects=(handle.external_id,),
                        cleanup_complete=False,
                    )
                )
            return "terminated", True


def wait_for_readiness(
    plan: RunPlan,
    *,
    secret_values: Mapping[str, str] | None = None,
    handle: LocalProcessHandle | DockerHandle | None = None,
) -> CapabilityProbeReceipt:
    """Poll the declared readiness contract until it passes or times out."""
    deadline = time.monotonic() + plan.readiness.timeout_seconds
    last_error: RuntimeEffectError | None = None
    while time.monotonic() < deadline:
        if handle is not None and not is_handle_running(handle):
            log_hint = f"; inspect {handle.stderr_path}" if isinstance(handle, LocalProcessHandle) else ""
            raise RuntimeEffectError(
                RuntimeDiagnostic(
                    code="launch-exited",
                    message=f"managed service exited before readiness{log_hint}",
                )
            )
        try:
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
    match plan.intent.task:
        case Generation():
            response = client.post(
                f"{plan.endpoint.url}/chat/completions",
                json={
                    "model": plan.expected_model,
                    "messages": [{"role": "user", "content": "Reply with the word ready."}],
                    "max_tokens": 8,
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
                    "model": plan.expected_model,
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
    raise TypeError(f"unsupported task type {type(plan.intent.task)!r}")


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


def _resolve_secrets(plan: RunPlan, values: dict[str, str]) -> dict[str, str]:
    required = {
        variable.source_environment_variable
        for variable in plan.command.environment
        if isinstance(variable, SecretEnvironmentVariable)
    }
    missing = sorted(name for name in required if name not in values)
    if missing:
        raise RuntimeEffectError(
            RuntimeDiagnostic(
                code="missing-secret",
                message=f"secret environment variable {missing[0]!r} is not resolved",
            )
        )
    return {name: values[name] for name in required}


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
    suffix = marker or "unknown"
    return LocalProcessHandle(
        external_id=f"{process.pid}:{suffix}",
        pid=process.pid,
        process_group_id=process.pid,
        start_marker=marker,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def _launch_docker(argv: tuple[str, ...], environment: dict[str, str]) -> DockerHandle:
    completed = subprocess.run(argv, env=environment, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeEffectError(
            RuntimeDiagnostic(code="docker-launch-failed", message=completed.stderr.strip() or "docker run failed")
        )
    container_id = completed.stdout.strip()
    if not container_id:
        raise RuntimeEffectError(
            RuntimeDiagnostic(code="docker-launch-failed", message="docker run returned no container identity")
        )
    return DockerHandle(external_id=container_id, container_id=container_id)


def _now() -> str:
    return datetime.now(UTC).isoformat()
