# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lifecycle management for the notebook-local GLiNER2 process."""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import platform
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import httpx
from data_designer.config.models import ModelProvider

from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.notebooks._model_config import (
    LOCAL_TOKEN_ENV,
    build_notebook_model_configuration,
    validate_notebook_model_inputs,
)
from anonymizer.notebooks.local_inference.gliner2 import MODEL_ID, MODEL_REVISION

logger = logging.getLogger(__name__)

_DEVICE_ENV = "ANONYMIZER_LOCAL_GLINER2_DEVICE"
_STARTUP_TIMEOUT_SECONDS = 300.0
_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_MAX_PORT_ATTEMPTS = 3
_SERVER_REQUIREMENTS = (
    "gliner2[local]==2.0.0",
    "fastapi>=0.115,<1",
    "huggingface-hub>=0.33,<1",
    "protobuf>=5,<7",
    "sentencepiece>=0.2,<1",
    "uvicorn>=0.30,<1",
)


@dataclass
class _LocalRuntime:
    process: subprocess.Popen[str]
    endpoint: str
    requested_device: str
    selected_device: str
    token: str
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    log_thread: threading.Thread | None = None


_runtime: _LocalRuntime | None = None
_runtime_lock = threading.RLock()
_previous_token: str | None = None
_token_environment_captured = False


def create_anonymizer(
    *,
    model_configs: str | Path | None = None,
    model_providers: list[ModelProvider] | str | Path | None = None,
    gliner_device: str = "auto",
) -> Anonymizer:
    """Return an Anonymizer backed by an owned notebook-local GLiNER2 process."""
    with _runtime_lock:
        validate_notebook_model_inputs(model_configs=model_configs, model_providers=model_providers)
        runtime = _ensure_runtime(gliner_device)
        configuration = build_notebook_model_configuration(
            model_configs=model_configs,
            model_providers=model_providers,
            endpoint=runtime.endpoint,
        )
        return Anonymizer(
            model_configs=configuration.model_configs,
            model_providers=configuration.model_providers,
        )


def stop_local_runtime() -> None:
    """Stop the owned child process; repeated calls are safe."""
    global _runtime
    with _runtime_lock:
        runtime = _runtime
        _runtime = None
        if runtime is not None and runtime.process.poll() is None:
            runtime.process.terminate()
            try:
                runtime.process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                runtime.process.kill()
                runtime.process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        _restore_token_environment()


def _ensure_runtime(requested_device: str) -> _LocalRuntime:
    global _runtime
    if _runtime is not None and _runtime.process.poll() is None:
        if _runtime.requested_device == requested_device:
            _verify_readiness(_runtime)
            return _runtime
        logger.warning(
            "Replacing the notebook GLiNER2 runtime (%s -> %s); previously returned Anonymizer instances "
            "are no longer supported.",
            _runtime.requested_device,
            requested_device,
        )
        stop_local_runtime()
    elif _runtime is not None:
        _runtime = None

    last_error: Exception | None = None
    for _ in range(_MAX_PORT_ATTEMPTS):
        runtime = _start_runtime(requested_device)
        try:
            _wait_until_ready(runtime)
        except Exception as exc:
            last_error = exc
            _stop_failed_runtime(runtime)
            continue
        _runtime = runtime
        print(
            f"GLiNER2 ready: model={MODEL_ID} revision={MODEL_REVISION} "
            f"device={runtime.selected_device} endpoint={runtime.endpoint}"
        )
        return runtime
    if last_error is None:
        raise RuntimeError("Unable to start the local GLiNER2 runtime.")
    _restore_token_environment()
    raise RuntimeError(f"Unable to start the local GLiNER2 runtime.\n{_format_log_tail(runtime)}") from last_error


def _start_runtime(requested_device: str) -> _LocalRuntime:
    server_packages = _ensure_server_environment()
    port = _find_available_port()
    token = secrets.token_urlsafe(32)
    _set_token_environment(token)
    child_environment = os.environ.copy()
    for name in list(child_environment):
        if name != LOCAL_TOKEN_ENV and _looks_sensitive(name):
            child_environment.pop(name)
    child_environment[LOCAL_TOKEN_ENV] = token
    child_environment[_DEVICE_ENV] = requested_device
    existing_python_path = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(server_packages), existing_python_path)))
    command = [
        sys.executable,
        str(_server_script()),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    try:
        process = subprocess.Popen(
            command,
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception:
        _restore_token_environment()
        raise
    runtime = _LocalRuntime(
        process=process,
        endpoint=f"http://127.0.0.1:{port}/v1",
        requested_device=requested_device,
        selected_device="starting",
        token=token,
    )
    runtime.log_thread = threading.Thread(
        target=_capture_output,
        args=(process.stdout, runtime.log_lines),
        name="gliner2-log-reader",
        daemon=True,
    )
    runtime.log_thread.start()
    return runtime


def _looks_sensitive(name: str) -> bool:
    upper = name.upper()
    return upper.endswith(("_API_KEY", "_ACCESS_TOKEN", "_AUTH_TOKEN", "_PASSWORD", "_SECRET")) or upper in {
        "HF_TOKEN",
    }


def _server_script() -> Path:
    return Path(__file__).with_name("local_inference") / "gliner2" / "server.py"


def _ensure_server_environment() -> Path:
    """Create a cached dependency-isolated import path for native GLiNER2."""
    identity = "\n".join((*_SERVER_REQUIREMENTS, sys.version, platform.platform()))
    cache_key = hashlib.sha256(identity.encode()).hexdigest()[:16]
    cache_root = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "nemo-anonymizer"
    environment = cache_root / f"gliner2-{cache_key}"
    packages = environment / "site-packages"
    marker = environment / ".ready"
    if packages.is_dir() and marker.is_file() and marker.read_text().strip() == identity:
        return packages

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="gliner2-build-", dir=cache_root))
    try:
        print("Preparing the isolated GLiNER2 notebook runtime (first start only)...")
        temporary_packages = temporary / "site-packages"
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-warn-conflicts",
                "--target",
                str(temporary_packages),
                *_SERVER_REQUIREMENTS,
            ]
        )
        (temporary / ".ready").write_text(identity)
        if environment.exists():
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, environment)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return environment / "site-packages"


def run_development_server() -> None:
    """Replace this process with the dependency-isolated development server."""
    server_packages = _ensure_server_environment()
    environment = os.environ.copy()
    for name in list(environment):
        if name != LOCAL_TOKEN_ENV and _looks_sensitive(name):
            environment.pop(name)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(server_packages), environment.get("PYTHONPATH"))))
    command = [sys.executable, str(_server_script()), *sys.argv[1:]]
    os.execve(sys.executable, command, environment)


def _capture_output(stream: IO[str] | None, lines: deque[str]) -> None:
    if stream is None:
        return
    for line in stream:
        lines.append(line.rstrip())


def _wait_until_ready(runtime: _LocalRuntime) -> None:
    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if runtime.process.poll() is not None:
            raise RuntimeError(f"Local GLiNER2 exited during startup.\n{_format_log_tail(runtime)}")
        try:
            metadata = _read_metadata(runtime)
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            time.sleep(0.25)
            continue
        runtime.selected_device = str(metadata["device"])
        return
    raise RuntimeError(f"Timed out waiting for local GLiNER2.\n{_format_log_tail(runtime)}") from last_error


def _verify_readiness(runtime: _LocalRuntime) -> None:
    metadata = _read_metadata(runtime)
    runtime.selected_device = str(metadata["device"])


def _read_metadata(runtime: _LocalRuntime) -> dict[str, object]:
    response = httpx.get(
        f"{runtime.endpoint}/models",
        headers={"Authorization": f"Bearer {runtime.token}"},
        timeout=2.0,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ValueError("Local GLiNER2 returned malformed model metadata.")
    metadata = data[0]
    if metadata.get("id") != MODEL_ID or metadata.get("revision") != MODEL_REVISION:
        raise ValueError("Local endpoint did not report the expected pinned GLiNER2 model.")
    return metadata


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _stop_failed_runtime(runtime: _LocalRuntime) -> None:
    if runtime.process.poll() is None:
        runtime.process.terminate()
        try:
            runtime.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            runtime.process.kill()
            runtime.process.wait(timeout=2.0)


def _format_log_tail(runtime: _LocalRuntime) -> str:
    return "\n".join(runtime.log_lines) or "No child-process output was captured."


def _restore_token_environment() -> None:
    global _previous_token, _token_environment_captured
    if not _token_environment_captured:
        return
    if _previous_token is None:
        os.environ.pop(LOCAL_TOKEN_ENV, None)
    else:
        os.environ[LOCAL_TOKEN_ENV] = _previous_token
    _previous_token = None
    _token_environment_captured = False


def _set_token_environment(token: str) -> None:
    global _previous_token, _token_environment_captured
    if not _token_environment_captured:
        _previous_token = os.environ.get(LOCAL_TOKEN_ENV)
        _token_environment_captured = True
    os.environ[LOCAL_TOKEN_ENV] = token


atexit.register(stop_local_runtime)
