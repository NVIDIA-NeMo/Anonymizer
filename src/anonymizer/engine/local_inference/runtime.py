# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lifecycle management for the native GLiNER2 process."""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
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
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, NamedTuple

import httpx

from anonymizer.engine.local_inference.gliner2 import MODEL_ID, MODEL_REVISION
from anonymizer.engine.local_inference.gliner2.contract import (
    AttentionRequest,
    DeviceRequest,
    DtypeRequest,
    ModelCard,
    ModelList,
)

LOCAL_TOKEN_ENV = "ANONYMIZER_LOCAL_GLINER2_TOKEN"

logger = logging.getLogger(__name__)

_DEVICE_ENV = "ANONYMIZER_LOCAL_GLINER2_DEVICE"
_DTYPE_ENV = "ANONYMIZER_LOCAL_GLINER2_DTYPE"
_ATTENTION_ENV = "ANONYMIZER_LOCAL_GLINER2_ATTENTION"
_STARTUP_TIMEOUT_SECONDS = 300.0
_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_MAX_PORT_ATTEMPTS = 3
_CHILD_SECRET_ALLOWLIST = {LOCAL_TOKEN_ENV, "HF_TOKEN"}
_SERVER_REQUIREMENTS = (
    "gliner2[local,disentangled-flash]==2.1.0",
    "fastapi>=0.115,<1",
    "huggingface-hub>=0.33,<1",
    "protobuf>=5,<7",
    "sentencepiece>=0.2,<1",
    "uvicorn>=0.30,<1",
)


class _LogTail:
    """The last lines of the child's combined output, read by a daemon thread."""

    def __init__(self) -> None:
        self._lines: deque[str] = deque(maxlen=200)
        self._thread: threading.Thread | None = None

    def follow(self, stream: IO[str] | None) -> None:
        self._thread = threading.Thread(target=self._read, args=(stream,), name="gliner2-log-reader", daemon=True)
        self._thread.start()

    def _read(self, stream: IO[str] | None) -> None:
        if stream is None:
            return
        for line in stream:
            self._lines.append(line.rstrip())

    def join(self) -> None:
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    def text(self) -> str:
        self.join()
        return "\n".join(self._lines) or "No child-process output was captured."


class NativeRequest(NamedTuple):
    """What a caller asked of the native runtime; a running runtime is reused only for an equal request."""

    device: DeviceRequest = "auto"
    dtype: DtypeRequest = "auto"
    attention: AttentionRequest = "auto"


@dataclass(frozen=True)
class _LocalRuntime:
    process: subprocess.Popen[str]
    endpoint: str
    request: NativeRequest
    token: str
    log: _LogTail = field(default_factory=_LogTail)


@dataclass(frozen=True)
class _SavedVariable:
    """An environment variable's value before this module overwrote it."""

    name: str
    previous: str | None

    @classmethod
    def capture(cls, name: str) -> _SavedVariable:
        return cls(name, os.environ.get(name))

    def restore(self) -> None:
        if self.previous is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = self.previous


class _RuntimeRegistry:
    """The one native runtime of this process, how many contexts hold it, and the saved token variable."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.runtime: _LocalRuntime | None = None
        self.context_count = 0
        self.context_owned = False
        self._saved_token: _SavedVariable | None = None

    @contextmanager
    def singleton(self, request: NativeRequest) -> Iterator[_LocalRuntime]:
        """Get the persistent singleton, respecting an active scoped runtime."""
        with self.lock:
            if self.context_count and self.context_owned:
                raise RuntimeError("Use the native runtime session to construct clients inside its context.")
            yield self.ensure(request)

    @contextmanager
    def borrow(self, request: NativeRequest) -> Iterator[_LocalRuntime]:
        """Borrow a singleton or own a runtime until the last nested context exits."""
        with self.lock:
            previous = self.runtime
            if (
                not self.context_count
                and previous is not None
                and previous.request != request
                and previous.process.poll() is None
            ):
                raise RuntimeError("Cannot borrow a native runtime with a different device, precision or attention.")
            runtime = self.ensure(request)
            if self.context_count == 0:
                self.context_owned = runtime is not previous
            self.context_count += 1
        try:
            yield runtime
        finally:
            with self.lock:
                self.context_count -= 1
                if self.context_count == 0 and self.context_owned:
                    self.context_owned = False
                    self.stop()

    def stop(self) -> None:
        """Stop the owned child process; repeated calls are safe."""
        with self.lock:
            if self.context_count:
                raise RuntimeError("Cannot stop a native runtime while a context is active.")
            self.shutdown()

    def shutdown(self) -> None:
        """Stop the process even if an active context survives interpreter shutdown."""
        with self.lock:
            runtime, self.runtime = self.runtime, None
            try:
                if runtime is not None:
                    _stop_process(runtime, timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            finally:
                self.restore_token()

    def ensure(self, request: NativeRequest) -> _LocalRuntime:
        """Reuse the running runtime, or replace it, or start one."""
        runtime = self.runtime
        if self.context_count:
            if runtime is None or runtime.process.poll() is not None or runtime.request != request:
                raise RuntimeError(
                    "Cannot replace a native runtime while a context is active, including after the process exits."
                )
            _verify_readiness(runtime)
            return runtime
        if runtime is not None and runtime.process.poll() is None:
            if runtime.request == request:
                _verify_readiness(runtime)
                return runtime
            logger.warning(
                "Replacing the native GLiNER2 runtime (%s -> %s); previously returned Anonymizer instances "
                "are no longer supported.",
                tuple(runtime.request),
                tuple(request),
            )
            self.stop()
        elif runtime is not None:
            self.runtime = None
        self.runtime = self._start_ready(request)
        return self.runtime

    def _start_ready(self, request: NativeRequest) -> _LocalRuntime:
        """Start a child, retrying on a new port when it does not become ready."""
        last_error: Exception | None = None
        runtime: _LocalRuntime | None = None
        for _ in range(_MAX_PORT_ATTEMPTS):
            try:
                runtime = _start_runtime(request)
            except BaseException:
                self.restore_token()
                raise
            try:
                card = _wait_until_ready(runtime)
            except (KeyboardInterrupt, SystemExit):
                try:
                    _stop_process(runtime, timeout=2.0)
                finally:
                    self.restore_token()
                raise
            except Exception as exc:
                last_error = exc
                _stop_process(runtime, timeout=2.0)
                continue
            print(
                f"GLiNER2 ready: model={MODEL_ID} revision={MODEL_REVISION} "
                f"device={card.device} endpoint={runtime.endpoint}"
            )
            return runtime
        self.restore_token()
        if runtime is None or last_error is None:
            raise RuntimeError("Unable to start the local GLiNER2 runtime.")
        raise RuntimeError(f"Unable to start the local GLiNER2 runtime.\n{runtime.log.text()}") from last_error

    def set_token(self, token: str) -> None:
        """Expose ``token`` to this process, remembering the original value once."""
        if self._saved_token is None:
            self._saved_token = _SavedVariable.capture(LOCAL_TOKEN_ENV)
        os.environ[LOCAL_TOKEN_ENV] = token

    def restore_token(self) -> None:
        if self._saved_token is not None:
            self._saved_token.restore()
            self._saved_token = None


_registry = _RuntimeRegistry()


def singleton_runtime(
    requested_device: DeviceRequest = "auto",
    requested_dtype: DtypeRequest = "auto",
    requested_attention: AttentionRequest = "auto",
) -> AbstractContextManager[_LocalRuntime]:
    """Get the persistent singleton, respecting an active scoped runtime."""
    return _registry.singleton(NativeRequest(requested_device, requested_dtype, requested_attention))


def runtime_context(
    requested_device: DeviceRequest = "auto",
    requested_dtype: DtypeRequest = "auto",
    requested_attention: AttentionRequest = "auto",
) -> AbstractContextManager[_LocalRuntime]:
    """Borrow a singleton or own a runtime until the last nested context exits."""
    return _registry.borrow(NativeRequest(requested_device, requested_dtype, requested_attention))


def stop_local_runtime() -> None:
    """Stop the owned child process; repeated calls are safe."""
    _registry.stop()


def _start_runtime(request: NativeRequest) -> _LocalRuntime:
    server_packages = _ensure_server_environment()
    listener = _reserve_listener()
    port = int(listener.getsockname()[1])
    token = secrets.token_urlsafe(32)
    _registry.set_token(token)
    child_environment = _child_environment()
    child_environment[LOCAL_TOKEN_ENV] = token
    child_environment[_DEVICE_ENV] = request.device
    # An explicit request overrides the inherited variable; "auto" leaves it in effect.
    if request.dtype != "auto":
        child_environment[_DTYPE_ENV] = request.dtype
    if request.attention != "auto":
        child_environment[_ATTENTION_ENV] = request.attention
    existing_python_path = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(server_packages), existing_python_path)))
    command = [
        sys.executable,
        str(_server_script()),
        "--fd",
        str(listener.fileno()),
    ]
    try:
        process = subprocess.Popen(
            command,
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            pass_fds=(listener.fileno(),),
        )
    except Exception:
        _registry.restore_token()
        raise
    finally:
        listener.close()
    runtime = _LocalRuntime(
        process=process,
        endpoint=f"http://127.0.0.1:{port}/v1",
        request=request,
        token=token,
    )
    runtime.log.follow(process.stdout)
    return runtime


def _looks_sensitive(name: str) -> bool:
    upper = name.upper()
    return upper.endswith(("_API_KEY", "_ACCESS_TOKEN", "_AUTH_TOKEN", "_PASSWORD", "_SECRET")) or upper in {
        "HF_TOKEN",
    }


def _child_environment() -> dict[str, str]:
    """Return a scrubbed environment containing credentials required by the child."""
    environment = os.environ.copy()
    for name in list(environment):
        if name not in _CHILD_SECRET_ALLOWLIST and _looks_sensitive(name):
            environment.pop(name)
    return environment


def _server_script() -> Path:
    return Path(__file__).parent / "gliner2" / "server.py"


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
        print("Preparing the isolated native GLiNER2 runtime (first start only)...")
        temporary_packages = temporary / "site-packages"
        subprocess.check_call([*_installer_command(), "--target", str(temporary_packages), *_SERVER_REQUIREMENTS])
        (temporary / ".ready").write_text(identity)
        if environment.exists():
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, environment)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return environment / "site-packages"


def _installer_command() -> list[str]:
    """Return a ``pip install`` prefix for this interpreter; uv-created environments ship without pip."""
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-warn-conflicts"]
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Preparing native GLiNER2 needs pip in this environment or uv on PATH.")
    return [uv, "pip", "install", "--python", sys.executable]


def prepare_development_server() -> Path:
    """Build or reuse the isolated environment without starting the server; return its import path."""
    return _ensure_server_environment()


def run_development_server() -> None:
    """Replace this process with the dependency-isolated development server."""
    server_packages = _ensure_server_environment()
    environment = _child_environment()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(server_packages), environment.get("PYTHONPATH"))))
    command = [sys.executable, str(_server_script()), *sys.argv[1:]]
    os.execve(sys.executable, command, environment)


def _wait_until_ready(runtime: _LocalRuntime) -> ModelCard:
    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if runtime.process.poll() is not None:
            raise RuntimeError(f"Local GLiNER2 exited during startup.\n{runtime.log.text()}")
        try:
            card = _read_model_card(runtime)
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            time.sleep(0.25)
            continue
        return card
    raise RuntimeError(f"Timed out waiting for local GLiNER2.\n{runtime.log.text()}") from last_error


def _verify_readiness(runtime: _LocalRuntime) -> None:
    _read_model_card(runtime)


def _read_model_card(runtime: _LocalRuntime) -> ModelCard:
    response = httpx.get(
        f"{runtime.endpoint}/models",
        headers={"Authorization": f"Bearer {runtime.token}"},
        timeout=2.0,
    )
    response.raise_for_status()
    models = ModelList.model_validate_json(response.content)
    if len(models.data) != 1:
        raise ValueError("Local GLiNER2 returned malformed model metadata.")
    card = models.data[0]
    if card.id != MODEL_ID or card.revision != MODEL_REVISION:
        raise ValueError("Local endpoint did not report the expected pinned GLiNER2 model.")
    return card


def _reserve_listener() -> socket.socket:
    """Bind and retain an ephemeral loopback socket until the child inherits it."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(socket.SOMAXCONN)
    except Exception:
        listener.close()
        raise
    return listener


def _stop_process(runtime: _LocalRuntime, *, timeout: float) -> None:
    """Stop a child without allowing a stuck wait to block cleanup."""
    if runtime.process.poll() is None:
        runtime.process.terminate()
        try:
            runtime.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            runtime.process.kill()
            try:
                runtime.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("Local GLiNER2 did not exit within %.1f seconds after being killed.", timeout)
    runtime.log.join()


atexit.register(_registry.shutdown)
