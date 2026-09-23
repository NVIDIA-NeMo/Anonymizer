# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from anonymizer.notebooks import _runtime


def test_isolated_server_environment_requires_gliner2_local_extra() -> None:
    assert "gliner2[local]==2.0.0" in _runtime._SERVER_REQUIREMENTS


@pytest.mark.parametrize(
    "name",
    ["NVIDIA_API_KEY", "HF_TOKEN", "SERVICE_ACCESS_TOKEN", "DB_PASSWORD", "CLIENT_SECRET"],
)
def test_child_environment_secret_names_are_recognized(name: str) -> None:
    assert _runtime._looks_sensitive(name)


def test_child_environment_preserves_hugging_face_token_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hugging-face-token")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-token")
    monkeypatch.setenv("SERVICE_ACCESS_TOKEN", "service-token")

    environment = _runtime._child_environment()

    assert environment["HF_TOKEN"] == "hugging-face-token"
    assert "NVIDIA_API_KEY" not in environment
    assert "SERVICE_ACCESS_TOKEN" not in environment


def test_reserved_listener_keeps_ephemeral_port_owned_until_closed() -> None:
    listener = _runtime._reserve_listener()
    challenger = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError):
            challenger.bind(listener.getsockname())
    finally:
        challenger.close()
        listener.close()


def test_start_runtime_passes_reserved_listener_to_child(tmp_path: Path) -> None:
    _runtime.stop_local_runtime()
    listener = _runtime._reserve_listener()
    listener_fd = listener.fileno()
    process = Mock(stdout=None)
    process.poll.return_value = 0

    with (
        patch("anonymizer.notebooks._runtime._ensure_server_environment", return_value=tmp_path),
        patch("anonymizer.notebooks._runtime._reserve_listener", return_value=listener),
        patch("anonymizer.notebooks._runtime.subprocess.Popen", return_value=process) as popen,
    ):
        runtime = _runtime._start_runtime("cpu")

    command = popen.call_args.args[0]
    assert command[-2:] == ["--fd", str(listener_fd)]
    assert popen.call_args.kwargs["pass_fds"] == (listener_fd,)
    assert listener.fileno() == -1
    assert runtime.endpoint.startswith("http://127.0.0.1:")
    _runtime._restore_token_environment()


def test_create_anonymizer_uses_runtime_endpoint() -> None:
    runtime = _runtime._LocalRuntime(
        process=Mock(),
        endpoint="http://127.0.0.1:1234/v1",
        requested_device="cpu",
        selected_device="cpu",
        token="generated-token",
    )
    configuration = Mock(model_configs="models", model_providers=[])
    anonymizer = Mock()

    with (
        patch("anonymizer.notebooks._runtime.validate_notebook_model_inputs"),
        patch("anonymizer.notebooks._runtime._ensure_runtime", return_value=runtime),
        patch(
            "anonymizer.notebooks._runtime.build_notebook_model_configuration",
            return_value=configuration,
        ) as build_configuration,
        patch("anonymizer.notebooks._runtime.Anonymizer", return_value=anonymizer),
    ):
        result = _runtime.create_anonymizer(gliner_device="cpu")

    assert result is anonymizer
    build_configuration.assert_called_once_with(
        model_configs=None,
        model_providers=None,
        endpoint=runtime.endpoint,
    )


def test_stop_local_runtime_is_idempotent() -> None:
    _runtime.stop_local_runtime()
    _runtime.stop_local_runtime()


def test_stop_restores_token_when_process_remains_after_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    _runtime.stop_local_runtime()
    monkeypatch.delenv(_runtime.LOCAL_TOKEN_ENV, raising=False)
    _runtime._set_token_environment("generated-token")
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [
        subprocess.TimeoutExpired(cmd="gliner2", timeout=10.0),
        subprocess.TimeoutExpired(cmd="gliner2", timeout=10.0),
    ]
    _runtime._runtime = _runtime._LocalRuntime(
        process=process,
        endpoint="http://127.0.0.1:1234/v1",
        requested_device="auto",
        selected_device="starting",
        token="generated-token",
    )

    _runtime.stop_local_runtime()

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert _runtime.LOCAL_TOKEN_ENV not in _runtime.os.environ
    assert _runtime._runtime is None


def test_failed_runtime_joins_log_reader_before_tail_is_read() -> None:
    process = Mock()
    process.poll.return_value = 0
    log_thread = Mock()
    runtime = _runtime._LocalRuntime(
        process=process,
        endpoint="http://127.0.0.1:1234/v1",
        requested_device="auto",
        selected_device="starting",
        token="generated-token",
        log_thread=log_thread,
    )

    _runtime._stop_failed_runtime(runtime)
    tail = _runtime._format_log_tail(runtime)

    assert log_thread.join.call_count == 2
    log_thread.join.assert_called_with(timeout=1.0)
    assert tail == "No child-process output was captured."


def test_repeated_token_assignment_restores_original_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    _runtime.stop_local_runtime()
    monkeypatch.delenv(_runtime.LOCAL_TOKEN_ENV, raising=False)

    _runtime._set_token_environment("first-generated-token")
    _runtime._set_token_environment("replacement-generated-token")
    _runtime._restore_token_environment()

    assert _runtime.LOCAL_TOKEN_ENV not in _runtime.os.environ


def test_interrupted_startup_stops_candidate_and_restores_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _runtime.stop_local_runtime()
    monkeypatch.delenv(_runtime.LOCAL_TOKEN_ENV, raising=False)
    process = Mock()
    process.poll.return_value = None
    candidate = _runtime._LocalRuntime(
        process=process,
        endpoint="http://127.0.0.1:1234/v1",
        requested_device="auto",
        selected_device="starting",
        token="generated-token",
    )

    def start_runtime(_: str) -> _runtime._LocalRuntime:
        _runtime._set_token_environment(candidate.token)
        return candidate

    with (
        patch("anonymizer.notebooks._runtime._start_runtime", side_effect=start_runtime),
        patch("anonymizer.notebooks._runtime._wait_until_ready", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        _runtime._ensure_runtime("auto")

    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=2.0)
    assert _runtime.LOCAL_TOKEN_ENV not in _runtime.os.environ
    assert _runtime._runtime is None


def test_read_metadata_rejects_wrong_revision() -> None:
    runtime = _runtime._LocalRuntime(
        process=Mock(),
        endpoint="http://127.0.0.1:1234/v1",
        requested_device="auto",
        selected_device="starting",
        token="secret",
    )
    response = Mock()
    response.json.return_value = {"data": [{"id": _runtime.MODEL_ID, "revision": "moving-head", "device": "cpu"}]}
    with (
        patch("anonymizer.notebooks._runtime.httpx.get", return_value=response),
        pytest.raises(ValueError, match="pinned"),
    ):
        _runtime._read_metadata(runtime)
