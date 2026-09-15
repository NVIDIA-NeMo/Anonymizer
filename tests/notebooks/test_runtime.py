# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import socket
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


def test_stop_local_runtime_is_idempotent() -> None:
    _runtime.stop_local_runtime()
    _runtime.stop_local_runtime()


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
