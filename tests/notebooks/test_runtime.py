# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest

from anonymizer.engine.local_inference import runtime as _runtime


def test_isolated_server_environment_requires_gliner2_local_and_flash_extras() -> None:
    assert "gliner2[local,disentangled-flash]==2.1.0" in _runtime._SERVER_REQUIREMENTS


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
        patch("anonymizer.engine.local_inference.runtime._ensure_server_environment", return_value=tmp_path),
        patch("anonymizer.engine.local_inference.runtime._reserve_listener", return_value=listener),
        patch("anonymizer.engine.local_inference.runtime.subprocess.Popen", return_value=process) as popen,
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
        patch("anonymizer.engine.local_inference.runtime._start_runtime", side_effect=start_runtime),
        patch("anonymizer.engine.local_inference.runtime._wait_until_ready", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        _runtime._ensure_runtime("auto")

    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=2.0)
    assert _runtime.LOCAL_TOKEN_ENV not in _runtime.os.environ
    assert _runtime._runtime is None


def _model_card_response(**changes: str) -> httpx.Response:
    card = {
        "id": _runtime.MODEL_ID,
        "object": "model",
        "revision": _runtime.MODEL_REVISION,
        "device": "cpu",
        "dtype": "float32",
        "attention": "standard",
        **changes,
    }
    request = httpx.Request("GET", "http://127.0.0.1:1234/v1/models")
    return httpx.Response(200, json={"object": "list", "data": [card]}, request=request)


def _starting_runtime() -> _runtime._LocalRuntime:
    return _runtime._LocalRuntime(
        process=Mock(),
        endpoint="http://127.0.0.1:1234/v1",
        requested_device="auto",
        selected_device="starting",
        token="secret",
    )


def test_read_model_card_returns_the_pinned_checkpoint_card() -> None:
    with patch("anonymizer.engine.local_inference.runtime.httpx.get", return_value=_model_card_response()):
        card = _runtime._read_model_card(_starting_runtime())
    assert (card.device, card.dtype, card.attention) == ("cpu", "float32", "standard")


def test_read_model_card_rejects_wrong_revision() -> None:
    with (
        patch(
            "anonymizer.engine.local_inference.runtime.httpx.get",
            return_value=_model_card_response(revision="moving-head"),
        ),
        pytest.raises(ValueError, match="pinned"),
    ):
        _runtime._read_model_card(_starting_runtime())


@pytest.mark.parametrize("changes", [{"device": "tpu"}, {"dtype": "fp8"}])
def test_read_model_card_rejects_malformed_metadata(changes: dict[str, str]) -> None:
    with (
        patch("anonymizer.engine.local_inference.runtime.httpx.get", return_value=_model_card_response(**changes)),
        pytest.raises(ValueError),  # noqa: PT011 - pydantic's ValidationError is a ValueError
    ):
        _runtime._read_model_card(_starting_runtime())


def test_scoped_runtime_owns_process_until_last_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _runtime.stop_local_runtime()
    monkeypatch.setenv(_runtime.LOCAL_TOKEN_ENV, "original")
    process = Mock()
    process.poll.return_value = None
    candidate = _runtime._LocalRuntime(process, "http://127.0.0.1:1234/v1", "cpu", "cpu", "token")

    def start(_: str) -> _runtime._LocalRuntime:
        _runtime._set_token_environment(candidate.token)
        return candidate

    with (
        patch.object(_runtime, "_start_runtime", side_effect=start),
        patch.object(_runtime, "_wait_until_ready"),
        patch.object(_runtime, "_verify_readiness"),
    ):
        with _runtime.runtime_context("cpu") as outer:
            with _runtime.runtime_context("cpu") as inner:
                assert outer is inner
                with pytest.raises(RuntimeError, match="session"):
                    with _runtime.singleton_runtime("cpu"):
                        pass
                with pytest.raises(RuntimeError, match="context"):
                    _runtime.stop_local_runtime()
            process.terminate.assert_not_called()
        process.terminate.assert_called_once()
    assert _runtime.os.environ[_runtime.LOCAL_TOKEN_ENV] == "original"


def test_scoped_runtime_borrows_existing_singleton() -> None:
    _runtime.stop_local_runtime()
    process = Mock()
    process.poll.return_value = None
    candidate = _runtime._LocalRuntime(process, "http://127.0.0.1:1234/v1", "cpu", "cpu", "token")
    _runtime._runtime = candidate
    try:
        with patch.object(_runtime, "_verify_readiness"):
            with _runtime.runtime_context("cpu") as runtime:
                assert runtime is candidate
                with pytest.raises(RuntimeError, match="context"):
                    with _runtime.runtime_context("cuda"):
                        pass
        process.terminate.assert_not_called()
    finally:
        _runtime.stop_local_runtime()


def test_start_failure_restores_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    _runtime.stop_local_runtime()
    monkeypatch.setenv(_runtime.LOCAL_TOKEN_ENV, "original")

    def fail_start(_: str) -> _runtime._LocalRuntime:
        _runtime._set_token_environment("temporary")
        raise OSError("startup failed")

    with patch.object(_runtime, "_start_runtime", side_effect=fail_start), pytest.raises(OSError, match="startup"):
        with _runtime.runtime_context("cpu"):
            pass
    assert _runtime.os.environ[_runtime.LOCAL_TOKEN_ENV] == "original"
    assert _runtime._runtime is None


def test_exit_cleanup_stops_runtime_with_active_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _runtime.stop_local_runtime()
    monkeypatch.setenv(_runtime.LOCAL_TOKEN_ENV, "original")
    process = Mock()
    process.poll.return_value = None
    candidate = _runtime._LocalRuntime(process, "http://127.0.0.1:1234/v1", "cpu", "cpu", "token")

    def start(_: str) -> _runtime._LocalRuntime:
        _runtime._set_token_environment(candidate.token)
        return candidate

    with patch.object(_runtime, "_start_runtime", side_effect=start), patch.object(_runtime, "_wait_until_ready"):
        with _runtime.runtime_context("cpu"):
            _runtime._shutdown_runtime()
            process.terminate.assert_called_once()
            assert _runtime.os.environ[_runtime.LOCAL_TOKEN_ENV] == "original"
            assert _runtime._runtime is None
    process.terminate.assert_called_once()


@pytest.mark.parametrize("borrowed,acquire_singleton", [(False, False), (True, False), (True, True)])
def test_dead_runtime_is_not_replaced_during_active_context(
    monkeypatch: pytest.MonkeyPatch, borrowed: bool, acquire_singleton: bool
) -> None:
    # Active contexts retain their endpoint and credentials even when the child exits.
    _runtime.stop_local_runtime()
    monkeypatch.setenv(_runtime.LOCAL_TOKEN_ENV, "original")
    processes = [Mock(), Mock()]
    for process in processes:
        process.poll.return_value = None
    candidates = [
        _runtime._LocalRuntime(processes[0], "http://127.0.0.1:1111/v1", "cpu", "cpu", "first"),
        _runtime._LocalRuntime(processes[1], "http://127.0.0.1:2222/v1", "cpu", "cpu", "second"),
    ]

    def start(_: str) -> _runtime._LocalRuntime:
        candidate = candidates[start_mock.call_count - 1]
        _runtime._set_token_environment(candidate.token)
        return candidate

    try:
        with (
            patch.object(_runtime, "_start_runtime", side_effect=start) as start_mock,
            patch.object(_runtime, "_wait_until_ready"),
            patch.object(_runtime, "_verify_readiness"),
        ):
            if borrowed:
                with _runtime.singleton_runtime("cpu"):
                    pass
            with _runtime.runtime_context("cpu") as outer:
                processes[0].poll.return_value = 1
                error = None
                try:
                    acquire = _runtime.singleton_runtime if acquire_singleton else _runtime.runtime_context
                    with acquire("cpu"):
                        pass
                except RuntimeError as exc:
                    error = exc
                assert error is not None
                assert "context" in str(error)
                assert start_mock.call_count == 1
                assert _runtime._runtime is outer is candidates[0]
                assert _runtime.os.environ[_runtime.LOCAL_TOKEN_ENV] == "first"
                assert _runtime._context_count == 1
                assert _runtime._context_owned is not borrowed
            assert _runtime._context_count == 0
            assert not _runtime._context_owned
            with _runtime.runtime_context("cpu") as fresh:
                assert fresh is candidates[1]
                assert start_mock.call_count == 2
            assert _runtime.os.environ[_runtime.LOCAL_TOKEN_ENV] == "original"
    finally:
        _runtime.stop_local_runtime()


@pytest.mark.parametrize("alive_observations", [1, 2])
def test_child_exit_during_nested_acquisition_never_replaces_runtime(alive_observations: int) -> None:
    # A scoped endpoint must never silently move when its child exits (self-hosting docs).
    _runtime.stop_local_runtime()
    process = Mock()
    process.poll.return_value = None
    existing = _runtime._LocalRuntime(process, "http://127.0.0.1:1111/v1", "cpu", "cpu", "old")
    replacement = _runtime._LocalRuntime(Mock(), "http://127.0.0.1:2222/v1", "cpu", "cpu", "new")
    _runtime._runtime = existing
    try:
        with (
            patch.object(_runtime, "_verify_readiness"),
            patch.object(_runtime, "_wait_until_ready"),
            patch.object(_runtime, "_start_runtime", return_value=replacement) as start,
        ):
            with _runtime.runtime_context("cpu") as outer:
                observations = iter([None] * alive_observations)
                process.poll.side_effect = lambda: next(observations, 1)
                try:
                    with _runtime.runtime_context("cpu") as inner:
                        assert inner.endpoint == outer.endpoint
                except RuntimeError:
                    pass
                start.assert_not_called()
    finally:
        _runtime.stop_local_runtime()


def test_replacement_during_first_acquisition_is_owned_and_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    _runtime.stop_local_runtime()
    monkeypatch.setenv(_runtime.LOCAL_TOKEN_ENV, "original")
    old_process, new_process = Mock(), Mock()
    old_process.poll.return_value = None
    new_process.poll.return_value = None
    old = _runtime._LocalRuntime(old_process, "http://127.0.0.1:1111/v1", "cpu", "cpu", "old")
    new = _runtime._LocalRuntime(new_process, "http://127.0.0.1:2222/v1", "cpu", "cpu", "new")
    _runtime._runtime = old
    ensure_runtime = _runtime._ensure_runtime

    def acquire_after_exit(device: str) -> _runtime._LocalRuntime:
        old_process.poll.return_value = 1
        return ensure_runtime(device)

    def start(_: str) -> _runtime._LocalRuntime:
        _runtime._set_token_environment(new.token)
        return new

    try:
        with (
            patch.object(_runtime, "_ensure_runtime", side_effect=acquire_after_exit),
            patch.object(_runtime, "_start_runtime", side_effect=start),
            patch.object(_runtime, "_wait_until_ready"),
        ):
            with _runtime.runtime_context("cpu") as runtime:
                assert runtime is new
            new_process.terminate.assert_called_once()
            assert _runtime._runtime is None
            assert _runtime.os.environ[_runtime.LOCAL_TOKEN_ENV] == "original"
    finally:
        _runtime.stop_local_runtime()


@pytest.mark.parametrize(
    ("argv", "prepared", "served"),
    [(["serve", "--prepare"], True, False), (["serve", "--host", "127.0.0.1", "--port", "9001"], False, True)],
)
def test_serve_entry_prepares_only_or_execs_the_server(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], prepared: bool, served: bool
) -> None:
    from anonymizer.engine.local_inference import serve

    monkeypatch.setattr(serve.sys, "argv", argv)
    with (
        patch.object(serve, "prepare_development_server") as prepare,
        patch.object(serve, "run_development_server") as run,
    ):
        serve.main()
    assert prepare.called is prepared
    assert run.called is served


def test_installer_uses_uv_when_the_environment_has_no_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_runtime.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(_runtime.shutil, "which", lambda name: "/bin/uv" if name == "uv" else None)
    assert _runtime._installer_command() == ["/bin/uv", "pip", "install", "--python", _runtime.sys.executable]
    monkeypatch.setattr(_runtime.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="pip in this environment or uv on PATH"):
        _runtime._installer_command()
