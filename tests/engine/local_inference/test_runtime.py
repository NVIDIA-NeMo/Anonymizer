# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import httpx
import pytest

from anonymizer.engine.local_inference import runtime as native_runtime_module
from anonymizer.engine.local_inference.gliner2.contract import AttentionRequest, DtypeRequest
from anonymizer.engine.local_inference.runtime import NativeRequest


def test_isolated_server_environment_requires_gliner2_local_and_flash_extras() -> None:
    assert "gliner2[local,disentangled-flash]==2.1.0" in native_runtime_module._SERVER_REQUIREMENTS


@pytest.mark.parametrize(
    "name",
    ["NVIDIA_API_KEY", "SERVICE_ACCESS_TOKEN", "DB_PASSWORD", "CLIENT_SECRET"],
)
def test_child_environment_secret_names_are_recognized(name: str) -> None:
    assert native_runtime_module._looks_sensitive(name)


def test_child_environment_preserves_hugging_face_token_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hugging-face-token")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-token")
    monkeypatch.setenv("SERVICE_ACCESS_TOKEN", "service-token")

    environment = native_runtime_module._child_environment()

    assert environment["HF_TOKEN"] == "hugging-face-token"
    assert "NVIDIA_API_KEY" not in environment
    assert "SERVICE_ACCESS_TOKEN" not in environment


def test_reserved_listener_keeps_ephemeral_port_owned_until_closed() -> None:
    listener = native_runtime_module._reserve_listener()
    challenger = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError):
            challenger.bind(listener.getsockname())
    finally:
        challenger.close()
        listener.close()


def test_start_runtime_passes_reserved_listener_to_child(tmp_path: Path) -> None:
    native_runtime_module.stop_local_runtime()
    listener = native_runtime_module._reserve_listener()
    listener_fd = listener.fileno()
    process = Mock(stdout=None)
    process.poll.return_value = 0

    with (
        patch("anonymizer.engine.local_inference.runtime._ensure_server_environment", return_value=tmp_path),
        patch("anonymizer.engine.local_inference.runtime._reserve_listener", return_value=listener),
        patch("anonymizer.engine.local_inference.runtime.subprocess.Popen", return_value=process) as popen,
    ):
        runtime = native_runtime_module._start_runtime(NativeRequest("cpu"))

    command = popen.call_args.args[0]
    assert command[-2:] == ["--fd", str(listener_fd)]
    assert popen.call_args.kwargs["pass_fds"] == (listener_fd,)
    assert listener.fileno() == -1
    assert runtime.endpoint.startswith("http://127.0.0.1:")
    native_runtime_module._registry.restore_token()


def test_stop_local_runtime_is_idempotent() -> None:
    native_runtime_module.stop_local_runtime()
    native_runtime_module.stop_local_runtime()


def test_stop_restores_token_when_process_remains_after_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    native_runtime_module.stop_local_runtime()
    monkeypatch.delenv(native_runtime_module.LOCAL_TOKEN_ENV, raising=False)
    native_runtime_module._registry.set_token("generated-token")
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [
        subprocess.TimeoutExpired(cmd="gliner2", timeout=10.0),
        subprocess.TimeoutExpired(cmd="gliner2", timeout=10.0),
    ]
    native_runtime_module._registry.runtime = native_runtime_module._LocalRuntime(
        process=process,
        endpoint="http://127.0.0.1:1234/v1",
        request=NativeRequest("auto"),
        token="generated-token",
    )

    native_runtime_module.stop_local_runtime()

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert native_runtime_module.LOCAL_TOKEN_ENV not in native_runtime_module.os.environ
    assert native_runtime_module._registry.runtime is None


def test_failed_runtime_joins_log_reader_before_tail_is_read() -> None:
    process = Mock()
    process.poll.return_value = 0
    log_thread = Mock()
    log = native_runtime_module._LogTail()
    log._thread = log_thread
    runtime = native_runtime_module._LocalRuntime(
        process, "http://127.0.0.1:1234/v1", NativeRequest("auto"), "generated-token", log
    )

    native_runtime_module._stop_process(runtime, timeout=2.0)
    tail = runtime.log.text()

    assert log_thread.join.call_count == 2
    log_thread.join.assert_called_with(timeout=1.0)
    assert tail == "No child-process output was captured."


def test_repeated_token_assignment_restores_original_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    native_runtime_module.stop_local_runtime()
    monkeypatch.delenv(native_runtime_module.LOCAL_TOKEN_ENV, raising=False)

    native_runtime_module._registry.set_token("first-generated-token")
    native_runtime_module._registry.set_token("replacement-generated-token")
    native_runtime_module._registry.restore_token()

    assert native_runtime_module.LOCAL_TOKEN_ENV not in native_runtime_module.os.environ


def test_interrupted_startup_stops_candidate_and_restores_token(monkeypatch: pytest.MonkeyPatch) -> None:
    native_runtime_module.stop_local_runtime()
    monkeypatch.delenv(native_runtime_module.LOCAL_TOKEN_ENV, raising=False)
    process = Mock()
    process.poll.return_value = None
    candidate = native_runtime_module._LocalRuntime(
        process=process,
        endpoint="http://127.0.0.1:1234/v1",
        request=NativeRequest("auto"),
        token="generated-token",
    )

    def start_runtime(_: str, __: str = "auto") -> native_runtime_module._LocalRuntime:
        native_runtime_module._registry.set_token(candidate.token)
        return candidate

    with (
        patch("anonymizer.engine.local_inference.runtime._start_runtime", side_effect=start_runtime),
        patch("anonymizer.engine.local_inference.runtime._wait_until_ready", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        native_runtime_module._registry.ensure(NativeRequest("auto"))

    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=2.0)
    assert native_runtime_module.LOCAL_TOKEN_ENV not in native_runtime_module.os.environ
    assert native_runtime_module._registry.runtime is None


def _model_card_response(**changes: str) -> httpx.Response:
    card = {
        "id": native_runtime_module.MODEL_ID,
        "object": "model",
        "revision": native_runtime_module.MODEL_REVISION,
        "device": "cpu",
        "dtype": "float32",
        "attention": "standard",
        **changes,
    }
    request = httpx.Request("GET", "http://127.0.0.1:1234/v1/models")
    return httpx.Response(200, json={"object": "list", "data": [card]}, request=request)


def _starting_runtime() -> native_runtime_module._LocalRuntime:
    return native_runtime_module._LocalRuntime(
        process=Mock(),
        endpoint="http://127.0.0.1:1234/v1",
        request=NativeRequest("auto"),
        token="secret",
    )


def test_read_model_card_returns_the_pinned_checkpoint_card() -> None:
    with patch("anonymizer.engine.local_inference.runtime.httpx.get", return_value=_model_card_response()):
        card = native_runtime_module._read_model_card(_starting_runtime())
    assert (card.device, card.dtype, card.attention) == ("cpu", "float32", "standard")


def test_read_model_card_rejects_wrong_revision() -> None:
    with (
        patch(
            "anonymizer.engine.local_inference.runtime.httpx.get",
            return_value=_model_card_response(revision="moving-head"),
        ),
        pytest.raises(ValueError, match="pinned"),
    ):
        native_runtime_module._read_model_card(_starting_runtime())


@pytest.mark.parametrize("changes", [{"device": "tpu"}, {"dtype": "fp8"}])
def test_read_model_card_rejects_malformed_metadata(changes: dict[str, str]) -> None:
    with (
        patch("anonymizer.engine.local_inference.runtime.httpx.get", return_value=_model_card_response(**changes)),
        pytest.raises(ValueError),  # noqa: PT011 - pydantic's ValidationError is a ValueError
    ):
        native_runtime_module._read_model_card(_starting_runtime())


def test_scoped_runtime_owns_process_until_last_context(monkeypatch: pytest.MonkeyPatch) -> None:
    native_runtime_module.stop_local_runtime()
    monkeypatch.setenv(native_runtime_module.LOCAL_TOKEN_ENV, "original")
    process = Mock()
    process.poll.return_value = None
    candidate = native_runtime_module._LocalRuntime(process, "http://127.0.0.1:1234/v1", NativeRequest("cpu"), "token")

    def start(_: str, __: str = "auto") -> native_runtime_module._LocalRuntime:
        native_runtime_module._registry.set_token(candidate.token)
        return candidate

    with (
        patch.object(native_runtime_module, "_start_runtime", side_effect=start),
        patch.object(native_runtime_module, "_wait_until_ready"),
        patch.object(native_runtime_module, "_verify_readiness"),
    ):
        with native_runtime_module.runtime_context("cpu") as outer:
            with native_runtime_module.runtime_context("cpu") as inner:
                assert outer is inner
                with pytest.raises(RuntimeError, match="session"):
                    with native_runtime_module.singleton_runtime("cpu"):
                        pass
                with pytest.raises(RuntimeError, match="context"):
                    native_runtime_module.stop_local_runtime()
            process.terminate.assert_not_called()
        process.terminate.assert_called_once()
    assert native_runtime_module.os.environ[native_runtime_module.LOCAL_TOKEN_ENV] == "original"


def test_scoped_runtime_borrows_existing_singleton() -> None:
    native_runtime_module.stop_local_runtime()
    process = Mock()
    process.poll.return_value = None
    candidate = native_runtime_module._LocalRuntime(process, "http://127.0.0.1:1234/v1", NativeRequest("cpu"), "token")
    native_runtime_module._registry.runtime = candidate
    try:
        with patch.object(native_runtime_module, "_verify_readiness"):
            with native_runtime_module.runtime_context("cpu") as runtime:
                assert runtime is candidate
                with pytest.raises(RuntimeError, match="context"):
                    with native_runtime_module.runtime_context("cuda"):
                        pass
        process.terminate.assert_not_called()
    finally:
        native_runtime_module.stop_local_runtime()


def test_start_failure_restores_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    native_runtime_module.stop_local_runtime()
    monkeypatch.setenv(native_runtime_module.LOCAL_TOKEN_ENV, "original")

    def fail_start(_: str, __: str = "auto") -> native_runtime_module._LocalRuntime:
        native_runtime_module._registry.set_token("temporary")
        raise OSError("startup failed")

    with (
        patch.object(native_runtime_module, "_start_runtime", side_effect=fail_start),
        pytest.raises(OSError, match="startup"),
    ):
        with native_runtime_module.runtime_context("cpu"):
            pass
    assert native_runtime_module.os.environ[native_runtime_module.LOCAL_TOKEN_ENV] == "original"
    assert native_runtime_module._registry.runtime is None


def test_exit_cleanup_stops_runtime_with_active_context(monkeypatch: pytest.MonkeyPatch) -> None:
    native_runtime_module.stop_local_runtime()
    monkeypatch.setenv(native_runtime_module.LOCAL_TOKEN_ENV, "original")
    process = Mock()
    process.poll.return_value = None
    candidate = native_runtime_module._LocalRuntime(process, "http://127.0.0.1:1234/v1", NativeRequest("cpu"), "token")

    def start(_: str, __: str = "auto") -> native_runtime_module._LocalRuntime:
        native_runtime_module._registry.set_token(candidate.token)
        return candidate

    with (
        patch.object(native_runtime_module, "_start_runtime", side_effect=start),
        patch.object(native_runtime_module, "_wait_until_ready"),
    ):
        with native_runtime_module.runtime_context("cpu"):
            native_runtime_module._registry.shutdown()
            process.terminate.assert_called_once()
            assert native_runtime_module.os.environ[native_runtime_module.LOCAL_TOKEN_ENV] == "original"
            assert native_runtime_module._registry.runtime is None
    process.terminate.assert_called_once()


@pytest.mark.parametrize("borrowed,acquire_singleton", [(False, False), (True, False), (True, True)])
def test_dead_runtime_is_not_replaced_during_active_context(
    monkeypatch: pytest.MonkeyPatch, borrowed: bool, acquire_singleton: bool
) -> None:
    # Active contexts retain their endpoint and credentials even when the child exits.
    native_runtime_module.stop_local_runtime()
    monkeypatch.setenv(native_runtime_module.LOCAL_TOKEN_ENV, "original")
    processes = [Mock(), Mock()]
    for process in processes:
        process.poll.return_value = None
    candidates = [
        native_runtime_module._LocalRuntime(processes[0], "http://127.0.0.1:1111/v1", NativeRequest("cpu"), "first"),
        native_runtime_module._LocalRuntime(processes[1], "http://127.0.0.1:2222/v1", NativeRequest("cpu"), "second"),
    ]

    def start(_: str, __: str = "auto") -> native_runtime_module._LocalRuntime:
        candidate = candidates[start_mock.call_count - 1]
        native_runtime_module._registry.set_token(candidate.token)
        return candidate

    try:
        with (
            patch.object(native_runtime_module, "_start_runtime", side_effect=start) as start_mock,
            patch.object(native_runtime_module, "_wait_until_ready"),
            patch.object(native_runtime_module, "_verify_readiness"),
        ):
            if borrowed:
                with native_runtime_module.singleton_runtime("cpu"):
                    pass
            with native_runtime_module.runtime_context("cpu") as outer:
                processes[0].poll.return_value = 1
                error = None
                try:
                    acquire = (
                        native_runtime_module.singleton_runtime
                        if acquire_singleton
                        else native_runtime_module.runtime_context
                    )
                    with acquire("cpu"):
                        pass
                except RuntimeError as exc:
                    error = exc
                assert error is not None
                assert "context" in str(error)
                assert start_mock.call_count == 1
                assert native_runtime_module._registry.runtime is outer is candidates[0]
                assert native_runtime_module.os.environ[native_runtime_module.LOCAL_TOKEN_ENV] == "first"
                assert native_runtime_module._registry.context_count == 1
                assert native_runtime_module._registry.context_owned is not borrowed
            assert native_runtime_module._registry.context_count == 0
            assert not native_runtime_module._registry.context_owned
            with native_runtime_module.runtime_context("cpu") as fresh:
                assert fresh is candidates[1]
                assert start_mock.call_count == 2
            assert native_runtime_module.os.environ[native_runtime_module.LOCAL_TOKEN_ENV] == "original"
    finally:
        native_runtime_module.stop_local_runtime()


@pytest.mark.parametrize("alive_observations", [1, 2])
def test_child_exit_during_nested_acquisition_never_replaces_runtime(alive_observations: int) -> None:
    # A scoped endpoint must never silently move when its child exits (self-hosting docs).
    native_runtime_module.stop_local_runtime()
    process = Mock()
    process.poll.return_value = None
    existing = native_runtime_module._LocalRuntime(process, "http://127.0.0.1:1111/v1", NativeRequest("cpu"), "old")
    replacement = native_runtime_module._LocalRuntime(Mock(), "http://127.0.0.1:2222/v1", NativeRequest("cpu"), "new")
    native_runtime_module._registry.runtime = existing
    try:
        with (
            patch.object(native_runtime_module, "_verify_readiness"),
            patch.object(native_runtime_module, "_wait_until_ready"),
            patch.object(native_runtime_module, "_start_runtime", return_value=replacement) as start,
        ):
            with native_runtime_module.runtime_context("cpu") as outer:
                observations = iter([None] * alive_observations)
                process.poll.side_effect = lambda: next(observations, 1)
                try:
                    with native_runtime_module.runtime_context("cpu") as inner:
                        assert inner.endpoint == outer.endpoint
                except RuntimeError:
                    pass
                start.assert_not_called()
    finally:
        native_runtime_module.stop_local_runtime()


def test_replacement_during_first_acquisition_is_owned_and_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    native_runtime_module.stop_local_runtime()
    monkeypatch.setenv(native_runtime_module.LOCAL_TOKEN_ENV, "original")
    old_process, new_process = Mock(), Mock()
    old_process.poll.return_value = None
    new_process.poll.return_value = None
    old = native_runtime_module._LocalRuntime(old_process, "http://127.0.0.1:1111/v1", NativeRequest("cpu"), "old")
    new = native_runtime_module._LocalRuntime(new_process, "http://127.0.0.1:2222/v1", NativeRequest("cpu"), "new")
    native_runtime_module._registry.runtime = old
    ensure_runtime = native_runtime_module._registry.ensure

    def acquire_after_exit(request: NativeRequest) -> native_runtime_module._LocalRuntime:
        old_process.poll.return_value = 1
        return ensure_runtime(request)

    def start(_: str, __: str = "auto") -> native_runtime_module._LocalRuntime:
        native_runtime_module._registry.set_token(new.token)
        return new

    try:
        with (
            patch.object(native_runtime_module._registry, "ensure", side_effect=acquire_after_exit),
            patch.object(native_runtime_module, "_start_runtime", side_effect=start),
            patch.object(native_runtime_module, "_wait_until_ready"),
        ):
            with native_runtime_module.runtime_context("cpu") as runtime:
                assert runtime is new
            new_process.terminate.assert_called_once()
            assert native_runtime_module._registry.runtime is None
            assert native_runtime_module.os.environ[native_runtime_module.LOCAL_TOKEN_ENV] == "original"
    finally:
        native_runtime_module.stop_local_runtime()


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
    monkeypatch.setattr(native_runtime_module.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(native_runtime_module.shutil, "which", lambda name: "/bin/uv" if name == "uv" else None)
    assert native_runtime_module._installer_command() == [
        "/bin/uv",
        "pip",
        "install",
        "--python",
        native_runtime_module.sys.executable,
    ]
    monkeypatch.setattr(native_runtime_module.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="pip in this environment or uv on PATH"):
        native_runtime_module._installer_command()


def _running_runtime(dtype: DtypeRequest) -> native_runtime_module._LocalRuntime:
    process = Mock()
    process.poll.return_value = None
    return native_runtime_module._LocalRuntime(
        process, "http://127.0.0.1:1234/v1", NativeRequest("cpu", dtype), "token"
    )


@pytest.mark.parametrize(
    ("running", "requested"),
    [
        (NativeRequest("cpu", "bfloat16"), ("cpu", "float32", "auto")),
        (NativeRequest("cpu", "auto", "standard"), ("cpu", "auto", "disentangled_flash")),
    ],
)
def test_borrowing_rejects_a_runtime_started_for_a_different_request(
    running: NativeRequest, requested: tuple[str, str, str]
) -> None:
    native_runtime_module.stop_local_runtime()
    process = Mock()
    process.poll.return_value = None
    native_runtime_module._registry.runtime = native_runtime_module._LocalRuntime(
        process, "http://127.0.0.1:1234/v1", running, "token"
    )
    try:
        with pytest.raises(RuntimeError, match="different device, precision or attention"):
            with native_runtime_module.runtime_context(*requested):  # ty: ignore[invalid-argument-type]
                pass
    finally:
        native_runtime_module._registry.runtime = None


def test_a_different_dtype_is_not_served_by_a_running_runtime() -> None:
    native_runtime_module.stop_local_runtime()
    running = _running_runtime("bfloat16")
    native_runtime_module._registry.runtime = running
    try:
        with patch.object(native_runtime_module, "_verify_readiness"):
            with native_runtime_module.runtime_context("cpu", "bfloat16") as borrowed:
                assert borrowed is running
                with pytest.raises(RuntimeError, match="context"):
                    with native_runtime_module.runtime_context("cpu", "float32"):
                        pass
        with pytest.raises(RuntimeError, match="different device, precision or attention"):
            with native_runtime_module.runtime_context("cpu", "float32"):
                pass
    finally:
        native_runtime_module._registry.runtime = None


def test_a_different_dtype_replaces_an_idle_singleton() -> None:
    native_runtime_module.stop_local_runtime()
    old = _running_runtime("bfloat16")
    new = _running_runtime("float32")
    native_runtime_module._registry.runtime = old
    try:
        with (
            patch.object(native_runtime_module, "_start_runtime", return_value=new) as start,
            patch.object(native_runtime_module, "_wait_until_ready"),
        ):
            with native_runtime_module.singleton_runtime("cpu", "float32") as runtime:
                assert runtime is new
        start.assert_called_once_with(NativeRequest("cpu", "float32"))
        cast(Mock, old.process).terminate.assert_called_once()
    finally:
        native_runtime_module._registry.runtime = None


@pytest.mark.parametrize(
    ("requested", "inherited", "expected"),
    [("bfloat16", "float32", "bfloat16"), ("auto", "bfloat16", "bfloat16"), ("auto", None, None)],
)
def test_start_runtime_gives_an_explicit_dtype_precedence_over_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: DtypeRequest,
    inherited: str | None,
    expected: str | None,
) -> None:
    native_runtime_module.stop_local_runtime()
    if inherited is None:
        monkeypatch.delenv(native_runtime_module.DTYPE_ENV, raising=False)
    else:
        monkeypatch.setenv(native_runtime_module.DTYPE_ENV, inherited)
    listener = native_runtime_module._reserve_listener()
    process = Mock(stdout=None)
    with (
        patch("anonymizer.engine.local_inference.runtime._ensure_server_environment", return_value=tmp_path),
        patch("anonymizer.engine.local_inference.runtime._reserve_listener", return_value=listener),
        patch("anonymizer.engine.local_inference.runtime.subprocess.Popen", return_value=process) as popen,
    ):
        runtime = native_runtime_module._start_runtime(NativeRequest("cpu", requested))
    assert popen.call_args.kwargs["env"].get(native_runtime_module.DTYPE_ENV) == expected
    assert runtime.request.dtype == requested
    native_runtime_module._registry.restore_token()


@pytest.mark.parametrize(
    ("requested", "inherited", "expected"),
    [
        ("disentangled_flash", "standard", "disentangled_flash"),
        ("auto", "disentangled_flash", "disentangled_flash"),
        ("auto", None, None),
    ],
)
def test_start_runtime_gives_an_explicit_attention_precedence_over_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: AttentionRequest,
    inherited: str | None,
    expected: str | None,
) -> None:
    native_runtime_module.stop_local_runtime()
    if inherited is None:
        monkeypatch.delenv(native_runtime_module.ATTENTION_ENV, raising=False)
    else:
        monkeypatch.setenv(native_runtime_module.ATTENTION_ENV, inherited)
    listener = native_runtime_module._reserve_listener()
    with (
        patch("anonymizer.engine.local_inference.runtime._ensure_server_environment", return_value=tmp_path),
        patch("anonymizer.engine.local_inference.runtime._reserve_listener", return_value=listener),
        patch("anonymizer.engine.local_inference.runtime.subprocess.Popen", return_value=Mock(stdout=None)) as popen,
    ):
        runtime = native_runtime_module._start_runtime(NativeRequest("cpu", "auto", requested))
    assert popen.call_args.kwargs["env"].get(native_runtime_module.ATTENTION_ENV) == expected
    assert runtime.request.attention == requested
    native_runtime_module._registry.restore_token()
