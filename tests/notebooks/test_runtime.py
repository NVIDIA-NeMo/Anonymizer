# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
