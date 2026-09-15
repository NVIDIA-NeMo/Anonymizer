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
