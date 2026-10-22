# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test access to the execution the harness keeps alive for a raised error."""

from __future__ import annotations

from typing import Any

from anonymizer.engine.harness.kernel import retained_execution


def retained(error: BaseException) -> Any:
    """Return the retained execution, failing the test when the raise site kept none."""
    execution = retained_execution(error)
    assert execution is not None, f"no execution retained for {type(error).__name__}"
    return execution
