# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def private_socket_path() -> Iterator[Path]:
    # Keep the private parent path short enough for Unix-domain socket limits.
    prefix = f"na-{os.getpid()}-{uuid4().hex[:8]}-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir="/tmp") as directory:
        yield Path(directory) / "exporter.sock"
