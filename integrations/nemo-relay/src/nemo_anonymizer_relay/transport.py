# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small framed-JSON transport between the Relay worker and exporter service."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

Json = Any
_HEADER = struct.Struct("!I")
DEFAULT_MAX_FRAME_BYTES = 8 * 1024 * 1024


class FrameWriter(Protocol):
    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...


@dataclass(frozen=True)
class Endpoint:
    scheme: str
    path: Path | None = None


def parse_endpoint(value: str) -> Endpoint:
    parsed = urlparse(value)
    if parsed.scheme == "unix":
        path = Path(parsed.path)
        if not path.is_absolute():
            raise ValueError("unix exporter endpoint must use an absolute path")
        return Endpoint(scheme="unix", path=path)
    raise ValueError("exporter endpoint must use unix://")


def validate_private_unix_parent(path: Path) -> None:
    """Require an owner-only directory so another user cannot pre-bind the socket."""

    parent = path.parent
    metadata = parent.stat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"unix exporter parent is not a directory: {parent}")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise ValueError(f"unix exporter parent must be owned by the current user: {parent}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError(f"unix exporter parent must not be accessible by group or other: {parent}")


def _validate_private_unix_socket(path: Path) -> None:
    validate_private_unix_parent(path)
    metadata = path.lstat()
    if not stat.S_ISSOCK(metadata.st_mode):
        raise ValueError(f"unix exporter endpoint is not a socket: {path}")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise ValueError(f"unix exporter socket must be owned by the current user: {path}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError(f"unix exporter socket must not be accessible by group or other: {path}")


async def exchange(
    endpoint: str,
    message: Json,
    *,
    timeout_seconds: float,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> Json:
    """Send one request and read one response."""

    async def communicate() -> Json:
        parsed = parse_endpoint(endpoint)
        if parsed.scheme == "unix":
            if parsed.path is None:
                raise ValueError("unix exporter endpoint is missing a socket path")
            _validate_private_unix_socket(parsed.path)
            reader, writer = await asyncio.open_unix_connection(str(parsed.path))
        else:  # pragma: no cover - parse_endpoint currently permits only Unix
            raise ValueError("exporter endpoint must use unix://")
        try:
            await write_frame(writer, message, max_frame_bytes=max_frame_bytes)
            return await read_frame(reader, max_frame_bytes=max_frame_bytes)
        finally:
            writer.close()
            await writer.wait_closed()

    return await asyncio.wait_for(communicate(), timeout_seconds)


async def read_frame(
    reader: asyncio.StreamReader,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> Json:
    header = await reader.readexactly(_HEADER.size)
    (length,) = _HEADER.unpack(header)
    if length > max_frame_bytes:
        raise ValueError(f"frame size {length} exceeds limit {max_frame_bytes}")
    payload = await reader.readexactly(length)
    return json.loads(payload)


async def write_frame(
    writer: FrameWriter,
    message: Json,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> None:
    payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > max_frame_bytes:
        raise ValueError(f"frame size {len(payload)} exceeds limit {max_frame_bytes}")
    writer.write(_HEADER.pack(len(payload)))
    writer.write(payload)
    await writer.drain()


__all__ = [
    "DEFAULT_MAX_FRAME_BYTES",
    "Endpoint",
    "exchange",
    "parse_endpoint",
    "read_frame",
    "validate_private_unix_parent",
    "write_frame",
]
