# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import errno
import os
import struct
from pathlib import Path

import pytest

from nemo_anonymizer_relay.transport import (
    exchange,
    parse_endpoint,
    read_frame,
    write_frame,
)


def test_endpoint_accepts_only_absolute_unix_paths() -> None:
    unix = parse_endpoint("unix:///tmp/nemo-anonymizer.sock")

    assert unix.scheme == "unix"
    assert str(unix.path) == "/tmp/nemo-anonymizer.sock"
    with pytest.raises(ValueError, match="absolute"):
        parse_endpoint("unix://relative.sock")
    with pytest.raises(ValueError, match="unix://"):
        parse_endpoint("tcp://127.0.0.1:8123")


async def test_framed_json_round_trip_over_unix_socket(private_socket_path: Path) -> None:

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request = await read_frame(reader, max_frame_bytes=1024)
        await write_frame(writer, {"echo": request}, max_frame_bytes=1024)
        writer.close()
        await writer.wait_closed()

    try:
        server = await asyncio.start_unix_server(handle, path=str(private_socket_path))
        os.chmod(private_socket_path, 0o600)
    except PermissionError as error:
        if error.errno == errno.EPERM:
            pytest.skip("execution sandbox blocks Unix-domain socket creation")
        raise
    try:
        response = await exchange(
            f"unix://{private_socket_path}",
            {"name": "Marisol", "unicode": "café"},
            timeout_seconds=1,
            max_frame_bytes=1024,
        )
    finally:
        server.close()
        await server.wait_closed()
        private_socket_path.unlink(missing_ok=True)

    assert response == {"echo": {"name": "Marisol", "unicode": "café"}}


async def test_reader_rejects_frame_before_reading_oversize_payload() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(struct.pack("!I", 4096))
    reader.feed_eof()

    with pytest.raises(ValueError, match="exceeds limit"):
        await read_frame(reader, max_frame_bytes=128)


async def test_writer_rejects_oversize_payload() -> None:
    class Writer:
        def write(self, data: bytes) -> None:
            raise AssertionError("oversize frame must not be written")

        async def drain(self) -> None:
            raise AssertionError("oversize frame must not be drained")

    with pytest.raises(ValueError, match="exceeds limit"):
        await write_frame(Writer(), {"value": "x" * 256}, max_frame_bytes=32)
