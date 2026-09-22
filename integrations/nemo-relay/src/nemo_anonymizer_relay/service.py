# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long-lived local Anonymizer service for the Relay subscriber worker."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import stat
from pathlib import Path
from typing import Any

from nemo_anonymizer_relay.backend import AnonymizerBackend, BackendConfig, require_api_keys
from nemo_anonymizer_relay.exporter import JsonlSink, ProtectedExporter
from nemo_anonymizer_relay.sanitizer import ObservationSanitizer
from nemo_anonymizer_relay.transport import (
    DEFAULT_MAX_FRAME_BYTES,
    exchange,
    parse_endpoint,
    read_frame,
    validate_private_unix_parent,
    write_frame,
)


class ExporterServer:
    def __init__(
        self,
        endpoint: str,
        exporter: ProtectedExporter,
        *,
        max_frame_bytes: int,
        connection_timeout_seconds: float = 5.0,
        max_connections: int = 64,
    ) -> None:
        self._endpoint = endpoint
        self._exporter = exporter
        self._max_frame_bytes = max_frame_bytes
        self._connection_timeout_seconds = connection_timeout_seconds
        self._connections = asyncio.Semaphore(max_connections)
        self._server: asyncio.AbstractServer | None = None
        self._stopped = asyncio.Event()
        self._closed = False

    async def start(self) -> None:
        parsed = parse_endpoint(self._endpoint)
        if parsed.scheme == "unix":
            if parsed.path is None:
                raise ValueError("unix exporter endpoint is missing a socket path")
            parsed.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            validate_private_unix_parent(parsed.path)
            _remove_stale_socket(parsed.path)
            self._server = await asyncio.start_unix_server(
                self._handle,
                path=str(parsed.path),
            )
            os.chmod(parsed.path, 0o600)
        else:  # pragma: no cover - parse_endpoint currently permits only Unix
            raise ValueError("exporter endpoint must use unix://")
        try:
            self._exporter.start()
        except Exception:
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()
            if parsed.scheme == "unix" and parsed.path is not None:
                parsed.path.unlink(missing_ok=True)
            raise

    async def wait(self) -> None:
        await self._stopped.wait()

    async def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._exporter.close()
        except Exception:
            # A failed sink is already visible in exporter status. Teardown
            # must still release the socket and stop the service.
            pass
        finally:
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()
            parsed = parse_endpoint(self._endpoint)
            if parsed.scheme == "unix" and parsed.path is not None:
                parsed.path.unlink(missing_ok=True)
            self._stopped.set()

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if self._connections.locked():
            try:
                await write_frame(
                    writer,
                    {"status": "busy"},
                    max_frame_bytes=self._max_frame_bytes,
                )
            finally:
                writer.close()
                await writer.wait_closed()
            return
        try:
            async with self._connections:
                try:
                    request = await asyncio.wait_for(
                        read_frame(reader, max_frame_bytes=self._max_frame_bytes),
                        timeout=self._connection_timeout_seconds,
                    )
                    response = await self._dispatch(request)
                except Exception as error:
                    response = {"status": "error", "error": type(error).__name__}
                try:
                    await write_frame(writer, response, max_frame_bytes=self._max_frame_bytes)
                finally:
                    writer.close()
                    await writer.wait_closed()
        except (BrokenPipeError, ConnectionError):
            pass

    async def _dispatch(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise TypeError("request must be an object")
        kind = request.get("kind")
        if kind == "health":
            snapshot = self._exporter.snapshot()
            status = "ready" if snapshot["accepting"] and not snapshot["fatal_error"] else "failed"
            return {"status": status, **snapshot}
        if kind == "status":
            return {"status": "ok", **self._exporter.snapshot()}
        if kind == "event":
            event = request.get("event")
            if not isinstance(event, dict):
                raise TypeError("event must be an object")
            encoded_bytes = len(json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            return self._exporter.enqueue(event, encoded_bytes)
        if kind == "flush":
            try:
                await self._exporter.flush()
            except RuntimeError:
                return {"status": "failed", **self._exporter.snapshot()}
            return {"status": "flushed", **self._exporter.snapshot()}
        if kind == "shutdown":
            self._exporter.quiesce()
            try:
                await self._exporter.flush()
            except RuntimeError:
                pass
            response = {"status": "stopping", **self._exporter.snapshot()}
            asyncio.get_running_loop().call_soon(lambda: asyncio.create_task(self.stop()))
            return response
        raise ValueError("unsupported request kind")


def _remove_stale_socket(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise RuntimeError(f"refusing to replace non-socket endpoint: {path}")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.25)
        probe.connect(str(path))
    except (ConnectionRefusedError, FileNotFoundError):
        pass
    except OSError as error:
        raise RuntimeError(f"cannot verify existing exporter socket: {path}") from error
    else:
        raise RuntimeError(f"exporter endpoint is already active: {path}")
    finally:
        probe.close()
    path.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--endpoint", required=True)
    serve.add_argument("--output", type=Path, required=True)
    serve.add_argument("--metrics", type=Path)
    serve.add_argument("--queue-capacity", type=int, default=2048)
    serve.add_argument("--queue-max-bytes", type=int, default=64 * 1024 * 1024)
    serve.add_argument("--max-event-bytes", type=int, default=2 * 1024 * 1024)
    serve.add_argument("--max-frame-bytes", type=int, default=DEFAULT_MAX_FRAME_BYTES)
    serve.add_argument("--batch-max-events", type=int, default=8)
    serve.add_argument("--batch-max-bytes", type=int, default=2 * 1024 * 1024)
    serve.add_argument("--batch-wait-ms", type=int, default=250)
    serve.add_argument("--max-text-leaves", type=int, default=8192)
    serve.add_argument("--max-text-bytes", type=int, default=4 * 1024 * 1024)
    serve.add_argument("--cache-entries", type=int, default=16384)
    serve.add_argument("--detector-endpoint", default="http://127.0.0.1:8001/v1")
    serve.add_argument("--detector-model", default="fastino/gliner2-privacy-filter-PII-multi")
    serve.add_argument("--detector-api-key-env", default="EMPTY")
    serve.add_argument("--evaluator-endpoint", default="https://integrate.api.nvidia.com/v1")
    serve.add_argument("--evaluator-model", default="nvidia/nemotron-3.5-lightning-30b-a3b")
    serve.add_argument("--evaluator-api-key-env", default="NVIDIA_API_KEY")
    serve.add_argument("--threshold", type=float, default=0.3)
    serve.add_argument("--provider-timeout-seconds", type=int, default=300)
    serve.add_argument(
        "--data-summary",
        default="Copied AI-agent observability containing prompts, model responses, and tool traffic.",
    )
    for command in ("health", "status", "flush", "shutdown"):
        control = subparsers.add_parser(command)
        control.add_argument("--endpoint", required=True)
        control.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser


async def _serve(args: argparse.Namespace) -> None:
    backend_config = BackendConfig(
        detector_endpoint=args.detector_endpoint,
        detector_model=args.detector_model,
        detector_api_key_env=args.detector_api_key_env,
        evaluator_endpoint=args.evaluator_endpoint,
        evaluator_model=args.evaluator_model,
        evaluator_api_key_env=args.evaluator_api_key_env,
        threshold=args.threshold,
        timeout_seconds=args.provider_timeout_seconds,
        data_summary=args.data_summary,
    )
    require_api_keys(backend_config)
    sink = JsonlSink(args.output)
    exporter = ProtectedExporter(
        ObservationSanitizer(
            AnonymizerBackend(backend_config),
            max_leaves=args.max_text_leaves,
            max_bytes=args.max_text_bytes,
            cache_entries=args.cache_entries,
            replacement_template="[REDACTED]",
        ),
        sink,
        queue_capacity=args.queue_capacity,
        queue_max_bytes=args.queue_max_bytes,
        max_event_bytes=args.max_event_bytes,
        batch_max_events=args.batch_max_events,
        batch_max_bytes=args.batch_max_bytes,
        batch_wait_seconds=args.batch_wait_ms / 1000,
        metrics_path=args.metrics,
    )
    server = ExporterServer(
        args.endpoint,
        exporter,
        max_frame_bytes=args.max_frame_bytes,
    )
    await server.start()
    loop = asyncio.get_running_loop()
    for handled_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                handled_signal,
                lambda: asyncio.create_task(server.stop()),
            )
        except NotImplementedError:
            pass
    try:
        await server.wait()
    finally:
        await server.stop()


async def _control(args: argparse.Namespace) -> None:
    response = await exchange(
        args.endpoint,
        {"kind": args.command},
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(response, indent=2, sort_keys=True))
    if (
        response.get("status") in {"busy", "error", "failed", "full", "oversize", "stopping"}
        and args.command != "shutdown"
    ):
        raise SystemExit(1)


def main() -> None:
    args = _parser().parse_args()
    if args.command == "serve":
        asyncio.run(_serve(args))
    else:
        asyncio.run(_control(args))


if __name__ == "__main__":
    main()
