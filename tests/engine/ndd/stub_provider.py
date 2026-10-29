# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A loopback chat-completions server whose replies a test scripts per request."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from data_designer.config.models import ModelProvider

# (model name, prompt text, number of earlier requests with the same prompt and model) -> reply
Reply = tuple[int, str, str]  # (http status, content, finish_reason)
Script = Callable[[str, str, int], Reply]

COMPLETE = (200, '{"label": "ok", "score": 1}', "stop")


def cut_off(prefix: str = "") -> Reply:
    """A JSON object that stops mid-field because the token budget ran out."""
    return 200, f'{{"label": "{prefix}', "length"


@contextmanager
def stub_provider(script: Script) -> Iterator[ModelProvider]:
    seen: dict[tuple[str, str], int] = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = json.dumps(body["messages"])
            with lock:
                key = (body["model"], prompt)
                earlier = seen.get(key, 0)
                seen[key] = earlier + 1
            status, content, finish_reason = script(body["model"], prompt, earlier)
            if status == 200:
                payload = {
                    "id": "stub",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
                }
            else:
                payload = {"error": {"message": content, "type": "stub_error"}}
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ModelProvider(
            name="stub",
            endpoint=f"http://127.0.0.1:{server.server_address[1]}/v1",
            provider_type="openai",
            api_key="EMPTY",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
