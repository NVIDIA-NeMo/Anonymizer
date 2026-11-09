# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standard-library server for testing the managed interpreter boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


class Handler(BaseHTTPRequestHandler):
    def authenticated(self) -> bool:
        key = os.environ.get("VLLM_API_KEY")
        if key is not None and self.headers.get("Authorization") != f"Bearer {key}":
            self.send_error(401)
            return False
        return True

    def do_GET(self) -> None:
        if not self.authenticated():
            return
        if self.path == "/_anonymizer/launch-ownership":
            token = os.environ["ANONYMIZER_INFERENCE_LAUNCH_TOKEN"]
            if self.headers.get("X-Anonymizer-Launch-Token") != token:
                self.send_error(404)
                return
            self.respond({"launch_token_sha256": hashlib.sha256(token.encode()).hexdigest()})
        else:
            self.respond({"data": [{"id": "openai/gpt-oss-20b"}]})

    def do_POST(self) -> None:
        if not self.authenticated():
            return
        self.rfile.read(int(self.headers["Content-Length"]))
        self.respond({"choices": [{"message": {"content": "ready"}}]})

    def respond(self, payload: object) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--validate", action="store_true")
    args, _ = parser.parse_known_args()
    if args.validate:
        return
    config = json.loads(args.config_json)
    Path("observed.json").write_text(
        json.dumps(
            {
                "prefix": sys.prefix,
                "pid": os.getpid(),
                "path": sys.path,
                "environment": {
                    name: os.environ.get(name)
                    for name in (
                        "PYTHONHOME",
                        "PYTHONPATH",
                        "VIRTUAL_ENV",
                        "CUDA_VISIBLE_DEVICES",
                        "HF_HOME",
                        "VLLM_API_KEY",
                    )
                },
            }
        )
    )
    HTTPServer((config["host"], config["port"]), Handler).serve_forever()


if __name__ == "__main__":
    main()
