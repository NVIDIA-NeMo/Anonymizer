# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standard-library stand-ins for managed services.

Default mode serves the managed interpreter boundary. ``group`` mode starts a process
group of sleeping workers for stop and ownership tests:

    inference_process.py group --workers N [--ignore-term] [--leader-exits] [--hold-mib M] --ready PATH

The leader writes the worker PIDs to ``PATH`` once every worker has installed its
signal handling, then sleeps, or exits when ``--leader-exits`` is given. ``--hold-mib`` makes each
worker hold that much touched memory, so it takes measurable time to die after SIGKILL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
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


def run_group(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ignore-term", action="store_true")
    parser.add_argument("--leader-exits", action="store_true")
    parser.add_argument("--hold-mib", type=int, default=0)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args(argv)
    worker_argv = [
        sys.executable,
        __file__,
        "worker",
        str(args.hold_mib),
        *(["--ignore-term"] if args.ignore_term else []),
    ]
    workers = [subprocess.Popen(worker_argv, stdout=subprocess.PIPE) for _ in range(args.workers)]
    for worker in workers:
        assert worker.stdout is not None
        worker.stdout.readline()
    if args.ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    args.ready.write_text(json.dumps({"workers": [worker.pid for worker in workers]}))
    if args.leader_exits:
        os._exit(0)
    sleep_forever()


def run_worker(argv: list[str]) -> None:
    if "--ignore-term" in argv:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    held = bytearray(int(argv[0]) << 20)
    print("ready", flush=True)
    sleep_forever()
    del held


def sleep_forever() -> None:
    while True:
        time.sleep(60)


def main() -> None:
    if sys.argv[1:2] == ["group"]:
        run_group(sys.argv[2:])
        return
    if sys.argv[1:2] == ["worker"]:
        run_worker(sys.argv[2:])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-spec-json", required=True)
    parser.add_argument("--validate", action="store_true")
    args, _ = parser.parse_known_args()
    if args.validate:
        return
    spec = json.loads(args.launch_spec_json)
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
    HTTPServer((spec["host"], spec["port"]), Handler).serve_forever()


if __name__ == "__main__":
    main()
