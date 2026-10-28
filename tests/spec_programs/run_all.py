# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run every spec program and write one JSON line per check: `uv run python -m tests.spec_programs.run_all [out]`."""

from __future__ import annotations

import importlib
import json
import sys

PROGRAMS = [
    "sp01_execution_only",
    "sp02_verification",
    "sp03_admission",
    "sp04_initial_binding",
    "sp05_inspection",
    "sp06_context_sources",
    "sp07_offsets",
    "sp09_request_forecast",
    "sp10_nested_frames",
    "sp11_cancellation",
    "sp12_ownership",
    "sp13_identity",
    "sp14_invocation_operation",
    "sp15_stage_sequence",
    "sp16_bounded_operation",
]
RETIRED_SP8 = [f"E8.{i}" for i in range(1, 6)]


def collect() -> list[dict[str, str]]:
    lines: list[dict[str, str]] = []
    for name in PROGRAMS:
        module = importlib.import_module(f"tests.spec_programs.{name}")
        try:
            lines.extend(module.program())
        except Exception as exc:  # noqa: BLE001 - a crashed program is reported, not hidden
            lines.append({"program": name, "check": "*", "result": "FAIL", "detail": f"crashed: {type(exc).__name__}"})
        if name == "sp07_offsets":
            lines += [{"program": "SP-8", "check": e, "result": "RETIRED", "detail": "retired"} for e in RETIRED_SP8]
    return lines


def main() -> int:
    lines = collect()
    text = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 1 if any(line["result"] == "FAIL" for line in lines) else 0


if __name__ == "__main__":
    raise SystemExit(main())
