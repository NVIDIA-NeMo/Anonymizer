#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scan a dogfood_run.py --out directory's log.txt for entity-value leaks.

log.txt should contain no entity values (see README.md "What to look at"). This script
checks that mechanically instead of hand-grepping: it collects entity values (len >= 4)
from result.{parquet,csv}'s `final_entities` column -- falling back to trace.csv's
`final_entities` when the result lacks that column, which is always true for Rewrite:
Rewrite's public result has no `final_entities` column at all (only trace.csv does), so
without the fallback every Rewrite run reports zero probes and exits 3 regardless of how
many entities detection actually found -- and searches for them in log.txt, with one
filter: lines that echo the detector's configured label list
(`extra_body={'labels': [...]}`) are ignored, because a short entity value can
legitimately appear inside a label name there (e.g. the label `http_cookie` containing
the entity value `http`). Any other match is a `[privacy]` finding.

This is a privacy check, so it fails closed: a run directory counts as a FAIL, not a
silent pass, whenever the check could not actually run. Exit codes:

  0  every given directory was scanned and had zero (word-bounded) matches.
  1  at least one directory is missing (not a directory, no log.txt, or no result
     file), or at least one directory had a leak -- a real `[privacy]` finding.
  3  no directory was missing or leaked, but at least one had zero entity-value probes
     (e.g. a run over an all-null/no-PII input variant) -- nothing was checked for that
     directory, which is not the same claim as "no leaks found", so it's reported
     distinctly from exit 0 and from exit 1's input problems.

Usage: python scan_log.py OUT_DIR [OUT_DIR ...]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

_LABEL_LIST_MARKER = "extra_body={'labels'"
_MIN_VALUE_LEN = 4


def _entity_values_from_cells(cells: list[Any]) -> set[str]:
    """Extract entity `value`s (len >= _MIN_VALUE_LEN) from a `final_entities` column.

    Handles both forms the column can arrive in: a pandas ArrowDtype struct (from
    reading result.parquet with pyarrow -- already plain Python dicts/lists via
    to_pylist()) and a stringified dict/list (from reading a CSV -- JSON first, then a
    best-effort `ast.literal_eval` fallback for the numpy-repr form pandas sometimes
    produces, e.g. `array([...])`).
    """
    values: set[str] = set()
    for cell in cells:
        parsed: Any = cell
        if isinstance(cell, str):
            try:
                parsed = json.loads(cell)
            except (json.JSONDecodeError, TypeError):
                try:
                    parsed = ast.literal_eval(re.sub(r"\barray\(", "list(", cell))
                except (ValueError, SyntaxError):
                    parsed = {}
        entities = parsed.get("entities") if isinstance(parsed, dict) else parsed
        for entity in list(entities) if entities is not None else []:
            value = dict(entity).get("value")
            if isinstance(value, str) and len(value) >= _MIN_VALUE_LEN:
                values.add(value)
    return values


def _load_entity_values(out_dir: Path) -> set[str]:
    """Collect entity-value probes from the result, falling back to the trace.

    Rewrite's public `result.dataframe` has no `final_entities` column at all -- it
    only exists in `trace_dataframe` (`trace.csv`). Without this fallback, every
    Rewrite run would report zero probes (status "no_entities", exit 3) regardless of
    how many entities detection actually found, which made the privacy check blind on
    the one route (Rewrite) where the rewritten text most needs checking for leaks.
    """
    parquet_path = out_dir / "result.parquet"
    csv_path = out_dir / "result.csv"
    cells: list[Any] = []
    if parquet_path.exists():
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        if "final_entities" in table.column_names:
            cells = table.column("final_entities").to_pylist()
    elif csv_path.exists():
        import pandas as pd

        df = pd.read_csv(csv_path)
        if "final_entities" in df.columns:
            cells = list(df["final_entities"])

    if not cells:
        trace_path = out_dir / "trace.csv"
        if trace_path.exists():
            import pandas as pd

            trace_df = pd.read_csv(trace_path)
            if "final_entities" in trace_df.columns:
                cells = list(trace_df["final_entities"])

    return _entity_values_from_cells(cells)


def _filtered_log_text(out_dir: Path) -> str:
    log_path = out_dir / "log.txt"
    lines = log_path.read_text(errors="replace").splitlines(keepends=True)
    return "".join(line for line in lines if _LABEL_LIST_MARKER not in line)


def scan(out_dir: Path) -> dict[str, Any]:
    """Scan one run directory. Always returns a dict with a "status" key:

    - "missing_inputs": the check couldn't run at all (no such directory, no log.txt,
      or no result file). Exit 1.
    - "no_entities": the check ran but found zero entity-value probes to search for
      (e.g. an all-null/no-PII input variant) -- "nothing was checked" is not the same
      claim as "no leaks found", so this is reported distinctly from both "clean" and
      "missing_inputs". Exit 3.
    - "leak": the check ran, had probes, and found at least one match. Exit 1.
    - "clean": the check ran, had probes, and found zero matches. Exit 0.
    """
    run_name = out_dir.name or str(out_dir)
    if not out_dir.is_dir():
        return {"run": run_name, "status": "missing_inputs", "reason": f"not a directory: {out_dir}"}
    if not (out_dir / "log.txt").exists():
        return {"run": run_name, "status": "missing_inputs", "reason": "log.txt not found"}
    if not (out_dir / "result.parquet").exists() and not (out_dir / "result.csv").exists():
        return {"run": run_name, "status": "missing_inputs", "reason": "no result.parquet or result.csv found"}

    values = _load_entity_values(out_dir)
    if not values:
        return {
            "run": run_name,
            "status": "no_entities",
            "reason": "zero entity-value probes collected from final_entities -- nothing was checked",
        }

    text = _filtered_log_text(out_dir)
    # (?<!\w) / (?!\w) rather than \b: \b only fires at a transition between a word
    # char and a non-word char, so it never matches at either end of a value that
    # itself starts/ends with a non-word char (e.g. a phone number or a
    # punctuation-leading token) -- the match would be silently skipped.
    hits = {value: len(re.findall(rf"(?<!\w){re.escape(value)}(?!\w)", text)) for value in values}
    hits = {value: count for value, count in hits.items() if count > 0}
    return {
        "run": run_name,
        "status": "leak" if hits else "clean",
        "log_bytes": len(text),
        "probes": len(values),
        "hits": len(hits),
        "examples": sorted(hits, key=lambda v: -hits[v])[:5],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dirs", nargs="+", metavar="OUT_DIR", help="dogfood_run.py --out directory to scan.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    statuses: set[str] = set()
    for raw_dir in args.out_dirs:
        result = scan(Path(raw_dir))
        statuses.add(result["status"])
        print(json.dumps(result))
    if statuses & {"missing_inputs", "leak"}:
        return 1
    if "no_entities" in statuses:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
