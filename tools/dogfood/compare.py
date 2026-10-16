#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare two dogfood_run.py output directories (e.g. branch vs released wheel).

Loads result.parquet with pyarrow directly rather than pandas.read_parquet. The
Substitute route's `final_entities` column round-trips through parquet as a
pandas ArrowDtype-backed struct column
(`struct<entities: list<item: struct<value: string, label: string, ...>>>[pyarrow]`,
confirmed from the engine's EntitiesSchema -> DataDesigner output). Reading that
back with `pandas.read_parquet` raises, because pandas reconstructs the column's
extension dtype from the nested-struct type string recorded in the parquet
file's pandas metadata, and that reconstruction does not handle nested structs:
`ValueError: format number 1 of "struct<entities: list<item: struct<...>>>
[pyarrow]" is not recognized`. (A plain object-dtype column of Python dicts does
not hit this — only pandas' own ArrowDtype struct columns do, which is what NDD
produces.) Reading the Arrow table directly and converting each column with
`to_pylist()` sidesteps the pandas dtype reconstruction and keeps nested
structs usable; falls back to result.csv (with best-effort literal parsing of
the stringified struct) if no parquet file is present.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


class CompareError(Exception):
    """A run directory can't be compared (missing meta.json/result file, or the run errored)."""


def _load(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise CompareError(f"{run_dir}: no meta.json — did dogfood_run.py finish writing this directory?")
    meta: dict[str, Any] = json.loads(meta_path.read_text())

    parquet_path = run_dir / "result.parquet"
    csv_path = run_dir / "result.csv"
    if not parquet_path.exists() and not csv_path.exists():
        if meta.get("status") == "error":
            raise CompareError(
                f"{run_dir}: dogfood_run.py errored ({meta.get('error_type')}: {meta.get('error')}) "
                "— no result file to compare"
            )
        raise CompareError(f"{run_dir}: no result.parquet or result.csv found")

    if parquet_path.exists():
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        df = pd.DataFrame({name: table.column(name).to_pylist() for name in table.column_names})
    else:
        df = pd.read_csv(csv_path)
    return df, meta


def _row0(df: pd.DataFrame, col: str, width: int = 260) -> str:
    if len(df) == 0:
        return "(0 rows)"
    return str(df[col].iloc[0])[:width]


def _entities(df: pd.DataFrame) -> list[list[dict[str, Any]]]:
    if "final_entities" not in df.columns:
        return [[] for _ in range(len(df))]
    rows: list[list[dict[str, Any]]] = []
    for cell in df["final_entities"]:
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
        try:
            rows.append([dict(e) for e in (list(entities) if entities is not None else [])])
        except (TypeError, ValueError):
            rows.append([])
    return rows


def _entity_key(entity: dict[str, Any]) -> tuple[Any, Any]:
    return (entity.get("value"), entity.get("label"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir_a", help="First dogfood_run.py --out directory.")
    parser.add_argument("dir_b", help="Second dogfood_run.py --out directory.")
    parser.add_argument(
        "--mode", choices=["redact", "substitute", "rewrite"], default="redact", help="Route that produced both runs."
    )
    parser.add_argument("--text-column", default="biography", help="Text column name used for the dogfood_run.py runs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dir_a, dir_b = Path(args.dir_a), Path(args.dir_b)
    try:
        df_a, meta_a = _load(dir_a)
        df_b, meta_b = _load(dir_b)
    except CompareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    ents_a, ents_b = _entities(df_a), _entities(df_b)

    print(f"== {dir_a} vs {dir_b} ({args.mode})")
    failed_a: list[Any] = meta_a.get("failed_records") or []
    failed_b: list[Any] = meta_b.get("failed_records") or []
    print(
        f"status: {meta_a['status']}/{meta_b['status']}  seconds: {meta_a['seconds']} vs {meta_b['seconds']}  "
        f"rows: {meta_a['rows']}/{meta_b['rows']}  "
        f"failed_records: {len(failed_a)}/{len(failed_b)}"
    )
    cols_equal = list(df_a.columns) == list(df_b.columns)
    print("columns equal:", cols_equal, "| A:", list(df_a.columns))
    if not cols_equal:
        print("  B:", list(df_b.columns))

    dtypes_a: dict[str, Any] = meta_a.get("dtypes", {})
    dtypes_b: dict[str, Any] = meta_b.get("dtypes", {})
    dtype_diff = {
        c: (dtypes_a.get(c), dtypes_b.get(c))
        for c in set(dtypes_a) | set(dtypes_b)
        if dtypes_a.get(c) != dtypes_b.get(c)
    }
    print("dtype differences (A,B):", dtype_diff or "none")

    print("entities/row A:", [len(x) for x in ents_a], "total", sum(map(len, ents_a)))
    print("entities/row B:", [len(x) for x in ents_b], "total", sum(map(len, ents_b)))
    overlaps = [
        (
            len({_entity_key(e) for e in a} & {_entity_key(e) for e in b}),
            len({_entity_key(e) for e in a} - {_entity_key(e) for e in b}),
            len({_entity_key(e) for e in b} - {_entity_key(e) for e in a}),
        )
        for a, b in zip(ents_a, ents_b, strict=False)
    ]
    totals = tuple(map(sum, zip(*overlaps, strict=False))) if overlaps else None
    print("per-row (shared, A-only, B-only):", overlaps, "| totals:", totals)

    labels_a: set[str] = {str(e.get("label")) for row in ents_a for e in row}
    labels_b: set[str] = {str(e.get("label")) for row in ents_b for e in row}
    print(
        "labels A-only:",
        sorted(labels_a - labels_b),
        "B-only:",
        sorted(labels_b - labels_a),
        "shared:",
        len(labels_a & labels_b),
    )

    text_col = args.text_column
    out_col = f"{text_col}_rewritten" if args.mode == "rewrite" else f"{text_col}_replaced"
    if args.mode != "rewrite":
        for name, df, entities in ((str(dir_a), df_a, ents_a), (str(dir_b), df_b, ents_b)):
            if out_col not in df.columns:
                continue
            leaks = [
                (i, e.get("value"))
                for i, row in enumerate(entities)
                for e in row
                if e.get("value") and str(e["value"]) in str(df[out_col].iloc[i])
            ]
            print(f"{name}: entity values surviving in {out_col}: {len(leaks)} {leaks[:8]}")
        if out_col in df_a.columns:
            print(f"{dir_a} row0: {_row0(df_a, out_col)}")
        if out_col in df_b.columns:
            print(f"{dir_b} row0: {_row0(df_b, out_col)}")
    else:
        for name, df in ((str(dir_a), df_a), (str(dir_b), df_b)):
            metric_cols = [
                c
                for c in (
                    "utility_score",
                    "leakage_mass",
                    "weighted_leakage_rate",
                    "any_high_leaked",
                    "needs_human_review",
                )
                if c in df.columns
            ]
            summary = {}
            for col in metric_cols:
                if col in ("any_high_leaked", "needs_human_review"):
                    summary[col] = df[col].astype(str).value_counts().to_dict()
                else:
                    numeric = pd.to_numeric(df[col], errors="coerce")
                    summary[col] = (round(float(numeric.min()), 3), round(float(numeric.max()), 3))
            print(f"{name}: rewrite metrics:", summary)
            if out_col in df.columns:
                print(f"{name}: row0 rewritten: {_row0(df, out_col, width=300)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
