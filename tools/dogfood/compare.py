#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare two dogfood_run.py output directories (e.g. branch vs released wheel).

Loads result.parquet with pyarrow directly rather than pandas.read_parquet: the
Substitute route's `final_entities` column is an Arrow struct<entities: list<struct<...>>>,
which pandas cannot deserialize back into Python objects from parquet. Reading
the Arrow table and converting each column with `to_pylist()` keeps nested
structs usable; falls back to result.csv (with best-effort literal parsing of
the stringified struct) if no parquet file is present.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _load(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = json.loads((run_dir / "meta.json").read_text())
    parquet_path = run_dir / "result.parquet"
    if parquet_path.exists():
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        df = pd.DataFrame({name: table.column(name).to_pylist() for name in table.column_names})
    else:
        df = pd.read_csv(run_dir / "result.csv")
    return df, meta


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
    df_a, meta_a = _load(dir_a)
    df_b, meta_b = _load(dir_b)
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
        print(f"{dir_a} row0: {str(df_a[out_col].iloc[0])[:260]}")
        print(f"{dir_b} row0: {str(df_b[out_col].iloc[0])[:260]}")
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
                print(f"{name}: row0 rewritten: {str(df[out_col].iloc[0])[:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
