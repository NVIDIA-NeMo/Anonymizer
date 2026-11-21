#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate input-variant files for dogfooding `dogfood_run.py --data`.

Builds a small set of CSV/Parquet files (5 rows each, from the head of a source CSV) that
each reproduce one "nullable or non-string columns" surface: `None`, a CSV-empty field that
reads back as NaN, `int`, `string[pyarrow]` with `pd.NA`, an all-null column, empty string,
whitespace-only string, `category`, `large_string` (pyarrow, written without going through
pandas' own large_string dtype), zero rows, and a row with odd passthrough-column dtypes
(category, nullable Int64, datetime64[ns, UTC], nullable boolean, float with NaN, and a list
column). Each variant is small enough to dogfood a single surface in one `dogfood_run.py`
command, pointed at `--text-column biography` (or your own `--text-column`, matching
whatever text column name you pass to `--text-column` below).

Uses only pandas/pyarrow — no Anonymizer import at all, so it has nothing to be out of sync
with across Anonymizer versions.

Usage: python make_variants.py SOURCE_CSV OUT_DIR [--text-column biography]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_csv", help="CSV with at least one text column and >=5 rows.")
    parser.add_argument("out_dir", help="Directory to write variant files into.")
    parser.add_argument("--text-column", default="biography", help="Name of the text column (default: biography).")
    parser.add_argument("--rows", type=int, default=5, help="Row count per variant (default: 5).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source_csv, out_dir = Path(args.source_csv), Path(args.out_dir)
    text_col = args.text_column
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(source_csv).head(args.rows)
    if text_col not in base.columns:
        raise SystemExit(f"{source_csv}: no column {text_col!r} (columns: {list(base.columns)})")
    base.insert(0, "record_id", pd.Series([f"r{i:03d}" for i in range(len(base))]))
    texts = list(base[text_col])

    def write(name: str, frame: pd.DataFrame) -> None:
        path = out_dir / name
        if name.endswith(".parquet"):
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)

    def with_text(values: object) -> pd.DataFrame:
        frame = base.copy()
        frame[text_col] = values
        return frame

    write("D1_none.parquet", with_text(pd.Series(texts[:2] + [None] + texts[3:], dtype=object)))
    write("D2_nan.csv", with_text(texts[:2] + [""] + texts[3:]))  # empty CSV field -> NaN on read
    write("D3_empty_string.parquet", with_text(texts[:2] + [""] + texts[3:]))
    write("D4_whitespace.parquet", with_text(texts[:2] + ["   "] + texts[3:]))
    write("D5_int.csv", with_text([1000 + i for i in range(len(base))]))
    write("D6_string_pyarrow_na.parquet", with_text(pd.array(texts[:2] + [pd.NA] + texts[3:], dtype="string[pyarrow]")))
    write("D7_string_pyarrow.parquet", with_text(pd.array(texts, dtype="string[pyarrow]")))
    write("D8_category.parquet", with_text(pd.Categorical(texts)))
    write("D9_all_null.parquet", with_text(pd.Series([None] * len(base), dtype=object)))
    write("D10_zero_rows.csv", base.iloc[:0])
    write("D10_zero_rows.parquet", base.iloc[:0])

    pq.write_table(
        pa.table({text_col: pa.array(texts, type=pa.large_string()), "record_id": pa.array(list(base["record_id"]))}),
        out_dir / "D11_large_string.parquet",
    )

    odd = base.copy()
    n = len(odd)
    odd["cat"] = pd.Categorical((["a", "b", "c"] * n)[:n])
    odd["nullable_int"] = pd.array([(i if i % 2 == 0 else None) for i in range(n)], dtype="Int64")
    odd["ts"] = pd.to_datetime([None if i % 3 == 0 else "2024-01-01" for i in range(n)], utc=True)
    odd["nullable_bool"] = pd.array([bool(i % 2) if i % 2 else None for i in range(n)], dtype="boolean")
    odd["amount"] = [float(i) if i % 2 else np.nan for i in range(n)]
    odd["tags"] = [["x"] if i % 2 else [] for i in range(n)]
    write("E1_odd_passthrough_dtypes.parquet", odd)

    print(f"wrote {len(list(out_dir.iterdir()))} variant files to {out_dir}")
    for path in sorted(out_dir.iterdir()):
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        dtype = frame[text_col].dtype if text_col in frame else "-"
        nonstr = [i for i, v in enumerate(frame[text_col]) if not isinstance(v, str)] if text_col in frame else "-"
        print(f"  {path.name:34s} rows={len(frame):2d} {text_col}={dtype} nonstr_rows={nonstr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
