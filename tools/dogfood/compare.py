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
                f"{run_dir}: dogfood_run.py errored ({meta.get('error_type')}) — no result file to compare. "
                "The exception message isn't in meta.json (never persisted); see that run's console output."
            )
        raise CompareError(f"{run_dir}: no result.parquet or result.csv found")

    if parquet_path.exists():
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        df = pd.DataFrame({name: table.column(name).to_pylist() for name in table.column_names})
    else:
        df = pd.read_csv(csv_path)
    return df, meta


def _load_trace(run_dir: Path) -> pd.DataFrame | None:
    trace_path = run_dir / "trace.csv"
    if not trace_path.exists():
        return None
    return pd.read_csv(trace_path)


def _load_input(run_dir: Path) -> pd.DataFrame | None:
    """Load the `input.{csv,parquet}` dogfood_run.py wrote for a `--call run` invocation.

    Not present for `--call preview` runs (preview reads `--data` directly, outside `--out`).
    """
    parquet_path = run_dir / "input.parquet"
    csv_path = run_dir / "input.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def _row0(df: pd.DataFrame, col: str, width: int = 260) -> str:
    if len(df) == 0:
        return "(0 rows)"
    return str(df[col].iloc[0])[:width]


def _entities_source(df: pd.DataFrame, trace_df: pd.DataFrame | None) -> pd.DataFrame:
    """Pick the frame to pull `final_entities` from.

    Rewrite's public `result.dataframe` has no `final_entities` column at all -- it
    only exists in `trace_dataframe` (`trace.csv`). Reading only `df` for a Rewrite run
    silently reports zero entities/row even when detection found plenty, which made the
    privacy/entity-overlap sections below look like a clean run instead of an untested
    one. Fall back to the trace frame whenever the primary frame lacks the column.
    """
    if "final_entities" in df.columns:
        return df
    if trace_df is not None and "final_entities" in trace_df.columns:
        return trace_df
    return df


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


# Columns the pipeline itself produces (metrics, entities, output text) rather than
# passthrough columns copied verbatim from the caller's input. Anything in
# trace_dataframe outside this set and not prefixed with "_" (internal/engine-only) is
# treated as a passthrough column for the parity check below.
_PIPELINE_COLUMNS = {
    "final_entities",
    "detected_entities",
    "utility_score",
    "leakage_mass",
    "weighted_leakage_rate",
    "any_high_leaked",
    "needs_human_review",
    "entity_coverage",
    "leaked_entities",
    "judge_evaluation",
    "detection_valid",
    "detection_invalid_entities",
}


def _passthrough_columns(trace_df: pd.DataFrame, text_col: str, out_col: str) -> list[str]:
    exclude = _PIPELINE_COLUMNS | {text_col, out_col}
    return sorted(
        c
        for c in trace_df.columns
        if not c.startswith("_") and c not in exclude and not c.endswith("_replaced") and not c.endswith("_rewritten")
    )


def _strategy_token_oracle(
    df: pd.DataFrame, entities: list[list[dict[str, Any]]], out_col: str
) -> list[tuple[int, Any]]:
    """Word-bounded leak check for Redact/Hash/Annotate: did an original entity value

    survive, as a whole word, in the strategy-token output? A plain substring test
    over-reports: a 1-2 digit entity value can appear inside a hex digest
    (e.g. Hash output) or inside another token without being a real leak. `\\b` is not
    enough here: `\\b` only fires at a transition between a word char and a non-word
    char, so it never matches at either end of a value that itself starts/ends with a
    non-word char (e.g. a phone number `"(555) 123-4567"` or a `+`-prefixed value) --
    `\\b` immediately before/after the escaped value then requires a word char there,
    which doesn't exist, and the match is silently skipped. `(?<!\\w)` / `(?!\\w)`
    anchor on the *absence* of a word char on the outside instead, which also matches
    at a string boundary or next to punctuation.
    """
    if out_col not in df.columns:
        return []
    leaks: list[tuple[int, Any]] = []
    for i, row in enumerate(entities):
        output_text = str(df[out_col].iloc[i])
        for entity in row:
            value = entity.get("value")
            if not value:
                continue
            if re.search(rf"(?<!\w){re.escape(str(value))}(?!\w)", output_text):
                leaks.append((i, value))
    return leaks


def _null_rows(input_df: pd.DataFrame, text_col: str) -> list[int]:
    if text_col not in input_df.columns:
        return []
    return [
        i
        for i, v in enumerate(input_df[text_col])
        if v is None or (isinstance(v, float) and pd.isna(v)) or not isinstance(v, str)
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir_a", help="First dogfood_run.py --out directory.")
    parser.add_argument("dir_b", help="Second dogfood_run.py --out directory.")
    parser.add_argument(
        "--mode",
        choices=["redact", "substitute", "annotate", "hash", "rewrite"],
        default="redact",
        help="Route that produced both runs.",
    )
    parser.add_argument("--text-column", default="biography", help="Text column name used for the dogfood_run.py runs.")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Also run pd.testing.assert_frame_equal(df_a, df_b) and exit 1 on mismatch. NOT a "
        "branch-vs-release regression gate for any route, including redact/hash/annotate: entity "
        "detection calls the configured chat model for LLM augmentation/validation regardless of "
        "replace strategy, so two separate dogfood_run.py invocations aren't guaranteed to detect "
        "the same entities. Meaningful only for a fixed-detection-input comparison (e.g. the same "
        "directory compared to itself). See README.md 'Running'.",
    )
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
    trace_a, trace_b = _load_trace(dir_a), _load_trace(dir_b)
    # Rewrite's public result has no final_entities column (see _entities_source);
    # pull entities from trace.csv instead when that's the case.
    ents_a, ents_b = _entities(_entities_source(df_a, trace_a)), _entities(_entities_source(df_b, trace_b))

    print(f"== {dir_a} vs {dir_b} ({args.mode})")

    # Status + module-qualified error-class parity first: the deciding signal for
    # whether two runs even failed the same way (dogfood_run.py records
    # f"{type(exc).__module__}.{type(exc).__qualname__}" as error_type).
    status_a, status_b = meta_a.get("status"), meta_b.get("status")
    error_type_a, error_type_b = meta_a.get("error_type"), meta_b.get("error_type")
    print(
        f"status: {status_a}/{status_b}  error_type parity: {error_type_a == error_type_b}  "
        f"({error_type_a!r} vs {error_type_b!r})"
    )
    print(
        f"seconds: {meta_a.get('seconds')} vs {meta_b.get('seconds')}  rows: {meta_a.get('rows')}/{meta_b.get('rows')}"
    )

    failed_a: list[Any] = meta_a.get("failed_records") or []
    failed_b: list[Any] = meta_b.get("failed_records") or []
    print(
        f"failed_records: {len(failed_a)}/{len(failed_b)}  "
        f"steps A: {meta_a.get('failed_record_steps') or {}}  steps B: {meta_b.get('failed_record_steps') or {}}"
    )
    print(
        f"skipped_span_count_total: {meta_a.get('skipped_span_count_total')} vs "
        f"{meta_b.get('skipped_span_count_total')}  "
        f"applied_span_count_total: {meta_a.get('applied_span_count_total')} vs "
        f"{meta_b.get('applied_span_count_total')}"
    )

    cols_equal = list(df_a.columns) == list(df_b.columns)
    cols_set_equal = set(df_a.columns) == set(df_b.columns)
    print(
        "result columns equal (order-sensitive):",
        cols_equal,
        "| set-equal:",
        cols_set_equal,
        "| A:",
        list(df_a.columns),
    )
    if not cols_equal:
        print("  B:", list(df_b.columns))

    # Trace columns compared as a set: order of independent DataDesigner columns
    # varies run to run in both versions (see H0 in the campaign RESULTS doc), so
    # order differences there are informational, not a parity finding.
    if trace_a is not None and trace_b is not None:
        trace_cols_a, trace_cols_b = list(trace_a.columns), list(trace_b.columns)
        print(
            "trace columns set-equal:",
            set(trace_cols_a) == set(trace_cols_b),
            "| order equal (informational only):",
            trace_cols_a == trace_cols_b,
        )
        if set(trace_cols_a) != set(trace_cols_b):
            print("  A-only:", sorted(set(trace_cols_a) - set(trace_cols_b)))
            print("  B-only:", sorted(set(trace_cols_b) - set(trace_cols_a)))
    else:
        print("trace columns: skipped (trace.csv missing on one or both sides)")

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
        # Word-bounded oracle for all four replace strategies. For
        # redact/annotate/hash the "entity value" should never survive as a whole
        # word in the strategy token; for substitute it is an LLM-generated
        # synthetic, so survival there is a direct privacy signal either way.
        for name, df, entities in ((str(dir_a), df_a, ents_a), (str(dir_b), df_b, ents_b)):
            leaks = _strategy_token_oracle(df, entities, out_col)
            print(f"{name}: entity values surviving (word-bounded) in {out_col}: {len(leaks)} {leaks[:8]}")
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

    # Passthrough-column dtype and value parity. Passthrough columns (caller
    # columns copied through unchanged) exist only in trace_dataframe, not in
    # result.dataframe.
    #
    # Dtype comparison uses meta.json["trace_dtypes"], recorded by dogfood_run.py from
    # the in-memory trace_dataframe at run time -- NOT trace_a[col].dtype /
    # trace_b[col].dtype read back from trace.csv. Plain CSV round-tripping infers
    # dtypes from values, so e.g. a nullable Int64 passthrough column reads back as
    # float64 on both sides regardless of whether the two trace_dataframes actually
    # agree; comparing post-CSV-read dtypes would always "match" and the check would be
    # vacuous. Value comparison still uses the CSV-read columns (values, not dtype,
    # survive that round-trip well enough for an equality check).
    if trace_a is not None and trace_b is not None:
        passthrough_a = _passthrough_columns(trace_a, text_col, out_col)
        passthrough_b = _passthrough_columns(trace_b, text_col, out_col)
        shared_passthrough = sorted(set(passthrough_a) & set(passthrough_b))
        print("passthrough columns A:", passthrough_a, "| B:", passthrough_b)
        trace_dtypes_a: dict[str, Any] = meta_a.get("trace_dtypes") or {}
        trace_dtypes_b: dict[str, Any] = meta_b.get("trace_dtypes") or {}
        if shared_passthrough and (not trace_dtypes_a or not trace_dtypes_b):
            print(
                "  dtype parity: skipped -- meta.json has no trace_dtypes on one or both sides "
                "(run produced by an older dogfood_run.py?)"
            )
        for col in shared_passthrough:
            if not trace_dtypes_a or not trace_dtypes_b:
                break
            dtype_a, dtype_b = trace_dtypes_a.get(col, "?"), trace_dtypes_b.get(col, "?")
            dtype_match = dtype_a == dtype_b
            n = min(len(trace_a), len(trace_b))
            value_match = (
                trace_a[col].iloc[:n].reset_index(drop=True).equals(trace_b[col].iloc[:n].reset_index(drop=True))
            )
            print(
                f"  {col}: dtype (at run time) {dtype_a}/{dtype_b} match={dtype_match}  "
                f"values (from trace.csv) match={value_match}"
            )

    # Null-row report: rows whose input text was null or not a str. Only
    # available for `--call run` directories, where dogfood_run.py writes
    # input.{csv,parquet} into --out.
    input_a, input_b = _load_input(dir_a), _load_input(dir_b)
    if input_a is not None and input_b is not None:
        null_rows_a = _null_rows(input_a, text_col)
        null_rows_b = _null_rows(input_b, text_col)
        print("null/non-str input text rows A:", null_rows_a, "B:", null_rows_b)
        for i in sorted(set(null_rows_a) | set(null_rows_b)):
            cell_a = df_a[out_col].iloc[i] if out_col in df_a.columns and i < len(df_a) else "(n/a)"
            cell_b = df_b[out_col].iloc[i] if out_col in df_b.columns and i < len(df_b) else "(n/a)"
            print(f"  row {i}: A[{out_col}]={cell_a!r}  B[{out_col}]={cell_b!r}")
    else:
        print("null-row report: skipped (input.csv/input.parquet not found in one or both --out dirs; --call preview?)")

    if args.exact:
        try:
            pd.testing.assert_frame_equal(df_a.reset_index(drop=True), df_b.reset_index(drop=True))
            print("--exact: PASS (df_a and df_b are identical)")
        except AssertionError as exc:
            print(f"--exact: FAIL\n{exc}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
