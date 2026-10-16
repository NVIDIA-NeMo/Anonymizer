#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dogfood runner: exercises one route (Redact / Substitute / Rewrite) against one
Anonymizer environment and writes the result plus a diagnostic bundle to --out.

Run the same command against a checked-out branch and a released wheel (see
tools/dogfood/README.md for the two-environment setup), then compare the two
--out directories with compare.py.

Uses only the public Anonymizer API — no engine/graph imports.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd


def _data_designer_version() -> str:
    try:
        from importlib.metadata import version

        return version("data-designer")
    except Exception:
        return "?"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["redact", "substitute", "rewrite"], required=True, help="Replace/rewrite route to exercise."
    )
    parser.add_argument("--call", choices=["run", "preview"], default="run", help="Anonymizer method to call.")
    parser.add_argument(
        "--n", type=int, default=10, help="Row count for run (head of --data) or preview (num_records)."
    )
    parser.add_argument("--data", required=True, help="Path to a CSV/Parquet input file.")
    parser.add_argument("--text-column", default="biography", help="Name of the text column in --data.")
    parser.add_argument("--id-column", default=None, help="Optional record-identifier column in --data.")
    parser.add_argument("--data-summary", default=None, help="Optional short dataset description to improve detection.")
    parser.add_argument("--configs", required=True, help="Path to a unified model_configs YAML.")
    parser.add_argument("--providers", required=True, help="Path to a providers YAML.")
    parser.add_argument(
        "--out", required=True, help="Output directory for result.{csv,parquet}, trace.csv, log.txt, meta.json."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import anonymizer
    from anonymizer import (
        Anonymizer,
        AnonymizerConfig,
        AnonymizerInput,
        LoggingConfig,
        Redact,
        Rewrite,
        Substitute,
        configure_logging,
    )

    configure_logging(LoggingConfig(anonymizer_level="DEBUG", data_designer_level="INFO"))
    file_handler = logging.FileHandler(out_dir / "log.txt", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.DEBUG)

    meta: dict[str, object] = {
        "anonymizer_version": anonymizer.__version__,
        "data_designer_version": _data_designer_version(),
        "python": sys.version.split()[0],
        "mode": args.mode,
        "call": args.call,
        "n": args.n,
    }

    source_df = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_parquet(args.data)
    if args.call == "run":
        source_df = source_df.head(args.n)
        data_path = out_dir / "input.csv"
        source_df.to_csv(data_path, index=False)
    else:
        data_path = Path(args.data)

    strategy = {"redact": Redact(), "substitute": Substitute()}.get(args.mode)
    config = (
        AnonymizerConfig(rewrite=Rewrite(), emit_telemetry=False)
        if strategy is None
        else AnonymizerConfig(replace=strategy, emit_telemetry=False)
    )
    data = AnonymizerInput(
        source=str(data_path),
        text_column=args.text_column,
        id_column=args.id_column,
        data_summary=args.data_summary,
    )

    start = time.monotonic()
    try:
        anon = Anonymizer(model_configs=args.configs, model_providers=args.providers)
        result = (
            anon.run(config=config, data=data)
            if args.call == "run"
            else anon.preview(config=config, data=data, num_records=args.n)
        )
    except BaseException as exc:  # noqa: BLE001 - this is a diagnostic harness, not library code
        meta.update(
            status="error",
            seconds=round(time.monotonic() - start, 1),
            error_type=type(exc).__name__,
            error=str(exc)[:2000],
            traceback=traceback.format_exc()[-6000:],
        )
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        print(json.dumps({k: meta[k] for k in ("status", "error_type", "seconds")}))
        return 2

    meta["status"] = "ok"
    meta["seconds"] = round(time.monotonic() - start, 1)

    df = result.dataframe
    df.to_csv(out_dir / "result.csv", index=False)
    try:
        df.to_parquet(out_dir / "result.parquet")
    except Exception as exc:  # noqa: BLE001
        meta["parquet_error"] = str(exc)[:200]

    meta["columns"] = list(df.columns)
    meta["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    meta["rows"] = int(len(df))
    meta["input_rows"] = int(len(source_df)) if args.call == "run" else args.n

    trace_df = result.trace_dataframe
    try:
        trace_df.to_csv(out_dir / "trace.csv", index=False)
        meta["trace_columns"] = list(trace_df.columns)
        meta["trace_rows"] = int(len(trace_df))
    except Exception as exc:  # noqa: BLE001
        meta["trace_error"] = str(exc)[:200]

    failed = [{"record_id": r.record_id, "step": r.step, "reason": r.reason[:300]} for r in result.failed_records]
    meta["failed_records"] = failed

    # --- lightweight oracle checks (not a substitute for compare.py) ---
    checks: dict[str, object] = {}
    text_col = args.text_column
    out_col = f"{text_col}_rewritten" if args.mode == "rewrite" else f"{text_col}_replaced"
    checks["output_column_present"] = out_col in df.columns
    checks["row_count_matches"] = meta["rows"] == meta["input_rows"]

    if args.mode == "rewrite":
        expected = {
            f"{text_col}_rewritten",
            "utility_score",
            "leakage_mass",
            "weighted_leakage_rate",
            "any_high_leaked",
            "needs_human_review",
        }
        checks["rewrite_missing_columns"] = sorted(expected - set(df.columns))
        for col in ("utility_score", "leakage_mass"):
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                checks[col] = {
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                    "nan": int(numeric.isna().sum()),
                }

    meta["checks"] = checks
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(
        json.dumps(
            {
                "status": "ok",
                "seconds": meta["seconds"],
                "rows": meta["rows"],
                "cols": len(df.columns),
                "failed_records": len(failed),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
