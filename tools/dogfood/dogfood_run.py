#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dogfood runner: exercises one route (Redact / Substitute / Annotate / Hash / Rewrite)
against one Anonymizer environment and writes the result plus a diagnostic bundle to --out.

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
from collections import Counter
from pathlib import Path
from typing import Any

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
        "--mode",
        choices=["redact", "substitute", "annotate", "hash", "rewrite"],
        required=True,
        help="Replace/rewrite route to exercise.",
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

    strategy_group = parser.add_argument_group("replace strategy options (redact / annotate / hash)")
    strategy_group.add_argument(
        "--format-template",
        default=None,
        help="Redaction/annotation/hash template override (see Redact/Annotate/Hash docstrings for placeholders).",
    )
    strategy_group.add_argument(
        "--no-normalize-label",
        action="store_true",
        help="Redact only: disable uppercasing/cleaning the label before substitution.",
    )
    strategy_group.add_argument(
        "--hash-algorithm",
        choices=["sha256", "sha1", "md5"],
        default=None,
        help="Hash only: digest algorithm (default sha256).",
    )
    strategy_group.add_argument(
        "--digest-length", type=int, default=None, help="Hash only: hex characters kept from the digest (default 12)."
    )

    rewrite_group = parser.add_argument_group("rewrite options")
    rewrite_group.add_argument(
        "--strict",
        action="store_true",
        help="Rewrite only: strict_entity_protection (require a protective disposition for every entity).",
    )
    rewrite_group.add_argument(
        "--risk-tolerance",
        choices=["minimal", "low", "moderate", "high"],
        default=None,
        help="Rewrite only: RiskTolerance preset (default low).",
    )
    rewrite_group.add_argument(
        "--max-repair-iterations", type=int, default=None, help="Rewrite only: max repair rounds (default 3)."
    )
    rewrite_group.add_argument(
        "--combined",
        action="store_true",
        help="Rewrite only: use_combined_graph (run rewrite + conditional repair in one Data Designer graph).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import anonymizer
    from anonymizer import (
        Annotate,
        Anonymizer,
        AnonymizerConfig,
        AnonymizerInput,
        Hash,
        LoggingConfig,
        Redact,
        Rewrite,
        RiskTolerance,
        Substitute,
        configure_logging,
    )

    configure_logging(LoggingConfig(anonymizer_level="DEBUG", data_designer_level="INFO"))
    # `configure_logging` attaches its console handler directly to the
    # "anonymizer" and "data_designer" loggers and sets propagate=False on
    # both (src/anonymizer/logging.py), so a handler on the root logger never
    # sees these records. Attach the file handler to the same two loggers,
    # with the same formatter, so log.txt mirrors what the console prints.
    file_handler = logging.FileHandler(out_dir / "log.txt", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"))
    logging.getLogger("anonymizer").addHandler(file_handler)
    logging.getLogger("data_designer").addHandler(file_handler)

    meta: dict[str, object] = {
        "anonymizer_version": anonymizer.__version__,
        "data_designer_version": _data_designer_version(),
        "python": sys.version.split()[0],
        "mode": args.mode,
        "call": args.call,
        "n": args.n,
        "started_epoch": time.time(),
    }

    is_parquet = not args.data.endswith(".csv")
    source_df = pd.read_parquet(args.data) if is_parquet else pd.read_csv(args.data)
    if args.call == "run":
        source_df = source_df.head(args.n)
        # Keep the input format: a parquet source stays parquet so dtypes
        # (string[pyarrow] NA, category, large_string, nullable Int64, list
        # columns, …) reach the pipeline instead of collapsing through CSV.
        if is_parquet:
            data_path = out_dir / "input.parquet"
            source_df.to_parquet(data_path, index=False)
        else:
            data_path = out_dir / "input.csv"
            source_df.to_csv(data_path, index=False)
    else:
        data_path = Path(args.data)

    strategy_kwargs: dict[str, Any] = {}
    if args.format_template is not None:
        strategy_kwargs["format_template"] = args.format_template
    strategy: Redact | Substitute | Annotate | Hash | None
    if args.mode == "redact":
        if args.no_normalize_label:
            strategy_kwargs["normalize_label"] = False
        strategy = Redact(**strategy_kwargs)
    elif args.mode == "substitute":
        strategy = Substitute()
    elif args.mode == "annotate":
        strategy = Annotate(**strategy_kwargs)
    elif args.mode == "hash":
        if args.hash_algorithm is not None:
            strategy_kwargs["algorithm"] = args.hash_algorithm
        if args.digest_length is not None:
            strategy_kwargs["digest_length"] = args.digest_length
        strategy = Hash(**strategy_kwargs)
    else:
        strategy = None

    if strategy is None:
        rewrite_kwargs: dict[str, Any] = {
            "strict_entity_protection": args.strict,
            "use_combined_graph": args.combined,
        }
        if args.risk_tolerance is not None:
            rewrite_kwargs["risk_tolerance"] = RiskTolerance(args.risk_tolerance)
        if args.max_repair_iterations is not None:
            rewrite_kwargs["max_repair_iterations"] = args.max_repair_iterations
        config = AnonymizerConfig(rewrite=Rewrite(**rewrite_kwargs), emit_telemetry=False)
    else:
        config = AnonymizerConfig(replace=strategy, emit_telemetry=False)
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
        exc_type = type(exc)
        error_type = f"{exc_type.__module__}.{exc_type.__qualname__}"
        # Module-qualified so e.g. an engine-internal rejection type can't be confused with a
        # public anonymizer.interface.errors class of the same name.
        #
        # The exception's own message/str() is deliberately never persisted to
        # meta.json (or anywhere else this kit writes to disk). A backend exception can
        # embed input record content verbatim -- not just the offending cell's full
        # text, but a fragment of it, a pydantic ValidationError's truncated
        # "input_value=...", text from a different (non-text) column, or LLM-generated
        # text that was never a literal substring of the input. A whole-cell-substring
        # scrub cannot catch all of those, so there is no safe redaction of the message
        # short of not writing it at all. Only the error class and a bare stack
        # (file:function:lineno per frame, explicitly excluding
        # `FrameSummary.line` -- the source line text) go to meta.json; the full
        # message is printed to the console only, for whoever is running this
        # interactively.
        print(f"[dogfood_run] {error_type}: {exc}", file=sys.stderr)
        traceback_frames = [
            f"{frame.filename}:{frame.name}:{frame.lineno}" for frame in traceback.extract_tb(exc.__traceback__)
        ]
        meta.update(
            status="error",
            seconds=round(time.monotonic() - start, 1),
            ended_epoch=time.time(),
            error_type=error_type,
            traceback_frames=traceback_frames,
        )
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        print(json.dumps({k: meta[k] for k in ("status", "error_type", "seconds")}))
        return 2

    meta["status"] = "ok"
    meta["seconds"] = round(time.monotonic() - start, 1)
    meta["ended_epoch"] = time.time()

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
    # Record dtypes from the in-memory trace_dataframe, before any CSV round-trip.
    # trace.csv loses dtype fidelity on read-back (e.g. a nullable Int64 passthrough
    # column reads back as float64 on both sides of a comparison, via plain CSV type
    # inference), which would make a dtype-parity check that re-reads trace.csv
    # vacuous -- it would "match" even when the two trace_dataframes actually differ.
    meta["trace_dtypes"] = {c: str(t) for c, t in trace_df.dtypes.items()}
    try:
        trace_df.to_csv(out_dir / "trace.csv", index=False)
        meta["trace_columns"] = list(trace_df.columns)
        meta["trace_rows"] = int(len(trace_df))
    except Exception as exc:  # noqa: BLE001
        meta["trace_error"] = str(exc)[:200]

    failed = [{"record_id": r.record_id, "step": r.step, "reason": r.reason[:300]} for r in result.failed_records]
    meta["failed_records"] = failed
    meta["failed_record_steps"] = dict(Counter(r["step"] for r in failed))

    # Per-row replacement counters: trace_dataframe["_replacement_application"]
    # holds ReplacementApplication.to_metrics() per record
    # (src/anonymizer/engine/replace/strategies.py). skipped_span_count > 0
    # means the original (unredacted) text survived in the output for that
    # row with no failed record and no review flag — this is the only signal
    # for that (see README "Skipped replacement spans").
    replacement_col = "_replacement_application"
    if replacement_col in trace_df.columns:

        def _metrics(cell: object) -> dict[str, int | None]:
            parsed: object = cell
            if isinstance(cell, str):
                try:
                    parsed = json.loads(cell)
                except (json.JSONDecodeError, TypeError):
                    parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            return {
                "targeted_span_count": parsed.get("targeted_span_count"),
                "applied_span_count": parsed.get("applied_span_count"),
                "skipped_span_count": parsed.get("skipped_span_count"),
            }

        per_row = [_metrics(cell) for cell in trace_df[replacement_col]]
        meta["replacement_application"] = per_row
        for key in ("targeted_span_count", "applied_span_count", "skipped_span_count"):
            meta[f"{key}_total"] = sum(row[key] or 0 for row in per_row)

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
