<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dogfood kit

A small, in-repo kit for comparing a branch build of Anonymizer against the released wheel on your
own data. It uses only the public `anonymizer` API (`Anonymizer`, `AnonymizerConfig`,
`AnonymizerInput`, the replace strategies, `configure_logging`) — no engine or `anonymizer.graph`
imports — so it is safe to run against any build that exposes that surface.

Files:

- `dogfood_run.py` — runs one route (`redact` / `substitute` / `rewrite`) with one `call`
  (`run` / `preview`) against one environment, and writes `result.csv`, `result.parquet`,
  `trace.csv`, `log.txt` (DEBUG console log), and `meta.json` to `--out`.
- `compare.py` — diffs two `dogfood_run.py` output directories (columns, dtypes, entity overlap,
  surviving entity values in output, rewrite metrics).
- `model_configs.yaml` / `providers.yaml` — a small default model pool (local GLiNER2 detector +
  two chat models behind the NVIDIA inference gateway) and provider definitions, naming env vars
  only.

## Setup

1. `uv sync --frozen --extra notebooks` — the `notebooks` extra is the documented setup for
   `tools/serve_gliner.py`; the server's own packages (`gliner2[local]`, `fastapi`, …) are
   installed by the wrapper into an isolated environment (`_SERVER_REQUIREMENTS` in
   `src/anonymizer/notebooks/_runtime.py`). See
   [`docs/concepts/self-hosting-gliner.md`](../../docs/concepts/self-hosting-gliner.md#running-the-development-server-directly).
2. Start the detector server this kit's `providers.yaml` expects on `http://127.0.0.1:8001/v1`:
   ```bash
   uv run python tools/serve_gliner.py --host 127.0.0.1 --port 8001
   ```
   Do **not** use `anonymizer.notebooks.create_anonymizer()` for this kit: that helper starts its
   own server on a random port with a random bearer token and returns an in-process `Anonymizer`
   client wired to that token (`src/anonymizer/notebooks/_runtime.py`, `_start_runtime` /
   `_LocalRuntime`). `dogfood_run.py` is a separate process that talks to whatever endpoint
   `providers.yaml` names, so there's no way to point it at a `create_anonymizer()`-managed server.
   `tools/serve_gliner.py` is the only server option compatible with this kit.
3. Export the gateway provider key named in `providers.yaml` (`NVIDIA_INFERENCE_API_KEY`), and
   `ANONYMIZER_LOCAL_GLINER2_TOKEN` if your detector server requires one.
4. Create a second environment with the released wheel, pinned to the **same** `pandas` and
   `data-designer` versions as the branch (see "Version pin" below) — otherwise dtype and behavior
   differences you see may be dependency drift, not a code difference.

## Running

Exercise all three routes, both `run` and `preview`, against your own data (point `--data` at a CSV
or Parquet file with a text column):

```bash
# branch environment
.venv/bin/python tools/dogfood/dogfood_run.py --mode redact     --call run     --data mydata.csv --text-column mytext --configs tools/dogfood/model_configs.yaml --providers tools/dogfood/providers.yaml --out /tmp/dogfood/branch-redact-run
.venv/bin/python tools/dogfood/dogfood_run.py --mode substitute --call preview --data mydata.csv --text-column mytext --configs tools/dogfood/model_configs.yaml --providers tools/dogfood/providers.yaml --out /tmp/dogfood/branch-substitute-preview
.venv/bin/python tools/dogfood/dogfood_run.py --mode rewrite    --call run     --data mydata.csv --text-column mytext --configs tools/dogfood/model_configs.yaml --providers tools/dogfood/providers.yaml --out /tmp/dogfood/branch-rewrite-run

# released-wheel environment (same commands, different --out and venv)
<release-venv>/bin/python tools/dogfood/dogfood_run.py --mode redact --call run --data mydata.csv --text-column mytext --configs tools/dogfood/model_configs.yaml --providers tools/dogfood/providers.yaml --out /tmp/dogfood/release-redact-run
```

Then compare:

```bash
.venv/bin/python tools/dogfood/compare.py /tmp/dogfood/branch-redact-run /tmp/dogfood/release-redact-run --mode redact --text-column mytext
```

`compare.py` reads `result.parquet` with `pyarrow.parquet` directly (`to_pylist()` per column)
rather than `pandas.read_parquet`. The Substitute route's `final_entities` column round-trips
through parquet as a pandas ArrowDtype-backed struct column
(`struct<entities: list<item: struct<value: string, label: string, ...>>>[pyarrow]`); reading that
back with `pandas.read_parquet` raises `ValueError: format number 1 of "struct<...>[pyarrow]" is
not recognized` — pandas reconstructs the column's extension dtype from the nested-struct type
string recorded in the parquet file's pandas metadata, and that reconstruction doesn't handle
nested structs (reproduced directly: a plain object-dtype column of Python dicts round-trips fine;
only pandas' own ArrowDtype struct columns, which is what NDD produces for `final_entities`, hit
this). Reading the Arrow table directly sidesteps that dtype reconstruction. `compare.py` falls
back to `result.csv` if no parquet file was written, and exits 1 with a clear message (not a
traceback) if a run directory has no `meta.json` or no result file at all — including when the run
being compared itself errored.

## What to look at

- **`result.failed_records`** — each entry is a `FailedRecord(record_id, step, reason)`
  (`src/anonymizer/engine/ndd/adapter.py`): a row that was dropped by an NDD workflow, not merely
  flagged. Any non-empty list is worth understanding before trusting a route's output.
- **Skipped replacement spans** — `trace_dataframe["_replacement_application"]` holds one dict per
  record (`ReplacementApplication.to_metrics()` in `src/anonymizer/engine/replace/strategies.py`),
  not a scalar, so pull the count out with
  `trace_dataframe["_replacement_application"].map(lambda m: m["skipped_span_count"])` (equivalently
  `.str["skipped_span_count"]` — verified both work on a synthetic dict-valued `Series`). `> 0` means
  the *original* (unredacted) text survives in `{text}_replaced` for that record, with no failed
  record and no review flag — this trace column is the only signal. This is a known issue (DF-3): the
  replacement map is indexed by exact `(original, label)` text in
  `_apply_indexed_replacements`/`_resolve_synthetic`. There is a value-only fallback
  (`by_value`, used when `allow_value_fallback=True`, which the Substitute route's
  `apply_replacement_map` sets) that resolves a span whose exact `(value, label)` pair is missing but
  whose bare value has exactly one synthetic across all labels — so not every map miss causes a skip.
  But there is no Unicode/whitespace normalization anywhere in this lookup, so an LLM-generated map
  that paraphrases, retypes, or drops an entity's text still silently leaves the source text in
  place.
- **A null `{text}_rewritten` with `needs_human_review=True`** means the rewrite was *withheld by
  design*, not that anything crashed. In rewrite mode, when the replacement map required for a record
  isn't ready, `_extract_rewritten_text` in
  `src/anonymizer/engine/rewrite/rewrite_generation.py` sets `COL_REWRITTEN_TEXT` to `None` and logs
  `"Required rewrite replacement was unavailable; marking rewritten text unavailable."` before any
  evaluation runs. Check `failed_records` (should be empty for these rows) and `needs_human_review` to
  tell this apart from a real failure.
- **Output dtypes vs the release** — `meta.json["dtypes"]` and `compare.py`'s `dtype differences`
  line. Dtype choices (`object` vs `string[pyarrow]`/`ArrowDtype`, etc.) can differ between a branch
  and the release even when values match; a consumer that depends on `df.dtypes` or Arrow-backed
  nullability will see it.
- **`leakage_mass` / `needs_human_review`** (rewrite mode) — `compare.py` prints their min/max and
  value counts for both sides; large deltas on the same input are worth a closer look at individual
  rows via `trace.csv`.
- **The DEBUG console log (`log.txt`)** — `dogfood_run.py` configures `anonymizer_level="DEBUG"` and
  attaches a `FileHandler` directly to the `"anonymizer"` and `"data_designer"` loggers (the ones
  `configure_logging` itself configures; both have `propagate=False`, so a handler on the root logger
  would never see these records), with the same formatter the console uses — `log.txt` holds the
  `anonymizer` and `data_designer` log records shown on stderr (not prints, tracebacks, or other loggers). At DEBUG, library logging reports counts, labels, and workflow
  status (e.g. "chunked validation: N candidate(s) in M chunks", "Replacement map: requested=N
  raw=N filtered=N …") — it does **not** log entity text values or other record content, so do not
  grep `log.txt` for entity values from your input; that check will always come up empty by design.
  To see what entities Anonymizer detected or how they were replaced, use `result.dataframe`'s
  `final_entities` column and `trace.csv` instead.

## Known noise, not bugs

- **`OSError: [Errno 98] Address already in use`** from DataDesigner's OpenTelemetry metrics server
  (`data_designer/integrations/opentelemetry.py`, `_activate_run`), followed by "OpenTelemetry
  metrics are unavailable; continuing without them." Harmless; expect it whenever you run more than
  one Anonymizer process at a time (including a branch run and a release run in parallel).
- **`AnonymizerInput.id_column`** appears only in `trace_dataframe`, not in `result.dataframe` — by
  design, not a bug to report.

## Version pin

This branch pins `data-designer` to an exact version
(`src/anonymizer/engine/ndd/_graph_requests.py` raises `AnonymizerWorkflowError` if
`importlib.metadata.version("data-designer") != "0.9.4"`). When you set up the two comparison
environments, make sure both install `data-designer==0.9.4` — a fresh `uv sync` / `pip install` that
resolves a newer `data-designer` will make this branch fail every route after detection, which is a
dependency-resolution artifact, not necessarily a code regression.

## Claims not independently verified in this pass

- Exact wall-clock overhead of the branch vs. the release, and whether it reproduces outside the
  original dogfood environment — not re-measured here; this kit only gives you the `seconds` field in
  `meta.json` to compare on your own data and hardware.
