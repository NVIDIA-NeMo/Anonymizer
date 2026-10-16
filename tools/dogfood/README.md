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
  one gateway chat model) and provider definitions, naming env vars only.

## Setup

1. `uv sync --frozen`
2. Start a GLiNER2-compatible detector server. Either:
   - run `tools/serve_gliner.py` directly, or
   - use the notebook runtime (`anonymizer.notebooks.create_anonymizer()`).
   See [`docs/concepts/self-hosting-gliner.md`](../../docs/concepts/self-hosting-gliner.md) for the
   server contract and both options. `providers.yaml` in this kit expects it on
   `http://127.0.0.1:8001/v1`.
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
rather than `pandas.read_parquet`: the Substitute route's `final_entities` column comes back as an
Arrow `struct<entities: list<struct<...>>>`, which plain pandas parquet loading cannot deserialize
into Python objects. It falls back to `result.csv` if no parquet file was written.

## What to look at

- **`result.failed_records`** — each entry is a `FailedRecord(record_id, step, reason)`
  (`src/anonymizer/engine/ndd/adapter.py`): a row that was dropped by an NDD workflow, not merely
  flagged. Any non-empty list is worth understanding before trusting a route's output.
- **`trace_dataframe["_replacement_application"].skipped_span_count`** — a per-record count of
  replacement spans that were targeted but never applied. `> 0` means the *original* (unredacted)
  text survives in `{text}_replaced` for that record, with no failed record and no review flag — the
  only signal is this trace column. This is a known issue (DF-3): the replacement map is indexed by
  exact `(original, label)` text in `src/anonymizer/engine/replace/strategies.py`
  (`_apply_indexed_replacements`), with no Unicode/whitespace normalization, so an LLM-generated map
  that paraphrases or drops an entity silently leaves the source text in place.
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
- **The DEBUG console log (`log.txt`)** — `dogfood_run.py` configures `anonymizer_level="DEBUG"` so
  entity values that leak into logging are visible; grep `log.txt` for entity text from your input to
  spot-check that detected PII values aren't being logged in the clear.

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
