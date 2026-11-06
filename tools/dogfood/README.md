<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dogfood kit

A small, in-repo kit for comparing a branch build of Anonymizer against the released wheel on your
own data. It uses only the public `anonymizer` API (`Anonymizer`, `AnonymizerConfig`,
`AnonymizerInput`, `RiskTolerance`, the replace strategies `Redact`/`Substitute`/`Annotate`/`Hash`,
`configure_logging`) — no engine or `anonymizer.graph` imports — so it is safe to run against any
build that exposes that surface.

Files:

- `dogfood_run.py` — runs one route (`redact` / `substitute` / `annotate` / `hash` / `rewrite`) with
  one `call` (`run` / `preview`) against one environment, and writes `result.csv`, `result.parquet`
  (parquet input keeps its dtypes — see "Running" below), `trace.csv`, `log.txt` (DEBUG console
  log), and `meta.json` to `--out`.
- `compare.py` — diffs two `dogfood_run.py` output directories (status/error-class parity, columns,
  dtypes, entity overlap (falling back to `trace.csv` for Rewrite, whose public result has no
  `final_entities` column), a word-bounded surviving-entity-value oracle, passthrough-column parity,
  a null-input-row report, failed-record/skipped-span totals, rewrite metrics, and an optional
  `--exact` `assert_frame_equal` check — see "Running" below for what `--exact` actually proves).
- `scan_log.py` — scans a run directory's `log.txt` for entity-value leaks, with the detector
  label-list line filtered out. See "What to look at" below.
- `make_variants.py` — generates a set of input files (`None`, CSV-empty→NaN, `int`,
  `string[pyarrow]` with NA, all-null, empty string, whitespace, `category`, `large_string`, zero
  rows, odd passthrough dtypes) for dogfooding the "nullable or non-string columns" surface.
- `model_configs.yaml` / `providers.yaml` — a small default model pool (local GLiNER2 detector +
  two chat models behind the NVIDIA inference gateway) and provider definitions, naming env vars
  only. See "Model IDs" below if a model 403s.

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

## Model IDs

Gateway model IDs are **key- and gateway-dependent**: the same alias can need a different `model:`
string depending on which key and which gateway route it is called through. The IDs currently in
`model_configs.yaml` (`nvcf/openai/gpt-oss-120b`, `nvidia/nvidia/nemotron-3-nano-30b-a3b`) are
operator-verified against the `nvidia-gateway` provider in `providers.yaml`
(`https://inference-api.nvidia.com/v1`); the shorter forms (`openai/gpt-oss-120b`,
`nvidia/nemotron-3-nano-30b-a3b`) returned `403 key_model_access_denied` on that gateway for the
keys this kit was verified with. If a model 403s for your key, don't guess at prefixes — consult
your gateway's model list (however your provider/gateway documents or exposes it, e.g. an
OpenAI-compatible `/v1/models` endpoint, a gateway dashboard, or your provider's own docs) for the
exact ID your key is entitled to, and update `model:` in `model_configs.yaml` accordingly.

## Running

Exercise routes, both `run` and `preview`, against your own data (point `--data` at a CSV or
Parquet file with a text column; a Parquet `--data` file keeps its own dtypes end to end —
`dogfood_run.py` writes `input.parquet` into `--out` for a parquet source instead of collapsing it
to CSV, so `string[pyarrow]` NA, `category`, `large_string`, nullable `Int64`, and list columns all
reach the pipeline):

```bash
# branch environment
.venv/bin/python tools/dogfood/dogfood_run.py --mode redact     --call run     --data mydata.csv --text-column mytext --configs tools/dogfood/model_configs.yaml --providers tools/dogfood/providers.yaml --out /tmp/dogfood/branch-redact-run
.venv/bin/python tools/dogfood/dogfood_run.py --mode substitute --call preview --data mydata.csv --text-column mytext --configs tools/dogfood/model_configs.yaml --providers tools/dogfood/providers.yaml --out /tmp/dogfood/branch-substitute-preview
.venv/bin/python tools/dogfood/dogfood_run.py --mode annotate   --call run     --data mydata.csv --text-column mytext --configs tools/dogfood/model_configs.yaml --providers tools/dogfood/providers.yaml --out /tmp/dogfood/branch-annotate-run
.venv/bin/python tools/dogfood/dogfood_run.py --mode hash       --call run     --data mydata.parquet --text-column mytext --hash-algorithm sha1 --digest-length 16 --configs tools/dogfood/model_configs.yaml --providers tools/dogfood/providers.yaml --out /tmp/dogfood/branch-hash-run
.venv/bin/python tools/dogfood/dogfood_run.py --mode rewrite    --call run     --data mydata.csv --text-column mytext --risk-tolerance moderate --max-repair-iterations 1 --strict --combined --configs tools/dogfood/model_configs.yaml --providers tools/dogfood/providers.yaml --out /tmp/dogfood/branch-rewrite-run

# released-wheel environment (same commands, different --out and venv)
<release-venv>/bin/python tools/dogfood/dogfood_run.py --mode redact --call run --data mydata.csv --text-column mytext --configs tools/dogfood/model_configs.yaml --providers tools/dogfood/providers.yaml --out /tmp/dogfood/release-redact-run
```

`--format-template` overrides the redaction/annotation/hash template (see the `Redact` / `Annotate`
/ `Hash` docstrings in `anonymizer.config.replace_strategies` for the placeholders each one
accepts); `--no-normalize-label` disables `Redact.normalize_label`; `--hash-algorithm` and
`--digest-length` configure `Hash`. `--strict`, `--risk-tolerance`, `--max-repair-iterations`, and
`--combined` map to `Rewrite.strict_entity_protection`, `.risk_tolerance`, `.max_repair_iterations`,
and `.use_combined_graph`. Run `--help` for the full list.

Then compare:

```bash
.venv/bin/python tools/dogfood/compare.py /tmp/dogfood/branch-redact-run /tmp/dogfood/release-redact-run --mode redact --text-column mytext
```

`--exact` (`pd.testing.assert_frame_equal(df_a, df_b)`) is **not** a branch-vs-release regression
gate, even for Redact/Annotate/Hash: those routes apply their replacement locally without an LLM
call, but entity *detection* itself calls the configured chat model for LLM augmentation and
validation regardless of replace strategy (see `AGENTS.md`'s detection pipeline), so two separate
`dogfood_run.py` invocations — even same route, same input, same configs — are not guaranteed to
detect the identical entity set, and `--exact` will typically FAIL on real branch-vs-release runs
(exit 1) for that reason alone, not because of an actual branch regression. Scope it to a
fixed-detection-input comparison instead — e.g. verifying that `compare.py` would treat an unchanged
directory as identical, or that re-serializing the same `result.dataframe` round-trips cleanly:

```bash
.venv/bin/python tools/dogfood/compare.py /tmp/dogfood/branch-redact-run /tmp/dogfood/branch-redact-run --mode redact --text-column mytext --exact
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
  (`src/anonymizer/engine/ndd/failed_record.py`): a row that was dropped by an NDD workflow, not merely
  flagged. Any non-empty list is worth understanding before trusting a route's output.
  `meta.json["failed_record_steps"]` is a `{step: count}` summary of the same list, and
  `compare.py` prints both sides' step breakdown next to the `skipped_span_count` totals below.
- **`meta.json["error_type"]`** — module-qualified (`f"{type(exc).__module__}.{type(exc).__qualname__}"`),
  not just the bare class name, so e.g. an engine-internal rejection type can't be confused with a public
  `anonymizer.interface.errors` class of the same name. `meta.json["started_epoch"]` /
  `["ended_epoch"]` give wall-clock epoch times (in addition to `["seconds"]`) so you can compute
  e.g. Ctrl-C drain time. On error, `meta.json["traceback_frames"]` is a bare stack
  (`"file:function:lineno"` per frame, from `traceback.extract_tb`) — **the exception's message and
  source-line text are never written to meta.json or any other file this kit writes.** A backend
  exception can embed input record content verbatim, and not only as the offending cell's full text:
  a fragment of it, a pydantic `ValidationError`'s truncated `input_value=...`, text from a
  non-text column, or LLM-generated text that was never a literal substring of the input. A
  whole-cell-substring scrub can't catch all of those, so there is no safe redaction short of not
  persisting the message at all. `dogfood_run.py` prints the full `f"{error_type}: {exc}"` to
  **stderr only**, for whoever is running it interactively — read that, not meta.json, for the
  actual error text.
- **Skipped replacement spans** — `meta.json["replacement_application"]` holds one
  `{targeted_span_count, applied_span_count, skipped_span_count}` dict per row (and
  `meta.json["skipped_span_count_total"]` / `["targeted_span_count_total"]` /
  `["applied_span_count_total"]` sum them), pulled from
  `trace_dataframe["_replacement_application"]` — one dict per record
  (`ReplacementApplication.to_metrics()` in `src/anonymizer/engine/replace/strategies.py`), not a
  scalar, so to read it directly from `trace_dataframe` yourself use
  `trace_dataframe["_replacement_application"].map(lambda m: m["skipped_span_count"])` (equivalently
  `.str["skipped_span_count"]` — verified both work on a synthetic dict-valued `Series`). `> 0` means
  the *original* (unredacted) text survives in `{text}_replaced` for that record, with no failed
  record and no review flag — this trace column is the only signal. This is a known issue (DF-3): the
  replacement map is indexed by exact `(original, label)` text in
  `_apply_indexed_replacements`/`_resolve_synthetic`. There is a value-only fallback
  (`by_value`, used when `allow_value_fallback=True`, which the Substitute and local
  routes set) that resolves a span whose exact `(value, label)` pair is missing but
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
  raw=N filtered=N …"). `log.txt` should contain no entity values. Ignore matches inside the
  detector label-list line (`extra_body={'labels': [...]}`) — a short entity value can legitimately
  appear inside a label name there (e.g. `http_cookie`). Any other match is a `[privacy]` finding.
  Run `scan_log.py` instead of hand-grepping; it collects entity-value probes from the result,
  falling back to `trace.csv` for Rewrite (whose public result has no `final_entities` column — see
  above), and applies the same label-list filter. It fails closed — it reports a non-zero exit
  whenever it couldn't actually check something, not just when it finds a leak: exit 1 for a leak
  **or** missing inputs (no such directory, no `log.txt`, no result file); exit 3 when the check ran
  but collected zero entity-value probes — this is a routine outcome, not a rare edge case: it
  happens for any genuinely PII-free input (an all-null/no-PII variant, or a dataset your detector
  just doesn't find anything in), not only as a bug symptom, so treat exit 3 as "nothing was checked,
  go look at why" rather than either a pass or a failure; exit 0 only when everything it checked came
  back clean:
  ```bash
  .venv/bin/python tools/dogfood/scan_log.py /tmp/dogfood/branch-redact-run
  ```
  To see what entities Anonymizer detected or how they were replaced, use `result.dataframe`'s
  `final_entities` column and `trace.csv` instead.

## Input variants

`make_variants.py` builds a set of small input files from the head of any source CSV, each
reproducing one "nullable or non-string columns" surface (`None`, a CSV-empty field that reads back
as `NaN`, `int`, `string[pyarrow]` with `pd.NA`, an all-null column, empty string, whitespace-only
string, `category`, `large_string`, zero rows, and a row with odd passthrough-column dtypes):

```bash
.venv/bin/python tools/dogfood/make_variants.py mydata.csv /tmp/dogfood/variants --text-column mytext
.venv/bin/python tools/dogfood/dogfood_run.py --mode redact --call run --data /tmp/dogfood/variants/D6_string_pyarrow_na.parquet --text-column mytext --configs tools/dogfood/model_configs.yaml --providers tools/dogfood/providers.yaml --out /tmp/dogfood/branch-d6-run
```

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
