# CLI Display Improvements — Implementation Plan

## Problem

The CLI output is bare:

- `run` prints only `"Output written to: <path>"` — no stats, no entity summary, no failure info
- `preview` calls `result.dataframe.to_string(max_colwidth=80)` — raw pandas dump, hard to read, no summary
- `failed_records` are never shown to the user
- No way to save the `trace_dataframe` to a file

## Scope

No new dependencies. All rendering uses stdlib + pandas only.

---

## Files Changed

| File | Change |
|---|---|
| `src/anonymizer/interface/cli/_output.py` | Major expansion — new display functions |
| `src/anonymizer/interface/cli/main.py` | Add `--trace` flag; wire up new display calls |
| `tests/interface/cli/test_cli_output.py` | Update 1 existing test; add ~12 new tests |

`src/anonymizer/interface/anonymizer.py` — no changes needed. The private helpers `_unwrap_entities`, `_count_labels`, `_count_entities`, `_count_labels_for_row` are module-level functions and can be imported directly by `_output.py`. The import graph stays acyclic (`cli/_output.py` → `interface/anonymizer.py` → `engine/`).

---

## Step 1 — Expand `_output.py`

Keep `write_result()` signature identical (existing tests depend on it). Add:

### New public functions

```python
def write_trace(result: AnonymizerResult | PreviewResult, trace_path: str | Path) -> Path
def print_run_summary(result: AnonymizerResult, written: Path, *, trace_written: Path | None = None) -> None
def print_preview(result: PreviewResult) -> None
```

### New private helpers

```python
def _is_rewrite_mode(df: pd.DataFrame) -> bool
    # True if any column ends with "_rewritten"

def _entity_stats(df: pd.DataFrame) -> tuple[int, Counter[str]]
    # Reads COL_FINAL_ENTITIES, falls back to COL_DETECTED_ENTITIES
    # Returns (total_count, per_label_counter)
    # Returns (0, Counter()) when neither column is present (rewrite mode)

def _format_entity_table(total: int, labels: Counter[str]) -> list[str]
    # Returns lines; top 20 labels sorted by count descending
    # Appends "  ... and N more" if truncated

def _format_failed_records(failed: list[FailedRecord]) -> list[str]
    # Returns lines; empty list when len(failed) == 0

def _truncate(text: str, max_len: int = 120) -> str
    # Appends "..." if text exceeds max_len

def _format_preview_record(idx: int, row: pd.Series, text_col: str, rewrite_mode: bool) -> list[str]
    # Returns lines for one record block
```

Helpers return `list[str]` so they are unit-testable without capturing stdout.

### `print_run_summary` output — replace mode

```
Output written to: /path/to/data_anonymized.csv
Trace written to : /path/to/trace.csv          ← only when --trace given

--- Summary ---
Records processed : 1,234
Failed records    : 0
Total entities    : 4,521
  first_name      :   891
  email           :   312
  ssn             :    47
  ... and 3 more
```

### `print_run_summary` output — rewrite mode

```
Output written to: /path/to/data_rewritten.csv

--- Summary ---
Records processed    : 1,234
Failed records       : 0
Avg utility score    : 0.87
Avg leakage mass     : 0.03
Avg weighted leakage : 0.01
Needs human review   : 12 (1.0%)
```

### Failed records block (appended when failures > 0)

```
--- Failed Records (3) ---
  record_id=abc123  step=detect   reason=LLM timeout
  record_id=def456  step=replace  reason=JSON parse error
```

### `print_preview` output — replace mode

```
--- Preview (3 records, replace mode) ---

Record 1
  original : Alice works at Acme Corp, call 555-1212
  replaced : [REDACTED_FIRST_NAME] works at [REDACTED_COMPANY_NAME], ...
  entities : first_name, company_name, phone_number

Record 2
  ...

--- 0 failed records ---
```

### `print_preview` output — rewrite mode

```
--- Preview (3 records, rewrite mode) ---

Record 1
  original  : Alice works at Acme Corp
  rewritten : A healthcare professional works at a technology company
  utility   : 0.91
  leakage   : 0.00
  review?   : No

--- 0 failed records ---
```

### Edge cases

| Scenario | Behaviour |
|---|---|
| 0 entities detected | Print `"Total entities : 0"` with no label block |
| Rewrite mode — no entity column | Skip entity section entirely |
| Metric column absent in dataframe | Skip that metric line (guard with `if col in df.columns`) |
| 0 rows in dataframe (all failed) | Means are NaN — skip metric lines |
| Text > 120 chars | Truncate with `"..."` |
| > 20 entity labels | Show top 20 + `"... and N more"` |

---

## Step 2 — Update `main.py`

### Add `--trace` to `CliOpts` (under `# -- shared --`)

```python
trace: Annotated[
    str | None,
    cyclopts.Parameter(help="Write full pipeline trace to this path (.csv or .parquet)."),
] = None
```

`--trace` requires an explicit path argument (consistent with `--output`). No auto-derivation.

### Update `run` command

```python
# before
written = write_result(result, output)
print(f"Output written to: {written}")

# after
written = write_result(result, output)
trace_written = write_trace(result, opts.trace) if opts.trace else None
print_run_summary(result, written, trace_written=trace_written)
```

### Update `preview` command

```python
# before
print(result.dataframe.to_string(max_colwidth=80))

# after
print_preview(result)
```

### No `--trace` on `preview`

Preview is a quick inspection tool. Trace export belongs to `run`.

### Updated imports from `_output`

```python
from anonymizer.interface.cli._output import print_preview, print_run_summary, write_result, write_trace
```

---

## Step 3 — Tests

### Update existing: `test_cli_output.py`

- `test_preview_prints_dataframe`: currently asserts `"bio_replaced" in out` (column name). After the change `print_preview` shows values, not column headers. Update assertion to check for a value from the replaced text (e.g. `"REDACTED_0"`).
- `test_run_default_output_path` and `test_run_explicit_output`: assert the output path appears in stdout — still true because `print_run_summary` prints it on the first line. No change needed.

### New tests to add

```
# print_run_summary — replace mode
test_run_summary_replace_prints_entity_stats
test_run_summary_replace_zero_entities
test_run_summary_with_failures_prints_failed_block
test_run_summary_no_failures_omits_failed_block
test_run_summary_trace_line_appears

# print_run_summary — rewrite mode
test_run_summary_rewrite_prints_metrics
test_run_summary_rewrite_missing_columns_no_crash

# print_preview
test_preview_replace_shows_record_blocks
test_preview_rewrite_shows_metrics_per_record
test_preview_truncates_long_text
test_preview_failed_records_shown
test_preview_empty_dataframe_no_crash

# write_trace
test_write_trace_csv
test_write_trace_parquet

# CLI integration
test_run_with_trace_flag_writes_trace_file
```

All new tests construct result objects directly (using an extended `_make_result` helper) — no real pipeline or LLM calls.

---

## Implementation Order

1. Implement `_output.py` (new helpers + display functions)
2. Update `main.py` (add `--trace`, wire display calls, update imports)
3. Update + extend `test_cli_output.py`
4. Run `make check` (format, lint, copyright)
5. Run `make test`
