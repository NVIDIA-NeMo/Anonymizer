# Changelog

## v1.0.0 (unreleased)

This section lists what changes for users between v0.5.0 and v1.0.0. It will be updated until the release, and
becomes the v1.0.0 release notes after #319 lands on `main`.

- **Describes:** commit `3dc8c7eb`, the head of the #319 branch.
- **Compared against:** v0.5.0 (`81bcea43`).
- **Last reconciled with main at `628be95a`.** The branch has since merged main at `6c5eb897` (#438), which this
  file has not been checked against. The next update reads `git log 628be95a..main` and the new commits on
  the #319 branch.

v1.0.0 may break compatibility with v0.5.0. Each entry says what changed, where it changed, and what to do.

## Breaking and behaviour changes since v0.5.0

### Install and dependencies

**DataDesigner 1.0 is required.** The requirement moves from `data-designer>=0.9.4,<0.10` to
`data-designer>=1.0.0,<1.1`.
- What to do: upgrade DataDesigner with Anonymizer. Code that passes DataDesigner objects (for example a
  `DataDesigner` or `RunConfig`) must follow DataDesigner's 1.0 changes.
- Changed in: #375, #404, #410.

**New and updated dependencies.** `regex` is a new required dependency. The `notebooks` extra moves to
`gliner2==2.1.0` and `pillow>=13,<14`.
- What to do: nothing, unless you pin these packages yourself.
- Changed in: #265, #212, #375.

**Notebook runtime internals are removed.** `anonymizer.notebooks.local_inference` and the private helpers in
`anonymizer.notebooks._runtime` and `_model_config` (such as `build_notebook_model_configuration`) are gone.
`anonymizer.notebooks.create_anonymizer`, `stop_local_runtime` and `required_api_key_environment_variables` remain.
The native GLiNER2 server runs float32 with standard attention by default.
- What to do: replace imports of the private modules with `create_anonymizer(gliner=NativeGliner(...))` or
  `native_runtime(...)` from `anonymizer`.
- Changed in: #212.

### Configuration

**Built-in regex detection is on by default.** `Detect.builtin_regexes` defaults to `True`. Rules run for the labels
`credit_debit_card`, `email`, `ipv4`, `ipv6`, `mac_address` and `url` when they are in the effective label set. By
default, an LLM validates each match. Detection output can differ from v0.5.0 for these labels.
- What to do: set `Detect(builtin_regexes=False)` (CLI: `--no-builtin-regexes`) to keep v0.5.0 detection.
- Changed in: #265.

**An empty Redact template is rejected.** `Redact(format_template="")`, or a template that renders empty, raises a
validation error when the strategy is built. v0.5.0 accepted it and left the original detected value in the output.
- What to do: give a template that renders to at least one character.
- Changed in: #319 (`0b51cfef`).

**Labels named `nagroup_g<N>` are reserved.** Such a label in `entity_labels`, `excluded_entity_labels` or
`entity_label_examples` raises an error when the run starts.
- What to do: rename the label.
- Changed in: #305.

**`Hash` requires a secret key.** The digest is HMAC-SHA256 of the entity text with `Hash.key`, after casefolding and
collapsing runs of whitespace (the fold Substitute uses to match variants of a value). v0.5.0 hashed the exact text
with an unkeyed digest, which could be reversed for low-entropy values, such as dates of birth or ZIP codes, by hashing
candidates. `Hash()` without a key raises a validation error that says how to pass one. `algorithm` accepts only
`sha256`; v0.5.0 also accepted `sha1` and `md5`. The CLI reads the key from `ANONYMIZER_HASH_KEY`. Results keep the
strategy without its key, and a config built from one is refused. JSON dumps of a config mask the key, so a saved JSON
config does not load until you pass the key again; a pickled config keeps it. Validation errors from `Hash` and
`AnonymizerConfig`, including their `errors()` and `json()`, do not carry the key.
- What to do: pass `Hash(key=...)`, for example from `os.environ["ANONYMIZER_HASH_KEY"]`, and set that variable for
  `--replace hash`. Drop `algorithm="sha1"` or `"md5"`. Every token changes, and case and spacing variants of a value
  now share one token. Keep the key to get the same tokens in later runs, and pass it again when you load a saved
  config, including a v0.5.0 dump that holds `Hash`.
- Changed in: #381.

**An input column named like an internal column is rejected.** `run()`, `preview()` and `export_detection_config()`
raise `InvalidInputError`, naming each such column, when the input has a column that the pipeline uses as its own, such
as `_rewritten_text` or `_anonymizer_record_id`. v0.5.0 rejected only the internal text column, and read any other
such column as pipeline data: in Rewrite mode an input `_rewritten_text` was released as the rewritten text. A column
named `_anonymizer_record_id` is rejected even when it is the `id_column`. Your own text column is unaffected. Result
columns are renamed with an `__input` suffix when an input column carries their name: v0.5.0 did this for the output
columns `<text>_replaced`, `<text>_with_spans` and `<text>_rewritten` and for `final_entities`, `utility_score`,
`leakage_mass`, `weighted_leakage_rate`, `any_high_leaked` and `needs_human_review`, and it now also covers the
evaluation columns, such as `judge_evaluation`, `entity_coverage`, `missed_entities` and `detection_valid`, so an input
value is no longer read as the pipeline's own result.
- What to do: rename or drop the column before running.
- Changed in: #440.

### Detection

**Detection candidates follow a new, deterministic lifecycle.** Detected values are no longer propagated to other
occurrences, and full names are no longer split into parts. Augmentation adds every exact match of a suggested value
in the same context. An entity kept only as fallback protection has `source="fallback"`. An entity with no native
score reports `0.0`; augmented entities used to report `1.0`.
- What to do: expect different `final_entities` on the same input. Code that relied on name-part entities, or on
  `score == 1.0` for augmented entities, must change.
- Changed in: #305.

**Validation chunks keep overlapping candidates together.** `validation_max_entities_per_call` is now a target.
Overlapping alternatives stay in one chunk, and a group larger than the target is sent alone with a warning.
- What to do: nothing. Expect larger validator requests on texts with many overlapping candidates.
- Changed in: #305.

**`data_summary` sensitivity rules take priority.** When the summary says a kind of value is not sensitive,
validation drops it and augmentation does not add it.
- What to do: review summaries that describe sensitivity. They now change detection output.
- Changed in: #389.

**The augmenter asks for schema-constrained output.** On OpenAI-compatible providers, augmenter requests send the
response JSON schema as `response_format`. If the provider rejects the field, the request is retried once without it.
A malformed augmenter response is retried under the run's restart and correction limits instead of dropping the record.
- What to do: nothing for most providers. To turn the schema off, set `response_format` in the model's or provider's
  `extra_body`.
- Changed in: #319 (`21a10a08`, `1302866d`), #422.

**Detector spans are checked against the text.** A custom detector's spans must be half-open, inside the text,
non-empty, labelled, and equal to the text they cover. v0.5.0 Replace skipped a span that broke these rules, or that
overlapped an earlier span, and left any of its text that no earlier span covered in the output. Now:
- Spans with identical bounds collapse into the first labelled one, keeping its label, and a span inside a labelled
  span gives way to the outer span. Later steps and `final_entities` see the collapsed spans.
- Any other break fails the record at step `detect`. The reason is a fixed code, with no offsets and no text:
  `overlapping_output_ownership` (a partial overlap), `mention_out_of_range` (past the end of the text, negative or
  zero-width), `mention_slice_mismatch` (the text differs from `value`) or `mention_unlabelled` (no label, and no
  labelled span covers it).
- In Rewrite, a built-in detector's `final_entities` payload that does not parse fails the record at `detect` with
  `invalid_output`; Replace fails it at step `replace` (see Replace). A custom detector returns typed spans in a
  `DetectorResult` instead, and a malformed one raises `InvalidDetectorResultError` (see "Detection contract").
- A custom detector no longer returns `_entities_by_value`; the engine builds it. For a row with spans, it is built
  from the kept spans plus every value the detector listed, and the value of every collapsed span, that still occurs
  in the text outside the kept spans. A value occurs as a whole word, ignoring case and accents (a value of up to three
  uppercase characters matches with its case, and a value of two or more characters in Han, Thai or a similar script
  matches anywhere); a one-character value in an unspaced script, such as 王, counts wherever it appears. A row
  without spans keeps the detector's view.
- The built-in detector does not produce such spans.
- What to do: a custom detector must report non-overlapping code-point offsets into the text it was given, and a
  label for every span. These checks run on every custom detector's result: it names its input rows by record id and
  carries no text of its own (see "Detection contract").
- Changed in: #305, #423, #319, #432.

**The regex time budget is measured in CPU time.** Each rule's budget counts CPU seconds of its scanning thread, with
a wall-clock ceiling of 20 times the budget. A record whose scan exceeds the CPU budget and one whose scan was starved
of time fail with different reasons. `RegexLimits` is new since v0.5.0, so no released field changes meaning.
- What to do: update code that matches regex failure reasons.
- Changed in: #319 (`35d68f4e`).

### Replace

**Variants of a value share one substitute within a record.** Case, whitespace and Unicode-space variants of a value
(for example `Paris`, `PARIS`, and a value spelled with a narrow no-break space) get one substitute, re-cased to each
mention. A mention keeps its own substitute only where the shared one would clash with another value of the record.
- What to do: nothing. Output differs where v0.5.0 gave variants different substitutes.
- Changed in: #319 (`4fcecaeb`, `18da9734`), with #396.

**A substitute that matches a protected value becomes a placeholder.** When the model proposes a substitute equal to
a detected value of the same record, ignoring case and whitespace, the value gets `[SUBSTITUTE_<LABEL>_<n>]`. Two
different values of one record can still share an ordinary substitute, as in v0.5.0.
- What to do: expect occasional placeholders in Substitute output.
- Changed in: #382, #403.

**Incomplete substitution maps fail the record.** Substitute checks the model's map against the requested values. It
retries missing pairs once, then reports a record that is still incomplete in `failed_records`. Empty and
whitespace-only substitutes are rejected.
- What to do: read `failed_records`. Records that v0.5.0 returned with a value left unreplaced are now failed.
- Changed in: #396.

**Malformed entity payloads fail the record.** A malformed `final_entities` value, for example from a custom
detector, fails the record at step `replace`. v0.5.0 returned the record with its original text.
- What to do: read `failed_records`.
- Changed in: #367.

**Replace rows have a checked outcome.** Each Redact, Hash and Substitute row has the required check
`replacements_applied`: the replacements rebuild the text. It also has the advisory `no_detected_value_added`: the
replaced text holds a detected value more often than the source holds it outside the detected spans. Each detected value
is counted as whole-word matches, ignoring case and accents (a value of up to three uppercase characters matches with
its case, and a value of two or more characters in Han, Thai or a similar script matches anywhere), in the source (N),
inside the detected spans (D) and in the replaced text (O), and the advisory fails when O is greater than N minus D. The
count covers the whole replaced text, replacements included, so a substitute that brings back another detected value
fails it, and a value that the source repeats outside the detected spans may stay. A failing check states
`e<position>:N<n>,D<d>,O<o>` for each value over the bound in its `cause`: the entity's place among the detected spans
and the counts, never record text. Annotate keeps the source text and is never marked protected. Rewrite rows carry no
such advisory. A `not_met` row stays in the output frame.
- What to do: read `result.outcome(row)` before releasing a row, and `result.leftovers(row)` to see the counted values.
- Changed in: #319, #424, #426.

**A custom detector must report a span for every occurrence of a value it reports.** With a custom detector, Redact,
Hash, Annotate and Substitute fail a row at step `detect` with `unspanned_value` when a value the detector reports, in
a span or in `_entities_by_value`, occurs in the text outside every kept span. That includes a span's value repeated
without a span, and a value that crosses two spans. Replace changes spans only, so v0.5.0 returned such a value
unreplaced. Built-in detection protects such a repeat instead: Replace adds a span there (see "Detection contract"),
and a repeat that crosses a kept span's edge fails the row with `unspanned_value`.
- What to do: report a span for each occurrence, or read `failed_records`.
- Changed in: #432.

### Rewrite

**Every repair iteration takes effect.** In v0.5.0, repairs after the first were discarded and the same text was
scored again. Now each repair sees the previous repair's text.
- What to do: nothing. With `max_repair_iterations` above 1 (the default is 3), output and metrics can differ.
- Changed in: #354.

**The review flag follows the final checks.** `needs_human_review` is set when a required check on the final text
does not pass, when the row keeps a replaced value or a placeholder as described below, or when the utility score is
missing or below the preset's floor. The checks are `high_leak`, `replacements_applied`,
`leakage_within_threshold` and `privacy_coverage`. Compared with v0.5.0:
- `replacements_applied` counts each value the sensitivity policy replaces, as the Replace advisory does (N, D and O,
  above). A row whose final text holds such a value more often than the source holds it outside the detected spans
  fails, and its `cause` gives the counts. A row whose policy is missing is flagged too;
- a row that keeps a replaced value within that bound, because the source holds it outside the detected spans (a
  homograph, or a repeat that detection missed), stays met but is flagged;
- a row whose final text holds a placeholder from its replacement map (`[SUBSTITUTE_<LABEL>_<n>]`) stays met but is
  flagged. The match is loose: any case, brackets optional, `_`, `-` or one space between words, and the separator
  before the number optional. The number is required;
- a row with a missing or NaN metric is flagged; NaN used to pass and `None` used to raise;
- "replacement unavailable" alone no longer flags a row.
- What to do: read `result.outcome(row)` for the reason a row is flagged, and `result.leftovers(row)` for the counts.
- Changed in: #368, #319 (`3af11141`, `cb015ae3`, `7523154b`), #426, #431.

**Rewrite keeps a record whose model response cannot be parsed.** Domain, meaning-unit, quality-question, rewrite,
repair and evaluation responses that still cannot be parsed after their retries keep the record. v0.5.0 dropped it and
listed it in `failed_records`. Each kept record logs a warning of its own, and the run ends with one warning counting
such records per step; an evaluation step counts a record whose response was missing in an earlier repair round, not
only in the last one. What the record gets:
- Domain: classified as `OTHER`.
- Meaning units or quality questions: the quality evaluation is skipped, `utility_score` is null and the row is
  flagged. The privacy evaluation runs as usual.
- Rewrite or repair: no rewritten text (null), null metrics, a flagged row and a `not_met` outcome.
- Privacy evaluation: the rewritten text is withheld (null), the row is flagged, and its outcome is `not_met` with
  `could_not_assess` reasons. Each evaluation call now retries in a second conversation, at most 8 requests where
  v0.5.0 allowed 4. A privacy answer's `reason` that is too long is shortened and an empty one gets a default; v0.5.0
  rejected the whole answer.
- Sensitivity disposition: a response with a different number of entries than supplied entities is asked again, like
  one that cannot be parsed. If none has the right count, the record has no disposition and is not rewritten: the
  text is null, the row is flagged and its outcome is `not_met`. The per-row warning names the column and the request
  count, and for a wrong entry count both counts, as in `entry_count_mismatch: 4 entries, 2 expected`.
- What to do: read the warning. A step listed for most records points at that step's model.
- Changed in: #427, #431.

**Rewritten text that fails a privacy check is withheld.** A record whose final candidate fails `high_leak` or
`leakage_within_threshold` has its rewritten text withheld (null), whether or not it was repaired: after the last
repair round, when a repair response cannot be parsed, and when no repair ran, which includes every failing record under
`max_repair_iterations=0` and a record whose high leak the `high` preset does not send to repair. The row stays flagged
and `not_met`, and its outcome has `available=False`. v0.5.0 returned the text and flagged the row. Withheld and
missing Rewrite text is `None` in the default staged mode and in the combined mode
(`Rewrite(use_combined_graph=True)`).
- What to do: treat null rewritten text as "not released". For an audit pass, run with `max_repair_iterations=0`:
  `trace_dataframe` keeps the withheld first rewrite in its internal `_full_rewrite` column.
- Changed in: #431.

**A Rewrite step whose model keeps giving no usable response stops early.** When the share of a step's last
`shutdown_error_window` rows without a usable response reaches `shutdown_error_rate` (by default 5 of 10), that step
sends no further requests and logs one warning naming the step, the rate, the window and the threshold. Rows that had
already passed the stopped step are not dropped with the rows not run: they finish and are returned. Only rows the
stopped step refused are listed in `failed_records`, each once, with the reason `stopped_early (<step>)` naming that
step by its column name, for example `stopped_early (_domain)`. The stop does not raise, even when no row finished. Each
repair round counts on its own, in the staged and the combined mode. A row that failed for another reason keeps that
reason, and an error the stop did not cause still fails the run. A dataset smaller than the window never stops. In
v0.5.0 DataDesigner's own shutdown ended the run and raised `AnonymizerWorkflowError` when no row had finished.
- What to do: check the named step's model. To run every row anyway, set `disable_early_shutdown=True` in
  `data_designer_run_config`.
- Changed in: #431, #446.

**Rows that fail detection in Rewrite are failed records.** They are absent from the output and listed in
`failed_records` with step `detect`. They used to be rewritten anyway.
- What to do: read `failed_records`.
- Changed in: #392.

**Sensitivity disposition was reworked.** Default labels are classified locally as direct or quasi identifiers, and
custom labels by a new `entity_classifier` model role. The model must return exactly one decision per entity. The
disposition prompt no longer receives the domain supplement. It still receives `data_summary`, now as a separate block
whose explicit sensitivity guidance the model applies where it agrees with the privacy goal. The
`_sensitivity_disposition` trace column has a new shape.
- What to do: nothing required. `selected_models.rewrite.entity_classifier` defaults to the `disposition_analyzer`
  alias. Expect one more model call for a record with custom labels.
- Changed in: #333.

**A record whose only finding is latent is rewritten.** A latent entity is sensitive information the text implies
without naming it. A Rewrite record with latent entities and no explicit entity is now rewritten and evaluated under
the privacy goal. It used to be returned unchanged and reported as protected. See "Latent-only Rewrite records" under
"Pending changes".

### Evaluation

**Missed entities flag a Rewrite row.** In Rewrite mode, `evaluate()` sets `needs_human_review` on any row whose
coverage judge reports missed entities. The `run()` result is not changed.
- What to do: nothing.
- Changed in: #356.

**Each Replace judge runs in its own workflow.** `evaluate()` on a Replace result ran every judge as a column of one
DataDesigner workflow, and DataDesigner dropped a record when any of its columns failed, so one judge's unusable
response emptied every judge's score for that record. Now each active judge runs as its own workflow, one after
another. A judge that fails for a record leaves only its own score empty, and each record failure DataDesigner returns
for a judge adds its own entry to `failed_records`. A judge whose whole workflow raises is logged at WARNING and leaves
its scores empty for every row without adding to `failed_records`. `evaluate()` takes the sum of the judge times
instead of the longest.
- What to do: nothing. Expect one `failed_records` entry per judge and row that DataDesigner reports as failed; its
  `reason` names the judge's column when the cause was recorded. Expect a longer `evaluate()` whenever more than one
  judge is active (always for Substitute).
- Changed in: #434.

**Judges read only the columns they need, and a failed judge is logged.** Each evaluation judge sends DataDesigner only
the columns its prompt reads, then puts its verdict back on the full trace by source position, so the returned trace
keeps every column with its original values. A result whose index repeats is scored row by row. A stage that reads
nested cells from a frame loaded from parquet now seeds them as plain JSON (numpy arrays become lists, models their
fields); the seed used to hold their printed form. A judge failure that leaves its scores empty (the Replace
judges, the Rewrite final judge and the Rewrite detection validity judge) is logged at WARNING with the exception class
and the number of rows, never the message. v0.5.0 logged these at DEBUG.
- What to do: nothing. Alert on the new WARNING lines if you watch for unavailable scores.
- Changed in: #411, #420.

### Results

**Nested result columns are plain Python objects.** `final_entities` and `judge_evaluation` are returned as Python
objects, not Arrow structs. They round-trip through `to_parquet` and `pd.read_parquet`.
- What to do: read these columns as Python dicts.
- Changed in: #352.

**`skipped_span_label_counts` is JSON text.** In `trace_dataframe`, in `_replacement_application`,
`skipped_span_label_counts` is JSON text (for example `"{}"`) instead of a dict, so trace frames can be written to
Parquet. An empty dict became a struct with no fields, which Parquet cannot store.
- What to do: parse it with `json.loads`.
- Changed in: #429.

**Trace frame columns changed.** Internal columns of `trace_dataframe`, whose names start with `_`, were added,
removed, renamed and reordered. For example, `_validated_entities` and `_seed_tagged_text` are gone, and
`_augmentation_response_format` is new. In `evaluate()` traces, the judge columns come in a different order. The public
columns of each mode, and their order, are unchanged.
- What to do: read trace columns by name, and do not depend on underscore columns.
- Changed in: #265, #305, #333, #319, #422.

**`FailedRecord.reason` names the cause.** The reason keeps the prefix `Record missing from workflow output` and adds
the column and a fixed summary of the cause. Detection and map failures have their own steps and reasons.
- What to do: match on the prefix or the step, not on the whole text. Import `FailedRecord` from
  `anonymizer.interface.results`; its defining module moved.
- Changed in: #349, #392, #396.

**Custom-runner `run()` and `preview()` results have no outcomes.** A `replace_runner` or `rewrite_runner` whose class
overrides `run()` keeps the older detect-then-run path. Its `run()` and `preview()` results carry no outcomes, and
`result.outcome(row)` raises `IndexError`. An `evaluate()` result carries one outcome per row, as the next entry
says.
- What to do: use the built-in runners to get outcomes from `run()` and `preview()`.
- Changed in: #319.

**`evaluate()` results report outcomes.** `result.outcome(row)` on an `evaluate()` result no longer raises
`IndexError`. It returns the row's outcome from the evaluated `run()` or `preview()` result, with one advisory per judge
appended. The advisory's `check` and `verifier` name the judge (`coverage`, `rewrite-judge`, `detection-validity`,
`type-fidelity`, `relational-consistency` or `attribute-fidelity`), and its `status` is `passed`, `failed`,
`could_not_assess`, `scored` (the rewrite judge, which has no pass threshold, returned scores) or `not_applicable` (the
rewrite judge skips a row with no detected entities). A judge never changes qualification or release, so
`qualification` and `protected` still say what the run decided. The evaluate frame does not change.
- Matching: judge rows and the run's outcomes are matched to records only by a non-null `_anonymizer_record_id` that
  occurs once in the evaluated result, once among the judge's rows and once among the run's own outcomes. Row order
  and text are never used. A row without an id, or whose id does not occur exactly once in the evaluated result and
  among the judge's rows, cannot be placed: it is `not_assessed` and unprotected, and each judge reads
  `could_not_assess` with cause `unplaced`. A placed row whose id does not occur exactly once among the run's outcomes
  is `not_assessed` and unprotected, but keeps its judges' actual advisories.
  Outcomes are looked up by id, so reordering or filtering the result's rows before `evaluate()` keeps each row's own
  outcome. A result with no outcomes, such as one from a custom runner or one saved before outcomes were kept, gives
  `not_assessed` and unprotected rows with their judges' advisories.
- What to do: a custom `replace_runner` or `rewrite_runner` must keep each row's `_anonymizer_record_id` in the
  trace it returns and in its `evaluate()` output. Every built-in runner does. A row a custom runner gives another
  record's id is reported as that record.
- Changed in: #387.

**A detector result that cannot be placed raises.** A custom detector returns a `DetectorResult` whose rows name
their input rows by record id and carry no text, so the run never works over a detector's frame as returned. A result
that names no input row, or names one twice, raises `InvalidDetectorResultError` before any model call (see "Detection
contract"). On the #319 branch before the contract, such a result was rerouted without per-row outcomes or span checks.
- What to do: return a `DetectorResult` (see "Detection contract").
- Changed in: #319 (`845f4106`), #432.

**`display_record` reads a null metric as "not available".** A null Utility, Leakage or Weighted Leakage Rate (`None`,
NA or NaN) shows "not available". v0.5.0 omitted the metric for `None`, and showed `nan` for NaN and a blank for NA.
- What to do: nothing.
- Changed in: #431.

**Custom workflow results must say which input row they answer.** A result without source positions corresponds to
its input only through unique record IDs. Equal row counts or equal text are no longer enough.
- What to do: a custom detector or runner that drops or reorders rows must keep record IDs or report positions.
- Changed in: #319 (`4477453e`, `14eb166e`, `8c2e74a6`).

### Runtime, logging and telemetry

**Runs are isolated by context and thread.** `run()`, `preview()` and `evaluate()` run in a copy of the caller's
`contextvars` context. Built-in steps run their work on the calling thread. Runs that share one `DataDesigner` no
longer see each other's overrides and can run at the same time.
- What to do: nothing. A context value set inside a custom runner no longer leaks back to the caller.
- Changed in: #321, #319 (`6a9e48a1`, `0b2173b4`, `d335b8da`).

**Log lines carry fewer values.** Library log lines omit entity values, label lists, model aliases and record IDs.
The `effective detection scope` line no longer lists label names; read `result.detection_scope` instead. Per-record
failure debug lines are gone; read `result.failed_records`. Two failure warnings, the workflow-failure warning in
the model adapter and the entity coverage judge's "scores may be unavailable" warning, name the exception class but
not its message, because the message can carry record content; backend class names are shortened to a neutral form.
- What to do: update log parsers.
- Changed in: #319 (`372bbd0b`).

**The failure warning lists counts by step and reason.** The warning that counts failed records ends with
`step/reason=count` for each kind of failure, for example `detect/mention_slice_mismatch=1`. v0.5.0 gave only the
total. Only known steps and reason codes are named: any other reason counts under its step alone and any other step
under `other`. Record IDs and values stay out of the line. In the progress lines, a record rejected at `detect` counts
under detection, not under replacement or rewrite.
- What to do: update log parsers.
- Changed in: #432.

**DataDesigner task tracing is always on.** It is used to report failure causes, and costs about 250 bytes per task
attempt.
- What to do: nothing. Expect higher peak memory on large runs.
- Changed in: #349.

**Measurement records gain graph stages.** With a measurement session active, each run also records the stages
`graph_prepare`, `graph_protect` and `graph_reconcile`, and the Replace stages `ReplacementWorkflow.run` and
`LlmReplaceWorkflow.fold`. These records carry counts and timings only.
- What to do: measurement tooling that expects a fixed set of stages must accept the new ones.
- Changed in: #319 (`86dc269e`, `0b51cfef`), #397.

**The CLI shows a traceback for a graph rejection.** A `GraphRejection` raised during a CLI command is not turned into
an `Error: ...` line. It ends the command with its traceback, as a wiring fault.
- What to do: scripts that parse CLI errors should also expect a traceback for this error type.
- Changed in: #319 (`90e957da`).

**Telemetry key for the combined Rewrite path.** Failures from `Rewrite(use_combined_graph=True)` count under
`rewrite_pipeline`, not `unknown`.
- Changed in: #373.

### Internal changes

These change no public name and no behaviour for supported input.

- **Unused private code is removed.** The private structured-column subclass in the detection workflow, the adapter
  branch that converted it, and `_resolve_model_providers` in `anonymizer.interface.anonymizer`, which only forwarded
  to `parse_model_providers`. Nothing created the subclass after #305 (#433).
- **Nested cells have one reader.** Six parse functions in `anonymizer.engine.rewrite.parsers` and four hand-written
  replacement-map parsers read cells through `read_cell` and `EntityReplacementMapSchema.from_raw`. The sensitivity
  disposition parser's error message uses the wording of the other parsers (#412).
- **Stages run through one helper.** `run_columns` in `anonymizer.engine.ndd.workflow_utils` derives the seed columns,
  seeds the frame and runs the columns for the Rewrite, evaluation and Replace stages (#411).
- **Character ranges have one type.** Detection, replacement and the leftover checks carry character ranges as `Span`
  and `Spans` (`anonymizer.engine.spans`) instead of raw offset pairs. No behaviour changes for any reachable input
  (#430).
- **A missing import is restored.** #429 began writing skipped span label counts with `json.dumps` in
  `anonymizer.engine.replace.strategies` while #412 removed that module's `json` import. Together every local
  replacement raised `NameError` on this branch. #435 restores the import. v0.5.0 never had the fault.

### New public names

These are additions and break nothing.

- `result.outcome(row)`, `result.detection_scope` and `result.effective_entity_labels` on `run()` and `preview()`
  results (#319, #385).
- `Substitute(consistency="record" | "run")`. `"record"` is the default and matches v0.5.0. `"run"` keeps one
  substitute per value across the run, and two values never share one (#319, #394, #403).
- Regex detection: `Detect.builtin_regexes`, `Detect.regex_rules`, `Detect.regex_limits`, and `BuiltinRegex`,
  `RegexRule`, `RegexLimits`, `RegexCandidate`, `RegexValidationResult` and `BUILTIN_REGEX_LABELS`, with matching CLI
  flags (#265).
- Native GLiNER2: `create_anonymizer`, `native_runtime`, `NativeGliner`, `GlinerEndpoint` and `NativeRuntimeSession`
  (#212).
- `anonymizer.graph`, the graph SDK (#319). It is provisional, and its names may change. `prepare` rejects two
  operations with the same flattened name (`duplicate_operation_name`).
- `Finding` in `anonymizer.config.rewrite`, and `EvaluationCriteria.repair_on` (#368).
- `result.leftovers(row)` on `run()` and `preview()` results. It returns the counted values of a row as `Leftover`
  records, in the order of the check's `cause`, for auditing; it holds record text. It raises `ValueError` for an
  Annotate result, and for a Rewrite row whose rewritten text is null, which has no output to count. `outcome(row)`
  keeps that row's reasons (#426, #431).

## Pending changes

These are not on the #319 branch yet. Each lands with its change.

### #455: one rule for applying edits

**Five places that rebuild a text from replacement spans now follow one rule**: local replacement, tagged text for
prompts, the `verify-patches` check and Rewrite's baseline use `anonymizer.engine.edits`, and `assemble_regions` keeps
its own copy of the rule, since the graph package imports nothing outside itself. Three inputs give a different
result:
- A tag whose range is not in the text is dropped from the tagged text a prompt sees. Before, a tag past the end was
  appended empty, a tag reaching past the end swallowed the tail, and an inverted range repeated text. Built-in
  detection never produces such a range; a custom detector's or runner override's entity can.
- On the Rewrite runner-override path, a row whose entities overlap is "not ready" (the rewrite gets the original
  tagged text and no baseline). Before, the row stayed ready and one tag was dropped silently. Stock detection never
  hands Rewrite overlapping entities.
- `assemble_regions` (`anonymizer.graph`) refuses a range with a `bool` or float bound as `region_out_of_bounds`.
  Before, `True` was sliced as 1 and a float raised `TypeError`.
- What to do: nothing for built-in detection. A custom detector or runner should return in-range, non-overlapping
  entities, and graph code should pass integer ranges.

### #NNN: privacy coverage reason

**A Rewrite row whose final privacy evaluation left a question unanswered is withheld.** An omitted question or a
null answer is still scored as leaked with full confidence. The row's outcome is now `not_met` with
`could_not_assess` on the check `privacy_coverage` (cause `privacy_items_omitted`), its rewritten text is withheld
(null, `available=False`) and `needs_human_review` is `True`.
- Availability cost: such rows used to be released when the leak scored for the unanswered question stayed under the
  threshold. They are now withheld.
- An omission in an earlier repair round does not count against a repaired candidate that answered every question.
- The run ends with a warning counting the rows whose privacy evaluation omitted a question in any round.
- What to do: treat these rows like rows whose privacy evaluation was unavailable; the warning's count shows how
  often your evaluator model leaves questions out.

### Detection contract

**A custom `detection_workflow` returns a `DetectorResult`.** This replaces the frame with `final_entities` and
`_entities_by_value` columns, and it applies to every detector other than the built-in `EntityDetectionWorkflow`
class itself, including its subclasses and `Mock(spec=EntityDetectionWorkflow)`.
- A `DetectorResult` holds one `DetectorRow(record_id, spans, extra_values, latent)` per answered input record, named
  by its `_anonymizer_record_id`, and the records the detector failed. `DetectedSpan`, `DetectorRow`,
  `DetectorResult`, `FailedRecord` (for `DetectorResult.failed`), `InvalidDetectorResultError` and `TEXT_COLUMN` (the
  name of the text column a detector receives) are exported from `anonymizer`.
- A result that cannot be placed against the input raises `InvalidDetectorResultError` before any model call: any
  other return type, a malformed span, extra value or failure, an id that is not an input id, or an id answered or
  failed twice. Before, each case went differently: a frame whose rows could not be placed was rerouted, and the run
  worked over it as returned, without these checks; a return value that is not a frame, such as a dict, raised
  `AttributeError`; and a span with a null label failed its record at `replace`.
- A record the result neither answers nor fails fails at `detect` with `dropped_output`.
- The engine renders `final_entities`, `_entities_by_value`, `tagged_text` and `_latent_entities` from the result. A
  detector-supplied text column, `tagged_text` or `source_positions` is no longer read.
- Failures from a custom `detection_workflow` are named by the input record's id (the `TextRecord` id, or a
  generated id for file input), not by the returned row's index.
- What to do: build one `DetectorRow` per input row from `dataframe["_anonymizer_record_id"]`, move each span from
  the `final_entities` dict into `DetectedSpan(start, end, label, value)`, and move any value listed only in
  `_entities_by_value` into `extra_values`. See "Custom detection workflows".

**A custom runner's `run()` must keep `_anonymizer_record_id` on every row it returns.** This applies to a
`replace_runner` or `rewrite_runner` that overrides `run()`.
- A returned row with no record id, an id that is not an input id, or an id returned twice is not released, and each
  input record that no released row answers fails at that runner's step with `unplaced_output`. Such rows were
  released before.
- What to do: keep the `_anonymizer_record_id` column of the frame the runner receives, unchanged, on every row it
  returns, and return at most one row per record.

**Other detection and Replace changes.**
- The "Detection complete — N entities found" log line counts the final (kept) entities for every detector, the
  built-in one included. Before, it counted the built-in detector's validated entities before label filtering, and 0
  for a custom detector.
- Built-in detection in the Replace modes now replaces a missed repeat: where a detected value occurs again without a
  span, a span with the first detected span's label is added (`"source": "repeat"` in `final_entities`), and the
  occurrence is replaced like the others. It was released unreplaced before. The added span covers the whole matched
  word, so `order #W3916020` becomes `order [REDACTED_ORDER_ID]`. A repeat that crosses a kept span's edge gets no
  span, and its row fails at `detect` with `unspanned_value`; such a row was released before. Rewrite is unchanged.
- A `replace_runner` whose `run_with_detection` override does not accept the `detector` keyword (or `**kwargs`) now
  raises `InvalidConfigError` before detection, naming the keyword. The facade passes
  `detector="builtin" | "custom"` on every call. With a custom detector such an override raised `TypeError` on an
  `unspanned_check` keyword, which `detector` replaces. A `rewrite_runner` override receives no new keyword.
- Substitute rows gain a `placeholder_in_text` column, `True` when the replaced text keeps a placeholder substitute
  (`[SUBSTITUTE_<LABEL>_<n>]`) of the row's own replacement map. The row is not failed and stays `met`, as in Rewrite,
  where the same situation flags the row for review.

### Latent-only Rewrite records

**A Rewrite record whose only finding is latent is now rewritten and evaluated.** A latent entity is sensitive
information the text implies without naming it, such as an occupation implied by "night shifts on the cardiac ward".
Before, a record with latent entities and no explicit entity was returned unchanged and reported as protected,
without any model request.
- Each such record now goes through the full rewrite: about 8 model requests per record, and its text changes. The
  privacy goal decides whether each latent entity is protected, as it already did for latent entities on records with
  explicit ones. A record whose latent entities are all judged low sensitivity is still rewritten, as a record whose
  explicit entities are all low is.
- A record whose rewrite still leaks a latent entity is withheld and flagged for review, as any Rewrite record is.
- `evaluate()` now runs the rewrite judge on these records. The detection validity judge still reads explicit
  entities only, so these records keep `detection_valid` 1.0 and the judge is not asked.
- A record whose latent detection fails still fails, and is neither rewritten nor released.
- Records with neither explicit nor latent entities still pass through unchanged, with no model request.
- A custom detector's `DetectorRow(latent=())` means "no latent entity found". There is no way yet to say "latent
  entities not assessed"; that state is open.
- What to do: expect more model requests and changed text for records whose only finding is latent. A custom
  detector that does not look for latent entities should return `latent=()`.

### #407: frozen config and one options object

**Config fields cannot be reassigned.** `AnonymizerConfig`, `Detect`, `Rewrite`, `PrivacyGoal`, the replace
strategies and the regex models are frozen.
- What to do: use `model_copy(update=...)`.

**Unknown keys are rejected.** The config and input models raise a validation error that names an unknown key,
without echoing its value. `TextRecord` still ignores extra keys. Dumps written by v0.5.0 still load, except one
that holds `Hash`, which needs its key (see "`Hash` requires a secret key").
- What to do: remove or correct the key the error names.

**`Rewrite.instructions` is removed.** It never reached the rewrite model. Passing a value raises an error that points
to `privacy_goal`. A dump with `instructions: null` still loads. `Rewrite.privacy_goal` is typed `PrivacyGoal`, and
`None` gives the default goal.
- What to do: move the text into `privacy_goal.protect` and `privacy_goal.preserve`.

**Custom detection workflows take an options object.** An object passed as `Anonymizer(detection_workflow=...)` is
called as `run(dataframe, *, options, context, latent_goal=None)`. Import `DetectionWorkflow`, `DetectionOptions` and
`RunContext` from `anonymizer.interface.workflows`. The per-setting keywords, and the signature check that kept older
detectors working, are gone. `validation_single_chunk_full_text` and `compute_grouped_entities` are no longer
keywords. A detector that still takes the old keywords fails with `TypeError` at its first call.
- What to do: change the detector's signature. The #407 description maps each old keyword to its new place.

**Custom rewrite runners take options and context.** A `rewrite_runner` or `combined_rewrite_runner` that overrides
`run()` is called as `run(dataframe, *, options, context)`, where `options` is the `Rewrite` config. It used to be
called with eight keywords: `model_configs`, `selected_models`, `replace_model_selection`, `privacy_goal`,
`evaluation`, `data_summary`, `preview_num_records` and `strict_entity_protection`.
- What to do: change the override's signature.

**Detection export needs the builders.** `export_detection_config` and `export_detection_builder_for_seed` raise
`InvalidConfigError` when the injected detection workflow does not define the export builders.

**Reserved labels are rejected earlier.** A `nagroup_g<N>` label is rejected when `Detect` or `RegexRule` is built,
not when the run starts.

### #408: caller-held Session

**`session=` on `run` and `preview`.** A `Session` from `anonymizer.graph` keeps substitutes stable across calls.
Under a stable session, `run` and `preview` reject Rewrite and a custom replace runner that overrides `run()`. The
session and its snapshots are provisional.
- In run mode (`consistency="run"`), a value detected under two labels can keep one substitute, where run mode used
  to give the second label a placeholder; for such a value, record-mode (default) output is unchanged from v0.5.0.
- What to do: in run mode, if you relied on a placeholder for the second label, read `result.replacement_map(row)`,
  whose entries carry each label and its substitute.
- Without a stable session, including `Session()`, a custom replace runner's `run()` and its map workflow's
  `generate_validated_maps()` are called as before `session` existed, so overrides with the older signatures keep
  working. A `run_with_detection()` override must also accept `detector` (or `**kwargs`); one that does not raises
  `InvalidConfigError` before detection (see "Detection contract").
- Under a stable session, a custom replace runner is also rejected before detection when it overrides
  `run_with_detection()` or its map workflow's `generate_validated_maps()`, even when the override forwards
  `**kwargs` or `session`, and when the stock method is bound to another object.
- Under a stable session, a value detected under two labels never gets one substitute as a record-local fallback
  under one label and as the shared substitute under the other. The second use gets a placeholder instead, in either
  row order and across calls. Replaying the same rows, when their values all take their shared substitutes, gives
  the same text and snapshot, whatever order the model returns its entries or the detector its rows in; letter case
  follows each value's earliest mention, and a substitute that re-casing would turn into another value (`Işıl`) is
  used as assigned. A call whose earliest spelling of a value differs can get another letter case. A row that itself
  needs a fallback gets a new placeholder on each replay, because fallbacks are not reused.
- `result.replacement_map(row)` entries carry `placeholder`, a token-shape check tied to the entry's label: it is
  true for a substitute shaped like that label's placeholder, such as `[SUBSTITUTE_FIRST_NAME_1]` for `first_name`, in
  any letter case, whether Anonymizer minted it, the session held it, or the model proposed it.
  `result.session_changes` counts only the placeholders minted in the call.

**Placeholders are kept as minted.** A variant of a value that received a placeholder gets the placeholder exactly as
minted, not re-cased (for example `[substitute_first_name_1]`). v0.5.0 had no re-casing, so this changes output only
against the #319 branch.

**Graph SDK additions.** `prepare` and `protect` take `session=`. `PreparedProtection` and `ProtectionResult` gain one
trailing field each. Operation outputs named `session`, `identities` or `record` are rejected, and the subject keys
`by-value` and `substitute-fallback` are reserved.

### #413, #416, #419

**#413: graph port values must be plain.** The built-in dict port kinds accept only plain JSON-like values. A value
with no plain form, such as `pd.NA` or a timestamp, fails its target as `invalid_output`. This affects
`anonymizer.graph` users only.

**#416: one value under two labels keeps a substitute.** A value detected under two labels no longer loses its
replacement when the model joins the labels or changes case. In record mode, one string under two labels gets one
substitute.

**#419: detector facts are checked privately.** No user-visible change. The detector declaration fields removed from
`Detect` in #418 were never released.
