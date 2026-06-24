---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: anonymizer
description: Use when the user wants to anonymize a text dataset, redact PII, de-identify free-text data, or rewrite text to remove sensitive or inferable identifying information. Produces a runnable Python script that calls the NeMo Anonymizer pipeline (detection → replace or rewrite).
argument-hint: [describe the dataset and how you want it anonymized]
---

# Before You Start

Do not explore the workspace first. The workflow's data-inspection step shows you what you need.

# Goal

Anonymize a text dataset using NeMo Anonymizer in the way the user describes:

$ARGUMENTS

The output is a single runnable Python script that builds an `AnonymizerConfig`, previews results on a few rows, inspects failures and quality metrics, optionally scores Replace output with LLM-as-judge evaluation, and (on user approval) runs the full pipeline. The script is the durable artifact — the user keeps it for re-runs, version control, and production.

# Workflow

Read `workflows/interactive.md` and follow it. Anonymization is high-stakes — there is no autopilot mode. Even when the user says "you decide" or "be opinionated", you still ask the minimum questions needed to choose `risk_tolerance` and phrase `privacy_goal`, because those are fundamentally the user's calls based on their regulatory and business context.

# Rules

- **Always preview before running the full pipeline.** Preview is cheap; a full run can be expensive and slow.
- **If `result.failed_records` is non-empty after preview, fix that *before* tweaking strategy.** Dropped rows are a model/provider/infra problem (rate limits, auth, etc.), not a config problem. Strategy knobs won't help. See `docs/troubleshooting.md` "Did the run actually complete cleanly?".
- **Ask the user which mode.** Briefly describe both: Replace detects entities and replaces each in place (faster, cheaper, keeps shape); Rewrite transforms the full text to also remove inferable identifiers (more expensive, may restructure). Use the data shape as a hint — free-text with implicit identifiers (clinical notes, biographies, depositions) leans Rewrite; structured records / log lines lean Replace — but the user picks.
- **For cross-record consistency** (same value → same replacement everywhere), use `Hash`, not `Substitute`. `Substitute` is consistent within a row only.
- **In Replace mode, default to `Substitute`** if the user hasn't specified a strategy. It's the most general-purpose choice and matches the bulk of production usage.
- **`Annotate` is for inspection, not production.** Its output keeps the original entity text and is not privacy-safe. Use it during iteration to confirm detection is working, then switch.
- **Evaluation is opt-in and runs as a separate step** (Replace mode). After `preview()` / `run()`, call `anonymizer.evaluate(result)` to score the output with LLM-as-judge. `Substitute` gets four judges (detection validity, type fidelity, relational consistency, attribute fidelity); `Redact` / `Annotate` / `Hash` get the detection-validity judge only. Evaluation is diagnostic — it scores quality, it does not change the anonymized output.
- **Always set `AnonymizerInput.data_summary`**, even briefly. It is the single cheapest quality lever and it improves both detection and rewrite.
- **Never claim privacy guarantees.** Anonymizer is best-effort. Outputs may need human review depending on `risk_tolerance`. Tell the user this when you finalize.

# Usage Tips and Common Pitfalls

- **`Detect.entity_labels=None` (the default) is permissive** — the augmenter LLM may invent labels not in `DEFAULT_ENTITY_LABELS`. Setting an explicit list switches to **strict mode** where *only* the listed labels are detected. To add domain labels, *extend* the default, don't replace it: `entity_labels=[*DEFAULT_ENTITY_LABELS, "clinical_facility", ...]` (`DEFAULT_ENTITY_LABELS` is a tuple, so unpack it into a list). Match the snake_case convention of `DEFAULT_ENTITY_LABELS`.
- **GLiNER is zero-shot** — entity labels are natural-language concept names (e.g. `"clinical_facility"`, `"internal_project_codename"`), not codes or enum values. Any concept you can name in English is a label GLiNER can detect.
- **`Rewrite.instructions` is a dead field today** — it exists on the model but the rewrite engine never reads it. Do not use it. Put rewriter guidance in `privacy_goal.protect` / `privacy_goal.preserve` instead.
- **`risk_tolerance` only applies to Rewrite mode**, not Replace.
- **`PrivacyGoal.protect` and `.preserve` must each be 10–1000 chars and at least 3 words.** Be specific (categories, named identifiers, structural facets); avoid generic phrasing like "preserve meaning".
- **Validator pool is the only model role with built-in load-spreading.** Set `entity_validator: [a, b, c]` in `models.yaml` if rate limits drop rows. Other roles (rewriter, evaluator, etc.) are single-alias.
- **Self-hosted GLiNER** — when detection must not call `build.nvidia.com` (PHI on-prem, air-gapped, latency), run the reference server from a **source checkout** (`python tools/serve_gliner.py`; not installed via `pip install nemo-anonymizer`). Add a provider with `endpoint: http://localhost:8001/v1`, route `entity_detector` via a `gliner-pii-detector` alias with `provider: local-gliner` and `skip_health_check: true`. If you pass `--port` or `--host` to the server, set the provider `endpoint` to the same host/port. `model_configs` is a **complete** model pool, not an overlay — copy `src/anonymizer/config/default_model_configs/models.yaml` and change only the detector entry (keep `gpt-oss-120b` and `nemotron-30b-thinking`). See [`docs/concepts/self-hosting-gliner.md`](../../docs/concepts/self-hosting-gliner.md).
- **The evaluation judges use their own model roles** (`detection_validity_judge`, `replace_type_fidelity_judge`, `replace_relational_consistency_judge`, `replace_attribute_fidelity_judge`), configured in the `evaluate` section of `models.yaml`. They are **not** consumed by `preview()` / `run()`, so a config that anonymizes fine can still fail validation at `evaluate()` if those roles are unset. Defaults ship in `src/anonymizer/config/default_model_configs/evaluate.yaml`.
- **`*_valid` verdict columns are `True` / `False` / `None`.** `None` means the judge was unavailable (model/infra failure), **not** that the row passed — treat it as "unscored", never as a pass. Inspect verdicts per record with `evaluated.display_record(i)`.
- **`EvaluateConfig` is an empty placeholder today** — no knobs to set. `anonymizer.evaluate(result)` is the whole API; pass nothing else.

# Reference Docs

The agent should consult these as it goes — *do not* try to enumerate field reference inline:

- [`docs/concepts/choosing-a-strategy.md`](../../docs/concepts/choosing-a-strategy.md) — decision tree for picking mode (Replace vs Rewrite), strategy (Substitute / Redact / Annotate / Hash), risk tolerance, privacy goal phrasing, and detection knobs.
- [`docs/troubleshooting.md`](../../docs/troubleshooting.md) — symptom-first guide for diagnosing dropped rows, leakage, low utility, and pipeline failures. Read **section by section as symptoms appear**, not all at once.
- [`docs/concepts/detection.md`](../../docs/concepts/detection.md) — detection internals (GLiNER threshold semantics, label catalogue, augmenter/validator behavior).
- [`docs/concepts/models.md`](../../docs/concepts/models.md) — model role configuration, validator pools.
- [`docs/concepts/self-hosting-gliner.md`](../../docs/concepts/self-hosting-gliner.md) — local `entity_detector` server (`tools/serve_gliner.py`), OpenAI-compatible contract, YAML wiring.

# Troubleshooting

Environment-level issues only. Quality and pipeline issues are in `docs/troubleshooting.md`.

- **`anonymizer` not installed:** Tell the user `nemo-anonymizer` is not in this Python environment (requires Python ≥ 3.11). Ask if they want you to install it (`pip install nemo-anonymizer`) or do it themselves. Do not install without permission.
- **Model/provider setup:** Plain `Anonymizer()` ships with bundled `models.yaml` and `providers.yaml` (see `src/anonymizer/config/default_model_configs/`). For the default path, confirm `NVIDIA_API_KEY` is set. Pass custom `model_configs` and/or `model_providers` only when targeting non-default endpoints or model pools — see `docs/concepts/models.md`.
- **LLM calls failing at preview:** Usually an auth issue (missing or invalid API key), a network problem, or a wrong endpoint URL. See `docs/troubleshooting.md` "Validation passed but `preview` errors at LLM call".
- **Local / on-prem GLiNER:** Clone or download `tools/serve_gliner.py` from the Anonymizer repo, start the server, add a provider with `endpoint: http://localhost:8001/v1`, and point `gliner-pii-detector` at `provider: local-gliner` with `skip_health_check: true`. Preflight errors about missing aliases usually mean `model_configs` only listed the detector — include the full default pool. Wrong `endpoint` or a down server surfaces as detection failures at preview — see [`docs/concepts/self-hosting-gliner.md`](../../docs/concepts/self-hosting-gliner.md).

# Output Template

Write a Python script to the current directory. Name it after the dataset (e.g. `anonymize_clinical_notes.py`, `anonymize_support_logs.py`). Use this template — fill in the TODO markers, drop unused sections.

```python
"""Anonymize <dataset> using NeMo Anonymizer.

Generated by the anonymizer agent skill.

Usage:
    python <this_script>.py                 # preview on 5 rows (fast, cheap)
    python <this_script>.py --full          # run on the full dataset
    python <this_script>.py --evaluate      # preview 5 rows, then LLM-judge-score those rows
    python <this_script>.py --full --evaluate  # run full dataset, then score the full output
"""

from __future__ import annotations

import argparse
import sys

from anonymizer import (
    Anonymizer,
    AnonymizerConfig,
    AnonymizerInput,
    DEFAULT_ENTITY_LABELS,
    Detect,
    # Pick what you need:
    # Replace mode:
    Substitute, Redact, Annotate, Hash,
    # Rewrite mode:
    Rewrite, PrivacyGoal,
)


def build_config() -> tuple[AnonymizerInput, AnonymizerConfig]:
    """Single source of truth for what we anonymize and how."""
    data = AnonymizerInput(
        source="TODO: path to .csv / .parquet / .jsonl",
        text_column="TODO: name of the text column",
        data_summary="TODO: one-line description of the data (domain, genre, anything non-obvious)",
    )

    detect = Detect(
        # Add domain labels by *extending* the default, not replacing it.
        # entity_labels=[*DEFAULT_ENTITY_LABELS, "clinical_facility", "diagnosis_code"],
        gliner_threshold=0.3,  # default; lower (0.2) for recall, raise (0.5) for cost savings
    )

    # ---- Pick ONE of the two strategies below ----

    # Replace mode (Substitute | Redact | Annotate | Hash):
    # config = AnonymizerConfig(detect=detect, replace=Substitute(
    #     instructions="TODO: short hint about the domain (e.g. names should remain plausible "
    #                  "for the original cultural context)",
    # ))

    # Rewrite mode (free-text de-identification with inferable-identifier suppression):
    config = AnonymizerConfig(
        detect=detect,
        rewrite=Rewrite(
            privacy_goal=PrivacyGoal(
                protect="TODO: what must not appear in the output, even by inference",
                preserve="TODO: what must be kept so the rewritten text is still useful",
            ),
            risk_tolerance="low",          # minimal | low | moderate | high
            strict_entity_protection=False, # True = force every detected entity into a protective disposition
            max_repair_iterations=3,
        ),
    )
    return data, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Run on full dataset (default: preview 5 rows)")
    parser.add_argument("--num-records", type=int, default=5, help="Rows to preview (ignored with --full)")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="LLM-judge-score the output produced this run (preview rows, or full output with --full)",
    )
    args = parser.parse_args()

    anonymizer = Anonymizer()
    data, config = build_config()

    if args.full:
        result = anonymizer.run(config=config, data=data)
        out_path = "output.parquet"  # TODO: change path/format (.csv, .jsonl) as needed
        result.dataframe.to_parquet(out_path)
        print(f"Wrote {len(result.dataframe)} rows to {out_path}")
    else:
        result = anonymizer.preview(config=config, data=data, num_records=args.num_records)
        print(f"Previewed {len(result.dataframe)} rows.")

        # Save preview output so you can investigate without re-running.
        # trace_dataframe is a superset of dataframe — it has the user-facing
        # columns plus internal columns (validation decisions, sensitivity
        # dispositions, etc.) that explain why entities were kept, dropped,
        # or rewritten.
        result.trace_dataframe.to_parquet("preview.parquet")
        print("Saved: preview.parquet (load with pd.read_parquet)")

    # Failure-first protocol: dropped rows are infra issues, not strategy issues.
    if result.failed_records:
        print(f"\n⚠️  {len(result.failed_records)} record(s) failed:")
        for fr in result.failed_records[:3]:
            print(f"   - record_id={fr.record_id} step={fr.step} reason={fr.reason}")
        print("\nFix dropped rows before tweaking strategy. See docs/troubleshooting.md.")
        sys.exit(1)

    # Optional LLM-as-judge evaluation (Replace mode). Opt-in, separate step:
    # scores how well detection + replacement worked without changing the
    # output. Substitute -> 4 judges (detection validity, type fidelity,
    # relational consistency, attribute fidelity); Redact/Annotate/Hash ->
    # detection-validity judge only. Needs the `evaluate` model roles in
    # models.yaml (see src/anonymizer/config/default_model_configs/evaluate.yaml).
    if args.evaluate and config.replace is not None:
        result = anonymizer.evaluate(result)
        df = result.dataframe
        for col in (
            "detection_valid",
            "type_fidelity_valid",
            "relational_consistency_valid",
            "attribute_fidelity_valid",
        ):
            if col in df.columns:
                passed = int(df[col].eq(True).sum())  # None = unscored, never a pass
                scored = int(df[col].notna().sum())
                print(f"{col}: {passed}/{scored} passed ({len(df) - scored} unscored)")
        # In a notebook, inspect per-record verdicts visually:
        #   result.display_record(0)

    # Rewrite-mode quality summary (skip for Replace mode).
    if config.rewrite is not None:
        df = result.dataframe
        print(f"\nleakage_mass:   mean={df['leakage_mass'].mean():.3f}  max={df['leakage_mass'].max():.3f}")
        print(f"utility_score:  mean={df['utility_score'].mean():.3f}  min={df['utility_score'].min():.3f}")
        print(f"flagged for review: {int(df['needs_human_review'].sum())} / {len(df)}")


if __name__ == "__main__":
    main()
```
