---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: anonymizer
description: Use when the user wants to anonymize a text dataset, redact PII, de-identify free-text data, or rewrite text to remove sensitive or inferable identifying information. Produces a runnable Python script that calls the NeMo Anonymizer pipeline (detection → replace or rewrite).
license: Apache-2.0
metadata:
  author: Aaron Gonzales <aagonzales@nvidia.com>
---

# Before You Start

Do not explore the workspace first. The workflow's data-inspection step shows you what you need.

# Goal

Anonymize a text dataset using NeMo Anonymizer in the way the user describes:

$ARGUMENTS

The output is a single runnable Python script that builds an `AnonymizerConfig`, previews results on a few rows, inspects failures and quality metrics, optionally scores output with LLM-as-judge evaluation (Replace and Rewrite modes), and (on user approval) runs the full pipeline. The script is the durable artifact — the user keeps it for re-runs, version control, and production.

# Workflow

Read `references/interactive.md` and follow it. Anonymization is high-stakes,
so there is no autopilot mode. Even when the user says "you decide" or "be
opinionated", ask the minimum questions needed to choose `risk_tolerance` and
phrase `privacy_goal`. The user must make those choices based on their
regulatory and business context.

# Rules

- **Always preview before running the full pipeline.** Preview is cheap; a full run can be expensive and slow.
- **If `result.failed_records` is non-empty after preview, fix that *before* tweaking strategy.** Dropped rows are a model/provider/infra problem (rate limits, auth, etc.), not a config problem. Strategy knobs won't help. See `docs/troubleshooting.md` "Did the run actually complete cleanly?" or the [published troubleshooting guide](https://nvidia-nemo.github.io/Anonymizer/dev/troubleshooting/).
- **Ask the user which mode.** Briefly describe both: Replace detects entities and replaces each in place (faster, cheaper, keeps shape); Rewrite transforms the full text to also remove inferable identifiers (more expensive, may restructure). Use the data shape as a hint — free-text with implicit identifiers (clinical notes, biographies, depositions) leans Rewrite; structured records / log lines lean Replace — but the user picks.
- **For cross-record consistency** (same value → same replacement everywhere), use `Hash`, not `Substitute`. `Substitute` is consistent within a row only.
- **In Replace mode, default to `Substitute`** if the user hasn't specified a strategy. It's the most general-purpose choice and matches the bulk of production usage.
- **`Annotate` is for inspection, not production.** Its output keeps the original entity text and is not privacy-safe. Use it during iteration to confirm detection is working, then switch.
- **Evaluation is opt-in and runs as a separate step** (Replace and Rewrite modes). After `preview()` / `run()`, call `anonymizer.evaluate(result)` to score the output with LLM-as-judge. Replace mode: `Substitute` gets four judges (detection validity, type fidelity, relational consistency, attribute fidelity); `Redact` / `Annotate` / `Hash` get detection-validity only. Rewrite mode: detection validity + holistic privacy/quality/style judge. Evaluation is diagnostic — it scores quality, it does not change the anonymized output.
- **Always set `AnonymizerInput.data_summary`**, even briefly. It is the single cheapest quality lever and it improves both detection and rewrite.
- **Never claim privacy guarantees.** Anonymizer is best-effort. Outputs may need human review depending on `risk_tolerance`. Tell the user this when you finalize.

# Usage Tips and Common Pitfalls

- **`Detect.entity_labels=None` (the default) is permissive** — the augmenter LLM may invent labels not in `DEFAULT_ENTITY_LABELS`. Setting an explicit list switches to **strict mode** where *only* the listed labels are detected. To add domain labels, *extend* the default, don't replace it: `entity_labels=[*DEFAULT_ENTITY_LABELS, "clinical_facility", ...]` (`DEFAULT_ENTITY_LABELS` is a tuple, so unpack it into a list). Match the snake_case convention of `DEFAULT_ENTITY_LABELS`.
- **GLiNER is zero-shot** — entity labels are natural-language concept names (e.g. `"clinical_facility"`, `"internal_project_codename"`), not codes or enum values. Any concept you can name in English is a label GLiNER can detect.
- **`Rewrite.instructions` is a dead field today** — it exists on the model but the rewrite engine never reads it. Do not use it. Put rewriter guidance in `privacy_goal.protect` / `privacy_goal.preserve` instead.
- **`risk_tolerance` only applies to Rewrite mode**, not Replace.
- **`PrivacyGoal.protect` and `.preserve` must each be 10–1000 chars and at least 3 words.** Be specific (categories, named identifiers, structural facets); avoid generic phrasing like "preserve meaning".
- **Validator pool is the only model role with built-in load-spreading.** Set `entity_validator: [a, b, c]` in `models.yaml` if rate limits drop rows. Other roles (rewriter, evaluator, etc.) are single-alias.
- **Self-hosted GLiNER:** When detection must not call `build.nvidia.com` (PHI on-prem, air-gapped, latency), run the reference server from a **source checkout** with `python tools/serve_gliner.py`. The server is not installed by `pip install nemo-anonymizer`. Add a provider with `endpoint: http://localhost:8001/v1`, then route `entity_detector` through a `gliner-pii-detector` alias with `provider: local-gliner` and `skip_health_check: true`. Match any custom `--port` or `--host` in the provider endpoint. `model_configs` is a **complete** model pool, not an overlay. Copy `src/anonymizer/config/default_model_configs/models.yaml` and change only the detector entry, keeping `gpt-oss-120b` and `nemotron-30b-thinking`. See [`docs/concepts/self-hosting-gliner.md`](../../docs/concepts/self-hosting-gliner.md) or the [published self-hosting guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/self-hosting-gliner/).
- **The evaluation judges use their own model roles** (`detection_validity_judge`, `replace_type_fidelity_judge`, `replace_relational_consistency_judge`, `replace_attribute_fidelity_judge`, `rewrite_judge`), configured in the `evaluate` section of `models.yaml`. They are **not** consumed by `preview()` / `run()`, so a config that anonymizes fine can still fail validation at `evaluate()` if those roles are unset. Defaults ship in `src/anonymizer/config/default_model_configs/evaluate.yaml`.
- **Verdict columns are null when the judge was unavailable** — `None` means "unscored", never a pass. Replace verdict columns (`type_fidelity_valid`, etc.) are `True` / `False` / `None`. Rewrite `detection_valid` is a `0–1` float fraction (or `None` if unscored). Inspect verdicts per record with `evaluated.display_record(i)`.
- **`EvaluateConfig` is an empty placeholder today** — no knobs to set. `anonymizer.evaluate(result)` is the whole API; pass nothing else.

# Reference Docs

The agent should consult these as it goes — *do not* try to enumerate field reference inline:

- [`docs/concepts/choosing-a-strategy.md`](../../docs/concepts/choosing-a-strategy.md) or the [published strategy guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/choosing-a-strategy/) for choosing a mode, replacement strategy, risk tolerance, privacy goal, and detection settings.
- [`docs/troubleshooting.md`](../../docs/troubleshooting.md) or the [published troubleshooting guide](https://nvidia-nemo.github.io/Anonymizer/dev/troubleshooting/) for dropped rows, leakage, low utility, and pipeline failures. Read the relevant section when a symptom appears.
- [`docs/concepts/detection.md`](../../docs/concepts/detection.md) or the [published detection guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/detection/) for GLiNER threshold semantics, entity labels, augmentation, and validation.
- [`docs/concepts/evaluation.md`](../../docs/concepts/evaluation.md) or the [published evaluation guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/evaluation/) for Replace and Rewrite evaluation, judge roles, result columns, and saved-result evaluation.
- [`docs/concepts/models.md`](../../docs/concepts/models.md) or the [published models guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/models/) for model roles and validator pools.
- [`docs/concepts/self-hosting-gliner.md`](../../docs/concepts/self-hosting-gliner.md) or the [published self-hosting guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/self-hosting-gliner/) for the local `entity_detector` server, OpenAI-compatible contract, and YAML configuration.

# Troubleshooting

This section covers environment-level issues. For quality and pipeline issues,
read `docs/troubleshooting.md` or the
[published troubleshooting guide](https://nvidia-nemo.github.io/Anonymizer/dev/troubleshooting/).

- **`anonymizer` not installed:** Tell the user `nemo-anonymizer` is not in this Python environment (requires Python ≥ 3.11). Ask if they want you to install it (`pip install nemo-anonymizer`) or do it themselves. Do not install without permission.
- **Model/provider setup:** Plain `Anonymizer()` ships with bundled `models.yaml` and `providers.yaml` (see `src/anonymizer/config/default_model_configs/`). For the default path, confirm `NVIDIA_API_KEY` is set. Pass custom `model_configs` or `model_providers` only for non-default endpoints or model pools. See `docs/concepts/models.md` or the [published models guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/models/).
- **LLM calls failing at preview:** Check for a missing or invalid API key, a network problem, or a wrong endpoint URL. See `docs/troubleshooting.md` "Validation passed but `preview` errors at LLM call" or the [published troubleshooting guide](https://nvidia-nemo.github.io/Anonymizer/dev/troubleshooting/).
- **Local / on-prem GLiNER:** Clone or download `tools/serve_gliner.py` from the Anonymizer repo, start the server, add a provider with `endpoint: http://localhost:8001/v1`, and point `gliner-pii-detector` at `provider: local-gliner` with `skip_health_check: true`. Preflight errors about missing aliases usually mean `model_configs` lists only the detector. Include the full default pool. A wrong endpoint or stopped server surfaces as a detection failure during preview. See [`docs/concepts/self-hosting-gliner.md`](../../docs/concepts/self-hosting-gliner.md) or the [published self-hosting guide](https://nvidia-nemo.github.io/Anonymizer/dev/concepts/self-hosting-gliner/).

# Output Template

Write a Python script to the current directory. Name it after the dataset (for
example, `anonymize_clinical_notes.py` or `anonymize_support_logs.py`). Fill in
the TODO markers in this template and remove unused sections.

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
        print("\nFix dropped rows before tweaking strategy. See docs/troubleshooting.md or https://nvidia-nemo.github.io/Anonymizer/dev/troubleshooting/.")
        sys.exit(1)

    # Optional LLM-as-judge evaluation (Replace and Rewrite modes). Opt-in, separate
    # step — scores quality without changing the anonymized output.
    # Replace: Substitute -> 4 judges (detection validity, type fidelity, relational
    #   consistency, attribute fidelity); Redact/Annotate/Hash -> detection-validity only.
    # Rewrite: detection validity + holistic privacy/quality/style judge.
    # Needs the `evaluate` model roles in models.yaml
    # (see src/anonymizer/config/default_model_configs/evaluate.yaml).
    if args.evaluate:
        result = anonymizer.evaluate(result)
        df = result.dataframe
        if config.replace is not None:
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
        else:
            # Rewrite: detection_valid is a 0–1 fraction (mean is more informative than pass count).
            if "detection_valid" in df.columns:
                scored = int(df["detection_valid"].notna().sum())
                mean_val = df["detection_valid"].mean()
                print(f"detection_valid: mean={mean_val:.2f}  scored={scored}/{len(df)}")
            if "judge_evaluation" in df.columns:
                scored = int(df["judge_evaluation"].notna().sum())
                print(f"judge_evaluation: {scored}/{len(df)} scored")
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
