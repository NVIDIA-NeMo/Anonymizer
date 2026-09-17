# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# <!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# -->
# # 🕵️ Rewriting Biographies
#
# Instead of replacing entities with tokens, rewrite mode generates a
# privacy-safe transformation of the entire text. The `run()` / `preview()` pipeline:
#
# 1. Detects entities (same as replace mode, plus latent entity detection)
# 2. Classifies the domain and assigns sensitivity dispositions
# 3. Generates a rewritten version that obscures sensitive entities
# 4. Evaluates quality (utility) and privacy (leakage) with an automated repair loop
#
# Afterward, a separate optional `evaluate()` call runs LLM judges for
# detection validity and holistic privacy, quality, and style scores.
#
#
# #### 📚 What you'll learn
#
# - Configure rewrite mode with `PrivacyGoal` to specify what to protect and what to preserve
# - Set evaluation criteria and risk tolerance for automated quality checks
# - Preview rewritten text and inspect utility / leakage scores
# - Triage flagged records with `needs_human_review`
# - Run `evaluate()` for detection validity and holistic judge scores (privacy, quality, style)
#
# > **Tip:** First time running notebooks? Start with
# > [setup instructions](https://nvidia-nemo.github.io/Anonymizer/latest/tutorials/).

# %% [markdown]
# ## ⚙️ Setup
#
# - Install the notebook extra, then provide credentials for the configured external LLM providers.
# - `create_anonymizer()` starts pinned GLiNER2 locally and selects CUDA, MPS, or CPU automatically.
# - The default external LLM models currently use [OpenRouter](https://openrouter.ai); its terms and privacy practices apply.
#
# > **Data boundary:** GLiNER2 detection runs locally in this notebook environment. LLM-assisted validation,
# > augmentation, replacement, rewriting, repair, and evaluation use configured external hosts and may send
# > them original or tagged input text. Do not treat this configuration as an all-local privacy boundary.
# - `configure_logging(LoggingConfig.default())` keeps logs at INFO. Switch to `LoggingConfig.debug()` when troubleshooting.

# %%
import getpass
import os
import subprocess
import sys

package_spec = os.getenv("ANONYMIZER_NOTEBOOK_PACKAGE", "nemo-anonymizer[notebooks]")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package_spec])

# %%
from anonymizer.notebooks import required_api_key_environment_variables

for variable in required_api_key_environment_variables():
    key = getpass.getpass(f"Enter {variable}: ").strip()
    if not key:
        raise RuntimeError(f"{variable} is required by the configured external model providers.")
    os.environ[variable] = key

# %%
from anonymizer import (
    AnonymizerConfig,
    AnonymizerInput,
    LoggingConfig,
    PrivacyGoal,
    Rewrite,
    configure_logging,
)
from anonymizer.notebooks import create_anonymizer, stop_local_runtime

configure_logging(LoggingConfig.default())

# %%
anonymizer = create_anonymizer()

# %% [markdown]
# ## 📦 Input data
#
# - Same biographies dataset used in earlier notebooks -- familiar data makes it
#   easy to compare rewrite output against replace output.

# %%
input_data = AnonymizerInput(
    source="https://raw.githubusercontent.com/NVIDIA-NeMo/Anonymizer/refs/heads/main/docs/data/NVIDIA_synthetic_biographies.csv",
    text_column="biography",
    data_summary="Biographical profiles",
)

# %% [markdown]
# ## 🎛️ Configure
#
# - `PrivacyGoal` spells out what to **protect** and what to **preserve** --
#   this gives the rewriter clear, domain-specific guidance.
# - `risk_tolerance` (default `"low"`) and `max_repair_iterations` (default `3`)
#   control the automated quality gate --
#   see [Risk tolerance](../../concepts/rewrite/#risk-tolerance) for presets.

# %%
config = AnonymizerConfig(
    rewrite=Rewrite(
        privacy_goal=PrivacyGoal(
            protect="All direct identifiers and quasi-identifier combinations (names, locations, employers, dates)",
            preserve="Career trajectory, educational background, and professional accomplishments",
        ),
        risk_tolerance="low",
        max_repair_iterations=3,
    ),
)

# %% [markdown]
# ## 👁️ Preview
#
# - `preview()` runs on a small sample so you can iterate on privacy goals
#   and evaluation criteria before committing to a full run.

# %%
preview = anonymizer.preview(
    config=config,
    data=input_data,
    num_records=3,
)

preview.display_record(0)

# %%
preview.display_record(1)

# %% [markdown]
# > **How to interpret leakage:** Leakage is measured against the sensitivity
# > disposition. Details marked `leave_as_is` may remain without increasing
# > `leakage_mass`. If an output retains something you expected the privacy goal
# > to protect, inspect the Entity Disposition table.
#
# ## 🚀 Full run
#
# - `result.dataframe` has user-facing columns: rewritten text, scores, and the review flag.
# - `result.trace_dataframe` has every intermediate column for debugging.

# %%
result = anonymizer.run(config=config, data=input_data)

result.dataframe.head()

# %%
result.dataframe[["biography_rewritten", "utility_score", "leakage_mass", "needs_human_review"]].head()

# %%
result.trace_dataframe.columns.tolist()

# %% [markdown]
# ## 🚩 Filter by review flag
#
# - Records where automated metrics exceed thresholds are flagged for manual review.
# - `needs_human_review` is threshold-based, so a record can have small nonzero
#   leakage without being flagged.
# - Use this to prioritize human attention on the records that need it most.
# - See [Working with flagged records](../../concepts/rewrite/#working-with-flagged-records)
#   for guidance on diagnosing and resolving flagged records.

# %%
df = result.dataframe
flagged = df[df["needs_human_review"] == True]  # noqa: E712
print(f"{len(flagged)} of {len(df)} records flagged for human review")
flagged.head()

# %% [markdown]
# ## 🔬 Evaluate (optional)
#
# Call `evaluate()` to run LLM-as-judge scoring on the rewrite result — detection validity and three quality rubrics (privacy, quality, style).
# Evaluation makes additional LLM calls per record. For larger datasets, evaluate
# a preview first; this tutorial evaluates all 25 rows to demonstrate the complete workflow.
# This holistic judge is independent of pipeline leakage scoring, so their assessments may differ.
# See [Evaluation](../../concepts/evaluation/#rewrite-evaluation) for details.

# %%
evaluated = anonymizer.evaluate(result)

# %%
evaluated.display_record(0)

# %% [markdown]
# ## ⏭️ Next steps
#
# - **[⚖️ Rewriting Legal Documents](../05_rewriting_legal_documents/)** --
#   rewrite legal text with custom entity labels and domain-specific privacy goals.
# - **[📊 Evaluation](../../concepts/evaluation/#rewrite-evaluation)** --
#   learn about the detection validity and rewrite quality judges in detail.
# - **[🎯 Choosing a Replacement Strategy](../03_choosing_a_replacement_strategy/)** --
#   compare Redact, Annotate, Hash, and Substitute if you prefer token-level replacement.
# - **[🔍 Inspecting Detected Entities](../02_inspecting_detected_entities/)** --
#   debug what the detection pipeline found before rewriting.

# %%
stop_local_runtime()
