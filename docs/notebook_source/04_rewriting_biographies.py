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
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# %% [markdown]
# # 🕵️ Rewriting Biographies
#
# Instead of replacing entities with tokens, rewrite mode generates a
# privacy-safe paraphrase of the entire text. The pipeline:
#
# 1. Detects entities (same as replace mode)
# 2. Classifies the domain and assigns sensitivity dispositions
# 3. Generates a rewritten version that obscures sensitive entities
# 4. Evaluates quality (utility) and privacy (leakage) with an automated repair loop
# 5. Runs a final LLM judge for informational scores
#
# #### 📚 What you'll learn
#
# - Configure rewrite mode with `PrivacyGoal` to specify what to protect and what to preserve
# - Set evaluation criteria and risk tolerance for automated quality checks
# - Preview rewritten text and inspect utility / leakage scores
# - Triage flagged records with `needs_human_review`
#
# > **Tip:** First time running notebooks? Start with
# > [setup instructions](https://nvidia-nemo.github.io/Anonymizer/latest/tutorials/).

# %% [markdown]
# ## ⚙️ Setup
#
# - Check if your `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com) is registered for model access.
# - Import `Rewrite` and its config classes: `PrivacyGoal`, `EvaluationCriteria`, `RiskTolerance`.
# - `Anonymizer()` initializes with the default model provider -- no extra config needed.

# %%
import getpass
import os

if not os.getenv("NVIDIA_API_KEY"):
    key = getpass.getpass("Enter NVIDIA_API_KEY from build.nvidia.com: ").strip()
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is required to run these notebooks.")
    os.environ["NVIDIA_API_KEY"] = key

# %%
from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal, RiskTolerance

# %%
anonymizer = Anonymizer()

# %% [markdown]
# ## 📦 Input data
#
# - Same biographies dataset used in earlier notebooks -- familiar data makes it
#   easy to compare rewrite output against replace output.

# %%
input_data = AnonymizerInput(
    source="../data/NVIDIA_synthetic_biographies.csv",
    text_column="biography",
    data_summary="Biographical profiles",
)

# %% [markdown]
# ## 🎛️ Configure
#
# - `PrivacyGoal` spells out what to **protect** and what to **preserve** --
#   this gives the rewriter clear, domain-specific guidance.
# - `EvaluationCriteria` controls the automated quality gate: `risk_tolerance`
#   sets the leakage threshold and `max_repair_iterations` caps how many times
#   the rewriter retries when evaluation fails.

# %%
config = AnonymizerConfig(
    rewrite=Rewrite(
        privacy_goal=PrivacyGoal(
            protect="All direct identifiers and quasi-identifier combinations (names, locations, employers, dates)",
            preserve="Career trajectory, educational background, and professional accomplishments",
        ),
        evaluation=EvaluationCriteria(
            risk_tolerance=RiskTolerance.strict,
            max_repair_iterations=3,
        ),
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
# - Use this to prioritize human attention on the records that need it most.

# %%
df = result.dataframe
flagged = df[df["needs_human_review"] == True]  # noqa: E712
print(f"{len(flagged)} of {len(df)} records flagged for human review")
flagged.head()

# %% [markdown]
# ## ⏭️ Next steps
#
# - **[⚖️ Rewriting Legal Documents](05_rewriting_legal_documents.ipynb)** --
#   rewrite legal text with custom entity labels and domain-specific privacy goals.
# - **[🎯 Choosing a Replacement Strategy](03_choosing_a_replacement_strategy.ipynb)** --
#   compare Redact, Annotate, Hash, and Substitute if you prefer token-level replacement.
# - **[🔍 Inspecting Detected Entities](02_inspecting_detected_entities.ipynb)** --
#   debug what the detection pipeline found before rewriting.
