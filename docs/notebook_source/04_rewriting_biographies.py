# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
# # Rewriting Biographies
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
# The result includes **utility_score**, **leakage_mass**, and a
# **needs_human_review** flag so you can triage records that need attention.

# %% [markdown]
# ## Setup

# %%
from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, LoggingConfig, Rewrite, configure_logging
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal, RiskTolerance

configure_logging(LoggingConfig.debug())

# %%
anonymizer = Anonymizer()

# %% [markdown]
# ## Input data

# %%
input_data = AnonymizerInput(
    source="../data/NVIDIA_synthetic_biographies.csv",
    text_column="biography",
    data_summary="Biographical profiles",
)

# %% [markdown]
# ## Configure
#
# Spell out what to protect and what to preserve. This gives the rewriter
# clear guidance for your domain.

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
# ## Preview

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
# ## Full run
#
# `result.dataframe` has user-facing columns only.
# `result.trace_dataframe` has every intermediate column for debugging.

# %%
result = anonymizer.run(config=config, data=input_data)

print(result)
result.dataframe.head()

# %%
result.dataframe[["biography_rewritten", "utility_score", "leakage_mass", "needs_human_review"]].head()

# %%
result.trace_dataframe.columns.tolist()

# %% [markdown]
# ## Filter by review flag
#
# Records where automated metrics exceed thresholds are flagged for manual review.

# %%
df = result.dataframe
flagged = df[df["needs_human_review"] == True]  # noqa: E712
print(f"{len(flagged)} of {len(df)} records flagged for human review")
flagged.head()
