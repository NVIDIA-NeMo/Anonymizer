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
# # Rewrite Mode
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
from pathlib import Path

try:
    NOTEBOOK_SOURCE_DIR = Path(__file__).resolve().parent
except NameError:
    NOTEBOOK_SOURCE_DIR = Path.cwd().parent / "notebook_source"

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, LoggingConfig, Rewrite, configure_logging

configure_logging(LoggingConfig.debug())

# %%
anonymizer = Anonymizer()

# %% [markdown]
# ## Input data

# %%
input_data = AnonymizerInput(
    source=str(NOTEBOOK_SOURCE_DIR / "data" / "synth_bios_sample10.csv"),
    text_column="bio",
    data_summary="Biographical profiles",
)

# %% [markdown]
# ## Basic rewrite
#
# `Rewrite()` with no arguments uses sensible defaults:
# conservative risk tolerance, up to 2 repair iterations, and an
# auto-populated privacy goal.

# %%
config = AnonymizerConfig(rewrite=Rewrite())

preview = anonymizer.preview(
    config=config,
    data=input_data,
    num_records=3,
)

preview.display_record(0)

# %%
preview.display_record(1)

# %% [markdown]
# ## Inspect the results
#
# `result.dataframe` has user-facing columns only.
# `result.trace_dataframe` has every intermediate column for debugging.

# %%
result = anonymizer.run(config=config, data=input_data)

print(result)
result.dataframe.head()

# %%
result.dataframe[["bio_rewritten", "utility_score", "leakage_mass", "needs_human_review"]].head()

# %%
result.trace_dataframe.columns.tolist()

# %% [markdown]
# ## Custom privacy goal
#
# For domain-specific data you can spell out exactly what to protect
# and what to preserve.

# %%
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal, RiskTolerance

custom_config = AnonymizerConfig(
    rewrite=Rewrite(
        privacy_goal=PrivacyGoal(
            protect="All direct identifiers, quasi-identifier combinations, and medical record numbers",
            preserve="Clinical meaning, temporal relationships, and treatment outcomes",
        ),
        evaluation=EvaluationCriteria(
            risk_tolerance=RiskTolerance.strict,
            max_repair_iterations=3,
        ),
    ),
)

custom_preview = anonymizer.preview(
    config=custom_config,
    data=input_data,
    num_records=3,
)

custom_preview.display_record(0)

# %% [markdown]
# ## Filter by review flag
#
# Records where automated metrics exceed thresholds are flagged for manual review.

# %%
df = result.dataframe
flagged = df[df["needs_human_review"] == True]  # noqa: E712
print(f"{len(flagged)} of {len(df)} records flagged for human review")
flagged.head()
