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
# # 🕵️ Rewriting Legal Documents
#
# Rewriting legal text (TAB dataset) with a domain-specific privacy goal
# and custom entity labels tailored for legal proceedings.
#
# #### 📚 What you'll learn
#
# - Define domain-specific entity labels for legal text (case numbers, court names, etc.)
# - Configure rewrite mode with legal-specific privacy goals
# - Preview and run on court decision documents
# - Triage flagged records with `needs_human_review`
#
# > **Tip:** First time running notebooks? Start with
# > [setup instructions](https://nvidia-nemo.github.io/Anonymizer/latest/tutorials/).

# %% [markdown]
# ## ⚙️ Setup
#
# - Install the notebook extra, then provide credentials for the configured external LLM providers.
# - `create_anonymizer()` starts pinned GLiNER2 locally and selects CUDA, MPS, or CPU automatically.
# - The default external LLM models currently use [NVIDIA Build](https://build.nvidia.com); its terms and privacy practices apply.
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
subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])

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
    Detect,
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
# - [TAB (Text Anonymization Benchmark)](https://github.com/NorskRegnesentral/text-anonymization-benchmark)
#   legal documents -- court decisions containing names, dates, case numbers, and other legal identifiers.
# - `LEGAL_ENTITY_LABELS` defines the domain-specific entity types to detect.
#   This replaces the default label set with one tailored to legal text.

# %%
LEGAL_ENTITY_LABELS = [
    "first_name",
    "last_name",
    "court_name",
    "organization_name",
    "company_name",
    "prison_detention_facility",
    "street_address",
    "city",
    "state",
    "country",
    "date",
    "date_time",
    "time",
    "date_of_birth",
    "age",
    "email",
    "phone_number",
    "ssn",
    "unique_id",
    "legal_role",
    "case_number",
    "application_number",
    "monetary_amount",
    "sentence_duration",
    "nationality",
]

input_data = AnonymizerInput(
    source="https://raw.githubusercontent.com/NVIDIA-NeMo/Anonymizer/refs/heads/main/docs/data/TAB_legal_sample25.csv",
    text_column="text",
    data_summary="Legal court decisions containing personal identifiers, case numbers, and institutional references",
)

# %% [markdown]
# ## 🎛️ Configure
#
# - `Detect(entity_labels=...)` overrides the default entity set with legal-specific labels.
#   The explicit list is a strict allowlist for both detection and LLM augmentation:
#   labels not included here are filtered out, so include every entity type you need.
# - `PrivacyGoal` tells the rewriter what to **protect** (identifiers, case numbers,
#   institutional references) and what to **preserve** (legal reasoning, statutory references,
#   ruling structure).

# %%
config = AnonymizerConfig(
    detect=Detect(
        entity_labels=LEGAL_ENTITY_LABELS,
    ),
    rewrite=Rewrite(
        privacy_goal=PrivacyGoal(
            protect="All personal identifiers, case numbers, court names, and institutional references that could identify parties",
            preserve="Legal reasoning, procedural facts, statutory references, and the structure of the ruling",
        ),
        risk_tolerance="minimal",
        max_repair_iterations=3,
    ),
)

# %% [markdown]
# ## 👁️ Preview
#
# - Preview on a few records to check that legal entities are detected
#   and the rewrite preserves the ruling's structure.

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
# - This notebook uses `risk_tolerance="minimal"`, which applies stricter repair
#   and review thresholds than notebook 04.

# %%
result = anonymizer.run(config=config, data=input_data)

result.dataframe.head()

# %%
result.dataframe[["text_rewritten", "utility_score", "leakage_mass", "needs_human_review"]].head()

# %% [markdown]
# ## 🚩 Filter by review flag
#
# - Records where automated metrics exceed thresholds are flagged for manual review.
# - The repair loop stops after `max_repair_iterations`; records that still need
#   repair remain flagged for human review but are not pipeline failures.
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
# - **[📊 Evaluation](../../concepts/evaluation/#rewrite-evaluation)** --
#   learn about the detection validity and rewrite quality judges in detail.
# - **[🔍 Inspecting Detected Entities](../02_inspecting_detected_entities/)** --
#   debug what the detection pipeline found before rewriting.
# - **Try it on your own data!** Swap in your CSV, define entity labels for your
#   domain, and set a `PrivacyGoal` that fits -- you've got all the building blocks.

# %%
stop_local_runtime()
