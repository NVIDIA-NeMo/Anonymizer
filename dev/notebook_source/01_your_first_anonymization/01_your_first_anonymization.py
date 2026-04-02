
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
# # 🕵️ Your First Anonymization
#
# Detect sensitive entities and replace them with LLM-generated substitutes --
# the simplest end-to-end example of Anonymizer.
#
# #### 📚 What you'll learn
#
# - Load a CSV dataset and configure Anonymizer in a few lines
# - Preview anonymized results on a small sample before committing to a full run
# - Inspect entity detection and replacement with `display_record()`
# - Process the full dataset with `run()`
#
# > **Tip:** First time running notebooks? Start with
# > [setup instructions](https://nvidia-nemo.github.io/Anonymizer/latest/tutorials/).

# %% [markdown]
# ## ⚙️ Setup
#
# - Check if your `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com) is registered for model access.
# - Import the core Anonymizer classes: `Anonymizer`, `AnonymizerConfig`, `AnonymizerInput`, and `Substitute`.
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
from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Substitute

# %%
anonymizer = Anonymizer()

# %% [markdown]
# ## 📦 Load data and configure
#
# - `AnonymizerInput` points to your CSV and names the text column. `data_summary`
#   gives the LLM context about the kind of text it will process.
# - `AnonymizerConfig` with `Substitute()` tells Anonymizer to replace detected
#   entities with LLM-generated synthetic values (e.g. fake names, cities, dates).

# %%
input_data = AnonymizerInput(
    source="../data/NVIDIA_synthetic_biographies.csv",
    text_column="biography",
    data_summary="Biographical profiles of individuals",
)

config = AnonymizerConfig(replace=Substitute())

# %% [markdown]
# ## 👁️ Preview
#
# - `preview()` runs on a small sample so you can iterate quickly.
# - Always preview before processing the full dataset -- it's the fastest way
#   to catch prompt or config issues early.

# %%
preview = anonymizer.preview(config=config, data=input_data, num_records=3)

# %% [markdown]
# ## 🔍 Inspect
#
# - `display_record()` shows the original text with highlighted entities,
#   the replacement map, and the anonymized output -- all in one view.
# - The result dataframe has original and substituted text side-by-side.

# %%
preview.display_record(0)

# %%
preview.display_record(1)

# %%
preview.dataframe

# %% [markdown]
# ## 🚀 Full run
#
# - `run()` processes the entire dataset with the same config you previewed.
# - Access the output via `result.dataframe`.

# %%
result = anonymizer.run(config=config, data=input_data)
print(result)

# %%
result.dataframe.head()

# %% [markdown]
# ## ⏭️ Next steps
#
# - **[🔍 Inspecting Detected Entities](02_inspecting_detected_entities.ipynb)** --
#   dig into what the detection pipeline found and debug quality.
# - **[🎯 Choosing a Replacement Strategy](03_choosing_a_replacement_strategy.ipynb)** --
#   compare Redact, Annotate, Hash, and Substitute side-by-side.
# - **[✏️ Rewriting Biographies](04_rewriting_biographies.ipynb)** --
#   generate privacy-safe paraphrases instead of token-level replacements.
