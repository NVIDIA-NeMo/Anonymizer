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
from anonymizer import AnonymizerConfig, AnonymizerInput, LoggingConfig, Substitute, configure_logging
from anonymizer.notebooks import create_anonymizer, stop_local_runtime

configure_logging(LoggingConfig.default())

# %%
anonymizer = create_anonymizer()

# %% [markdown]
# ## 📦 Load data and configure
#
# - `AnonymizerInput` points to your CSV and names the text column. `data_summary`
#   gives the LLM context about the kind of text it will process.
# - Records up to 2,000 tokens each work with the default model configs.
# - `AnonymizerConfig` with `Substitute()` tells Anonymizer to replace detected
#   entities with LLM-generated synthetic values for names, cities, dates, etc.

# %%
input_data = AnonymizerInput(
    source="https://raw.githubusercontent.com/NVIDIA-NeMo/Anonymizer/refs/heads/main/docs/data/NVIDIA_synthetic_biographies.csv",
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
# ## 📊 (Optional) Evaluate replacement quality
#
# - `evaluate()` is a separate, opt-in step that scores the output with LLM-as-judge metrics.
# - For Substitute, all four metrics run: **Detection Validity**, **Type Fidelity**, **Relational Consistency**, **Attribute Fidelity**.
# - Skip it for routine runs; call it when you want LLM-side confidence on the output. Costs LLM calls per record, so try it on `preview` first.

# %%
evaluated = anonymizer.evaluate(preview)
evaluated.display_record(0)

# %% [markdown]
# ## ⏭️ Next steps
#
# - **[🔍 Inspecting Detected Entities](../02_inspecting_detected_entities/)** --
#   dig into what the detection pipeline found and debug quality.
# - **[🎯 Choosing a Replacement Strategy](../03_choosing_a_replacement_strategy/)** --
#   compare Redact, Annotate, Hash, and Substitute side-by-side.
# - **[✏️ Rewriting Biographies](../04_rewriting_biographies/)** --
#   generate privacy-safe paraphrases instead of token-level replacements.

# %%
stop_local_runtime()
