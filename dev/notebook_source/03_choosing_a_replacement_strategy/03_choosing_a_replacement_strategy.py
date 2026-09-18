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
# # 🕵️ Choosing a Replacement Strategy
#
# Four [replace mode](../../concepts/replace/) strategies compared side-by-side on the same data.
#
# | Strategy | What it does |
# |----------|-------------|
# | **Substitute** | LLM-generated contextual replacements |
# | **Redact** | Label-based markers (`[REDACTED_FIRST_NAME]`) |
# | **Annotate** | Tags entities but keeps original text |
# | **Hash** | Deterministic hash digest |
#
# #### 📚 What you'll learn
#
# - Compare **Redact**, **Annotate**, **Hash**, and **Substitute** on the same input
# - Customize output formats with `format_template`
# - Understand which strategy fits your use case (readability, determinism, privacy)
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
    Annotate,
    AnonymizerConfig,
    AnonymizerInput,
    Hash,
    LoggingConfig,
    Redact,
    Substitute,
    configure_logging,
)
from anonymizer.notebooks import create_anonymizer, stop_local_runtime

configure_logging(LoggingConfig.default())
# %%
anonymizer = create_anonymizer()

# %% [markdown]
# ## 📦 Input data
#
# - We use the same biographies dataset throughout so each strategy is compared
#   on identical input.

# %%
input_data = AnonymizerInput(
    source="https://raw.githubusercontent.com/NVIDIA-NeMo/Anonymizer/refs/heads/main/docs/data/NVIDIA_synthetic_biographies.csv",
    text_column="biography",
    data_summary="Biographical profiles",
)

# %% [markdown]
# ## 🔄 Substitute
#
# - Uses an LLM to generate contextually appropriate synthetic replacements.
#   - The LLM considers the full document context matching names with emails, cities to states, etc.
# - Customize with `instructions` to steer the LLM's replacement choices.

# %%
substitute_config = AnonymizerConfig(replace=Substitute())

substitute_preview = anonymizer.preview(
    config=substitute_config,
    data=input_data,
    num_records=3,
)

# %%
substitute_preview.display_record(0)

# %% [markdown]
# ### Custom instructions
#
# - Pass `instructions` to guide the LLM -- e.g. keep replacements within
#   a specific region, culture, or naming convention.

# %%
substitute_custom_config = AnonymizerConfig(
    replace=Substitute(instructions="Use only Japanese names and locations for all replacements.")
)
substitute_custom_preview = anonymizer.preview(
    config=substitute_custom_config,
    data=input_data,
    num_records=3,
)
substitute_custom_preview.display_record(0)
# %% [markdown]
# ## 🚫 Redact
#
# - Replaces each entity with a label-based marker. Default: `[REDACTED_FIRST_NAME]`.
# - Customize with `Redact(format_template=...)`.

# %%
redact_config = AnonymizerConfig(replace=Redact())

redact_preview = anonymizer.preview(
    config=redact_config,
    data=input_data,
    num_records=3,
)

redact_preview.display_record(0)

# %% [markdown]
# ### Custom template
#
# - `format_template="***"` replaces every entity with the same constant.

# %%
custom_config = AnonymizerConfig(replace=Redact(format_template="***"))

custom_preview = anonymizer.preview(
    config=custom_config,
    data=input_data,
    num_records=3,
)

custom_preview.display_record(0)

# %% [markdown]
# ## 🏷️ Annotate
#
# - Tags each entity with its label but keeps the original text visible.
#   Default: `<Alice, first_name>`.
# - Customize with `format_template` -- must include `{text}` and `{label}`,
#   e.g. `Annotate(format_template="<{text}-|-{label}>")`.

# %%
annotate_config = AnonymizerConfig(replace=Annotate())

annotate_preview = anonymizer.preview(
    config=annotate_config,
    data=input_data,
    num_records=3,
)

annotate_preview.display_record(0)

# %% [markdown]
# ### Custom template
#
# - Override the default format with any string containing `{text}` and `{label}`.

# %%
annotate_custom_config = AnonymizerConfig(replace=Annotate(format_template="<{text}-|-{label}>"))
annotate_custom_preview = anonymizer.preview(
    config=annotate_custom_config,
    data=input_data,
    num_records=3,
)
annotate_custom_preview.display_record(0)

# %% [markdown]
# ## #️⃣ Hash
#
# - Deterministic -- same input always produces the same hash.
# - Customize with `format_template` (must include `{digest}`),
#   `algorithm` (`sha256`/`sha1`/`md5`), and `digest_length` (6-64 characters).

# %%
hash_config = AnonymizerConfig(replace=Hash())

hash_preview = anonymizer.preview(
    config=hash_config,
    data=input_data,
    num_records=3,
)

hash_preview.display_record(0)

# %% [markdown]
# ### Custom template
#
# - Override the algorithm, digest length, and output format.

# %%
hash_custom_config = AnonymizerConfig(replace=Hash(algorithm="md5", digest_length=8, format_template="[{digest}]"))
hash_custom_preview = anonymizer.preview(
    config=hash_custom_config,
    data=input_data,
    num_records=3,
)
hash_custom_preview.display_record(0)


# %% [markdown]
# ## 📊 (Optional) Evaluate each strategy
#
# - `evaluate()` is a separate, opt-in step that scores the output with LLM-as-judge metrics. Which metrics fire depends on the strategy:
#   - **Substitute** → 4 metrics (Detection Validity + Type Fidelity + Relational Consistency + Attribute Fidelity).
#   - **Redact / Annotate / Hash** → Detection Validity only (no replacement map to score type/relational/attribute against).
# - Below shows it on the Substitute preview to surface all four; the same call works on `redact_preview`, `annotate_preview`, or `hash_preview`.

# %%
substitute_evaluated = anonymizer.evaluate(substitute_preview)
substitute_evaluated.display_record(0)


# %% [markdown]
# ## ⏭️ Next steps
#
# - **[🕵️ Inspecting Detected Entities](../02_inspecting_detected_entities/)** --
#   dig into what the detection pipeline found and debug quality.
# - **[✏️ Rewriting Biographies](../04_rewriting_biographies/)** --
#   generate privacy-safe paraphrases instead of token-level replacements.
# - **[⚖️ Rewriting Legal Documents](../05_rewriting_legal_documents/)** --
#   rewrite legal text with domain-specific privacy goals.

# %%
stop_local_runtime()
