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
# # 🕵️ Choosing a Replacement Strategy
#
# Four [replace mode](../concepts/replace.md) strategies compared side-by-side on the same data.
#
# | Strategy | What it does |
# |----------|-------------|
# | **Substitute** | LLM-generated contextual replacements |
# | **Redact** | Label-based markers (`[REDACTED_FIRST_NAME]`) |
# | **Annotate** | Tags entities but keeps original text |
# | **Hash** | Deterministic hash digest |
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
# - Check if your `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com) is registered for model access.
# - Import all four strategy classes: `Redact`, `Annotate`, `Hash`, `Substitute`.
# - `Anonymizer()` initializes with the default model provider -- no extra config needed.
# - `Anonymizer.configure_logging()` controls verbosity -- switch to `Anonymizer.configure_logging(LoggingConfig.debug())` when troubleshooting.

# %%
import getpass
import os

if not os.getenv("NVIDIA_API_KEY"):
    key = getpass.getpass("Enter NVIDIA_API_KEY from build.nvidia.com: ").strip()
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is required to run these notebooks.")
    os.environ["NVIDIA_API_KEY"] = key

# %%
from anonymizer import Annotate, Anonymizer, AnonymizerConfig, AnonymizerInput, Hash, Redact, Substitute

# %%
anonymizer = Anonymizer()

# %% [markdown]
# ## 📦 Input data
#
# - We use the same biographies dataset throughout so each strategy is compared
#   on identical input.

# %%
input_data = AnonymizerInput(
    source="../data/NVIDIA_synthetic_biographies.csv",
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
# ## ⏭️ Next steps
#
# - **[🕵️ Inspecting Detected Entities](02_inspecting_detected_entities.ipynb)** --
#   dig into what the detection pipeline found and debug quality.
# - **[✏️ Rewriting Biographies](04_rewriting_biographies.ipynb)** --
#   generate privacy-safe paraphrases instead of token-level replacements.
# - **[⚖️ Rewriting Legal Documents](05_rewriting_legal_documents.ipynb)** --
#   rewrite legal text with domain-specific privacy goals.
