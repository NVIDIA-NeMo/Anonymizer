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
# # 🕵️ Inspecting Detected Entities
#
# Dig into the entity detection pipeline output -- what was detected,
# what the LLM validator kept or dropped, and where entities appear in the text.
#
# This notebook is for users who need to debug detection quality,
# tune labels and/or thresholds, or investigate downstream replacement or rewriting results.

# We use **Annotate** mode because it preserves the original text while tagging each entity
# with its label, making it ideal for reviewing detection quality.
#
# #### 📚 What you'll learn
#
# - Run the detection pipeline and inspect its output using Annotate mode
# - View tagged text with entities marked inline
# - Break down detected entities by label, source, and unique value
# - Identify and triage failed records
#
# > **Tip:** First time running notebooks? Start with
# > [setup instructions](https://nvidia-nemo.github.io/Anonymizer/latest/tutorials/).

# %% [markdown]
# ## ⚙️ Setup
#
# - Check if your `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com) is registered for model access.
#   - Treat the default `build.nvidia.com` setup as a convenient experimentation path. For privacy-sensitive or production data, switch to a secure endpoint you trust and to which you are comfortable sending data.
#   - Request/token rate limits on `build.nvidia.com` vary by account and model access, and lower-volume development access can be slow for full runs. Start with `preview()` on a small sample.
# - Import the core classes -- this notebook uses `Annotate` to keep original values visible.
# - `Anonymizer()` initializes with the default model provider -- no extra config needed.
# - `Anonymizer.configure_logging()` controls verbosity -- switch to `Anonymizer.configure_logging(LoggingConfig.debug())` when troubleshooting.

# %%
import getpass
import os
from collections import Counter

import pandas as pd

if not os.getenv("NVIDIA_API_KEY"):
    key = getpass.getpass("Enter NVIDIA_API_KEY from build.nvidia.com: ").strip()
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is required to run these notebooks.")
    os.environ["NVIDIA_API_KEY"] = key

# %%
from anonymizer import Annotate, Anonymizer, AnonymizerConfig, AnonymizerInput

# %%
anonymizer = Anonymizer()

# %% [markdown]
# ## 👁️ Preview
#
# - Detection runs as part of any strategy. `Annotate` keeps original text visible
#   alongside entity labels -- ideal for debugging.
# - `trace_dataframe` exposes every internal pipeline column; that's what we explore below.

# %%
config = AnonymizerConfig(replace=Annotate())

input_data = AnonymizerInput(
    source="https://raw.githubusercontent.com/NVIDIA-NeMo/Anonymizer/refs/heads/main/docs/data/NVIDIA_synthetic_biographies.csv",
    text_column="biography",
    data_summary="Biographical profiles",
)

result = anonymizer.preview(
    config=config,
    data=input_data,
    num_records=3,
)

# %% [markdown]
# ## 🔍 Inspect
#
# - `display_record()` renders an interactive view with entity highlights.

# %%
result.display_record(0)

# %% [markdown]
# ## 📋 Columns
#
# - `trace_dataframe` contains all internal columns from the pipeline
#   (detection, validation, replacement, etc.).

# %%
df = result.trace_dataframe
print(f"Records: {len(df)}")
print(f"Columns: {list(df.columns)}")

# %% [markdown]
# ## 🎯 Detected entities
#
# - Final entity list after validation. Each entity has `value`, `label`,
#   positions, `score`, and `source` (detector / augmenter / name_split / propagation).

# %%
row_idx = 0
raw = df.loc[row_idx, "_detected_entities"]
entities = raw["entities"] if isinstance(raw, dict) else raw
print(f"Record {row_idx}: {len(entities)} entities detected\n")

entity_df = pd.DataFrame(entities)
if not entity_df.empty:
    cols = [c for c in ["value", "label", "start_position", "end_position", "source"] if c in entity_df.columns]
    print(entity_df[cols].to_string())

# %% [markdown]
# ## 🏷️ Labels
#
# - Entity label distribution across all records -- which types are most common.

# %%
label_counts = Counter()
for raw in df["_detected_entities"]:
    entity_list = raw["entities"] if isinstance(raw, dict) else raw
    for entity in entity_list:
        label_counts[entity["label"]] += 1

for label, count in label_counts.most_common():
    print(f"  {label}: {count}")

# %% [markdown]
# ## 📡 Sources
#
# - Where each entity came from in the pipeline:
#     - `detector` -- GLiNER NER
#     - `augmenter` -- LLM-added (missed by GLiNER)
#     - `validator` -- LLM decision step over detector-seed entities (keep/reclass/drop); does not emit a separate source value
#     - `name_split` -- derived from splitting full names
#     - `propagation` -- expanded from validated entities to all text occurrences

# %%
source_counts = Counter()
for raw in df["_detected_entities"]:
    entity_list = raw["entities"] if isinstance(raw, dict) else raw
    for entity in entity_list:
        source_counts[entity.get("source", "unknown")] += 1

for source, count in source_counts.most_common():
    print(f"  {source}: {count}")

# %% [markdown]
# ## 📊 By value
#
# - Entities grouped by unique value -- this is what drives consistent replacement
#   downstream (same name always maps to the same substitute).

# %%
row_idx = 0
raw_bv = df.loc[row_idx, "_entities_by_value"]
by_value = raw_bv["entities_by_value"] if isinstance(raw_bv, dict) else raw_bv
print(f"Record {row_idx}: {len(by_value)} unique entity values\n")

for entry in by_value:
    print(f"  {entry['value']!r} -> labels: {entry['labels']}")

# %% [markdown]
# ## ❌ Failures
#
# - Records dropped during detection (LLM timeout, parse error, etc.).
# - Check this to understand data loss in your pipeline.

# %%
if result.failed_records:
    for fr in result.failed_records:
        print(f"  record_id={fr.record_id}, step={fr.step}, reason={fr.reason}")
else:
    print("No failed records.")

# %% [markdown]
# ## ⏭️ Next steps
#
# - **[🕵️ Your First Anonymization](../01_your_first_anonymization/)** --
#   the simplest end-to-end replace workflow if you haven't run it yet.
# - **[🎯 Choosing a Replacement Strategy](../03_choosing_a_replacement_strategy/)** --
#   compare Redact, Annotate, Hash, and Substitute side-by-side.
# - **[✏️ Rewriting Biographies](../04_rewriting_biographies/)** --
#   generate privacy-safe paraphrases instead of token-level replacements.
