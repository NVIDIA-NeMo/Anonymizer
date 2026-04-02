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
# # Detection Debug
#
# Inspect the entity detection pipeline output. Useful for understanding what was detected,
# what the LLM validator kept/dropped, and where entities appear in the text.
#
# Pipeline: NER detection → parse → LLM augmentation → merge → LLM validation → propagate.

# %% [markdown]
# ## Setup

# %%
from collections import Counter
from pathlib import Path

try:
    NOTEBOOK_SOURCE_DIR = Path(__file__).resolve().parent
except NameError:
    # Running as .ipynb — cwd is docs/notebooks/, data lives in docs/notebook_source/
    NOTEBOOK_SOURCE_DIR = Path.cwd().parent / "notebook_source"

import pandas as pd

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Redact

# %%
anonymizer = Anonymizer()

# %% [markdown]
# ## Run
#
# Detection runs as part of any strategy. We use `Redact` here since we only
# care about the detection columns. Set `data_summary` on the input to improve augmenter/validator accuracy.

# %%
config = AnonymizerConfig(replace=Redact())

input_data = AnonymizerInput(
    source=str(NOTEBOOK_SOURCE_DIR / "data" / "synth_bios_sample10.csv"),
    text_column="bio",
    data_summary="Biographical profiles",
)

result = anonymizer.preview(
    config=config,
    data=input_data,
    num_records=3,
)

# %% [markdown]
# ## Visual preview

# %%
result.display_record(0)

# %% [markdown]
# ## Columns

# %%
df = result.trace_dataframe
print(f"Records: {len(df)}")
print(f"Columns: {list(df.columns)}")

# %% [markdown]
# ## Tagged text
#
# Original text with entities marked inline. Notation adapts to avoid conflicts with existing markup.

# %%
for i in range(len(df)):
    print(f"--- Record {i} ---")
    print(df.loc[i, "bio_with_spans"][:1000])
    print()

# %% [markdown]
# ## Detected entities
#
# Final entity list after validation. Each has `value`, `label`, positions, `score`,
# and `source` (detector / augmenter / name_split / propagation).

# %%


row_idx = 0
entities = df.loc[row_idx, "_detected_entities"]
print(f"Record {row_idx}: {len(entities)} entities detected\n")

entity_df = pd.DataFrame(entities)
if not entity_df.empty:
    print(entity_df[["value", "label", "start_position", "end_position", "source"]].to_string())

# %% [markdown]
# ## Labels

# %%


label_counts = Counter()
for entities in df["_detected_entities"]:
    for entity in entities:
        label_counts[entity["label"]] += 1

for label, count in label_counts.most_common():
    print(f"  {label}: {count}")

# %% [markdown]
# ## Sources
#
# - `detector` — GLiNER NER
# - `augmenter` — LLM-added (missed by GLiNER)
# - `name_split` — derived from splitting full names
# - `propagation` — expanded from validated entities to all text occurrences

# %%
source_counts = Counter()
for entities in df["_detected_entities"]:
    for entity in entities:
        source_counts[entity.get("source", "unknown")] += 1

for source, count in source_counts.most_common():
    print(f"  {source}: {count}")

# %% [markdown]
# ## By value
#
# Entities grouped by unique value — this drives consistent replacement.

# %%
row_idx = 0
by_value = df.loc[row_idx, "_entities_by_value"]
print(f"Record {row_idx}: {len(by_value)} unique entity values\n")

for entry in by_value:
    print(f"  {entry['value']!r} -> labels: {entry['labels']}")

# %% [markdown]
# ## Failures
#
# Records dropped during detection (LLM timeout, parse error, etc.).

# %%
if result.failed_records:
    for fr in result.failed_records:
        print(f"  record_id={fr.record_id}, step={fr.step}, reason={fr.reason}")
else:
    print("No failed records.")
