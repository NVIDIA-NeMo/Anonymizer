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
# # Inspecting Detected Entities
#
# Inspect the entity detection pipeline output. Useful for understanding what was detected,
# what the LLM validator kept or dropped, and where entities appear in the text.
#
# We use **Annotate** mode here -- it preserves the original text while tagging each entity
# with its label, making it ideal for reviewing detection quality.

# %% [markdown]
# ## Setup

# %%
from collections import Counter

import pandas as pd

from anonymizer import Annotate, Anonymizer, AnonymizerConfig, AnonymizerInput

# %%
anonymizer = Anonymizer()

# %% [markdown]
# ## Run
#
# Detection runs as part of any strategy. Annotate is a good fit for debugging because
# the original values stay visible alongside their labels.

# %%
config = AnonymizerConfig(replace=Annotate())

input_data = AnonymizerInput(
    source="../data/NVIDIA_synthetic_biographies.csv",
    text_column="biography",
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
# Original text with entities marked inline.

# %%
for i in range(len(df)):
    print(f"--- Record {i} ---")
    print(df.loc[i, "biography_with_spans"][:1000])
    print()

# %% [markdown]
# ## Detected entities
#
# Final entity list after validation. Each has `value`, `label`, positions, `score`,
# and `source` (detector / augmenter / name_split / propagation).

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
# ## Labels

# %%
label_counts = Counter()
for raw in df["_detected_entities"]:
    entity_list = raw["entities"] if isinstance(raw, dict) else raw
    for entity in entity_list:
        label_counts[entity["label"]] += 1

for label, count in label_counts.most_common():
    print(f"  {label}: {count}")

# %% [markdown]
# ## Sources
#
# - `detector` -- GLiNER NER
# - `augmenter` -- LLM-added (missed by GLiNER)
# - `validator` -— LLM decision step over detector-seed entities (keep/reclass/drop); it does not emit a separate source value.
# - `name_split` -- derived from splitting full names
# - `propagation` -- expanded from validated entities to all text occurrences

# %%
source_counts = Counter()
for raw in df["_detected_entities"]:
    entity_list = raw["entities"] if isinstance(raw, dict) else raw
    for entity in entity_list:
        source_counts[entity.get("source", "unknown")] += 1

for source, count in source_counts.most_common():
    print(f"  {source}: {count}")

# %% [markdown]
# ## By value
#
# Entities grouped by unique value -- this drives consistent replacement.

# %%
row_idx = 0
raw_bv = df.loc[row_idx, "_entities_by_value"]
by_value = raw_bv["entities_by_value"] if isinstance(raw_bv, dict) else raw_bv
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
