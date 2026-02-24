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
import tempfile
from collections import Counter
from pathlib import Path

try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:
    NOTEBOOK_DIR = Path.cwd()

import pandas as pd

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, RedactReplace

# %%
MODEL_PROVIDERS_YAML = """
providers:
  - name: nvidia
    endpoint: https://integrate.api.nvidia.com/v1
    provider_type: openai
    api_key: NIM_API_KEY

  - name: nvidia-pii
    endpoint: https://gliner-xx5zzfj46.brevlab.com/v1
    provider_type: openai
    api_key: NIM_API_KEY
"""

MODEL_CONFIGS_YAML = """
model_configs:
  - alias: gliner-pii-detector
    model: nvidia/nemotron-pii
    provider: nvidia-pii

  - alias: gpt-oss-120b
    model: nvdev/openai/gpt-oss-120b
    provider: nvidia
    inference_parameters:
      max_parallel_requests: 8
      temperature: 0.3
      top_p: 0.95
      max_tokens: 10000
      timeout: 180

  - alias: nemotron-30b-thinking
    model: nvidia/nemotron-3-nano-30b-a3b
    provider: nvidia
    inference_parameters:
      max_parallel_requests: 8
      temperature: 0.4
      top_p: 1.0
      max_tokens: 4096
      timeout: 180
"""

tmp_dir = Path(tempfile.mkdtemp(prefix="anonymizer_notebook_"))
providers_path = tmp_dir / "model_providers.yaml"

providers_path.write_text(MODEL_PROVIDERS_YAML.strip() + "\n", encoding="utf-8")

anonymizer = Anonymizer(model_configs=MODEL_CONFIGS_YAML, model_providers=providers_path)

# %% [markdown]
# ## Run
#
# Detection runs as part of any strategy. We use `RedactReplace` here since we only
# care about the detection columns. Set `data_summary` on the input to improve augmenter/validator accuracy.

# %%
config = AnonymizerConfig(replace=RedactReplace())

input_data = AnonymizerInput(
    source=str(NOTEBOOK_DIR / "data" / "synth_bios_sample10.csv"),
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
