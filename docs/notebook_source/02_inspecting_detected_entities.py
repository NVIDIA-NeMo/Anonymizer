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
# > **Privacy warning:** `Annotate` does not anonymize the text. Sensitive values
# > remain in the output, so use it only for inspection -- not as a privacy-safe
# > production strategy.
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
from collections import Counter

package_spec = os.getenv("ANONYMIZER_NOTEBOOK_PACKAGE", "nemo-anonymizer[notebooks]")
subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])

# %%
import pandas as pd

# %%
from anonymizer.notebooks._model_config import required_api_key_environment_variables

for variable in required_api_key_environment_variables():
    key = getpass.getpass(f"Enter {variable}: ").strip()
    if not key:
        raise RuntimeError(f"{variable} is required by the configured external model providers.")
    os.environ[variable] = key

# %%
from anonymizer import Annotate, AnonymizerConfig, AnonymizerInput, LoggingConfig, configure_logging
from anonymizer.notebooks import create_anonymizer, stop_local_runtime

configure_logging(LoggingConfig.default())

# %%
anonymizer = create_anonymizer()

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
# - `result.dataframe["final_entities"]` is the stable, public entity output.
# - `trace_dataframe` contains internal pipeline columns for deeper debugging;
#   those underscore-prefixed columns may change between releases.

# %%
trace_df = result.trace_dataframe
final_entities = result.dataframe["final_entities"]
print(f"Records: {len(trace_df)}")
print(f"Columns: {list(trace_df.columns)}")

# %% [markdown]
# ## 🎯 Detected entities
#
# - Final entity list after validation. Each entity has `value`, `label`,
#   positions, `score`, and `source` (detector / augmenter / name_split / propagation).

# %%
row_idx = 0
raw = final_entities.iloc[row_idx]
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
for raw in final_entities:
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
for raw in final_entities:
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
raw_bv = trace_df.loc[row_idx, "_entities_by_value"]
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
# ## 📊 (Optional) Score the detections with an LLM judge
#
# - `evaluate()` is a separate, opt-in step that runs LLM-as-judge metrics on the output.
# - This notebook uses Annotate, so only **Detection Validity** runs — it flags entities the detector got wrong (false positives, mislabels, boundary errors). Substitute would also enable Type Fidelity, Relational Consistency, and Attribute Fidelity.

# %%
evaluated = anonymizer.evaluate(result)
evaluated.display_record(0)

# %% [markdown]
# ## ⏭️ Next steps
#
# - **[🕵️ Your First Anonymization](../01_your_first_anonymization/)** --
#   the simplest end-to-end replace workflow if you haven't run it yet.
# - **[🎯 Choosing a Replacement Strategy](../03_choosing_a_replacement_strategy/)** --
#   compare Redact, Annotate, Hash, and Substitute side-by-side.
# - **[✏️ Rewriting Biographies](../04_rewriting_biographies/)** --
#   generate privacy-safe paraphrases instead of token-level replacements.

# %%
stop_local_runtime()
