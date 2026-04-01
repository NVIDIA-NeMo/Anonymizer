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
# # Your First Anonymization
#
# Detect sensitive entities and replace them with LLM-generated substitutes.
# This is the simplest end-to-end example of Anonymizer.

# %% [markdown]
# ## Setup

# %%
from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Substitute

# %%
anonymizer = Anonymizer()

# %% [markdown]
# ## Load data and configure

# %%
input_data = AnonymizerInput(
    source="../data/NVIDIA_synthetic_biographies.csv",
    text_column="biography",
    data_summary="Biographical profiles of individuals",
)

config = AnonymizerConfig(replace=Substitute())

# %% [markdown]
# ## Preview
#
# Run on a small sample first to check the results.

# %%
preview = anonymizer.preview(config=config, data=input_data, num_records=3)

# %% [markdown]
# ## Inspect
#
# `display_record()` shows the original text with highlighted entities,
# the replacement map, and the anonymized output.

# %%
preview.display_record(0)

# %%
preview.display_record(1)

# %% [markdown]
# The result dataframe has the original text and the substituted version.

# %%
preview.dataframe

# %% [markdown]
# ## Full run
#
# Process the entire dataset.

# %%
result = anonymizer.run(config=config, data=input_data)
print(result)

# %%
result.dataframe.head()
