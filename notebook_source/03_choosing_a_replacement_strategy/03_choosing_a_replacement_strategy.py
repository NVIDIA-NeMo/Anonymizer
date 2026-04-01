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
# # Choosing a Replacement Strategy
#
# Four replacement strategies compared side by side:
# - **Redact** — remove entity, leave a marker
# - **Annotate** — tag entity with its label (no removal)
# - **Hash** — deterministic hash token
# - **Substitute** — LLM-generated synthetic values

# %% [markdown]
# ## Setup

# %%
from anonymizer import Annotate, Anonymizer, AnonymizerConfig, AnonymizerInput, Hash, Redact, Substitute

# %%
anonymizer = Anonymizer()

# %% [markdown]
# ## Input data

# %%
input_data = AnonymizerInput(
    source="../data/NVIDIA_synthetic_biographies.csv",
    text_column="biography",
    data_summary="Biographical profiles",
)

# %% [markdown]
# ## Redact
#
# Default: `[REDACTED_FIRST_NAME]`. Customize with `format_template`.

# %%
redact_config = AnonymizerConfig(replace=Redact())

redact_preview = anonymizer.preview(
    config=redact_config,
    data=input_data,
    num_records=3,
)

redact_preview.display_record(0)

# %%
redact_run = anonymizer.run(
    config=redact_config,
    data=input_data,
)

print(redact_run)
redact_run.display_record(0)

# %% [markdown]
# ### Custom template
#
# `format_template="***"` replaces every entity with the same constant.

# %%
custom_config = AnonymizerConfig(replace=Redact(format_template="***"))

custom_preview = anonymizer.preview(
    config=custom_config,
    data=input_data,
    num_records=3,
)

custom_preview.display_record(0)

# %% [markdown]
# ## Annotate
#
# Default: `<Alice, first_name>`. Customize with `format_template` — must include `{text}` and `{label}`.

# %%
annotate_config = AnonymizerConfig(replace=Annotate())

annotate_preview = anonymizer.preview(
    config=annotate_config,
    data=input_data,
    num_records=3,
)

annotate_preview.display_record(0)

# %% [markdown]
# ## Hash
#
# Deterministic — same input always produces the same hash. Customize `format_template`
# (must include `{digest}`), `algorithm` (`sha256`/`sha1`/`md5`), and `digest_length` (6–64).

# %%
hash_config = AnonymizerConfig(replace=Hash())

hash_preview = anonymizer.preview(
    config=hash_config,
    data=input_data,
    num_records=3,
)

hash_preview.display_record(0)

# %% [markdown]
# ## Substitute
#
# Uses an LLM to generate contextually appropriate synthetic replacements. Unlike the
# strategies above, the LLM considers the full document context — matching names to emails,
# cities to states, etc.

# %%
substitute_config = AnonymizerConfig(replace=Substitute())

substitute_preview = anonymizer.preview(
    config=substitute_config,
    data=input_data,
    num_records=3,
)

# %%
substitute_preview.display_record(0)
