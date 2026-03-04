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
# # Replace Mode
#
# Four replacement strategies compared side by side:
# - **Redact** — remove entity, leave a marker
# - **Annotate** — tag entity with its label (no removal)
# - **Hash** — deterministic hash token
# - **Substitute** — LLM-generated synthetic values

# %% [markdown]
# ## Setup

# %%
import tempfile
from pathlib import Path

try:
    NOTEBOOK_SOURCE_DIR = Path(__file__).resolve().parent
except NameError:
    # Running as .ipynb — cwd is docs/notebooks/, data lives in docs/notebook_source/
    NOTEBOOK_SOURCE_DIR = Path.cwd().parent / "notebook_source"

from anonymizer import Annotate, Anonymizer, AnonymizerConfig, AnonymizerInput, Hash, Redact, Substitute

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
    inference_parameters:
      max_parallel_requests: 8
      temperature: 0.0
      top_p: 1.0
      max_tokens: 1024
      timeout: 120

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
# ## Input data

# %%
input_data = AnonymizerInput(
    source=str(NOTEBOOK_SOURCE_DIR / "data" / "synth_bios_sample10.csv"),
    text_column="bio",
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
