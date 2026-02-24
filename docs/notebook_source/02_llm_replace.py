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
# # LLM Replace
#
# Uses an LLM to generate contextually appropriate synthetic replacements. Unlike local
# strategies, the LLM considers the full document context — matching names to emails,
# cities to states, etc.
#
# Requires: GLiNER endpoint (detection) + text LLM (augmentation, validation, replacement).

# %% [markdown]
# ## Setup

# %%
import tempfile
from pathlib import Path

try:
    NOTEBOOK_DIR = Path(__file__).resolve().parent
except NameError:
    NOTEBOOK_DIR = Path.cwd()

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, LLMReplace

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
# ## Run preview with LLM replacement
#
# `LLMReplace()` sends entities to an LLM that generates synthetic values.
# Set `data_summary` to help the LLM understand the domain.

# %%
config = AnonymizerConfig(replace=LLMReplace())

input_data = AnonymizerInput(
    source=str(NOTEBOOK_DIR / "data" / "synth_bios_sample10.csv"),
    text_column="bio",
    data_summary="Biographical profiles",
)

preview = anonymizer.preview(
    config=config,
    data=input_data,
    num_records=3,
)

# %%
preview.display_record(0)

# %%
preview.display_record(1)

# %%
preview.display_record(2)
