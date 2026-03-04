# %% [markdown]
# # Pipeline Logging Demo
#
# This script demonstrates the user-facing logging added by #14. It requires a
# running NDD/DataDesigner backend. Run cells individually in VS Code or PyCharm.

# %%
from anonymizer.logging import configure_logging

configure_logging(verbose=True)  # use configure_logging(verbose=True) to see DataDesigner engine logs

# %% [markdown]
# ## Configuration
#
# Set `GLINER_ENDPOINT` below to point to your GLiNER NIM instance.

# %%
# Change this to your GLiNER NIM endpoint URL
GLINER_ENDPOINT = ""

# %%
import tempfile
from pathlib import Path

import pandas as pd

tmp_dir = tempfile.mkdtemp(prefix="logging_demo_")
csv_path = Path(tmp_dir) / "sample.csv"

pd.DataFrame(
    {
        "text": [
            "Alice Johnson works at Acme Corp in Portland, Oregon.",
            "Contact Bob Smith at bob.smith@example.com or 555-0123.",
            "Dr. Carol Lee treated patient #12345 on 2024-03-15.",
        ]
    }
).to_csv(csv_path, index=False)

print(f"Sample data saved to {csv_path}")

# %% [markdown]
# ## Scenario 1: Full run with Redact replacement

# %%
from data_designer.config.default_model_settings import get_default_providers
from data_designer.config.models import ModelProvider

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import Redact
from anonymizer.interface.anonymizer import Anonymizer

MODEL_CONFIGS = """
model_configs:
  - alias: gliner-pii-detector
    model: nvidia/gliner-pii
    provider: gliner-nim
    inference_parameters:
      max_parallel_requests: 16
      temperature: 0.0
      top_p: 1.0
      max_tokens: 1024
      timeout: 120

  - alias: gpt-oss-120b
    model: nvdev/openai/gpt-oss-120b
    provider: nvidia
    inference_parameters:
      max_parallel_requests: 16
      max_tokens: 16384
      temperature: 0.3
      top_p: 0.95
      timeout: 300

  - alias: nemotron-30b-thinking
    model: nvidia/nemotron-3-nano-30b-a3b
    provider: nvidia
    inference_parameters:
      max_parallel_requests: 16
      max_tokens: 8192
      temperature: 0.4
      top_p: 1.0
      timeout: 300
"""

providers = get_default_providers() + [ModelProvider(name="gliner-nim", endpoint=GLINER_ENDPOINT)]

anonymizer = Anonymizer(
    model_configs=MODEL_CONFIGS,
    model_providers=providers,
)
result = anonymizer.run(
    config=AnonymizerConfig(replace=Redact()),
    data=AnonymizerInput(source=str(csv_path)),
)
result.dataframe

# %% [markdown]
# ## Scenario 2: Preview mode

# %%
preview = anonymizer.preview(
    config=AnonymizerConfig(replace=Redact()),
    data=AnonymizerInput(source=str(csv_path)),
    num_records=2,
)
preview.dataframe

# %% [markdown]
# ## Scenario 3: Detection only (no replacement)

# %%
from anonymizer.config.anonymizer_config import Rewrite

result_detect_only = anonymizer.run(
    config=AnonymizerConfig(rewrite=Rewrite()),
    data=AnonymizerInput(source=str(csv_path)),
)
result_detect_only.dataframe

# %% [markdown]
# ## Cleanup
#
# This file is temporary and should be removed before merging.
