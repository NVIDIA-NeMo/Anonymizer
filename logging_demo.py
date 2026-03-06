# %% [markdown]
# # Pipeline Logging Demo
#
# This script demonstrates the user-facing logging added by #14. It requires a
# running NDD/DataDesigner backend. Run cells individually in VS Code or PyCharm.

# %%
from anonymizer.logging import LoggingConfig, configure_logging

configure_logging()  # default: anonymizer INFO, DD suppressed
# configure_logging(LoggingConfig.verbose())  # anonymizer INFO + DD progress
# configure_logging(LoggingConfig.debug())    # everything DEBUG

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
from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import Redact
from anonymizer.interface.anonymizer import Anonymizer

anonymizer = Anonymizer()
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
