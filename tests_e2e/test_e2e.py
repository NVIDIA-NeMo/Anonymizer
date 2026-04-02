# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, LoggingConfig, Rewrite, configure_logging

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "docs" / "notebook_source" / "data"

configure_logging(LoggingConfig.debug())

anonymizer = Anonymizer()
input_data = AnonymizerInput(
    source=str(DATA_DIR / "NVIDIA_synthetic_biographies.csv"),
    text_column="biography",
    data_summary="Biographical profiles",
)
config = AnonymizerConfig(rewrite=Rewrite())

preview = anonymizer.preview(
    config=config,
    data=input_data,
    num_records=3,
)

result = anonymizer.run(config=config, data=input_data)

print(result)
result.dataframe.head()


result.dataframe[["biography_rewritten", "utility_score", "leakage_mass", "needs_human_review"]].head()

print(result.trace_dataframe.columns.tolist())
