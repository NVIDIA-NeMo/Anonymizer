# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test for the Anonymizer rewrite pipeline.

Hits real LLM APIs - expects NVIDIA_API_KEY in the environment.

Usage:
    python tests_e2e/test_e2e.py
"""

from pathlib import Path

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Rewrite, configure_logging

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "docs" / "data"

EXPECTED_REWRITE_COLUMNS = {
    "biography_rewritten",
    "utility_score",
    "leakage_mass",
    "weighted_leakage_rate",
    "any_high_leaked",
    "needs_human_review",
}

configure_logging()

anonymizer = Anonymizer()
input_data = AnonymizerInput(
    source=str(DATA_DIR / "NVIDIA_synthetic_biographies.csv"),
    text_column="biography",
    data_summary="Biographical profiles",
)
config = AnonymizerConfig(rewrite=Rewrite())

result = anonymizer.preview(config=config, data=input_data, num_records=3)

assert len(result.dataframe) > 0, "Preview returned no records"
missing = EXPECTED_REWRITE_COLUMNS - set(result.dataframe.columns)
assert not missing, f"Missing columns: {missing}"

print(f"\nPASS - {len(result.dataframe)} records, all expected columns present")
